"""
SentinelVision - FastAPI Main Application Server
CraftifAI Physical AI Hackathon | Track A: PipeGen
"""

import os
import sys
import time
import json
import asyncio
import cv2
from typing import List, Dict, Any, Optional, AsyncGenerator
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel

# Add workspace directory to path for imports
workspace_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if workspace_dir not in sys.path:
    sys.path.insert(0, workspace_dir)

from sentinel_core.models import RestrictedZone, SafetyIncident, Detection
from sentinel_core.pipegen_adapter import PipeGenAdapter
from backend.pipeline_manager import PipelineManager
from backend.sample_generator import simulate_detections_for_frame

pipeline_mgr = PipelineManager(workspace_dir=workspace_dir)
adapter = PipeGenAdapter(zone_monitor=pipeline_mgr.zone_monitor, source_name="sentinel_stream")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Modern lifespan handler for continuous pipeline management and resource cleanup."""
    task = asyncio.create_task(pipeline_mgr.run_continuous_pipeline())
    yield
    pipeline_mgr.is_running = False
    pipeline_mgr.stop_pipegen_pipeline()
    task.cancel()


app = FastAPI(
    title="SentinelVision AI API",
    description="Workplace Safety Monitoring Powered by CraftifAI PipeGen (Track A)",
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS for Vite frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request models
class ZoneCreateRequest(BaseModel):
    zone_id: str
    name: str
    polygon: List[List[float]]
    color: str = "#EF4444"
    risk_level: str = "HIGH"
    cooldown_seconds: float = 3.0


class ModeRequest(BaseModel):
    mode: str


@app.get("/")
def read_root():
    return {
        "system": "SentinelVision Workplace Safety Monitoring System",
        "hackathon": "CraftifAI Physical AI Hackathon — Track A: PipeGen",
        "status": "online",
        "docs": "/docs",
        "api_status": "/api/status"
    }


@app.get("/api/status")
def get_system_status():
    """Return pipeline, GPU, Docker, and incident statistics."""
    return pipeline_mgr.get_status()


@app.get("/api/incidents")
def list_incidents(limit: int = 50):
    """Retrieve recent safety incident events formatted as Track A structured JSON."""
    incidents = pipeline_mgr.zone_monitor.get_recent_incidents(limit=limit)
    return [inc.model_dump() for inc in incidents]


@app.get("/api/incidents/stats")
def incident_statistics():
    """Return summary statistics for dashboard counters."""
    return pipeline_mgr.zone_monitor.get_incident_counts()


@app.get("/api/zones")
def list_zones():
    """List all configured restricted zones."""
    return [z.model_dump() for z in pipeline_mgr.zone_monitor.get_zones()]


@app.post("/api/zones")
def save_zone(zone_req: ZoneCreateRequest):
    """Add or update a restricted zone configuration."""
    zone = RestrictedZone(**zone_req.model_dump())
    pipeline_mgr.zone_monitor.add_zone(zone)
    return {"status": "success", "zone": zone.model_dump()}


@app.delete("/api/zones/{zone_id}")
def delete_zone(zone_id: str):
    """Remove a restricted zone."""
    removed = pipeline_mgr.zone_monitor.remove_zone(zone_id)
    if not removed:
        raise HTTPException(status_code=404, detail="Zone not found")
    return {"status": "success", "zone_id": zone_id}


@app.get("/api/metrics")
def get_performance_metrics():
    """Return real-time inference latency, FPS, and GPU load."""
    gpu = pipeline_mgr.get_gpu_status()
    return {
        "fps": pipeline_mgr.current_fps,
        "inference_latency_ms": pipeline_mgr.inference_latency_ms,
        "pipeline_latency_ms": round(pipeline_mgr.inference_latency_ms + 4.8, 1),
        "target_fps": 30.0,
        "gpu_temperature_c": gpu.get("temperature_c", 0.0),
        "gpu_memory_used_mb": gpu.get("memory_used_mb", 0.0),
        "gpu_memory_total_mb": gpu.get("memory_total_mb", 4096.0),
        "gpu_utilization_pct": gpu.get("utilization_pct", 0.0),
        "total_frames_processed": pipeline_mgr.total_frames_processed,
        "mode": pipeline_mgr.mode
    }


@app.post("/api/mode")
def set_pipeline_mode(req: ModeRequest):
    """Toggle between Live Inference (PipeGen/GPU) and Sample Demo mode."""
    if req.mode not in ("live", "demo"):
        raise HTTPException(status_code=400, detail="Mode must be 'live' or 'demo'")
    pipeline_mgr.set_mode(req.mode)
    return {"status": "success", "mode": pipeline_mgr.mode}


class PipelineStartRequest(BaseModel):
    input_source: Optional[str] = None


@app.post("/api/pipegen/ingest")
async def ingest_pipegen_metadata(payload: Dict[str, Any]):
    """
    Direct webhook / socket endpoint to ingest real-time detection metadata
    emitted by CraftifAI PipeGen DeepStream pipeline.
    """
    t0 = time.time()
    pipeline_mgr.last_pipegen_ingest_time = t0

    detections = adapter.parse_frame_payload(payload)
    pipeline_mgr.latest_live_detections = detections
    pipeline_mgr.total_detections += len(detections)
    pipeline_mgr.total_frames_processed += 1

    incidents = pipeline_mgr.zone_monitor.process_detections(detections, frame_timestamp=t0)
    latency = (time.time() - t0) * 1000.0
    pipeline_mgr.inference_latency_ms = round(latency, 2)

    for inc in incidents:
        await pipeline_mgr.broadcast_incident(inc)

    return {
        "status": "processed",
        "detections_count": len(detections),
        "incidents_triggered": len(incidents),
        "latency_ms": round(latency, 2)
    }


@app.post("/api/pipeline/start")
def start_pipeline(req: Optional[PipelineStartRequest] = None):
    """Launch the real PipeGen / DeepStream container inference pipeline in background."""
    src = req.input_source if req else None
    return pipeline_mgr.start_pipegen_pipeline(input_source=src)


@app.post("/api/pipeline/stop")
def stop_pipeline():
    """Stop the running PipeGen pipeline container process."""
    return pipeline_mgr.stop_pipegen_pipeline()


@app.get("/api/pipeline/benchmark")
def get_benchmark_report():
    """Return measured empirical benchmark metrics for NVIDIA RTX 2050."""
    return pipeline_mgr.get_benchmark_data()


async def generate_mjpeg_stream() -> AsyncGenerator[bytes, None]:
    """
    Streams annotated frames as an HTTP multipart MJPEG feed.
    Draws active restricted zones, worker bounding box, and intrusion alerts.
    """
    while True:
        if pipeline_mgr.latest_frame_jpeg:
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + pipeline_mgr.latest_frame_jpeg + b"\r\n")
        await asyncio.sleep(0.033)


@app.get("/api/stream/feed")
def get_video_stream():
    """Video stream endpoint returning annotated MJPEG."""
    return StreamingResponse(
        generate_mjpeg_stream(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.get("/api/stream/events")
async def sse_incident_stream():
    """Server-Sent Events (SSE) pushing new safety incidents in real-time."""
    async def event_generator():
        queue = await pipeline_mgr.register_subscriber()
        try:
            while True:
                # Send heartbeat or new event
                try:
                    data = await asyncio.wait_for(queue.get(), timeout=15.0)
                    yield f"event: incident\ndata: {data}\n\n"
                except asyncio.TimeoutError:
                    yield f"event: ping\ndata: {json.dumps({'time': time.time()})}\n\n"
        finally:
            pipeline_mgr.unregister_subscriber(queue)

    return StreamingResponse(event_generator(), media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)

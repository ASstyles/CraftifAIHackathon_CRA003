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

# Enable CORS for Vite frontend (restricted to localhost/127.0.0.1 origins for secure credential handling)
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$",
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


from sentinel_core.launcher import PipelineLaunchConfig, RuntimeRequirement


@app.get("/api/metrics")
def get_performance_metrics():
    """Return real-time inference latency, FPS, and GPU load."""
    return pipeline_mgr.get_metrics()


@app.post("/api/mode")
def set_pipeline_mode(req: ModeRequest):
    """Toggle between Live Inference (PipeGen/GPU) and Sample Demo mode."""
    if req.mode not in ("live", "demo"):
        raise HTTPException(status_code=400, detail="Mode must be 'live' or 'demo'")
    pipeline_mgr.set_mode(req.mode)
    return {"status": "success", "mode": pipeline_mgr.mode}


class PipelineStartRequest(BaseModel):
    input_source: str = "assets/workplace_safety_demo.mp4"
    yolo_model: Optional[str] = "pipegen_pipeline/official_pipegen_build/model_stage/yolov8n.fp32.plan"
    pose_model: Optional[str] = "pipegen_pipeline/official_pipegen_build/model_stage/6-yolo11n-pose/yolo11n-pose.fp32.plan"
    output_video: Optional[str] = "assets/annotated_pipeline_run.mp4"
    runtime_mode: str = "tensorrt_required"
    no_show: bool = True
    loop: bool = False


@app.post("/api/pipegen/ingest")
async def ingest_pipegen_metadata(payload: Dict[str, Any]):
    """
    Direct webhook / socket endpoint to ingest real-time detection metadata
    emitted by CraftifAI PipeGen DeepStream pipeline.
    """
    if not isinstance(payload, dict):
        raise HTTPException(status_code=422, detail="Payload must be a JSON object")

    raw_objects = payload.get("objects", [])
    if not isinstance(raw_objects, list):
        raise HTTPException(status_code=422, detail="'objects' must be a list")
    if len(raw_objects) > 200:
        raise HTTPException(status_code=400, detail="Too many objects in frame (max 200)")

    t0 = time.perf_counter()
    detections = adapter.parse_frame_payload(payload)
    incidents = pipeline_mgr.zone_monitor.process_detections(detections, frame_timestamp=time.time())
    for inc in incidents:
        await pipeline_mgr.broadcast_incident(inc)
    handler_ms = (time.perf_counter() - t0) * 1000.0
    pipeline_mgr.record_ingested_frame(detections, payload, ingest_handler_ms=handler_ms)

    return {
        "status": "processed",
        "detections_count": len(detections),
        "incidents_triggered": len(incidents),
        "latency_ms": round(handler_ms, 2),
        "ingest_handler_latency_ms": round(handler_ms, 2),
        "runner_inference_latency_ms": payload.get("inference_latency_ms")
    }


@app.post("/api/pipeline/reset-telemetry")
def reset_pipeline_telemetry():
    """Reset live measured telemetry back to uninitialized state for test isolation."""
    pipeline_mgr.reset_telemetry()
    return {"status": "success", "message": "Live telemetry reset to uninitialized state"}


@app.post("/api/pipeline/start")
def start_pipeline(req: Optional[PipelineStartRequest] = None):
    """Launch the real PipeGen / DeepStream container inference pipeline in background."""
    if req is not None:
        try:
            runtime_enum = RuntimeRequirement(req.runtime_mode)
        except ValueError:
            valid_opts = [r.value for r in RuntimeRequirement]
            raise HTTPException(
                status_code=400,
                detail=f"Invalid runtime_mode '{req.runtime_mode}'. Must be one of: {valid_opts}"
            )
        cfg = PipelineLaunchConfig(
            input_source=req.input_source,
            yolo_model=req.yolo_model or "pipegen_pipeline/official_pipegen_build/model_stage/yolov8n.fp32.plan",
            pose_model=req.pose_model or "pipegen_pipeline/official_pipegen_build/model_stage/6-yolo11n-pose/yolo11n-pose.fp32.plan",
            output_video=req.output_video,
            runtime_mode=runtime_enum,
            no_show=req.no_show,
            loop=req.loop
        )
    else:
        cfg = PipelineLaunchConfig()

    result = pipeline_mgr.start_pipegen_pipeline(launch_config=cfg)
    if result.get("status") in ("validation_failed", "command_construction_failed", "startup_failed", "error"):
        raise HTTPException(status_code=400, detail=result)
    return result


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

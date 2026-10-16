"""
SentinelVision - Pipeline & Inference Manager
CraftifAI Physical AI Hackathon | Track A: PipeGen
"""

import os
import sys
import shutil
import time
import json
import cv2
import asyncio
import subprocess
import numpy as np
from typing import Optional, Dict, Any, List, AsyncGenerator
from sentinel_core.models import Detection, RestrictedZone, SafetyIncident, PipelineStatus
from sentinel_core.zone_monitor import ZoneMonitor
from sentinel_core.launcher import PipelineProcessManager, PipelineLaunchConfig, RuntimeRequirement
from .sample_generator import generate_sample_hazard_video, simulate_detections_for_frame


class PipelineManager:
    """
    Manages the video inference pipeline, PipeGen container interface,
    GPU telemetry monitoring, and real-time streaming feed.
    """

    def __init__(self, workspace_dir: str):
        self.workspace_dir = workspace_dir
        self.sample_video_path = os.path.join(workspace_dir, "assets", "workplace_safety_demo.mp4")
        self.zone_monitor = ZoneMonitor(source_name="workplace_safety_demo.mp4 [DEMO_FIXTURE]")

        # Safe Subprocess Lifecycle Manager
        self.process_manager = PipelineProcessManager(workspace_dir=workspace_dir)

        # Operating Mode
        self.mode = "demo"  # 'demo' (synthetic test fixture) or 'live' (real GPU PipeGen pipeline)
        self.is_running = True

        # Metric Contract: Target values vs Measured runtime values (null until measured)
        self.target_fps: float = 30.0
        self.target_latency_ms: float = 10.0

        self.measured_fps: Optional[float] = None
        self.measured_inference_latency_ms: Optional[float] = None
        self.measured_pipeline_latency_ms: Optional[float] = None
        self.measured_ingest_latency_ms: Optional[float] = None
        self.inference_backend: Optional[str] = None
        self.metrics_source: str = "none"  # "pipegen_runner_telemetry", "demo_synthetic_fixture", "none"
        self.last_measurement_time: Optional[float] = None

        # Rolling window for FPS calculation (monotonic timestamps)
        self._frame_timestamps: List[float] = []

        # Incident and state accumulators
        self.total_frames_processed: int = 0
        self.total_detections: int = 0
        self.active_detections: List[Detection] = []
        self.latest_incident: Optional[SafetyIncident] = None
        self.latest_frame_jpeg: bytes = b""
        self.subscribers: List[asyncio.Queue] = []
        self.latest_live_detections: List[Detection] = []

        # Ensure sample video exists for reproducible evaluation
        if not os.path.exists(self.sample_video_path):
            generate_sample_hazard_video(self.sample_video_path)

    # Backwards-compatible properties
    @property
    def current_fps(self) -> Optional[float]:
        return self.measured_fps

    @property
    def inference_latency_ms(self) -> Optional[float]:
        return self.measured_inference_latency_ms

    def get_gpu_status(self) -> Dict[str, Any]:
        """Query real NVIDIA GPU telemetry via nvidia-smi."""
        try:
            cmd = [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total,memory.used,memory.free,temperature.gpu,utilization.gpu",
                "--format=csv,noheader,nounits"
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=3)
            if result.returncode == 0:
                parts = [p.strip() for p in result.stdout.strip().split(",")]
                if len(parts) >= 7:
                    return {
                        "name": parts[0],
                        "driver": parts[1],
                        "memory_total_mb": float(parts[2]),
                        "memory_used_mb": float(parts[3]),
                        "memory_free_mb": float(parts[4]),
                        "temperature_c": float(parts[5]),
                        "utilization_pct": float(parts[6]),
                        "available": True,
                        "source": "nvidia-smi"
                    }
        except Exception:
            pass

        return {
            "name": "NVIDIA GPU Unavailable",
            "driver": "N/A",
            "memory_total_mb": 0.0,
            "memory_used_mb": 0.0,
            "memory_free_mb": 0.0,
            "temperature_c": 0.0,
            "utilization_pct": 0.0,
            "available": False,
            "source": "none",
            "error": "nvidia-smi query returned no telemetry on current host"
        }

    def check_pipegen_container(self) -> Dict[str, Any]:
        """Check if CraftifAI PipeGen DeepStream container or image is present."""
        try:
            cmd = ["docker", "ps", "-a", "--filter", "name=craftifai", "--format", "{{.Names}}\t{{.Status}}"]
            if sys.platform == "win32" and shutil.which("docker") is None and shutil.which("wsl") is not None:
                cmd = ["wsl", "-d", "Ubuntu", "--", "docker", "ps", "-a", "--filter", "name=craftifai", "--format", "{{.Names}}\t{{.Status}}"]

            res = subprocess.run(cmd, capture_output=True, text=True, timeout=4)
            running = "craftifai" in res.stdout or self.process_manager.is_running()
            output = res.stdout.strip()

            img_ready = False
            if not running:
                img_cmd = ["docker", "images", "craftifaidocker/deepstream:7.1-amd64-v3", "--format", "{{.Repository}}:{{.Tag}}"]
                if sys.platform == "win32" and shutil.which("docker") is None and shutil.which("wsl") is not None:
                    img_cmd = ["wsl", "-d", "Ubuntu", "--", "docker", "images", "craftifaidocker/deepstream:7.1-amd64-v3", "--format", "{{.Repository}}:{{.Tag}}"]
                res_img = subprocess.run(img_cmd, capture_output=True, text=True, timeout=4)
                img_ready = "craftifaidocker/deepstream" in res_img.stdout

            container_found = running or len(output) > 0 or img_ready
            status_desc = "Active" if running else (output if output else ("Image Ready (craftifaidocker/deepstream:7.1-amd64-v3)" if img_ready else "Stopped / Standby"))
            return {
                "container_found": container_found,
                "container_status": status_desc,
                "pipegen_installed": True,
                "pipegen_version": "0.5.1",
                "pipeline_process_active": running
            }
        except Exception as e:
            return {
                "container_found": False,
                "container_status": f"Docker check: {str(e)}",
                "pipegen_installed": True,
                "pipegen_version": "0.5.1",
                "pipeline_process_active": False
            }

    def start_pipegen_pipeline(self, launch_config: Optional[PipelineLaunchConfig] = None) -> Dict[str, Any]:
        """
        Starts the real PipeGen / DeepStream container inference pipeline.
        Uses unified PipelineProcessManager with pre-flight configuration validation.
        """
        config = launch_config or PipelineLaunchConfig(
            input_source="assets/workplace_safety_demo.mp4",
            runtime_mode=RuntimeRequirement.TENSORRT_REQUIRED,
            loop=True
        )

        res = self.process_manager.start(config)
        if res.get("status") == "started":
            self.mode = "live"
            self.zone_monitor.source_name = "PipeGen DeepStream GPU Pipeline (Live)"
        return res

    def stop_pipegen_pipeline(self) -> Dict[str, Any]:
        """Stops the running PipeGen pipeline container process."""
        return self.process_manager.stop()

    def is_pipegen_active(self) -> bool:
        """Check if PipeGen container process or live ingest stream is active."""
        if self.process_manager.is_running():
            return True
        if self.last_measurement_time and (time.time() - self.last_measurement_time < 5.0):
            return True
        return False

    def record_ingested_frame(self, detections: List[Detection], payload: Dict[str, Any], ingest_handler_ms: float):
        """
        Processes real inference metadata received via webhook.
        Maintains strict separation between model inference time and backend handler time.
        """
        now = time.monotonic()
        wall_now = time.time()
        self.last_measurement_time = wall_now

        self.latest_live_detections = detections
        self.active_detections = detections
        self.total_detections += len(detections)
        self.total_frames_processed += 1

        # Calculate rolling FPS over last 30 frames
        self._frame_timestamps.append(now)
        if len(self._frame_timestamps) > 30:
            self._frame_timestamps.pop(0)

        if len(self._frame_timestamps) >= 2:
            dt = self._frame_timestamps[-1] - self._frame_timestamps[0]
            if dt > 0:
                self.measured_fps = round((len(self._frame_timestamps) - 1) / dt, 1)

        # Record ingest handler latency separately
        self.measured_ingest_latency_ms = round(ingest_handler_ms, 2)

        # Extract pure model inference latency from runner payload if present
        if "inference_latency_ms" in payload and payload["inference_latency_ms"] is not None:
            self.measured_inference_latency_ms = float(payload["inference_latency_ms"])
            self.metrics_source = "pipegen_runner_telemetry"
        else:
            self.measured_inference_latency_ms = None
            self.metrics_source = "ingest_stream_only"

        if "inference_backend" in payload and payload["inference_backend"]:
            self.inference_backend = str(payload["inference_backend"])

        # Calculate pipeline latency if model latency is known
        if self.measured_inference_latency_ms is not None:
            self.measured_pipeline_latency_ms = round(
                self.measured_inference_latency_ms + self.measured_ingest_latency_ms, 2
            )
        else:
            self.measured_pipeline_latency_ms = None

    def reset_telemetry(self):
        """Reset live measured telemetry back to uninitialized state (for clean test isolation)."""
        self._frame_timestamps.clear()
        self.measured_fps = None
        self.measured_inference_latency_ms = None
        self.measured_pipeline_latency_ms = None
        self.measured_ingest_latency_ms = None
        self.metrics_source = "uninitialized"
        self.last_measurement_time = None
        self.latest_live_detections = []
        self.active_detections = []

    def get_benchmark_data(self) -> Dict[str, Any]:
        """Load empirical measured benchmark data from actual_measured_benchmark.json."""
        benchmark_file = os.path.join(self.workspace_dir, "pipegen_pipeline", "actual_measured_benchmark.json")
        if os.path.exists(benchmark_file):
            try:
                with open(benchmark_file, "r") as f:
                    return json.load(f)
            except Exception as e:
                return {"error": f"Failed to load benchmark data: {str(e)}"}
        return {"error": "Benchmark data file not found"}

    def get_status(self) -> Dict[str, Any]:
        """Return comprehensive system and pipeline status with strict metric provenance."""
        gpu = self.get_gpu_status()
        pipegen_status = self.check_pipegen_container()
        counts = self.zone_monitor.get_incident_counts()
        is_live_ingesting = self.last_measurement_time is not None and (time.time() - self.last_measurement_time < 5.0)

        if self.mode == "live":
            operational_status = "ACTIVE_GPU_STREAM" if is_live_ingesting else "AWAITING_PIPEGEN_INGESTION"
        else:
            operational_status = "DEMO_SYNTHETIC_FIXTURE"

        is_stale = self.last_measurement_time is None or (time.time() - self.last_measurement_time > 5.0)

        return {
            "pipeline_id": "sentinel_pipegen_track_a",
            "pipeline_name": "SentinelVision Industrial Safety AI Pipeline",
            "pipegen_version": "0.5.1",
            "mode": self.mode,
            "operational_status": operational_status,
            "pipegen_installed": pipegen_status["pipegen_installed"],
            "container_ready": pipegen_status["container_found"],
            "container_status": pipegen_status["container_status"],
            "live_pipeline_active": self.is_pipegen_active(),
            "pipeline_process_state": "running" if self.process_manager.is_running() else (
                "failed" if self.process_manager.last_error_message else "stopped"
            ),
            "last_error": self.process_manager.last_error_message,
            "last_pipegen_ingest_seconds_ago": round(time.time() - self.last_measurement_time, 1) if self.last_measurement_time else None,
            "model_architecture": "YOLOv8n Person Detector + YOLO11n-Pose (Parallel DAG)",
            "inference_backend": self.inference_backend,
            "tensorrt_engine": "yolov8n.fp32.plan (22.3 MB, 248 TRT layers, TensorRT 10.3.0)",
            "pose_engine": "yolo11n-pose.fp32.plan (18.9 MB, 252 TRT layers, TensorRT 10.3.0)",
            "target_hardware": "NVIDIA GeForce RTX 2050 (4 GB VRAM)",
            "target_fps": self.target_fps,
            "target_latency_ms": self.target_latency_ms,
            "measured_fps": self.measured_fps if not is_stale else None,
            "measured_inference_latency_ms": self.measured_inference_latency_ms if not is_stale else None,
            "measured_pipeline_latency_ms": self.measured_pipeline_latency_ms if not is_stale else None,
            "measured_ingest_latency_ms": self.measured_ingest_latency_ms if not is_stale else None,
            "metrics_source": self.metrics_source if not is_stale else "none",
            "is_stale": is_stale,
            "total_frames_processed": self.total_frames_processed,
            "total_detections": self.total_detections,
            "total_incidents": counts["total_incidents"],
            "active_intrusions": counts["active_intrusions"],
            "gpu": gpu,
            "active_zones_count": len(self.zone_monitor.get_zones())
        }

    def get_metrics(self) -> Dict[str, Any]:
        """Return real-time performance telemetry with explicit nulls for unmeasured values."""
        gpu = self.get_gpu_status()
        is_stale = self.last_measurement_time is None or (time.time() - self.last_measurement_time > 5.0)

        # In demo mode, clearly mark metrics as synthetic fixture
        if self.mode == "demo":
            return {
                "mode": "demo",
                "metrics_source": "demo_synthetic_fixture",
                "is_stale": False,
                "fps": 30.0,
                "target_fps": self.target_fps,
                "inference_latency_ms": 3.8,  # Approximate CPU simulation time
                "pipeline_latency_ms": 4.5,
                "ingest_latency_ms": 0.5,
                "inference_backend": "CPU_FIXTURE",
                "gpu_temperature_c": gpu.get("temperature_c", 0.0),
                "gpu_memory_used_mb": gpu.get("memory_used_mb", 0.0),
                "gpu_memory_total_mb": gpu.get("memory_total_mb", 4096.0),
                "gpu_utilization_pct": gpu.get("utilization_pct", 0.0),
                "total_frames_processed": self.total_frames_processed,
                "last_measurement_time": self.last_measurement_time
            }

        # In live mode: return measured metrics or null if unmeasured / stale
        return {
            "mode": "live",
            "metrics_source": self.metrics_source if not is_stale else "none",
            "is_stale": is_stale,
            "fps": self.measured_fps if not is_stale else None,
            "target_fps": self.target_fps,
            "inference_latency_ms": self.measured_inference_latency_ms if not is_stale else None,
            "pipeline_latency_ms": self.measured_pipeline_latency_ms if not is_stale else None,
            "ingest_latency_ms": self.measured_ingest_latency_ms if not is_stale else None,
            "inference_backend": self.inference_backend if not is_stale else None,
            "gpu_temperature_c": gpu.get("temperature_c", 0.0) if gpu.get("available") else None,
            "gpu_memory_used_mb": gpu.get("memory_used_mb", 0.0) if gpu.get("available") else None,
            "gpu_memory_total_mb": gpu.get("memory_total_mb", 4096.0) if gpu.get("available") else None,
            "gpu_utilization_pct": gpu.get("utilization_pct", 0.0) if gpu.get("available") else None,
            "total_frames_processed": self.total_frames_processed,
            "last_measurement_time": self.last_measurement_time
        }

    def set_mode(self, mode: str):
        """Switch between 'live' (real GPU pipeline) and 'demo' (sample test fixture)."""
        if mode in ("live", "demo"):
            self.mode = mode
            if mode == "demo":
                self.zone_monitor.source_name = "workplace_safety_demo.mp4 [DEMO_FIXTURE]"
                self.metrics_source = "demo_synthetic_fixture"
            else:
                self.zone_monitor.source_name = "PipeGen DeepStream GPU Pipeline (Live)"
                self.metrics_source = "none"

    async def register_subscriber(self) -> asyncio.Queue:
        """Register a queue to receive real-time incident event pushes."""
        q = asyncio.Queue()
        self.subscribers.append(q)
        return q

    def unregister_subscriber(self, q: asyncio.Queue):
        if q in self.subscribers:
            self.subscribers.remove(q)

    async def broadcast_incident(self, incident: SafetyIncident):
        """Send incident to all active websocket/SSE subscribers."""
        self.latest_incident = incident
        data = incident.model_dump_json()
        for q in list(self.subscribers):
            try:
                await q.put(data)
            except Exception:
                pass

    def render_annotated_frame(self, frame: np.ndarray, detections: List[Detection], new_incidents: List[SafetyIncident]) -> bytes:
        """
        Draws zones, bounding boxes, foot contact points, and warning banners on video frame.
        Encodes as JPEG bytes for HTTP MJPEG streaming.
        """
        annotated = frame.copy()
        zones = self.zone_monitor.get_zones()

        # 1. Draw zones
        for zone in zones:
            pts = np.array(zone.polygon, np.int32)
            zone_color = (0, 0, 230) if zone.risk_level == "CRITICAL" else (0, 160, 240)
            cv2.polylines(annotated, [pts], isClosed=True, color=zone_color, thickness=2)

        # 2. Draw person detections
        for det in detections:
            xmin, ymin, xmax, ymax = [int(c) for c in det.bounding_box]
            foot_x, foot_y = int((xmin + xmax) / 2), int(ymax)
            if det.detection_point:
                foot_x, foot_y = int(det.detection_point[0]), int(det.detection_point[1])

            # Check if this person is currently in any zone
            in_zone = False
            for tstate in self.zone_monitor.tracks.values():
                if tstate.last_bbox and abs(tstate.last_bbox[0] - det.bounding_box[0]) < 10:
                    if len(tstate.active_zones) > 0:
                        in_zone = True
                        break

            color = (0, 0, 255) if in_zone else (0, 230, 100)
            cv2.rectangle(annotated, (xmin, ymin), (xmax, ymax), color, 2)
            cv2.circle(annotated, (foot_x, foot_y), 5, (0, 255, 255), -1)

            # Label with contact source
            src_tag = " [ANKLE]" if det.contact_source == "pose_ankles" else ""
            label = f"Person #{det.track_id or 1} ({det.confidence*100:.0f}%){src_tag}"
            if in_zone:
                label += " [INTRUSION]"
            cv2.putText(annotated, label, (xmin, max(20, ymin - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # 3. Draw Watermark & Pipeline Status
        if self.mode == "live":
            is_active = self.last_measurement_time and (time.time() - self.last_measurement_time < 5.0)
            status_text = "STREAMING" if is_active else "AWAITING INGESTION"
            fps_str = f"{self.measured_fps:.1f} FPS" if self.measured_fps is not None else "N/A FPS"
            lat_str = f"{self.measured_inference_latency_ms:.1f} ms" if self.measured_inference_latency_ms is not None else "N/A ms"
            backend_str = f" | {self.inference_backend}" if self.inference_backend else ""
            badge = f"SentinelVision Track A | Mode: LIVE GPU ({status_text}){backend_str} | {fps_str} | {lat_str}"
        else:
            badge = f"SentinelVision Track A | Mode: DEMO (SYNTHETIC FIXTURE) | 30.0 FPS | 3.8 ms"
        cv2.putText(annotated, badge, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 2)

        # Encode to JPEG
        ret, jpeg = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
        return jpeg.tobytes() if ret else b""

    async def run_continuous_pipeline(self):
        """
        Runs continuous display loop in the background.
        Ensures video monitor, detections, and safety incidents update cleanly
        without confusing live mode with demo data.
        """
        cap = cv2.VideoCapture(self.sample_video_path)
        frame_idx = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 300

        while self.is_running:
            if not cap.isOpened() or not cap.grab():
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                frame_idx = 0
                await asyncio.sleep(0.033)
                continue

            ret, frame = cap.retrieve()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                frame_idx = 0
                await asyncio.sleep(0.033)
                continue

            now = time.time()

            if self.mode == "demo":
                detections = simulate_detections_for_frame(frame_idx, total_frames=total_frames)
                self.active_detections = detections
                self.total_detections += len(detections)
                self.total_frames_processed += 1
                new_incidents = self.zone_monitor.process_detections(detections, frame_timestamp=now)
                for inc in new_incidents:
                    await self.broadcast_incident(inc)
            else:
                is_active = self.last_measurement_time and (now - self.last_measurement_time < 5.0)
                if is_active:
                    detections = self.latest_live_detections
                    self.active_detections = detections
                else:
                    detections = []
                    self.active_detections = []
                new_incidents = []

            self.latest_frame_jpeg = self.render_annotated_frame(frame, detections, new_incidents)

            frame_idx = (frame_idx + 1) % total_frames
            await asyncio.sleep(0.033)

        cap.release()

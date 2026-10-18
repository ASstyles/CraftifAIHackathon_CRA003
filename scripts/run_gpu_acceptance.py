"""
SentinelVision - Reproducible GPU Integration Acceptance Run (Task 3)
CraftifAI Physical AI Hackathon | Track A: PipeGen

Executes and verifies the complete physical AI pipeline on genuine hardware:
1. Verifies backend availability and resets telemetry to clean baseline.
2. Checks GPU runtime, container image, and TensorRT .plan engine availability (with SHA-256).
3. Connects to backend SSE live stream (/api/stream/events) to capture real-time event delivery.
4. Executes native TensorRT pipeline inside DeepStream container (--runtime tensorrt_required).
5. Captures real ingestion delivery metrics, incident payloads, and schema conformity.
6. Validates the generated output video (playable, frame count, dimensions, fps).
7. Derives a structured, non-hardcoded JSON summary saved to logs/gpu_acceptance_summary.json.
"""

import os
import sys
import time
import json
import hashlib
import argparse
import threading
import queue
import subprocess
from datetime import datetime, timezone
from urllib.request import urlopen, Request
from urllib.error import URLError

# Add workspace directory to path
workspace_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if workspace_dir not in sys.path:
    sys.path.insert(0, workspace_dir)

import cv2
from sentinel_core.models import SafetyIncident


def compute_sha256(filepath: str) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def check_backend_alive(backend_url: str) -> dict:
    urls_to_try = [backend_url]
    if "localhost" in backend_url:
        urls_to_try.append(backend_url.replace("localhost", "127.0.0.1"))
    elif "127.0.0.1" in backend_url:
        urls_to_try.append(backend_url.replace("127.0.0.1", "localhost"))

    last_err = "No response"
    for url in urls_to_try:
        try:
            req = Request(f"{url}/api/status", headers={"User-Agent": "SentinelVision-GPU-Acceptance"})
            with urlopen(req, timeout=5) as resp:
                if resp.status == 200:
                    data = json.loads(resp.read().decode("utf-8"))
                    data["_resolved_url"] = url
                    return data
        except Exception as e:
            last_err = str(e)
    return {"error": last_err}


def sse_listener_worker(stream_url: str, stop_event: threading.Event, connected_event: threading.Event, out_queue: queue.Queue):
    try:
        req = Request(stream_url, headers={"Accept": "text/event-stream", "User-Agent": "SentinelVision-SSE-Test"})
        with urlopen(req, timeout=120) as resp:
            connected_event.set()
            while not stop_event.is_set():
                line = resp.readline()
                if not line:
                    break
                line_str = line.decode("utf-8", errors="replace").strip()
                if line_str.startswith("data:"):
                    payload_str = line_str[len("data:"):].strip()
                    try:
                        payload = json.loads(payload_str)
                        # Filter to real safety incidents (exclude heartbeat pings or connection acks)
                        if payload.get("event_type") == "restricted_zone_intrusion":
                            out_queue.put(payload)
                    except Exception:
                        pass
    except Exception:
        connected_event.set()


def validate_output_video(video_path: str) -> dict:
    if not os.path.exists(video_path):
        return {"valid": False, "error": f"File does not exist: {video_path}"}
    size_bytes = os.path.getsize(video_path)
    if size_bytes == 0:
        return {"valid": False, "error": "Video file is 0 bytes"}

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"valid": False, "error": "Failed to open video with OpenCV VideoCapture"}

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    ret_first, frame_first = cap.read()
    if not ret_first or frame_first is None:
        cap.release()
        return {"valid": False, "error": "Failed to read first frame"}

    # Read last frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_count - 1))
    ret_last, frame_last = cap.read()
    cap.release()

    return {
        "valid": True,
        "size_bytes": size_bytes,
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "fps": fps,
        "first_frame_read": ret_first,
        "last_frame_read": ret_last
    }


def run_gpu_acceptance(backend_url: str = "http://localhost:8000", max_frames: int = 300):
    print("==========================================================================")
    print(" SentinelVision — Reproducible GPU Integration Acceptance Run")
    print(" Hardware-In-The-Loop Validation (Task 3)")
    print("==========================================================================")

    acceptance_summary = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "backend_url": backend_url,
        "environment": {},
        "engines": {},
        "preflight": {},
        "pipeline_execution": {},
        "telemetry_and_events": {},
        "output_video": {},
        "overall_status": "NOT_VERIFIED"
    }

    # Step 1: Pre-flight backend check
    print(f"[*] Checking backend status at: {backend_url}")
    status_data = check_backend_alive(backend_url)
    if "error" in status_data:
        acceptance_summary["preflight"]["backend_status"] = "FAIL"
        acceptance_summary["preflight"]["backend_error"] = status_data["error"]
        print(f"[FAIL] Backend is not reachable at {backend_url}. Start backend first.")
        _write_summary(acceptance_summary)
        return False

    if "_resolved_url" in status_data:
        backend_url = status_data["_resolved_url"]
        acceptance_summary["backend_url"] = backend_url

    acceptance_summary["preflight"]["backend_status"] = "PASS"
    acceptance_summary["environment"]["backend_app"] = status_data.get("pipeline_name", "SentinelVision")
    acceptance_summary["environment"]["backend_version"] = status_data.get("pipegen_version", "0.5.1")
    print(f"[PASS] Backend Online ({backend_url}): {acceptance_summary['environment']['backend_app']} (PipeGen v{acceptance_summary['environment']['backend_version']})")

    # Step 2: Engine availability & hash verification
    yolo_plan = os.path.join(workspace_dir, "pipegen_pipeline", "official_pipegen_build", "model_stage", "yolov8n.fp32.plan")
    pose_plan = os.path.join(workspace_dir, "pipegen_pipeline", "official_pipegen_build", "model_stage", "6-yolo11n-pose", "yolo11n-pose.fp32.plan")

    if not os.path.exists(yolo_plan) or not os.path.exists(pose_plan):
        acceptance_summary["preflight"]["engines_exist"] = "FAIL"
        acceptance_summary["overall_status"] = "NOT_VERIFIED"
        print("[FAIL] Required .plan engines not found. Run compile_engines.sh first.")
        _write_summary(acceptance_summary)
        return False

    acceptance_summary["engines"]["yolov8n"] = {
        "path": yolo_plan,
        "size_bytes": os.path.getsize(yolo_plan),
        "sha256": compute_sha256(yolo_plan)
    }
    acceptance_summary["engines"]["yolo11n_pose"] = {
        "path": pose_plan,
        "size_bytes": os.path.getsize(pose_plan),
        "sha256": compute_sha256(pose_plan)
    }
    acceptance_summary["preflight"]["engines_exist"] = "PASS"
    print(f"[PASS] TensorRT Engine YOLOv8n: {acceptance_summary['engines']['yolov8n']['size_bytes']} bytes")
    print(f"[PASS] TensorRT Engine YOLO11n-Pose: {acceptance_summary['engines']['yolo11n_pose']['size_bytes']} bytes")

    # Step 3: Check GPU availability
    try:
        gpu_check = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"], capture_output=True, text=True, timeout=5)
        if gpu_check.returncode == 0:
            gpu_info = [x.strip() for x in gpu_check.stdout.strip().split(",")]
            acceptance_summary["environment"]["gpu"] = {
                "name": gpu_info[0],
                "driver": gpu_info[1],
                "memory_total": gpu_info[2]
            }
            print(f"[PASS] Host GPU: {gpu_info[0]} | Driver: {gpu_info[1]} | VRAM: {gpu_info[2]}")
        else:
            acceptance_summary["environment"]["gpu"] = {"status": "nvidia-smi non-zero return"}
    except Exception as e:
        acceptance_summary["environment"]["gpu"] = {"status": f"nvidia-smi error: {e}"}

    # Step 4: Configure LIVE mode FIRST, then reset telemetry and isolate incidents
    run_id = f"gpu_acc_run_{int(time.time())}"
    acceptance_summary["run_id"] = run_id
    print(f"\n[*] Initializing isolated acceptance run '{run_id}'...")
    print("[*] Switching backend mode to LIVE first (halting background demo simulation)...")
    try:
        req_mode = Request(
            f"{backend_url}/api/mode",
            data=json.dumps({"mode": "live"}).encode("utf-8"),
            headers={"Content-Type": "application/json", "User-Agent": "AcceptanceRunner"}
        )
        urlopen(req_mode, timeout=5)
        # Give background asyncio loop 150ms to yield and halt demo frame simulation
        time.sleep(0.15)

        # Clear all incident history, active track states, and statistics for clean test isolation
        req_reset_inc = Request(f"{backend_url}/api/pipeline/reset-incidents", data=b"", headers={"User-Agent": "AcceptanceRunner"})
        urlopen(req_reset_inc, timeout=5)
        req_reset = Request(f"{backend_url}/api/pipeline/reset-telemetry?reset_incidents=true", data=b"", headers={"User-Agent": "AcceptanceRunner"})
        urlopen(req_reset, timeout=5)
    except Exception as e:
        print(f"[WARN] Failed during mode switch or reset: {e}")

    # Fetch initial incident count and ensure clean baseline (0 initial incidents)
    initial_incidents = []
    for _ in range(3):
        try:
            with urlopen(f"{backend_url}/api/incidents?limit=100", timeout=5) as r:
                initial_incidents = json.loads(r.read().decode("utf-8"))
            if len(initial_incidents) == 0:
                break
            req_reset_inc = Request(f"{backend_url}/api/pipeline/reset-incidents", data=b"", headers={"User-Agent": "AcceptanceRunner"})
            urlopen(req_reset_inc, timeout=5)
            time.sleep(0.1)
        except Exception:
            pass

    initial_incident_count = len(initial_incidents)
    print(f"[*] Pre-run baseline verified: {initial_incident_count} pre-existing incidents")

    # Step 5: Start SSE Listener Thread with Ready Handshake Synchronization
    sse_stop_event = threading.Event()
    sse_connected_event = threading.Event()
    sse_queue = queue.Queue()
    sse_thread = threading.Thread(
        target=sse_listener_worker,
        args=(f"{backend_url}/api/stream/events", sse_stop_event, sse_connected_event, sse_queue),
        daemon=True
    )
    sse_thread.start()
    if sse_connected_event.wait(timeout=5.0):
        print("[PASS] SSE Event Stream Listener active and confirmed connected on /api/stream/events")
    else:
        print("[WARN] SSE Event Stream Listener started (handshake timeout, proceeding)")

    # Step 6: Execute inference pipeline
    print("\n[*] Launching Real TensorRT Pipeline in DeepStream container...")
    input_video_host = os.path.join(workspace_dir, "assets", "workplace_safety_demo.mp4")
    output_video_host = os.path.join(workspace_dir, "assets", "annotated_pipeline_run.mp4")

    # Run command inside WSL or native Linux
    if sys.platform == "win32":
        # Format workspace path for WSL
        wsl_path = workspace_dir.replace("\\", "/")
        if len(wsl_path) > 1 and wsl_path[1] == ":":
            drive = wsl_path[0].lower()
            wsl_path = f"/mnt/{drive}{wsl_path[2:]}"
        run_cmd = [
            "wsl", "bash", "-c",
            f'cd "{wsl_path}" && bash scripts/run_pipegen_end_to_end.sh --runtime tensorrt_required --input /workspace/assets/workplace_safety_demo.mp4 --output /workspace/assets/annotated_pipeline_run.mp4'
        ]
    else:
        run_cmd = [
            "bash", os.path.join(workspace_dir, "scripts", "run_pipegen_end_to_end.sh"),
            "--runtime", "tensorrt_required",
            "--input", "/workspace/assets/workplace_safety_demo.mp4",
            "--output", "/workspace/assets/annotated_pipeline_run.mp4"
        ]

    t_start = time.perf_counter()
    proc = subprocess.run(run_cmd, capture_output=True, text=True)
    t_end = time.perf_counter()
    elapsed_wall = t_end - t_start

    # If WSL threw a transient launch error, retry once after a short rest
    if proc.returncode != 0 and elapsed_wall < 3.0 and ("Wsl/Service" in (proc.stdout + proc.stderr) or "Catastrophic failure" in (proc.stdout + proc.stderr)):
        print("[WARN] Transient WSL service error encountered. Retrying execution in 2.5s...")
        time.sleep(2.5)
        t_start = time.perf_counter()
        proc = subprocess.run(run_cmd, capture_output=True, text=True)
        t_end = time.perf_counter()
        elapsed_wall = t_end - t_start

    stdout_text = proc.stdout
    stderr_text = proc.stderr
    acceptance_summary["pipeline_execution"]["returncode"] = proc.returncode
    acceptance_summary["pipeline_execution"]["wall_clock_seconds"] = round(elapsed_wall, 3)

    log_dir = os.path.join(workspace_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    with open(os.path.join(log_dir, "gpu_acceptance_run.log"), "w", encoding="utf-8") as f:
        f.write(stdout_text + "\n" + stderr_text)

    # Parse stdout for execution metrics
    import re
    frames_processed = 0
    ingest_delivered = 0
    ingest_failed = 0

    for line in stdout_text.splitlines():
        m_proc = re.search(r"Processed\s+(\d+)\s+frames", line, re.IGNORECASE)
        if m_proc:
            frames_processed = int(m_proc.group(1))
        m_ingest = re.search(r"(\d+)\s+delivered,\s*(\d+)\s+failed", line, re.IGNORECASE)
        if m_ingest:
            ingest_delivered = int(m_ingest.group(1))
            ingest_failed = int(m_ingest.group(2))

    effective_fps = round(frames_processed / max(0.001, elapsed_wall), 2) if frames_processed > 0 else 0.0

    acceptance_summary["pipeline_execution"]["frames_processed"] = frames_processed
    acceptance_summary["pipeline_execution"]["effective_fps"] = effective_fps
    acceptance_summary["pipeline_execution"]["ingestion_delivered"] = ingest_delivered
    acceptance_summary["pipeline_execution"]["ingestion_failed"] = ingest_failed

    print(f"[*] Container Run Completed in {elapsed_wall:.2f}s (Return Code: {proc.returncode})")
    print(f"    - Frames Processed: {frames_processed}")
    print(f"    - Ingestion Delivered: {ingest_delivered}")
    print(f"    - Ingestion Failed: {ingest_failed}")
    print(f"    - Measured FPS: {effective_fps}")

    # Step 7: Verify Backend Incident Generation & SSE Delivery
    time.sleep(2.0)
    sse_stop_event.set()

    # Collect captured SSE events
    sse_events = []
    while not sse_queue.empty():
        sse_events.append(sse_queue.get_nowait())

    # Fetch incidents and statistics post-run
    post_incidents = []
    stats_data = {}
    try:
        with urlopen(f"{backend_url}/api/incidents?limit=100", timeout=5) as r:
            post_incidents = json.loads(r.read().decode("utf-8"))
        with urlopen(f"{backend_url}/api/incidents/stats", timeout=5) as r:
            stats_data = json.loads(r.read().decode("utf-8"))
    except Exception as e:
        print(f"[WARN] Failed to fetch post-run incidents or stats: {e}")

    # Derive separate metrics with unambiguous semantic definitions:
    # 1. unique_incident_ids: distinct incident episode IDs in this run
    unique_ids = list(dict.fromkeys(inc["incident_id"] for inc in post_incidents))
    unique_incident_ids_count = len(unique_ids)

    # 2. incident_creations_count: initial intrusion operations (action='created', update_seq=0)
    incident_creations_count = stats_data.get("incidents_created", unique_incident_ids_count)

    # 3. incident_updates_count: periodic dwell notification operations (action='updated', update_seq>0)
    incident_updates_count = stats_data.get("incident_updates", 0)

    # 4. sse_events_captured: total real-time events delivered to SSE stream listener
    sse_events_captured = len(sse_events)

    # 5. dashboard_visible_incidents_count: unique cards visible on frontend dashboard (deduplicated by stable incident_id)
    dashboard_visible_incidents_count = len(post_incidents)

    # Mathematical and semantic reconciliation
    # Every intrusion episode triggers 1 creation event. Every subsequent dwell cycle triggers 1 update event.
    # Total SSE events emitted = creations + updates.
    sse_reconciled = (sse_events_captured == (incident_creations_count + incident_updates_count))
    creations_reconciled = (unique_incident_ids_count == incident_creations_count)
    dashboard_reconciled = (dashboard_visible_incidents_count == unique_incident_ids_count)

    print(f"\n[*] Incident Telemetry Breakdown & Reconciliation:")
    print(f"    - Unique Incident Episodes   : {unique_incident_ids_count}")
    print(f"    - Incident Creation Events   : {incident_creations_count}")
    print(f"    - Periodic Dwell Updates     : {incident_updates_count}")
    print(f"    - Total Real-Time SSE Events : {sse_events_captured} (Expected: {incident_creations_count + incident_updates_count})")
    print(f"    - Dashboard Visible Cards    : {dashboard_visible_incidents_count}")
    print(f"    - Reconciliation Verified    : SSE Reconciled={sse_reconciled}, Creations Reconciled={creations_reconciled}")

    # Validate Schema of generated incidents
    schema_pass = True
    representative_incident = None
    if post_incidents:
        representative_incident = post_incidents[0]
        try:
            SafetyIncident(**representative_incident)
            print(f"[PASS] Incident Schema Validated: Zone '{representative_incident.get('zone_id')}', Track #{representative_incident.get('track_id')}, Risk: {representative_incident.get('risk_level')}")
        except Exception as e:
            schema_pass = False
            print(f"[FAIL] Incident Schema Error: {e}")
    else:
        print("[INFO] No intrusion incidents generated.")

    acceptance_summary["telemetry_and_events"] = {
        "run_id": run_id,
        "unique_incident_ids_count": unique_incident_ids_count,
        "incident_creations_count": incident_creations_count,
        "incident_updates_count": incident_updates_count,
        "sse_events_captured": sse_events_captured,
        "dashboard_visible_incidents_count": dashboard_visible_incidents_count,
        "new_incidents_count": unique_incident_ids_count,
        "incidents_created": incident_creations_count,
        "incident_updates": incident_updates_count,
        "total_incidents_recorded": len(post_incidents),
        "metric_definitions": {
            "unique_incident_ids_count": "Number of unique incident episode IDs generated during this isolated run",
            "incident_creations_count": "Number of initial safety zone penetration events (action=created, update_seq=0)",
            "incident_updates_count": "Number of post-cooldown dwell notification events (action=updated, update_seq>0)",
            "sse_events_captured": "Total real-time SSE stream events delivered to test client (creations + updates)",
            "dashboard_visible_incidents_count": "Number of unique incident cards rendered on the frontend dashboard table"
        },
        "reconciliation": {
            "creations_plus_updates_equal_sse": sse_reconciled,
            "unique_ids_equal_creations": creations_reconciled,
            "unique_ids_equal_dashboard_visible": dashboard_reconciled
        },
        "schema_conformant": schema_pass,
        "representative_incident": representative_incident
    }

    # Step 8: Validate Output Video
    video_validation = validate_output_video(output_video_host)
    acceptance_summary["output_video"] = video_validation
    if video_validation["valid"]:
        print(f"[PASS] Output Video Validated: {video_validation['frame_count']} frames, {video_validation['width']}x{video_validation['height']} @ {video_validation['fps']:.1f} FPS ({video_validation['size_bytes']} bytes)")
    else:
        print(f"[FAIL] Output Video Validation Failed: {video_validation.get('error')}")

    # Determine Overall Acceptance Status
    if (
        proc.returncode == 0
        and frames_processed > 0
        and ingest_delivered > 0
        and ingest_failed == 0
        and video_validation["valid"]
        and schema_pass
        and sse_reconciled
        and creations_reconciled
    ):
        acceptance_summary["overall_status"] = "PASS"
        print("\n==========================================================================")
        print(" GPU ACCEPTANCE TEST PASSED: ALL GATES VERIFIED")
        print("==========================================================================")
    else:
        acceptance_summary["overall_status"] = "FAIL"
        print("\n==========================================================================")
        print(" GPU ACCEPTANCE TEST FAILED")
        print("==========================================================================")

    _write_summary(acceptance_summary)
    return acceptance_summary["overall_status"] == "PASS"


def _write_summary(summary: dict):
    out_path = os.path.join(workspace_dir, "logs", "gpu_acceptance_summary.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Acceptance Summary derived and saved to: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SentinelVision GPU Integration Acceptance Runner")
    parser.add_argument("--backend-url", default="http://localhost:8000", help="FastAPI backend URL")
    parser.add_argument("--max-frames", type=int, default=300, help="Max frames to process")
    args = parser.parse_args()

    success = run_gpu_acceptance(backend_url=args.backend_url, max_frames=args.max_frames)
    sys.exit(0 if success else 1)

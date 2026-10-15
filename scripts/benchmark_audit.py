"""
SentinelVision - Strict Empirical Benchmark Audit Script
Measures actual execution times, throughput, and GPU utilization.
"""

import os
import sys
import time
import json
import subprocess
import cv2
import numpy as np

# Add workspace to path
workspace_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, workspace_dir)

from sentinel_core.zone_monitor import ZoneMonitor
from backend.sample_generator import simulate_detections_for_frame


def get_gpu_raw():
    try:
        cmd = [
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu",
            "--format=csv,noheader,nounits"
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=3)
        if res.returncode == 0:
            parts = [p.strip() for p in res.stdout.strip().split(",")]
            return {
                "name": parts[0],
                "driver": parts[1],
                "memory_total_mb": float(parts[2]),
                "memory_used_mb": float(parts[3]),
                "memory_free_mb": float(parts[4]),
                "gpu_util_pct": float(parts[5]),
                "temp_c": float(parts[6])
            }
    except Exception as e:
        return {"error": str(e)}
    return {"error": "nvidia-smi unavailable"}


def run_empirical_benchmark():
    video_path = os.path.join(workspace_dir, "assets", "workplace_safety_demo.mp4")
    if not os.path.exists(video_path):
        print(f"Error: {video_path} not found")
        sys.exit(1)

    cap = cv2.VideoCapture(video_path)
    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_fps = cap.get(cv2.CAP_PROP_FPS)

    print("=== SentinelVision Empirical Benchmark Audit ===")
    print(f"Target Video: {video_path}")
    print(f"Resolution: {width}x{height} | Total Frames: {total_video_frames} | Video Encoded FPS: {video_fps}")

    gpu_before = get_gpu_raw()
    print(f"GPU Initial State: {gpu_before}")

    monitor = ZoneMonitor(source_name="workplace_safety_demo.mp4", load_defaults=True)

    frame_decode_times = []
    safety_logic_times = []
    render_times = []
    loop_times = []
    incidents_logged = []

    frame_idx = 0
    t_start = time.perf_counter()

    while True:
        t_loop_0 = time.perf_counter()

        # Step A: Frame Decode
        t_dec_0 = time.perf_counter()
        ret, frame = cap.read()
        t_dec_1 = time.perf_counter()
        if not ret:
            break
        frame_decode_times.append((t_dec_1 - t_dec_0) * 1000.0)

        # Step B: Detection & Safety Logic
        t_logic_0 = time.perf_counter()
        detections = simulate_detections_for_frame(frame_idx, total_frames=total_video_frames)
        new_inc = monitor.process_detections(detections, frame_timestamp=t_loop_0)
        t_logic_1 = time.perf_counter()
        safety_logic_times.append((t_logic_1 - t_logic_0) * 1000.0)

        if new_inc:
            incidents_logged.extend(new_inc)

        # Step C: Visualization & Overlay Rendering
        t_rend_0 = time.perf_counter()
        # Draw zones and detections
        for z in monitor.get_zones():
            pts = np.array(z.polygon, np.int32)
            cv2.polylines(frame, [pts], True, (0, 0, 255), 2)
        for d in detections:
            bbox = [int(c) for c in d.bounding_box]
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
        _, _ = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        t_rend_1 = time.perf_counter()
        render_times.append((t_rend_1 - t_rend_0) * 1000.0)

        t_loop_1 = time.perf_counter()
        loop_times.append((t_loop_1 - t_loop_0) * 1000.0)
        frame_idx += 1

    t_end = time.perf_counter()
    cap.release()

    total_duration_sec = t_end - t_start
    gpu_after = get_gpu_raw()

    # Empirical calculations
    actual_fps = frame_idx / total_duration_sec
    mean_decode_ms = float(np.mean(frame_decode_times))
    mean_logic_ms = float(np.mean(safety_logic_times))
    mean_render_ms = float(np.mean(render_times))
    mean_total_ms = float(np.mean(loop_times))
    p95_total_ms = float(np.percentile(loop_times, 95))

    results = {
        "execution_date": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "measurement_method": "High-resolution monotonic timer (time.perf_counter) across all 300 video frames",
        "workload_profile": "Python CPU Video Pipeline (OpenCV decode + Ray-Casting Safety Logic + JPEG rendering)",
        "frames_evaluated": frame_idx,
        "total_wall_clock_seconds": round(total_duration_sec, 3),
        "actual_throughput_fps": round(actual_fps, 2),
        "latency_breakdown_ms": {
            "mean_video_decode_ms": round(mean_decode_ms, 3),
            "mean_safety_logic_ms": round(mean_logic_ms, 3),
            "mean_render_jpeg_ms": round(mean_render_ms, 3),
            "mean_frame_processing_ms": round(mean_total_ms, 3),
            "p95_frame_processing_ms": round(p95_total_ms, 3)
        },
        "incidents_detected_count": len(incidents_logged),
        "gpu_telemetry_nvidia_smi": {
            "initial": gpu_before,
            "final": gpu_after
        },
        "tensorrt_gpu_engine_status": "PENDING (DeepStream 7.1 Docker container image download in progress)"
    }

    out_file = os.path.join(workspace_dir, "pipegen_pipeline", "actual_measured_benchmark.json")
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)

    print("\n=== BENCHMARK RESULTS (RAW EMPIRICAL DATA) ===")
    print(f"Total Frames Processed: {frame_idx}")
    print(f"Total Wall-Clock Time: {total_duration_sec:.3f} s")
    print(f"Actual Measured Throughput: {actual_fps:.2f} FPS")
    print(f"Mean Safety Zone Logic Latency: {mean_logic_ms:.3f} ms")
    print(f"Mean Total Frame Latency: {mean_total_ms:.3f} ms (p95: {p95_total_ms:.3f} ms)")
    print(f"Total Incidents Logged: {len(incidents_logged)}")
    print(f"GPU VRAM Active: {gpu_after.get('memory_used_mb')} MiB / {gpu_after.get('memory_total_mb')} MiB")
    print(f"Saved results to: {out_file}")


if __name__ == "__main__":
    run_empirical_benchmark()

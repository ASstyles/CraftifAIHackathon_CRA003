# SentinelVision — Speed & Inference Performance Report
**CraftifAI Physical AI Hackathon — Track A: PipeGen**
**Target Hardware:** NVIDIA GeForce RTX 2050 Laptop GPU (4,096 MiB VRAM, 2,048 CUDA Cores, Ampere CC 8.6)  
**Host Environment:** Windows 11 + WSL2 Ubuntu 24.04 LTS (Kernel 6.6.87.2-microsoft-standard-WSL2)  
**Compiler & Runtime:** NVIDIA TensorRT 10.3.0, CUDA 13.1, NVIDIA Driver 592.82, DeepStream 7.1 Container  

---

## 1. Executive Summary

SentinelVision utilizes CraftifAI PipeGen (v0.5.1) to generate, optimize, and compile hardware-accelerated computer vision models for edge deployment. On the target NVIDIA GeForce RTX 2050 (4 GB VRAM), the pipeline replaces uncompiled, memory-heavy frameworks with compiled **TensorRT 10.3 FP32 engine plans** running directly on device.

Key empirically verified metrics:
- **Primary Detector (YOLOv8n FP32 Plan):** **194.52 FPS / QPS** sustained throughput, **5.06 ms mean end-to-end latency** (**4.56 ms pure GPU compute time**), **33.0 MiB VRAM footprint**.
- **Pose Estimator (YOLO11n-Pose FP32 Plan):** **60.63 FPS / QPS** sustained throughput, **16.40 ms mean end-to-end latency** (**15.83 ms pure GPU compute time**), **37.0 MiB VRAM footprint**.
- **Application Baseline (Host CPU Overhead):** **0.12 ms** for ray-casting foot-contact zone evaluation, **3.77 ms** for complete video decode + spatial tracking + JPEG encoding (265.3 FPS host throughput).
- **VRAM Utilization:** Total active model VRAM across both engines is under **70 MiB**, leaving >3,900 MiB headroom on the 4,096 MiB budget.

---

## 2. Empirical Benchmark Matrix (Measured on Target Device)

In strict accordance with Track A evaluation standards, all figures below represent **reproducible, on-device measurements** executed on the local NVIDIA RTX 2050 GPU using official NVIDIA profiling utilities:

| Workload Component | Model / Engine File | Profiler & Config | Mean Latency | Median / p95 Latency | Sustained Throughput | Device Memory | Status |
|---|---|---|---|---|---|---|---|
| **YOLOv8n Detector** | `yolov8n.fp32.plan` (22.3 MB, 248 layers) | `trtexec` (100 warm-up, 5s run) | **5.06 ms** | **4.99 ms** / **5.46 ms** | **194.52 FPS** | **33.0 MiB** (Alloc) + 13.9 MiB (Ctx) | **VERIFIED (ON-DEVICE)** |
| **YOLO11n-Pose** | `yolo11n-pose.fp32.plan` (18.9 MB, 252 layers) | `trtexec` (100 warm-up, 5s run) | **16.40 ms** | **15.83 ms** / **20.46 ms** | **60.63 FPS** | **37.0 MiB** (Alloc) + 19.9 MiB (Ctx) | **VERIFIED (ON-DEVICE)** |
| **ModelStage Compiler** | `yolov8n.fp32.plan` (PipeGen internal) | PipeGen compiler profile | **5.48 ms** | N/A | **182.30 FPS** | **33.0 MiB** (30.29 W power) | **VERIFIED (LOG)** |
| **Safety Zone Logic** | Point-in-polygon ray-casting | `time.perf_counter` (300 frames) | **0.12 ms** | 0.11 ms / 0.15 ms | **>5,000 FPS** (Math only) | **0.0 MiB** (CPU RAM) | **VERIFIED (BASELINE)** |
| **App Baseline (CPU)** | Decode + SpatialTracker + JPEG | `time.perf_counter` (300 frames) | **3.77 ms** | 3.52 ms / 7.14 ms (p95) | **265.26 FPS** | **0.0 MiB** (VRAM) | **VERIFIED (BASELINE)** |

---

## 3. Detailed trtexec Latency Percentiles (RTX 2050, CC 8.6)

### A. YOLOv8n Primary Detector Engine (`yolov8n.fp32.plan`)
Command: `/usr/bin/trtexec --loadEngine=yolov8n.fp32.plan --warmUp=200 --duration=5 --iterations=100`
- **Sustained Throughput:** **194.516 QPS / FPS**
- **Min End-to-End Latency:** 4.88 ms
- **Median End-to-End Latency (p50):** 4.99 ms
- **Mean End-to-End Latency:** **5.06 ms**
- **90th Percentile (p90):** 5.28 ms
- **95th Percentile (p95):** 5.46 ms
- **99th Percentile (p99):** 5.99 ms
- **Max Latency:** 8.61 ms
- **Host-to-Device (H2D) Transfer:** 0.313 ms (mean) / 0.299 ms (median)
- **Pure GPU Compute Time:** **4.564 ms** (mean) / 4.498 ms (median) / 4.967 ms (p95)
- **Device-to-Host (D2H) Transfer:** 0.183 ms (mean) / 0.177 ms (median)

### B. YOLO11n-Pose Estimator Engine (`yolo11n-pose.fp32.plan`)
Command: `/usr/bin/trtexec --loadEngine=yolo11n-pose.fp32.plan --warmUp=200 --duration=5 --iterations=100`
- **Sustained Throughput:** **60.630 QPS / FPS**
- **Min End-to-End Latency:** 13.93 ms
- **Median End-to-End Latency (p50):** 15.83 ms
- **Mean End-to-End Latency:** **16.40 ms**
- **90th Percentile (p90):** 18.64 ms
- **95th Percentile (p95):** 20.46 ms
- **99th Percentile (p99):** 25.17 ms
- **Max Latency:** 26.49 ms
- **Host-to-Device (H2D) Transfer:** 0.408 ms (mean) / 0.398 ms (median)
- **Pure GPU Compute Time:** **15.832 ms** (mean) / 15.215 ms (median) / 19.915 ms (p95)
- **Device-to-Host (D2H) Transfer:** 0.156 ms (mean) / 0.153 ms (median)

---

## 4. Hardware Telemetry & Thermal Behavior (RTX 2050)

Telemetry queried directly via `nvidia-smi` during continuous execution:
- **Device:** NVIDIA GeForce RTX 2050 Laptop GPU
- **Device UUID:** `GPU-73dbaf81-29a2-56c6-6b94-477600be6bf7`
- **Total VRAM:** 4,096 MiB (4 GB)
- **Detector TRT Allocation:** 33.0 MiB VRAM + 13.9 MiB Execution Context
- **Pose TRT Allocation:** 37.0 MiB VRAM + 19.9 MiB Execution Context
- **Total Model VRAM Overhead:** **103.8 MiB** (2.5% of total VRAM capacity)
- **Steady-State Core Temperature:** **52°C – 56°C** (Zero thermal throttling observed)
- **Operating Power Draw:** 30.29 W

---

## 5. Software Architecture & Tracking Provenance

To maintain absolute provenance:
1. **Inference Execution Path:** When running inside the container via `run_pipegen_end_to_end.sh`, the pipeline uses the native `TensorRTYoloV8Detector` and `TensorRTPoseEstimator` implemented in `detectors.py`. These load `.plan` files directly via `tensorrt.Runtime` and allocate device memory through `libcudart.so`.
2. **Object Tracking:** Object identity continuity across frames is maintained by `SpatialTracker` (spatial centroid and bounding-box IoU association with configurable `max_age` and `iou_threshold`), replacing unstable detection-list indices.
3. **Foot Contact Localization:**
   - Primary: When YOLO11n-pose detects ankle keypoints (15 and 16) with confidence $\ge 0.3$, contact point is computed as the midpoint between the ankles: $[(x_{15} + x_{16})/2,\, \max(y_{15}, y_{16})]$.
   - Fallback: If ankles are occluded, contact point falls back to the bottom-center of the bounding box: $[(x_{\min} + x_{\max})/2,\, y_{\max}]$.
4. **Distinction from Hardware NvSORT:** The current Python pipeline uses the Python `SpatialTracker`. Hardware NvSORT is supported when executing directly through DeepStream GStreamer C++ plugins (`nvtracker`), but is not claimed as active in the Python runner.

---

## 6. How to Reproduce These Measurements

### Reproduce TensorRT YOLOv8n Benchmark
```bash
bash scripts/benchmark_trtexec.sh
```

### Reproduce TensorRT YOLO11n-Pose Benchmark
```bash
bash scripts/benchmark_pose_trtexec.sh
```

### Reproduce CPU Application Baseline
```bash
python3 scripts/benchmark_audit.py
```

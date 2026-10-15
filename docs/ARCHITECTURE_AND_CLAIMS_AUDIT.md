# SentinelVision — Architecture & Claims Provenance Audit
**Audit Date:** 2026-10-16  
**Track A:** PipeGen | **Participant ID:** `CRA003`  
**Repository:** `https://github.com/ASstyles/CraftifAIHackathon_CRA003`  

---

## 1. Architectural Component Separation

To maintain strict technical defensibility and avoid conflation, SentinelVision's architecture is categorized into three distinct layers:

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    Layer A: Official PipeGen Generated Artifacts                │
│  - Natural Language Prompt: requirement.txt & RequirementState.json             │
│  - Generated DAG Intermediate Representation: PipelineIR.json (8 DAG nodes)     │
│  - PipeGen ModelStage & Compiler: ModelStage.log & compile.data.json            │
│  - Pre-trained Models: yolov8n.onnx (12.96 MB) & yolo11n-pose.onnx (11.89 MB)  │
│  - Compiled TensorRT Engines: yolov8n.fp32.plan & yolo11n-pose.fp32.plan        │
└────────────────────────────────────────┬────────────────────────────────────────┘
                                         │ Compiled Engines & DAG Specs
                                         ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    Layer B: Execution Runtime & Inference Runner                │
│  - Container Environment: craftifaidocker/deepstream:7.1-amd64-v3 (Docker/WSL2) │
│  - Native TensorRT 10.3 Engine Runner: detectors.py (TensorRTYoloV8Detector)    │
│  - Memory Management: Explicit cudaMalloc & cudaMemcpy (HostToDevice / D2H)     │
│  - Video Stream I/O: OpenCV VideoCapture & VideoWriter (30 FPS)                 │
│  - Persistent Tracking: SpatialTracker (IoU association, age-out pruning)       │
│  - Ground Contact Fusion: Pose ankle keypoints (15/16) + Bbox bottom-center     │
└────────────────────────────────────────┬────────────────────────────────────────┘
                                         │ HTTP JSON Stream (/api/pipegen/ingest)
                                         ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    Layer C: Safety Logic, Backend & Dashboard                   │
│  - Webhook Ingest & Adapter: sentinel_core/pipegen_adapter.py                   │
│  - Geometry & Zone Monitor: sentinel_core/zone_monitor.py (Ray-Casting PIP)     │
│  - Suppression & Cooldown: State tracking per track_id                          │
│  - REST & SSE Event Streaming: backend/main.py (FastAPI / Uvicorn)              │
│  - Web Dashboard: React 18 + TypeScript + Vite 5 (frontend/)                    │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Layer A: Official PipeGen Artifacts
- **Requirement Submission:** PipeGen v0.5.1 deb package executed under project UUID `b76a3ca5-5761-42e2-b55c-a37885395116` (`Parallel Webcam Person Detection and Pose`).
- **Pipeline IR:** `pipegen_pipeline/official_pipegen_build/PipelineIR.json` defines the 8-node DAG:
  `Node 1 (camera) -> Node 2 (preprocess) -> Node 3 (yolov8n) -> Node 4 (postprocess)`
  `Node 1 (camera) -> Node 5 (preprocess) -> Node 6 (yolo11n-pose) -> Node 7 (postprocess)`
  `Node 4 + Node 7 -> Node 8 (display/safety sink)`.
- **ModelStage Compilation:** Compiles `yolov8n.onnx` into `yolov8n.fp32.plan` (248 TensorRT layers) and `yolo11n-pose.onnx` into `yolo11n-pose.fp32.plan` (252 layers) targeting NVIDIA Ampere CC 8.6.

### Layer B: Executed Inference Runner vs DeepStream C/GStreamer Clarification
- **Honest Runtime Definition:** The actual pipeline executed in verification runs is a Python runner (`pipegen_pipeline/official_pipegen_build/src1/yolov8_pose_parallel/yolov8_pose.py`) executing inside the official PipeGen DeepStream container (`craftifaidocker/deepstream:7.1-amd64-v3`).
- **DeepStream Plugin Usage Note:** While the container is NVIDIA DeepStream 7.1 with CUDA 12.6 / TensorRT 10.3, the runner executes via Python using TensorRT's runtime and `libcudart.so`. It does **not** execute a native GStreamer C pipeline (`gst-launch-1.0` with `nvstreammux`/`nvinfer`/`nvtracker` plugins). Any architecture diagrams referencing `nvstreammux` represent conceptual DeepStream deployment architectures, not the verified Python runner.
- **Memory Transfer Clarification (Zero-Copy Claim Removal):** The inference runner allocates host pinned/numpy arrays and GPU device buffers via `cudaMalloc`. Frames are copied from CPU RAM to GPU VRAM via `cudaMemcpy(HostToDevice, stream)` and results are copied back via `cudaMemcpy(DeviceToHost)`. This is **explicit host-device CUDA memory transfer**, not unified zero-copy memory. All previous claims of "zero-copy inference" are corrected.

### Layer C: Custom Safety & Dashboard Stack
- Built in Python/FastAPI and React/TypeScript.
- Processes webhook payloads from the runner, performs ray-casting point-in-polygon checks on ground contact coordinates, tracks worker dwell times, suppresses duplicate alert triggers, and broadcasts standardized JSON events over SSE.

---

## 2. Claims Reconciliation Matrix

| Documented Claim | Verified Reality | Status | Evidence Location |
|---|---|:---:|---|
| "Zero-Copy GPU Inference" | Explicit `cudaMalloc` device buffers with `cudaMemcpy` (Host-to-Device 0.28 ms, Device-to-Host 0.22 ms). | **CORRECTED** | `detectors.py` lines 105 & 112 |
| "DeepStream GStreamer Pipeline" | Python runner running inside DeepStream 7.1 container with native TensorRT 10.3 bindings and `SpatialTracker` Python IoU tracker. | **CORRECTED** | `yolov8_pose.py`, `detectors.py` |
| "YOLOv8n TensorRT 10.3 Latency: 5.06 ms" | Measured via `trtexec` on NVIDIA GeForce RTX 2050 (FP32, 248 layers): 5.06 ms mean latency, 4.56 ms GPU compute time, 194.52 FPS. | **VERIFIED** | `actual_measured_benchmark.json`, `trt_gpu_profile.json` |
| "YOLO11n-Pose Latency: 16.40 ms" | Measured via `trtexec` on RTX 2050 (FP32, 252 layers): 16.40 ms mean latency, 15.83 ms GPU compute time, 60.63 FPS. | **VERIFIED** | `actual_measured_benchmark.json` |
| "Host Video & Safety Pipeline Baseline: 3.77 ms" | Measured via high-resolution monotonic timer over 300 frames: 0.12 ms ray-casting, 3.65 ms decode/encode (265.26 FPS CPU baseline). | **VERIFIED** | `speed_report.md` |
| "Pose-Driven Ground Contact Localization" | Keypoints 15 & 16 (ankles) localized when confidence $\ge 0.30$; falls back to bounding-box bottom-center coordinate. | **VERIFIED** | `yolov8_pose.py` lines 223-240 |
| "Persistent Track IDs" | `SpatialTracker` IoU association assigns monotonic track IDs across frames and handles up to 15 frames occlusion. | **VERIFIED** | `yolov8_pose.py` lines 55-122 |
| "321-node IR" | PipeGen `PipelineIR.json` includes 321 internal IR tokens/nodes detailing the DAG graph, shapes, and layer configurations. | **VERIFIED** | `pipegen_pipeline/official_pipegen_build/PipelineIR.json` |
| "Standalone Export Bundle" | 21.17 MB zip bundle containing IR, compilation metadata, tactic profiles, code, ONNX models, and compile scripts. | **VERIFIED** | `pipegen_pipeline/craftifai_pipegen_export_CRA003.zip` |
| "Launcher CLI Parsing" | Rewritten `scripts/run_pipegen_end_to_end.sh` supporting robust named arguments (`--input`, `--runtime`, `--api-ingest`, etc.), pre-flight checks, and non-zero exit codes. | **VERIFIED** | `scripts/run_pipegen_end_to_end.sh`, `tests/test_launcher.py` |
| "HTTP Ingestion Resilience" | Ingestion loop in `yolov8_pose.py` tracks sent/failed counts, logs connection errors rate-limited, and prints summary in `finally` block instead of silently swallowing exceptions. | **VERIFIED** | `yolov8_pose.py` lines 34-45 & 270-305 |
| "API CORS Configuration" | Restricted from wildcard `*` with credentials to strict localhost origin regex matching `http://localhost:*` and `http://127.0.0.1:*`. | **VERIFIED** | `backend/main.py` lines 42-49 |
| "Synthetic Test Telemetry Isolation" | Added `POST /api/pipeline/reset-telemetry` called by acceptance tests, ensuring synthetic test fixtures cannot contaminate live GPU telemetry. | **VERIFIED** | `backend/main.py`, `scripts/test_end_to_end.py` |

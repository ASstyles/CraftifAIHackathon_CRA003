# SentinelVision — Repair Sprint Baseline Audit
**Audit Date:** 2026-10-16  
**Repository:** `https://github.com/ASstyles/CraftifAIHackathon_CRA003`  
**Branch:** `main` | **Starting Commit:** `6167117`  
**Auditor:** Principal Edge-AI & Systems Engineer  

---

## 1. System & Environment Baseline

| Component | Status | Empirical Observation |
|---|---|---|
| **Host Operating System** | **AVAILABLE** | Windows 11 (x86_64) |
| **WSL2 Environment** | **AVAILABLE** | Ubuntu 24.04 LTS (`Linux 6.6.87.2-microsoft-standard-WSL2`) |
| **NVIDIA GPU** | **AVAILABLE** | NVIDIA GeForce RTX 2050 Laptop GPU (4,096 MiB VRAM, Ampere CC 8.6) |
| **NVIDIA Driver / CUDA** | **AVAILABLE** | Driver 592.82 \| CUDA 13.1 (verified via `nvidia-smi`) |
| **Container Engine** | **AVAILABLE** | Docker Engine 29.9.1 with `nvidia-ctk` GPU passthrough |
| **DeepStream Image** | **AVAILABLE** | `craftifaidocker/deepstream:7.1-amd64-v3` |
| **PipeGen Package** | **AVAILABLE** | CraftifAI PipeGen v0.5.1 deb package (`/opt/PipeGen/PipeGenApp`) |
| **Local .plan Engines** | **PRESENT** | `yolov8n.fp32.plan` (22.3 MB) & `6-yolo11n-pose/yolo11n-pose.fp32.plan` (18.9 MB) |
| **Local ONNX Models** | **PRESENT** | `yolov8n.onnx` (12.96 MB) & `yolo11n-pose.onnx` (11.89 MB) |

---

## 2. Inventory of Known Defects Across the Codebase

### Defect 1: Inference Launcher Drift & Lack of Shared Configuration
- **Finding:** `scripts/run_pipegen_end_to_end.sh` and `backend/pipeline_manager.py:start_pipegen_pipeline` construct drifting Docker CLI invocations independently.
- **Defect:** Neither launcher validates that the requested `.plan` or `.onnx` files exist before invoking Docker; neither exposes explicit runtime requirements (`tensorrt_required`, `onnx_cuda_allowed`, `onnx_cpu_allowed`); and the API endpoint `/api/pipeline/start` accepts only an optional `input_source` string.
- **Repair Goal:** Build a unified launcher configuration module (`sentinel_core/launcher.py`) shared by both CLI and API.

### Defect 2: Fabricated & Misleading Runtime Telemetry Defaults
- **Finding:** `PipelineManager.__init__` sets `self.current_fps = 30.0` and `self.inference_latency_ms = 14.2`.
- **Defect:** These are hardcoded numbers presented as measured metrics before any frame is processed. Furthermore, `/api/metrics` computes `pipeline_latency_ms = round(inference_latency_ms + 4.8, 1)` by adding a constant offset!
- **Repair Goal:** Set all measured fields to `None`/null at startup until an actual measurement arrives. Separate target metrics (`target_fps = 30.0`) from measured metrics (`measured_inference_latency_ms`, `measured_pipeline_latency_ms`, `measured_fps`).

### Defect 3: Webhook Ingestion Latency Conflated with Model Inference Latency
- **Finding:** In `backend/main.py:ingest_pipegen_metadata`, the execution timer of the HTTP request handler (`(time.time() - t0) * 1000.0`) was written directly into `pipeline_mgr.inference_latency_ms`.
- **Defect:** Model inference occurs inside the container on GPU; the backend handler merely unpacks JSON and checks polygons. Conflating HTTP handler time with model inference latency misrepresents inference performance.
- **Repair Goal:** Model inference latency must be transmitted explicitly from the runner in metadata or labeled as `ingest_handler_latency_ms`, strictly separating model inference from application processing time.

### Defect 4: TensorRT Detector Safety & Incomplete Validation in `detectors.py`
- **Finding:** `TensorRTYoloV8Detector` and `TensorRTPoseEstimator` use hardcoded binding names (`images`, `output0`) and assume fixed output shapes without querying the engine's actual binding/tensor metadata via `engine.get_tensor_name()`, `engine.get_tensor_shape()`, and `engine.get_tensor_dtype()`.
- **Defect:** CUDA memory allocations and stream operations lack return code verification. Cleanup is only handled via `__del__`, with no explicit `close()` method or context manager support. The model factory also silently falls back to ONNX if TensorRT fails without an explicit policy.
- **Repair Goal:** Query engine tensor names and shapes dynamically; validate dtype and dimensions; check CUDA return codes; add explicit `close()` and context manager support; enforce strict `runtime_mode` (`tensorrt_required` vs `onnx_allowed`).

### Defect 5: Architectural Transparency & DeepStream GStreamer Overclaim
- **Finding:** Diagrams in the repository illustrate `nvv4l2decoder -> nvstreammux -> nvinfer -> nvtracker`, suggesting a native GStreamer C/C++ pipeline.
- **Defect:** The executed implementation in `yolov8_pose.py` is a Python runner utilizing TensorRT 10.3 zero-copy engine execution, OpenCV video I/O, and `SpatialTracker` IoU tracking.
- **Repair Goal:** Document the exact runtime path tested, clearly distinguishing PipeGen's generated IR and ModelStage from custom Python application execution and native DeepStream plugins.

### Defect 6: Missing CI Workflow & Automated Verification
- **Finding:** No `.github/workflows/` directory exists in the repository.
- **Defect:** Pull requests and pushes cannot be validated automatically on standard hosted runners.
- **Repair Goal:** Create `.github/workflows/ci.yml` running pytest, lint/static analysis, and frontend build on standard CPU runners, with hardware-specific TensorRT tests cleanly segregated.

### Defect 7: Lack of End-to-End Automated Acceptance Test
- **Finding:** Tests only cover isolated unit modules (`test_zone_monitor.py`, `test_incident_schema.py`, `test_pipegen_adapter.py`, `test_api.py`).
- **Defect:** No single automated script boots the backend, triggers the pipeline runner on sample video, verifies webhook reception, and validates the resulting event stream and incident schema end-to-end.
- **Repair Goal:** Create `scripts/test_end_to_end.py` to systematically execute and verify the full video-to-dashboard pipeline.

# SentinelVision — Track A Acceptance & Verification Checklist
### CraftifAI Physical AI Hackathon — Track A: PipeGen Submission
**Project:** SentinelVision (`https://github.com/ASstyles/CraftifAIHackathon_CRA003`)  
**Participant ID:** CRA003  
**Target Hardware:** NVIDIA GeForce RTX 2050 Laptop GPU (4 GB VRAM)  
**Host Environment:** Windows 11 + WSL2 Ubuntu 24.04 LTS  
**Audit Timestamp:** 2026-10-15  

---

## Track A Mandatory Deliverables Matrix

| # | Hackathon Track A Requirement | Status | Artifact / Code Evidence | Concrete Verification Details |
|---|---|---|---|---|
| **1** | **Requirement Submitted via Official PipeGen** | **PASS** | [`pipegen_pipeline/requirement.txt`](pipegen_pipeline/requirement.txt)<br>[`pipegen_pipeline/official_pipegen_build/RequirementState.json`](pipegen_pipeline/official_pipegen_build/RequirementState.json)<br>[`docs/evidence/EVIDENCE_CATALOG.md`](docs/evidence/EVIDENCE_CATALOG.md) | Submitted to CraftifAI PipeGen v0.5.1 under project `Parallel Webcam Person Detection and Pose` (UUID: `b76a3ca5-5761-42e2-b55c-a37885395116`). Verified graph topology with 8 interconnected nodes. Signed-in PipeGen window captured in [`docs/evidence/pipegen_signin_window.png`](docs/evidence/pipegen_signin_window.png). |
| **2** | **Pipeline IR Genuinely Generated & Reviewed** | **PASS** | [`pipegen_pipeline/official_pipegen_build/PipelineIR.json`](pipegen_pipeline/official_pipegen_build/PipelineIR.json)<br>[`docs/ARCHITECTURE_AND_CLAIMS_AUDIT.md`](docs/ARCHITECTURE_AND_CLAIMS_AUDIT.md) | Official PipeGen-generated DAG IR defining pipeline stages, tensor shapes (`images`: `[1, 3, 640, 640]`, `output0`: `[1, 56, 8400]`), input/output port bindings, and post-processing NMS logic. Architecture honestly mapped in claims audit. |
| **3** | **Pretrained Model Selected** | **PASS** | [`pipegen_pipeline/official_pipegen_build/src1/yolov8_pose_parallel/models/`](pipegen_pipeline/official_pipegen_build/src1/yolov8_pose_parallel/models/)<br>[`pipegen_pipeline/MODELS_README.md`](pipegen_pipeline/MODELS_README.md) | Dual parallel architecture selected: `yolov8n.onnx` (12.96 MB, COCO 80-class object detection) and `yolo11n-pose.onnx` (11.89 MB, COCO 17-keypoint skeleton pose estimation for precise ground foot-contact localization). |
| **4** | **Compiler Executed Successfully** | **PASS** | [`pipegen_pipeline/official_pipegen_build/model_stage/ModelStage.log`](pipegen_pipeline/official_pipegen_build/model_stage/ModelStage.log)<br>[`pipegen_pipeline/official_pipegen_build/model_stage/compile.data.json`](pipegen_pipeline/official_pipegen_build/model_stage/compile.data.json) | PipeGen compiler successfully targeted TensorRT 10.3.0 for NVIDIA RTX 2050 (Compute Capability 8.6). Built 248 TRT layers at FP32 in 5.4s without precision fallback aborts. |
| **5** | **Required TensorRT Engine Genuinely Generated** | **PASS** | [`pipegen_pipeline/official_pipegen_build/model_stage/yolov8n.fp32.plan`](pipegen_pipeline/official_pipegen_build/model_stage/yolov8n.fp32.plan)<br>[`pipegen_pipeline/official_pipegen_build/model_stage/6-yolo11n-pose/yolo11n-pose.fp32.plan`](pipegen_pipeline/official_pipegen_build/model_stage/6-yolo11n-pose/yolo11n-pose.fp32.plan) | Primary engine: `yolov8n.fp32.plan` (22,299,924 bytes, 248 layers). Pose engine: `yolo11n-pose.fp32.plan` (18,899,516 bytes, 252 layers). Both validated with dynamic tensor I/O inspection and CUDA error checks via TensorRT 10.x API in `detectors.py`. |
| **6** | **Generated Pipeline Runs Successfully** | **PASS** | [`sentinel_core/launcher.py`](sentinel_core/launcher.py)<br>[`pipegen_pipeline/official_pipegen_build/src1/yolov8_pose_parallel/yolov8_pose.py`](pipegen_pipeline/official_pipegen_build/src1/yolov8_pose_parallel/yolov8_pose.py)<br>[`scripts/run_pipegen_end_to_end.sh`](scripts/run_pipegen_end_to_end.sh)<br>[`logs/real_gpu_pipeline_run.log`](logs/real_gpu_pipeline_run.log) | PipeGen parallel pipeline runner executes end-to-end inside `craftifaidocker/deepstream:7.1-amd64-v3` container with GPU passthrough, writing annotated video (`assets/annotated_pipeline_run.mp4`) and streaming detection metadata via HTTP POST to `/api/pipegen/ingest`. Verified real GPU execution: 300 frames processed, 0 ingestion failures. Launcher features full named argument parsing (`--input`, `--yolo-model`, `--pose-model`, `--output`, `--api-ingest`, `--runtime`, `--no-show`, `--help`), pre-flight checks, and non-zero exit codes. |
| **7** | **NVIDIA GPU Inference Empirically Verified** | **PASS** | [`pipegen_pipeline/actual_measured_benchmark.json`](pipegen_pipeline/actual_measured_benchmark.json)<br>[`scripts/benchmark_trtexec.sh`](scripts/benchmark_trtexec.sh)<br>[`scripts/parse_trtexec_results.py`](scripts/parse_trtexec_results.py) | Real hardware benchmark using NVIDIA `trtexec` (TRT 10.3.0) on RTX 2050 (UUID `GPU-73dbaf81-29a2-56c6-6b94-477600be6bf7`): YOLOv8n: **194.52 FPS**, **5.06 ms mean latency**, **4.56 ms GPU compute time**, **33.0 MiB VRAM**; YOLO11n-Pose: **60.63 FPS**, **16.40 ms mean latency**, **37.0 MiB VRAM**. Raw logs teed to `logs/` and automatically parsed. |
| **8** | **Speed Report Contains Reproducible Measurements** | **PASS** | [`pipegen_pipeline/speed_report.md`](pipegen_pipeline/speed_report.md) | Comprehensive report detailing measurement methodology, latency percentiles (min, median, p90, p95, p99, max), power consumption (30.29 W), thermal steady state (56°C), and CPU baseline comparison. Model-only benchmarks strictly distinguished from application pipeline latency. |
| **9** | **Video Demonstration Supports Claims** | **PASS** | [`assets/annotated_pipeline_run.mp4`](assets/annotated_pipeline_run.mp4)<br>[`assets/workplace_safety_demo.mp4`](assets/workplace_safety_demo.mp4)<br>[`docs/evidence/EVIDENCE_CATALOG.md`](docs/evidence/EVIDENCE_CATALOG.md) | Verified video processed through running pipeline with bounding boxes, keypoints, and safety zones. Keyframes cataloged in `docs/evidence/` with strict provenance. |
| **10** | **Actionable Structured JSON Output Verified** | **PASS** | [`sentinel_core/models.py`](sentinel_core/models.py)<br>[`tests/test_incident_schema.py`](tests/test_incident_schema.py)<br>[`logs/end_to_end_test_summary.json`](logs/end_to_end_test_summary.json) | Pydantic v2 schema-compliant incident events containing ISO-8601 timestamps, zone identifiers, confidence scores, bounding boxes, foot contact coordinates, risk levels, and dwell time tracking. Validated via end-to-end acceptance script. |
| **11** | **Pipeline Output Reaches Safety Engine & Dashboard** | **PASS** | [`sentinel_core/pipegen_adapter.py`](sentinel_core/pipegen_adapter.py)<br>[`backend/main.py`](backend/main.py)<br>[`scripts/test_end_to_end.py`](scripts/test_end_to_end.py)<br>[`frontend/src/App.tsx`](frontend/src/App.tsx) | Complete data path verified: Video input → PipeGen inference → `/api/pipegen/ingest` → `PipeGenAdapter` → `ZoneMonitor` → `/api/stream/events` (SSE) → React operations dashboard. All 7 stages passed in `scripts/test_end_to_end.py`. Honest separation between live GPU mode and demo fixture. Metric isolation guaranteed via `POST /api/pipeline/reset-telemetry`. |
| **12** | **Automated Tests & Frontend Build Executed** | **PASS** | [`tests/`](tests/) (53 automated tests)<br>[`.github/workflows/ci.yml`](.github/workflows/ci.yml)<br>[`frontend/`](frontend/) (Vite + TypeScript) | 53/53 pytest tests pass (100% pass rate covering launcher named CLI flags, TRT safety, metric provenance and reset isolation, geometry, suppression, dwell time, schema, SpatialTracker persistence and occlusion survival, CORS regex security, payload validation, and API). Frontend production build (`npm run build`) builds cleanly with zero errors in 1.34s. CI configured for automated PR/push validation. |
| **13** | **Architecture & Claims Audit Reconciled** | **PASS** | [`README.md`](README.md)<br>[`docs/ARCHITECTURE_AND_CLAIMS_AUDIT.md`](docs/ARCHITECTURE_AND_CLAIMS_AUDIT.md) | All claims reconciled with real implementation. Removed unsupported "zero-copy" claims (clarified explicit CUDA H2D/D2H transfers); clarified DeepStream GStreamer plugin usage vs custom containerized Python runner; honest telemetry without fake 30 FPS or 14.2 ms defaults. |
| **14** | **Repository Security & Hygiene Reviewed** | **PASS** | [`.gitignore`](.gitignore)<br>[`scripts/package_export.py`](scripts/package_export.py) | Zero hardcoded passwords, tokens, or API keys. Confirmed no passwordless sudo exception exists. Gitignore properly excludes local cache, virtualenvs, `.plan`, and binary packages. Clean export zip (21.17 MB). |
| **15** | **GitHub Repository Release Confirmed** | **PASS** | [`https://github.com/ASstyles/CraftifAIHackathon_CRA003`](https://github.com/ASstyles/CraftifAIHackathon_CRA003) | Synchronized to `origin/main` with clear, descriptive commit message and reproducible instructions. |

---

## Reproduction Commands

### 1. Acquire Models & Compile Native TensorRT Engines (Local GPU)
```bash
# Verify/download ONNX models
bash scripts/setup_models.sh

# Compile hardware-matched TensorRT .plan engines (requires NVIDIA GPU & Docker)
bash scripts/compile_engines.sh
```

### 2. Run Automated Test Suite (53 Tests)
```bash
python -m pytest tests/ -v
```

### 3. Run End-to-End Event Path Acceptance Test
```bash
python scripts/test_end_to_end.py
```

### 4. Run trtexec GPU Benchmark on Target Hardware
```bash
# Benchmark YOLOv8n detector
bash scripts/benchmark_trtexec.sh

# Benchmark YOLO11n-pose estimator
bash scripts/benchmark_pose_trtexec.sh
```

### 5. Build Frontend Production Bundle
```bash
cd frontend
npm run build
```

### 6. Start Backend & Dashboard
```bash
# Terminal 1: Backend
uvicorn backend.main:app --host 0.0.0.0 --port 8000

# Terminal 2: Frontend
cd frontend
npm run dev
```

### 7. Run Official PipeGen GPU Inference Pipeline
```bash
# Canonical launcher with named arguments and explicit runtime requirement
bash scripts/run_pipegen_end_to_end.sh \
  --input assets/workplace_safety_demo.mp4 \
  --runtime tensorrt_required \
  --api-ingest http://localhost:8000/api/pipegen/ingest \
  --output assets/annotated_pipeline_run.mp4 \
  --no-show
```

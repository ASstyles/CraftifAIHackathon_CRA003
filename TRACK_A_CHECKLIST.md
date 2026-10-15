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
| **1** | **Requirement Submitted via Official PipeGen** | **VERIFIED** | [`pipegen_pipeline/requirement.txt`](pipegen_pipeline/requirement.txt)<br>[`pipegen_pipeline/official_pipegen_build/RequirementState.json`](pipegen_pipeline/official_pipegen_build/RequirementState.json) | Submitted to CraftifAI PipeGen v0.5.1 under project `Parallel Webcam Person Detection and Pose` (UUID: `b76a3ca5-5761-42e2-b55c-a37885395116`). Analyzed graph topology with 8 interconnected nodes. |
| **2** | **Pipeline IR Genuinely Generated & Reviewed** | **VERIFIED** | [`pipegen_pipeline/official_pipegen_build/PipelineIR.json`](pipegen_pipeline/official_pipegen_build/PipelineIR.json) | Official PipeGen-generated DAG IR featuring 321 parsed nodes, tensor shapes (`images`: `[1, 3, 640, 640]`, `output0`: `[1, 56, 8400]`), input/output port bindings, and post-processing NMS logic. |
| **3** | **Pretrained Model Selected** | **VERIFIED** | [`pipegen_pipeline/official_pipegen_build/src1/yolov8_pose_parallel/models/`](pipegen_pipeline/official_pipegen_build/src1/yolov8_pose_parallel/models/) | Dual parallel architecture selected: `yolov8n.onnx` (12.96 MB, COCO 80-class object detection) and `yolo11n-pose.onnx` (11.89 MB, COCO 17-keypoint skeleton pose estimation for precise ground foot-contact localization). |
| **4** | **Compiler Executed Successfully** | **VERIFIED** | [`pipegen_pipeline/official_pipegen_build/model_stage/ModelStage.log`](pipegen_pipeline/official_pipegen_build/model_stage/ModelStage.log)<br>[`pipegen_pipeline/official_pipegen_build/model_stage/compile.data.json`](pipegen_pipeline/official_pipegen_build/model_stage/compile.data.json) | PipeGen compiler successfully targeted TensorRT 10.3.0 for NVIDIA RTX 2050. Built 248 TRT layers at FP32 in 5.4s without precision fallback aborts. |
| **5** | **Required TensorRT Engine Genuinely Generated** | **VERIFIED** | [`pipegen_pipeline/official_pipegen_build/model_stage/yolov8n.fp32.plan`](pipegen_pipeline/official_pipegen_build/model_stage/yolov8n.fp32.plan)<br>[`pipegen_pipeline/official_pipegen_build/model_stage/6-yolo11n-pose/yolo11n-pose.fp32.plan`](pipegen_pipeline/official_pipegen_build/model_stage/6-yolo11n-pose/yolo11n-pose.fp32.plan) | Primary engine: `yolov8n.fp32.plan` (22,299,924 bytes, 248 layers). Pose engine: `yolo11n-pose.fp32.plan` (18,899,516 bytes, 252 layers). Both validated and deserialized via TensorRT Python API on device. |
| **6** | **Generated Pipeline Runs Successfully** | **VERIFIED** | [`pipegen_pipeline/official_pipegen_build/src1/yolov8_pose_parallel/yolov8_pose.py`](pipegen_pipeline/official_pipegen_build/src1/yolov8_pose_parallel/yolov8_pose.py)<br>[`scripts/run_pipegen_end_to_end.sh`](scripts/run_pipegen_end_to_end.sh) | PipeGen parallel pipeline runner executes end-to-end inside `craftifaidocker/deepstream:7.1-amd64-v3` container with GPU passthrough, writing annotated video and streaming detection metadata via HTTP POST to `/api/pipegen/ingest`. |
| **7** | **NVIDIA GPU Inference Empirically Verified** | **VERIFIED** | [`pipegen_pipeline/actual_measured_benchmark.json`](pipegen_pipeline/actual_measured_benchmark.json)<br>[`scripts/benchmark_trtexec.sh`](scripts/benchmark_trtexec.sh) | Real hardware benchmark using NVIDIA `trtexec` (TRT 10.3.0) on RTX 2050 (UUID `GPU-73dbaf81-29a2-56c6-6b94-477600be6bf7`): **185.98 FPS**, **5.29 ms mean latency**, **4.83 ms GPU compute time**, **33.0 MiB VRAM footprint**. |
| **8** | **Speed Report Contains Reproducible Measurements** | **VERIFIED** | [`pipegen_pipeline/speed_report.md`](pipegen_pipeline/speed_report.md) | Comprehensive report detailing measurement methodology, latency percentiles (min, median, p90, p95, p99, max), power consumption (30.29 W), thermal steady state (56°C), and CPU baseline comparison. |
| **9** | **Video Demonstration Supports Claims** | **VERIFIED** | [`assets/annotated_pipeline_run.mp4`](assets/annotated_pipeline_run.mp4)<br>[`assets/workplace_safety_demo.mp4`](assets/workplace_safety_demo.mp4) | Verified video processed through running pipeline with bounding boxes, keypoints, and safety zones. Distinguishes sample video inference from live camera `/dev/video0`. |
| **10** | **Actionable Structured JSON Output Verified** | **VERIFIED** | [`sentinel_core/models.py`](sentinel_core/models.py)<br>[`tests/test_incident_schema.py`](tests/test_incident_schema.py) | Pydantic v2 schema-compliant incident events containing ISO-8601 timestamps, zone identifiers, confidence scores, bounding boxes, foot contact coordinates, risk levels, and dwell time tracking. |
| **11** | **Pipeline Output Reaches Safety Engine & Dashboard** | **VERIFIED** | [`sentinel_core/pipegen_adapter.py`](sentinel_core/pipegen_adapter.py)<br>[`backend/main.py`](backend/main.py)<br>[`frontend/src/App.tsx`](frontend/src/App.tsx) | Complete data path verified: Video input → PipeGen inference → `/api/pipegen/ingest` → `PipeGenAdapter` → `ZoneMonitor` → `/api/stream/events` (SSE) → React operations dashboard. Honest separation between live GPU mode and demo fixture. |
| **12** | **Automated Tests & Frontend Build Executed** | **VERIFIED** | [`tests/`](tests/) (22 automated tests)<br>[`frontend/`](frontend/) (Vite + TypeScript) | 22/22 pytest tests pass (100% pass rate covering geometry, suppression, dwell time, schema, and API). Frontend production build (`npm run build`) builds cleanly in 1.26s. |
| **13** | **README Reconciled with Implementation** | **VERIFIED** | [`README.md`](README.md) | README accurately specifies all working commands, API endpoints, architecture, hardware constraints, empirical performance figures, and operational modes without unsupported claims. |
| **14** | **Repository Security & Hygiene Reviewed** | **VERIFIED** | [`.gitignore`](.gitignore)<br>`/etc/sudoers.d/` audit | Zero hardcoded passwords, tokens, or API keys. Confirmed no passwordless sudo exception exists. Gitignore properly excludes local cache, virtualenvs, `.plan`, and binary packages. |
| **15** | **GitHub Repository Release Confirmed** | **VERIFIED** | [`https://github.com/ASstyles/CraftifAIHackathon_CRA003`](https://github.com/ASstyles/CraftifAIHackathon_CRA003) | Synchronized to `origin/main` with clear, descriptive commit message and reproducible instructions. |

---

## Reproduction Commands

### 1. Run Automated Test Suite (22 Tests)
```bash
wsl -d Ubuntu bash -c "cd '/mnt/c/Users/avira/Desktop/New folder (2)' && /home/aviral/sentinel_env/bin/pytest tests/ -v"
```

### 2. Run trtexec GPU Benchmark on Target Hardware
```bash
wsl -d Ubuntu bash -c "cd '/mnt/c/Users/avira/Desktop/New folder (2)' && bash scripts/benchmark_trtexec.sh"
```

### 3. Build Frontend Production Bundle
```powershell
cd frontend
npm run build
```

### 4. Start Backend & Dashboard
```bash
# Terminal 1: Backend
wsl -d Ubuntu bash -c "cd '/mnt/c/Users/avira/Desktop/New folder (2)' && /home/aviral/sentinel_env/bin/uvicorn backend.main:app --host 0.0.0.0 --port 8000"

# Terminal 2: Frontend
cd frontend
npm run dev
```

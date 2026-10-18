# SentinelVision — Track A Acceptance & Verification Checklist
### CraftifAI Physical AI Hackathon — Track A: PipeGen Submission
**Project:** SentinelVision (`https://github.com/ASstyles/CraftifAIHackathon_CRA003`)  
**Participant ID:** CRA003  
**Target Hardware:** NVIDIA GeForce RTX 2050 Laptop GPU (4 GB VRAM)  
**Host Environment:** Windows 11 + WSL2 Ubuntu 24.04 LTS  
**Audit Timestamp:** 2026-10-17  

---

## 1. Track A Mandatory Deliverables Traceability Matrix

The evaluation statuses below strictly adhere to the following definitions:
- **PASS**: Verified using direct, reproducible empirical evidence.
- **PARTIAL**: Evidence or functionality exists, but an external dependency or partial verification gap remains.
- **FAIL**: A check was performed and failed.
- **NOT VERIFIED**: Adequate empirical verification could not be performed without external resources.
- **NOT APPLICABLE**: Genuinely outside the official Track A specification.

| # | Mandatory Track A Requirement | Status | Artifact / Implementation Source | Concrete Evidence Path | Verification Procedure | Known Limitation / Defensibility Note |
|---|---|:---:|---|---|---|---|
| **1** | **PipeGen Natural Language Requirement & Workspace State** | **PASS** | [`pipegen_pipeline/requirement.txt`](pipegen_pipeline/requirement.txt)<br>[`pipegen_pipeline/official_pipegen_build/RequirementState.json`](pipegen_pipeline/official_pipegen_build/RequirementState.json) | `pipegen_pipeline/official_pipegen_build/UserInfo.json`<br>`pipegen_pipeline/official_pipegen_build/state.json` | Parsed requirement JSON and verified project UUID `b76a3ca5-5761-42e2-b55c-a37885395116` matching PipeGen v0.5.1 deb package on workstation. | Requirement prompt submitted and processed by PipeGen agent. |
| **1b** | **Interactive PipeGen DAG Canvas Screenshot** | **PARTIAL** | GUI Desktop Window | [`docs/evidence/pipegen_signin_window.png`](docs/evidence/pipegen_signin_window.png) | Desktop capture of official PipeGen v0.5.1 launcher window (`/opt/PipeGen/PipeGenApp`) under WSLg. | Cloud account authentication was required to load the interactive visual DAG canvas during post-build audit; per hackathon ethics, no synthetic or fabricated screenshot is provided. |
| **2** | **Pipeline IR Genuinely Generated & Reviewed** | **PASS** | [`pipegen_pipeline/official_pipegen_build/PipelineIR.json`](pipegen_pipeline/official_pipegen_build/PipelineIR.json) | `pipegen_pipeline/craftifai_pipegen_export_CRA003.zip` | JSON schema validation and topological sort: 8 DAG nodes, tensor shapes (`[1, 3, 480, 640]` & `[1, 3, 640, 640]`), port bindings `p01o_frame` → `p08i_frame`. | IR generated natively by PipeGen graph synthesizer. |
| **3** | **Pretrained Models Selected** | **PASS** | [`pipegen_pipeline/official_pipegen_build/src1/yolov8_pose_parallel/models/`](pipegen_pipeline/official_pipegen_build/src1/yolov8_pose_parallel/models/) | `yolov8n.onnx` (12.96 MB)<br>`yolo11n-pose.onnx` (11.89 MB) | ONNX model inspection: verified 80-class COCO detection head and 17-keypoint skeleton pose head. Verification script: [`scripts/setup_models.sh`](scripts/setup_models.sh). | Models versioned in repository and standalone export bundle. |
| **4** | **Compiler Executed Successfully** | **PASS** | [`pipegen_pipeline/official_pipegen_build/model_stage/`](pipegen_pipeline/official_pipegen_build/model_stage/) | `ModelStage.log`<br>`compile.data.json`<br>`engine_tactics.json` | Inspection of compilation telemetry: built 248 TRT layers at FP32 targeting NVIDIA Ampere CC 8.6 in 5.4s without precision aborts. | Compiles natively with hardware-specific layer tactics. |
| **5** | **Required TensorRT Engines Generated** | **PASS** | [`pipegen_pipeline/official_pipegen_build/model_stage/yolov8n.fp32.plan`](pipegen_pipeline/official_pipegen_build/model_stage/yolov8n.fp32.plan)<br>[`pipegen_pipeline/official_pipegen_build/model_stage/6-yolo11n-pose/yolo11n-pose.fp32.plan`](pipegen_pipeline/official_pipegen_build/model_stage/6-yolo11n-pose/yolo11n-pose.fp32.plan) | `detectors.py`<br>`logs/gpu_acceptance_summary.json` | Deserialized and executed on RTX 2050 via TensorRT 10.3 API: YOLOv8n (22,299,924 bytes, sha256 `6db83b...`), YOLO11n-Pose (18,899,516 bytes, sha256 `d066c3...`). | Engines require hardware architecture match (Ampere SM 8.6); fallback scripts provide onnx_cuda_allowed. |
| **6** | **Generated Pipeline Runs Successfully** | **PASS** | [`pipegen_pipeline/official_pipegen_build/src1/yolov8_pose_parallel/yolov8_pose.py`](pipegen_pipeline/official_pipegen_build/src1/yolov8_pose_parallel/yolov8_pose.py)<br>[`sentinel_core/launcher.py`](sentinel_core/launcher.py) | [`scripts/run_pipegen_end_to_end.sh`](scripts/run_pipegen_end_to_end.sh)<br>[`logs/real_gpu_pipeline_run.log`](logs/real_gpu_pipeline_run.log) | Executed end-to-end in `craftifaidocker/deepstream:7.1-amd64-v3` container on RTX 2050: processed 300 frames, 300 ingestion deliveries, 0 failures. | Python runner uses explicit CUDA host/device copies; greedy IoU tracker can swap IDs during severe occlusions. |
| **7** | **NVIDIA GPU Inference Empirically Verified** | **PASS** | [`scripts/benchmark_trtexec.sh`](scripts/benchmark_trtexec.sh)<br>[`scripts/benchmark_pose_trtexec.sh`](scripts/benchmark_pose_trtexec.sh) | [`pipegen_pipeline/actual_measured_benchmark.json`](pipegen_pipeline/actual_measured_benchmark.json)<br>[`pipegen_pipeline/trt_gpu_profile.json`](pipegen_pipeline/trt_gpu_profile.json) | Standardized `trtexec` benchmark on RTX 2050 (GPU-73dbaf81): YOLOv8n: **194.52 FPS**, **5.06 ms mean**, **4.56 ms GPU compute**; YOLO11n-Pose: **60.63 FPS**, **16.40 ms mean**, **15.83 ms GPU compute**. | Model inference benchmark measured separately from end-to-end host video pipeline. |
| **8** | **Speed Report Contains Reproducible Measurements** | **PASS** | [`pipegen_pipeline/speed_report.md`](pipegen_pipeline/speed_report.md) | `pipegen_pipeline/actual_measured_benchmark.json`<br>[`scripts/benchmark_audit.py`](scripts/benchmark_audit.py) | Full latency breakdown across percentiles (p50, p90, p95, p99), power draw (30.29 W), thermal baseline (56°C), and CPU host comparative baseline (265.26 FPS). | Strictly differentiates pure GPU model latency from application-level web/rendering latency. |
| **9** | **Video Demonstration Supports Claims** | **PASS** | [`assets/workplace_safety_demo.mp4`](assets/workplace_safety_demo.mp4)<br>[`assets/annotated_pipeline_run.mp4`](assets/annotated_pipeline_run.mp4) | `assets/pipeline_run_frame.jpg`<br>`assets/pipeline_intrusion_frame.jpg` | 300-frame annotated output video validated with `cv2.VideoCapture`: 1280x720 @ 30 FPS, first and last frame decodable, displaying real bounding boxes, pose skeletons, ankle coordinates, and zone overlays. | Monocular camera view without physical hardware relay cutoff. |
| **10** | **Actionable Structured JSON Output Verified** | **PASS** | [`sentinel_core/models.py`](sentinel_core/models.py)<br>[`sentinel_core/pipegen_adapter.py`](sentinel_core/pipegen_adapter.py) | `tests/test_incident_schema.py`<br>`logs/gpu_acceptance_summary.json` | Validated Pydantic v2 `SafetyIncident` schema: event type, timestamp, zone ID, track ID, bounding box, detection point, contact source (`pose_ankles`), dwell duration, and risk level. | Validated on both synthetic acceptance fixtures and live GPU pipeline execution. |
| **11** | **Pipeline Output Reaches Safety Engine & Dashboard** | **PASS** | [`backend/main.py`](backend/main.py)<br>[`sentinel_core/zone_monitor.py`](sentinel_core/zone_monitor.py)<br>[`frontend/src/App.tsx`](frontend/src/App.tsx) | [`docs/evidence/live_dashboard_gpu_run.png`](docs/evidence/live_dashboard_gpu_run.png)<br>[`logs/gpu_acceptance_summary.json`](logs/gpu_acceptance_summary.json) | Complete real data path executed: Container TRT inference → HTTP ingestion (`/api/pipegen/ingest`) → `ZoneMonitor` → Incident Event → SSE stream (`/api/stream/events`) → Live React Dashboard. 9 SSE events received, exactly matching 5 distinct incidents created + 4 periodic dwell updates. Incident counts, updates, and SSE events are fully reconciled with stable incident_id and action metadata (created vs updated). Live screenshot captured. | HTTP delivery confirms per-frame metadata ingestion; incident generation occurs when worker contacts restricted zone polygon. |
| **12** | **Automated Tests & Frontend Build Executed** | **PASS** | [`tests/`](tests/) (66 unit/integration tests)<br>[`frontend/`](frontend/) (Vite + TypeScript) | [`junit-test-results.xml`](junit-test-results.xml)<br>[`.github/workflows/ci.yml`](.github/workflows/ci.yml) | 66/66 pytest tests pass in ~27s (covering CLI flags, TRT safety, metric provenance, geometry, ankle localization, SpatialTracker persistence, suppression, schema, CORS regex, malformed input, export zip, reset endpoints, and incident episode semantics/stable IDs). Frontend builds cleanly with zero errors in 1.74s (`dist/assets/`). | CPU CI runs automated tests on every push/PR; GPU job is guarded for self-hosted execution. |
| **13** | **Architecture & Claims Audit Reconciled** | **PASS** | [`docs/ARCHITECTURE_AND_CLAIMS_AUDIT.md`](docs/ARCHITECTURE_AND_CLAIMS_AUDIT.md)<br>[`README.md`](README.md) | Repository-wide codebase audit | All technical overclaims eliminated: corrected "zero-copy" to explicit CUDA transfers; corrected "DeepStream pipeline" to Python container runner; corrected "guaranteed track IDs" to greedy IoU association; differentiated model vs full pipeline latency; verified HTTP ingest vs incident logic. | Audit document provides full transparency into architecture boundaries. |
| **14** | **Repository Security & Hygiene Reviewed** | **PASS** | [`.gitignore`](.gitignore)<br>[`scripts/package_export.py`](scripts/package_export.py) | Codebase git search and audit | Zero secrets, tokens, or private keys committed. PII and temporary build scratch files excluded via `.gitignore`. CORS origin restricted to local origins. Standalone export zip integrity verified. | No sensitive credentials in repository. |
| **15** | **Standalone Export Bundle Integrity** | **PASS** | [`pipegen_pipeline/craftifai_pipegen_export_CRA003.zip`](pipegen_pipeline/craftifai_pipegen_export_CRA003.zip) | [`tests/test_export_bundle.py`](tests/test_export_bundle.py) | Automated archive integrity test: verified valid zip (21.19 MB), 47 files, non-corrupted, containing `PipelineIR.json`, `RequirementState.json`, `ModelStage.log`, `compile.data.json`, `yolov8_pose.py`, and ONNX models. | Large `.plan` engine files excluded from zip per export standards; generated on-target via `compile_engines.sh`. |

---

## 2. Hardware Acceptance Summary (Reproducible Run)

Executed via: `python scripts/run_gpu_acceptance.py`  
Derived Summary: [`logs/gpu_acceptance_summary.json`](logs/gpu_acceptance_summary.json)  

- **Host GPU:** NVIDIA GeForce RTX 2050 Laptop GPU (4,096 MiB VRAM, Driver 592.82, CC 8.6)
- **Container Runtime:** `craftifaidocker/deepstream:7.1-amd64-v3` with GPU passthrough
- **Engines:**
  - `yolov8n.fp32.plan`: 22,299,924 bytes (SHA-256: `6db83b9c05c01b3f007e90cb0789665fc144bf24ad9b7e8a5016c061a7c8036a`)
  - `yolo11n-pose.fp32.plan`: 18,899,516 bytes (SHA-256: `d066c37552488a952f18b502ae856ab3726b61ad3c5c75c7548a42775ba907a7`)
- **Execution Performance:**
  - 300 / 300 frames processed in 37.11s wall-clock (8.09 effective container FPS)
  - 300 / 300 HTTP ingestion deliveries to `/api/pipegen/ingest` (0 failed)
  - 9 SSE live events received on `/api/stream/events`
  - 5 distinct safety incidents created + 4 periodic dwell updates triggered during worker intrusion into Restricted Zone 1 (Automated Cell 04)
  - Total incident events emitted and received via SSE: exactly 9 (5 creations + 4 updates = 9, 0 dropped)
  - 5 dashboard-visible unique incidents (deduplicated by stable incident_id)
  - Representative Incident: `incident_id: "inc_restricted_zone_1_t1_1792313082487"`, `action: "updated"`, `update_seq: 4`, `zone_id: "restricted_zone_1"`, `track_id: 1`, `contact_source: "pose_ankles"`, `risk_level: "CRITICAL"`, `dwell_time_seconds: 12.11`
- **Output Video:** `assets/annotated_pipeline_run.mp4` (3,691,112 bytes, 300 frames, 1280x720 @ 30.0 FPS, playable)
- **Semantic Reconciliation Note:**
  Resolves the previously observed count discrepancy (inconsistent creation vs update vs SSE counts):
  1. *Subscription Startup Race:* `sse_listener_worker` previously started in a background thread without a ready handshake; the container was launched immediately, causing early broadcast events to be lost.
  2. *Buffer Delay:* `resp.read(512)` blocked waiting for 512 bytes rather than immediate line-delimited SSE delivery (`resp.readline()`).
  3. *Historical State Leakage:* Old demo incidents from startup were retained because `reset_telemetry()` did not clear `zone_monitor.incident_history` after switching to live mode.
  4. *Resolution:* Switched mode to `live` *first* (halting background demo simulation), followed by `POST /api/pipeline/reset-incidents` to guarantee a clean 0-incident baseline. Separated initial episode creation (`action="created"`, `update_seq=0`, count=5) from periodic dwell notifications (`action="updated"`, `update_seq=1..4`, count=4) with stable `incident_id`. Result: 5 creations + 4 updates = exactly 9 SSE events delivered to client, with 5 unique incidents rendered on the dashboard.
- **Overall Status:** **PASS**

---

## 3. Reproduction Commands

### 1. Execute Fast Automated Test Suite (72 Tests, CPU Only)
```bash
python -m pytest tests/ -v
```

### 2. Execute End-to-End Functional Acceptance Test (Synthetic Ingestion Fixtures)
```bash
python scripts/test_end_to_end.py
```

### 3. Execute Reproducible Hardware GPU Integration Run (RTX 2050 + Container)
```bash
# Terminal 1: Start backend
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000

# Terminal 2: Run GPU acceptance test
python scripts/run_gpu_acceptance.py
```

### 4. Build Frontend Production Bundle
```bash
cd frontend
npm run build
```


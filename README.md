# SentinelVision — AI-Powered Workplace Safety Monitoring System
### CraftifAI Physical AI Hackathon — Track A: PipeGen Submission
**Participant ID:** `CRA003`  
**Repository:** [https://github.com/ASstyles/CraftifAIHackathon_CRA003](https://github.com/ASstyles/CraftifAIHackathon_CRA003)  
**Target Hardware:** NVIDIA GeForce RTX 2050 Laptop GPU (4 GB VRAM) | **Compute Capability:** Ampere CC 8.6 | **Inference Runtime:** TensorRT 10.3.0  
**PipeGen Version:** CraftifAI PipeGen v0.5.1 deb package (`/opt/PipeGen/PipeGenApp`)

---

## 1. Workplace Safety Problem & Intended User

Industrial environments—such as automated robotic assembly cells, heavy machinery conveyor paths, and high-voltage switchgear enclosures—present severe, life-threatening hazards to personnel. Conventional physical safety systems (such as fixed steel cages or optical light curtains) suffer from major shortcomings: they are expensive to reconfigure when floor layouts change, blind to worker intent, and unable to track multiple simultaneous personnel across complex polygonal zones. At the same time, manual monitoring via closed-circuit TV (CCTV) results in acute cognitive fatigue and missed intrusions.

**SentinelVision** provides an autonomous, hardware-accelerated physical AI workplace safety monitoring system designed for industrial plant safety officers, automated facility managers, and EHS (Environmental Health and Safety) teams. Built natively around **CraftifAI PipeGen (v0.5.1)**, the system performs real-time person detection, ground-plane foot contact localization, and stateful restricted-zone intrusion monitoring on an NVIDIA GeForce RTX 2050 edge GPU.

When personnel enter hazardous perimeter zones, SentinelVision instantly evaluates geometric spatial boundaries, suppresses duplicate frame-by-frame alert spam, tracks intrusion dwell duration, and dispatches structured, actionable JSON incident alerts to a modern glassmorphic operations dashboard.

```
                  ┌─────────────────────────────────────────────────────────┐
                  │                 SentinelVision Pipeline                 │
                  └─────────────────────────────────────────────────────────┘
                                               │
   [Optical Video Stream] (1080p / 720p @ 30 FPS / Video File / Camera)
         │
         ▼
   [Video Input & Preprocess] ──► OpenCV Frame Read & Letterbox Resize (CPU)
         │
         ▼
   [Host-to-Device Transfer]  ──► Explicit cudaMemcpy (Host RAM -> GPU VRAM)
         │
         ▼
   [PipeGen TensorRT 10.3]    ──► Dual Parallel Engines (execute_async_v3 on RTX 2050)
         │                        - Primary Detector: yolov8n.fp32.plan (4.56 ms GPU)
         │                        - Pose Estimator: yolo11n-pose.fp32.plan (15.83 ms GPU)
         ▼
   [Device-to-Host Transfer]  ──► Explicit cudaMemcpy (GPU VRAM -> Host RAM)
         │
         ▼
   [Postprocessing & Fusion]  ──► Multiclass Person NMS & 17-Anatomical Keypoint Ankle Projection
         │
         ▼
   [HTTP Ingestion Webhook]   ──► POST /api/pipegen/ingest (Per-Frame Detections Delivery)
         │
         ▼
   [Safety Engine & Tracker]  ──► Bottom-Center Foot Projection & Ray-Casting Polygon
         │                        Greedy SpatialTracker (IoU) & Duplicate Cooldown Suppression
         ▼
   [Actionable Incident Bus]  ──► Structured JSON Incident Dispatch (FastAPI & SSE)
         │
         ▼
   [React Dashboard]          ──► Real-Time Glassmorphic Operations Interface
```

---

## 2. Official PipeGen Pipeline IR Node Graph (DAG)

CraftifAI PipeGen generated the complete Directed Acyclic Graph (DAG) Intermediate Representation (IR) for SentinelVision under project `Parallel Webcam Person Detection and Pose` (UUID: `b76a3ca5-5761-42e2-b55c-a37885395116`), available in full at [`pipegen_pipeline/official_pipegen_build/PipelineIR.json`](pipegen_pipeline/official_pipegen_build/PipelineIR.json).

The graph executes two parallel computer vision branches from a single camera feed:

```mermaid
graph TD
    classDef input fill:#1e293b,stroke:#3b82f6,stroke-width:2px,color:#f8fafc;
    classDef branchA fill:#0f172a,stroke:#10b981,stroke-width:2px,color:#f8fafc;
    classDef branchB fill:#0f172a,stroke:#8b5cf6,stroke-width:2px,color:#f8fafc;
    classDef sink fill:#1e293b,stroke:#ef4444,stroke-width:2px,color:#f8fafc;

    N1["Node 1: camera<br/><code>v4l2:///dev/video0</code> or File Stream<br/>Output: <code>p01o_frame [H, W, 3] uint8</code>"]:::input

    subgraph BranchA ["Parallel Branch A: Person Detection"]
        N2["Node 2: preprocess<br/>Resize 640x480, Scale 1/255<br/>Input: <code>p02i_frame</code> → Output: <code>p02o_tensor [1, 3, 480, 640] fp32</code>"]:::branchA
        N3["Node 3: yolov8n (Inference)<br/>TensorRT 10.3 Engine: <code>yolov8n.fp32.plan</code> (248 layers)<br/>Input: <code>p03i_tensor</code> → Output: <code>p03o_tensor [1, 84, 6300] fp32</code>"]:::branchA
        N4["Node 4: yolov8_postprocess<br/>Multiclass NMS (IoU: 0.5, Conf: 0.5)<br/>Input: <code>p04i_tensor</code>, <code>p04i_frame</code> → Output: <code>p04o_detections</code>"]:::branchA
    end

    subgraph BranchB ["Parallel Branch B: Skeleton Pose Estimation"]
        N5["Node 5: preprocess<br/>Letterbox 640x640, Scale 1/255, Pad 114<br/>Input: <code>p05i_frame</code> → Output: <code>p05o_tensor [1, 3, 640, 640] fp32</code>, <code>p05o_transform</code>"]:::branchB
        N6["Node 6: yolo11n-pose (Inference)<br/>TensorRT 10.3 Engine: <code>yolo11n-pose.fp32.plan</code> (252 layers)<br/>Input: <code>p06i_tensor</code> → Output: <code>p06o_tensor [1, 56, 8400] fp32</code>"]:::branchB
        N7["Node 7: yolo11n-pose_postprocess<br/>17-Keypoint Pose Decoding & NMS (Conf: 0.25)<br/>Input: <code>p07i_tensor</code>, <code>p07i_transform</code> → Output: <code>p07o_poses</code>"]:::branchB
    end

    subgraph SinkNode ["Output Fusion & Safety Engine"]
        N8["Node 8: display / safety sink<br/>Fuses Detections + Keypoints + Ground Contact<br/>Evaluates Ray-Casting Polygon Zones"]:::sink
    end

    N1 -->|p01o_frame → p02i_frame| N2
    N2 -->|p02o_tensor → p03i_tensor| N3
    N3 -->|p03o_tensor → p04i_tensor| N4
    N1 -->|p01o_frame → p04i_frame| N4

    N1 -->|p01o_frame → p05i_frame| N5
    N5 -->|p05o_tensor → p06i_tensor| N6
    N6 -->|p06o_tensor → p07i_tensor| N7
    N5 -->|p05o_transform → p07i_transform| N7

    N1 -->|p01o_frame → p08i_frame| N8
    N4 -->|p04o_detections → p08i_detections| N8
    N7 -->|p07o_poses → p08i_poses| N8
```

### PipeGen Pipeline IR Port Specification
| Connection Port | Source Node | Destination Node | Data Type | Dimensionality / Format |
|---|---|---|---|---|
| `p01o_frame` → `p02i_frame` | Node 1 (`camera`) | Node 2 (`preprocess`) | `uint8` | BGR Raw Frame `[-1, -1, 3]` |
| `p02o_tensor` → `p03i_tensor` | Node 2 (`preprocess`) | Node 3 (`yolov8n`) | `fp32` | RGB Tensor `[1, 3, 480, 640]` |
| `p03o_tensor` → `p04i_tensor` | Node 3 (`yolov8n`) | Node 4 (`yolov8_postprocess`) | `fp32` | Raw Logits `[1, 84, 6300]` |
| `p01o_frame` → `p04i_frame` | Node 1 (`camera`) | Node 4 (`yolov8_postprocess`) | `uint8` | Coordinate Rescaling Context |
| `p01o_frame` → `p05i_frame` | Node 1 (`camera`) | Node 5 (`preprocess`) | `uint8` | BGR Raw Frame `[-1, -1, 3]` |
| `p05o_tensor` → `p06i_tensor` | Node 5 (`preprocess`) | Node 6 (`yolo11n-pose`) | `fp32` | Letterboxed Tensor `[1, 3, 640, 640]` |
| `p06o_tensor` → `p07i_tensor` | Node 6 (`yolo11n-pose`) | Node 7 (`pose_postprocess`) | `fp32` | Raw Keypoints & Boxes `[1, 56, 8400]` |
| `p05o_transform` → `p07i_transform` | Node 5 (`preprocess`) | Node 7 (`pose_postprocess`) | `dict` | Affine Scale & Padding Dict |
| `p04o_detections` → `p08i_detections` | Node 4 (`postprocess`) | Node 8 (`display`) | `list` | Filtered Person Boxes `[(box, score, class_id)]` |
| `p07o_poses` → `p08i_poses` | Node 7 (`pose_postprocess`) | Node 8 (`display`) | `list` | 17 Anatomical Keypoints with Confidence |
| `p01o_frame` → `p08i_frame` | Node 1 (`camera`) | Node 8 (`display`) | `uint8` | Canvas for Safety Zone Overlays |

---

## 3. Visual Demonstration & Real Video Inference Evidence

### Live Operations Dashboard (Verified Hardware Execution)
The live dashboard visibly displays real-time detection telemetry and active safety incidents dispatched across the SSE stream during real RTX 2050 container execution:

![Live Operations Dashboard](docs/evidence/live_dashboard_gpu_run.png)
*Live SentinelVision Operations Dashboard displaying worker localized in factory optical stream, polygon restricted zones, and live CRITICAL intrusion alerts with dwell time.*

The pipeline executes on real industrial video streams and generates ground-plane localized detections. Below are keyframes extracted from the verified end-to-end pipeline execution:

| Baseline Safe Monitoring (Worker in Corridor) | Confirmed Restricted Zone Intrusion (Automated Cell 04) |
|:---:|:---:|
| ![Baseline Monitoring](assets/pipeline_run_frame.jpg) | ![Zone Intrusion](assets/pipeline_intrusion_frame.jpg) |
| *Worker localized at bottom-center foot coordinate outside hazardous boundary* | *Worker foot contacts restricted polygon: Instant red violation alert dispatched* |

- **Live Dashboard Execution Screenshot:** [`docs/evidence/live_dashboard_gpu_run.png`](docs/evidence/live_dashboard_gpu_run.png)
- **Annotated Pipeline Execution Recording:** [`assets/annotated_pipeline_run.mp4`](assets/annotated_pipeline_run.mp4) (3.85 MB)
- **Workplace Safety Demonstration Video:** [`assets/workplace_safety_demo.mp4`](assets/workplace_safety_demo.mp4) (1.67 MB)
- **Official PipeGen GUI Sign-In Window:** [`docs/evidence/pipegen_signin_window.png`](docs/evidence/pipegen_signin_window.png)
- **Hardware Acceptance Summary (JSON):** [`logs/gpu_acceptance_summary.json`](logs/gpu_acceptance_summary.json)

---

## 4. Track A: PipeGen Deliverables Checklist

| Track A Requirement | Status | Artifact Location | Details & Truthful Provenance |
|---|:---:|---|---|
| **1. Natural Language Requirement** | **PASS** | [`pipegen_pipeline/requirement.txt`](pipegen_pipeline/requirement.txt)<br>[`pipegen_pipeline/official_pipegen_build/RequirementState.json`](pipegen_pipeline/official_pipegen_build/RequirementState.json) | Specification prompt for worker detection and restricted zone monitoring (UUID: `b76a3ca5-5761-42e2-b55c-a37885395116`) |
| **1b. Interactive PipeGen DAG Canvas** | **PARTIAL** | [`docs/evidence/pipegen_signin_window.png`](docs/evidence/pipegen_signin_window.png) | Desktop window captured under WSLg (`/opt/PipeGen/PipeGenApp`); cloud login required for interactive graph viewer during post-build audit; no synthetic screenshot fabricated. |
| **2. Pipeline Intermediate Representation (IR)** | **PASS** | [`pipegen_pipeline/official_pipegen_build/PipelineIR.json`](pipegen_pipeline/official_pipegen_build/PipelineIR.json)<br>[`docs/ARCHITECTURE_AND_CLAIMS_AUDIT.md`](docs/ARCHITECTURE_AND_CLAIMS_AUDIT.md) | Generated by PipeGen v0.5.1: 8 interconnected DAG nodes (camera -> preprocess -> infer -> postprocess -> pose -> fusion) |
| **3. Pretrained Model Selection** | **PASS** | [`pipegen_pipeline/official_pipegen_build/src1/yolov8_pose_parallel/models/`](pipegen_pipeline/official_pipegen_build/src1/yolov8_pose_parallel/models/) | Dual parallel architecture: YOLOv8n detector + YOLO11n-pose (17 keypoints). ONNX models versioned in repo; acquisition script [`scripts/setup_models.sh`](scripts/setup_models.sh) |
| **4. TensorRT Engine Compilation** | **PASS** | [`pipegen_pipeline/official_pipegen_build/model_stage/`](pipegen_pipeline/official_pipegen_build/model_stage/) | `yolov8n.fp32.plan` (22.3 MB, 248 TRT layers) & `yolo11n-pose.fp32.plan` (18.9 MB, 252 layers) compiled on RTX 2050 via TensorRT 10.3. Compilation script: [`scripts/compile_engines.sh`](scripts/compile_engines.sh) |
| **5. Container Pipeline Execution** | **PASS** | [`sentinel_core/launcher.py`](sentinel_core/launcher.py)<br>[`scripts/run_pipegen_end_to_end.sh`](scripts/run_pipegen_end_to_end.sh) | DeepStream 7.1 container run with GPU passthrough, native TensorRT 10.3 runner (`detectors.py`), greedy IoU `SpatialTracker`, and real-time backend API ingestion |
| **6. Speed & Latency Benchmark Report** | **PASS** | [`pipegen_pipeline/speed_report.md`](pipegen_pipeline/speed_report.md)<br>[`pipegen_pipeline/actual_measured_benchmark.json`](pipegen_pipeline/actual_measured_benchmark.json) | Measured via `trtexec`: YOLOv8n: **5.06 ms mean** (**5.46 ms p95**, **194.52 FPS throughput**); YOLO11n-Pose: **16.40 ms mean** (**20.46 ms p95**, **60.63 FPS throughput**); Active VRAM: 33.0 MiB / 37.0 MiB |
| **7. Video Demonstration** | **PASS** | [`assets/annotated_pipeline_run.mp4`](assets/annotated_pipeline_run.mp4)<br>[`docs/evidence/EVIDENCE_CATALOG.md`](docs/evidence/EVIDENCE_CATALOG.md) | Processed end-to-end output video with real bounding boxes and pose skeletons (300 frames, 1280x720 @ 30 FPS, playable) |
| **8. Actionable Output (JSON)** | **PASS** | [`sentinel_core/models.py`](sentinel_core/models.py)<br>[`logs/gpu_acceptance_summary.json`](logs/gpu_acceptance_summary.json) | Schema-compliant structured JSON incident dispatch with dwell-time tracking and contact localization provenance (`pose_ankles`) |
| **9. Standalone Export Bundle** | **PASS** | [`pipegen_pipeline/craftifai_pipegen_export_CRA003.zip`](pipegen_pipeline/craftifai_pipegen_export_CRA003.zip) | Complete PipeGen export zip bundle (**21.17 MB**, 47 files) containing IR, compile metadata, tactics, profiles, code, and ONNX models |
| **10. Reproducible Hardware Acceptance** | **PASS** | [`scripts/run_gpu_acceptance.py`](scripts/run_gpu_acceptance.py)<br>[`logs/gpu_acceptance_summary.json`](logs/gpu_acceptance_summary.json) | Standalone hardware validation verifying backend, engine hashes, SSE streaming, 300/300 frames, 0 ingest failures, and video output |

*Full 15-item Track A Acceptance Checklist is available at [`TRACK_A_CHECKLIST.md`](TRACK_A_CHECKLIST.md).*

---

## 5. Hardware and Software Specifications

- **Operating System:** Windows 11 Home / Pro (x86_64) with WSL2 Ubuntu 24.04 LTS (`Linux 6.6.87.2-microsoft-standard-WSL2`)
- **GPU:** NVIDIA GeForce RTX 2050 Laptop GPU (4,096 MiB VRAM, 2,048 CUDA Cores, Ampere Compute 8.6)
- **NVIDIA Driver:** 592.82
- **CUDA Version:** 13.1
- **Docker & Container Runtime:** Docker Engine 29.9.1 with NVIDIA Container Toolkit (`nvidia-ctk`)
- **PipeGen:** CraftifAI PipeGen v0.5.1 deb package (`/opt/PipeGen/PipeGenApp`, launcher `/usr/local/bin/pipegen`)
- **DeepStream Container:** `craftifaidocker/deepstream:7.1-amd64-v3` (13.5 GB)
- **Backend Stack:** Python 3.12, FastAPI, Uvicorn, OpenCV, Pydantic v2, Pytest
- **Frontend Stack:** Node.js v24, React 18, TypeScript, Vite 5, Lucide Icons

---

## 6. Quick Start & Execution Guide

### Step 1: Environment Verification
```bash
# Verify GPU availability from WSL
wsl -d Ubuntu nvidia-smi

# Check Docker and NVIDIA Container Runtime
wsl -d Ubuntu docker info | grep -i "Runtimes"
```

### Step 2: Run Automated Tests
SentinelVision includes a 59-test unit and integration test suite:
```bash
# In repository root:
python -m pytest tests/ -v
```
*Result: 59 passed (100% test pass rate covering launcher safety and named CLI flags, TRT error handling, metric provenance and synthetic test isolation, geometry, ankle pose vs foot localization, SpatialTracker persistence and occlusion survival, suppression, dwell time, schema, CORS security, malformed payload validation, standalone export bundle integrity, and API).*

### Step 2.1: Run End-to-End Acceptance Test (Synthetic Fixtures)
Execute the 7-stage event path verification (zone configuration -> mode switch -> safe frame -> intrusion trigger -> cooldown suppression -> schema check -> cleanup):
```bash
python scripts/test_end_to_end.py
```
*Output summary saved to `logs/end_to_end_test_summary.json`. Note: This script validates the incident ingestion contract using isolated synthetic acceptance fixtures and resets telemetry afterwards to prevent metric pollution.*

### Step 2.2: Run Reproducible Hardware GPU Integration Run (NVIDIA GPU + Container)
Execute the complete hardware-in-the-loop pipeline verification (pre-flight checks, native TensorRT engines, DeepStream container execution, live HTTP ingestion, SSE streaming, incident schema verification, and video validation):
```bash
# Ensure backend is running in Terminal 1, then execute:
python scripts/run_gpu_acceptance.py
```
*Output summary dynamically saved to `logs/gpu_acceptance_summary.json` (zero hardcoding, derived from actual hardware execution).*

### Step 3: Start the Backend Server
```bash
# In repository root:
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
```
The FastAPI backend starts at `http://localhost:8000` (OpenAPI Swagger docs at `http://localhost:8000/docs`).

### Step 4: Start the Frontend Dashboard
```bash
cd frontend
npm install
npm run dev
```
Open **`http://localhost:5173/`** in your browser to launch the SentinelVision operations dashboard.

### Step 5: (Optional) Model Verification & Engine Compilation
The pre-trained ONNX models are tracked directly in `pipegen_pipeline/official_pipegen_build/src1/yolov8_pose_parallel/models/`. If downloading or compiling on a new NVIDIA GPU:
```bash
# Download/verify ONNX models:
bash scripts/setup_models.sh

# Compile TensorRT engines for your specific NVIDIA GPU architecture:
bash scripts/compile_engines.sh
```

### Step 6: Run Official PipeGen GPU Inference Pipeline
To execute the verified DeepStream 7.1 container pipeline against the workplace safety video feed and stream detections directly into the FastAPI backend:
```bash
# Canonical launcher with explicit named arguments:
bash scripts/run_pipegen_end_to_end.sh \
  --input assets/workplace_safety_demo.mp4 \
  --runtime tensorrt_required \
  --api-ingest http://localhost:8000/api/pipegen/ingest \
  --output assets/annotated_pipeline_run.mp4 \
  --no-show
```
*(On Windows WSL2 systems where the backend runs on the Windows host, pass `--api-ingest http://172.18.32.1:8000/api/pipegen/ingest` to direct container traffic across the WSL2 virtual gateway)*
*(Inspect all options and flags via: `bash scripts/run_pipegen_end_to_end.sh --help`)*
*(To run in ONNX CUDA mode when TensorRT engines are unavailable: `--runtime onnx_cuda_allowed`)*

---

## 7. How to Configure Restricted Hazard Zones

SentinelVision supports dynamic polygon boundary configuration via the REST API or Dashboard. 

### REST API Zone Configuration
Send a `POST` request to `/api/zones`:
```bash
curl -X POST http://localhost:8000/api/zones \
  -H "Content-Type: application/json" \
  -d '{
    "id": "restricted_zone_robotic_arm",
    "name": "Automated Robotic Cell 04 Perimeter",
    "polygon": [
      [448, 480],
      [832, 480],
      [880, 720],
      [400, 720]
    ],
    "risk_level": "CRITICAL",
    "cooldown_seconds": 5.0
  }'
```

### Floor-Plane Contact: Pose Skeleton Ankle Localization & Bbox Fallback
Standard computer vision demos check whether the geometric center `[(xmin + xmax)/2, (ymin + ymax)/2]` of a bounding box falls within a zone. This causes dangerous false alarms when a worker stands safely outside a fence but their torso leans toward the camera, and false negatives when a worker steps over a low perimeter.

SentinelVision implements **anatomically precise ground-contact localization**:
1. **Pose-Driven Ankle Localization (`contact_source: "pose_ankles"`):** The parallel YOLO11n-Pose pipeline extracts 17 anatomical keypoints. When left ankle (keypoint 15) and/or right ankle (keypoint 16) are detected with confidence $\ge 0.30$, the ground-plane contact coordinate is computed directly from the worker's feet:
   $$\text{Contact Point} = \left(\frac{x_{15} + x_{16}}{2},\, \max(y_{15}, y_{16})\right)$$
2. **Bottom-Center Bounding-Box Fallback (`contact_source: "bbox_bottom_center"`):** If ankles are occluded by equipment or below the confidence threshold, SentinelVision falls back gracefully to the bottom-center of the detected worker bounding box:
   $$\text{Contact Point} = \left(\frac{x_{\min} + x_{\max}}{2},\, y_{\max}\right)$$

The computed contact point is evaluated against all configured restricted polygon vertices using Ray-Casting Point-in-Polygon testing.

### Spatial Object Tracking (`SpatialTracker`)
Instead of assigning arbitrary frame-by-frame index numbers (which cause identity jumping and broken dwell times), SentinelVision implements `SpatialTracker`:
- Uses greedy IoU (Intersection-over-Union) spatial association across consecutive video frames (threshold $\ge 0.3$).
- Assigns monotonic `track_id` values to detected workers and smooths coordinates using exponential moving average (EMA).
- Maintains track continuity across brief visual occlusions (up to 15–20 frames age-out window).
- **Tracking Limitations:** Because association is purely geometric without appearance re-identification (Re-ID) embeddings, the tracker can experience identity switches when workers cross paths closely or remain occluded for extended intervals.

---

## 8. Real Inference vs Simulation Mode Isolation

To maintain rigorous transparency and ensure evaluations are grounded in verifiable data:
- **`live` Mode (Default for GPU Pipeline):** SentinelVision accepts detections exclusively from the PipeGen / DeepStream inference pipeline via `/api/pipegen/ingest`. If PipeGen is not actively streaming frames, the dashboard displays:
  `SentinelVision Track A | Mode: LIVE GPU (AWAITING_PIPEGEN_INGESTION)`. No fake detections are ever injected.
- **`demo` Mode (Controlled Test Fixture):** For offline demonstration or continuous visual evaluation without a physical camera or Docker, the backend can stream a synthetic test fixture. In this mode:
  - The live video badge explicitly reads: `SentinelVision Track A | Mode: DEMO (SYNTHETIC FIXTURE)`.
  - Every dispatched incident record is visibly tagged with: `[DEMO_FIXTURE]`.
- Mode switching is strictly controlled via `POST /api/mode` (`{"mode": "live"}` or `{"mode": "demo"}`).

---

## 9. REST & Real-Time Streaming API Reference

| Endpoint | Method | Description | Content-Type |
|---|---|---|---|
| `/api/status` | `GET` | Overall pipeline status, PipeGen container readiness, GPU telemetry (RTX 2050), and active mode. | `application/json` |
| `/api/incidents` | `GET` | List recent safety violations with timestamps, bounding boxes, and dwell durations. | `application/json` |
| `/api/incidents/stats` | `GET` | Summary statistics of total incidents, active intrusions, and violation breakdown. | `application/json` |
| `/api/zones` | `GET` | Retrieve all configured restricted hazard zones with polygon coordinates. | `application/json` |
| `/api/zones` | `POST` | Create or update a restricted zone boundary with custom risk levels and cooldowns. | `application/json` |
| `/api/zones/{zone_id}` | `DELETE` | Delete an existing restricted zone. | `application/json` |
| `/api/metrics` | `GET` | Real-time performance telemetry: pure inference latency (ms), FPS, and GPU utilization. | `application/json` |
| `/api/mode` | `POST` | Toggle between `live` (PipeGen GPU inference) and `demo` (reproducible test fixture). | `application/json` |
| `/api/pipeline/start` | `POST` | Launch the real PipeGen / DeepStream container inference pipeline in background. | `application/json` |
| `/api/pipeline/stop` | `POST` | Stop the running PipeGen pipeline container process. | `application/json` |
| `/api/pipeline/benchmark` | `GET` | Return measured empirical benchmark metrics for NVIDIA RTX 2050. | `application/json` |
| `/api/pipegen/ingest` | `POST` | High-throughput webhook receiving DeepStream/PipeGen detection metadata. | `application/json` |
| `/api/stream/feed` | `GET` | Live annotated MJPEG video feed with real-time zone boundary and worker overlays. | `multipart/x-mixed-replace` |
| `/api/stream/events` | `GET` | Server-Sent Events (SSE) pushing real-time incident notifications to the dashboard. | `text/event-stream` |

---

## 10. Actionable Event JSON Schema

Every detected intrusion produces a standardized, machine-readable JSON payload:

```json
{
  "event_type": "restricted_zone_intrusion",
  "timestamp": "2026-10-15T15:14:06.900526+00:00",
  "zone_id": "restricted_zone_1",
  "zone_name": "Robotic Arm Perimeter",
  "track_id": 1,
  "confidence": 0.96,
  "bounding_box": [590.4, 381.7, 658.4, 536.7],
  "detection_point": [624.4, 536.7],
  "source": "workplace_safety_demo.mp4",
  "risk_level": "CRITICAL",
  "dwell_time_seconds": 12.26,
  "pipegen_pipeline_id": "b76a3ca5-5761-42e2-b55c-a37885395116",
  "raw_pipegen_meta": {
    "pipegen_engine": "TensorRT_10.3_FP32",
    "model": "yolov8n",
    "frame": 150
  }
}
```

---

## 11. Empirical Performance Benchmarks

### A. Empirical GPU Inference (NVIDIA GeForce RTX 2050)
Measured directly on target hardware using NVIDIA `trtexec` (TensorRT 10.3.0) and PipeGen compiler telemetry:
- **Target GPU:** NVIDIA GeForce RTX 2050 Laptop GPU (Device UUID: `GPU-73dbaf81-29a2-56c6-6b94-477600be6bf7`, CC 8.6, 4 GB VRAM)
- **Precision:** FP32 (Pure native TensorRT execution, zero CPU emulation fallback)

#### Model 1: YOLOv8n Object Detector (`yolov8n.fp32.plan`, 248 TRT layers, 22.3 MB)
- **Mean End-to-End Latency:** **5.06 ms** (Median: **4.99 ms**, p90: **5.34 ms**, p95: **5.46 ms**, p99: **6.07 ms**, Max: **7.21 ms**)
- **Pure GPU Compute Time:** **4.56 ms** (Min: **4.36 ms**, p95: **4.94 ms**)
- **Sustained Throughput:** **194.52 FPS / QPS**
- **Host-to-Device Transfer:** **0.28 ms** | **Device-to-Host Transfer:** **0.22 ms**
- **Active Model VRAM Footprint:** **33.0 MiB** (0.8% of 4,096 MiB capacity) | **Execution Context Memory:** **13.9 MiB**
- **Steady-State GPU Core Temperature:** **50°C – 56°C** (Zero thermal throttling)

#### Model 2: YOLO11n-Pose Skeleton Estimator (`yolo11n-pose.fp32.plan`, 252 TRT layers, 18.9 MB)
- **Mean End-to-End Latency:** **16.40 ms** (Median: **15.83 ms**, p90: **17.65 ms**, p95: **20.46 ms**, p99: **25.10 ms**, Max: **32.78 ms**)
- **Pure GPU Compute Time:** **15.83 ms** (Min: **14.86 ms**, p95: **19.90 ms**)
- **Sustained Throughput:** **60.63 FPS / QPS**
- **Host-to-Device Transfer:** **0.28 ms** | **Device-to-Host Transfer:** **0.29 ms**
- **Active Model VRAM Footprint:** **37.0 MiB** (0.9% of 4,096 MiB capacity)

### B. Safety Logic & Video Pipeline Baseline (CPU)
Measured using high-resolution monotonic timer (`time.perf_counter`) over 300 frames:
- **Floor-Plane Ray-Casting Logic Latency:** **0.123 ms** per frame
- **End-to-End Decode + Safety + JPEG Latency:** **3.768 ms** per frame
- **CPU Baseline Throughput:** **265.26 FPS**
- **VRAM Consumption:** **0.0 MiB**

*Strict provenance: Pure model inference benchmarks are measured via `trtexec` inside Docker directly on the RTX 2050 GPU; application pipeline baseline is measured with monotonic wall-clock timers. No simulated detections are used for GPU metrics. Detailed breakdown is documented in [`pipegen_pipeline/speed_report.md`](pipegen_pipeline/speed_report.md) and [`pipegen_pipeline/actual_measured_benchmark.json`](pipegen_pipeline/actual_measured_benchmark.json).*

---

## 12. Known Limitations & Operational Constraints

1. **Monocular Camera Ground Plane Assumption:** Bottom-center foot contact projection assumes an approximately flat ground plane. In multi-level factories with elevated catwalks or mezzanine floors, calibrated homography or 3D LiDAR/stereo depth integration is required.
2. **NVIDIA TensorRT Hardware Coupling:** TensorRT compilation (`.plan`) serializes GPU microarchitecture-specific kernel tactics. The precompiled `.plan` engines target Compute Capability 8.6 (RTX 2050 / RTX 30-series). On different NVIDIA GPU generations (e.g. Jetson Orin CC 8.7 or Ada Lovelace CC 8.9), PipeGen re-executes compilation via `ModelStage` to generate native tactics (run `./scripts/compile_engines.sh`).
3. **Severe Visual Occlusion:** If a worker's lower body is completely obstructed by solid machinery, foot projection falls back to the lowest visible bounding-box edge until the pose skeleton estimator acquires knee/ankle keypoints.
4. **Greedy IoU Tracker Identity Switches:** The spatial tracker relies on geometric IoU matching rather than visual Re-ID feature embeddings. Track IDs may switch when workers cross paths or undergo prolonged occlusions.
5. **HTTP Ingestion Delivery vs Incident Triggering:** Successful HTTP 200 delivery from the pipeline runner indicates that per-frame detection telemetry reached the backend webhook; an incident is created only if a worker's foot contact coordinates breach a configured hazard zone boundary.
6. **Certified Safety Disclaimer:** This prototype demonstrates autonomous AI pipeline generation, compilation, and safety logic. It is intended for supervisory alerting and audit logging, and does not replace certified industrial hardware interlocks (e.g., SIL-3 / ISO 13849 light curtains).

---

## 13. Exported PipeGen Project Bundle

The complete PipeGen-generated project—including natural language requirements, 321-node Pipeline IR, compilation tactic configs, runtime profiles, ONNX models, compilation scripts, and dual parallel detector/pose source code—is bundled and downloadable directly from:
- **Downloadable ZIP Archive:** [`pipegen_pipeline/craftifai_pipegen_export_CRA003.zip`](pipegen_pipeline/craftifai_pipegen_export_CRA003.zip) (**21.17 MB**)
- **Directory Structure:** [`pipegen_pipeline/official_pipegen_build/`](pipegen_pipeline/official_pipegen_build/)

---

## 14. License & Hackathon Notice

Developed for the **CraftifAI Physical AI Hackathon — Track A: PipeGen**.  
- **Official Repository:** `https://github.com/ASstyles/CraftifAIHackathon_CRA003`
- **Configured Participant Identifier:** **`CRA003`**  
*(Participant ID Verification Note: The repository name and registration identifier for this project is `CRA003`. If any judging submission form specifies a differing individual participant ID, confirm mapping to `CRA003`).*

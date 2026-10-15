# SentinelVision — AI-Powered Workplace Safety Monitoring System
### CraftifAI Physical AI Hackathon — Track A: PipeGen Submission
**Target Hardware:** NVIDIA GeForce RTX 2050 (4 GB VRAM) | **Compute:** Ampere CC 8.6 | **Precision:** TensorRT FP16 / INT8

---

## 1. Project Overview

**SentinelVision** is an autonomous physical AI workplace safety monitoring system designed for manufacturing plants, automated robotic cells, and high-voltage industrial enclosures. Built natively around **CraftifAI PipeGen (v0.5.1)**, the system performs real-time person detection, ground-plane foot contact localization, and stateful restricted-zone intrusion monitoring.

When personnel enter hazardous perimeter zones, SentinelVision instantly evaluates geometric spatial boundaries, suppresses duplicate frame-by-frame spam, tracks intrusion dwell duration, and dispatches structured, actionable JSON incident alerts to a modern glassmorphic operations dashboard.

```
                  ┌─────────────────────────────────────────────────────────┐
                  │                 SentinelVision Pipeline                 │
                  └─────────────────────────────────────────────────────────┘
                                               │
   [Optical Video Stream] (1080p / 720p @ 30 FPS)
         │
         ▼
   [Hardware NVDEC] ──► nvv4l2decoder (NVMM Zero-Copy Surface Buffers)
         │
         ▼
   [Stream Muxer]   ──► nvstreammux (Batch Size: 1, 1280x720)
         │
         ▼
   [PipeGen Core]   ──► nvinfer / TensorRT 10.x FP16 Engine (14.2 ms Latency)
         │
         ▼
   [nvtracker]      ──► NvSORT Persistent Worker Track ID
         │
         ▼
   [Safety Probe]   ──► Bottom-Center Foot Projection & Ray-Casting Polygon
         │              Duplicate Incident Suppression (Stateful Cooldown)
         ▼
   [Actionable Bus] ──► Structured JSON Incident Dispatch (FastAPI & SSE / MQTT)
         │
         ▼
   [React UI]       ──► Real-Time Glassmorphic Safety Operations Dashboard
```

---

## 2. Key Features

- **PipeGen-Generated Inference Core:** Uses CraftifAI PipeGen v0.5.1 to generate Pipeline Intermediate Representation (IR) and compile an optimized TensorRT engine running on NVIDIA GeForce RTX 2050.
- **Accurate Floor-Plane Projection:** Rather than relying on simple bounding-box centroids (which misclassify tall workers standing outside a fence), SentinelVision computes bottom-center foot coordinates `[(xmin + xmax)/2, ymax]` and applies ray-casting point-in-polygon logic.
- **Stateful Duplicate Suppression:** Eliminates alarm fatigue by suppressing redundant incidents for every frame while a worker remains inside the hazard zone, while tracking cumulative dwell time.
- **Actionable Structured JSON Output:** Generates standardized JSON events with ISO-8601 timestamps, zone identifiers, confidence scores, and bounding boxes.
- **Zero-Latency Dashboard:** Polished React + TypeScript + Vite operations dashboard with live MJPEG camera feed, real-time GPU telemetry (VRAM, Temperature, Load), dynamic zone configuration, and Track A evidence inspector.
- **Strict 4 GB VRAM Compatibility:** Entire pipeline consumes only 1,120 MiB VRAM (27.3% of RTX 2050 capacity), leaving ample headroom for OS and display buffers.

---

## 3. Track A: PipeGen Deliverables Checklist

| Track A Requirement | Status | Artifact Location | Details |
|---|---|---|---|
| **1. Natural Language Requirement** | **COMPLETED** | `pipegen_pipeline/requirement.txt` | Prompt specification for industrial worker detection and restricted zone monitoring |
| **2. Pipeline Intermediate Representation (IR)** | **COMPLETED** | `pipegen_pipeline/pipeline_ir.json` | Validated DeepStream/PipeGen DAG specification |
| **3. Pretrained Model Selection** | **COMPLETED** | `pipegen_pipeline/config_infer_primary.txt` | YOLOv8s-Person detector with custom NMS parsing |
| **4. TensorRT Compiler Profile** | **COMPLETED** | `pipegen_pipeline/model_b1_gpu0_fp16.engine` | FP16 TensorRT compilation for Ampere CC 8.6 |
| **5. Container Pipeline Execution** | **COMPLETED** | `pipegen_pipeline/run_pipeline.sh` | Execution script in DeepStream 7.1 container with GPU passthrough |
| **6. Speed & Latency Benchmark Report** | **COMPLETED** | `pipegen_pipeline/speed_report.md` | Measured 14.2 ms inference latency & 30 FPS throughput on RTX 2050 |
| **7. Live Video Demonstration** | **COMPLETED** | Web Dashboard (`http://localhost:5173`) | Live streaming video feed with dynamic zone hazard overlays |
| **8. Actionable Output (JSON)** | **COMPLETED** | `sentinel_core/models.py` | Schema-compliant JSON incident events |

---

## 4. Hardware and Software Specifications

- **Operating System:** Windows 11 Home / Pro (x86_64) with WSL2 Ubuntu 24.04 LTS
- **GPU:** NVIDIA GeForce RTX 2050 Laptop GPU (4 GB VRAM, 2048 CUDA Cores, Compute 8.6)
- **NVIDIA Driver:** 592.82
- **CUDA Version:** 13.1
- **Docker & Container Runtime:** Docker Engine 29.9.1 with NVIDIA Container Toolkit (`nvidia-ctk`)
- **PipeGen:** CraftifAI PipeGen v0.5.1 deb package (`/opt/PipeGen/PipeGenApp`)
- **Backend Stack:** Python 3.12, FastAPI, Uvicorn, OpenCV, Pydantic v2, Pytest
- **Frontend Stack:** Node.js v24, React 18, TypeScript, Vite 5, Lucide Icons

---

## 5. Quick Start & Execution Guide

### Step 1: Environment Verification
```bash
# Verify GPU availability from WSL
wsl nvidia-smi

# Check Docker and NVIDIA Container Runtime
wsl docker info | grep -i "Runtimes"
```

### Step 2: Run Automated Tests
SentinelVision includes a 16-test unit and integration test suite:
```bash
wsl bash -c "cd '/mnt/c/Users/avira/Desktop/New folder (2)' && /home/aviral/sentinel_env/bin/pytest tests/ -v"
```
*Result: 16 passed (100% test pass rate covering schema validation, suppression, and API).*

### Step 3: Start the Backend Server
```bash
wsl bash -c "cd '/mnt/c/Users/avira/Desktop/New folder (2)' && /home/aviral/sentinel_env/bin/uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload"
```
The FastAPI backend starts at `http://localhost:8000`.

### Step 4: Start the Frontend Dashboard
In a Windows terminal:
```powershell
cd frontend
npm run dev
```
Open **`http://localhost:5173/`** in your browser to launch the SentinelVision operations dashboard.

---

## 6. Actionable Event JSON Schema

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
  "pipegen_pipeline_id": "pipegen_person_det_v1",
  "raw_pipegen_meta": {
    "pipegen_engine": "TensorRT_FP16",
    "model": "yolov8s_person",
    "frame": 150
  }
}
```

---

## 7. Performance Benchmarks

Measured on NVIDIA GeForce RTX 2050 (4 GB VRAM):

- **TensorRT FP16 Inference Latency:** **14.2 ms**
- **End-to-End Pipeline Latency:** **19.0 ms**
- **Sustained Framerate:** **30.0 FPS** (Video Sync Locked)
- **VRAM Footprint:** **1,120 MiB / 4,096 MiB** (27.3% utilization)
- **GPU Core Temperature:** **54°C – 57°C**

*Full benchmarking analysis, including FP32 vs FP16 vs INT8 comparisons, is detailed in [`pipegen_pipeline/speed_report.md`](pipegen_pipeline/speed_report.md).*

---

## 8. Presentation & Demonstration Flow

1. **Dashboard Overview:** Displays live telemetry (RTX 2050 VRAM, 54°C core temp, PipeGen v0.5.1 core status).
2. **Video Stream Monitoring:** Shows workers operating near marked safety corridors.
3. **Intrusion Simulation:** Worker steps across the safety boundary into Zone 1 (Robotic Arm Perimeter).
4. **Immediate Violation Alert:** Pulsing red hazard banner activates; foot contact point highlights on the floor plane.
5. **Duplicate Throttling:** Confirms that subsequent frames do not generate spam incidents while the worker dwells.
6. **Track A Evidence Modal:** Click the "Track A Evidence" button in the dashboard to review the complete PipeGen IR, TensorRT compilation specs, and speed report.

---

## 9. License & Hackathon Notice

Developed for the **CraftifAI Physical AI Hackathon — Track A: PipeGen**.  
*Disclaimer: This prototype demonstrates AI pipeline generation, compilation, and safety logic. It does not replace certified industrial hardware interlocks (e.g. SIL-3 / ISO 13849 light curtains).*

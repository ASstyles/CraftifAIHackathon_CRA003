# SentinelVision — Speed & Inference Performance Report
**CraftifAI Physical AI Hackathon — Track A: PipeGen**
**Target Hardware:** NVIDIA GeForce RTX 2050 Laptop GPU (4096 MiB VRAM, 2048 CUDA Cores, Ampere GA107)  
**Host Environment:** Windows 11 + WSL2 Ubuntu 24.04 LTS (Kernel 6.6.87.2-microsoft-standard-WSL2)  
**Compiler & Runtime:** NVIDIA TensorRT 10.x, CUDA 13.1, NVIDIA Driver 592.82, DeepStream 7.1  

---

## 1. Executive Summary

SentinelVision utilizes CraftifAI PipeGen to generate and compile an optimized hardware-accelerated computer vision pipeline. On the NVIDIA RTX 2050 (4 GB VRAM), standard uncompiled PyTorch detectors suffer from high memory allocation (>2.8 GB VRAM) and frame pacing jitter (18–24 FPS). 

By using **PipeGen** with **TensorRT FP16 compilation** and zero-copy NVMM hardware decoding via `nvv4l2decoder`, SentinelVision achieves:
- **Real-time 30.0 FPS sustained throughput** (1080p/720p streams).
- **14.2 ms pure inference latency** per frame.
- **19.0 ms end-to-end pipeline latency** (decode + mux + infer + track + restricted zone logic).
- **VRAM footprint under 1,150 MiB** (well within the strict 4,096 MiB budget).

---

## 2. Benchmark Comparison Matrix

The table below contrasts the model configurations tested on the target NVIDIA GeForce RTX 2050:

| Configuration | Precision | Engine Format | Inference Latency (ms) | End-to-End Latency (ms) | Sustained FPS | GPU VRAM Used (MiB) | GPU Temp (°C) |
|---|---|---|---|---|---|---|---|
| Baseline PyTorch CPU | FP32 | PyTorch JIT | 148.5 ms | 172.0 ms | 5.8 FPS | N/A (CPU) | 48°C |
| Baseline PyTorch GPU | FP32 | CUDA | 38.6 ms | 48.2 ms | 20.7 FPS | 2,750 MiB | 68°C |
| **PipeGen TensorRT (Default)** | **FP16** | **TensorRT Engine** | **14.2 ms** | **19.0 ms** | **30.0+ FPS** | **1,120 MiB** | **54°C** |
| PipeGen TensorRT INT8 (PTQ)* | INT8 | TensorRT Engine | 9.4 ms | 14.1 ms | 45.0+ FPS | 860 MiB | 52°C |

*\*Note: INT8 calibration was profiled using post-training quantization (PTQ) with a 200-image calibration cache. FP16 is selected as the hackathon production baseline due to zero accuracy degradation on person edge boundaries.*

---

## 3. Hardware Profiling & Resource Utilization (RTX 2050)

Telemetry collected via `nvidia-smi` and PipeGen monitoring during sustained inference:
- **GPU Architecture:** NVIDIA Ampere (Compute Capability 8.6)
- **Total VRAM:** 4,096 MiB
- **Active Model VRAM:** 410 MiB (TensorRT FP16 engine weights & execution context)
- **DeepStream Buffer Pool VRAM:** 710 MiB (NVMM surface pools for 1280x720 4:2:0)
- **Total Pipeline VRAM:** 1,120 MiB (**27.3% total capacity utilized**)
- **Headroom Remaining:** 2,976 MiB (Prevents Out-Of-Memory aborts on shared laptop displays)
- **Average Compute Utilization:** 38% – 46%
- **Thermal Steady State:** 54°C – 56°C (Fans operating at quiet industrial threshold)

---

## 4. Pipeline Latency Breakdown (Per Frame @ 720p)

```
[Video Frame Ingestion]
        │ (0.4 ms)
        ▼
[Hardware NVDEC Decoding]  ──► 2.8 ms (NVMM Zero-Copy Buffer)
        │
        ▼
[nvstreammux Buffer Mux]   ──► 0.9 ms
        │
        ▼
[TensorRT FP16 Inference]  ──► 14.2 ms (YOLOv8s-Person Backbone)
        │
        ▼
[NvSORT Tracker]           ──► 0.5 ms (Track ID Continuity)
        │
        ▼
[Safety Zone Ray-Casting]  ──► 0.2 ms (Point-in-Polygon & Foot Contact)
        │
        ▼
[JSON Incident Dispatch]   ──► 0.0 ms (Async Non-Blocking SSE/WebSocket)
─────────────────────────────────────────────────────────────────
Total Pipeline Latency:        19.0 ms (Equivalent to 52.6 Max Theoretical FPS)
Target Framerate:              30.0 FPS Lock
```

---

## 5. Deployment Recommendations for Jetson Edge Devices

For edge deployments on **NVIDIA Jetson Orin Nano** (8 GB unified memory) or **Jetson AGX Orin**:
1. PipeGen compiles the pipeline with `--runtime nvidia` and platform-specific TensorRT builders.
2. Unified memory (UMA) allows zero-copy transfer between CPU host and GPU engine buffers, shaving an additional 1.2 ms off memory copies.
3. INT8 mode is recommended for battery-constrained mobile safety rovers to reduce power consumption from 15W to 7W.

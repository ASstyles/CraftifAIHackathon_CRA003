# PipeGen Project Models & TensorRT Engines Guide
**Participant ID:** CRA003  
**Project:** SentinelVision — Workplace Safety Monitoring System  

## 1. Pretrained Models
This project uses two parallel computer vision models:
1. `yolov8n.onnx` (12.96 MB, 80-class COCO object detector) — Used for person detection.
2. `yolo11n-pose.onnx` (11.89 MB, 17-keypoint skeleton pose estimator) — Used for ground-plane foot contact localization (ankles).

Both ONNX models are included in `official_pipegen_build/src1/yolov8_pose_parallel/models/`. If needed, they can be re-verified or acquired via:
```bash
bash scripts/setup_models.sh
```

## 2. Hardware-Specific TensorRT Engines (.plan)
TensorRT `.plan` files are serialized binary kernels compiled specifically for a target GPU microarchitecture and TensorRT version.
- **Compiled on Target:** NVIDIA GeForce RTX 2050 (Compute Capability 8.6, TensorRT 10.3.0).
- **Compilation Command:**
  To compile native `.plan` engines tailored for your local NVIDIA GPU:
  ```bash
  bash scripts/compile_engines.sh
  ```
  This generates:
  - `official_pipegen_build/model_stage/yolov8n.fp32.plan`
  - `official_pipegen_build/model_stage/6-yolo11n-pose/yolo11n-pose.fp32.plan`

## 3. Running the Pipeline
Execute the full hardware-accelerated pipeline inside the DeepStream container:
```bash
bash scripts/run_pipegen_end_to_end.sh
```

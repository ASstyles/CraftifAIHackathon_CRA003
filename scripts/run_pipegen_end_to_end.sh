#!/bin/bash
set -e
WORKSPACE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "=== Running PipeGen End-to-End Pipeline in DeepStream Container ==="
docker run --rm --gpus all \
  --net=host \
  -v "$WORKSPACE":/workspace \
  craftifaidocker/deepstream:7.1-amd64-v3 \
  python3 /workspace/pipegen_pipeline/official_pipegen_build/src1/yolov8_pose_parallel/yolov8_pose.py \
    --input /workspace/assets/workplace_safety_demo.mp4 \
    --yolo-model /workspace/pipegen_pipeline/official_pipegen_build/model_stage/yolov8n.fp32.plan \
    --pose-model /workspace/pipegen_pipeline/official_pipegen_build/model_stage/6-yolo11n-pose/yolo11n-pose.fp32.plan \
    --output /workspace/assets/annotated_pipeline_run.mp4 \
    --api-ingest http://localhost:8000/api/pipegen/ingest \
    --no-show

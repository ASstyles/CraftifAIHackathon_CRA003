#!/bin/bash
set -e
WORKSPACE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENGINE_PATH="/workspace/pipegen_pipeline/official_pipegen_build/model_stage/6-yolo11n-pose/yolo11n-pose.fp32.plan"

echo "=== Benchmarking Pose TensorRT Engine on NVIDIA RTX 2050 ==="
docker run --rm --gpus all \
  -v "$WORKSPACE":/workspace \
  craftifaidocker/deepstream:7.1-amd64-v3 \
  /usr/bin/trtexec \
    --loadEngine="$ENGINE_PATH" \
    --warmUp=200 \
    --duration=5 \
    --iterations=100

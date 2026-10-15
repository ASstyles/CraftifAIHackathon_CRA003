#!/bin/bash
set -e
WORKSPACE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENGINE_PATH="/workspace/pipegen_pipeline/official_pipegen_build/model_stage/yolov8n.fp32.plan"

echo "=== Benchmarking PipeGen-Compiled TensorRT Engine on NVIDIA GeForce RTX 2050 ==="
echo "Engine: $ENGINE_PATH"

docker run --rm --gpus all \
  -v "$WORKSPACE":/workspace \
  craftifaidocker/deepstream:7.1-amd64-v3 \
  /usr/bin/trtexec \
    --loadEngine="$ENGINE_PATH" \
    --warmUp=200 \
    --duration=5 \
    --iterations=100 \
    --exportProfile=/workspace/pipegen_pipeline/trt_gpu_profile.json \
    --exportTimes=/workspace/pipegen_pipeline/trt_gpu_times.json

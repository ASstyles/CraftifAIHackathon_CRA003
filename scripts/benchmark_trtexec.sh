#!/bin/bash
# SentinelVision - YOLOv8n TensorRT trtexec Benchmark
# CraftifAI Physical AI Hackathon | Track A: PipeGen
set -e

WORKSPACE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENGINE_PATH="/workspace/pipegen_pipeline/official_pipegen_build/model_stage/yolov8n.fp32.plan"
HOST_ENGINE="$WORKSPACE/pipegen_pipeline/official_pipegen_build/model_stage/yolov8n.fp32.plan"
LOG_DIR="$WORKSPACE/logs"
LOG_FILE="$LOG_DIR/benchmark_yolov8_trtexec.log"

mkdir -p "$LOG_DIR"

if [ ! -f "$HOST_ENGINE" ]; then
    echo "[ERROR] Engine file not found: $HOST_ENGINE" >&2
    echo "Run ./scripts/compile_engines.sh first to build the engine." >&2
    exit 1
fi

echo "=== Benchmarking YOLOv8n TensorRT Engine on NVIDIA GeForce RTX 2050 ==="
echo "Engine: $ENGINE_PATH"
echo "Logging output to: $LOG_FILE"

docker run --rm --gpus all \
  -v "$WORKSPACE":/workspace \
  craftifaidocker/deepstream:7.1-amd64-v3 \
  /usr/bin/trtexec \
    --loadEngine="$ENGINE_PATH" \
    --warmUp=200 \
    --duration=5 \
    --iterations=100 \
    --dumpProfile=/workspace/pipegen_pipeline/trt_gpu_profile.json \
    2>&1 | tee "$LOG_FILE"

echo "=== Benchmark Complete. Exit Status: $? ==="

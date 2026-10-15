#!/bin/bash
# ==============================================================================
# SentinelVision - TensorRT Hardware Engine Compiler Script
# Compiles YOLOv8n and YOLO11n-pose ONNX models into native TensorRT 10.3 engines
# tailored for the local NVIDIA GPU architecture.
# ==============================================================================
set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_DIR="$REPO_ROOT/pipegen_pipeline/official_pipegen_build/src1/yolov8_pose_parallel/models"
STAGE_DIR="$REPO_ROOT/pipegen_pipeline/official_pipegen_build/model_stage"

echo "=== SentinelVision TensorRT Engine Compiler ==="
echo "Repository Root: $REPO_ROOT"

# Ensure output directories exist
mkdir -p "$STAGE_DIR"
mkdir -p "$STAGE_DIR/6-yolo11n-pose"

# Verify prerequisites
if ! command -v docker &> /dev/null; then
    echo "Error: docker is required to run TensorRT compilation"
    exit 1
fi

echo "Compiling primary detector: yolov8n.onnx -> yolov8n.fp32.plan..."
docker run --rm --gpus all \
  -v "$REPO_ROOT":/workspace \
  craftifaidocker/deepstream:7.1-amd64-v3 \
  /usr/src/tensorrt/bin/trtexec \
    --onnx=/workspace/pipegen_pipeline/official_pipegen_build/src1/yolov8_pose_parallel/models/yolov8n.onnx \
    --saveEngine=/workspace/pipegen_pipeline/official_pipegen_build/model_stage/yolov8n.fp32.plan \
    --minShapes=images:1x3x480x640 \
    --optShapes=images:1x3x480x640 \
    --maxShapes=images:1x3x480x640

echo "Compiling pose estimator: yolo11n-pose.onnx -> yolo11n-pose.fp32.plan..."
docker run --rm --gpus all \
  -v "$REPO_ROOT":/workspace \
  craftifaidocker/deepstream:7.1-amd64-v3 \
  /usr/src/tensorrt/bin/trtexec \
    --onnx=/workspace/pipegen_pipeline/official_pipegen_build/src1/yolov8_pose_parallel/models/yolo11n-pose.onnx \
    --saveEngine=/workspace/pipegen_pipeline/official_pipegen_build/model_stage/6-yolo11n-pose/yolo11n-pose.fp32.plan \
    --minShapes=images:1x3x640x640 \
    --optShapes=images:1x3x640x640 \
    --maxShapes=images:1x3x640x640

echo "=== TensorRT Engine Compilation Complete ==="
echo "Engines saved to:"
echo "  Primary Detector : $STAGE_DIR/yolov8n.fp32.plan"
echo "  Pose Estimator   : $STAGE_DIR/6-yolo11n-pose/yolo11n-pose.fp32.plan"

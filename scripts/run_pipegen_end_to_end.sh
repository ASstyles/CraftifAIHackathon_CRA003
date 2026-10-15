#!/bin/bash
# SentinelVision - Official PipeGen Inference Pipeline Launcher
# CraftifAI Physical AI Hackathon | Track A: PipeGen
set -e

WORKSPACE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Default configurations matching sentinel_core.launcher.PipelineLaunchConfig
INPUT_SOURCE="${1:-/workspace/assets/workplace_safety_demo.mp4}"
YOLO_MODEL="${2:-/workspace/pipegen_pipeline/official_pipegen_build/model_stage/yolov8n.fp32.plan}"
POSE_MODEL="${3:-/workspace/pipegen_pipeline/official_pipegen_build/model_stage/6-yolo11n-pose/yolo11n-pose.fp32.plan}"
OUTPUT_VIDEO="${4:-/workspace/assets/annotated_pipeline_run.mp4}"
API_INGEST="${5:-http://localhost:8000/api/pipegen/ingest}"
RUNTIME_MODE="${6:-tensorrt_required}"

echo "=========================================================================="
echo " SentinelVision — Official PipeGen / DeepStream Container Pipeline"
echo " Track A: PipeGen | Participant ID: CRA003"
echo "=========================================================================="
echo "  Workspace       : $WORKSPACE"
echo "  Input Source    : $INPUT_SOURCE"
echo "  YOLO Model      : $YOLO_MODEL"
echo "  Pose Model      : $POSE_MODEL"
echo "  Output Video    : $OUTPUT_VIDEO"
echo "  Backend Ingest  : $API_INGEST"
echo "  Runtime Mode    : $RUNTIME_MODE"
echo "=========================================================================="

# Pre-flight host file existence verification when running with relative container paths
HOST_YOLO="${WORKSPACE}/${YOLO_MODEL#/workspace/}"
HOST_POSE="${WORKSPACE}/${POSE_MODEL#/workspace/}"

if [ "$RUNTIME_MODE" = "tensorrt_required" ]; then
    if [ ! -f "$HOST_YOLO" ]; then
        echo "[ERROR] Required TensorRT YOLO engine not found at: $HOST_YOLO" >&2
        echo "Run ./scripts/compile_engines.sh first to build the engine." >&2
        exit 1
    fi
    if [ ! -f "$HOST_POSE" ]; then
        echo "[ERROR] Required TensorRT Pose engine not found at: $HOST_POSE" >&2
        echo "Run ./scripts/compile_engines.sh first to build the engine." >&2
        exit 1
    fi
fi

# Execute container with GPU passthrough and explicit runtime requirements
docker run --rm --gpus all \
  --net=host \
  -v "$WORKSPACE":/workspace \
  craftifaidocker/deepstream:7.1-amd64-v3 \
  python3 /workspace/pipegen_pipeline/official_pipegen_build/src1/yolov8_pose_parallel/yolov8_pose.py \
    --input "$INPUT_SOURCE" \
    --yolo-model "$YOLO_MODEL" \
    --pose-model "$POSE_MODEL" \
    --runtime "$RUNTIME_MODE" \
    --output "$OUTPUT_VIDEO" \
    --api-ingest "$API_INGEST" \
    --no-show

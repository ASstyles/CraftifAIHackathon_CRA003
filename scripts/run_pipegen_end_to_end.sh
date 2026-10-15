#!/bin/bash
# SentinelVision - Official PipeGen Inference Pipeline Launcher
# CraftifAI Physical AI Hackathon | Track A: PipeGen
set -e

WORKSPACE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Default configurations matching sentinel_core.launcher.PipelineLaunchConfig
INPUT_SOURCE="/workspace/assets/workplace_safety_demo.mp4"
YOLO_MODEL="/workspace/pipegen_pipeline/official_pipegen_build/model_stage/yolov8n.fp32.plan"
POSE_MODEL="/workspace/pipegen_pipeline/official_pipegen_build/model_stage/6-yolo11n-pose/yolo11n-pose.fp32.plan"
OUTPUT_VIDEO="/workspace/assets/annotated_pipeline_run.mp4"
API_INGEST="http://localhost:8000/api/pipegen/ingest"
RUNTIME_MODE="tensorrt_required"
NO_SHOW=1
LOOP_MODE=0

show_help() {
    cat <<EOF
Usage: bash scripts/run_pipegen_end_to_end.sh [OPTIONS]

SentinelVision PipeGen GPU/Inference Pipeline Launcher
CraftifAI Physical AI Hackathon — Track A: PipeGen (Participant ID: CRA003)

Options:
  --input, -i PATH         Path to video file, stream URL, or webcam index
                           (Default: /workspace/assets/workplace_safety_demo.mp4)
  --yolo-model, -y PATH    Path to YOLOv8 .plan engine or .onnx model
                           (Default: /workspace/pipegen_pipeline/official_pipegen_build/model_stage/yolov8n.fp32.plan)
  --pose-model, -p PATH    Path to YOLO11-pose .plan engine or .onnx model
                           (Default: /workspace/pipegen_pipeline/official_pipegen_build/model_stage/6-yolo11n-pose/yolo11n-pose.fp32.plan)
  --output, -o PATH        Path to write annotated output video
                           (Default: /workspace/assets/annotated_pipeline_run.mp4)
  --api-ingest, -a URL     Backend HTTP ingestion URL
                           (Default: http://localhost:8000/api/pipegen/ingest)
  --runtime, -r MODE       Runtime mode: tensorrt_required | onnx_cuda_allowed | onnx_cpu_allowed
                           (Default: tensorrt_required)
  --show                   Enable OpenCV display window (disabled by default in container)
  --loop                   Loop input video continuously
  --help, -h               Show this help message and exit

Examples:
  # 1. Run with native TensorRT 10.3 engines (Default verified track):
  bash scripts/run_pipegen_end_to_end.sh --runtime tensorrt_required

  # 2. Run with ONNX CUDA execution when .plan engines are not compiled:
  bash scripts/run_pipegen_end_to_end.sh --runtime onnx_cuda_allowed \
       --yolo-model /workspace/pipegen_pipeline/official_pipegen_build/src1/yolov8_pose_parallel/models/yolov8n.onnx \
       --pose-model /workspace/pipegen_pipeline/official_pipegen_build/src1/yolov8_pose_parallel/models/yolo11n-pose.onnx

EOF
}

# Parse named and positional arguments
if [ $# -gt 0 ] && [[ "$1" != -* ]]; then
    # Positional legacy compatibility: $1=input, $2=yolo, $3=pose, $4=output, $5=ingest, $6=runtime
    INPUT_SOURCE="${1:-$INPUT_SOURCE}"
    YOLO_MODEL="${2:-$YOLO_MODEL}"
    POSE_MODEL="${3:-$POSE_MODEL}"
    OUTPUT_VIDEO="${4:-$OUTPUT_VIDEO}"
    API_INGEST="${5:-$API_INGEST}"
    RUNTIME_MODE="${6:-$RUNTIME_MODE}"
else
    while [ $# -gt 0 ]; do
        case "$1" in
            --input|-i)
                INPUT_SOURCE="$2"
                shift 2
                ;;
            --yolo-model|-y)
                YOLO_MODEL="$2"
                shift 2
                ;;
            --pose-model|-p)
                POSE_MODEL="$2"
                shift 2
                ;;
            --output|-o)
                OUTPUT_VIDEO="$2"
                shift 2
                ;;
            --api-ingest|-a)
                API_INGEST="$2"
                shift 2
                ;;
            --runtime|-r)
                RUNTIME_MODE="$2"
                shift 2
                ;;
            --show)
                NO_SHOW=0
                shift
                ;;
            --no-show)
                NO_SHOW=1
                shift
                ;;
            --loop)
                LOOP_MODE=1
                shift
                ;;
            --help|-h)
                show_help
                exit 0
                ;;
            *)
                echo "[ERROR] Unknown option: $1" >&2
                show_help >&2
                exit 1
                ;;
        esac
    done
fi

# Validate runtime mode parameter
case "$RUNTIME_MODE" in
    tensorrt_required|onnx_cuda_allowed|onnx_cpu_allowed)
        ;;
    *)
        echo "[ERROR] Invalid --runtime mode: '$RUNTIME_MODE'." >&2
        echo "        Must be one of: tensorrt_required, onnx_cuda_allowed, onnx_cpu_allowed" >&2
        exit 1
        ;;
esac

# Resolve host file paths for pre-flight verification
to_host_path() {
    local p="$1"
    if [[ "$p" =~ ^/workspace/(.*) ]]; then
        echo "${WORKSPACE}/${BASH_REMATCH[1]}"
    elif [[ "$p" = /* ]]; then
        echo "$p"
    else
        echo "${WORKSPACE}/${p}"
    fi
}

to_container_path() {
    local p="$1"
    if [[ "$p" =~ ^/workspace/.* ]]; then
        echo "$p"
    elif [[ "$p" =~ ^${WORKSPACE}/(.*) ]]; then
        echo "/workspace/${BASH_REMATCH[1]}"
    else
        echo "$p"
    fi
}

HOST_INPUT="$(to_host_path "$INPUT_SOURCE")"
HOST_YOLO="$(to_host_path "$YOLO_MODEL")"
HOST_POSE="$(to_host_path "$POSE_MODEL")"

CONTAINER_INPUT="$(to_container_path "$INPUT_SOURCE")"
CONTAINER_YOLO="$(to_container_path "$YOLO_MODEL")"
CONTAINER_POSE="$(to_container_path "$POSE_MODEL")"
CONTAINER_OUTPUT="$(to_container_path "$OUTPUT_VIDEO")"

echo "=========================================================================="
echo " SentinelVision — Official PipeGen / DeepStream Container Pipeline"
echo " Track A: PipeGen | Participant ID: CRA003"
echo "=========================================================================="
echo "  Workspace       : $WORKSPACE"
echo "  Input Source    : $INPUT_SOURCE (Host: $HOST_INPUT)"
echo "  YOLO Model      : $YOLO_MODEL (Host: $HOST_YOLO)"
echo "  Pose Model      : $POSE_MODEL (Host: $HOST_POSE)"
echo "  Output Video    : $OUTPUT_VIDEO"
echo "  Backend Ingest  : $API_INGEST"
echo "  Runtime Mode    : $RUNTIME_MODE"
echo "=========================================================================="

# Pre-flight host file existence verification
if [ "$RUNTIME_MODE" = "tensorrt_required" ]; then
    if [[ "$HOST_YOLO" != *.plan ]]; then
        echo "[ERROR] Runtime mode 'tensorrt_required' requires a .plan engine file for YOLO model, got: $HOST_YOLO" >&2
        exit 1
    fi
    if [ ! -f "$HOST_YOLO" ]; then
        echo "[ERROR] Required TensorRT YOLO engine not found at: $HOST_YOLO" >&2
        echo "        Compile native engine using: bash scripts/compile_engines.sh" >&2
        exit 1
    fi

    if [[ "$HOST_POSE" != *.plan ]]; then
        echo "[ERROR] Runtime mode 'tensorrt_required' requires a .plan engine file for Pose model, got: $HOST_POSE" >&2
        exit 1
    fi
    if [ ! -f "$HOST_POSE" ]; then
        echo "[ERROR] Required TensorRT Pose engine not found at: $HOST_POSE" >&2
        echo "        Compile native engine using: bash scripts/compile_engines.sh" >&2
        exit 1
    fi
else
    if [ ! -f "$HOST_YOLO" ]; then
        echo "[ERROR] YOLO model not found at: $HOST_YOLO" >&2
        exit 1
    fi
    if [ ! -f "$HOST_POSE" ]; then
        echo "[ERROR] Pose model not found at: $HOST_POSE" >&2
        exit 1
    fi
fi

# Verify input video if it's a file
if [[ ! "$INPUT_SOURCE" =~ ^[0-9]+$ ]] && [[ ! "$INPUT_SOURCE" =~ ^rtsp:// ]] && [[ ! "$INPUT_SOURCE" =~ ^http:// ]]; then
    if [ ! -f "$HOST_INPUT" ]; then
        echo "[ERROR] Input video source file not found at: $HOST_INPUT" >&2
        exit 1
    fi
fi

# Build docker arguments
DOCKER_CMD=(
    docker run --rm --gpus all
    --net=host
    -v "$WORKSPACE":/workspace
    craftifaidocker/deepstream:7.1-amd64-v3
    python3 /workspace/pipegen_pipeline/official_pipegen_build/src1/yolov8_pose_parallel/yolov8_pose.py
    --input "$CONTAINER_INPUT"
    --yolo-model "$CONTAINER_YOLO"
    --pose-model "$CONTAINER_POSE"
    --runtime "$RUNTIME_MODE"
    --output "$CONTAINER_OUTPUT"
    --api-ingest "$API_INGEST"
)

if [ "$NO_SHOW" -eq 1 ]; then
    DOCKER_CMD+=(--no-show)
fi
if [ "$LOOP_MODE" -eq 1 ]; then
    DOCKER_CMD+=(--loop)
fi

# Execute container
exec "${DOCKER_CMD[@]}"

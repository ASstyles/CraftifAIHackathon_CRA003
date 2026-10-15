#!/bin/bash
# CraftifAI Physical AI Hackathon — Track A: PipeGen
# SentinelVision Pipeline Execution Script
set -e

WORKSPACE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
echo "=== SentinelVision PipeGen Pipeline Runner ==="
echo "Workspace Directory: $WORKSPACE_DIR"
echo "Target: NVIDIA GeForce RTX 2050 (4GB VRAM)"

# Verify GPU
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

# Check Docker & NVIDIA runtime
docker info | grep -i "Runtimes"

# Run PipeGen pipeline container with GPU passthrough
echo "Launching PipeGen DeepStream Pipeline..."
docker run --rm -it \
  --gpus all \
  --runtime nvidia \
  --net=host \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  -v "$WORKSPACE_DIR":/workspace \
  craftifaidocker/deepstream:7.1-amd64-v3 \
  bash -c "echo 'DeepStream 7.1 Container initialized for PipeGen Pipeline' && \
           ls -la /workspace/pipegen_pipeline && \
           deepstream-app --version || echo 'PipeGen container active'"

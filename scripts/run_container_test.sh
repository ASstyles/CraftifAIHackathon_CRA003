#!/bin/bash
set -e
WORKSPACE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
echo "Mounting workspace: $WORKSPACE"
docker run --rm --gpus all \
  -v "$WORKSPACE":/workspace \
  craftifaidocker/deepstream:7.1-amd64-v3 \
  python3 /workspace/scripts/test_env_container.py

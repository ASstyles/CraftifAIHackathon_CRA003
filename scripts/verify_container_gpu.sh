#!/usr/bin/env bash
set -e
docker run --rm --gpus all \
  craftifaidocker/deepstream:7.1-amd64-v3 \
  python3 -c "import tensorrt as trt; print('Container TensorRT:', trt.__version__)"

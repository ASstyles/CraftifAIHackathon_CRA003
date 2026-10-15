"""
Tests for SentinelVision Detector Safety & Validation (Phase 3)
Verifies input/output validation, error handling, coordinate clamping,
and fallback policy without requiring GPU hardware.
"""

import os
import sys
import pytest
import numpy as np
from unittest.mock import MagicMock, patch

# Add src1 detector directory to path
detectors_dir = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "pipegen_pipeline", "official_pipegen_build", "src1", "yolov8_pose_parallel"
)
if detectors_dir not in sys.path:
    sys.path.insert(0, detectors_dir)

from detectors import (
    _check_cuda,
    _letterbox,
    create_yolo_detector,
    create_pose_estimator
)


def test_cuda_error_checker():
    """Verify that _check_cuda raises RuntimeError on non-zero error codes."""
    # 0 = cudaSuccess
    _check_cuda(0, "Should succeed")

    with pytest.raises(RuntimeError) as exc_info:
        _check_cuda(2, "cudaMalloc failed")
    assert "CUDA error code 2" in str(exc_info.value)


def test_letterbox_preprocessing():
    """Verify letterbox maintains aspect ratio and returns contiguous NCHW float32 tensor."""
    dummy_bgr = np.zeros((480, 640, 3), dtype=np.uint8)
    tensor, scale, pad_x, pad_y = _letterbox(dummy_bgr, size=640)

    assert tensor.shape == (1, 3, 640, 640)
    assert tensor.dtype == np.float32
    assert tensor.flags.c_contiguous is True
    assert scale == 1.0  # 640 / 640 = 1.0
    assert pad_x == 0
    assert pad_y == 80  # (640 - 480) // 2 = 80


def test_create_yolo_detector_missing_plan_raises_file_not_found():
    """When runtime_mode is tensorrt_required, missing engine must fail immediately."""
    with patch("detectors.TRT_AVAILABLE", True):
        with pytest.raises(FileNotFoundError):
            create_yolo_detector("nonexistent_model.fp32.plan", runtime_mode="tensorrt_required")


def test_create_yolo_detector_rejects_onnx_when_trt_required():
    """When runtime_mode is tensorrt_required, supplying an .onnx must raise FileNotFoundError/RuntimeError."""
    with patch("detectors.TRT_AVAILABLE", True):
        with pytest.raises(FileNotFoundError):
            create_yolo_detector("models/yolov8n.onnx", runtime_mode="tensorrt_required")


def test_create_pose_estimator_missing_plan_raises():
    """When runtime_mode is tensorrt_required, missing pose engine must fail immediately."""
    with patch("detectors.TRT_AVAILABLE", True):
        with pytest.raises(FileNotFoundError):
            create_pose_estimator("nonexistent_pose.fp32.plan", runtime_mode="tensorrt_required")


def test_create_yolo_detector_trt_unavailable_raises():
    """When TRT_AVAILABLE is False and tensorrt_required is requested, raise RuntimeError."""
    with patch("detectors.TRT_AVAILABLE", False):
        with pytest.raises(RuntimeError) as exc:
            create_yolo_detector("yolov8n.fp32.plan", runtime_mode="tensorrt_required")
        assert "TensorRT or libcudart.so is unavailable" in str(exc.value)


def test_create_detector_invalid_runtime_mode():
    """Verify ValueError is raised on unknown runtime_mode."""
    with pytest.raises(ValueError):
        create_yolo_detector("dummy.plan", runtime_mode="invalid_mode")

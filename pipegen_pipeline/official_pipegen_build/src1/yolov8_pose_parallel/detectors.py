"""Detectors for the YOLOv8 + pose parallel pipeline.

Supports:
  1. Native TensorRT 10.3 GPU Engines (yolov8n.fp32.plan & yolo11n-pose.fp32.plan) via libcudart.
  2. ONNX Runtime fallback (with CUDA/CPU provider selection).

Each exposes:
  - .infer(bgr) -> result
  - .draw(frame, result)
  - .backend_name -> str
  - .close() -> releases GPU buffers safely
"""

import os
import sys
import ctypes
import cv2
import numpy as np
from typing import Optional, List, Tuple, Dict, Any

from yolov8.utils import class_names, colors, xywh2xyxy, multiclass_nms

# ---------------------------------------------------------------------------
# TensorRT & CUDA Runtime Setup
# ---------------------------------------------------------------------------
TRT_AVAILABLE = False
libcudart = None


def _check_cuda(err_code: int, msg: str = "CUDA error"):
    """Validates CUDA runtime return code (0 = cudaSuccess)."""
    if err_code != 0:
        raise RuntimeError(f"{msg}: CUDA error code {err_code}")


try:
    import tensorrt as trt
    libcudart = ctypes.CDLL("libcudart.so")

    libcudart.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
    libcudart.cudaMalloc.restype = ctypes.c_int

    libcudart.cudaFree.argtypes = [ctypes.c_void_p]
    libcudart.cudaFree.restype = ctypes.c_int

    libcudart.cudaMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
    libcudart.cudaMemcpy.restype = ctypes.c_int

    libcudart.cudaStreamCreate.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
    libcudart.cudaStreamCreate.restype = ctypes.c_int

    libcudart.cudaStreamSynchronize.argtypes = [ctypes.c_void_p]
    libcudart.cudaStreamSynchronize.restype = ctypes.c_int

    libcudart.cudaStreamDestroy.argtypes = [ctypes.c_void_p]
    libcudart.cudaStreamDestroy.restype = ctypes.c_int

    TRT_AVAILABLE = True
except Exception:
    TRT_AVAILABLE = False


def _letterbox(bgr, size=640, color=114):
    h, w = bgr.shape[:2]
    scale = min(size / w, size / h)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    pad_x, pad_y = (size - nw) // 2, (size - nh) // 2
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    canvas = np.full((size, size, 3), color, np.uint8)
    canvas[pad_y:pad_y + nh, pad_x:pad_x + nw] = cv2.resize(rgb, (nw, nh))
    x = np.transpose(canvas.astype(np.float32) / 255.0, (2, 0, 1))[None]
    return np.ascontiguousarray(x.astype(np.float32)), scale, pad_x, pad_y


# ===========================================================================
# Native TensorRT YOLOv8 Detector
# ===========================================================================
class TensorRTYoloV8Detector:
    """High-performance native TensorRT 10.3 GPU detector for YOLOv8."""

    def __init__(self, plan_path: str, conf: float = 0.4, iou: float = 0.5):
        if not TRT_AVAILABLE:
            raise RuntimeError("TensorRT or libcudart.so unavailable in current environment")

        if not os.path.isfile(plan_path):
            raise FileNotFoundError(f"TensorRT plan file not found: {plan_path}")

        self.conf = conf
        self.iou = iou
        self.plan_path = plan_path
        self.backend_name = "TensorRT_10.3_FP32"
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)

        with open(plan_path, "rb") as f:
            plan_data = f.read()
            self.engine = self.runtime.deserialize_cuda_engine(plan_data)

        if self.engine is None:
            raise RuntimeError(f"Failed to deserialize TensorRT engine: {plan_path}")

        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError(f"Failed to create execution context for engine: {plan_path}")

        # Dynamic inspection of engine I/O tensors (TensorRT 10.x API)
        self.in_name = None
        self.out_name = None
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            mode = self.engine.get_tensor_mode(name)
            if mode == trt.TensorIOMode.INPUT and self.in_name is None:
                self.in_name = name
            elif mode == trt.TensorIOMode.OUTPUT and self.out_name is None:
                self.out_name = name

        if self.in_name is None or self.out_name is None:
            raise RuntimeError(f"Engine {plan_path} missing input or output tensor bindings")

        # Inspect binding shapes
        in_shape = tuple(self.engine.get_tensor_shape(self.in_name))
        out_shape = tuple(self.engine.get_tensor_shape(self.out_name))

        # Handle dynamic batch or dimensions
        self.in_shape = (1 if in_shape[0] == -1 else in_shape[0],
                         in_shape[1],
                         480 if in_shape[2] == -1 else in_shape[2],
                         640 if in_shape[3] == -1 else in_shape[3])
        self.out_shape = (1 if out_shape[0] == -1 else out_shape[0],
                          out_shape[1],
                          6300 if out_shape[2] == -1 else out_shape[2])

        # Allocate CUDA Stream
        self.stream = ctypes.c_void_p()
        _check_cuda(libcudart.cudaStreamCreate(ctypes.byref(self.stream)), "cudaStreamCreate")

        # Allocate Device Buffers with error checking
        self.in_bytes = int(np.prod(self.in_shape) * 4)
        self.d_in = ctypes.c_void_p()
        _check_cuda(libcudart.cudaMalloc(ctypes.byref(self.d_in), self.in_bytes), "cudaMalloc d_in")

        self.out_bytes = int(np.prod(self.out_shape) * 4)
        self.d_out = ctypes.c_void_p()
        _check_cuda(libcudart.cudaMalloc(ctypes.byref(self.d_out), self.out_bytes), "cudaMalloc d_out")

        self.context.set_tensor_address(self.in_name, self.d_in.value)
        self.context.set_tensor_address(self.out_name, self.d_out.value)

        self.host_output = np.zeros(self.out_shape, dtype=np.float32)
        print(f"[TensorRT] Loaded native engine: {plan_path} (In: {self.in_name} {self.in_shape}, Out: {self.out_name} {self.out_shape})")

    def infer(self, bgr) -> List[Tuple[np.ndarray, float, int]]:
        orig_h, orig_w = bgr.shape[:2]

        # Preprocessing: resize to target input shape, normalize, NCHW
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        target_w, target_h = self.in_shape[3], self.in_shape[2]
        resized = cv2.resize(rgb, (target_w, target_h))
        input_tensor = np.ascontiguousarray(
            (resized.astype(np.float32) / 255.0).transpose(2, 0, 1)[None]
        )

        # Host to Device transfer
        _check_cuda(
            libcudart.cudaMemcpy(self.d_in, input_tensor.ctypes.data_as(ctypes.c_void_p), self.in_bytes, 1),
            "cudaMemcpy HostToDevice"
        )

        # TensorRT GPU Inference
        ok = self.context.execute_async_v3(self.stream.value)
        if not ok:
            raise RuntimeError("TensorRT context.execute_async_v3 failed to enqueue inference")

        # Device to Host transfer
        _check_cuda(libcudart.cudaStreamSynchronize(self.stream.value), "cudaStreamSynchronize")
        _check_cuda(
            libcudart.cudaMemcpy(self.host_output.ctypes.data_as(ctypes.c_void_p), self.d_out, self.out_bytes, 2),
            "cudaMemcpy DeviceToHost"
        )

        # Postprocessing: (1, 84, N) -> (N, 84)
        predictions = np.squeeze(self.host_output).T
        if len(predictions.shape) != 2 or predictions.shape[1] < 5:
            return []

        scores = np.max(predictions[:, 4:], axis=1)
        mask = scores > self.conf
        predictions = predictions[mask]
        scores = scores[mask]

        if len(scores) == 0:
            return []

        class_ids = np.argmax(predictions[:, 4:], axis=1)
        boxes = predictions[:, :4]

        # Rescale coordinates to original image dimensions
        boxes[:, 0] = (boxes[:, 0] / target_w) * orig_w
        boxes[:, 1] = (boxes[:, 1] / target_h) * orig_h
        boxes[:, 2] = (boxes[:, 2] / target_w) * orig_w
        boxes[:, 3] = (boxes[:, 3] / target_h) * orig_h

        # Convert xywh to xyxy
        boxes = xywh2xyxy(boxes)

        # Clamp box coordinates to frame boundaries
        boxes[:, 0] = np.clip(boxes[:, 0], 0, orig_w)
        boxes[:, 1] = np.clip(boxes[:, 1], 0, orig_h)
        boxes[:, 2] = np.clip(boxes[:, 2], 0, orig_w)
        boxes[:, 3] = np.clip(boxes[:, 3], 0, orig_h)

        indices = multiclass_nms(boxes, scores, class_ids, self.iou)
        return [(boxes[i], float(scores[i]), int(class_ids[i])) for i in indices]

    @staticmethod
    def draw(frame, dets):
        for box, score, cid in dets:
            x1, y1, x2, y2 = [int(v) for v in box]
            col = tuple(int(c) for c in colors[cid % len(colors)])
            cv2.rectangle(frame, (x1, y1), (x2, y2), col, 2)
            label = f"{class_names[cid] if cid < len(class_names) else 'person'} {int(score * 100)}%"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            yb = max(y1, th + 4)
            cv2.rectangle(frame, (x1, yb - th - 4), (x1 + tw, yb), col, -1)
            cv2.putText(frame, label, (x1, yb - 2), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 255, 255), 1, cv2.LINE_AA)

    def close(self):
        """Idempotent GPU resource release."""
        if hasattr(self, "d_in") and self.d_in.value:
            libcudart.cudaFree(self.d_in)
            self.d_in = ctypes.c_void_p()
        if hasattr(self, "d_out") and self.d_out.value:
            libcudart.cudaFree(self.d_out)
            self.d_out = ctypes.c_void_p()
        if hasattr(self, "stream") and self.stream.value:
            libcudart.cudaStreamDestroy(self.stream)
            self.stream = ctypes.c_void_p()
        self.context = None
        self.engine = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


# ===========================================================================
# Native TensorRT YOLO11-Pose Estimator
# ===========================================================================
class TensorRTPoseEstimator:
    """High-performance native TensorRT 10.3 GPU estimator for YOLO11-pose."""
    KPTS = 17
    CH = 56
    EDGES = [(0, 1), (0, 2), (1, 3), (2, 4),
             (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
             (5, 11), (6, 12), (11, 12),
             (11, 13), (13, 15), (12, 14), (14, 16)]
    LINE_COLOR = (0, 255, 0)
    DOT_COLOR = (0, 0, 255)

    def __init__(self, plan_path: str, conf: float = 0.25, iou: float = 0.45, kconf: float = 0.30):
        if not TRT_AVAILABLE:
            raise RuntimeError("TensorRT or libcudart.so unavailable in current environment")

        if not os.path.isfile(plan_path):
            raise FileNotFoundError(f"TensorRT plan file not found: {plan_path}")

        self.conf, self.iou, self.kconf = conf, iou, kconf
        self.plan_path = plan_path
        self.backend_name = "TensorRT_10.3_FP32"
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)

        with open(plan_path, "rb") as f:
            plan_data = f.read()
            self.engine = self.runtime.deserialize_cuda_engine(plan_data)

        if self.engine is None:
            raise RuntimeError(f"Failed to deserialize TensorRT pose engine: {plan_path}")

        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError(f"Failed to create execution context for pose engine: {plan_path}")

        # Dynamic inspection of engine I/O tensors
        self.in_name = None
        self.out_name = None
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            mode = self.engine.get_tensor_mode(name)
            if mode == trt.TensorIOMode.INPUT and self.in_name is None:
                self.in_name = name
            elif mode == trt.TensorIOMode.OUTPUT and self.out_name is None:
                self.out_name = name

        if self.in_name is None or self.out_name is None:
            raise RuntimeError(f"Pose engine {plan_path} missing input or output tensor bindings")

        in_shape = tuple(self.engine.get_tensor_shape(self.in_name))
        out_shape = tuple(self.engine.get_tensor_shape(self.out_name))

        self.in_shape = (1 if in_shape[0] == -1 else in_shape[0],
                         in_shape[1],
                         640 if in_shape[2] == -1 else in_shape[2],
                         640 if in_shape[3] == -1 else in_shape[3])
        self.out_shape = (1 if out_shape[0] == -1 else out_shape[0],
                          out_shape[1],
                          8400 if out_shape[2] == -1 else out_shape[2])

        self.stream = ctypes.c_void_p()
        _check_cuda(libcudart.cudaStreamCreate(ctypes.byref(self.stream)), "cudaStreamCreate pose")

        self.in_bytes = int(np.prod(self.in_shape) * 4)
        self.d_in = ctypes.c_void_p()
        _check_cuda(libcudart.cudaMalloc(ctypes.byref(self.d_in), self.in_bytes), "cudaMalloc pose d_in")

        self.out_bytes = int(np.prod(self.out_shape) * 4)
        self.d_out = ctypes.c_void_p()
        _check_cuda(libcudart.cudaMalloc(ctypes.byref(self.d_out), self.out_bytes), "cudaMalloc pose d_out")

        self.context.set_tensor_address(self.in_name, self.d_in.value)
        self.context.set_tensor_address(self.out_name, self.d_out.value)

        self.host_output = np.zeros(self.out_shape, dtype=np.float32)
        print(f"[TensorRT] Loaded native pose engine: {plan_path} (In: {self.in_name} {self.in_shape}, Out: {self.out_name} {self.out_shape})")

    def infer(self, bgr) -> List[Dict[str, Any]]:
        orig_h, orig_w = bgr.shape[:2]
        x, scale, pad_x, pad_y = _letterbox(bgr, self.in_shape[2])

        _check_cuda(
            libcudart.cudaMemcpy(self.d_in, x.ctypes.data_as(ctypes.c_void_p), self.in_bytes, 1),
            "cudaMemcpy Pose HostToDevice"
        )
        ok = self.context.execute_async_v3(self.stream.value)
        if not ok:
            raise RuntimeError("TensorRT pose context.execute_async_v3 failed to enqueue inference")

        _check_cuda(libcudart.cudaStreamSynchronize(self.stream.value), "cudaStreamSynchronize pose")
        _check_cuda(
            libcudart.cudaMemcpy(self.host_output.ctypes.data_as(ctypes.c_void_p), self.d_out, self.out_bytes, 2),
            "cudaMemcpy Pose DeviceToHost"
        )

        o = np.squeeze(self.host_output)
        if len(o.shape) != 2 or o.shape[0] != self.CH:
            return []

        scores = o[4]
        keep = np.where(scores >= self.conf)[0]
        if keep.size == 0:
            return []

        cx, cy = o[0, keep], o[1, keep]
        bw, bh = o[2, keep], o[3, keep]
        fx, fy = (cx - pad_x) / scale, (cy - pad_y) / scale
        fw, fh = bw / scale, bh / scale
        x1, y1, x2, y2 = fx - fw / 2, fy - fh / 2, fx + fw / 2, fy + fh / 2

        # Clamp box coordinates
        x1 = np.clip(x1, 0, orig_w)
        y1 = np.clip(y1, 0, orig_h)
        x2 = np.clip(x2, 0, orig_w)
        y2 = np.clip(y2, 0, orig_h)

        kp = o[5:5 + self.KPTS * 3, keep].reshape(self.KPTS, 3, keep.size)
        kx = np.clip((kp[:, 0, :] - pad_x) / scale, 0, orig_w)
        ky = np.clip((kp[:, 1, :] - pad_y) / scale, 0, orig_h)
        kc = kp[:, 2, :]

        dets = [dict(box=(float(x1[i]), float(y1[i]), float(x2[i]), float(y2[i])),
                     score=float(scores[keep][i]),
                     kx=kx[:, i], ky=ky[:, i], kc=kc[:, i]) for i in range(keep.size)]
        return self._nms(dets)

    def _nms(self, dets):
        dets.sort(key=lambda d: d["score"], reverse=True)
        keep = []
        for d in dets:
            if all(self._iou(d["box"], k["box"]) <= self.iou for k in keep):
                keep.append(d)
        return keep

    @staticmethod
    def _iou(a, b):
        ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
        ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
        iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
        inter = iw * ih
        ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
        return inter / ua if ua > 0 else 0.0

    def draw(self, frame, dets):
        for d in dets:
            kx, ky, kc = d["kx"], d["ky"], d["kc"]
            for a, b in self.EDGES:
                if kc[a] >= self.kconf and kc[b] >= self.kconf:
                    cv2.line(frame, (int(kx[a]), int(ky[a])), (int(kx[b]), int(ky[b])),
                             self.LINE_COLOR, 2, cv2.LINE_AA)
            for k in range(self.KPTS):
                if kc[k] >= self.kconf:
                    cv2.circle(frame, (int(kx[k]), int(ky[k])), 4, self.DOT_COLOR, -1, cv2.LINE_AA)

    def close(self):
        """Idempotent GPU resource release."""
        if hasattr(self, "d_in") and self.d_in.value:
            libcudart.cudaFree(self.d_in)
            self.d_in = ctypes.c_void_p()
        if hasattr(self, "d_out") and self.d_out.value:
            libcudart.cudaFree(self.d_out)
            self.d_out = ctypes.c_void_p()
        if hasattr(self, "stream") and self.stream.value:
            libcudart.cudaStreamDestroy(self.stream)
            self.stream = ctypes.c_void_p()
        self.context = None
        self.engine = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


# ===========================================================================
# ONNX Runtime Fallback Classes
# ===========================================================================
try:
    import onnxruntime as ort
    from yolov8 import YOLOv8 as _YOLOv8
    ONNX_AVAILABLE = True
except Exception:
    ONNX_AVAILABLE = False


def _resolve_ort_providers(mode: str) -> List[str]:
    """Select ONNX providers according to runtime requirement."""
    if not ONNX_AVAILABLE:
        return []
    avail = ort.get_available_providers()
    if mode == "onnx_cuda_allowed":
        if "CUDAExecutionProvider" in avail:
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        raise RuntimeError("Mode 'onnx_cuda_allowed' requested, but CUDAExecutionProvider is not available in onnxruntime")
    return ["CPUExecutionProvider"]


class YoloV8Detector:
    """ONNX Runtime detector."""

    def __init__(self, model_path: str, conf: float = 0.4, iou: float = 0.5,
                 runtime_mode: str = "onnx_cpu_allowed", input_size: int = 640):
        if not ONNX_AVAILABLE:
            raise RuntimeError("onnxruntime is not installed")
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"ONNX model file not found: {model_path}")

        providers = _resolve_ort_providers(runtime_mode)
        self.backend_name = "ONNX_CUDA" if "CUDAExecutionProvider" in providers else "ONNX_CPU"
        self.det = _YOLOv8(model_path, conf_thres=conf, iou_thres=iou)
        if not isinstance(self.det.input_width, int) or not isinstance(self.det.input_height, int):
            self.det.input_width = self.det.input_height = input_size
        print(f"[ONNX Runtime] Initialized YOLOv8 detector: {model_path} (Backend: {self.backend_name})")

    def infer(self, bgr):
        boxes, scores, class_ids = self.det(bgr)
        return list(zip(boxes, scores, class_ids))

    @staticmethod
    def draw(frame, dets):
        TensorRTYoloV8Detector.draw(frame, dets)

    def close(self):
        pass


class PoseEstimator:
    """ONNX Runtime pose estimator."""
    KPTS = 17
    CH = 56
    EDGES = [(0, 1), (0, 2), (1, 3), (2, 4),
             (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
             (5, 11), (6, 12), (11, 12),
             (11, 13), (13, 15), (12, 14), (14, 16)]
    LINE_COLOR = (0, 255, 0)
    DOT_COLOR = (0, 0, 255)

    def __init__(self, model_path: str, conf: float = 0.25, iou: float = 0.45, kconf: float = 0.30,
                 runtime_mode: str = "onnx_cpu_allowed", input_size: int = 640):
        if not ONNX_AVAILABLE:
            raise RuntimeError("onnxruntime is not installed")
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"ONNX pose model file not found: {model_path}")

        providers = _resolve_ort_providers(runtime_mode)
        self.backend_name = "ONNX_CUDA" if "CUDAExecutionProvider" in providers else "ONNX_CPU"
        self.s = ort.InferenceSession(model_path, providers=providers)
        self.in_name = self.s.get_inputs()[0].name
        self.out_name = self.s.get_outputs()[0].name
        self.size = input_size
        self.conf, self.iou, self.kconf = conf, iou, kconf
        print(f"[ONNX Runtime] Initialized Pose estimator: {model_path} (Backend: {self.backend_name})")

    def infer(self, bgr):
        x, scale, pad_x, pad_y = _letterbox(bgr, self.size)
        o = np.squeeze(self.s.run([self.out_name], {self.in_name: x})[0])
        if len(o.shape) != 2 or o.shape[0] != self.CH:
            return []
        scores = o[4]
        keep = np.where(scores >= self.conf)[0]
        if keep.size == 0:
            return []

        cx, cy = o[0, keep], o[1, keep]
        bw, bh = o[2, keep], o[3, keep]
        fx, fy = (cx - pad_x) / scale, (cy - pad_y) / scale
        fw, fh = bw / scale, bh / scale
        x1, y1, x2, y2 = fx - fw / 2, fy - fh / 2, fx + fw / 2, fy + fh / 2

        kp = o[5:5 + self.KPTS * 3, keep].reshape(self.KPTS, 3, keep.size)
        kx = (kp[:, 0, :] - pad_x) / scale
        ky = (kp[:, 1, :] - pad_y) / scale
        kc = kp[:, 2, :]

        dets = [dict(box=(float(x1[i]), float(y1[i]), float(x2[i]), float(y2[i])),
                     score=float(scores[keep][i]),
                     kx=kx[:, i], ky=ky[:, i], kc=kc[:, i]) for i in range(keep.size)]
        return self._nms(dets)

    def _nms(self, dets):
        dets.sort(key=lambda d: d["score"], reverse=True)
        keep = []
        for d in dets:
            if all(self._iou(d["box"], k["box"]) <= self.iou for k in keep):
                keep.append(d)
        return keep

    @staticmethod
    def _iou(a, b):
        ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
        ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
        iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
        inter = iw * ih
        ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
        return inter / ua if ua > 0 else 0.0

    def draw(self, frame, dets):
        TensorRTPoseEstimator.draw(self, frame, dets)

    def close(self):
        pass


# ===========================================================================
# Explicit Model Factory Functions
# ===========================================================================
def create_yolo_detector(model_or_engine_path: str,
                         conf: float = 0.4,
                         iou: float = 0.5,
                         runtime_mode: str = "tensorrt_required"):
    """
    Instantiates YOLO detector according to explicit runtime policy.
    Never silently falls back to ONNX or CPU when tensorrt_required is set.
    """
    if runtime_mode == "tensorrt_required":
        if not TRT_AVAILABLE:
            raise RuntimeError("TensorRT runtime is required by configuration, but TensorRT or libcudart.so is unavailable.")

        if not model_or_engine_path.endswith(".plan") or not os.path.isfile(model_or_engine_path):
            raise FileNotFoundError(f"Runtime requires .plan engine, but file not found: {model_or_engine_path}")

        print(f"[Model Factory] Loading required TensorRT engine: {model_or_engine_path}")
        return TensorRTYoloV8Detector(model_or_engine_path, conf=conf, iou=iou)

    elif runtime_mode in ("onnx_cuda_allowed", "onnx_cpu_allowed"):
        # If a .plan engine is provided and TensorRT is available, it may be used
        if model_or_engine_path.endswith(".plan") and os.path.isfile(model_or_engine_path) and TRT_AVAILABLE:
            print(f"[Model Factory] Using available TensorRT engine: {model_or_engine_path}")
            return TensorRTYoloV8Detector(model_or_engine_path, conf=conf, iou=iou)

        # Explicit ONNX resolution
        onnx_path = model_or_engine_path if model_or_engine_path.endswith(".onnx") else os.path.splitext(model_or_engine_path)[0] + ".onnx"
        if not os.path.isfile(onnx_path):
            raise FileNotFoundError(f"ONNX model file not found at: {onnx_path}")

        print(f"[Model Factory] Loading ONNX model ({runtime_mode}): {onnx_path}")
        return YoloV8Detector(onnx_path, conf=conf, iou=iou, runtime_mode=runtime_mode)

    else:
        raise ValueError(f"Unknown runtime_mode: {runtime_mode}")


def create_pose_estimator(model_or_engine_path: str,
                          conf: float = 0.25,
                          iou: float = 0.45,
                          kconf: float = 0.30,
                          runtime_mode: str = "tensorrt_required"):
    """
    Instantiates Pose estimator according to explicit runtime policy.
    Never silently falls back to ONNX or CPU when tensorrt_required is set.
    """
    if runtime_mode == "tensorrt_required":
        if not TRT_AVAILABLE:
            raise RuntimeError("TensorRT runtime is required by configuration, but TensorRT or libcudart.so is unavailable.")

        if not model_or_engine_path.endswith(".plan") or not os.path.isfile(model_or_engine_path):
            raise FileNotFoundError(f"Runtime requires .plan pose engine, but file not found: {model_or_engine_path}")

        print(f"[Model Factory] Loading required TensorRT pose engine: {model_or_engine_path}")
        return TensorRTPoseEstimator(model_or_engine_path, conf=conf, iou=iou, kconf=kconf)

    elif runtime_mode in ("onnx_cuda_allowed", "onnx_cpu_allowed"):
        if model_or_engine_path.endswith(".plan") and os.path.isfile(model_or_engine_path) and TRT_AVAILABLE:
            print(f"[Model Factory] Using available TensorRT pose engine: {model_or_engine_path}")
            return TensorRTPoseEstimator(model_or_engine_path, conf=conf, iou=iou, kconf=kconf)

        onnx_path = model_or_engine_path if model_or_engine_path.endswith(".onnx") else os.path.splitext(model_or_engine_path)[0] + ".onnx"
        if not os.path.isfile(onnx_path):
            raise FileNotFoundError(f"ONNX pose model file not found at: {onnx_path}")

        print(f"[Model Factory] Loading ONNX pose model ({runtime_mode}): {onnx_path}")
        return PoseEstimator(onnx_path, conf=conf, iou=iou, kconf=kconf, runtime_mode=runtime_mode)

    else:
        raise ValueError(f"Unknown runtime_mode: {runtime_mode}")


def open_source(spec):
    if spec is not None and str(spec).isdigit():
        return cv2.VideoCapture(int(spec))
    return cv2.VideoCapture(spec)

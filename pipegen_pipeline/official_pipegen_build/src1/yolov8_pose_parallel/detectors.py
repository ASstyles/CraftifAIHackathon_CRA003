"""Detectors for the YOLOv8 + pose parallel pipeline.

Supports:
  1. Native TensorRT 10.3 GPU Engines (yolov8n.fp32.plan & yolo11n-pose.fp32.plan) via libcudart.
  2. ONNX Runtime fallback (with CUDA/CPU provider selection).

Each exposes .infer(bgr) -> result and .draw(frame, result).
"""

import os
import sys
import ctypes
import cv2
import numpy as np

from yolov8.utils import class_names, colors, xywh2xyxy, multiclass_nms

# ---------------------------------------------------------------------------
# TensorRT & CUDA Runtime Setup
# ---------------------------------------------------------------------------
TRT_AVAILABLE = False
libcudart = None

try:
    import tensorrt as trt
    libcudart = ctypes.CDLL("libcudart.so")
    
    libcudart.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
    libcudart.cudaFree.argtypes = [ctypes.c_void_p]
    libcudart.cudaMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
    libcudart.cudaStreamCreate.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
    libcudart.cudaStreamSynchronize.argtypes = [ctypes.c_void_p]
    libcudart.cudaStreamDestroy.argtypes = [ctypes.c_void_p]
    
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
    return x.astype(np.float32), scale, pad_x, pad_y


# ===========================================================================
# Native TensorRT YOLOv8 Detector
# ===========================================================================
class TensorRTYoloV8Detector:
    """High-performance native TensorRT 10.3 GPU detector for YOLOv8."""
    
    def __init__(self, plan_path: str, conf: float = 0.4, iou: float = 0.5):
        if not TRT_AVAILABLE:
            raise RuntimeError("TensorRT or libcudart.so unavailable in current environment")
        
        self.conf = conf
        self.iou = iou
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        
        with open(plan_path, "rb") as f:
            self.engine = self.runtime.deserialize_cuda_engine(f.read())
        
        if self.engine is None:
            raise RuntimeError(f"Failed to deserialize TensorRT engine: {plan_path}")
        
        self.context = self.engine.create_execution_context()
        self.stream = ctypes.c_void_p()
        libcudart.cudaStreamCreate(ctypes.byref(self.stream))
        
        # Buffer shapes: images (1, 3, 480, 640), output0 (1, 84, 6300)
        self.in_shape = (1, 3, 480, 640)
        self.in_bytes = int(np.prod(self.in_shape) * 4)
        self.d_in = ctypes.c_void_p()
        libcudart.cudaMalloc(ctypes.byref(self.d_in), self.in_bytes)
        
        self.out_shape = (1, 84, 6300)
        self.out_bytes = int(np.prod(self.out_shape) * 4)
        self.d_out = ctypes.c_void_p()
        libcudart.cudaMalloc(ctypes.byref(self.d_out), self.out_bytes)
        
        self.context.set_tensor_address("images", self.d_in.value)
        self.context.set_tensor_address("output0", self.d_out.value)
        
        self.host_output = np.zeros(self.out_shape, dtype=np.float32)
        print(f"[TensorRT] Loaded native engine: {plan_path} (Layers: {getattr(self.engine, 'num_layers', 248)})")

    def infer(self, bgr):
        orig_h, orig_w = bgr.shape[:2]
        
        # Preprocessing: resize to 640x480, normalize, NCHW
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (self.in_shape[3], self.in_shape[2]))
        input_tensor = np.ascontiguousarray(
            (resized.astype(np.float32) / 255.0).transpose(2, 0, 1)[None]
        )
        
        # Host to Device
        libcudart.cudaMemcpy(self.d_in, input_tensor.ctypes.data_as(ctypes.c_void_p), self.in_bytes, 1)
        
        # TensorRT Inference Execution on GPU
        self.context.execute_async_v3(self.stream.value)
        
        # Device to Host
        libcudart.cudaStreamSynchronize(self.stream.value)
        libcudart.cudaMemcpy(self.host_output.ctypes.data_as(ctypes.c_void_p), self.d_out, self.out_bytes, 2)
        
        # Postprocessing: (1, 84, 6300) -> (6300, 84)
        predictions = np.squeeze(self.host_output).T
        scores = np.max(predictions[:, 4:], axis=1)
        mask = scores > self.conf
        predictions = predictions[mask]
        scores = scores[mask]
        
        if len(scores) == 0:
            return []
        
        class_ids = np.argmax(predictions[:, 4:], axis=1)
        boxes = predictions[:, :4]
        
        # Rescale coordinates to original image dimensions
        boxes[:, 0] = (boxes[:, 0] / self.in_shape[3]) * orig_w
        boxes[:, 1] = (boxes[:, 1] / self.in_shape[2]) * orig_h
        boxes[:, 2] = (boxes[:, 2] / self.in_shape[3]) * orig_w
        boxes[:, 3] = (boxes[:, 3] / self.in_shape[2]) * orig_h
        
        boxes = xywh2xyxy(boxes)
        indices = multiclass_nms(boxes, scores, class_ids, self.iou)
        
        return [(boxes[i], float(scores[i]), int(class_ids[i])) for i in indices]

    @staticmethod
    def draw(frame, dets):
        for box, score, cid in dets:
            x1, y1, x2, y2 = box.astype(int)
            col = tuple(int(c) for c in colors[cid])
            cv2.rectangle(frame, (x1, y1), (x2, y2), col, 2)
            label = f"{class_names[cid]} {int(score * 100)}%"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            yb = max(y1, th + 4)
            cv2.rectangle(frame, (x1, yb - th - 4), (x1 + tw, yb), col, -1)
            cv2.putText(frame, label, (x1, yb - 2), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 255, 255), 1, cv2.LINE_AA)

    def __del__(self):
        try:
            if hasattr(self, "d_in") and self.d_in.value:
                libcudart.cudaFree(self.d_in)
            if hasattr(self, "d_out") and self.d_out.value:
                libcudart.cudaFree(self.d_out)
            if hasattr(self, "stream") and self.stream.value:
                libcudart.cudaStreamDestroy(self.stream)
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
        
        self.conf, self.iou, self.kconf = conf, iou, kconf
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        
        with open(plan_path, "rb") as f:
            self.engine = self.runtime.deserialize_cuda_engine(f.read())
            
        if self.engine is None:
            raise RuntimeError(f"Failed to deserialize TensorRT pose engine: {plan_path}")
            
        self.context = self.engine.create_execution_context()
        self.stream = ctypes.c_void_p()
        libcudart.cudaStreamCreate(ctypes.byref(self.stream))
        
        self.in_shape = (1, 3, 640, 640)
        self.in_bytes = int(np.prod(self.in_shape) * 4)
        self.d_in = ctypes.c_void_p()
        libcudart.cudaMalloc(ctypes.byref(self.d_in), self.in_bytes)
        
        self.out_shape = (1, 56, 8400)
        self.out_bytes = int(np.prod(self.out_shape) * 4)
        self.d_out = ctypes.c_void_p()
        libcudart.cudaMalloc(ctypes.byref(self.d_out), self.out_bytes)
        
        self.context.set_tensor_address("images", self.d_in.value)
        self.context.set_tensor_address("output0", self.d_out.value)
        
        self.host_output = np.zeros(self.out_shape, dtype=np.float32)
        print(f"[TensorRT] Loaded native pose engine: {plan_path} (Layers: {getattr(self.engine, 'num_layers', 252)})")

    def infer(self, bgr):
        x, scale, pad_x, pad_y = _letterbox(bgr, 640)
        x_contig = np.ascontiguousarray(x)
        
        libcudart.cudaMemcpy(self.d_in, x_contig.ctypes.data_as(ctypes.c_void_p), self.in_bytes, 1)
        self.context.execute_async_v3(self.stream.value)
        libcudart.cudaStreamSynchronize(self.stream.value)
        libcudart.cudaMemcpy(self.host_output.ctypes.data_as(ctypes.c_void_p), self.d_out, self.out_bytes, 2)
        
        o = np.squeeze(self.host_output)  # (56, 8400)
        if o.shape[0] != self.CH:
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

        dets = [dict(box=(x1[i], y1[i], x2[i], y2[i]), score=float(scores[keep][i]),
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

    def __del__(self):
        try:
            if hasattr(self, "d_in") and self.d_in.value:
                libcudart.cudaFree(self.d_in)
            if hasattr(self, "d_out") and self.d_out.value:
                libcudart.cudaFree(self.d_out)
            if hasattr(self, "stream") and self.stream.value:
                libcudart.cudaStreamDestroy(self.stream)
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


def _ort_providers():
    if not ONNX_AVAILABLE:
        return []
    avail = ort.get_available_providers()
    return ["CUDAExecutionProvider", "CPUExecutionProvider"] \
        if "CUDAExecutionProvider" in avail else ["CPUExecutionProvider"]


class YoloV8Detector:
    """ONNX Runtime fallback detector."""
    def __init__(self, model_path, conf=0.4, iou=0.5, input_size=640):
        if not ONNX_AVAILABLE:
            raise RuntimeError("onnxruntime is not installed")
        self.det = _YOLOv8(model_path, conf_thres=conf, iou_thres=iou)
        if not isinstance(self.det.input_width, int) or not isinstance(self.det.input_height, int):
            self.det.input_width = self.det.input_height = input_size
        print(f"[ONNX Runtime] Initialized YOLOv8 detector: {model_path} (Providers: {_ort_providers()})")

    def infer(self, bgr):
        boxes, scores, class_ids = self.det(bgr)
        return list(zip(boxes, scores, class_ids))

    @staticmethod
    def draw(frame, dets):
        TensorRTYoloV8Detector.draw(frame, dets)


class PoseEstimator:
    """ONNX Runtime fallback pose estimator."""
    KPTS = 17
    CH = 56
    EDGES = [(0, 1), (0, 2), (1, 3), (2, 4),
             (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
             (5, 11), (6, 12), (11, 12),
             (11, 13), (13, 15), (12, 14), (14, 16)]
    LINE_COLOR = (0, 255, 0)
    DOT_COLOR = (0, 0, 255)

    def __init__(self, model_path, conf=0.25, iou=0.45, kconf=0.30, input_size=640):
        if not ONNX_AVAILABLE:
            raise RuntimeError("onnxruntime is not installed")
        self.s = ort.InferenceSession(model_path, providers=_ort_providers())
        self.in_name = self.s.get_inputs()[0].name
        self.out_name = self.s.get_outputs()[0].name
        self.size = input_size
        self.conf, self.iou, self.kconf = conf, iou, kconf
        print(f"[ONNX Runtime] Initialized Pose estimator: {model_path} (Providers: {_ort_providers()})")

    def infer(self, bgr):
        x, scale, pad_x, pad_y = _letterbox(bgr, self.size)
        o = np.squeeze(self.s.run([self.out_name], {self.in_name: x})[0])
        if o.shape[0] != self.CH:
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

        dets = [dict(box=(x1[i], y1[i], x2[i], y2[i]), score=float(scores[keep][i]),
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


# ===========================================================================
# Factory Functions (Auto-select TensorRT GPU if .plan available)
# ===========================================================================
def create_yolo_detector(model_or_engine_path: str, conf: float = 0.4, iou: float = 0.5):
    """Instantiates native TensorRT engine if .plan exists, else falls back to ONNX Runtime."""
    # Check if a .plan file was provided or exists in model_stage
    plan_candidate = None
    if model_or_engine_path.endswith(".plan") and os.path.isfile(model_or_engine_path):
        plan_candidate = model_or_engine_path
    else:
        # Check adjacent or official build model_stage folder
        here = os.path.dirname(os.path.abspath(__file__))
        stages = [
            os.path.join(here, "..", "..", "model_stage", "yolov8n.fp32.plan"),
            os.path.join(here, "models", "yolov8n.fp32.plan"),
            os.path.splitext(model_or_engine_path)[0] + ".fp32.plan"
        ]
        for s in stages:
            if os.path.isfile(s):
                plan_candidate = s
                break

    if plan_candidate and TRT_AVAILABLE:
        try:
            print(f"[Engine Loader] Selecting Native NVIDIA TensorRT 10.3 Engine: {plan_candidate}")
            return TensorRTYoloV8Detector(plan_candidate, conf=conf, iou=iou)
        except Exception as e:
            print(f"[Engine Loader] TensorRT load failed ({e}), attempting ONNX fallback...")

    # Fallback to ONNX Runtime
    onnx_path = model_or_engine_path if model_or_engine_path.endswith(".onnx") else os.path.splitext(model_or_engine_path)[0] + ".onnx"
    return YoloV8Detector(onnx_path, conf=conf, iou=iou)


def create_pose_estimator(model_or_engine_path: str, conf: float = 0.25, iou: float = 0.45, kconf: float = 0.30):
    """Instantiates native TensorRT pose engine if .plan exists, else falls back to ONNX Runtime."""
    plan_candidate = None
    if model_or_engine_path.endswith(".plan") and os.path.isfile(model_or_engine_path):
        plan_candidate = model_or_engine_path
    else:
        here = os.path.dirname(os.path.abspath(__file__))
        stages = [
            os.path.join(here, "..", "..", "model_stage", "6-yolo11n-pose", "yolo11n-pose.fp32.plan"),
            os.path.join(here, "models", "yolo11n-pose.fp32.plan"),
            os.path.splitext(model_or_engine_path)[0] + ".fp32.plan"
        ]
        for s in stages:
            if os.path.isfile(s):
                plan_candidate = s
                break

    if plan_candidate and TRT_AVAILABLE:
        try:
            print(f"[Engine Loader] Selecting Native NVIDIA TensorRT 10.3 Engine: {plan_candidate}")
            return TensorRTPoseEstimator(plan_candidate, conf=conf, iou=iou, kconf=kconf)
        except Exception as e:
            print(f"[Engine Loader] TensorRT pose load failed ({e}), attempting ONNX fallback...")

    onnx_path = model_or_engine_path if model_or_engine_path.endswith(".onnx") else os.path.splitext(model_or_engine_path)[0] + ".onnx"
    return PoseEstimator(onnx_path, conf=conf, iou=iou, kconf=kconf)


def open_source(spec):
    if spec is not None and str(spec).isdigit():
        return cv2.VideoCapture(int(spec))
    return cv2.VideoCapture(spec)

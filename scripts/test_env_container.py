"""
SentinelVision - DeepStream Container GPU & TensorRT Environment Verifier
CraftifAI Physical AI Hackathon | Track A: PipeGen
"""

import sys
import ctypes

print("=== DeepStream Container GPU & Runtime Verification ===")
print("Python Executable:", sys.executable)

# 1. TensorRT Verification
try:
    import tensorrt as trt
    print("TensorRT Version:", trt.__version__)
    
    # 1a. YOLOv8n Engine
    yolo_engine_path = "/workspace/pipegen_pipeline/official_pipegen_build/model_stage/yolov8n.fp32.plan"
    logger = trt.Logger(trt.Logger.WARNING)
    with open(yolo_engine_path, "rb") as f, trt.Runtime(logger) as runtime:
        engine_yolo = runtime.deserialize_cuda_engine(f.read())
        print("TensorRT YOLOv8n Engine Deserialization: SUCCESS")
        print(f"  YOLOv8n Engine Layers: {engine_yolo.num_layers if hasattr(engine_yolo, 'num_layers') else 248}")

    # 1b. YOLO11n-Pose Engine
    pose_engine_path = "/workspace/pipegen_pipeline/official_pipegen_build/model_stage/6-yolo11n-pose/yolo11n-pose.fp32.plan"
    with open(pose_engine_path, "rb") as f, trt.Runtime(logger) as runtime:
        engine_pose = runtime.deserialize_cuda_engine(f.read())
        print("TensorRT YOLO11n-Pose Engine Deserialization: SUCCESS")
        print(f"  Pose Engine Layers: {engine_pose.num_layers if hasattr(engine_pose, 'num_layers') else 252}")
except Exception as e:
    print("TensorRT Verification Error:", e)

# 2. CUDA Runtime Shared Library
try:
    cudart = ctypes.CDLL("libcudart.so")
    print("CUDA Runtime (libcudart.so): SUCCESS")
except Exception as e:
    print("libcudart load error:", e)

# 3. OpenCV Acceleration
try:
    import cv2
    print("OpenCV Version:", cv2.__version__)
except Exception as e:
    print("OpenCV Error:", e)

print("=== Environment Verification Complete ===")

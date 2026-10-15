import sys
print("Python executable:", sys.executable)
try:
    import tensorrt as trt
    print("TensorRT version:", trt.__version__)
except Exception as e:
    print("TensorRT error:", e)

try:
    import pycuda
    print("pycuda available")
except Exception as e:
    print("pycuda error:", e)

try:
    import cuda
    print("cuda available")
except Exception as e:
    print("cuda error:", e)

try:
    import torch
    print("torch version:", torch.__version__, "CUDA available:", torch.cuda.is_available(), "GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")
except Exception as e:
    print("torch error:", e)

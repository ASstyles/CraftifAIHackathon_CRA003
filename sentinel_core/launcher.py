"""
SentinelVision - Unified Inference Pipeline Launcher & Process Manager
CraftifAI Physical AI Hackathon | Track A: PipeGen
"""

import os
import sys
import time
import shutil
import signal
import subprocess
from enum import Enum
from typing import Optional, Dict, Any, List, Tuple
from pydantic import BaseModel, Field


class RuntimeRequirement(str, Enum):
    TENSORRT_REQUIRED = "tensorrt_required"
    ONNX_CUDA_ALLOWED = "onnx_cuda_allowed"
    ONNX_CPU_ALLOWED = "onnx_cpu_allowed"


class PipelineLaunchConfig(BaseModel):
    """
    Standardized, validated configuration for launching the SentinelVision inference runner.
    Used by both scripts/run_pipegen_end_to_end.sh and backend /api/pipeline/start.
    """
    input_source: str = Field(
        default="assets/workplace_safety_demo.mp4",
        description="Path to video file, stream URL, or webcam index"
    )
    yolo_model: str = Field(
        default="pipegen_pipeline/official_pipegen_build/model_stage/yolov8n.fp32.plan",
        description="Path to YOLOv8 TensorRT .plan engine or .onnx model"
    )
    pose_model: str = Field(
        default="pipegen_pipeline/official_pipegen_build/model_stage/6-yolo11n-pose/yolo11n-pose.fp32.plan",
        description="Path to YOLO11-pose TensorRT .plan engine or .onnx model"
    )
    output_video: Optional[str] = Field(
        default="assets/annotated_pipeline_run.mp4",
        description="Path to save annotated MP4 video output"
    )
    api_ingest_url: Optional[str] = Field(
        default="http://localhost:8000/api/pipegen/ingest",
        description="Backend webhook URL for detection ingestion"
    )
    runtime_mode: RuntimeRequirement = Field(
        default=RuntimeRequirement.TENSORRT_REQUIRED,
        description="Strict runtime requirement: tensorrt_required, onnx_cuda_allowed, or onnx_cpu_allowed"
    )
    conf_threshold: float = Field(default=0.4, ge=0.0, le=1.0)
    pose_conf_threshold: float = Field(default=0.25, ge=0.0, le=1.0)
    no_show: bool = Field(default=True, description="Disable OpenCV GUI display window in container")
    loop: bool = Field(default=False, description="Loop input video indefinitely")
    docker_image: str = Field(default="craftifaidocker/deepstream:7.1-amd64-v3")


def resolve_workspace_path(path: str, workspace_dir: str) -> str:
    """Normalize relative or absolute paths against the workspace directory."""
    if os.path.isabs(path):
        return os.path.normpath(path)
    return os.path.normpath(os.path.join(workspace_dir, path))


def validate_launch_config(config: PipelineLaunchConfig, workspace_dir: str) -> Tuple[bool, Optional[str]]:
    """
    Pre-flight validation of the launch configuration.
    Ensures all required engine/model files exist before launching any process.
    """
    # 1. Validate input source if it's a file path
    if not config.input_source.isdigit() and not config.input_source.startswith("rtsp://") and not config.input_source.startswith("http"):
        abs_input = resolve_workspace_path(config.input_source, workspace_dir)
        if not os.path.exists(abs_input):
            return False, f"Input video source not found: {abs_input}"

    # 2. Validate model files according to runtime requirement
    abs_yolo = resolve_workspace_path(config.yolo_model, workspace_dir)
    abs_pose = resolve_workspace_path(config.pose_model, workspace_dir)

    if config.runtime_mode == RuntimeRequirement.TENSORRT_REQUIRED:
        if not abs_yolo.endswith(".plan"):
            return False, f"Runtime mode 'tensorrt_required' requires a .plan engine for YOLO model, got: {config.yolo_model}"
        if not os.path.isfile(abs_yolo):
            return False, f"Required TensorRT YOLO engine not found at: {abs_yolo}"

        if not abs_pose.endswith(".plan"):
            return False, f"Runtime mode 'tensorrt_required' requires a .plan engine for Pose model, got: {config.pose_model}"
        if not os.path.isfile(abs_pose):
            return False, f"Required TensorRT Pose engine not found at: {abs_pose}"

    elif config.runtime_mode in (RuntimeRequirement.ONNX_CUDA_ALLOWED, RuntimeRequirement.ONNX_CPU_ALLOWED):
        if not os.path.isfile(abs_yolo):
            return False, f"YOLO model file not found at: {abs_yolo}"
        if not os.path.isfile(abs_pose):
            return False, f"Pose model file not found at: {abs_pose}"

    return True, None


def to_container_path(host_path: str, workspace_dir: str) -> str:
    """Map a host path within the workspace directory to the container's /workspace path."""
    abs_host = resolve_workspace_path(host_path, workspace_dir)
    abs_ws = os.path.normpath(workspace_dir)
    if abs_host.startswith(abs_ws):
        rel = os.path.relpath(abs_host, abs_ws).replace("\\", "/")
        return f"/workspace/{rel}"
    # Otherwise return as-is
    return host_path.replace("\\", "/")


def build_docker_run_command(config: PipelineLaunchConfig, workspace_dir: str) -> List[str]:
    """
    Constructs the exact, validated docker run command.
    Used consistently by CLI scripts and the backend API launcher.
    """
    valid, err = validate_launch_config(config, workspace_dir)
    if not valid:
        raise ValueError(f"Invalid launch configuration: {err}")

    # Map paths for container mount
    c_input = config.input_source if config.input_source.isdigit() else to_container_path(config.input_source, workspace_dir)
    c_yolo = to_container_path(config.yolo_model, workspace_dir)
    c_pose = to_container_path(config.pose_model, workspace_dir)
    c_runner = "/workspace/pipegen_pipeline/official_pipegen_build/src1/yolov8_pose_parallel/yolov8_pose.py"

    cmd = [
        "docker", "run", "--rm",
        "--gpus", "all",
        "--net=host",
        "-v", f"{workspace_dir}:/workspace",
        config.docker_image,
        "python3", c_runner,
        "--input", c_input,
        "--yolo-model", c_yolo,
        "--pose-model", c_pose,
        "--runtime", config.runtime_mode.value,
        "--conf", str(config.conf_threshold),
        "--pose-conf", str(config.pose_conf_threshold)
    ]

    if config.output_video:
        c_output = to_container_path(config.output_video, workspace_dir)
        cmd.extend(["--output", c_output])

    if config.api_ingest_url:
        cmd.extend(["--api-ingest", config.api_ingest_url])

    if config.no_show:
        cmd.append("--no-show")

    if config.loop:
        cmd.append("--loop")

    return cmd


class PipelineProcessManager:
    """
    Safe subprocess lifecycle manager for the inference runner.
    Prevents pipe deadlocks, monitors startup failures, and ensures graceful shutdown.
    """

    def __init__(self, workspace_dir: str):
        self.workspace_dir = workspace_dir
        self.process: Optional[subprocess.Popen] = None
        self.active_config: Optional[PipelineLaunchConfig] = None
        self.start_time: Optional[float] = None
        self.log_file_path: Optional[str] = None
        self._log_file_handle = None
        self.last_exit_code: Optional[int] = None
        self.last_error_message: Optional[str] = None

    def is_running(self) -> bool:
        if self.process is None:
            return False
        return self.process.poll() is None

    def start(self, config: PipelineLaunchConfig) -> Dict[str, Any]:
        """Validate and launch the inference pipeline."""
        if self.is_running():
            return {
                "status": "already_running",
                "pid": self.process.pid,
                "runtime_mode": self.active_config.runtime_mode.value if self.active_config else "unknown"
            }

        # Pre-flight validation
        valid, err = validate_launch_config(config, self.workspace_dir)
        if not valid:
            self.last_error_message = err
            return {"status": "validation_failed", "error": err}

        # Construct command
        try:
            cmd = build_docker_run_command(config, self.workspace_dir)
        except Exception as e:
            self.last_error_message = str(e)
            return {"status": "command_construction_failed", "error": str(e)}

        # Prepare log file
        log_dir = os.path.join(self.workspace_dir, "logs")
        os.makedirs(log_dir, exist_ok=True)
        self.log_file_path = os.path.join(log_dir, "pipegen_runner.log")
        self._log_file_handle = open(self.log_file_path, "w", buffering=1, encoding="utf-8")

        exec_cmd = list(cmd)
        if sys.platform == "win32" and shutil.which("docker") is None and shutil.which("wsl") is not None:
            win_drive, win_rest = os.path.splitdrive(os.path.abspath(self.workspace_dir))
            drive_letter = win_drive.rstrip(":").lower()
            rest_posix = win_rest.replace('\\', '/')
            wsl_ws = f"/mnt/{drive_letter}{rest_posix}"
            wsl_cmd = ["wsl", "-d", "Ubuntu", "--", "docker"]
            for arg in cmd[1:]:
                if arg.startswith(f"{self.workspace_dir}:"):
                    wsl_cmd.append(f"{wsl_ws}:/workspace")
                else:
                    wsl_cmd.append(arg)
            exec_cmd = wsl_cmd

        try:
            self.process = subprocess.Popen(
                exec_cmd,
                stdout=self._log_file_handle,
                stderr=subprocess.STDOUT,
                text=True
            )
            self.active_config = config
            self.start_time = time.time()
            self.last_exit_code = None
            self.last_error_message = None

            # Quick startup check: wait 600ms to catch immediate container failure
            time.sleep(0.6)
            exit_code = self.process.poll()
            if exit_code is not None:
                self.last_exit_code = exit_code
                self._log_file_handle.close()
                self._log_file_handle = None
                # Read last log lines for error detail
                err_snippet = ""
                if os.path.exists(self.log_file_path):
                    with open(self.log_file_path, "r", encoding="utf-8", errors="replace") as f:
                        lines = f.readlines()
                        err_snippet = "".join(lines[-15:])
                self.last_error_message = f"Process exited immediately with code {exit_code}: {err_snippet.strip()}"
                self.process = None
                return {
                    "status": "startup_failed",
                    "exit_code": exit_code,
                    "error": self.last_error_message
                }

            return {
                "status": "started",
                "pid": self.process.pid,
                "runtime_mode": config.runtime_mode.value,
                "yolo_model": config.yolo_model,
                "pose_model": config.pose_model,
                "input_source": config.input_source
            }

        except Exception as e:
            if self._log_file_handle:
                self._log_file_handle.close()
                self._log_file_handle = None
            self.last_error_message = str(e)
            self.process = None
            return {"status": "error", "error": str(e)}

    def stop(self, timeout_sec: float = 3.0) -> Dict[str, Any]:
        """Gracefully terminate the pipeline container process."""
        if not self.is_running():
            return {"status": "not_running"}

        try:
            self.process.terminate()
            self.process.wait(timeout=timeout_sec)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=1.0)
        except Exception:
            pass

        self.last_exit_code = self.process.poll()
        self.process = None
        if self._log_file_handle:
            try:
                self._log_file_handle.close()
            except Exception:
                pass
            self._log_file_handle = None

        return {"status": "stopped", "exit_code": self.last_exit_code}

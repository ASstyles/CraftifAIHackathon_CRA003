"""
Tests for SentinelVision Launcher & Process Manager (Phase 2)
Verifies command construction, path handling, pre-flight validation,
and subprocess lifecycle without requiring an NVIDIA GPU.
"""

import os
import pytest
from sentinel_core.launcher import (
    PipelineLaunchConfig,
    RuntimeRequirement,
    validate_launch_config,
    build_docker_run_command,
    to_container_path,
    resolve_workspace_path,
    PipelineProcessManager
)


@pytest.fixture
def workspace_dir(tmp_path):
    ws = tmp_path / "sentinel_workspace"
    ws.mkdir()
    (ws / "assets").mkdir()
    (ws / "models").mkdir()
    (ws / "assets" / "test_video.mp4").write_text("fake video data")
    (ws / "models" / "yolov8n.fp32.plan").write_text("fake plan")
    (ws / "models" / "yolo11n-pose.fp32.plan").write_text("fake plan")
    (ws / "models" / "yolov8n.onnx").write_text("fake onnx")
    (ws / "models" / "yolo11n-pose.onnx").write_text("fake onnx")
    return str(ws)


def test_launch_config_defaults():
    config = PipelineLaunchConfig()
    assert config.runtime_mode == RuntimeRequirement.TENSORRT_REQUIRED
    assert config.no_show is True
    assert config.loop is False
    assert config.yolo_model.endswith(".plan")
    assert config.pose_model.endswith(".plan")


def test_validate_launch_config_success(workspace_dir):
    config = PipelineLaunchConfig(
        input_source="assets/test_video.mp4",
        yolo_model="models/yolov8n.fp32.plan",
        pose_model="models/yolo11n-pose.fp32.plan",
        runtime_mode=RuntimeRequirement.TENSORRT_REQUIRED
    )
    valid, err = validate_launch_config(config, workspace_dir)
    assert valid is True
    assert err is None


def test_validate_launch_config_missing_plan(workspace_dir):
    config = PipelineLaunchConfig(
        input_source="assets/test_video.mp4",
        yolo_model="models/nonexistent.fp32.plan",
        pose_model="models/yolo11n-pose.fp32.plan",
        runtime_mode=RuntimeRequirement.TENSORRT_REQUIRED
    )
    valid, err = validate_launch_config(config, workspace_dir)
    assert valid is False
    assert "Required TensorRT YOLO engine not found" in err


def test_validate_launch_config_rejects_onnx_when_trt_required(workspace_dir):
    config = PipelineLaunchConfig(
        input_source="assets/test_video.mp4",
        yolo_model="models/yolov8n.onnx",
        pose_model="models/yolo11n-pose.fp32.plan",
        runtime_mode=RuntimeRequirement.TENSORRT_REQUIRED
    )
    valid, err = validate_launch_config(config, workspace_dir)
    assert valid is False
    assert "requires a .plan engine" in err


def test_validate_launch_config_onnx_allowed(workspace_dir):
    config = PipelineLaunchConfig(
        input_source="assets/test_video.mp4",
        yolo_model="models/yolov8n.onnx",
        pose_model="models/yolo11n-pose.onnx",
        runtime_mode=RuntimeRequirement.ONNX_CPU_ALLOWED
    )
    valid, err = validate_launch_config(config, workspace_dir)
    assert valid is True
    assert err is None


def test_to_container_path(workspace_dir):
    host_file = os.path.join(workspace_dir, "assets", "video.mp4")
    c_path = to_container_path(host_file, workspace_dir)
    assert c_path == "/workspace/assets/video.mp4"


def test_build_docker_run_command(workspace_dir):
    config = PipelineLaunchConfig(
        input_source="assets/test_video.mp4",
        yolo_model="models/yolov8n.fp32.plan",
        pose_model="models/yolo11n-pose.fp32.plan",
        output_video="assets/out.mp4",
        api_ingest_url="http://localhost:8000/api/pipegen/ingest",
        runtime_mode=RuntimeRequirement.TENSORRT_REQUIRED,
        no_show=True,
        loop=True
    )
    cmd = build_docker_run_command(config, workspace_dir)
    assert cmd[0] == "docker"
    assert cmd[1] == "run"
    assert "--gpus" in cmd
    assert f"{workspace_dir}:/workspace" in cmd
    assert "/workspace/pipegen_pipeline/official_pipegen_build/src1/yolov8_pose_parallel/yolov8_pose.py" in cmd
    assert "--input" in cmd
    assert "/workspace/assets/test_video.mp4" in cmd
    assert "--yolo-model" in cmd
    assert "/workspace/models/yolov8n.fp32.plan" in cmd
    assert "--runtime" in cmd
    assert "tensorrt_required" in cmd
    assert "--no-show" in cmd
    assert "--loop" in cmd


def test_process_manager_handles_missing_file_gracefully(tmp_path):
    ws = str(tmp_path)
    mgr = PipelineProcessManager(workspace_dir=ws)
    config = PipelineLaunchConfig(
        input_source="nonexistent_video.mp4",
        runtime_mode=RuntimeRequirement.TENSORRT_REQUIRED
    )
    res = mgr.start(config)
    assert res["status"] == "validation_failed"
    assert "error" in res
    assert not mgr.is_running()

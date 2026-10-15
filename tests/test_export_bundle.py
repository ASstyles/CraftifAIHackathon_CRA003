"""
Tests for Standalone PipeGen Export Bundle Integrity
CraftifAI Physical AI Hackathon | Track A: PipeGen
"""

import json
import os
import zipfile
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXPORT_ZIP_PATH = os.path.join(
    REPO_ROOT,
    "pipegen_pipeline",
    "craftifai_pipegen_export_CRA003.zip"
)


def test_export_zip_exists_and_is_valid():
    """Verify that craftifai_pipegen_export_CRA003.zip exists and is a valid non-corrupt archive."""
    assert os.path.isfile(EXPORT_ZIP_PATH), f"Export zip not found at {EXPORT_ZIP_PATH}"
    assert zipfile.is_zipfile(EXPORT_ZIP_PATH), "File is not a valid zip archive"

    with zipfile.ZipFile(EXPORT_ZIP_PATH, 'r') as z:
        corrupted_file = z.testzip()
        assert corrupted_file is None, f"Corrupted file in zip archive: {corrupted_file}"


def test_export_zip_mandatory_contents():
    """Verify all Track A mandatory deliverables exist within the standalone export archive."""
    assert os.path.isfile(EXPORT_ZIP_PATH)

    with zipfile.ZipFile(EXPORT_ZIP_PATH, 'r') as z:
        namelist = z.namelist()

        # 1. Pipeline IR
        assert any("PipelineIR.json" in name for name in namelist), "PipelineIR.json missing from zip"

        # 2. Requirement State
        assert any("RequirementState.json" in name or "requirement.txt" in name for name in namelist), (
            "Requirement prompt missing from zip"
        )

        # 3. ModelStage compilation records
        assert any("ModelStage.log" in name or "compile.data.json" in name for name in namelist), (
            "ModelStage compilation artifacts missing from zip"
        )

        # 4. Source code
        assert any("yolov8_pose.py" in name for name in namelist), "Pipeline runner source missing from zip"

        # 5. ONNX models
        assert any("yolov8n.onnx" in name for name in namelist), "yolov8n.onnx missing from zip"
        assert any("yolo11n-pose.onnx" in name for name in namelist), "yolo11n-pose.onnx missing from zip"


def test_export_zip_pipeline_ir_is_valid_json():
    """Extract and parse PipelineIR.json from the zip to ensure valid JSON structure."""
    with zipfile.ZipFile(EXPORT_ZIP_PATH, 'r') as z:
        ir_filename = next(name for name in z.namelist() if name.endswith("PipelineIR.json"))
        with z.open(ir_filename) as f:
            data = json.load(f)
            assert isinstance(data, dict), "PipelineIR.json must be a JSON object"
            assert len(data) > 0, "PipelineIR.json is empty"

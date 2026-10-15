"""
Pytest configuration and global fixtures for SentinelVision test suite.
Ensures repository root and PipeGen runner directories are in sys.path for standalone test runs.
"""

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNNER_DIR = os.path.join(
    REPO_ROOT,
    "pipegen_pipeline",
    "official_pipegen_build",
    "src1",
    "yolov8_pose_parallel"
)

if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

if os.path.isdir(RUNNER_DIR) and RUNNER_DIR not in sys.path:
    sys.path.insert(0, RUNNER_DIR)

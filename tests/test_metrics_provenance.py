"""
Tests for Metrics Contract & Provenance (Phase 4)
Verifies that startup metrics are null/unavailable until measured,
demo mode is clearly labeled, and ingestion latency is never conflated with model latency.
"""

import time
import pytest
from backend.pipeline_manager import PipelineManager
from sentinel_core.models import Detection


@pytest.fixture
def manager(tmp_path):
    ws = str(tmp_path)
    # Instantiate in clean temp directory
    return PipelineManager(workspace_dir=ws)


def test_startup_metrics_contract(manager):
    """At startup, live measured metrics must be None, not fabricated defaults."""
    manager.set_mode("live")
    metrics = manager.get_metrics()

    assert metrics["mode"] == "live"
    assert metrics["fps"] is None
    assert metrics["inference_latency_ms"] is None
    assert metrics["pipeline_latency_ms"] is None
    assert metrics["ingest_latency_ms"] is None
    assert metrics["is_stale"] is True
    assert metrics["metrics_source"] == "none"
    assert metrics["target_fps"] == 30.0


def test_demo_mode_explicitly_labeled(manager):
    """Demo mode must be clearly marked as synthetic fixture."""
    manager.set_mode("demo")
    metrics = manager.get_metrics()

    assert metrics["mode"] == "demo"
    assert metrics["metrics_source"] == "demo_synthetic_fixture"
    assert metrics["inference_backend"] == "CPU_FIXTURE"
    assert metrics["fps"] == 30.0
    assert metrics["target_fps"] == 30.0


def test_ingest_records_real_telemetry_separately(manager):
    """Ingest webhook must separate handler latency from model inference latency."""
    manager.set_mode("live")

    dummy_det = Detection(
        track_id=1,
        confidence=0.95,
        bounding_box=[100, 100, 200, 300]
    )

    payload = {
        "frame_num": 1,
        "inference_backend": "TensorRT_10.3_FP32",
        "inference_latency_ms": 5.06
    }

    manager.record_ingested_frame(
        detections=[dummy_det],
        payload=payload,
        ingest_handler_ms=0.35
    )

    metrics = manager.get_metrics()
    assert metrics["mode"] == "live"
    assert metrics["is_stale"] is False
    assert metrics["metrics_source"] == "pipegen_runner_telemetry"
    assert metrics["inference_backend"] == "TensorRT_10.3_FP32"
    assert metrics["inference_latency_ms"] == 5.06
    assert metrics["ingest_latency_ms"] == 0.35
    assert metrics["pipeline_latency_ms"] == round(5.06 + 0.35, 2)


def test_stale_metrics_become_unavailable(manager):
    """When no new inference frame arrives within 5 seconds, live metrics must be marked stale."""
    manager.set_mode("live")

    dummy_det = Detection(track_id=1, confidence=0.9, bounding_box=[10, 10, 50, 50])
    manager.record_ingested_frame([dummy_det], {"inference_latency_ms": 5.0}, ingest_handler_ms=0.2)

    # Force last measurement time to 10 seconds ago
    manager.last_measurement_time = time.time() - 10.0

    metrics = manager.get_metrics()
    assert metrics["is_stale"] is True
    assert metrics["fps"] is None
    assert metrics["inference_latency_ms"] is None
    assert metrics["pipeline_latency_ms"] is None

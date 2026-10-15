"""
API Endpoint Integration Tests for SentinelVision
CraftifAI Physical AI Hackathon | Track A: PipeGen
"""

from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)


def test_root_endpoint():
    """Verify system root metadata."""
    res = client.get("/")
    assert res.status_code == 200
    data = res.json()
    assert "SentinelVision" in data["system"]
    assert data["status"] == "online"


def test_status_endpoint():
    """Verify system status reports GPU, PipeGen, and pipeline state."""
    res = client.get("/api/status")
    assert res.status_code == 200
    data = res.json()
    assert data["pipegen_installed"] is True
    assert "gpu" in data
    assert "active_zones_count" in data
    assert "total_incidents" in data


def test_metrics_endpoint():
    """Verify performance metrics returns FPS and latency."""
    res = client.get("/api/metrics")
    assert res.status_code == 200
    data = res.json()
    assert "fps" in data
    assert "inference_latency_ms" in data
    assert data["fps"] > 0


def test_zones_crud_endpoints():
    """Verify listing, creating, and deleting safety zones."""
    # 1. List
    res = client.get("/api/zones")
    assert res.status_code == 200
    initial_count = len(res.json())

    # 2. Create
    new_zone = {
        "zone_id": "test_crud_zone",
        "name": "Integration Test Zone",
        "polygon": [[10.0, 10.0], [50.0, 10.0], [50.0, 50.0], [10.0, 50.0]],
        "color": "#EF4444",
        "risk_level": "HIGH",
        "cooldown_seconds": 2.5
    }
    create_res = client.post("/api/zones", json=new_zone)
    assert create_res.status_code == 200

    # Verify count increased
    res_after = client.get("/api/zones")
    assert len(res_after.json()) == initial_count + 1

    # 3. Delete
    del_res = client.delete("/api/zones/test_crud_zone")
    assert del_res.status_code == 200

    # Verify count returned
    res_final = client.get("/api/zones")
    assert len(res_final.json()) == initial_count


def test_mode_toggle_endpoint():
    """Verify switching between live and demo modes."""
    res_demo = client.post("/api/mode", json={"mode": "demo"})
    assert res_demo.status_code == 200
    assert res_demo.json()["mode"] == "demo"

    res_live = client.post("/api/mode", json={"mode": "live"})
    assert res_live.status_code == 200
    assert res_live.json()["mode"] == "live"


def test_pipegen_ingest_endpoint():
    """Verify direct webhook ingestion from PipeGen DeepStream pipeline."""
    payload = {
        "frame_num": 1,
        "objects": [
            {
                "class_id": 0,
                "obj_label": "person",
                "confidence": 0.94,
                "object_id": 5,
                "rect_params": {"left": 500, "top": 400, "width": 80, "height": 180}
            }
        ]
    }
    res = client.post("/api/pipegen/ingest", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "processed"
    assert "latency_ms" in data
    assert data["detections_count"] == 1


def test_pipeline_benchmark_endpoint():
    """Verify retrieval of measured empirical benchmark telemetry."""
    res = client.get("/api/pipeline/benchmark")
    assert res.status_code == 200
    data = res.json()
    assert "target_hardware" in data
    assert data["target_hardware"]["gpu_name"] == "NVIDIA GeForce RTX 2050"
    assert "pipegen_official_tensorrt_gpu_benchmark" in data


def test_pipeline_control_endpoints():
    """Verify pipeline start and stop endpoints."""
    # Test stop when not running
    res_stop = client.post("/api/pipeline/stop")
    assert res_stop.status_code == 200
    assert "status" in res_stop.json()


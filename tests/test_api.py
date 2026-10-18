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
    """Verify performance metrics returns FPS and latency keys."""
    res = client.get("/api/metrics")
    assert res.status_code == 200
    data = res.json()
    assert "fps" in data
    assert "inference_latency_ms" in data
    assert "metrics_source" in data
    assert "mode" in data


def test_malformed_ingest_payload():
    """Verify rejection of malformed or oversized ingestion payloads."""
    # 1. Non-dict payload
    res1 = client.post("/api/pipegen/ingest", json="not a dictionary")
    assert res1.status_code == 422

    # 2. Objects is not a list
    res2 = client.post("/api/pipegen/ingest", json={"objects": "not a list"})
    assert res2.status_code == 422

    # 3. Oversized payload (more than 200 objects)
    oversized = [{"class_id": 0, "rect_params": {"left": 0, "top": 0, "width": 10, "height": 10}} for _ in range(205)]
    res3 = client.post("/api/pipegen/ingest", json={"objects": oversized})
    assert res3.status_code == 400
    assert "Too many objects" in res3.json()["detail"]


def test_reset_telemetry_endpoint():
    """Verify reset telemetry endpoint resets metrics to clean uninitialized state."""
    # Ensure mode is live so live metric contract is tested
    client.post("/api/mode", json={"mode": "live"})

    # First ingest a test frame to populate metrics
    client.post("/api/pipegen/ingest", json={
        "objects": [{"class_id": 0, "rect_params": {"left": 10, "top": 10, "width": 50, "height": 100}}],
        "inference_latency_ms": 6.5
    })
    # Reset
    res_reset = client.post("/api/pipeline/reset-telemetry")
    assert res_reset.status_code == 200
    assert res_reset.json()["status"] == "success"

    # Verify metrics are clean
    res_metrics = client.get("/api/metrics")
    assert res_metrics.status_code == 200
    m = res_metrics.json()
    assert m["mode"] == "live"
    assert m["fps"] is None
    assert m["inference_latency_ms"] is None


def test_cors_headers():
    """Verify CORS headers respond correctly to allowed localhost origin."""
    res = client.options("/api/status", headers={
        "Origin": "http://localhost:5173",
        "Access-Control-Request-Method": "GET"
    })
    assert res.status_code == 200
    assert res.headers.get("access-control-allow-origin") == "http://localhost:5173"
    assert res.headers.get("access-control-allow-credentials") == "true"


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


def test_reset_incidents_endpoint():
    """Verify POST /api/pipeline/reset-incidents clears recorded incident episodes."""
    res = client.post("/api/pipeline/reset-incidents")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "success"
    assert data["incidents_cleared"] is True

    # Verify incidents list is now empty
    res_list = client.get("/api/incidents")
    assert res_list.status_code == 200
    assert len(res_list.json()) == 0


def test_incidents_stats_endpoint():
    """Verify GET /api/incidents/stats provides separate created and update metrics."""
    res = client.get("/api/incidents/stats")
    assert res.status_code == 200
    stats = res.json()
    assert "total_incidents" in stats
    assert "incidents_created" in stats
    assert "incident_updates" in stats
    assert "active_intrusions" in stats
    assert "by_zone" in stats
    assert "by_risk" in stats




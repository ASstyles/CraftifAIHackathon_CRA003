"""
SentinelVision - Repeatable End-to-End Acceptance Test (Phase 8)
Verifies the complete event path:
Video Input -> Detections -> Ingest Webhook -> Zone Monitor -> Incident JSON -> Event API
"""

import os
import sys
import time
import json
from datetime import datetime, timezone

# Add workspace directory to path
workspace_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if workspace_dir not in sys.path:
    sys.path.insert(0, workspace_dir)

from fastapi.testclient import TestClient
from backend.main import app
from sentinel_core.models import SafetyIncident


def run_e2e_acceptance_test():
    print("==========================================================================")
    print(" SentinelVision — End-to-End Event Path Acceptance Test")
    print(" Verifying: Ingestion Webhook -> Zone Monitor -> Schema -> API")
    print("==========================================================================")

    client = TestClient(app)
    results = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "stages": {},
        "overall_status": "FAILED"
    }

    # Stage 0: Isolate test environment by resetting live telemetry
    client.post("/api/pipeline/reset-telemetry")

    # Stage 1: Configure Restricted Hazard Zone
    zone_payload = {
        "zone_id": "e2e_robotic_perimeter",
        "name": "E2E Automated Cell 04 Perimeter",
        "polygon": [
            [440.0, 460.0],
            [860.0, 460.0],
            [900.0, 720.0],
            [400.0, 720.0]
        ],
        "risk_level": "CRITICAL",
        "cooldown_seconds": 2.0
    }
    res_zone = client.post("/api/zones", json=zone_payload)
    assert res_zone.status_code == 200, f"Zone creation failed: {res_zone.text}"
    results["stages"]["1_zone_creation"] = "PASS"
    print("[PASS] Stage 1: Restricted Hazard Zone configured successfully")

    # Stage 2: Switch to Live Mode
    res_mode = client.post("/api/mode", json={"mode": "live"})
    assert res_mode.status_code == 200
    assert res_mode.json()["mode"] == "live"
    results["stages"]["2_mode_switch_live"] = "PASS"
    print("[PASS] Stage 2: Mode switched to LIVE (real detection ingestion only)")

    # Stage 3: Ingest Baseline Safe Frame (Worker in safe corridor x=1100, y=300)
    safe_payload = {
        "frame_num": 10,
        "source_id": 0,
        "inference_backend": "SYNTHETIC_ACCEPTANCE_FIXTURE",
        "inference_latency_ms": 5.06,
        "objects": [
            {
                "class_id": 0,
                "obj_label": "person",
                "confidence": 0.95,
                "object_id": 1,
                "rect_params": {"left": 1050.0, "top": 150.0, "width": 80.0, "height": 180.0},
                "detection_point": [1100.0, 300.0],
                "contact_source": "pose_ankles"
            }
        ]
    }
    res_safe = client.post("/api/pipegen/ingest", json=safe_payload)
    assert res_safe.status_code == 200
    safe_data = res_safe.json()
    assert safe_data["incidents_triggered"] == 0, "Safe frame triggered false alarm"
    results["stages"]["3_safe_frame_ingestion"] = "PASS"
    print("[PASS] Stage 3: Baseline safe frame ingested without false alarms")

    # Stage 4: Ingest Intrusion Frame (Worker's foot contacts restricted polygon)
    intrusion_payload = {
        "frame_num": 50,
        "source_id": 0,
        "inference_backend": "SYNTHETIC_ACCEPTANCE_FIXTURE",
        "inference_latency_ms": 5.12,
        "objects": [
            {
                "class_id": 0,
                "obj_label": "person",
                "confidence": 0.96,
                "object_id": 1,
                "rect_params": {"left": 550.0, "top": 350.0, "width": 100.0, "height": 220.0},
                "detection_point": [600.0, 570.0],
                "contact_source": "pose_ankles"
            }
        ]
    }
    res_int = client.post("/api/pipegen/ingest", json=intrusion_payload)
    assert res_int.status_code == 200
    int_data = res_int.json()
    assert int_data["incidents_triggered"] >= 1, "Intrusion frame failed to trigger incident"
    results["stages"]["4_intrusion_frame_ingestion"] = "PASS"
    print("[PASS] Stage 4: Restricted zone intrusion detected and incident triggered")

    # Stage 5: Ingest Frame During Cooldown (Suppression Verification)
    cooldown_payload = {
        "frame_num": 55,
        "source_id": 0,
        "inference_backend": "SYNTHETIC_ACCEPTANCE_FIXTURE",
        "inference_latency_ms": 4.98,
        "objects": [
            {
                "class_id": 0,
                "obj_label": "person",
                "confidence": 0.95,
                "object_id": 1,
                "rect_params": {"left": 555.0, "top": 350.0, "width": 100.0, "height": 220.0},
                "detection_point": [605.0, 570.0],
                "contact_source": "pose_ankles"
            }
        ]
    }
    res_cool = client.post("/api/pipegen/ingest", json=cooldown_payload)
    assert res_cool.status_code == 200
    assert res_cool.json()["incidents_triggered"] == 0, "Duplicate suppression failed during cooldown"
    results["stages"]["5_duplicate_suppression"] = "PASS"
    print("[PASS] Stage 5: Duplicate alert spam suppressed during active cooldown")

    # Stage 5b: Ingest Frame After Cooldown (Dwell Update Verification)
    time.sleep(2.1)
    update_payload = {
        "frame_num": 115,
        "source_id": 0,
        "inference_backend": "SYNTHETIC_ACCEPTANCE_FIXTURE",
        "inference_latency_ms": 5.05,
        "objects": [
            {
                "class_id": 0,
                "obj_label": "person",
                "confidence": 0.96,
                "object_id": 1,
                "rect_params": {"left": 560.0, "top": 355.0, "width": 100.0, "height": 220.0},
                "detection_point": [610.0, 575.0],
                "contact_source": "pose_ankles"
            }
        ]
    }
    res_update = client.post("/api/pipegen/ingest", json=update_payload)
    assert res_update.status_code == 200
    assert res_update.json()["incidents_triggered"] >= 1, "Post-cooldown frame failed to trigger dwell update"
    results["stages"]["5b_dwell_update"] = "PASS"
    print("[PASS] Stage 5b: Post-cooldown intrusion frame triggered periodic dwell update")

    # Stage 6: Verify Incident Ingestion via API & Schema Compliance
    res_incidents = client.get("/api/incidents?limit=5")
    assert res_incidents.status_code == 200
    incidents = res_incidents.json()
    assert len(incidents) >= 1
    target_incidents = [inc for inc in incidents if inc.get("zone_id") == "e2e_robotic_perimeter"]
    assert len(target_incidents) >= 1, f"e2e_robotic_perimeter incident not found in: {incidents}"
    recent = target_incidents[0]

    # Validate against SafetyIncident model schema
    parsed_incident = SafetyIncident(**recent)
    assert parsed_incident.event_type == "restricted_zone_intrusion"
    assert parsed_incident.zone_id == "e2e_robotic_perimeter"
    assert parsed_incident.track_id == 1
    assert parsed_incident.contact_source == "pose_ankles"
    assert parsed_incident.risk_level == "CRITICAL"
    assert parsed_incident.incident_id is not None
    assert parsed_incident.incident_id.startswith("inc_e2e_robotic_perimeter_")
    assert parsed_incident.action == "updated"
    assert parsed_incident.is_update is True
    assert parsed_incident.update_seq >= 1
    assert parsed_incident.dwell_time_seconds >= 2.0

    # Also verify statistics endpoint separates creations vs updates
    res_stats = client.get("/api/incidents/stats")
    assert res_stats.status_code == 200
    stats = res_stats.json()
    assert stats["incidents_created"] >= 1
    assert stats["incident_updates"] >= 1
    assert stats["total_incidents"] >= 1

    results["stages"]["6_schema_compliance"] = "PASS"
    print(f"[PASS] Stage 6: Incident schema verified: {parsed_incident.event_type} (Zone: {parsed_incident.zone_id}, Track: {parsed_incident.track_id}, Action: {parsed_incident.action}, UpdateSeq: {parsed_incident.update_seq})")

    # Stage 7: Clean up test zone and reset telemetry
    client.delete("/api/zones/e2e_robotic_perimeter")
    client.post("/api/pipeline/reset-telemetry")
    client.post("/api/pipeline/reset-incidents")
    res_clean_metrics = client.get("/api/metrics")
    assert res_clean_metrics.status_code == 200
    metrics_data = res_clean_metrics.json()
    assert metrics_data["fps"] is None, "Measured FPS was not reset after test teardown"
    assert metrics_data["inference_latency_ms"] is None, "Inference latency was not reset after test teardown"
    results["stages"]["7_cleanup"] = "PASS"
    print("[PASS] Stage 7: Test zone and test telemetry cleaned up successfully")

    results["overall_status"] = "PASS"

    log_dir = os.path.join(workspace_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    summary_path = os.path.join(log_dir, "end_to_end_test_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("==========================================================================")
    print(f" End-to-End Acceptance Test PASSED. Summary saved to: {summary_path}")
    print("==========================================================================")


if __name__ == "__main__":
    run_e2e_acceptance_test()

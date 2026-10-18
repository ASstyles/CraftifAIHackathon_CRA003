"""
Tests for Incident Semantics, Stable Identifiers, and SSE Lifecycle (Phase 4)
CraftifAI Physical AI Hackathon | Track A: PipeGen
"""

import json
import pytest
from datetime import datetime
from sentinel_core.models import RestrictedZone, Detection, SafetyIncident
from sentinel_core.zone_monitor import ZoneMonitor


def test_stable_incident_id_across_dwell_updates():
    """Verify that a single intrusion episode maintains a stable incident_id across periodic dwell updates."""
    monitor = ZoneMonitor(source_name="test_runner", load_defaults=False)
    zone = RestrictedZone(
        zone_id="robotic_cell_01",
        name="Robotic Cell 01",
        polygon=[[100.0, 100.0], [400.0, 100.0], [400.0, 400.0], [100.0, 400.0]],
        cooldown_seconds=3.0
    )
    monitor.add_zone(zone)

    det = Detection(track_id=7, confidence=0.94, bounding_box=[150.0, 150.0, 250.0, 300.0])

    # 1. Initial intrusion at t=10.0s
    incidents_t0 = monitor.process_detections([det], frame_timestamp=10.0)
    assert len(incidents_t0) == 1
    inc0 = incidents_t0[0]
    assert inc0.is_update is False
    assert inc0.action == "created"
    assert inc0.update_seq == 0
    assert inc0.incident_id.startswith("inc_robotic_cell_01_t7_")
    stable_id = inc0.incident_id

    # 2. Frame within cooldown at t=11.0s -> Suppressed
    incidents_t1 = monitor.process_detections([det], frame_timestamp=11.0)
    assert len(incidents_t1) == 0

    # 3. Dwell update at t=13.5s (> 3.0s cooldown elapsed)
    incidents_t2 = monitor.process_detections([det], frame_timestamp=13.5)
    assert len(incidents_t2) == 1
    inc2 = incidents_t2[0]
    assert inc2.is_update is True
    assert inc2.action == "updated"
    assert inc2.update_seq == 1
    assert inc2.incident_id == stable_id  # Stable ID preserved!
    assert inc2.dwell_time_seconds >= 3.0

    # 4. Another dwell update at t=17.0s (> 3.0s further)
    incidents_t3 = monitor.process_detections([det], frame_timestamp=17.0)
    assert len(incidents_t3) == 1
    inc3 = incidents_t3[0]
    assert inc3.is_update is True
    assert inc3.update_seq == 2
    assert inc3.incident_id == stable_id  # Stable ID preserved!


def test_exit_and_reentry_assigns_new_incident_id():
    """Verify that when a person leaves the hazard zone and re-enters, a new incident episode is created with a new incident_id."""
    monitor = ZoneMonitor(source_name="test_runner", load_defaults=False)
    zone = RestrictedZone(
        zone_id="press_zone",
        name="Hydraulic Press",
        polygon=[[0.0, 0.0], [200.0, 0.0], [200.0, 200.0], [0.0, 200.0]],
        cooldown_seconds=4.0
    )
    monitor.add_zone(zone)

    det_in = Detection(track_id=12, confidence=0.91, bounding_box=[50.0, 50.0, 100.0, 150.0])
    det_out = Detection(track_id=12, confidence=0.91, bounding_box=[300.0, 300.0, 350.0, 400.0])

    # Initial entry
    inc1 = monitor.process_detections([det_in], frame_timestamp=100.0)
    assert len(inc1) == 1
    id1 = inc1[0].incident_id

    # Exit
    monitor.process_detections([det_out], frame_timestamp=102.0)

    # Re-entry
    inc2 = monitor.process_detections([det_in], frame_timestamp=105.0)
    assert len(inc2) == 1
    id2 = inc2[0].incident_id

    assert id1 != id2
    assert inc2[0].action == "created"
    assert inc2[0].is_update is False
    assert inc2[0].update_seq == 0


def test_incident_counts_separation():
    """Verify that get_incident_counts provides distinct definitions for created, updates, and total recorded incidents."""
    monitor = ZoneMonitor(source_name="test_runner", load_defaults=False)
    zone = RestrictedZone(
        zone_id="zone_a",
        name="Zone A",
        polygon=[[0.0, 0.0], [200.0, 0.0], [200.0, 200.0], [0.0, 200.0]],
        cooldown_seconds=2.0
    )
    monitor.add_zone(zone)

    det1 = Detection(track_id=1, confidence=0.95, bounding_box=[10.0, 10.0, 50.0, 80.0])
    det2 = Detection(track_id=2, confidence=0.92, bounding_box=[100.0, 100.0, 150.0, 180.0])

    # Initial entry for two workers (2 created)
    monitor.process_detections([det1, det2], frame_timestamp=10.0)

    # Dwell update for worker 1 (1 update)
    monitor.process_detections([det1], frame_timestamp=13.0)

    stats = monitor.get_incident_counts()
    assert stats["incidents_created"] == 2
    assert stats["incident_updates"] == 1
    assert stats["total_incidents"] == 2  # 2 distinct ongoing/recorded incidents in history


def test_clear_incidents_and_tracks():
    """Verify that clear_incidents_and_tracks resets all state and counters cleanly."""
    monitor = ZoneMonitor(source_name="test_runner", load_defaults=False)
    zone = RestrictedZone(
        zone_id="zone_z",
        name="Zone Z",
        polygon=[[0.0, 0.0], [100.0, 0.0], [100.0, 100.0], [0.0, 100.0]],
        cooldown_seconds=1.0
    )
    monitor.add_zone(zone)

    det = Detection(track_id=99, confidence=0.9, bounding_box=[10.0, 10.0, 30.0, 50.0])
    monitor.process_detections([det], frame_timestamp=1.0)
    monitor.process_detections([det], frame_timestamp=3.0)

    assert len(monitor.incident_history) > 0
    assert len(monitor.tracks) > 0

    monitor.clear_incidents_and_tracks()
    assert len(monitor.incident_history) == 0
    assert len(monitor.tracks) == 0
    assert monitor.total_incidents_created == 0
    assert monitor.total_incident_updates == 0


def test_safety_incident_json_roundtrip_with_episode_fields():
    """Verify serialization and deserialization with episode fields."""
    incident = SafetyIncident(
        incident_id="inc_zone_1_t5_1697000000",
        action="updated",
        update_seq=3,
        is_update=True,
        event_type="restricted_zone_intrusion",
        zone_id="restricted_zone_1",
        confidence=0.89,
        bounding_box=[100.0, 200.0, 150.0, 300.0],
        dwell_time_seconds=9.5,
        source="PipeGen GPU"
    )

    data_json = incident.model_dump_json()
    parsed = json.loads(data_json)

    assert parsed["incident_id"] == "inc_zone_1_t5_1697000000"
    assert parsed["action"] == "updated"
    assert parsed["update_seq"] == 3
    assert parsed["is_update"] is True
    assert parsed["event_type"] == "restricted_zone_intrusion"

    rebuilt = SafetyIncident.model_validate_json(data_json)
    assert rebuilt.incident_id == incident.incident_id
    assert rebuilt.update_seq == 3
    assert rebuilt.is_update is True

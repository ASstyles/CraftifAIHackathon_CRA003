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


# =========================================================================
# Blocker 2 Deterministic Regression Tests
# =========================================================================

def test_regression_incident_ids_never_duplicated_across_distinct_tracks():
    """
    Regression Test 1: Incident IDs are never duplicated unexpectedly across distinct tracks or zones.
    Fails if distinct intrusion episodes collide on the same incident_id.
    """
    monitor = ZoneMonitor(source_name="test_runner", load_defaults=False)
    zone1 = RestrictedZone(
        zone_id="zone_robot_1",
        name="Robot Zone 1",
        polygon=[[0.0, 0.0], [100.0, 0.0], [100.0, 100.0], [0.0, 100.0]],
        cooldown_seconds=2.0
    )
    zone2 = RestrictedZone(
        zone_id="zone_robot_2",
        name="Robot Zone 2",
        polygon=[[200.0, 200.0], [300.0, 200.0], [300.0, 300.0], [200.0, 300.0]],
        cooldown_seconds=2.0
    )
    monitor.add_zone(zone1)
    monitor.add_zone(zone2)

    dets = [
        Detection(track_id=1, confidence=0.92, bounding_box=[10.0, 10.0, 50.0, 80.0]),
        Detection(track_id=2, confidence=0.91, bounding_box=[20.0, 20.0, 60.0, 90.0]),
        Detection(track_id=3, confidence=0.88, bounding_box=[210.0, 210.0, 250.0, 280.0]),
    ]

    incidents = monitor.process_detections(dets, frame_timestamp=100.0)
    assert len(incidents) == 3

    incident_ids = [inc.incident_id for inc in incidents]
    # All IDs must be unique
    assert len(set(incident_ids)) == len(incident_ids) == 3

    # IDs must contain zone_id and track_id
    assert "zone_robot_1_t1" in incident_ids[0]
    assert "zone_robot_1_t2" in incident_ids[1]
    assert "zone_robot_2_t3" in incident_ids[2]


def test_regression_creation_counts_agree_with_actual_records():
    """
    Regression Test 2: Creation counts agree strictly with actual creation records.
    Fails if total_incidents_created does not match observed initial creation events.
    """
    monitor = ZoneMonitor(source_name="test_runner", load_defaults=False)
    zone = RestrictedZone(
        zone_id="press_zone",
        name="Hydraulic Press",
        polygon=[[0.0, 0.0], [200.0, 0.0], [200.0, 200.0], [0.0, 200.0]],
        cooldown_seconds=2.0
    )
    monitor.add_zone(zone)

    dets = [
        Detection(track_id=10, confidence=0.95, bounding_box=[20.0, 20.0, 50.0, 80.0]),
        Detection(track_id=20, confidence=0.93, bounding_box=[30.0, 30.0, 60.0, 90.0]),
    ]

    # Two workers enter at t=10.0
    created_events = monitor.process_detections(dets, frame_timestamp=10.0)
    assert len(created_events) == 2

    stats = monitor.get_incident_counts()
    assert stats["incidents_created"] == 2
    assert stats["incident_updates"] == 0
    assert stats["total_incidents"] == 2

    # Verify directly against history records
    recorded_creations = [inc for inc in monitor.incident_history if inc.action == "created" or inc.update_seq == 0]
    assert len(recorded_creations) == stats["incidents_created"]


def test_regression_update_events_not_counted_as_new_creations():
    """
    Regression Test 3: Update events are NEVER counted as newly created incidents.
    Fails if dwelling in a hazard zone inflates the incidents_created counter.
    """
    monitor = ZoneMonitor(source_name="test_runner", load_defaults=False)
    zone = RestrictedZone(
        zone_id="saw_cell",
        name="Rotary Saw Cell",
        polygon=[[0.0, 0.0], [100.0, 0.0], [100.0, 100.0], [0.0, 100.0]],
        cooldown_seconds=1.5
    )
    monitor.add_zone(zone)

    det = Detection(track_id=5, confidence=0.95, bounding_box=[10.0, 10.0, 40.0, 70.0])

    # Initial intrusion (creation)
    inc0 = monitor.process_detections([det], frame_timestamp=10.0)
    assert len(inc0) == 1
    assert inc0[0].action == "created"

    # 3 subsequent dwell updates (> 1.5s cooldown between each)
    inc1 = monitor.process_detections([det], frame_timestamp=12.0)
    assert len(inc1) == 1
    assert inc1[0].action == "updated"

    inc2 = monitor.process_detections([det], frame_timestamp=14.0)
    assert len(inc2) == 1
    assert inc2[0].action == "updated"

    inc3 = monitor.process_detections([det], frame_timestamp=16.0)
    assert len(inc3) == 1
    assert inc3[0].action == "updated"

    stats = monitor.get_incident_counts()
    # CRITICAL INVARIANT: 1 unique incident created, 3 updates, 1 episode recorded in history
    assert stats["incidents_created"] == 1
    assert stats["incident_updates"] == 3
    assert stats["total_incidents"] == 1
    assert len(monitor.incident_history) == 1


def test_regression_sse_events_delivery_count_and_no_loss():
    """
    Regression Test 4: SSE events are neither lost nor duplicated.
    Fails if subscriber queue receives fewer or more events than total creations + updates.
    """
    import asyncio
    from backend.pipeline_manager import PipelineManager
    import os

    async def _run_test():
        workspace_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        mgr = PipelineManager(workspace_dir=workspace_dir)
        mgr.mode = "live"
        mgr.zone_monitor.clear_zones()
        mgr.zone_monitor.clear_incidents_and_tracks()

        zone = RestrictedZone(
            zone_id="test_sse_zone",
            name="SSE Validation Zone",
            polygon=[[0.0, 0.0], [200.0, 0.0], [200.0, 200.0], [0.0, 200.0]],
            cooldown_seconds=2.0
        )
        mgr.zone_monitor.add_zone(zone)

        # Register an SSE subscriber queue
        q = await mgr.register_subscriber()

        det = Detection(track_id=88, confidence=0.96, bounding_box=[20.0, 20.0, 60.0, 100.0])

        # Trigger 1 creation (t=10.0) and 2 updates (t=13.0, t=16.0)
        for t in [10.0, 13.0, 16.0]:
            incidents = mgr.zone_monitor.process_detections([det], frame_timestamp=t)
            for inc in incidents:
                await mgr.broadcast_incident(inc)

        # Collect events from subscriber queue
        received = []
        while not q.empty():
            item = q.get_nowait()
            received.append(json.loads(item))

        mgr.unregister_subscriber(q)

        # Expected: exactly 3 events received (1 creation + 2 updates)
        assert len(received) == 3
        assert received[0]["action"] == "created"
        assert received[0]["update_seq"] == 0
        assert received[1]["action"] == "updated"
        assert received[1]["update_seq"] == 1
        assert received[2]["action"] == "updated"
        assert received[2]["update_seq"] == 2

        # All share the exact same incident_id
        assert received[0]["incident_id"] == received[1]["incident_id"] == received[2]["incident_id"]

    asyncio.run(_run_test())


def test_regression_event_order_and_update_sequence_strictly_monotonic():
    """
    Regression Test 5: Event order and update sequence is strictly monotonic.
    Fails if update sequence decrements or skips, or if dwell time goes backwards.
    """
    monitor = ZoneMonitor(source_name="test_runner", load_defaults=False)
    zone = RestrictedZone(
        zone_id="conveyor_zone",
        name="Conveyor Area",
        polygon=[[0.0, 0.0], [100.0, 0.0], [100.0, 100.0], [0.0, 100.0]],
        cooldown_seconds=1.0
    )
    monitor.add_zone(zone)

    det = Detection(track_id=42, confidence=0.90, bounding_box=[10.0, 10.0, 30.0, 60.0])

    timestamps = [100.0, 101.5, 103.0, 105.0]
    collected_incidents = []

    for t in timestamps:
        res = monitor.process_detections([det], frame_timestamp=t)
        collected_incidents.extend(res)

    assert len(collected_incidents) == 4

    last_seq = -1
    last_dwell = -1.0
    initial_id = collected_incidents[0].incident_id

    for idx, inc in enumerate(collected_incidents):
        assert inc.incident_id == initial_id
        assert inc.update_seq == idx
        assert inc.update_seq > last_seq
        assert inc.dwell_time_seconds >= last_dwell
        if idx == 0:
            assert inc.action == "created"
            assert inc.is_update is False
        else:
            assert inc.action == "updated"
            assert inc.is_update is True
        last_seq = inc.update_seq
        last_dwell = inc.dwell_time_seconds


def test_regression_run_isolation_prevents_stale_data_leakage():
    """
    Regression Test 6: Run isolation reset prevents stale data from another run leaking into acceptance metrics.
    Fails if resetting state leaves behind remnants of prior runs.
    """
    monitor = ZoneMonitor(source_name="test_runner", load_defaults=False)
    zone = RestrictedZone(
        zone_id="cell_alpha",
        name="Cell Alpha",
        polygon=[[0.0, 0.0], [100.0, 0.0], [100.0, 100.0], [0.0, 100.0]],
        cooldown_seconds=1.0
    )
    monitor.add_zone(zone)

    # 1. Run A executes (generating 2 incidents from 2 workers)
    det_a1 = Detection(track_id=1, confidence=0.9, bounding_box=[10.0, 10.0, 30.0, 50.0])
    det_a2 = Detection(track_id=2, confidence=0.9, bounding_box=[40.0, 40.0, 60.0, 70.0])
    monitor.process_detections([det_a1, det_a2], frame_timestamp=10.0)
    stats_a = monitor.get_incident_counts()
    assert stats_a["incidents_created"] == 2
    assert len(monitor.incident_history) == 2
    run_a_ids = set(inc.incident_id for inc in monitor.incident_history)

    # 2. ISOLATION RESET
    monitor.clear_incidents_and_tracks()

    # Verify complete zeroing
    stats_post_reset = monitor.get_incident_counts()
    assert stats_post_reset["incidents_created"] == 0
    assert stats_post_reset["incident_updates"] == 0
    assert stats_post_reset["total_incidents"] == 0
    assert stats_post_reset["active_intrusions"] == 0
    assert len(monitor.incident_history) == 0

    # 3. Run B executes (worker 3 enters at t=20.0 and dwells until t=22.0)
    det_b = Detection(track_id=3, confidence=0.95, bounding_box=[15.0, 15.0, 35.0, 55.0])
    monitor.process_detections([det_b], frame_timestamp=20.0)  # created
    monitor.process_detections([det_b], frame_timestamp=22.0)  # updated

    stats_b = monitor.get_incident_counts()
    # Invariant: Run B stats MUST NOT include Run A (creation must be 1, NOT 2+1=3)
    assert stats_b["incidents_created"] == 1
    assert stats_b["incident_updates"] == 1
    assert stats_b["total_incidents"] == 1
    assert len(monitor.incident_history) == 1

    # Verify none of Run A's incident IDs appear in Run B
    run_b_ids = set(inc.incident_id for inc in monitor.incident_history)
    assert len(run_a_ids.intersection(run_b_ids)) == 0


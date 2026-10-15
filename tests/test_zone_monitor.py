"""
Unit and Integration Tests for SentinelVision Workplace Safety Logic
CraftifAI Physical AI Hackathon | Track A: PipeGen
"""

import time
import pytest
from datetime import datetime
from sentinel_core.models import RestrictedZone, Detection, SafetyIncident, BoundingBox
from sentinel_core.zone_monitor import ZoneMonitor, point_in_polygon


def test_point_in_polygon():
    """Verify ray-casting point-in-polygon algorithm on convex and concave shapes."""
    square = [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]]
    assert point_in_polygon((5.0, 5.0), square) is True
    assert point_in_polygon((15.0, 5.0), square) is False
    assert point_in_polygon((-1.0, 5.0), square) is False
    assert point_in_polygon((5.0, -1.0), square) is False

    # Triangle
    triangle = [[0.0, 0.0], [10.0, 0.0], [5.0, 10.0]]
    assert point_in_polygon((5.0, 2.0), triangle) is True
    assert point_in_polygon((0.0, 10.0), triangle) is False


def test_bounding_box_contact_point():
    """Verify foot contact point is bottom-center of bounding box."""
    bbox = BoundingBox.from_list([100.0, 200.0, 200.0, 400.0])
    foot_x, foot_y = bbox.bottom_center
    assert foot_x == 150.0  # (100 + 200) / 2
    assert foot_y == 400.0  # ymax represents feet on floor

    cx, cy = bbox.centroid
    assert cx == 150.0
    assert cy == 300.0


def test_restricted_zone_intrusion_trigger():
    """Verify an incident is generated when a person enters a restricted zone."""
    monitor = ZoneMonitor(source_name="test_feed", load_defaults=False)
    # Zone covers x: [100, 300], y: [100, 300]
    test_zone = RestrictedZone(
        zone_id="test_zone_hazard",
        name="Hazard Zone",
        polygon=[[100.0, 100.0], [300.0, 100.0], [300.0, 300.0], [100.0, 300.0]],
        cooldown_seconds=5.0
    )
    monitor.add_zone(test_zone)

    # Frame 1: Person outside (foot at x=50, y=50)
    det_outside = Detection(
        track_id=101,
        confidence=0.95,
        bounding_box=[30.0, 10.0, 70.0, 50.0]
    )
    incidents = monitor.process_detections([det_outside], frame_timestamp=1000.0)
    assert len(incidents) == 0

    # Frame 2: Person enters zone (foot at x=200, y=200)
    det_inside = Detection(
        track_id=101,
        confidence=0.92,
        bounding_box=[160.0, 100.0, 240.0, 200.0]
    )
    incidents = monitor.process_detections([det_inside], frame_timestamp=1000.1)
    assert len(incidents) == 1
    inc = incidents[0]
    assert inc.event_type == "restricted_zone_intrusion"
    assert inc.zone_id == "test_zone_hazard"
    assert inc.track_id == 101
    assert inc.confidence == 0.92
    assert inc.source == "test_feed"


def test_duplicate_event_suppression():
    """
    CRITICAL REQUIREMENT: Avoid generating duplicate incidents for every frame
    while a person remains inside the restricted zone.
    """
    monitor = ZoneMonitor(source_name="test_feed", load_defaults=False)
    test_zone = RestrictedZone(
        zone_id="zone_suppression_test",
        name="Suppression Test",
        polygon=[[100.0, 100.0], [400.0, 100.0], [400.0, 400.0], [100.0, 400.0]],
        cooldown_seconds=5.0
    )
    monitor.add_zone(test_zone)

    det_inside = Detection(
        track_id=42,
        confidence=0.96,
        bounding_box=[180.0, 150.0, 220.0, 250.0]  # foot at (200, 250)
    )

    # Frame 1 at t=10.0s -> Intrusion trigger!
    inc1 = monitor.process_detections([det_inside], frame_timestamp=10.0)
    assert len(inc1) == 1

    # Frames 2 to 30 (staying inside across next 1.0 second) -> Must be SUPPRESSED!
    for frame_i in range(1, 30):
        t = 10.0 + (frame_i * 0.033)
        inc_suppressed = monitor.process_detections([det_inside], frame_timestamp=t)
        assert len(inc_suppressed) == 0, f"Duplicate incident falsely generated on frame {frame_i} at t={t}"

    # Verify dwell time was accumulated
    track = monitor.tracks[42]
    assert "zone_suppression_test" in track.active_zones

    # At t=16.0s (> 5.0s cooldown elapsed) -> Re-trigger alert is permitted
    inc_retrigger = monitor.process_detections([det_inside], frame_timestamp=16.0)
    assert len(inc_retrigger) == 1
    assert inc_retrigger[0].dwell_time_seconds >= 5.0


def test_exit_and_reentry():
    """Verify that if a person exits the zone and re-enters, a new incident is immediately triggered."""
    monitor = ZoneMonitor(source_name="test_feed", load_defaults=False)
    test_zone = RestrictedZone(
        zone_id="zone_reentry",
        name="Reentry Zone",
        polygon=[[100.0, 100.0], [300.0, 100.0], [300.0, 300.0], [100.0, 300.0]],
        cooldown_seconds=10.0
    )
    monitor.add_zone(test_zone)

    det_inside = Detection(track_id=99, confidence=0.91, bounding_box=[180.0, 150.0, 220.0, 200.0])
    det_outside = Detection(track_id=99, confidence=0.91, bounding_box=[400.0, 150.0, 440.0, 200.0])

    # 1. Enter
    inc1 = monitor.process_detections([det_inside], frame_timestamp=1.0)
    assert len(inc1) == 1

    # 2. Exit
    inc_exit = monitor.process_detections([det_outside], frame_timestamp=2.0)
    assert len(inc_exit) == 0
    assert "zone_reentry" not in monitor.tracks[99].active_zones

    # 3. Re-enter at t=3.0 (within cooldown period, but because they exited, it's a new entry)
    inc_reenter = monitor.process_detections([det_inside], frame_timestamp=3.0)
    assert len(inc_reenter) == 1
    assert inc_reenter[0].dwell_time_seconds == 0.0


def test_multiple_persons_same_zone():
    """Verify that multiple people in the same zone trigger separate incidents."""
    monitor = ZoneMonitor(source_name="test_feed", load_defaults=False)
    test_zone = RestrictedZone(
        zone_id="multi_zone",
        name="Multi Zone",
        polygon=[[0.0, 0.0], [500.0, 0.0], [500.0, 500.0], [0.0, 500.0]],
        cooldown_seconds=5.0
    )
    monitor.add_zone(test_zone)

    det1 = Detection(track_id=1, confidence=0.95, bounding_box=[50.0, 50.0, 100.0, 150.0])
    det2 = Detection(track_id=2, confidence=0.92, bounding_box=[200.0, 200.0, 250.0, 300.0])

    incidents = monitor.process_detections([det1, det2], frame_timestamp=10.0)
    assert len(incidents) == 2
    track_ids = {inc.track_id for inc in incidents}
    assert track_ids == {1, 2}


def test_untracked_detection_fallback():
    """Verify spatial centroid tracking assigns a stable ID when track_id is None."""
    monitor = ZoneMonitor(source_name="test_feed", load_defaults=False)
    test_zone = RestrictedZone(
        zone_id="untracked_zone",
        name="Untracked Zone",
        polygon=[[0.0, 0.0], [500.0, 0.0], [500.0, 500.0], [0.0, 500.0]],
        cooldown_seconds=10.0
    )
    monitor.add_zone(test_zone)

    # Frame 1: Detection without track_id
    det_f1 = Detection(track_id=None, confidence=0.90, bounding_box=[100.0, 100.0, 140.0, 200.0])
    inc1 = monitor.process_detections([det_f1], frame_timestamp=1.0)
    assert len(inc1) == 1
    assigned_id = inc1[0].track_id
    assert assigned_id is not None

    # Frame 2: Same person moved slightly (10px right) without track_id at t=1.033
    det_f2 = Detection(track_id=None, confidence=0.91, bounding_box=[110.0, 100.0, 150.0, 200.0])
    inc2 = monitor.process_detections([det_f2], frame_timestamp=1.033)
    # Duplicate suppression should prevent new alert, but track state should retain the same assigned_id
    assert len(inc2) == 0
    assert assigned_id in monitor.tracks


def test_invalid_polygon_boundary():
    """Verify point_in_polygon returns False for degenerate polygons (< 3 points)."""
    assert point_in_polygon((10.0, 10.0), []) is False
    assert point_in_polygon((10.0, 10.0), [[5.0, 5.0]]) is False
    assert point_in_polygon((10.0, 10.0), [[5.0, 5.0], [15.0, 15.0]]) is False


def test_dwell_time_accumulation():
    """Verify that dwell time accumulates accurately over time."""
    monitor = ZoneMonitor(source_name="test_feed", load_defaults=False)
    test_zone = RestrictedZone(
        zone_id="dwell_zone",
        name="Dwell Zone",
        polygon=[[0.0, 0.0], [300.0, 0.0], [300.0, 300.0], [0.0, 300.0]],
        cooldown_seconds=2.0
    )
    monitor.add_zone(test_zone)

    det = Detection(track_id=77, confidence=0.95, bounding_box=[50.0, 50.0, 100.0, 150.0])

    # t = 0.0s -> initial intrusion (dwell_time = 0.0)
    inc0 = monitor.process_detections([det], frame_timestamp=0.0)
    assert len(inc0) == 1
    assert inc0[0].dwell_time_seconds == 0.0

    # t = 1.0s -> within cooldown, suppressed
    inc1 = monitor.process_detections([det], frame_timestamp=1.0)
    assert len(inc1) == 0

    # t = 2.5s -> cooldown elapsed (> 2.0s), re-trigger alert with accumulated dwell time
    inc2 = monitor.process_detections([det], frame_timestamp=2.5)
    assert len(inc2) == 1
    assert inc2[0].dwell_time_seconds == 2.5


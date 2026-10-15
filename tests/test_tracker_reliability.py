"""
Tests for SpatialTracker Reliability and Identity Continuity
CraftifAI Physical AI Hackathon | Track A: PipeGen
"""

import pytest
import numpy as np
from pipegen_pipeline.official_pipegen_build.src1.yolov8_pose_parallel.yolov8_pose import (
    SpatialTracker,
    Track
)


def test_tracker_initializes_unique_tracks():
    """Verify separate track IDs are created for distinct detections."""
    tracker = SpatialTracker(max_age=10, iou_threshold=0.3)
    # Two people far apart
    dets = [
        ([100.0, 100.0, 150.0, 250.0], 0.95, 0),
        ([400.0, 100.0, 450.0, 250.0], 0.92, 0)
    ]
    tracks = tracker.update(dets)
    assert len(tracks) == 2
    assert tracks[0].track_id != tracks[1].track_id
    assert tracks[0].hits == 1
    assert tracks[1].hits == 1


def test_tracker_preserves_id_across_frames():
    """Verify track ID remains stable when bounding box moves smoothly."""
    tracker = SpatialTracker(max_age=10, iou_threshold=0.3)
    dets_f1 = [([100.0, 100.0, 150.0, 250.0], 0.95, 0)]
    tracks_f1 = tracker.update(dets_f1)
    tid = tracks_f1[0].track_id

    # Frame 2: Slight movement (high IoU)
    dets_f2 = [([105.0, 102.0, 155.0, 252.0], 0.94, 0)]
    tracks_f2 = tracker.update(dets_f2)
    assert len(tracks_f2) == 1
    assert tracks_f2[0].track_id == tid
    assert tracks_f2[0].hits == 2


def test_tracker_survives_temporary_occlusion():
    """Verify track survives missing detections up to max_age."""
    tracker = SpatialTracker(max_age=5, iou_threshold=0.3)
    dets = [([100.0, 100.0, 150.0, 250.0], 0.95, 0)]
    tracks_f1 = tracker.update(dets)
    tid = tracks_f1[0].track_id

    # Frames 2-4: Missing detection (occluded by forklift)
    for _ in range(3):
        active = tracker.update([])
        assert len(active) == 0

    # Frame 5: Person reappears near last location
    dets_reappear = [([108.0, 104.0, 158.0, 254.0], 0.91, 0)]
    tracks_reappear = tracker.update(dets_reappear)
    assert len(tracks_reappear) == 1
    assert tracks_reappear[0].track_id == tid


def test_tracker_retires_dead_tracks_after_max_age():
    """Verify track is removed from internal state after max_age expires."""
    tracker = SpatialTracker(max_age=3, iou_threshold=0.3)
    dets = [([100.0, 100.0, 150.0, 250.0], 0.95, 0)]
    tracks_f1 = tracker.update(dets)
    tid = tracks_f1[0].track_id

    # 4 frames of absence (exceeds max_age=3)
    for _ in range(4):
        tracker.update([])

    # Internal tracks list should be empty
    assert len(tracker.tracks) == 0

    # Person reappears - must be assigned a new track ID
    dets_new = [([100.0, 100.0, 150.0, 250.0], 0.95, 0)]
    tracks_new = tracker.update(dets_new)
    assert len(tracks_new) == 1
    assert tracks_new[0].track_id != tid


def test_tracker_iou_computation():
    """Verify IoU calculation on identical, overlapping, and disjoint boxes."""
    b1 = [0.0, 0.0, 10.0, 10.0]
    b2 = [0.0, 0.0, 10.0, 10.0]
    assert SpatialTracker._compute_iou(b1, b2) == pytest.approx(1.0)

    b3 = [100.0, 100.0, 110.0, 110.0]
    assert SpatialTracker._compute_iou(b1, b3) == 0.0

    # 50% overlap: intersection 5x10=50, union 100+100-50=150 -> IoU 1/3
    b4 = [5.0, 0.0, 15.0, 10.0]
    assert SpatialTracker._compute_iou(b1, b4) == pytest.approx(50.0 / 150.0)

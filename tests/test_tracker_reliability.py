"""
Tests for SpatialTracker Reliability and Identity Continuity
CraftifAI Physical AI Hackathon | Track A: PipeGen
"""

import pytest
import numpy as np
from pipegen_pipeline.official_pipegen_build.src1.yolov8_pose_parallel.yolov8_pose import (
    SpatialTracker,
    Track,
    compute_ground_contact
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


def test_tracker_crossing_paths_with_spatial_separation():
    """Verify that two tracks moving in opposing directions preserve IDs when trajectories retain dominant IoU."""
    tracker = SpatialTracker(max_age=5, iou_threshold=0.3)
    # Person 1 moving left to right: [100, 100, 160, 200] (width 60)
    # Person 2 moving right to left: [260, 100, 320, 200] (width 60)
    f1 = [([100.0, 100.0, 160.0, 200.0], 0.9, 0), ([260.0, 100.0, 320.0, 200.0], 0.9, 0)]
    t_f1 = tracker.update(f1)
    id1, id2 = t_f1[0].track_id, t_f1[1].track_id

    # Frame 2: Natural video frame motion (12 px step)
    f2 = [([112.0, 100.0, 172.0, 200.0], 0.9, 0), ([248.0, 100.0, 308.0, 200.0], 0.9, 0)]
    t_f2 = tracker.update(f2)
    # Both IDs preserved
    t_ids = {t.track_id for t in t_f2}
    assert t_ids == {id1, id2}

    # Frame 3: Continued motion
    f3 = [([125.0, 100.0, 185.0, 200.0], 0.9, 0), ([235.0, 100.0, 295.0, 200.0], 0.9, 0)]
    t_f3 = tracker.update(f3)
    t_ids3 = {t.track_id for t in t_f3}
    assert t_ids3 == {id1, id2}


def test_tracker_handles_out_of_frame_or_degenerate_boxes():
    """Verify tracker does not crash or corrupt state on zero-width or out-of-frame boxes."""
    tracker = SpatialTracker(max_age=5, iou_threshold=0.3)
    # Box with inverted or degenerate dimensions
    degenerate = [([50.0, 50.0, 50.0, 50.0], 0.5, 0)]
    tracks = tracker.update(degenerate)
    assert len(tracks) == 1
    assert np.allclose(tracks[0].box, [50.0, 50.0, 50.0, 50.0])

    # Negative coordinates (partially off-screen top-left)
    offscreen = [([-20.0, -10.0, 40.0, 80.0], 0.8, 0)]
    tracks_off = tracker.update(offscreen)
    assert len(tracks_off) >= 1


def test_ground_contact_pose_ankles_and_fallback():
    """Verify compute_ground_contact logic across both ankles, single ankle, and bbox fallback."""
    person_box = [100.0, 100.0, 200.0, 300.0]

    # Case 1: Both ankles visible with high confidence
    kx = [0.0] * 17
    ky = [0.0] * 17
    kc = [0.0] * 17
    kx[15], ky[15], kc[15] = 140.0, 295.0, 0.85  # Left ankle
    kx[16], ky[16], kc[16] = 160.0, 298.0, 0.90  # Right ankle
    pose_res_both = [{
        "box": [95.0, 95.0, 205.0, 305.0],
        "score": 0.9,
        "kx": kx,
        "ky": ky,
        "kc": kc
    }]
    pt, src = compute_ground_contact(person_box, pose_res_both)
    assert src == "pose_ankles"
    assert pt[0] == pytest.approx(150.0)  # (140 + 160) / 2
    assert pt[1] == pytest.approx(298.0)  # max(295, 298)

    # Case 2: Only left ankle visible
    kc_left_only = list(kc)
    kc_left_only[16] = 0.1  # Right ankle occluded
    pose_res_left = [{
        "box": [95.0, 95.0, 205.0, 305.0],
        "score": 0.9,
        "kx": kx,
        "ky": ky,
        "kc": kc_left_only
    }]
    pt, src = compute_ground_contact(person_box, pose_res_left)
    assert src == "pose_left_ankle"
    assert pt[0] == pytest.approx(140.0)
    assert pt[1] == pytest.approx(295.0)

    # Case 3: No pose overlap or low confidence -> fallback to bbox bottom-center
    pt_fallback, src_fallback = compute_ground_contact(person_box, [])
    assert src_fallback == "bbox_bottom_center"
    assert pt_fallback[0] == pytest.approx(150.0)  # (100 + 200) / 2
    assert pt_fallback[1] == pytest.approx(300.0)  # y2


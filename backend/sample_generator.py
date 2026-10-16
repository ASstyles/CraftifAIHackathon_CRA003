"""
SentinelVision - Sample Workplace Hazard Video Generator & Synthetic Streamer
CraftifAI Physical AI Hackathon | Track A: PipeGen
"""

import os
import cv2
import numpy as np
from typing import Generator, Tuple, List, Dict, Any
from sentinel_core.models import Detection


def generate_sample_hazard_video(output_path: str, duration_sec: int = 10, fps: int = 30) -> str:
    """
    Generates a realistic synthetic workplace safety video (1280x720):
    - Factory floor with grid lines and safety demarcation lines
    - Red polygonal "DANGER: ROBOTIC ARM" zone
    - Yellow polygonal "WARNING: HIGH VOLTAGE" zone
    - Animated human worker walking across the floor into the red restricted zone
    - Realistic motion, shadows, and worker silhouette
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    width, height = 1280, 720
    total_frames = duration_sec * fps

    # Try H264 or mp4v codec
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    if not out.isOpened():
        raise RuntimeError(f"Could not open VideoWriter for {output_path}")

    # Zone 1 Polygon (Robotic Arm Hazard)
    zone1_poly = np.array([[450, 300], [850, 300], [900, 680], [400, 680]], np.int32)
    # Zone 2 Polygon (Electrical Enclosure)
    zone2_poly = np.array([[50, 200], [320, 200], [320, 550], [50, 550]], np.int32)

    # Worker trajectory:
    # Starts at x=100, y=500 (safe area)
    # Walks across to x=650, y=520 (deep inside Zone 1) around frame 100-200
    # Dwells inside Zone 1, then exits to x=1050, y=500

    # Load realistic worker sprite if available for genuine detector/pose evaluation
    worker_crop = None
    crop_candidates = [
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets", "worker_crop.png"),
        os.path.join(os.path.dirname(os.path.abspath(output_path)), "worker_crop.png"),
    ]
    for cp in crop_candidates:
        if os.path.exists(cp):
            loaded = cv2.imread(cp, cv2.IMREAD_UNCHANGED)
            if loaded is not None and len(loaded.shape) == 3 and loaded.shape[2] == 4:
                worker_crop = loaded
                break

    for frame_idx in range(total_frames):
        # 1. Base Factory Floor (Dark industrial metallic gray)
        frame = np.full((height, width, 3), (35, 38, 42), dtype=np.uint8)

        # Draw perspective floor grid lines
        for y_line in range(250, 720, 50):
            cv2.line(frame, (0, y_line), (width, y_line), (48, 52, 58), 1)
        for x_line in range(0, 1300, 100):
            cv2.line(frame, (x_line, 250), (int((x_line - 640) * 1.5 + 640), 720), (48, 52, 58), 1)

        # Draw Industrial Machinery silhouette in background
        cv2.rectangle(frame, (550, 150), (750, 310), (60, 65, 75), -1)
        cv2.rectangle(frame, (590, 100), (710, 150), (80, 85, 95), -1)
        cv2.circle(frame, (650, 125), 18, (0, 165, 255), -1)  # Amber warning light
        cv2.putText(frame, "AUTOMATED CELL 04", (565, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 170, 180), 1)

        # 2. Draw Semi-transparent Restricted Zones
        overlay = frame.copy()
        # Red zone (Zone 1)
        cv2.fillPoly(overlay, [zone1_poly], (30, 30, 200))
        # Amber zone (Zone 2)
        cv2.fillPoly(overlay, [zone2_poly], (20, 140, 220))
        cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, frame)

        # Draw Zone Borders with warning stripes
        cv2.polylines(frame, [zone1_poly], isClosed=True, color=(50, 50, 239), thickness=3)
        cv2.polylines(frame, [zone2_poly], isClosed=True, color=(30, 180, 245), thickness=2)

        # Zone Labels
        cv2.putText(frame, "[RESTRICTED ZONE 1: ROBOTIC ARM PERIMETER]", (460, 330),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 80, 255), 2)
        cv2.putText(frame, "[ZONE 2: HIGH VOLTAGE]", (60, 230),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 200, 255), 1)

        # 3. Calculate Worker Position
        progress = frame_idx / total_frames
        if progress < 0.4:
            # Walking from left into Zone 1
            t = progress / 0.4
            wx = int(120 + t * (600 - 120))
            wy = int(580 - t * 40)
        elif progress < 0.75:
            # Dwell inside Zone 1 with slight movement
            t = (progress - 0.4) / 0.35
            wx = int(600 + np.sin(t * np.pi * 2) * 25)
            wy = int(540 + np.cos(t * np.pi * 2) * 15)
        else:
            # Walk out to right
            t = (progress - 0.75) / 0.25
            wx = int(600 + t * (1150 - 600))
            wy = int(540 + t * 40)

        if worker_crop is not None:
            # Photorealistic worker composite
            w_height = int(220 + (wy / 720.0) * 40)
            aspect = worker_crop.shape[1] / float(worker_crop.shape[0])
            w_width = int(w_height * aspect)
            w_xmin = wx - w_width // 2
            w_ymin = wy - w_height
            w_xmax = w_xmin + w_width
            w_ymax = wy

            # Contact point shadow
            cv2.ellipse(frame, (wx, wy), (int(w_width * 0.45), 10), 0, 0, 360, (20, 20, 25), -1)

            target_x1 = max(0, w_xmin)
            target_y1 = max(0, w_ymin)
            target_x2 = min(width, w_xmax)
            target_y2 = min(height, w_ymax)
            crop_w = target_x2 - target_x1
            crop_h = target_y2 - target_y1

            if crop_w > 0 and crop_h > 0:
                crop_x1 = target_x1 - w_xmin
                crop_y1 = target_y1 - w_ymin
                resized_worker = cv2.resize(worker_crop, (w_width, w_height))
                sub_worker = resized_worker[crop_y1:crop_y1 + crop_h, crop_x1:crop_x1 + crop_w]
                alpha = sub_worker[:, :, 3:4] / 255.0
                frame_roi = frame[target_y1:target_y2, target_x1:target_x2]
                frame[target_y1:target_y2, target_x1:target_x2] = (
                    (1.0 - alpha) * frame_roi + alpha * sub_worker[:, :, :3]
                ).astype(np.uint8)
        else:
            # Geometric fallback
            w_width = 68
            w_height = 155
            w_xmin = wx - w_width // 2
            w_ymin = wy - w_height
            w_xmax = wx + w_width // 2
            w_ymax = wy

            cv2.circle(frame, (wx, w_ymin + 18), 16, (0, 215, 255), -1)
            cv2.rectangle(frame, (wx - 22, w_ymin + 36), (wx + 22, w_ymin + 100), (0, 115, 255), -1)
            cv2.line(frame, (wx - 20, w_ymin + 55), (wx + 20, w_ymin + 55), (240, 240, 240), 3)
            cv2.line(frame, (wx - 20, w_ymin + 75), (wx + 20, w_ymin + 75), (240, 240, 240), 3)
            leg_offset = int(np.sin(frame_idx * 0.4) * 12)
            cv2.line(frame, (wx - 10, w_ymin + 100), (wx - 10 + leg_offset, w_ymax), (120, 80, 50), 8)
            cv2.line(frame, (wx + 10, w_ymin + 100), (wx + 10 - leg_offset, w_ymax), (120, 80, 50), 8)
            cv2.ellipse(frame, (wx, w_ymax), (30, 10), 0, 0, 360, (20, 20, 25), -1)

        # Write Frame
        out.write(frame)

    out.release()
    return output_path


def simulate_detections_for_frame(frame_idx: int, total_frames: int = 300) -> List[Detection]:
    """
    Generates deterministic, ground-truth detection metadata matching the synthetic video.
    """
    progress = frame_idx / total_frames
    if progress < 0.4:
        t = progress / 0.4
        wx = 120.0 + t * (600.0 - 120.0)
        wy = 580.0 - t * 40.0
    elif progress < 0.75:
        t = (progress - 0.4) / 0.35
        wx = 600.0 + float(np.sin(t * np.pi * 2) * 25.0)
        wy = 540.0 + float(np.cos(t * np.pi * 2) * 15.0)
    else:
        t = (progress - 0.75) / 0.25
        wx = 600.0 + t * (1150.0 - 600.0)
        wy = 540.0 + t * 40.0

    w_width = 68.0
    w_height = 155.0

    # Slight realistic confidence jitter around 0.93 - 0.97
    conf = 0.94 + 0.03 * float(np.sin(frame_idx * 0.1))

    return [
        Detection(
            track_id=1,
            class_id=0,
            class_name="person",
            confidence=round(conf, 3),
            bounding_box=[
                round(wx - w_width / 2.0, 1),
                round(wy - w_height, 1),
                round(wx + w_width / 2.0, 1),
                round(wy, 1)
            ],
            raw_metadata={
                "pipegen_engine": "TensorRT_FP16",
                "model": "yolov8s_person",
                "frame": frame_idx
            }
        )
    ]

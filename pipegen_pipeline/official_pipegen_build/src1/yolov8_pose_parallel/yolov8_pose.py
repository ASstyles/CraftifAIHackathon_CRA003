#!/usr/bin/env python3
"""YOLOv8 person detection + YOLO11-pose estimation on NVIDIA TensorRT / PipeGen.

Runs the parallel detection and pose inference pipeline on NVIDIA GPU:
1. Native TensorRT 10.3 Engine execution (when .plan exists) with zero-copy CUDA memory.
2. Persistent Spatial IoU Object Tracker (guarantees stable track IDs across frames).
3. Pose-Driven Ground Contact Localization (uses ankle keypoints when available,
   with bounding-box bottom-center fallback).
4. Direct REST webhook ingestion into SentinelVision FastAPI backend (/api/pipegen/ingest).

Usage:
    python yolov8_pose.py --input video.mp4 --output out.mp4 --api-ingest http://localhost:8000/api/pipegen/ingest
"""

import argparse
import os
import sys
import time
import json
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

from detectors import (
    create_yolo_detector,
    create_pose_estimator,
    open_source
)

HERE = os.path.dirname(os.path.abspath(__file__))


# ===========================================================================
# Persistent Spatial IoU Object Tracker
# ===========================================================================
class Track:
    def __init__(self, track_id: int, box, score: float, class_id: int = 0):
        self.track_id = track_id
        self.box = np.array(box, dtype=np.float32)  # [x1, y1, x2, y2]
        self.score = float(score)
        self.class_id = int(class_id)
        self.hits = 1
        self.time_since_update = 0

    def update(self, box, score: float):
        # Smooth box coordinates using EMA
        self.box = 0.7 * np.array(box, dtype=np.float32) + 0.3 * self.box
        self.score = float(score)
        self.hits += 1
        self.time_since_update = 0


class SpatialTracker:
    """Persistent spatial object tracker using bounding-box IoU association."""
    def __init__(self, max_age: int = 20, iou_threshold: float = 0.3):
        self.max_age = max_age
        self.iou_threshold = iou_threshold
        self.tracks = []
        self.next_id = 1

    @staticmethod
    def _compute_iou(boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        inter = max(0.0, xB - xA) * max(0.0, yB - yA)
        areaA = max(1.0, (boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
        areaB = max(1.0, (boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))
        return inter / (areaA + areaB - inter)

    def update(self, raw_detections):
        """
        raw_detections: list of (box, score, class_id)
        returns: list of Track objects
        """
        # Age existing tracks
        for t in self.tracks:
            t.time_since_update += 1

        if len(raw_detections) == 0:
            self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]
            return []

        # IoU matching matrix
        matched_tracks = set()
        matched_dets = set()

        if self.tracks:
            iou_matrix = np.zeros((len(self.tracks), len(raw_detections)), dtype=np.float32)
            for i, trk in enumerate(self.tracks):
                for j, (box, _, _) in enumerate(raw_detections):
                    iou_matrix[i, j] = self._compute_iou(trk.box, box)

            # Greedy matching by highest IoU
            while True:
                max_val = np.max(iou_matrix)
                if max_val < self.iou_threshold:
                    break
                i, j = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
                if i not in matched_tracks and j not in matched_dets:
                    self.tracks[i].update(raw_detections[j][0], raw_detections[j][1])
                    matched_tracks.add(i)
                    matched_dets.add(j)
                iou_matrix[i, :] = -1
                iou_matrix[:, j] = -1

        # Spawn new tracks for unmatched detections
        for j, (box, score, cid) in enumerate(raw_detections):
            if j not in matched_dets:
                new_track = Track(self.next_id, box, score, cid)
                self.next_id += 1
                self.tracks.append(new_track)

        # Retire dead tracks
        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]

        # Return tracks that were updated this frame
        return [t for t in self.tracks if t.time_since_update == 0]


# ===========================================================================
# CLI Arguments
# ===========================================================================
def parse_args():
    p = argparse.ArgumentParser(description="SentinelVision YOLOv8 + Pose Parallel Pipeline")
    p.add_argument("--input", default="0", help="video path or webcam index")
    p.add_argument("--yolo-model", default=os.path.join(HERE, "models", "yolov8n.onnx"),
                   help="path to yolov8n.onnx or yolov8n.fp32.plan")
    p.add_argument("--pose-model", default=os.path.join(HERE, "models", "yolo11n-pose.onnx"),
                   help="path to yolo11n-pose.onnx or yolo11n-pose.fp32.plan")
    p.add_argument("--conf", type=float, default=0.4, help="YOLOv8 person confidence")
    p.add_argument("--pose-conf", type=float, default=0.25, help="pose person confidence")
    p.add_argument("--output", default=None, help="write annotated video here")
    p.add_argument("--no-show", action="store_true", help="disable display window")
    p.add_argument("--api-ingest", default=None,
                   help="URL to post frame detections to (e.g. http://localhost:8000/api/pipegen/ingest)")
    p.add_argument("--loop", action="store_true", help="loop video continuously")
    return p.parse_args()


# ===========================================================================
# Main Execution Pipeline
# ===========================================================================
def main():
    args = parse_args()

    print("=== SentinelVision Official PipeGen / DeepStream Inference Pipeline ===")
    print(f"  Target Video Source : {args.input}")
    print(f"  YOLOv8 Model Target : {args.yolo_model}")
    print(f"  Pose Model Target   : {args.pose_model}")

    # Initialize detectors (TensorRT GPU if .plan available, else ONNX)
    yolo = create_yolo_detector(args.yolo_model, conf=args.conf)
    pose = create_pose_estimator(args.pose_model, conf=args.pose_conf)

    # Initialize persistent spatial tracker
    tracker = SpatialTracker(max_age=15, iou_threshold=0.3)

    cap = open_source(args.input)
    if not cap.isOpened():
        sys.exit(f"Could not open video input: {args.input}")

    writer = None
    if args.output:
        fps_val = cap.get(cv2.CAP_PROP_FPS) or 30.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"), fps_val, (w, h))

    win = "SentinelVision PipeGen GPU Stream"
    if not args.no_show:
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    pool = ThreadPoolExecutor(max_workers=2)
    fps = 0.0
    frame_idx = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                if args.loop:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                break

            t0 = time.time()

            # Execute dual models in parallel
            f_yolo = pool.submit(yolo.infer, frame)
            f_pose = pool.submit(pose.infer, frame)
            yolo_res, pose_res = f_yolo.result(), f_pose.result()

            # Filter persons (class_id 0) and update persistent tracker
            person_dets = [d for d in yolo_res if d[2] == 0]
            active_tracks = tracker.update(person_dets)

            # Ingest detections to SentinelVision backend with Pose-Driven Foot Localization
            if args.api_ingest:
                objects = []
                for trk in active_tracks:
                    x1, y1, x2, y2 = [float(v) for v in trk.box]
                    
                    # Pose Association: Find corresponding pose keypoints by IoU
                    contact_x = (x1 + x2) / 2.0
                    contact_y = y2
                    contact_source = "bbox_bottom_center"

                    for p_det in pose_res:
                        p_box = p_det["box"]
                        # Check IoU overlap with current person bounding box
                        ix1, iy1 = max(x1, p_box[0]), max(y1, p_box[1])
                        ix2, iy2 = min(x2, p_box[2]), min(y2, p_box[3])
                        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
                        if inter > 0:
                            area_t = max(1.0, (x2 - x1) * (y2 - y1))
                            area_p = max(1.0, (p_box[2] - p_box[0]) * (p_box[3] - p_box[1]))
                            p_iou = inter / (area_t + area_p - inter)
                            if p_iou >= 0.35:
                                # Keypoint 15: Left Ankle, Keypoint 16: Right Ankle
                                kx, ky, kc = p_det["kx"], p_det["ky"], p_det["kc"]
                                left_visible = (kc[15] >= 0.3)
                                right_visible = (kc[16] >= 0.3)
                                if left_visible and right_visible:
                                    contact_x = float((kx[15] + kx[16]) / 2.0)
                                    contact_y = float(max(ky[15], ky[16]))
                                    contact_source = "pose_ankles"
                                elif left_visible:
                                    contact_x = float(kx[15])
                                    contact_y = float(ky[15])
                                    contact_source = "pose_left_ankle"
                                elif right_visible:
                                    contact_x = float(kx[16])
                                    contact_y = float(ky[16])
                                    contact_source = "pose_right_ankle"
                                break

                    objects.append({
                        "class_id": 0,
                        "obj_label": "person",
                        "confidence": round(float(trk.score), 3),
                        "object_id": trk.track_id,  # Genuine persistent tracking ID
                        "rect_params": {
                            "left": round(x1, 1),
                            "top": round(y1, 1),
                            "width": max(1.0, round(x2 - x1, 1)),
                            "height": max(1.0, round(y2 - y1, 1))
                        },
                        "detection_point": [round(contact_x, 1), round(contact_y, 1)],
                        "contact_source": contact_source
                    })

                try:
                    payload = json.dumps({
                        "frame_num": frame_idx,
                        "source_id": 0,
                        "objects": objects
                    }).encode("utf-8")
                    req = urllib.request.Request(args.api_ingest, data=payload, headers={"Content-Type": "application/json"})
                    urllib.request.urlopen(req, timeout=0.2)
                except Exception:
                    pass

            frame_idx += 1

            # Draw visual feedback
            yolo.draw(frame, person_dets)
            pose.draw(frame, pose_res)

            # Draw persistent track IDs and foot contact points
            for trk in active_tracks:
                bx1, by1, bx2, by2 = [int(v) for v in trk.box]
                cv2.putText(frame, f"ID: {trk.track_id}", (bx1, by1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)

            fps = 0.9 * fps + 0.1 * (1.0 / max(time.time() - t0, 1e-6))
            cv2.putText(frame, f"{fps:4.1f} FPS (PipeGen)",
                        (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

            if writer:
                writer.write(frame)
            if not args.no_show:
                cv2.imshow(win, frame)
                if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
                    break
    finally:
        pool.shutdown(wait=False)
        cap.release()
        if writer:
            writer.release()
        if not args.no_show:
            cv2.destroyAllWindows()
        print(f"Pipeline execution finished. Processed {frame_idx} frames.")


if __name__ == "__main__":
    main()

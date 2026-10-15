#!/usr/bin/env python3
"""YOLOv8 object detection + YOLO11-pose estimation, run IN PARALLEL per frame.

Two ONNX models run concurrently on each frame (separate threads), then the
detection boxes and the pose skeletons are drawn on the same frame.

    ./run.sh                                  # default webcam
    python yolov8_pose.py --input people.mp4 --output out.mp4
"""

import argparse
import os
import sys
import time
import json
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import cv2

from detectors import YoloV8Detector, PoseEstimator, open_source

HERE = os.path.dirname(os.path.abspath(__file__))


def parse_args():
    p = argparse.ArgumentParser(description="YOLOv8 + YOLO11-pose in parallel")
    p.add_argument("--input", default="0", help="video path or webcam index")
    p.add_argument("--yolo-model", default=os.path.join(HERE, "models", "yolov8n.onnx"))
    p.add_argument("--pose-model", default=os.path.join(HERE, "models", "yolo11n-pose.onnx"))
    p.add_argument("--conf", type=float, default=0.4, help="YOLOv8 confidence")
    p.add_argument("--pose-conf", type=float, default=0.25, help="pose person confidence")
    p.add_argument("--output", default=None, help="write annotated video here")
    p.add_argument("--no-show", action="store_true")
    p.add_argument("--api-ingest", default=None, help="URL to post frame detections to (e.g. http://localhost:8000/api/pipegen/ingest)")
    p.add_argument("--loop", action="store_true", help="loop video continuously")
    return p.parse_args()


def main():
    args = parse_args()

    print("pipeline: yolov8 detection + yolo11-pose estimation (parallel)")
    print(f"  model 1 (yolov8): {args.yolo_model}")
    print(f"  model 2 (pose)  : {args.pose_model}")
    yolo = YoloV8Detector(args.yolo_model, conf=args.conf)
    pose = PoseEstimator(args.pose_model, conf=args.pose_conf)

    cap = open_source(args.input)
    if not cap.isOpened():
        sys.exit(f"could not open input: {args.input}")

    writer = None
    if args.output:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    win = "YOLOv8 + Pose (parallel)"
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
            f_yolo = pool.submit(yolo.infer, frame)
            f_pose = pool.submit(pose.infer, frame)
            yolo_res, pose_res = f_yolo.result(), f_pose.result()

            # Ingest detections to SentinelVision backend if configured
            if args.api_ingest:
                objects = []
                for i, (box, score, cid) in enumerate(yolo_res):
                    if cid == 0:  # person detection
                        x1, y1, x2, y2 = [float(v) for v in box]
                        objects.append({
                            "class_id": 0,
                            "obj_label": "person",
                            "confidence": round(float(score), 3),
                            "object_id": i + 1,
                            "rect_params": {
                                "left": x1,
                                "top": y1,
                                "width": max(1.0, x2 - x1),
                                "height": max(1.0, y2 - y1)
                            }
                        })
                try:
                    payload = json.dumps({"frame_num": frame_idx, "source_id": 0, "objects": objects}).encode("utf-8")
                    req = urllib.request.Request(args.api_ingest, data=payload, headers={"Content-Type": "application/json"})
                    urllib.request.urlopen(req, timeout=0.2)
                except Exception:
                    pass

            frame_idx += 1

            yolo.draw(frame, yolo_res)     # boxes first
            pose.draw(frame, pose_res)     # skeletons on top

            fps = 0.9 * fps + 0.1 * (1.0 / max(time.time() - t0, 1e-6))
            cv2.putText(frame, f"{fps:4.1f} FPS",
                        (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

            if writer:
                writer.write(frame)
            if not args.no_show:
                cv2.imshow(win, frame)
                if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
                    break
    finally:
        pool.shutdown()
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

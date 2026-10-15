"""
SentinelVision - PipeGen Metadata Adapter
CraftifAI Physical AI Hackathon | Track A: PipeGen
"""

import json
from typing import List, Dict, Any, Union, Optional
from .models import Detection, SafetyIncident
from .zone_monitor import ZoneMonitor


class PipeGenAdapter:
    """
    Adapter between CraftifAI PipeGen / DeepStream output format and SentinelVision's
    safety incident schema.
    
    Supports:
    1. DeepStream NvDsObjectMeta JSON export (e.g. from nvmsgconv schema or probe hooks).
    2. PipeGen pipeline IR detection messages.
    3. Standard COCO/YOLO detection dicts.
    """

    def __init__(self, zone_monitor: Optional[ZoneMonitor] = None, source_name: str = "sample_video"):
        self.source_name = source_name
        self.zone_monitor = zone_monitor or ZoneMonitor(source_name=source_name)

    def parse_pipegen_detection(self, raw_obj: Dict[str, Any]) -> Optional[Detection]:
        """
        Parses a single detected object from PipeGen/DeepStream metadata into a Detection model.
        Handles both [left, top, width, height] and [xmin, ymin, xmax, ymax].
        """
        # 1. Check DeepStream style: rect_params: {left, top, width, height}
        if "rect_params" in raw_obj:
            rp = raw_obj["rect_params"]
            left = float(rp.get("left", 0))
            top = float(rp.get("top", 0))
            width = float(rp.get("width", 0))
            height = float(rp.get("height", 0))
            bbox = [left, top, left + width, top + height]
            conf = float(raw_obj.get("confidence", 0.0))
            track_id = raw_obj.get("object_id")
            class_id = int(raw_obj.get("class_id", 0))
            class_name = raw_obj.get("obj_label", "person")

        # 2. Check PipeGen standard message schema
        elif "bbox" in raw_obj or "bounding_box" in raw_obj:
            raw_box = raw_obj.get("bbox") or raw_obj.get("bounding_box")
            if isinstance(raw_box, dict):
                # {xmin, ymin, xmax, ymax} or {x, y, w, h}
                if "xmin" in raw_box:
                    bbox = [float(raw_box["xmin"]), float(raw_box["ymin"]), float(raw_box["xmax"]), float(raw_box["ymax"])]
                else:
                    x = float(raw_box.get("x", 0))
                    y = float(raw_box.get("y", 0))
                    w = float(raw_box.get("w", 0))
                    h = float(raw_box.get("h", 0))
                    bbox = [x, y, x + w, y + h]
            elif isinstance(raw_box, (list, tuple)) and len(raw_box) == 4:
                bbox = [float(c) for c in raw_box]
            else:
                return None

            conf = float(raw_obj.get("confidence", raw_obj.get("score", 0.0)))
            track_id = raw_obj.get("track_id", raw_obj.get("id"))
            class_id = int(raw_obj.get("class_id", 0))
            class_name = raw_obj.get("class_name", raw_obj.get("label", "person"))

        # 3. Direct coordinates [xmin, ymin, xmax, ymax]
        elif all(k in raw_obj for k in ("xmin", "ymin", "xmax", "ymax")):
            bbox = [float(raw_obj["xmin"]), float(raw_obj["ymin"]), float(raw_obj["xmax"]), float(raw_obj["ymax"])]
            conf = float(raw_obj.get("confidence", 0.0))
            track_id = raw_obj.get("track_id")
            class_id = int(raw_obj.get("class_id", 0))
            class_name = raw_obj.get("class_name", "person")
        else:
            return None

        # Filter: only evaluate persons (class_id 0 or class_name 'person')
        if class_name.lower() not in ("person", "pedestrian", "worker", "human") and class_id != 0:
            return None

        det_point = raw_obj.get("detection_point")
        contact_source = raw_obj.get("contact_source", "bbox_bottom_center")

        return Detection(
            track_id=int(track_id) if track_id is not None else None,
            class_id=class_id,
            class_name=class_name,
            confidence=max(0.0, min(1.0, conf)),
            bounding_box=bbox,
            detection_point=det_point,
            contact_source=contact_source,
            raw_metadata=raw_obj
        )

    def parse_frame_payload(self, payload: Union[str, Dict[str, Any]]) -> List[Detection]:
        """
        Parses a complete frame message payload from PipeGen / DeepStream.
        """
        if isinstance(payload, str):
            try:
                data = json.loads(payload)
            except json.JSONDecodeError:
                return []
        else:
            data = payload

        # Could be a list of objects or a dict containing 'objects' / 'detections'
        raw_list = []
        if isinstance(data, list):
            raw_list = data
        elif isinstance(data, dict):
            raw_list = data.get("objects", data.get("detections", data.get("predictions", [])))

        detections: List[Detection] = []
        for item in raw_list:
            if isinstance(item, dict):
                parsed = self.parse_pipegen_detection(item)
                if parsed:
                    detections.append(parsed)

        return detections

    def ingest_pipegen_frame(
        self,
        payload: Union[str, Dict[str, Any]],
        frame_timestamp: Optional[float] = None
    ) -> List[SafetyIncident]:
        """
        Ingests a frame message from PipeGen, converts to standard Detection objects,
        and triggers restricted zone evaluation.
        Returns any newly generated SafetyIncident objects.
        """
        detections = self.parse_frame_payload(payload)
        return self.zone_monitor.process_detections(detections, frame_timestamp=frame_timestamp)

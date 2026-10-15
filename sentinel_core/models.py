"""
SentinelVision - Core Data Models
CraftifAI Physical AI Hackathon | Track A: PipeGen
"""

from typing import List, Tuple, Dict, Any, Optional
from pydantic import BaseModel, Field
from datetime import datetime, timezone


class BoundingBox(BaseModel):
    """
    Normalized or pixel coordinates [xmin, ymin, xmax, ymax]
    """
    xmin: float
    ymin: float
    xmax: float
    ymax: float

    def to_list(self) -> List[float]:
        return [self.xmin, self.ymin, self.xmax, self.ymax]

    @classmethod
    def from_list(cls, coords: List[float]) -> "BoundingBox":
        if len(coords) != 4:
            raise ValueError("Bounding box list must contain exactly 4 coordinates [xmin, ymin, xmax, ymax]")
        return cls(xmin=coords[0], ymin=coords[1], xmax=coords[2], ymax=coords[3])

    @property
    def bottom_center(self) -> Tuple[float, float]:
        """
        Bottom-center point (x, y) representing the ground contact/foot position of the person.
        Critical for floor-plane restricted zone intrusion accuracy.
        """
        return ((self.xmin + self.xmax) / 2.0, self.ymax)

    @property
    def centroid(self) -> Tuple[float, float]:
        """Center coordinate of the bounding box."""
        return ((self.xmin + self.xmax) / 2.0, (self.ymin + self.ymax) / 2.0)


class Detection(BaseModel):
    """Individual person detection output from PipeGen detector."""
    track_id: Optional[int] = None
    class_id: int = 0  # 0 = person
    class_name: str = "person"
    confidence: float = Field(ge=0.0, le=1.0)
    bounding_box: List[float] = Field(description="[xmin, ymin, xmax, ymax]")
    detection_point: Optional[List[float]] = Field(default=None, description="[x, y] ground foot contact coordinate")
    contact_source: str = Field(default="bbox_bottom_center", description="Method used: pose_ankles or bbox_bottom_center")
    raw_metadata: Optional[Dict[str, Any]] = None


class RestrictedZone(BaseModel):
    """
    Configurable restricted zone definition.
    Supports arbitrary 2D polygon vertices [[x1, y1], [x2, y2], ...] or rectangular bounds.
    """
    zone_id: str
    name: str
    polygon: List[List[float]] = Field(description="Ordered list of [x, y] coordinates forming polygon")
    color: str = "#EF4444"  # Default red alert color
    risk_level: str = "HIGH"  # CRITICAL, HIGH, MEDIUM, LOW
    is_active: bool = True
    cooldown_seconds: float = Field(
        default=3.0,
        description="Cooldown period in seconds before a re-trigger is permitted for the same track ID"
    )


class SafetyIncident(BaseModel):
    """
    Standardized safety incident event complying with Hackathon Track A requirements:
    {
      "event_type": "restricted_zone_intrusion",
      "timestamp": "ISO-8601 timestamp",
      "zone_id": "restricted_zone_1",
      "confidence": 0.0,
      "bounding_box": [0, 0, 0, 0],
      "source": "sample_video"
    }
    """
    event_type: str = "restricted_zone_intrusion"
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    zone_id: str
    zone_name: Optional[str] = None
    track_id: Optional[int] = None
    confidence: float = Field(ge=0.0, le=1.0)
    bounding_box: List[float] = Field(description="[xmin, ymin, xmax, ymax]")
    detection_point: Optional[List[float]] = Field(default=None, description="[x, y] foot contact point")
    contact_source: str = Field(default="bbox_bottom_center", description="Source of contact point (pose_ankles or bbox_bottom_center)")
    source: str = "sample_video"
    risk_level: str = "HIGH"
    dwell_time_seconds: float = 0.0
    pipegen_pipeline_id: Optional[str] = "pipegen_person_det_v1"
    raw_pipegen_meta: Optional[Dict[str, Any]] = None


class PipelineStatus(BaseModel):
    """Current runtime status of PipeGen and SentinelVision."""
    pipeline_id: str = "pipegen_person_track_a"
    pipegen_installed: bool = True
    pipegen_version: str = "0.5.1"
    container_ready: bool = False
    active_model: str = "yolov8s_person_fp16"
    trt_engine_loaded: bool = False
    gpu_device: str = "NVIDIA GeForce RTX 2050"
    gpu_driver: str = "592.82"
    cuda_version: str = "13.1"
    current_fps: float = 0.0
    inference_latency_ms: float = 0.0
    mode: str = "live"  # "live" | "demo"
    total_frames_processed: int = 0
    total_detections: int = 0
    total_incidents: int = 0

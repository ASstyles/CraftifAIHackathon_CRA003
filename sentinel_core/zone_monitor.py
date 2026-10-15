"""
SentinelVision - Workplace Safety Restricted Zone Monitor
CraftifAI Physical AI Hackathon | Track A: PipeGen
"""

import time
from datetime import datetime, timezone
from typing import List, Dict, Optional, Tuple, Any
from .models import RestrictedZone, Detection, SafetyIncident, BoundingBox


def point_in_polygon(point: Tuple[float, float], polygon: List[List[float]]) -> bool:
    """
    Ray-casting algorithm to determine if a point (x, y) is inside a polygon.
    Handles arbitrary convex and concave polygons.
    """
    x, y = point
    n = len(polygon)
    inside = False

    if n < 3:
        return False

    p1x, p1y = polygon[0]
    for i in range(1, n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        x_inters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= x_inters:
                        inside = not inside
        p1x, p1y = p2x, p2y

    return inside


class TrackState:
    """Tracks state of an individual entity within or near restricted zones."""
    def __init__(self, track_id: int):
        self.track_id = track_id
        self.active_zones: Dict[str, float] = {}  # zone_id -> entry_time (timestamp)
        self.last_incident_time: Dict[str, float] = {}  # zone_id -> last_incident_time (timestamp)
        self.last_seen: float = time.time()
        self.last_bbox: Optional[List[float]] = None


class ZoneMonitor:
    """
    Monitors person detections against configurable restricted zones.
    Provides stateful tracking, intrusion detection, duplicate event suppression,
    and structured JSON incident event generation.
    """

    def __init__(self, source_name: str = "sample_video", load_defaults: bool = True):
        self.source_name = source_name
        self.zones: Dict[str, RestrictedZone] = {}
        self.tracks: Dict[int, TrackState] = {}
        self.incident_history: List[SafetyIncident] = []
        self.max_history: int = 500
        self.track_timeout_seconds: float = 15.0
        self.next_untracked_id: int = 1000

        # Load default industrial zones if requested
        if load_defaults:
            self._initialize_default_zones()

    def clear_zones(self):
        """Clear all active and inactive zones."""
        self.zones.clear()

    def _initialize_default_zones(self):
        """Create sample realistic industrial restricted zones (pixel coords 0-1280, 0-720)."""
        zone_1 = RestrictedZone(
            zone_id="restricted_zone_1",
            name="Robotic Arm Perimeter",
            polygon=[
                [450.0, 300.0],
                [850.0, 300.0],
                [900.0, 680.0],
                [400.0, 680.0]
            ],
            color="#EF4444",
            risk_level="CRITICAL",
            is_active=True,
            cooldown_seconds=3.0
        )
        zone_2 = RestrictedZone(
            zone_id="restricted_zone_2",
            name="High-Voltage Electrical Enclosure",
            polygon=[
                [50.0, 200.0],
                [320.0, 200.0],
                [320.0, 550.0],
                [50.0, 550.0]
            ],
            color="#F59E0B",
            risk_level="HIGH",
            is_active=True,
            cooldown_seconds=4.0
        )
        self.add_zone(zone_1)
        self.add_zone(zone_2)

    def add_zone(self, zone: RestrictedZone):
        """Add or update a restricted zone."""
        self.zones[zone.zone_id] = zone

    def remove_zone(self, zone_id: str) -> bool:
        """Remove a zone by ID."""
        if zone_id in self.zones:
            del self.zones[zone_id]
            return True
        return False

    def get_zones(self) -> List[RestrictedZone]:
        """Return all active and inactive zones."""
        return list(self.zones.values())

    def process_detections(
        self,
        detections: List[Detection],
        frame_timestamp: Optional[float] = None
    ) -> List[SafetyIncident]:
        """
        Process a batch of person detections for a single video frame.
        Evaluates intrusions, updates track state, applies duplicate suppression,
        and returns any newly triggered safety incidents.
        """
        current_time = frame_timestamp if frame_timestamp is not None else time.time()
        self._prune_stale_tracks(current_time)

        new_incidents: List[SafetyIncident] = []

        for det in detections:
            # Assign temporary track ID if detector does not have tracking enabled
            if det.track_id is None:
                track_id = self._match_or_assign_track(det, current_time)
            else:
                track_id = det.track_id

            if track_id not in self.tracks:
                self.tracks[track_id] = TrackState(track_id)

            track = self.tracks[track_id]
            track.last_seen = current_time
            track.last_bbox = det.bounding_box

            bbox = BoundingBox.from_list(det.bounding_box)
            # Contact point: bottom-center of bounding box represents feet on ground
            foot_x, foot_y = bbox.bottom_center

            # Check every active zone
            for zone_id, zone in self.zones.items():
                if not zone.is_active:
                    continue

                is_inside = point_in_polygon((foot_x, foot_y), zone.polygon)

                if is_inside:
                    if zone_id not in track.active_zones:
                        # NEW ENTRY INTO ZONE
                        track.active_zones[zone_id] = current_time
                        dwell_time = 0.0
                        trigger_incident = True
                    else:
                        # ALREADY INSIDE ZONE
                        dwell_time = current_time - track.active_zones[zone_id]
                        last_alert = track.last_incident_time.get(zone_id, 0.0)
                        
                        # DUPLICATE SUPPRESSION LOGIC:
                        # Only re-trigger if cooldown period has elapsed
                        if current_time - last_alert >= zone.cooldown_seconds:
                            trigger_incident = True
                        else:
                            trigger_incident = False

                    if trigger_incident:
                        track.last_incident_time[zone_id] = current_time
                        incident = SafetyIncident(
                            event_type="restricted_zone_intrusion",
                            timestamp=datetime.fromtimestamp(current_time, timezone.utc).isoformat(),
                            zone_id=zone.zone_id,
                            zone_name=zone.name,
                            track_id=track_id,
                            confidence=round(det.confidence, 4),
                            bounding_box=[round(c, 2) for c in det.bounding_box],
                            detection_point=[round(foot_x, 2), round(foot_y, 2)],
                            source=self.source_name,
                            risk_level=zone.risk_level,
                            dwell_time_seconds=round(dwell_time, 2),
                            raw_pipegen_meta=det.raw_metadata
                        )
                        new_incidents.append(incident)
                        self.incident_history.insert(0, incident)
                        if len(self.incident_history) > self.max_history:
                            self.incident_history.pop()

                else:
                    # Person is outside this zone - if they were previously inside, mark exited
                    if zone_id in track.active_zones:
                        del track.active_zones[zone_id]

        return new_incidents

    def _match_or_assign_track(self, det: Detection, current_time: float) -> int:
        """Simple spatial centroid proximity tracker for untracked detections."""
        bbox = BoundingBox.from_list(det.bounding_box)
        cx, cy = bbox.centroid

        best_track_id = None
        min_dist = float("inf")

        for tid, tstate in self.tracks.items():
            if tstate.last_bbox and (current_time - tstate.last_seen < 1.0):
                prev_bbox = BoundingBox.from_list(tstate.last_bbox)
                pcx, pcy = prev_bbox.centroid
                dist = ((cx - pcx)**2 + (cy - pcy)**2)**0.5
                if dist < 80.0 and dist < min_dist:
                    min_dist = dist
                    best_track_id = tid

        if best_track_id is not None:
            return best_track_id

        assigned = self.next_untracked_id
        self.next_untracked_id += 1
        return assigned

    def _prune_stale_tracks(self, current_time: float):
        """Remove tracks that have disappeared for longer than timeout."""
        stale = [
            tid for tid, state in self.tracks.items()
            if current_time - state.last_seen > self.track_timeout_seconds
        ]
        for tid in stale:
            del self.tracks[tid]

    def get_recent_incidents(self, limit: int = 50) -> List[SafetyIncident]:
        """Return the most recent safety incidents."""
        return self.incident_history[:limit]

    def get_incident_counts(self) -> Dict[str, Any]:
        """Return aggregate statistics for safety incidents."""
        total = len(self.incident_history)
        by_zone: Dict[str, int] = {}
        by_risk: Dict[str, int] = {}

        for inc in self.incident_history:
            by_zone[inc.zone_id] = by_zone.get(inc.zone_id, 0) + 1
            by_risk[inc.risk_level] = by_risk.get(inc.risk_level, 0) + 1

        active_intrusions = 0
        for tstate in self.tracks.values():
            active_intrusions += len(tstate.active_zones)

        return {
            "total_incidents": total,
            "active_intrusions": active_intrusions,
            "by_zone": by_zone,
            "by_risk": by_risk
        }

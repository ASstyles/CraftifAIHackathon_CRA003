"""
Unit Tests for CraftifAI PipeGen Metadata Adapter
CraftifAI Physical AI Hackathon | Track A: PipeGen
"""

from sentinel_core.pipegen_adapter import PipeGenAdapter
from sentinel_core.zone_monitor import ZoneMonitor
from sentinel_core.models import RestrictedZone


def test_parse_deepstream_nvds_metadata():
    """Verify parsing DeepStream rect_params format into standard SentinelVision detection."""
    adapter = PipeGenAdapter()

    raw_nvds_meta = {
        "class_id": 0,
        "obj_label": "person",
        "confidence": 0.892,
        "object_id": 7,
        "rect_params": {
            "left": 150.0,
            "top": 200.0,
            "width": 80.0,
            "height": 190.0
        }
    }

    det = adapter.parse_pipegen_detection(raw_nvds_meta)
    assert det is not None
    assert det.track_id == 7
    assert det.class_id == 0
    assert det.class_name == "person"
    assert det.confidence == 0.892
    assert det.bounding_box == [150.0, 200.0, 230.0, 390.0]  # left, top, left+width, top+height


def test_filter_non_person_classes():
    """Verify that vehicles or non-person classes are ignored."""
    adapter = PipeGenAdapter()

    car_meta = {
        "class_id": 2,
        "obj_label": "car",
        "confidence": 0.95,
        "rect_params": {"left": 10, "top": 10, "width": 100, "height": 100}
    }
    assert adapter.parse_pipegen_detection(car_meta) is None


def test_pipegen_frame_ingestion_end_to_end():
    """Verify full ingestion of a PipeGen frame payload triggering an incident."""
    monitor = ZoneMonitor(load_defaults=False)
    zone = RestrictedZone(
        zone_id="test_ingest_zone",
        name="Ingest Zone",
        polygon=[[100.0, 100.0], [500.0, 100.0], [500.0, 500.0], [100.0, 500.0]],
        cooldown_seconds=3.0
    )
    monitor.add_zone(zone)
    adapter = PipeGenAdapter(zone_monitor=monitor)

    payload = {
        "frame_num": 142,
        "source_id": 0,
        "objects": [
            {
                "class_id": 0,
                "obj_label": "person",
                "confidence": 0.93,
                "object_id": 12,
                "rect_params": {
                    "left": 200.0,
                    "top": 200.0,
                    "width": 60.0,
                    "height": 150.0  # foot at (230, 350) -> inside zone
                }
            }
        ]
    }

    incidents = adapter.ingest_pipegen_frame(payload, frame_timestamp=500.0)
    assert len(incidents) == 1
    assert incidents[0].track_id == 12
    assert incidents[0].zone_id == "test_ingest_zone"

"""
Unit Tests for Incident JSON Serialization & Schema Compliance
CraftifAI Physical AI Hackathon | Track A: PipeGen
"""

import json
from datetime import datetime
from sentinel_core.models import SafetyIncident


def test_hackathon_json_schema_compliance():
    """
    Validates that SafetyIncident JSON serialization satisfies the exact Hackathon Track A schema:
    {
      "event_type": "restricted_zone_intrusion",
      "timestamp": "ISO-8601 timestamp",
      "zone_id": "restricted_zone_1",
      "confidence": 0.0,
      "bounding_box": [0, 0, 0, 0],
      "source": "sample_video"
    }
    """
    incident = SafetyIncident(
        event_type="restricted_zone_intrusion",
        timestamp="2026-10-15T15:30:00.000Z",
        zone_id="restricted_zone_1",
        confidence=0.9421,
        bounding_box=[120.5, 340.0, 240.2, 580.0],
        source="sample_video"
    )

    json_str = incident.model_dump_json()
    data = json.loads(json_str)

    # Required primary fields
    assert data["event_type"] == "restricted_zone_intrusion"
    assert data["zone_id"] == "restricted_zone_1"
    assert data["source"] == "sample_video"
    assert isinstance(data["confidence"], (int, float))
    assert 0.0 <= data["confidence"] <= 1.0
    assert isinstance(data["bounding_box"], list)
    assert len(data["bounding_box"]) == 4

    # Timestamp verification (valid ISO-8601)
    ts = data["timestamp"]
    assert ts.endswith("Z") or "+" in ts
    # Parses without error
    parsed_dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    assert parsed_dt.year == 2026


def test_json_roundtrip_serialization():
    """Verify serialization to JSON string and deserialization back into model."""
    original = SafetyIncident(
        zone_id="zone_assembly_line",
        confidence=0.88,
        bounding_box=[50.0, 60.0, 150.0, 200.0],
        source="rtsp://camera01.factory.internal"
    )
    raw_json = original.model_dump_json()
    reconstructed = SafetyIncident.model_validate_json(raw_json)

    assert reconstructed.zone_id == original.zone_id
    assert reconstructed.confidence == original.confidence
    assert reconstructed.bounding_box == original.bounding_box
    assert reconstructed.source == original.source

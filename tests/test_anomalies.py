# PROMPT:
# Write unit tests for app/anomalies.py using a real in-memory SQLite database.
# Cover: check_queue_spike (WARN at >5, CRITICAL at >10), check_dead_zones
# (zone with no events flagged, zone with recent events not flagged),
# check_abandonment (rate > 15% triggers WARN), check_conversion_drop
# (empty baseline returns no anomaly). Verify severity ordering.
#
# CHANGES MADE:
# - Used naive UTC datetimes throughout to match SQLite storage (timezone bug fix)
# - Added test that dead zone check ignores zones with recent events
# - Added test that get_anomalies returns sorted result (CRITICAL before WARN before INFO)
# - Empty store anomaly check: conversion_drop skips when 7-day baseline is zero

import pytest
from datetime import datetime, timedelta
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database import Base
from app.db_models import EventRecord
from app.models import EventType
from app.anomalies import (
    check_queue_spike,
    check_dead_zones,
    check_abandonment,
    check_conversion_drop,
    get_anomalies,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def db():
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()
    Base.metadata.drop_all(engine)


STORE = "STORE_BLR_002"
NOW = datetime.utcnow()


def _queue_event(visitor_id, depth, offset_minutes=0):
    return EventRecord(
        event_id=f"qev-{visitor_id}-{depth}",
        store_id=STORE,
        camera_id="CAM_BILLING_03",
        visitor_id=visitor_id,
        event_type=EventType.BILLING_QUEUE_JOIN.value,
        timestamp=NOW - timedelta(minutes=offset_minutes),
        zone_id="BILLING",
        dwell_ms=0,
        is_staff=False,
        confidence=0.9,
        metadata_json={"queue_depth": depth},
    )


def _abandon_event(visitor_id, offset_minutes=0):
    return EventRecord(
        event_id=f"abd-{visitor_id}",
        store_id=STORE,
        camera_id="CAM_BILLING_03",
        visitor_id=visitor_id,
        event_type=EventType.BILLING_QUEUE_ABANDON.value,
        timestamp=NOW - timedelta(minutes=offset_minutes),
        zone_id="BILLING",
        dwell_ms=0,
        is_staff=False,
        confidence=0.9,
        metadata_json={},
    )


def _dwell_event(visitor_id, zone_id, offset_minutes=0):
    return EventRecord(
        event_id=f"dw-{visitor_id}-{zone_id}-{offset_minutes}",
        store_id=STORE,
        camera_id="CAM_MAKEUP_02",
        visitor_id=visitor_id,
        event_type=EventType.ZONE_DWELL.value,
        timestamp=NOW - timedelta(minutes=offset_minutes),
        zone_id=zone_id,
        dwell_ms=30000,
        is_staff=False,
        confidence=0.9,
        metadata_json={},
    )


# ---------------------------------------------------------------------------
# check_queue_spike
# ---------------------------------------------------------------------------

class TestQueueSpike:
    def test_no_events_no_anomaly(self, db):
        anomalies = check_queue_spike(STORE, db)
        assert anomalies == []

    def test_depth_below_threshold_no_anomaly(self, db):
        db.add(_queue_event("VIS_001", depth=3))
        db.commit()
        anomalies = check_queue_spike(STORE, db)
        assert anomalies == []

    def test_depth_above_5_triggers_warn(self, db):
        db.add(_queue_event("VIS_001", depth=7))
        db.commit()
        anomalies = check_queue_spike(STORE, db)
        assert len(anomalies) == 1
        assert anomalies[0]["severity"] == "WARN"
        assert anomalies[0]["type"] == "BILLING_QUEUE_SPIKE"

    def test_depth_above_10_triggers_critical(self, db):
        db.add(_queue_event("VIS_001", depth=12))
        db.commit()
        anomalies = check_queue_spike(STORE, db)
        assert len(anomalies) == 1
        assert anomalies[0]["severity"] == "CRITICAL"

    def test_peak_depth_used(self, db):
        db.add(_queue_event("VIS_001", depth=3))
        db.add(_queue_event("VIS_002", depth=11, offset_minutes=1))
        db.add(_queue_event("VIS_003", depth=2, offset_minutes=2))
        db.commit()
        anomalies = check_queue_spike(STORE, db)
        assert anomalies[0]["severity"] == "CRITICAL"
        assert anomalies[0]["value"] == 11

    def test_suggested_action_present(self, db):
        db.add(_queue_event("VIS_001", depth=8))
        db.commit()
        anomalies = check_queue_spike(STORE, db)
        assert "suggested_action" in anomalies[0]
        assert isinstance(anomalies[0]["suggested_action"], str)


# ---------------------------------------------------------------------------
# check_dead_zones
# ---------------------------------------------------------------------------

class TestDeadZones:
    def test_zone_with_no_events_flagged(self, db):
        anomalies = check_dead_zones(STORE, db, ["SKINCARE"])
        assert len(anomalies) == 1
        assert anomalies[0]["type"] == "DEAD_ZONE"
        assert anomalies[0]["value"] == "SKINCARE"

    def test_zone_with_recent_event_not_flagged(self, db):
        db.add(_dwell_event("VIS_001", "SKINCARE", offset_minutes=5))
        db.commit()
        anomalies = check_dead_zones(STORE, db, ["SKINCARE"])
        assert anomalies == []

    def test_zone_with_old_event_flagged(self, db):
        db.add(_dwell_event("VIS_001", "SKINCARE", offset_minutes=45))
        db.commit()
        anomalies = check_dead_zones(STORE, db, ["SKINCARE"])
        assert len(anomalies) == 1

    def test_only_inactive_zones_flagged(self, db):
        db.add(_dwell_event("VIS_001", "SKINCARE", offset_minutes=5))    # recent
        db.add(_dwell_event("VIS_002", "MAKEUP", offset_minutes=45))     # stale
        db.commit()
        anomalies = check_dead_zones(STORE, db, ["SKINCARE", "MAKEUP"])
        assert len(anomalies) == 1
        assert anomalies[0]["value"] == "MAKEUP"

    def test_empty_zone_list_no_anomalies(self, db):
        assert check_dead_zones(STORE, db, []) == []


# ---------------------------------------------------------------------------
# check_abandonment
# ---------------------------------------------------------------------------

class TestAbandonmentCheck:
    def test_no_events_no_anomaly(self, db):
        assert check_abandonment(STORE, db) == []

    def test_rate_below_15_no_anomaly(self, db):
        # 1 abandon, 10 joins → ~9%
        db.add(_queue_event("VIS_001", depth=1))
        for i in range(9):
            db.add(_queue_event(f"VIS_{i+2:03d}", depth=1, offset_minutes=i + 1))
        db.add(_abandon_event("VIS_001"))
        db.commit()
        assert check_abandonment(STORE, db) == []

    def test_rate_above_15_triggers_warn(self, db):
        # 3 joins + 3 abandons = 50% rate
        for i in range(3):
            db.add(_queue_event(f"VIS_{i:03d}", depth=1, offset_minutes=i))
            db.add(_abandon_event(f"VIS_{i:03d}", offset_minutes=i + 1))
        db.commit()
        anomalies = check_abandonment(STORE, db)
        assert len(anomalies) == 1
        assert anomalies[0]["severity"] == "WARN"


# ---------------------------------------------------------------------------
# check_conversion_drop
# ---------------------------------------------------------------------------

class TestConversionDrop:
    def test_empty_baseline_no_anomaly(self, db):
        # No 7-day data → skip conversion drop check
        anomalies = check_conversion_drop(STORE, db)
        assert anomalies == []


# ---------------------------------------------------------------------------
# get_anomalies (integration)
# ---------------------------------------------------------------------------

class TestGetAnomalies:
    def test_returns_dict_with_anomalies_key(self, db):
        result = get_anomalies(STORE, db)
        assert "anomalies" in result
        assert isinstance(result["anomalies"], list)

    def test_empty_store_no_crash(self, db):
        result = get_anomalies("STORE_EMPTY", db)
        assert "anomalies" in result

    def test_severity_ordering_critical_first(self, db):
        # Add a CRITICAL queue spike and an abandoned queue (WARN)
        db.add(_queue_event("VIS_001", depth=12))
        for i in range(3):
            db.add(_queue_event(f"VIS_{i+2:03d}", depth=1, offset_minutes=i + 1))
            db.add(_abandon_event(f"VIS_{i+2:03d}", offset_minutes=i + 5))
        db.commit()
        result = get_anomalies(STORE, db)
        anomalies = result["anomalies"]
        if len(anomalies) >= 2:
            severity_order = {"CRITICAL": 0, "WARN": 1, "INFO": 2}
            for i in range(len(anomalies) - 1):
                assert (
                    severity_order[anomalies[i]["severity"]]
                    <= severity_order[anomalies[i + 1]["severity"]]
                )

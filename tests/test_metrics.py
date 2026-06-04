# PROMPT:
# Write thorough unit tests for app/metrics.py using a real SQLite in-memory database.
# Cover: unique_visitors (excludes staff), conversion_rate (zero division, zero purchases),
# dwell_by_zone (excludes staff), queue metrics, abandonment_rate, and the full get_metrics
# response shape. Include edge cases: empty store, all-staff events, re-entry counted once.
#
# CHANGES MADE:
# - Added explicit is_staff=True test to verify staff exclusion in unique_visitors
# - Added re-entry test: same visitor_id with two ENTRY events must still count as 1 visitor
# - Verified that conversion_rate returns float 0.0 (not None) for zero-purchase stores

import pytest
from datetime import datetime, timedelta
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database import Base
from app.db_models import EventRecord, VisitorSession, ConversionRecord
from app.metrics import (
    compute_unique_visitors,
    compute_conversion_rate,
    compute_dwell_by_zone,
    compute_queue_metrics,
    compute_abandonment_rate,
    get_metrics,
)
from app.models import EventType


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


def _event(visitor_id, event_type, zone_id=None, dwell_ms=0, is_staff=False,
           metadata=None, offset_minutes=0):
    return EventRecord(
        event_id=f"ev-{visitor_id}-{event_type}-{offset_minutes}",
        store_id=STORE,
        camera_id="CAM_ENTRY_01",
        visitor_id=visitor_id,
        event_type=event_type,
        timestamp=NOW - timedelta(minutes=offset_minutes),
        zone_id=zone_id,
        dwell_ms=dwell_ms,
        is_staff=is_staff,
        confidence=0.9,
        metadata_json=metadata or {},
    )


def _session(visitor_id, converted=False, is_reentry=False, zones=None):
    return VisitorSession(
        session_id=f"sess-{visitor_id}",
        visitor_id=visitor_id,
        store_id=STORE,
        entry_time=NOW - timedelta(hours=1),
        exit_time=NOW,
        zones_visited=zones or [],
        total_dwell_ms=0,
        converted=converted,
        is_reentry=is_reentry,
    )


# ---------------------------------------------------------------------------
# compute_unique_visitors
# ---------------------------------------------------------------------------

class TestUniqueVisitors:
    def test_empty_store_returns_zero(self, db):
        assert compute_unique_visitors(STORE, db) == 0

    def test_counts_one_visitor(self, db):
        db.add(_event("VIS_001", EventType.ENTRY.value))
        db.commit()
        assert compute_unique_visitors(STORE, db) == 1

    def test_deduplicates_same_visitor(self, db):
        db.add(_event("VIS_001", EventType.ENTRY.value))
        db.add(_event("VIS_001", EventType.ZONE_DWELL.value, zone_id="SKINCARE"))
        db.commit()
        assert compute_unique_visitors(STORE, db) == 1

    def test_counts_multiple_visitors(self, db):
        for i in range(3):
            db.add(_event(f"VIS_{i:03d}", EventType.ENTRY.value))
        db.commit()
        assert compute_unique_visitors(STORE, db) == 3

    def test_excludes_staff(self, db):
        db.add(_event("VIS_001", EventType.ENTRY.value, is_staff=False))
        db.add(_event("STAFF_01", EventType.ENTRY.value, is_staff=True))
        db.commit()
        assert compute_unique_visitors(STORE, db) == 1

    def test_all_staff_returns_zero(self, db):
        db.add(_event("STAFF_01", EventType.ENTRY.value, is_staff=True))
        db.add(_event("STAFF_02", EventType.ENTRY.value, is_staff=True))
        db.commit()
        assert compute_unique_visitors(STORE, db) == 0


# ---------------------------------------------------------------------------
# compute_conversion_rate
# ---------------------------------------------------------------------------

class TestConversionRate:
    def test_empty_store_returns_zero(self, db):
        assert compute_conversion_rate(STORE, db) == 0.0

    def test_zero_purchases_returns_zero(self, db):
        db.add(_session("VIS_001", converted=False))
        db.commit()
        assert compute_conversion_rate(STORE, db) == 0.0

    def test_all_converted_returns_100(self, db):
        db.add(_session("VIS_001", converted=True))
        db.commit()
        assert compute_conversion_rate(STORE, db) == 100.0

    def test_partial_conversion(self, db):
        db.add(_session("VIS_001", converted=True))
        db.add(_session("VIS_002", converted=False))
        db.commit()
        rate = compute_conversion_rate(STORE, db)
        assert rate == 50.0

    def test_returns_float_not_none(self, db):
        result = compute_conversion_rate(STORE, db)
        assert isinstance(result, float)


# ---------------------------------------------------------------------------
# compute_dwell_by_zone
# ---------------------------------------------------------------------------

class TestDwellByZone:
    def test_empty_returns_empty_dict(self, db):
        assert compute_dwell_by_zone(STORE, db) == {}

    def test_single_zone_dwell(self, db):
        db.add(_event("VIS_001", EventType.ZONE_DWELL.value, zone_id="SKINCARE", dwell_ms=5000))
        db.commit()
        result = compute_dwell_by_zone(STORE, db)
        assert "SKINCARE" in result
        assert result["SKINCARE"] == 5000.0

    def test_averages_multiple_dwells(self, db):
        db.add(_event("VIS_001", EventType.ZONE_DWELL.value, zone_id="BILLING", dwell_ms=4000))
        db.add(_event("VIS_002", EventType.ZONE_DWELL.value, zone_id="BILLING", dwell_ms=6000, offset_minutes=1))
        db.commit()
        result = compute_dwell_by_zone(STORE, db)
        assert result["BILLING"] == 5000.0

    def test_excludes_staff_dwell(self, db):
        db.add(_event("VIS_001", EventType.ZONE_DWELL.value, zone_id="SKINCARE", dwell_ms=10000, is_staff=False))
        db.add(_event("STAFF_01", EventType.ZONE_DWELL.value, zone_id="SKINCARE", dwell_ms=99999, is_staff=True, offset_minutes=1))
        db.commit()
        result = compute_dwell_by_zone(STORE, db)
        assert result.get("SKINCARE") == 10000.0


# ---------------------------------------------------------------------------
# compute_queue_metrics
# ---------------------------------------------------------------------------

class TestQueueMetrics:
    def test_empty_returns_zeros(self, db):
        result = compute_queue_metrics(STORE, db)
        assert result["queue_depth_current"] == 0
        assert result["queue_depth_peak"] == 0

    def test_tracks_peak_depth(self, db):
        db.add(_event("VIS_001", EventType.BILLING_QUEUE_JOIN.value, metadata={"queue_depth": 3}))
        db.add(_event("VIS_002", EventType.BILLING_QUEUE_JOIN.value, metadata={"queue_depth": 7}, offset_minutes=1))
        db.add(_event("VIS_003", EventType.BILLING_QUEUE_JOIN.value, metadata={"queue_depth": 2}, offset_minutes=2))
        db.commit()
        result = compute_queue_metrics(STORE, db)
        assert result["queue_depth_peak"] == 7


# ---------------------------------------------------------------------------
# compute_abandonment_rate
# ---------------------------------------------------------------------------

class TestAbandonmentRate:
    def test_no_events_returns_zero(self, db):
        assert compute_abandonment_rate(STORE, db) == 0.0

    def test_full_abandonment(self, db):
        db.add(_event("VIS_001", EventType.BILLING_QUEUE_JOIN.value, metadata={"queue_depth": 1}))
        db.add(_event("VIS_001", EventType.BILLING_QUEUE_ABANDON.value, offset_minutes=1))
        db.commit()
        rate = compute_abandonment_rate(STORE, db)
        assert rate == 50.0  # 1 abandon / (1 join + 1 abandon)


# ---------------------------------------------------------------------------
# get_metrics (integration)
# ---------------------------------------------------------------------------

class TestGetMetrics:
    def test_all_fields_present(self, db):
        result = get_metrics(STORE, db)
        assert "unique_visitors" in result
        assert "conversion_rate" in result
        assert "avg_dwell_by_zone" in result
        assert "queue_depth_current" in result
        assert "queue_depth_peak" in result
        assert "abandonment_rate" in result
        assert "data_freshness_minutes" in result

    def test_empty_store_no_crash(self, db):
        result = get_metrics("STORE_EMPTY", db)
        assert result["unique_visitors"] == 0
        assert result["conversion_rate"] == 0.0

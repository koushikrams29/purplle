"""
Store metrics computation service.
"""

from datetime import datetime, timedelta, timezone
from typing import Dict, Optional

from sqlalchemy import func
from sqlalchemy.orm import Session

from app.db_models import (
    EventRecord,
    VisitorSession,
)
from app.models import EventType


def _get_since(since: Optional[datetime] = None) -> datetime:
    """
    Default to last 24 hours if no timestamp provided.
    """
    return since or (datetime.now(timezone.utc) - timedelta(hours=24))


def compute_unique_visitors(
    store_id: str,
    db_session: Session,
    since: Optional[datetime] = None,
) -> int:
    """
    Count unique non-staff visitors.
    """
    since = _get_since(since)

    result = (
        db_session.query(
            func.count(func.distinct(EventRecord.visitor_id))
        )
        .filter(
            EventRecord.store_id == store_id,
            EventRecord.timestamp >= since,
            EventRecord.is_staff.is_(False),
        )
        .scalar()
    )

    return int(result or 0)


def compute_conversion_rate(
    store_id: str,
    db_session: Session,
    since: Optional[datetime] = None,
) -> float:
    """
    Conversion Rate =
    converted visitors / unique visitors * 100
    """
    since = _get_since(since)

    total_visitors = (
        db_session.query(VisitorSession)
        .filter(
            VisitorSession.store_id == store_id,
            VisitorSession.entry_time >= since,
        )
        .count()
    )

    if total_visitors == 0:
        return 0.0

    converted_visitors = (
        db_session.query(VisitorSession)
        .filter(
            VisitorSession.store_id == store_id,
            VisitorSession.entry_time >= since,
            VisitorSession.converted.is_(True),
        )
        .count()
    )

    return round(
        (converted_visitors / total_visitors) * 100,
        2,
    )


def compute_dwell_by_zone(
    store_id: str,
    db_session: Session,
    since: Optional[datetime] = None,
) -> Dict[str, float]:
    """
    Average dwell time (milliseconds) by zone.
    """
    since = _get_since(since)

    rows = (
        db_session.query(
            EventRecord.zone_id,
            func.avg(EventRecord.dwell_ms),
        )
        .filter(
            EventRecord.store_id == store_id,
            EventRecord.timestamp >= since,
            EventRecord.is_staff.is_(False),
            EventRecord.event_type == EventType.ZONE_DWELL.value,
        )
        .group_by(EventRecord.zone_id)
        .all()
    )

    return {
        zone: round(avg_dwell or 0.0, 2)
        for zone, avg_dwell in rows
        if zone
    }


def compute_queue_metrics(
    store_id: str,
    db_session: Session,
    since: Optional[datetime] = None,
) -> Dict[str, int]:
    """
    Compute queue depth metrics.
    """
    since = _get_since(since)

    queue_events = (
        db_session.query(EventRecord)
        .filter(
            EventRecord.store_id == store_id,
            EventRecord.timestamp >= since,
            EventRecord.event_type
            == EventType.BILLING_QUEUE_JOIN.value,
        )
        .all()
    )

    queue_depths = []

    for event in queue_events:
        metadata = event.metadata_json or {}

        depth = metadata.get("queue_depth")

        if depth is not None:
            queue_depths.append(depth)

    current_depth = queue_depths[-1] if queue_depths else 0
    peak_depth = max(queue_depths) if queue_depths else 0

    return {
        "queue_depth_current": current_depth,
        "queue_depth_peak": peak_depth,
    }


def compute_abandonment_rate(
    store_id: str,
    db_session: Session,
    since: Optional[datetime] = None,
) -> float:
    """
    Abandonment Rate =
    queue abandon events / total queue events * 100
    """
    since = _get_since(since)

    joins = (
        db_session.query(EventRecord)
        .filter(
            EventRecord.store_id == store_id,
            EventRecord.timestamp >= since,
            EventRecord.event_type
            == EventType.BILLING_QUEUE_JOIN.value,
        )
        .count()
    )

    abandons = (
        db_session.query(EventRecord)
        .filter(
            EventRecord.store_id == store_id,
            EventRecord.timestamp >= since,
            EventRecord.event_type
            == EventType.BILLING_QUEUE_ABANDON.value,
        )
        .count()
    )

    total = joins + abandons

    if total == 0:
        return 0.0

    return round(
        (abandons / total) * 100,
        2,
    )


def get_metrics(
    store_id: str,
    db_session: Session,
    since: Optional[datetime] = None,
) -> Dict:
    """
    Compute complete store metrics response.
    """
    since = _get_since(since)

    latest_event = (
        db_session.query(EventRecord)
        .filter(
            EventRecord.store_id == store_id
        )
        .order_by(EventRecord.timestamp.desc())
        .first()
    )

    freshness_minutes = 0

    if latest_event:
        event_time = latest_event.timestamp

        if event_time.tzinfo is not None:
            event_time = event_time.replace(tzinfo=None)

        freshness_minutes = int(
            (
                datetime.utcnow()
                - event_time
            ).total_seconds()
            / 60
        )

    queue_metrics = compute_queue_metrics(
        store_id,
        db_session,
        since,
    )

    return {
        "store_id": store_id,
        "timestamp": datetime.now(
            timezone.utc
        ).isoformat(),
        "unique_visitors": compute_unique_visitors(
            store_id,
            db_session,
            since,
        ),
        "conversion_rate": compute_conversion_rate(
            store_id,
            db_session,
            since,
        ),
        "avg_dwell_by_zone": compute_dwell_by_zone(
            store_id,
            db_session,
            since,
        ),
        "queue_depth_current": queue_metrics[
            "queue_depth_current"
        ],
        "queue_depth_peak": queue_metrics[
            "queue_depth_peak"
        ],
        "abandonment_rate": compute_abandonment_rate(
            store_id,
            db_session,
            since,
        ),
        "data_freshness_minutes": freshness_minutes,
    }
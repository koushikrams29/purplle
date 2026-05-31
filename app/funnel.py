"""
Conversion funnel analytics.
"""

from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

from sqlalchemy.orm import Session

from app.db_models import EventRecord, VisitorSession
from app.models import EventType


def _get_since(since: Optional[datetime] = None) -> datetime:
    """
    Default to last 24 hours.
    """
    return since or (
        datetime.now(timezone.utc) - timedelta(hours=24)
    )


def get_unique_visitors(
    store_id: str,
    db_session: Session,
    since: Optional[datetime] = None,
) -> List[str]:
    """
    Stage 1:
    Visitors with ENTRY events.
    """
    since = _get_since(since)

    rows = (
        db_session.query(EventRecord.visitor_id)
        .filter(
            EventRecord.store_id == store_id,
            EventRecord.timestamp >= since,
            EventRecord.is_staff.is_(False),
            EventRecord.event_type == EventType.ENTRY.value,
        )
        .distinct()
        .all()
    )

    return [r[0] for r in rows]


def get_zone_visitors(
    store_id: str,
    zone_id: Optional[str],
    db_session: Session,
    since: Optional[datetime] = None,
) -> List[str]:
    """
    Visitors who visited a zone.

    If zone_id=None:
    return visitors who visited any zone.
    """
    since = _get_since(since)

    query = (
        db_session.query(EventRecord.visitor_id)
        .filter(
            EventRecord.store_id == store_id,
            EventRecord.timestamp >= since,
            EventRecord.is_staff.is_(False),
            EventRecord.event_type == EventType.ZONE_DWELL.value,
        )
    )

    if zone_id:
        query = query.filter(
            EventRecord.zone_id == zone_id
        )

    rows = query.distinct().all()

    return [r[0] for r in rows]


def get_converted_visitors(
    store_id: str,
    db_session: Session,
    since: Optional[datetime] = None,
) -> List[str]:
    """
    Visitors with completed purchase.
    """
    since = _get_since(since)

    rows = (
        db_session.query(VisitorSession.visitor_id)
        .filter(
            VisitorSession.store_id == store_id,
            VisitorSession.entry_time >= since,
            VisitorSession.converted.is_(True),
        )
        .distinct()
        .all()
    )

    return [r[0] for r in rows]


def _drop_off(
    previous_count: int,
    current_count: int,
) -> float:
    """
    Calculate drop-off percentage.
    """
    if previous_count == 0:
        return 0.0

    return round(
        ((previous_count - current_count) / previous_count)
        * 100,
        1,
    )


def compute_funnel(
    store_id: str,
    db_session: Session,
    since: Optional[datetime] = None,
) -> Dict:
    """
    Compute complete conversion funnel.
    """
    since = _get_since(since)

    entry_visitors = set(
        get_unique_visitors(
            store_id,
            db_session,
            since,
        )
    )

    zone_visitors = set(
        get_zone_visitors(
            store_id,
            None,
            db_session,
            since,
        )
    )

    billing_visitors = set(
        get_zone_visitors(
            store_id,
            "BILLING",
            db_session,
            since,
        )
    )

    purchase_visitors = set(
        get_converted_visitors(
            store_id,
            db_session,
            since,
        )
    )

    entry_count = len(entry_visitors)
    zone_count = len(zone_visitors)
    billing_count = len(billing_visitors)
    purchase_count = len(purchase_visitors)

    funnel = [
        {
            "stage": "Entry",
            "count": entry_count,
            "drop_off_pct": 0.0,
        },
        {
            "stage": "Zone Visit",
            "count": zone_count,
            "drop_off_pct": _drop_off(
                entry_count,
                zone_count,
            ),
        },
        {
            "stage": "Billing Queue",
            "count": billing_count,
            "drop_off_pct": _drop_off(
                zone_count,
                billing_count,
            ),
        },
        {
            "stage": "Purchase",
            "count": purchase_count,
            "drop_off_pct": _drop_off(
                billing_count,
                purchase_count,
            ),
        },
    ]

    total_conversion_rate = (
        round(
            (purchase_count / entry_count) * 100,
            1,
        )
        if entry_count > 0
        else 0.0
    )

    re_entry_count = (
        db_session.query(VisitorSession)
        .filter(
            VisitorSession.store_id == store_id,
            VisitorSession.entry_time >= since,
            VisitorSession.is_reentry.is_(True),
        )
        .count()
    )

    return {
        "funnel": funnel,
        "total_conversion_rate": total_conversion_rate,
        "re_entry_count": re_entry_count,
    }
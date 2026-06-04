"""
Heatmap computation for zone visit frequency and dwell time.

Returns per-zone metrics normalised to 0-100 for grid heatmap rendering.
Includes a data_confidence flag when fewer than 20 sessions exist in the window.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional

from sqlalchemy import func
from sqlalchemy.orm import Session

from app.db_models import EventRecord, VisitorSession
from app.models import EventType


_MIN_SESSIONS_FOR_CONFIDENCE = 20


def _get_since(since: Optional[datetime] = None) -> datetime:
    if since is not None:
        return since.replace(tzinfo=None) if since.tzinfo else since
    return datetime.utcnow() - timedelta(hours=24)


def compute_heatmap(
    store_id: str,
    db_session: Session,
    since: Optional[datetime] = None,
) -> Dict:
    """
    Compute zone visit frequency and avg dwell normalised to 0-100.

    Returns:
        {
            "store_id": ...,
            "data_confidence": bool,   # False if < 20 sessions in window
            "zones": [
                {
                    "zone_id": "SKINCARE",
                    "visit_count": 42,
                    "avg_dwell_ms": 18500.0,
                    "heat_score": 87     # normalised 0-100
                },
                ...
            ]
        }
    """
    since = _get_since(since)

    # Count total sessions in window for confidence flag
    total_sessions = (
        db_session.query(VisitorSession)
        .filter(
            VisitorSession.store_id == store_id,
            VisitorSession.entry_time >= since,
        )
        .count()
    )
    data_confidence = total_sessions >= _MIN_SESSIONS_FOR_CONFIDENCE

    # Zone visit count (distinct visitors per zone via ZONE_ENTER or ZONE_DWELL)
    visit_rows = (
        db_session.query(
            EventRecord.zone_id,
            func.count(func.distinct(EventRecord.visitor_id)).label("visit_count"),
        )
        .filter(
            EventRecord.store_id == store_id,
            EventRecord.timestamp >= since,
            EventRecord.is_staff.is_(False),
            EventRecord.zone_id.isnot(None),
            EventRecord.event_type.in_([
                EventType.ZONE_ENTER.value,
                EventType.ZONE_DWELL.value,
            ]),
        )
        .group_by(EventRecord.zone_id)
        .all()
    )

    # Avg dwell per zone (from ZONE_DWELL events only)
    dwell_rows = (
        db_session.query(
            EventRecord.zone_id,
            func.avg(EventRecord.dwell_ms).label("avg_dwell_ms"),
        )
        .filter(
            EventRecord.store_id == store_id,
            EventRecord.timestamp >= since,
            EventRecord.is_staff.is_(False),
            EventRecord.zone_id.isnot(None),
            EventRecord.event_type == EventType.ZONE_DWELL.value,
        )
        .group_by(EventRecord.zone_id)
        .all()
    )

    dwell_by_zone = {row.zone_id: round(row.avg_dwell_ms or 0.0, 2) for row in dwell_rows}

    # Build zone records
    zones: List[Dict] = []
    for row in visit_rows:
        zones.append({
            "zone_id": row.zone_id,
            "visit_count": row.visit_count,
            "avg_dwell_ms": dwell_by_zone.get(row.zone_id, 0.0),
            "heat_score": 0,  # filled after normalisation
        })

    # Normalise visit_count to 0-100
    if zones:
        max_visits = max(z["visit_count"] for z in zones)
        for z in zones:
            z["heat_score"] = (
                round((z["visit_count"] / max_visits) * 100)
                if max_visits > 0 else 0
            )

    # Sort by heat_score descending
    zones.sort(key=lambda z: z["heat_score"], reverse=True)

    return {
        "store_id": store_id,
        "window_hours": 24,
        "total_sessions": total_sessions,
        "data_confidence": data_confidence,
        "zones": zones,
    }

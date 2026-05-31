"""
Anomaly detection service.
"""

from datetime import datetime, timedelta, timezone
from typing import Dict, List

from sqlalchemy import func
from sqlalchemy.orm import Session

from app.db_models import EventRecord
from app.metrics import (
    compute_abandonment_rate,
    compute_conversion_rate,
)
from app.models import EventType


SEVERITY_ORDER = {
    "CRITICAL": 0,
    "WARN": 1,
    "INFO": 2,
}


def check_queue_spike(
    store_id: str,
    db_session: Session,
) -> List[Dict]:
    """
    Detect billing queue spikes.
    """
    anomalies = []

    queue_events = (
        db_session.query(EventRecord)
        .filter(
            EventRecord.store_id == store_id,
            EventRecord.event_type
            == EventType.BILLING_QUEUE_JOIN.value,
        )
        .all()
    )

    max_depth = 0

    for event in queue_events:
        metadata = event.metadata_json or {}
        depth = metadata.get("queue_depth", 0)

        if depth > max_depth:
            max_depth = depth

    if max_depth > 10:
        anomalies.append(
            {
                "type": "BILLING_QUEUE_SPIKE",
                "severity": "CRITICAL",
                "value": max_depth,
                "threshold": 10,
                "suggested_action": "Add more billing staff",
            }
        )

    elif max_depth > 5:
        anomalies.append(
            {
                "type": "BILLING_QUEUE_SPIKE",
                "severity": "WARN",
                "value": max_depth,
                "threshold": 5,
                "suggested_action": "Monitor billing queue closely",
            }
        )

    return anomalies


def check_conversion_drop(
    store_id: str,
    db_session: Session,
) -> List[Dict]:
    """
    Detect conversion drop against 7-day baseline.
    """
    anomalies = []

    now = datetime.now(timezone.utc)

    today_rate = compute_conversion_rate(
        store_id,
        db_session,
        since=now - timedelta(days=1),
    )

    seven_day_rate = compute_conversion_rate(
        store_id,
        db_session,
        since=now - timedelta(days=7),
    )

    if seven_day_rate <= 0:
        return anomalies

    ratio = today_rate / seven_day_rate

    if ratio < 0.5:
        anomalies.append(
            {
                "type": "CONVERSION_DROP",
                "severity": "CRITICAL",
                "value": round(today_rate, 2),
                "threshold": round(seven_day_rate, 2),
                "suggested_action": (
                    "Investigate conversion funnel and staffing"
                ),
            }
        )

    elif ratio < 0.75:
        anomalies.append(
            {
                "type": "CONVERSION_DROP",
                "severity": "WARN",
                "value": round(today_rate, 2),
                "threshold": round(seven_day_rate, 2),
                "suggested_action": (
                    "Review merchandising and customer flow"
                ),
            }
        )

    return anomalies


def check_dead_zones(
    store_id: str,
    db_session: Session,
    zone_list: List[str],
) -> List[Dict]:
    """
    Detect zones with no activity for 30 minutes.
    """
    anomalies = []

    cutoff = datetime.now(timezone.utc) - timedelta(minutes=30)

    for zone in zone_list:
        latest_event = (
            db_session.query(EventRecord)
            .filter(
                EventRecord.store_id == store_id,
                EventRecord.zone_id == zone,
                EventRecord.event_type
                == EventType.ZONE_DWELL.value,
            )
            .order_by(EventRecord.timestamp.desc())
            .first()
        )

        if latest_event is None:
            anomalies.append(
                {
                    "type": "DEAD_ZONE",
                    "severity": "INFO",
                    "value": zone,
                    "threshold": "30_minutes",
                    "suggested_action": (
                        "Review zone placement and promotions"
                    ),
                }
            )
            continue

        if latest_event.timestamp < cutoff:
            anomalies.append(
                {
                    "type": "DEAD_ZONE",
                    "severity": "INFO",
                    "value": zone,
                    "threshold": "30_minutes",
                    "suggested_action": (
                        "Review zone placement and promotions"
                    ),
                }
            )

    return anomalies


def check_abandonment(
    store_id: str,
    db_session: Session,
) -> List[Dict]:
    """
    Detect excessive billing abandonment.
    """
    anomalies = []

    abandonment_rate = compute_abandonment_rate(
        store_id,
        db_session,
    )

    if abandonment_rate > 15:
        anomalies.append(
            {
                "type": "HIGH_ABANDONMENT",
                "severity": "WARN",
                "value": round(abandonment_rate, 2),
                "threshold": 15,
                "suggested_action": (
                    "Reduce queue wait times"
                ),
            }
        )

    return anomalies


def get_anomalies(
    store_id: str,
    db_session: Session,
) -> Dict:
    """
    Run all anomaly checks and return results.
    """
    try:
        anomalies = []

        anomalies.extend(
            check_queue_spike(
                store_id,
                db_session,
            )
        )

        anomalies.extend(
            check_conversion_drop(
                store_id,
                db_session,
            )
        )

        zone_rows = (
            db_session.query(
                EventRecord.zone_id
            )
            .filter(
                EventRecord.store_id == store_id,
                EventRecord.zone_id.isnot(None),
            )
            .distinct()
            .all()
        )

        zone_list = [
            row[0]
            for row in zone_rows
            if row[0]
        ]

        anomalies.extend(
            check_dead_zones(
                store_id,
                db_session,
                zone_list,
            )
        )

        anomalies.extend(
            check_abandonment(
                store_id,
                db_session,
            )
        )

        anomalies.sort(
            key=lambda x: SEVERITY_ORDER.get(
                x["severity"],
                99,
            )
        )

        return {
            "anomalies": anomalies
        }

    except Exception as exc:
        return {
            "anomalies": [],
            "error": str(exc),
        }
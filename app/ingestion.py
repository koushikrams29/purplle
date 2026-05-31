"""
Event ingestion service.

Handles:
- Event validation
- Duplicate detection
- Event storage
- Visitor session management
- POS correlation
"""

import logging
from datetime import timedelta
from typing import Dict, List, Tuple

from sqlalchemy.orm import Session

from app.models import VisitorEvent, EventType
from app.db_models import (
    EventRecord,
    VisitorSession,
    ConversionRecord,
)

logger = logging.getLogger(__name__)

MAX_BATCH_SIZE = 500
def create_or_update_session(
    event: VisitorEvent,
    db_session: Session
) -> None:
    """
    Create or update visitor session based on incoming event.
    """

    active_session = (
        db_session.query(VisitorSession)
        .filter(
            VisitorSession.visitor_id == event.visitor_id,
            VisitorSession.store_id == event.store_id,
            VisitorSession.exit_time.is_(None)
        )
        .first()
    )

    if event.event_type == EventType.ENTRY:
        if active_session:
            return

        session = VisitorSession(
            session_id=f"{event.visitor_id}_{int(event.timestamp.timestamp())}",
            visitor_id=event.visitor_id,
            store_id=event.store_id,
            entry_time=event.timestamp,
            zones_visited=[],
            total_dwell_ms=0,
            converted=False,
            is_reentry=False,
        )

        db_session.add(session)
        return

    if not active_session:
        return

    if event.event_type == EventType.ZONE_DWELL:
        zones = active_session.zones_visited or []

        if event.zone_id not in zones:
            zones.append(event.zone_id)

        active_session.zones_visited = zones
        active_session.total_dwell_ms += event.dwell_ms

    elif event.event_type == EventType.EXIT:
        active_session.exit_time = event.timestamp

    elif event.event_type == EventType.REENTRY:
        active_session.is_reentry = True

def ingest_events(
    event_list: List[Dict],
    db_session: Session
) -> Tuple[int, List[Dict]]:
    """
    Ingest a batch of events.

    Args:
        event_list: Raw event dictionaries
        db_session: SQLAlchemy session

    Returns:
        (ingested_count, failed_events)
    """

    if len(event_list) > MAX_BATCH_SIZE:
        raise ValueError(
            f"Batch size exceeds limit of {MAX_BATCH_SIZE}"
        )

    ingested_count = 0
    failed_events = []

    for idx, raw_event in enumerate(event_list):
        try:
            # Validate against Pydantic schema
            event = VisitorEvent.model_validate(raw_event)

            # Check duplicate event_id
            existing = (
                db_session.query(EventRecord)
                .filter(
                    EventRecord.event_id == event.event_id
                )
                .first()
            )

            if existing:
                logger.info(
                    f"Duplicate event skipped: {event.event_id}"
                )
                continue

            # Store event
            record = EventRecord(
                event_id=event.event_id,
                store_id=event.store_id,
                camera_id=event.camera_id,
                visitor_id=event.visitor_id,
                event_type=event.event_type.value,
                timestamp=event.timestamp,
                zone_id=event.zone_id,
                dwell_ms=event.dwell_ms,
                is_staff=event.is_staff,
                confidence=event.confidence,
                metadata_json=event.metadata.model_dump(),
            )

            db_session.add(record)

            # Update visitor session
            create_or_update_session(
                event=event,
                db_session=db_session,
            )

            ingested_count += 1

        except Exception as exc:
            logger.exception(
                f"Failed to ingest event at index {idx}"
            )

            failed_events.append(
                {
                    "index": idx,
                    "error": str(exc),
                    "event": raw_event,
                }
            )

    try:
        db_session.commit()

    except Exception as exc:
        db_session.rollback()

        logger.exception(
            "Database commit failed"
        )

        raise RuntimeError(
            f"Database commit failed: {str(exc)}"
        )

    return ingested_count, failed_events

def correlate_with_pos(
    session: VisitorSession,
    pos_data: List[Dict],
    db_session: Session
) -> bool:
    """
    Correlate a visitor session with POS transactions.

    Conversion Rule:
    A visitor is considered converted if:
    - They visited the BILLING zone
    - A transaction occurred within 5 minutes after their visit

    Args:
        session: VisitorSession instance
        pos_data: List of POS transaction dictionaries
        db_session: SQLAlchemy session

    Returns:
        True if conversion matched, False otherwise
    """

    try:
        if not session:
            return False

        zones = session.zones_visited or []

        if "BILLING" not in zones:
            return False

        if not session.exit_time:
            return False

        session_time = session.exit_time

        for txn in pos_data:
            try:
                txn_time = txn.get("transaction_time")

                if txn_time is None:
                    continue

                if isinstance(txn_time, str):
                    from datetime import datetime

                    txn_time = datetime.fromisoformat(
                        txn_time.replace("Z", "+00:00")
                    )

                time_diff = abs(
                    (txn_time - session_time).total_seconds()
                )

                if time_diff <= 300:
                    conversion = ConversionRecord(
                        conversion_id=(
                            f"CONV_{session.session_id}_"
                            f"{txn.get('transaction_id', 'UNKNOWN')}"
                        ),
                        session_id=session.session_id,
                        store_id=session.store_id,
                        transaction_id=txn.get("transaction_id"),
                        transaction_time=txn_time,
                        transaction_amount=txn.get(
                            "transaction_amount"
                        ),
                        matched=True,
                    )

                    db_session.add(conversion)

                    session.converted = True

                    logger.info(
                        f"Conversion matched for "
                        f"session {session.session_id}"
                    )

                    return True

            except Exception as txn_exc:
                logger.warning(
                    f"Failed processing transaction: {txn_exc}"
                )

        return False

    except Exception as exc:
        logger.exception(
            f"POS correlation failed: {exc}"
        )
        return False
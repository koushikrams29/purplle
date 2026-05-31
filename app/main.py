"""
Store Intelligence API
Main FastAPI application
"""

from fastapi import (
    FastAPI,
    Depends,
    HTTPException,
    Request
)

import logging

from sqlalchemy import text

from app.metrics import get_metrics
from app.funnel import compute_funnel
from app.anomalies import get_anomalies
from app.database import SessionLocal
from app.db_models import EventRecord
from datetime import datetime
from app.config import settings
from app.database import init_db
from sqlalchemy.orm import Session
from app.models import EventIngestionRequest
from app.database import get_db
from app.ingestion import ingest_events
from app.logging_config import (
    setup_logging,
    LoggingMiddleware,
)


# Initialize database
init_db()
setup_logging()

# Create FastAPI app
tags_metadata = [
    {
        "name": "System",
        "description": "Health and system endpoints"
    },
    {
        "name": "Event Ingestion",
        "description": "CCTV event ingestion APIs"
    },
    {
        "name": "Metrics",
        "description": "Store metrics and analytics"
    },
    {
        "name": "Funnel",
        "description": "Conversion funnel analytics"
    },
    {
        "name": "Anomalies",
        "description": "Anomaly detection APIs"
    }
]

app = FastAPI(
    title=settings.api_title,
    description="Real-time retail analytics from CCTV footage",
    version=settings.api_version,
    debug=settings.debug,
    openapi_tags=tags_metadata
)

app.add_middleware(
    LoggingMiddleware
)

logger = logging.getLogger(__name__)



@app.on_event("startup")
async def startup_event():
    """Startup event"""
    print("🚀 Store Intelligence API starting...")
    print(f"   Environment: {settings.environment}")
    print(f"   Database: {settings.database_url}")
    print(f"   YOLO Model: {settings.yolo_model}")


@app.on_event("shutdown")
async def shutdown_event():
    """Shutdown event"""
    print("🛑 Store Intelligence API shutting down...")


@app.get(
    "/health",
    tags=["System"]
)
async def health():
    try:
        db = SessionLocal()

        db.execute(text("SELECT 1"))

        latest_event = (
            db.query(EventRecord)
            .order_by(
                EventRecord.timestamp.desc()
            )
            .first()
        )

        lag_minutes = 0
        warnings = []
        last_event_timestamp = None

        if latest_event:
            last_event_timestamp = (
                latest_event.timestamp.isoformat()
            )

            event_time = latest_event.timestamp

            if event_time.tzinfo is not None:
                event_time = event_time.replace(tzinfo=None)

            lag_minutes = int(
                (
                    datetime.utcnow()
                    - event_time
                ).total_seconds()
                / 60
            )

            if lag_minutes > 10:
                warnings.append(
                    "STALE_FEED"
                )

        db = SessionLocal()

        try:
            ...
        finally:
            db.close()

        return {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "service": "store-intelligence-api",
            "version": settings.api_version,
            "database": "healthy",
            "last_event_timestamp": last_event_timestamp,
            "lag_minutes": lag_minutes,
            "warnings": warnings
        }

    except Exception:
        raise HTTPException(
            status_code=503,
            detail="Database unavailable"
        )


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Store Intelligence API",
        "docs": "/docs",
        "health": "/health"
    }

@app.post(
    "/events/ingest",
    tags=["Event Ingestion"]
)
async def ingest_events_endpoint(
    payload: EventIngestionRequest,
    db: Session = Depends(get_db)
):
    """
    Ingest CCTV-generated events.

    Supports:
    - Batch ingestion (max 500)
    - Duplicate detection
    - Partial success handling
    - Visitor session creation/update
    """

    try:
        ingested_count, failed_events = ingest_events(
            event_list=payload.events,
            db_session=db
        )

        return {
            "status": "ok",
            "ingested": ingested_count,
            "failed": len(failed_events),
            "errors": failed_events,
        }

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc)
        )

    except Exception:
        logger.exception("Event ingestion failed")

    raise HTTPException(
        status_code=500,
        detail="Internal server error"
    )

@app.get(
    "/stores/{store_id}/metrics",
    tags=["Metrics"]
)
async def metrics_endpoint(
    store_id: str,
    db: Session = Depends(get_db)
):
    result = get_metrics(store_id, db)
    result["timestamp"] = datetime.utcnow().isoformat() + "Z"
    return result

@app.get(
    "/stores/{store_id}/funnel",
    tags=["Funnel"]
)
async def funnel_endpoint(
    store_id: str,
    db: Session = Depends(get_db)
):
    result = compute_funnel(store_id,db)
    result["timestamp"] = datetime.utcnow().isoformat() + "Z"
    return result


@app.get(
    "/stores/{store_id}/anomalies",
    tags=["Anomalies"]
)
async def anomalies_endpoint(
    store_id: str,
    db: Session = Depends(get_db)
):
    result = get_anomalies(store_id,db)
    result["timestamp"] = datetime.utcnow().isoformat() + "Z"
    return result

if __name__ == "__main__":

    import uvicorn
    uvicorn.run(
        app,
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug
    )
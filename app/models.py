"""
Event schema for Store Intelligence System CCTV analysis.

Defines Pydantic models for visitor events, including entry/exit tracking,
zone dwell time, queue interactions, and re-entry detection.
"""

from enum import Enum
from typing import Optional
from uuid import uuid4
from datetime import datetime, timezone

from pydantic import BaseModel, Field, field_validator, ConfigDict

from typing import Any, Dict, List
from pydantic import BaseModel, Field


class EventIngestionRequest(BaseModel):
    """
    Request model for batch event ingestion.
    """

    events: List[Dict[str, Any]] = Field(
        ...,
        max_length=500,
        description="Batch of CCTV events"
    )

class EventType(str, Enum):
    """Enumeration of supported event types."""
    
    ENTRY = "ENTRY"
    EXIT = "EXIT"
    ZONE_ENTER = "ZONE_ENTER"
    ZONE_EXIT = "ZONE_EXIT"
    ZONE_DWELL = "ZONE_DWELL"
    BILLING_QUEUE_JOIN = "BILLING_QUEUE_JOIN"
    BILLING_QUEUE_ABANDON = "BILLING_QUEUE_ABANDON"
    REENTRY = "REENTRY"


class EventMetadata(BaseModel):
    """
    Metadata associated with an event.
    
    Attributes:
        queue_depth: Number of customers in billing queue (null if not applicable)
        sku_zone: Specific product zone or SKU category
        session_seq: Sequential position in visitor session (1-indexed)
    """
    
    queue_depth: Optional[int] = Field(
        default=None,
        ge=0,
        description="Number of customers in billing queue"
    )
    sku_zone: Optional[str] = Field(
        default=None,
        description="Specific product zone or SKU category"
    )
    session_seq: int = Field(
        default=1,
        ge=1,
        description="Sequential position in visitor session"
    )
    
    model_config = ConfigDict(extra="forbid")


class VisitorEvent(BaseModel):
    """
    Represents a visitor event detected by CCTV analysis.
    
    This model captures temporal and spatial visitor interactions within a store,
    including zone transitions, dwell patterns, and queue behavior.
    
    Attributes:
        event_id: UUID v4 identifier for the event
        store_id: Store identifier (e.g., STORE_BLR_002)
        camera_id: Camera identifier (e.g., CAM_ENTRY_01)
        visitor_id: Unique visitor identifier
        event_type: Type of event (ENTRY, EXIT, ZONE_DWELL, etc.)
        timestamp: ISO 8601 timestamp with timezone
        zone_id: Zone or section identifier (e.g., SKINCARE, BILLING)
        dwell_ms: Duration in zone in milliseconds
        is_staff: Whether visitor is store staff
        confidence: Detection confidence score [0.0, 1.0]
        metadata: Additional event-specific metadata
    """
    
    event_id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="UUID v4 identifier"
    )
    store_id: str = Field(
        ...,
        pattern=r"^STORE_[A-Z]{3}_\d{3}$",
        description="Store identifier format: STORE_XXX_###"
    )
    camera_id: str = Field(
        ...,
        pattern=r"^CAM_[A-Z_]+_\d{2}$",
        description="Camera identifier format: CAM_LOCATION_##"
    )
    visitor_id: str = Field(
        ...,
        min_length=1,
        max_length=50,
        description="Unique visitor identifier"
    )
    event_type: EventType = Field(
        ...,
        description="Type of event"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="ISO 8601 timestamp with timezone"
    )
    zone_id: str = Field(
        ...,
        min_length=1,
        max_length=50,
        description="Zone or section identifier"
    )
    dwell_ms: int = Field(
        default=0,
        ge=0,
        description="Duration in zone in milliseconds"
    )
    is_staff: bool = Field(
        default=False,
        description="Whether visitor is store staff"
    )
    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Detection confidence score [0.0, 1.0]"
    )
    metadata: EventMetadata = Field(
        default_factory=EventMetadata,
        description="Event-specific metadata"
    )
    
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "event_id": "550e8400-e29b-41d4-a716-446655440000",
                "store_id": "STORE_BLR_002",
                "camera_id": "CAM_ENTRY_01",
                "visitor_id": "VIS_c8a2f1",
                "event_type": "ZONE_DWELL",
                "timestamp": "2026-03-03T14:22:10Z",
                "zone_id": "SKINCARE",
                "dwell_ms": 8400,
                "is_staff": False,
                "confidence": 0.91,
                "metadata": {
                    "queue_depth": None,
                    "sku_zone": "MOISTURISER",
                    "session_seq": 5
                }
            }
        }
    )
    
    @field_validator("timestamp", mode="before")
    @classmethod
    def ensure_utc_timezone(cls, v):
        """Ensure timestamp has UTC timezone."""
        if isinstance(v, datetime):
            if v.tzinfo is None:
                return v.replace(tzinfo=timezone.utc)
            return v
        return v
    
    @staticmethod
    def create_entry_event(
        store_id: str,
        camera_id: str,
        visitor_id: str,
        zone_id: str,
        confidence: float,
        is_staff: bool = False,
        timestamp: Optional[datetime] = None
    ) -> "VisitorEvent":
        """
        Factory method to create an ENTRY event.
        
        Args:
            store_id: Store identifier
            camera_id: Camera identifier
            visitor_id: Visitor identifier
            zone_id: Entry zone identifier
            confidence: Detection confidence
            is_staff: Whether visitor is staff
            timestamp: Event timestamp (defaults to now UTC)
        
        Returns:
            VisitorEvent configured as ENTRY type
        """
        return VisitorEvent(
            store_id=store_id,
            camera_id=camera_id,
            visitor_id=visitor_id,
            event_type=EventType.ENTRY,
            timestamp=timestamp or datetime.now(timezone.utc),
            zone_id=zone_id,
            is_staff=is_staff,
            confidence=confidence
        )
    
    @staticmethod
    def create_zone_dwell_event(
        store_id: str,
        camera_id: str,
        visitor_id: str,
        zone_id: str,
        dwell_ms: int,
        confidence: float,
        session_seq: int = 1,
        sku_zone: Optional[str] = None,
        is_staff: bool = False,
        timestamp: Optional[datetime] = None
    ) -> "VisitorEvent":
        """
        Factory method to create a ZONE_DWELL event.
        
        Args:
            store_id: Store identifier
            camera_id: Camera identifier
            visitor_id: Visitor identifier
            zone_id: Zone identifier
            dwell_ms: Dwell time in milliseconds
            confidence: Detection confidence
            session_seq: Position in visitor session
            sku_zone: Specific product zone
            is_staff: Whether visitor is staff
            timestamp: Event timestamp (defaults to now UTC)
        
        Returns:
            VisitorEvent configured as ZONE_DWELL type
        """
        return VisitorEvent(
            store_id=store_id,
            camera_id=camera_id,
            visitor_id=visitor_id,
            event_type=EventType.ZONE_DWELL,
            timestamp=timestamp or datetime.now(timezone.utc),
            zone_id=zone_id,
            dwell_ms=dwell_ms,
            is_staff=is_staff,
            confidence=confidence,
            metadata=EventMetadata(
                session_seq=session_seq,
                sku_zone=sku_zone
            )
        )
    
    def to_dict(self) -> dict:
        """Convert event to dictionary representation."""
        return self.model_dump(mode="json")
    
    def to_json(self) -> str:
        """Convert event to JSON string representation."""
        return self.model_dump_json()

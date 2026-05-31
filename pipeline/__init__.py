"""
CCTV detection pipeline for retail store analytics.

Modules:
    detect: YOLOv8 detection pipeline
    tracker: Centroid-based visitor tracking
    emit: Event generation and zone classification
"""

try:
    from pipeline.detect import StoreDetector, CentroidTracker, ZoneMapper
except ImportError:
    pass

try:
    from pipeline.tracker import SimpleTracker
except ImportError:
    pass

try:
    from pipeline.emit import (
        point_in_polygon,
        classify_zone,
        emit_events,
        handle_disappeared_visitors,
        get_visitor_statistics,
        reset_visitor_state,
    )
except ImportError:
    pass


__all__ = [
    # Detection
    "StoreDetector",
    "CentroidTracker",
    "ZoneMapper",

    # Tracking
    "SimpleTracker",

    # Event emission
    "point_in_polygon",
    "classify_zone",
    "emit_events",
    "handle_disappeared_visitors",
    "get_visitor_statistics",
    "reset_visitor_state",
]
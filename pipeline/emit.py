"""
Event emission logic for CCTV detection pipeline.

Generates structured visitor events based on tracked detections,
zone classification, and temporal analysis.
"""

from typing import List, Dict, Optional, Tuple, Set
from datetime import datetime, timedelta

from app.models import VisitorEvent, EventType, EventMetadata


# ============================================================================
# POINT-IN-POLYGON DETECTION
# ============================================================================

def point_in_polygon(
    point: Tuple[float, float],
    polygon: List[List[int]]
) -> bool:
    """
    Check if a point is inside a polygon using ray-casting algorithm.
    
    Algorithm:
    Cast a ray from the point to infinity (horizontal right).
    Count intersections with polygon edges.
    Odd count = inside, even count = outside.
    
    Args:
        point: Tuple of (x, y) coordinates
        polygon: List of [x, y] vertices defining polygon (clockwise or counter-clockwise)
        
    Returns:
        True if point is inside polygon, False otherwise
        
    Raises:
        ValueError: If polygon has fewer than 3 vertices
    """
    if len(polygon) < 3:
        raise ValueError(f"Polygon must have at least 3 vertices, got {len(polygon)}")
    
    x, y = point
    n = len(polygon)
    inside = False
    
    x1, y1 = polygon[0]
    
    for i in range(1, n + 1):
        x2, y2 = polygon[i % n]
        
        # Check if point is at y-range of edge
        if y > min(y1, y2):
            if y <= max(y1, y2):
                if x <= max(x1, x2):
                    # Calculate x-intersection of ray with edge
                    if y1 != y2:
                        xinters = (y - y1) * (x2 - x1) / (y2 - y1) + x1
                    
                    # If point is on edge or to left of edge, toggle inside flag
                    if x1 == x2 or x <= xinters:
                        inside = not inside
        
        x1, y1 = x2, y2
    
    return inside


def classify_zone(
    centroid: Tuple[float, float],
    zones: Dict[str, List[List[int]]]
) -> Optional[str]:
    """
    Classify a centroid to a zone using point-in-polygon detection.
    
    When multiple zones contain the point, returns the first match
    (iteration order depends on dict insertion order).
    
    Args:
        centroid: Tuple of (cx, cy) coordinates
        zones: Dict mapping zone_id to list of polygon vertices
               Example: {"SKINCARE": [[200, 150], [400, 150], [400, 300], [200, 300]]}
        
    Returns:
        Zone ID if centroid is inside a zone, None otherwise
        
    Raises:
        ValueError: If zones dict is empty
    """
    if not zones:
        raise ValueError("zones dictionary cannot be empty")
    
    for zone_id, polygon in zones.items():
        if point_in_polygon(centroid, polygon):
            return zone_id
    
    return None


def emit_events(
    detections: List[Tuple[str, float, float, float, float, float]],
    frame_num: int,
    fps: int,
    store_id: str,
    camera_id: str,
    zones: Dict[str, List[List[int]]],
    start_timestamp: datetime,
    visitor_state: Dict[str, Dict]
) -> List[VisitorEvent]:
    """
    Generate structured events from tracked detections.
    
    Maintains state about visitor presence, zone transitions, and dwell time.
    
    Args:
        detections: List of (visitor_id, x1, y1, x2, y2, confidence) tuples
        frame_num: Current frame number
        fps: Video frames per second (e.g., 15)
        store_id: Store identifier (e.g., 'STORE_BLR_002')
        camera_id: Camera identifier (e.g., 'CAM_ENTRY_01')
        zones: Dict mapping zone_id to polygon vertices
        start_timestamp: Timestamp of video start (datetime with timezone)
        visitor_state: Mutable dict to track visitor state across calls
                      Format: {visitor_id: {'zone': zone_id, 'frame': frame_num, 'session_seq': int}}
        
    Returns:
        List of VisitorEvent objects
        
    Raises:
        ValueError: If zones or store_id/camera_id invalid
        TypeError: If detections format invalid
    """
    if not zones:
        raise ValueError("zones dictionary cannot be empty")
    if not isinstance(store_id, str) or not store_id.strip():
        raise ValueError("store_id must be non-empty string")
    if not isinstance(camera_id, str) or not camera_id.strip():
        raise ValueError("camera_id must be non-empty string")
    if not isinstance(detections, list):
        raise TypeError("detections must be a list")
    
    events = []
    current_visitors: Set[str] = set()
    
    # Calculate timestamp for this frame
    frame_time_sec = frame_num / fps
    timestamp = start_timestamp + timedelta(seconds=frame_time_sec)
    
    # Process each detection
    for detection in detections:
        try:
            visitor_id, x1, y1, x2, y2, confidence = detection
        except (ValueError, TypeError) as e:
            raise TypeError(f"Detection format invalid: {str(e)}")
        
        if not isinstance(visitor_id, str):
            raise TypeError(f"visitor_id must be string, got {type(visitor_id)}")
        if not all(isinstance(v, (int, float)) for v in [x1, y1, x2, y2, confidence]):
            raise TypeError("Bounding box and confidence must be numeric")
        
        current_visitors.add(visitor_id)
        
        # Calculate centroid
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        centroid = (cx, cy)
        
        # Classify to zone
        zone_id = classify_zone(centroid, zones)
        
        # Initialize visitor state if not seen before
        if visitor_id not in visitor_state:
            visitor_state[visitor_id] = {
                'zone': zone_id,
                'frame': frame_num,
                'session_seq': 1,
                'last_event_frame': frame_num,
                'entered': False
            }
            
            # Emit ENTRY event
            entry_event = VisitorEvent(
                store_id=store_id,
                camera_id=camera_id,
                visitor_id=visitor_id,
                event_type=EventType.ENTRY,
                timestamp=timestamp,
                zone_id=zone_id or "UNKNOWN",
                confidence=confidence,
                metadata=EventMetadata(
                    session_seq=visitor_state[visitor_id]['session_seq']
                )
            )
            events.append(entry_event)
            visitor_state[visitor_id]['entered'] = True
        
        else:
            # Visitor previously seen
            prev_state = visitor_state[visitor_id]
            prev_zone = prev_state['zone']
            
            # Check for zone transition
            if zone_id != prev_zone:
                # ZONE_EXIT event
                if prev_zone is not None:
                    exit_event = VisitorEvent(
                        store_id=store_id,
                        camera_id=camera_id,
                        visitor_id=visitor_id,
                        event_type=EventType.ZONE_EXIT,
                        timestamp=timestamp,
                        zone_id=prev_zone,
                        confidence=confidence,
                        metadata=EventMetadata(
                            session_seq=prev_state['session_seq']
                        )
                    )
                    events.append(exit_event)
                
                # ZONE_ENTER event
                if zone_id is not None:
                    enter_event = VisitorEvent(
                        store_id=store_id,
                        camera_id=camera_id,
                        visitor_id=visitor_id,
                        event_type=EventType.ZONE_ENTER,
                        timestamp=timestamp,
                        zone_id=zone_id,
                        confidence=confidence,
                        metadata=EventMetadata(
                            session_seq=prev_state['session_seq']
                        )
                    )
                    events.append(enter_event)
                
                # Update state
                prev_state['zone'] = zone_id
                prev_state['frame'] = frame_num
                prev_state['last_event_frame'] = frame_num
            
            else:
                # Same zone - potentially emit ZONE_DWELL event
                frames_in_zone = frame_num - prev_state['frame']
                
                # Emit ZONE_DWELL every 30 frames (1 second at 30fps, ~2 seconds at 15fps)
                frames_since_last_event = frame_num - prev_state['last_event_frame']
                
                if zone_id is not None and frames_since_last_event >= 30:
                    dwell_ms = int((frames_in_zone / fps) * 1000)
                    
                    dwell_event = VisitorEvent(
                        store_id=store_id,
                        camera_id=camera_id,
                        visitor_id=visitor_id,
                        event_type=EventType.ZONE_DWELL,
                        timestamp=timestamp,
                        zone_id=zone_id,
                        dwell_ms=dwell_ms,
                        confidence=confidence,
                        metadata=EventMetadata(
                            session_seq=prev_state['session_seq']
                        )
                    )
                    events.append(dwell_event)
                    prev_state['last_event_frame'] = frame_num
    
    return events


def handle_disappeared_visitors(
    visitor_state: Dict[str, Dict],
    current_visitors: Set[str],
    frame_num: int,
    fps: int,
    store_id: str,
    camera_id: str,
    start_timestamp: datetime,
    max_disappear_frames: int = 30
) -> List[VisitorEvent]:
    """
    Generate EXIT events for visitors that have disappeared.
    
    Args:
        visitor_state: State dict tracking all visitors
        current_visitors: Set of visitor_ids seen in current frame
        frame_num: Current frame number
        fps: Video frames per second
        store_id: Store identifier
        camera_id: Camera identifier
        start_timestamp: Timestamp of video start
        max_disappear_frames: Max frames to track without detection before exit
        
    Returns:
        List of EXIT VisitorEvent objects for disappeared visitors
    """
    events = []
    
    frame_time_sec = frame_num / fps
    timestamp = start_timestamp + timedelta(seconds=frame_time_sec)
    
    for visitor_id, state in list(visitor_state.items()):
        if visitor_id not in current_visitors:
            # Visitor not in current frame
            frames_missing = frame_num - state['frame']
            
            # Emit EXIT if exceeded threshold
            if frames_missing > max_disappear_frames:
                exit_event = VisitorEvent(
                    store_id=store_id,
                    camera_id=camera_id,
                    visitor_id=visitor_id,
                    event_type=EventType.EXIT,
                    timestamp=timestamp,
                    zone_id=state['zone'] or "UNKNOWN",
                    confidence=0.90,
                    metadata=EventMetadata(
                        session_seq=state['session_seq']
                    )
                )
                events.append(exit_event)
                
                # Remove from state
                del visitor_state[visitor_id]
            else:
                # Mark as missing but keep tracking
                state['missing_frames'] = frames_missing
    
    return events


def get_visitor_statistics(visitor_state: Dict[str, Dict]) -> Dict:
    """
    Get summary statistics about tracked visitors.
    
    Args:
        visitor_state: State dict tracking all visitors
        
    Returns:
        Dict with statistics: total, active, exited, by_zone
    """
    total = len(visitor_state)
    active = sum(1 for v in visitor_state.values() if v.get('entered', False))
    
    by_zone = {}
    for visitor_id, state in visitor_state.items():
        zone = state.get('zone', 'UNKNOWN')
        if zone not in by_zone:
            by_zone[zone] = 0
        by_zone[zone] += 1
    
    return {
        'total_tracked': total,
        'active': active,
        'by_zone': by_zone
    }


def reset_visitor_state() -> Dict[str, Dict]:
    """
    Create a fresh visitor state dictionary.
    
    Returns:
        Empty visitor state dict ready for a new video processing session
    """
    return {}

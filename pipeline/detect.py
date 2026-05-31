"""
Production-ready CCTV detection pipeline for retail store analysis.

Orchestrates YOLOv8 inference, multi-person tracking, zone classification,
and structured event generation from retail CCTV footage.
"""

import logging
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
import uuid

import cv2
import numpy as np
from ultralytics import YOLO

from app.models import VisitorEvent, EventType, EventMetadata


logger = logging.getLogger(__name__)


# ============================================================================
# ZONE MAPPER
# ============================================================================

@dataclass
class Point:
    """2D point representation."""
    x: float
    y: float


class ZoneMapper:
    """
    Maps pixel coordinates to store zones using polygon containment.
    
    Supports multiple overlapping zones with configurable priority.
    Uses ray-casting algorithm for point-in-polygon detection.
    """
    
    def __init__(self, zones: Dict[str, List[List[int]]]):
        """
        Initialize zone mapper with zone definitions.
        
        Args:
            zones: Dict mapping zone_id to list of polygon vertices.
                   Example: {"SKINCARE": [[200, 150], [400, 150], [400, 300], [200, 300]]}
        """
        self.zones = zones
        self._validate_zones()
    
    def _validate_zones(self) -> None:
        """Validate zone polygon definitions."""
        for zone_id, polygon in self.zones.items():
            if not isinstance(polygon, list) or len(polygon) < 3:
                raise ValueError(
                    f"Zone '{zone_id}' requires at least 3 vertices. "
                    f"Got {len(polygon) if isinstance(polygon, list) else 'invalid'}"
                )
    
    @staticmethod
    def _point_in_polygon(point: Point, polygon: List[List[int]]) -> bool:
        """
        Check if point is inside polygon using ray-casting algorithm.
        
        Args:
            point: 2D point to check
            polygon: List of [x, y] vertices defining polygon
            
        Returns:
            True if point is inside polygon, False otherwise
        """
        x, y = point.x, point.y
        n = len(polygon)
        inside = False
        
        x1, y1 = polygon[0]
        for i in range(1, n + 1):
            x2, y2 = polygon[i % n]
            
            if y > min(y1, y2):
                if y <= max(y1, y2):
                    if x <= max(x1, x2):
                        if y1 != y2:
                            xinters = (y - y1) * (x2 - x1) / (y2 - y1) + x1
                        if x1 == x2 or x <= xinters:
                            inside = not inside
            
            x1, y1 = x2, y2
        
        return inside
    
    def classify_point(self, x: float, y: float) -> Optional[str]:
        """
        Classify a point to a zone.
        
        When multiple zones contain the point, returns the first match
        (iteration order depends on dict insertion order).
        
        Args:
            x: X coordinate (pixel)
            y: Y coordinate (pixel)
            
        Returns:
            Zone ID if point is inside a zone, None otherwise
        """
        point = Point(x=x, y=y)
        
        for zone_id, polygon in self.zones.items():
            if self._point_in_polygon(point, polygon):
                return zone_id
        
        return None
    
    def classify_centroid(self, centroid: Tuple[float, float]) -> Optional[str]:
        """
        Classify a centroid (x, y) to a zone.
        
        Args:
            centroid: Tuple of (x, y) coordinates
            
        Returns:
            Zone ID if centroid is inside a zone, None otherwise
        """
        x, y = centroid
        return self.classify_point(x, y)
    
    def get_zone_bounds(self, zone_id: str) -> Optional[Tuple[int, int, int, int]]:
        """
        Get bounding box (xmin, ymin, xmax, ymax) for a zone.
        
        Args:
            zone_id: Zone identifier
            
        Returns:
            Tuple of (xmin, ymin, xmax, ymax) or None if zone doesn't exist
        """
        if zone_id not in self.zones:
            return None
        
        polygon = self.zones[zone_id]
        xs = [p[0] for p in polygon]
        ys = [p[1] for p in polygon]
        
        return (min(xs), min(ys), max(xs), max(ys))
    
    def get_zone_polygon(self, zone_id: str) -> Optional[List[List[int]]]:
        """
        Get polygon vertices for a zone.
        
        Args:
            zone_id: Zone identifier
            
        Returns:
            List of [x, y] vertices or None if zone doesn't exist
        """
        return self.zones.get(zone_id)


# ============================================================================
# CENTROID TRACKER
# ============================================================================

@dataclass
class Track:
    """
    Represents a tracked visitor/person.
    
    Attributes:
        visitor_id: Unique identifier for this visitor in session
        centroid: (x, y) coordinates
        frame_idx: Frame number where track was last observed
        entered: Whether visitor has crossed entry threshold
        frames_in_zone: Dict mapping zone_id to frame count
        last_zone: Last detected zone
        frames_since_last_detection: Frames without detection (for timeout)
    """
    
    visitor_id: str
    centroid: Tuple[float, float]
    frame_idx: int
    entered: bool = False
    frames_in_zone: Dict[str, int] = field(default_factory=dict)
    last_zone: Optional[str] = None
    frames_since_last_detection: int = 0


class CentroidTracker:
    """
    Track people across video frames using centroid movement.
    
    Uses distance-based association to link detections across frames.
    Handles entry/exit detection and assigns persistent visitor IDs.
    """
    
    def __init__(
        self,
        entry_zone: str = "ENTRY",
        max_disappear: int = 30,
        distance_threshold: float = 50.0
    ):
        """
        Initialize centroid tracker.
        
        Args:
            entry_zone: Zone ID for entry/exit threshold
            max_disappear: Max frames to track without detection before timeout
            distance_threshold: Max distance (pixels) to associate detection to track
        """
        self.entry_zone = entry_zone
        self.max_disappear = max_disappear
        self.distance_threshold = distance_threshold
        
        self.tracks: Dict[str, Track] = {}
        self.next_id_counter = 0
        self.exited_visitors: set = set()
    
    @staticmethod
    def _euclidean_distance(
        p1: Tuple[float, float],
        p2: Tuple[float, float]
    ) -> float:
        """Calculate Euclidean distance between two points."""
        return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5
    
    def _register_new_track(
        self,
        centroid: Tuple[float, float],
        frame_idx: int,
        zone_id: Optional[str] = None
    ) -> str:
        """
        Register a new track for an unmatched detection.
        
        Args:
            centroid: (x, y) coordinates
            frame_idx: Current frame index
            zone_id: Zone where detection occurred
            
        Returns:
            Visitor ID for new track
        """
        visitor_id = f"VIS_{uuid.uuid4().hex[:6]}"
        
        track = Track(
            visitor_id=visitor_id,
            centroid=centroid,
            frame_idx=frame_idx,
            last_zone=zone_id
        )
        
        self.tracks[visitor_id] = track
        self.next_id_counter += 1
        
        return visitor_id
    
    def update(
        self,
        detections: List[Tuple[float, float]],
        zones: List[Optional[str]],
        frame_idx: int
    ) -> Dict[str, Dict]:
        """
        Update tracker with new detections from frame.
        
        Args:
            detections: List of (x, y) centroids from YOLOv8
            zones: Corresponding zone IDs for each detection
            frame_idx: Current frame index
            
        Returns:
            Dict mapping visitor_id to track info including entry/exit events
        """
        events = {}
        
        if len(detections) == 0:
            # Increment disappearance counter for all tracks
            for visitor_id, track in list(self.tracks.items()):
                track.frames_since_last_detection += 1
                
                # Remove track if disappeared too long
                if track.frames_since_last_detection > self.max_disappear:
                    if track.entered and visitor_id not in self.exited_visitors:
                        events[visitor_id] = {
                            "event": "EXIT",
                            "zone_id": track.last_zone or "UNKNOWN"
                        }
                        self.exited_visitors.add(visitor_id)
                    del self.tracks[visitor_id]
            
            return events
        
        # Match detections to existing tracks
        matched_detections = set()
        matched_tracks = set()
        
        for det_idx, (det_x, det_y) in enumerate(detections):
            best_track_id = None
            best_distance = self.distance_threshold
            
            for visitor_id, track in self.tracks.items():
                if visitor_id in matched_tracks:
                    continue
                
                distance = self._euclidean_distance(
                    (det_x, det_y),
                    track.centroid
                )
                
                if distance < best_distance:
                    best_distance = distance
                    best_track_id = visitor_id
            
            if best_track_id is not None:
                # Match found
                matched_detections.add(det_idx)
                matched_tracks.add(best_track_id)
                
                track = self.tracks[best_track_id]
                track.centroid = (det_x, det_y)
                track.frame_idx = frame_idx
                track.frames_since_last_detection = 0
                
                # Update zone tracking
                zone_id = zones[det_idx]
                if zone_id:
                    if zone_id not in track.frames_in_zone:
                        track.frames_in_zone[zone_id] = 0
                    track.frames_in_zone[zone_id] += 1
                    
                    # Detect entry/exit
                    if zone_id == self.entry_zone and not track.entered:
                        track.entered = True
                        events[best_track_id] = {
                            "event": "ENTRY",
                            "zone_id": zone_id
                        }
                    
                    track.last_zone = zone_id
        
        # Register new tracks for unmatched detections
        for det_idx, (det_x, det_y) in enumerate(detections):
            if det_idx not in matched_detections:
                visitor_id = self._register_new_track(
                    (det_x, det_y),
                    frame_idx,
                    zones[det_idx]
                )
                if zones[det_idx] == self.entry_zone:
                    self.tracks[visitor_id].entered = True
                    events[visitor_id] = {
                        "event": "ENTRY",
                        "zone_id": zones[det_idx]
                    }
        
        # Mark disappeared tracks
        for visitor_id, track in list(self.tracks.items()):
            if visitor_id not in matched_tracks:
                track.frames_since_last_detection += 1
                
                if track.frames_since_last_detection > self.max_disappear:
                    if track.entered and visitor_id not in self.exited_visitors:
                        events[visitor_id] = {
                            "event": "EXIT",
                            "zone_id": track.last_zone or "UNKNOWN"
                        }
                        self.exited_visitors.add(visitor_id)
                    del self.tracks[visitor_id]
        
        return events
    
    def get_active_tracks(self) -> Dict[str, Track]:
        """Get currently active tracks."""
        return dict(self.tracks)
    
    def get_zone_dwell_frames(
        self,
        visitor_id: str,
        zone_id: str
    ) -> int:
        """
        Get number of frames visitor spent in zone.
        
        Args:
            visitor_id: Visitor identifier
            zone_id: Zone identifier
            
        Returns:
            Frame count or 0 if not applicable
        """
        if visitor_id not in self.tracks:
            return 0
        
        return self.tracks[visitor_id].frames_in_zone.get(zone_id, 0)
    
    def reset(self) -> None:
        """Reset tracker state (for new session)."""
        self.tracks.clear()
        self.exited_visitors.clear()
        self.next_id_counter = 0


# ============================================================================
# STORE DETECTOR
# ============================================================================

class StoreDetector:
    """
    Production-ready CCTV detector for retail store analysis.
    
    Orchestrates YOLOv8 inference, tracking, and event generation.
    
    Attributes:
        model: YOLOv8 model instance
        tracker: CentroidTracker for visitor tracking
        zone_mapper: ZoneMapper for coordinate-to-zone classification
    """
    
    # YOLOv8 person class ID
    PERSON_CLASS_ID = 0
    
    # Default frame extraction rate (frames per second to process)
    DEFAULT_FPS_SAMPLE = 1
    
    # Dwell time threshold (frames at ~1 FPS = seconds)
    DWELL_THRESHOLD_FRAMES = 30
    
    # Billing queue detection parameters
    QUEUE_ZONE = "BILLING"
    QUEUE_DENSITY_THRESHOLD = 3
    
    def __init__(
        self,
        model_name: str = "yolov8m",
        device: str = "cpu",
        confidence_threshold: float = 0.5,
        max_disappear_frames: int = 30,
        distance_threshold: float = 50.0,
        entry_zone: str = "ENTRY"
    ):
        """
        Initialize store detector.
        
        Args:
            model_name: YOLOv8 model size (nano, small, medium, large, xlarge)
            device: Device to run inference on ('cpu', 'cuda', 'mps')
            confidence_threshold: Min confidence for detections [0.0, 1.0]
            max_disappear_frames: Max frames to track without detection
            distance_threshold: Max pixel distance for track association
            entry_zone: Zone ID for entry/exit threshold
            
        Raises:
            RuntimeError: If YOLOv8 model fails to load
        """
        try:
            logger.info(f"Loading YOLOv8 {model_name} on {device}")
            self.model = YOLO(f"yolov8{model_name[0]}.pt")
            self.model.to(device)
            self.device = device
        except Exception as e:
            raise RuntimeError(f"Failed to load YOLOv8 model: {str(e)}")
        
        self.confidence_threshold = confidence_threshold
        self.tracker = CentroidTracker(
            entry_zone=entry_zone,
            max_disappear=max_disappear_frames,
            distance_threshold=distance_threshold
        )
        self.zone_mapper: Optional[ZoneMapper] = None
        
        # Session state
        self.session_seq: Dict[str, int] = {}
        self.visitor_zone_entry_frame: Dict[str, Dict[str, int]] = {}
        self.billing_queue_join_frame: Dict[str, int] = {}
    
    def setup_zones(self, store_layout: Dict) -> None:
        """
        Setup zone mapper from store layout.
        
        Args:
            store_layout: Store layout dict with 'zones' key containing zone polygons
            
        Raises:
            ValueError: If store_layout is invalid
        """
        if "zones" not in store_layout:
            raise ValueError("store_layout must contain 'zones' key")
        
        zones = {
            zone_id: data["polygon"]
            for zone_id, data in store_layout["zones"].items()
        }
        
        self.zone_mapper = ZoneMapper(zones)
        logger.info(f"Initialized {len(zones)} zones: {list(zones.keys())}")
    
    @staticmethod
    def _extract_frames_cv2(
        video_path: str,
        fps_sample: int = 1,
        max_frames: Optional[int] = None
    ) -> Tuple[List[np.ndarray], int, Tuple[int, int]]:
        """
        Extract frames from video file.
        
        Args:
            video_path: Path to MP4 video file
            fps_sample: Extract every Nth frame (N=1 every frame, N=15 every 15th frame)
            max_frames: Max frames to extract (None for all)
            
        Returns:
            Tuple of (frames_list, video_fps, frame_size)
            
        Raises:
            IOError: If video file cannot be opened
        """
        video_path_str = str(video_path)
        cap = cv2.VideoCapture(video_path_str)
        
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {video_path_str}")
        
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        logger.info(
            f"Video: {width}x{height} @ {fps}fps, {total_frames} total frames"
        )
        
        frames = []
        frame_count = 0
        extracted_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % fps_sample == 0:
                frames.append(frame)
                extracted_count += 1
                
                if max_frames and extracted_count >= max_frames:
                    break
            
            frame_count += 1
        
        cap.release()
        
        logger.info(f"Extracted {extracted_count} frames from {total_frames}")
        
        return frames, fps, (width, height)
    
    def _run_inference(
        self,
        frames: List[np.ndarray]
    ) -> List[List[Tuple[float, float]]]:
        """
        Run YOLOv8 inference on frames and extract person centroids.
        
        Args:
            frames: List of frame arrays
            
        Returns:
            List of detection lists, each containing (x, y) centroids
        """
        all_detections = []
        
        for frame_idx, frame in enumerate(frames):
            results = self.model(
                frame,
                conf=self.confidence_threshold,
                classes=[self.PERSON_CLASS_ID],
                verbose=False
            )
            
            detections = []
            
            if results and len(results) > 0:
                result = results[0]
                
                if result.boxes is not None and len(result.boxes) > 0:
                    for box in result.boxes:
                        # Get bounding box
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        
                        # Calculate centroid
                        cx = (x1 + x2) / 2
                        cy = (y1 + y2) / 2
                        
                        detections.append((float(cx), float(cy)))
            
            all_detections.append(detections)
            
            if (frame_idx + 1) % 10 == 0:
                logger.debug(
                    f"Processed frame {frame_idx + 1}/{len(frames)}, "
                    f"{len(detections)} detections"
                )
        
        return all_detections
    
    def _classify_detections(
        self,
        detections: List[List[Tuple[float, float]]]
    ) -> List[List[Optional[str]]]:
        """
        Classify detection centroids to zones.
        
        Args:
            detections: List of detection lists with (x, y) centroids
            
        Returns:
            List of zone classification lists (same structure as detections)
        """
        if not self.zone_mapper:
            raise RuntimeError("Zone mapper not initialized. Call setup_zones() first.")
        
        all_zones = []
        
        for detection_list in detections:
            zones = []
            for cx, cy in detection_list:
                zone_id = self.zone_mapper.classify_point(cx, cy)
                zones.append(zone_id)
            all_zones.append(zones)
        
        return all_zones
    
    def _detect_dwell_events(
        self,
        frame_idx: int,
        frame_count: int,
        fps: int,
        store_id: str,
        camera_id: str,
        timestamp_start: datetime
    ) -> List[VisitorEvent]:
        """
        Detect zone dwell events (visitor in zone 30+ seconds).
        
        Args:
            frame_idx: Current frame index
            frame_count: Total frames processed
            fps: Video FPS
            store_id: Store identifier
            camera_id: Camera identifier
            timestamp_start: Timestamp of video start
            
        Returns:
            List of ZONE_DWELL events
        """
        events = []
        
        # Process every ~10 frames to reduce dwell event frequency
        if frame_idx % 10 != 0:
            return events
        
        for visitor_id, track in self.tracker.get_active_tracks().items():
            zone_id = track.last_zone
            if not zone_id or zone_id not in track.frames_in_zone:
                continue
            
            frames_in_zone = track.frames_in_zone[zone_id]
            
            # Check if exceeded dwell threshold
            if frames_in_zone >= self.DWELL_THRESHOLD_FRAMES:
                # Check if we haven't emitted dwell for this zone yet
                if visitor_id not in self.visitor_zone_entry_frame:
                    self.visitor_zone_entry_frame[visitor_id] = {}
                
                entry_frame = self.visitor_zone_entry_frame[visitor_id].get(zone_id)
                
                # Only emit once per zone entry
                if entry_frame is None or frame_idx - entry_frame >= 60:
                    dwell_ms = int((frames_in_zone / fps) * 1000)
                    
                    if visitor_id not in self.session_seq:
                        self.session_seq[visitor_id] = 1
                    
                    timestamp = timestamp_start + timedelta(
                        seconds=(frame_idx / fps)
                    )
                    
                    event = VisitorEvent(
                        store_id=store_id,
                        camera_id=camera_id,
                        visitor_id=visitor_id,
                        event_type=EventType.ZONE_DWELL,
                        timestamp=timestamp,
                        zone_id=zone_id,
                        dwell_ms=dwell_ms,
                        confidence=0.85,  # Tracking confidence
                        metadata=EventMetadata(
                            session_seq=self.session_seq[visitor_id]
                        )
                    )
                    
                    events.append(event)
                    self.visitor_zone_entry_frame[visitor_id][zone_id] = frame_idx
        
        return events
    
    def _detect_queue_events(
        self,
        frame_idx: int,
        fps: int,
        store_id: str,
        camera_id: str,
        timestamp_start: datetime
    ) -> List[VisitorEvent]:
        """
        Detect queue join/abandon events in billing zone.
        
        Args:
            frame_idx: Current frame index
            fps: Video FPS
            store_id: Store identifier
            camera_id: Camera identifier
            timestamp_start: Timestamp of video start
            
        Returns:
            List of BILLING_QUEUE events
        """
        events = []
        
        if not self.zone_mapper:
            return events
        
        # Count people in billing zone
        queue_depth = 0
        queue_visitors = []
        
        for visitor_id, track in self.tracker.get_active_tracks().items():
            if track.last_zone == self.QUEUE_ZONE:
                queue_depth += 1
                queue_visitors.append(visitor_id)
        
        # Emit queue join events
        for visitor_id in queue_visitors:
            if visitor_id not in self.billing_queue_join_frame:
                self.billing_queue_join_frame[visitor_id] = frame_idx
                
                if visitor_id not in self.session_seq:
                    self.session_seq[visitor_id] = 1
                
                timestamp = timestamp_start + timedelta(
                    seconds=(frame_idx / fps)
                )
                
                event = VisitorEvent(
                    store_id=store_id,
                    camera_id=camera_id,
                    visitor_id=visitor_id,
                    event_type=EventType.BILLING_QUEUE_JOIN,
                    timestamp=timestamp,
                    zone_id=self.QUEUE_ZONE,
                    confidence=0.85,
                    metadata=EventMetadata(
                        queue_depth=queue_depth,
                        session_seq=self.session_seq[visitor_id]
                    )
                )
                
                events.append(event)
        
        return events
    
    def process_video(
        self,
        video_path: str,
        store_id: str,
        camera_id: str,
        store_layout: Dict,
        fps_sample: int = 1,
        max_frames: Optional[int] = None
    ) -> List[VisitorEvent]:
        """
        Process CCTV video and emit structured events.
        
        Main pipeline:
        1. Extract frames from video
        2. Run YOLOv8 inference
        3. Classify detections to zones
        4. Track visitors across frames
        5. Generate events (ENTRY, EXIT, ZONE_DWELL, QUEUE)
        
        Args:
            video_path: Path to MP4 video file
            store_id: Store identifier (e.g., 'STORE_BLR_002')
            camera_id: Camera identifier (e.g., 'CAM_ENTRY_01')
            store_layout: Store layout dict with zone definitions
            fps_sample: Frame sampling rate (1=every frame, 15=every 15th)
            max_frames: Max frames to process (None for all)
            
        Returns:
            List of detected VisitorEvent objects
            
        Raises:
            RuntimeError: If processing fails
            IOError: If video file cannot be read
            ValueError: If store_layout is invalid
        """
        # Setup
        self.setup_zones(store_layout)
        self.tracker.reset()
        self.session_seq.clear()
        self.visitor_zone_entry_frame.clear()
        self.billing_queue_join_frame.clear()
        
        logger.info(f"Processing {video_path} for {camera_id}")
        
        try:
            # Extract frames
            frames, fps, frame_size = self._extract_frames_cv2(
                video_path,
                fps_sample=fps_sample,
                max_frames=max_frames
            )
            
            # YOLOv8 inference
            logger.info("Running YOLOv8 inference...")
            detections = self._run_inference(frames)
            
            # Zone classification
            logger.info("Classifying zones...")
            all_zones = self._classify_detections(detections)
            
            # Track and emit events
            logger.info("Tracking and generating events...")
            events = []
            timestamp_start = datetime.now(timezone.utc)
            
            for frame_idx, (frame_detections, frame_zones) in enumerate(
                zip(detections, all_zones)
            ):
                # Update tracker
                track_events = self.tracker.update(
                    frame_detections,
                    frame_zones,
                    frame_idx
                )
                
                # Convert track events to VisitorEvent
                for visitor_id, event_info in track_events.items():
                    if event_info["event"] == "ENTRY":
                        if visitor_id not in self.session_seq:
                            self.session_seq[visitor_id] = 1
                        else:
                            self.session_seq[visitor_id] += 1
                        
                        timestamp = timestamp_start + timedelta(
                            seconds=(frame_idx / (fps / fps_sample))
                        )
                        
                        event = VisitorEvent(
                            store_id=store_id,
                            camera_id=camera_id,
                            visitor_id=visitor_id,
                            event_type=EventType.ENTRY,
                            timestamp=timestamp,
                            zone_id=event_info["zone_id"],
                            confidence=0.90,
                            metadata=EventMetadata(
                                session_seq=self.session_seq[visitor_id]
                            )
                        )
                        
                        events.append(event)
                    
                    elif event_info["event"] == "EXIT":
                        timestamp = timestamp_start + timedelta(
                            seconds=(frame_idx / (fps / fps_sample))
                        )
                        
                        event = VisitorEvent(
                            store_id=store_id,
                            camera_id=camera_id,
                            visitor_id=visitor_id,
                            event_type=EventType.EXIT,
                            timestamp=timestamp,
                            zone_id=event_info["zone_id"],
                            confidence=0.90,
                            metadata=EventMetadata(
                                session_seq=self.session_seq.get(visitor_id, 1)
                            )
                        )
                        
                        events.append(event)
                
                # Detect dwell events
                dwell_events = self._detect_dwell_events(
                    frame_idx,
                    len(detections),
                    fps // fps_sample,
                    store_id,
                    camera_id,
                    timestamp_start
                )
                events.extend(dwell_events)
                
                # Detect queue events
                queue_events = self._detect_queue_events(
                    frame_idx,
                    fps // fps_sample,
                    store_id,
                    camera_id,
                    timestamp_start
                )
                events.extend(queue_events)
            
            logger.info(f"Generated {len(events)} events from {len(frames)} frames")
            
            return events
        
        except Exception as e:
            logger.exception(f"Error processing video: {str(e)}")
            raise
    
    def process_clip(
        self,
        video_path: str,
        store_id: str,
        camera_id: str,
        store_layout: Dict
    ) -> List[VisitorEvent]:
        """
        Convenience method: process video clip with default parameters.
        
        Args:
            video_path: Path to MP4 video file
            store_id: Store identifier
            camera_id: Camera identifier
            store_layout: Store layout dict
            
        Returns:
            List of VisitorEvent objects
        """
        return self.process_video(
            video_path=video_path,
            store_id=store_id,
            camera_id=camera_id,
            store_layout=store_layout,
            fps_sample=self.DEFAULT_FPS_SAMPLE
        )

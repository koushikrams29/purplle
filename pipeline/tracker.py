"""
Simple stateful tracker for visitor tracking in CCTV detection pipeline.

Tracks people across frames using centroid-based distance matching.
Assigns and maintains persistent visitor_id for each person.
"""

import uuid
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np


@dataclass
class TrackedPerson:
    """
    Represents a tracked person/visitor.
    
    Attributes:
        visitor_id: Unique identifier (VIS_XXXXXX format)
        last_centroid: (x, y) center of last detection
        bbox: Last detected bounding box (x1, y1, x2, y2)
        confidence: Detection confidence from last update
        frames_since_detection: Counter for disappearance tracking
        status: 'active', 'lost', or 'exited'
        last_updated: Timestamp of last detection
    """
    
    visitor_id: str
    last_centroid: Tuple[float, float]
    bbox: Tuple[float, float, float, float]
    confidence: float
    frames_since_detection: int = 0
    status: str = 'active'
    last_updated: datetime = field(default_factory=datetime.now)


class SimpleTracker:
    """
    Stateful centroid-based tracker for multi-person tracking.
    
    Maintains state across multiple update() calls, assigning persistent
    visitor IDs based on centroid-to-centroid distance matching.
    """
    
    def __init__(
        self,
        max_disappear_frames: int = 30,
        distance_threshold: float = 50.0
    ):
        """
        Initialize simple tracker.
        
        Args:
            max_disappear_frames: Max consecutive frames without detection before 'lost' → 'exited'
            distance_threshold: Max euclidean distance (pixels) to match to existing person
        """
        if max_disappear_frames < 1:
            raise ValueError("max_disappear_frames must be >= 1")
        if distance_threshold < 1.0:
            raise ValueError("distance_threshold must be >= 1.0")
        
        self.max_disappear_frames = max_disappear_frames
        self.distance_threshold = distance_threshold
        
        self.persons: Dict[str, TrackedPerson] = {}
        self._id_counter = 0
    
    @staticmethod
    def _get_centroid(bbox: Tuple[float, float, float, float]) -> Tuple[float, float]:
        """
        Calculate centroid from bounding box.
        
        Args:
            bbox: Tuple of (x1, y1, x2, y2)
            
        Returns:
            Tuple of (cx, cy) centroid coordinates
        """
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        return (cx, cy)
    
    @staticmethod
    def _euclidean_distance(
        p1: Tuple[float, float],
        p2: Tuple[float, float]
    ) -> float:
        """
        Calculate Euclidean distance between two points.
        
        Args:
            p1: Tuple of (x, y)
            p2: Tuple of (x, y)
            
        Returns:
            Distance value
        """
        return float(np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2))
    
    def _generate_visitor_id(self) -> str:
        """Generate unique visitor ID in VIS_XXXXXX format."""
        hex_part = uuid.uuid4().hex[:6]
        return f"VIS_{hex_part}"
    
    def _register_new_person(
        self,
        bbox: Tuple[float, float, float, float],
        confidence: float
    ) -> str:
        """
        Register a new tracked person.
        
        Args:
            bbox: Bounding box (x1, y1, x2, y2)
            confidence: Detection confidence
            
        Returns:
            Generated visitor_id
        """
        visitor_id = self._generate_visitor_id()
        centroid = self._get_centroid(bbox)
        
        person = TrackedPerson(
            visitor_id=visitor_id,
            last_centroid=centroid,
            bbox=bbox,
            confidence=confidence,
            frames_since_detection=0,
            status='active',
            last_updated=datetime.now()
        )
        
        self.persons[visitor_id] = person
        self._id_counter += 1
        
        return visitor_id
    
    def update(
        self,
        detections: List[Tuple[float, float, float, float, float]]
    ) -> List[Tuple[str, float, float, float, float, float]]:
        """
        Update tracker with new detections.
        
        Matches detections to existing persons using centroid distance.
        Marks unmatched persons as 'lost' and increments disappearance counter.
        
        Args:
            detections: List of (x1, y1, x2, y2, confidence) tuples
            
        Returns:
            List of (visitor_id, x1, y1, x2, y2, confidence) tuples for detected persons
            
        Raises:
            TypeError: If detections format is invalid
        """
        if not isinstance(detections, list):
            raise TypeError("detections must be a list")
        
        output = []
        
        if len(detections) == 0:
            # No detections - increment disappearance counter for all persons
            for visitor_id, person in list(self.persons.items()):
                person.frames_since_detection += 1
                
                if person.status == 'active':
                    person.status = 'lost'
                elif person.frames_since_detection > self.max_disappear_frames:
                    person.status = 'exited'
            
            return []
        
        # Match detections to existing persons
        matched_detections: Dict[int, str] = {}  # det_idx -> visitor_id
        matched_persons = set()
        
        for det_idx, detection in enumerate(detections):
            try:
                x1, y1, x2, y2, confidence = detection
                if not all(isinstance(v, (int, float)) for v in detection):
                    raise TypeError(f"Detection {det_idx}: all values must be numeric")
            except (ValueError, TypeError) as e:
                raise TypeError(f"Detection {det_idx}: {str(e)}")
            
            det_centroid = self._get_centroid((x1, y1, x2, y2))
            
            # Find closest matching person
            best_visitor_id = None
            best_distance = self.distance_threshold
            
            for visitor_id, person in self.persons.items():
                if visitor_id in matched_persons:
                    continue
                
                if person.status == 'exited':
                    continue
                
                distance = self._euclidean_distance(det_centroid, person.last_centroid)
                
                if distance < best_distance:
                    best_distance = distance
                    best_visitor_id = visitor_id
            
            if best_visitor_id is not None:
                # Match found - update existing person
                matched_detections[det_idx] = best_visitor_id
                matched_persons.add(best_visitor_id)
                
                person = self.persons[best_visitor_id]
                person.last_centroid = det_centroid
                person.bbox = (x1, y1, x2, y2)
                person.confidence = confidence
                person.frames_since_detection = 0
                person.status = 'active'
                person.last_updated = datetime.now()
                
                output.append((best_visitor_id, x1, y1, x2, y2, confidence))
        
        # Register unmatched detections as new persons
        for det_idx, detection in enumerate(detections):
            if det_idx not in matched_detections:
                x1, y1, x2, y2, confidence = detection
                visitor_id = self._register_new_person((x1, y1, x2, y2), confidence)
                output.append((visitor_id, x1, y1, x2, y2, confidence))
        
        # Update unmatched persons (increment disappearance)
        for visitor_id, person in self.persons.items():
            if visitor_id not in matched_persons:
                person.frames_since_detection += 1
                
                if person.status == 'active':
                    person.status = 'lost'
                elif person.frames_since_detection > self.max_disappear_frames:
                    person.status = 'exited'
        
        return output
    
    def get_visitor_status(self, visitor_id: str) -> Optional[str]:
        """
        Get status of a visitor.
        
        Args:
            visitor_id: Visitor identifier
            
        Returns:
            'active', 'lost', 'exited', or None if visitor not tracked
        """
        if visitor_id not in self.persons:
            return None
        
        return self.persons[visitor_id].status
    
    def mark_as_exited(self, visitor_id: str) -> bool:
        """
        Manually mark a visitor as exited.
        
        Args:
            visitor_id: Visitor identifier
            
        Returns:
            True if visitor existed and was marked, False otherwise
        """
        if visitor_id not in self.persons:
            return False
        
        self.persons[visitor_id].status = 'exited'
        return True
    
    def get_active_persons(self) -> Dict[str, TrackedPerson]:
        """Get all currently active persons."""
        return {
            vid: person 
            for vid, person in self.persons.items() 
            if person.status == 'active'
        }
    
    def get_lost_persons(self) -> Dict[str, TrackedPerson]:
        """Get all currently lost persons."""
        return {
            vid: person 
            for vid, person in self.persons.items() 
            if person.status == 'lost'
        }
    
    def get_all_persons(self) -> Dict[str, TrackedPerson]:
        """Get all tracked persons (any status)."""
        return dict(self.persons)
    
    def reset(self) -> None:
        """Reset tracker state."""
        self.persons.clear()
        self._id_counter = 0
    
    def get_stats(self) -> Dict:
        """
        Get tracker statistics.
        
        Returns:
            Dict with counts of active, lost, exited persons
        """
        active_count = sum(1 for p in self.persons.values() if p.status == 'active')
        lost_count = sum(1 for p in self.persons.values() if p.status == 'lost')
        exited_count = sum(1 for p in self.persons.values() if p.status == 'exited')
        
        return {
            'total_tracked': len(self.persons),
            'active': active_count,
            'lost': lost_count,
            'exited': exited_count
        }

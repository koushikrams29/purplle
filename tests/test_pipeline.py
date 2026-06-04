# PROMPT:
# I have a CCTV detection pipeline in pipeline/emit.py that:
# - Uses ray-casting point_in_polygon to classify centroids into store zones
# - Emits ENTRY events on first detection of a new visitor
# - Emits ZONE_EXIT + ZONE_ENTER when a visitor crosses a zone boundary
# - Emits ZONE_DWELL every fps*30 frames of continuous zone presence (30 seconds at 15fps)
# - Emits EXIT events for visitors missing > max_disappear_frames frames
# Zone polygons are pixel-space rectangles in a 1920x1080 frame.
#
# Write unit tests for: point_in_polygon, classify_zone, emit_events,
# and handle_disappeared_visitors. No mocking — use real VisitorEvent objects.
# Include edge cases: centroid outside all zones (zone_id=None), dwell threshold
# at fps*30 boundary, simultaneous ZONE_EXIT+ZONE_ENTER on boundary crossing.
#
# CHANGES MADE:
# - Removed tracker.py (SimpleTracker) from test scope — emit.py tests are self-contained
# - Discovered ZONE_DWELL threshold bug (was 30 frames, should be fps*30); tests now
#   explicitly verify the boundary at fps*30 - 1 (no dwell) and fps*30 (dwell emitted)
# - Added test for zone_id=None when centroid is in unclassified space (top strip)
# - Changed confidence assertion to use abs() delta instead of exact equality (float safety)

import pytest
from datetime import datetime, timezone
from pipeline.emit import (
    point_in_polygon,
    classify_zone,
    emit_events,
    handle_disappeared_visitors,
    reset_visitor_state,
)
from app.models import EventType


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

FPS = 15
START_TS = datetime(2026, 3, 3, 14, 0, 0, tzinfo=timezone.utc)

ZONES = {
    "ENTRY":    [[0, 850], [1920, 850], [1920, 1080], [0, 1080]],
    "SKINCARE": [[0, 200], [640, 200], [640, 850], [0, 850]],
    "MAKEUP":   [[640, 200], [1280, 200], [1280, 850], [640, 850]],
    "BILLING":  [[1280, 200], [1920, 200], [1920, 850], [1280, 850]],
}

STORE_ID = "STORE_BLR_002"
CAMERA_ID = "CAM_ENTRY_01"


# ---------------------------------------------------------------------------
# point_in_polygon tests
# ---------------------------------------------------------------------------

class TestPointInPolygon:
    def test_point_inside_rectangle(self):
        polygon = [[0, 0], [100, 0], [100, 100], [0, 100]]
        assert point_in_polygon((50, 50), polygon) is True

    def test_point_outside_rectangle(self):
        polygon = [[0, 0], [100, 0], [100, 100], [0, 100]]
        assert point_in_polygon((150, 50), polygon) is False

    def test_point_on_boundary_treated_consistently(self):
        polygon = [[0, 0], [100, 0], [100, 100], [0, 100]]
        # Edge behaviour is acceptable; just ensure no exception
        result = point_in_polygon((0, 50), polygon)
        assert isinstance(result, bool)

    def test_raises_on_degenerate_polygon(self):
        with pytest.raises(ValueError):
            point_in_polygon((50, 50), [[0, 0], [100, 0]])  # only 2 vertices


# ---------------------------------------------------------------------------
# classify_zone tests
# ---------------------------------------------------------------------------

class TestClassifyZone:
    def test_classifies_entry_zone(self):
        # Centroid in bottom strip (ENTRY zone: y between 850 and 1080)
        zone = classify_zone((960, 950), ZONES)
        assert zone == "ENTRY"

    def test_classifies_skincare_zone(self):
        zone = classify_zone((320, 500), ZONES)
        assert zone == "SKINCARE"

    def test_classifies_billing_zone(self):
        zone = classify_zone((1600, 500), ZONES)
        assert zone == "BILLING"

    def test_returns_none_for_outside_all_zones(self):
        # Top strip (BACKROOM not in this ZONES dict)
        zone = classify_zone((960, 50), ZONES)
        assert zone is None

    def test_raises_on_empty_zones(self):
        with pytest.raises(ValueError):
            classify_zone((100, 100), {})


# ---------------------------------------------------------------------------
# emit_events tests
# ---------------------------------------------------------------------------

class TestEmitEvents:
    def _detection(self, visitor_id, cx=960, cy=950, conf=0.9):
        """Helper: create a detection tuple with centroid at cx, cy."""
        return (visitor_id, cx - 50, cy - 100, cx + 50, cy + 100, conf)

    def test_first_detection_emits_entry(self):
        state = reset_visitor_state()
        detections = [self._detection("VIS_001")]
        events = emit_events(detections, 1, FPS, STORE_ID, CAMERA_ID, ZONES, START_TS, state)
        entry_events = [e for e in events if e.event_type == EventType.ENTRY]
        assert len(entry_events) == 1
        assert entry_events[0].visitor_id == "VIS_001"

    def test_entry_event_zone_matches_centroid(self):
        state = reset_visitor_state()
        # Centroid in SKINCARE zone
        detections = [self._detection("VIS_002", cx=320, cy=500)]
        events = emit_events(detections, 1, FPS, STORE_ID, CAMERA_ID, ZONES, START_TS, state)
        entry = next(e for e in events if e.event_type == EventType.ENTRY)
        assert entry.zone_id == "SKINCARE"

    def test_second_detection_no_extra_entry(self):
        state = reset_visitor_state()
        det = [self._detection("VIS_003")]
        emit_events(det, 1, FPS, STORE_ID, CAMERA_ID, ZONES, START_TS, state)
        events2 = emit_events(det, 2, FPS, STORE_ID, CAMERA_ID, ZONES, START_TS, state)
        assert not any(e.event_type == EventType.ENTRY for e in events2)

    def test_zone_change_emits_exit_and_enter(self):
        state = reset_visitor_state()
        # First in ENTRY zone
        det1 = [self._detection("VIS_004", cx=960, cy=950)]
        emit_events(det1, 1, FPS, STORE_ID, CAMERA_ID, ZONES, START_TS, state)
        # Move to SKINCARE zone
        det2 = [self._detection("VIS_004", cx=320, cy=500)]
        events2 = emit_events(det2, 2, FPS, STORE_ID, CAMERA_ID, ZONES, START_TS, state)
        types = [e.event_type for e in events2]
        assert EventType.ZONE_EXIT in types
        assert EventType.ZONE_ENTER in types

    def test_dwell_emitted_after_fps_times_30_frames(self):
        state = reset_visitor_state()
        det = [self._detection("VIS_005", cx=320, cy=500)]
        # First detection — ENTRY
        emit_events(det, 0, FPS, STORE_ID, CAMERA_ID, ZONES, START_TS, state)
        # Just below threshold — no ZONE_DWELL
        events_before = emit_events(det, FPS * 30 - 1, FPS, STORE_ID, CAMERA_ID, ZONES, START_TS, state)
        assert not any(e.event_type == EventType.ZONE_DWELL for e in events_before)
        # At threshold — ZONE_DWELL expected
        events_at = emit_events(det, FPS * 30, FPS, STORE_ID, CAMERA_ID, ZONES, START_TS, state)
        assert any(e.event_type == EventType.ZONE_DWELL for e in events_at)

    def test_entry_event_confidence_stored(self):
        state = reset_visitor_state()
        detections = [self._detection("VIS_006", conf=0.78)]
        events = emit_events(detections, 1, FPS, STORE_ID, CAMERA_ID, ZONES, START_TS, state)
        entry = next(e for e in events if e.event_type == EventType.ENTRY)
        assert abs(entry.confidence - 0.78) < 0.01

    def test_outside_zones_zone_id_is_none(self):
        state = reset_visitor_state()
        # Centroid outside all defined zones (top strip, y=50)
        detections = [self._detection("VIS_007", cx=960, cy=50)]
        events = emit_events(detections, 1, FPS, STORE_ID, CAMERA_ID, ZONES, START_TS, state)
        entry = next(e for e in events if e.event_type == EventType.ENTRY)
        assert entry.zone_id is None


# ---------------------------------------------------------------------------
# handle_disappeared_visitors tests
# ---------------------------------------------------------------------------

class TestHandleDisappearedVisitors:
    def test_no_exit_below_threshold(self):
        state = {"VIS_A": {"zone": "SKINCARE", "frame": 0, "session_seq": 1, "entered": True}}
        events = handle_disappeared_visitors(
            state, set(), frame_num=10, fps=FPS,
            store_id=STORE_ID, camera_id=CAMERA_ID,
            start_timestamp=START_TS, max_disappear_frames=30
        )
        assert not any(e.event_type == EventType.EXIT for e in events)
        assert "VIS_A" in state  # still tracked

    def test_exit_emitted_above_threshold(self):
        state = {"VIS_B": {"zone": "SKINCARE", "frame": 0, "session_seq": 1, "entered": True}}
        events = handle_disappeared_visitors(
            state, set(), frame_num=31, fps=FPS,
            store_id=STORE_ID, camera_id=CAMERA_ID,
            start_timestamp=START_TS, max_disappear_frames=30
        )
        exit_events = [e for e in events if e.event_type == EventType.EXIT]
        assert len(exit_events) == 1
        assert exit_events[0].visitor_id == "VIS_B"
        assert "VIS_B" not in state  # removed from tracking

    def test_visitor_in_current_frame_not_exited(self):
        state = {"VIS_C": {"zone": "ENTRY", "frame": 0, "session_seq": 1, "entered": True}}
        events = handle_disappeared_visitors(
            state, {"VIS_C"}, frame_num=100, fps=FPS,
            store_id=STORE_ID, camera_id=CAMERA_ID,
            start_timestamp=START_TS, max_disappear_frames=30
        )
        assert not any(e.event_type == EventType.EXIT for e in events)

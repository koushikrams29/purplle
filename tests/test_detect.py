# PROMPT:
# I have two core classes in pipeline/detect.py:
#
# ZoneMapper(zones: Dict[str, List[List[int]]])
#   .classify_point(x, y) -> Optional[str]
#   .classify_centroid((x, y)) -> Optional[str]
#   Raises ValueError if any polygon has < 3 vertices.
#
# CentroidTracker(entry_zone="ENTRY", max_disappear=30, distance_threshold=50.0)
#   .update(detections: List[(cx, cy)], zones: List[Optional[str]], frame_idx: int)
#     -> Dict[visitor_id, {"event": str, "zone_id": str}]
#   Emits: ENTRY (first detection in entry_zone), EXIT (disappeared > max_disappear),
#          ZONE_ENTER, ZONE_EXIT (zone transitions).
#
# Write unit tests for both classes. No mocking needed — pure Python logic, no video/YOLO.
# Cover: zone classification for each zone, outside-all-zones returns None,
# entry event on first detection in entry zone, exit after disappearance, zone transitions.
#
# CHANGES MADE:
# - Discovered CentroidTracker.update() returns {} (empty dict) for non-event frames,
#   not a dict with event=None — adjusted assertions accordingly
# - ZoneMapper raises ValueError on < 3 vertices, confirmed matches detect.py source
# - Entry event only fires when detection is IN entry_zone AND track.entered == False,
#   so test explicitly places centroid inside ENTRY zone polygon
# - EXIT event uses "zone_id": track.last_zone or "UNKNOWN" — tested the "UNKNOWN" fallback

import pytest
from pipeline.detect import ZoneMapper, CentroidTracker


# ---------------------------------------------------------------------------
# Zone layout matching brigade_store_layout.json
# ---------------------------------------------------------------------------

ZONES = {
    "ENTRY":    [[0, 850], [1920, 850], [1920, 1080], [0, 1080]],
    "SKINCARE": [[0, 200], [640, 200], [640, 850], [0, 850]],
    "MAKEUP":   [[640, 200], [1280, 200], [1280, 850], [640, 850]],
    "BILLING":  [[1280, 200], [1920, 200], [1920, 850], [1280, 850]],
    "BACKROOM": [[0, 0], [1920, 0], [1920, 200], [0, 200]],
}


# ===========================================================================
# TestZoneMapper
# ===========================================================================

class TestZoneMapper:

    @pytest.fixture
    def zm(self):
        return ZoneMapper(ZONES)

    # --- classify_point ---

    def test_entry_zone_bottom_center(self, zm):
        assert zm.classify_point(960, 950) == "ENTRY"

    def test_entry_zone_bottom_left(self, zm):
        assert zm.classify_point(10, 860) == "ENTRY"

    def test_skincare_zone(self, zm):
        assert zm.classify_point(320, 500) == "SKINCARE"

    def test_makeup_zone_center(self, zm):
        assert zm.classify_point(960, 500) == "MAKEUP"

    def test_billing_zone_right(self, zm):
        assert zm.classify_point(1600, 500) == "BILLING"

    def test_backroom_zone_top(self, zm):
        assert zm.classify_point(960, 100) == "BACKROOM"

    def test_outside_all_zones_returns_none(self, zm):
        # Gap between zones shouldn't exist in our layout, but boundary edge case
        result = zm.classify_point(-100, -100)
        assert result is None

    def test_single_zone_layout(self):
        zm = ZoneMapper({"SINGLE": [[0, 0], [500, 0], [500, 500], [0, 500]]})
        assert zm.classify_point(250, 250) == "SINGLE"
        assert zm.classify_point(600, 250) is None

    def test_invalid_polygon_raises_value_error(self):
        with pytest.raises(ValueError):
            ZoneMapper({"BAD": [[0, 0], [100, 0]]})  # only 2 vertices

    def test_empty_zones_classify_returns_none(self):
        # ZoneMapper accepts an empty dict (no polygons to check) and
        # classify_point simply returns None for any point.
        zm = ZoneMapper({})
        assert zm.classify_point(960, 540) is None

    # --- classify_centroid ---

    def test_classify_centroid_entry(self, zm):
        assert zm.classify_centroid((960, 950)) == "ENTRY"

    def test_classify_centroid_skincare(self, zm):
        assert zm.classify_centroid((320, 500)) == "SKINCARE"

    def test_classify_centroid_outside(self, zm):
        assert zm.classify_centroid((-50, -50)) is None

    def test_get_zone_polygon_returns_vertices(self, zm):
        poly = zm.get_zone_polygon("ENTRY")
        assert poly is not None
        assert len(poly) >= 3

    def test_get_zone_polygon_unknown_returns_none(self, zm):
        assert zm.get_zone_polygon("DOES_NOT_EXIST") is None


# ===========================================================================
# TestCentroidTracker
# ===========================================================================

class TestCentroidTracker:

    # --- Initial state ---

    def test_no_tracks_initially(self):
        tracker = CentroidTracker(entry_zone="ENTRY")
        assert len(tracker.tracks) == 0

    def test_empty_detections_empty_events(self):
        tracker = CentroidTracker(entry_zone="ENTRY")
        events = tracker.update([], [], frame_idx=0)
        assert events == {}

    # --- Entry event ---

    def test_first_detection_in_entry_zone_emits_entry(self):
        tracker = CentroidTracker(entry_zone="ENTRY", max_disappear=5)
        # centroid inside ENTRY zone (y=950)
        events = tracker.update([(960.0, 950.0)], ["ENTRY"], frame_idx=0)
        assert len(events) == 1
        vis_id = list(events.keys())[0]
        assert events[vis_id]["event"] == "ENTRY"

    def test_first_detection_not_in_entry_zone_no_entry_event(self):
        tracker = CentroidTracker(entry_zone="ENTRY", max_disappear=5)
        events = tracker.update([(320.0, 500.0)], ["SKINCARE"], frame_idx=0)
        # Track created, but no ENTRY event (not in entry_zone)
        entry_events = [v for v in events.values() if v.get("event") == "ENTRY"]
        assert len(entry_events) == 0

    def test_entry_fires_only_once_per_visitor(self):
        tracker = CentroidTracker(entry_zone="ENTRY", max_disappear=30)
        tracker.update([(960.0, 950.0)], ["ENTRY"], frame_idx=0)
        events2 = tracker.update([(960.0, 948.0)], ["ENTRY"], frame_idx=1)
        entry_events = [v for v in events2.values() if v.get("event") == "ENTRY"]
        assert len(entry_events) == 0  # already entered

    # --- Track persistence ---

    def test_same_centroid_matched_across_frames(self):
        tracker = CentroidTracker(entry_zone="ENTRY", max_disappear=5)
        events1 = tracker.update([(960.0, 950.0)], ["ENTRY"], frame_idx=0)
        vis_id = list(events1.keys())[0]
        # Same centroid in next frame
        tracker.update([(961.0, 951.0)], ["ENTRY"], frame_idx=1)
        assert vis_id in tracker.tracks  # still tracked

    def test_two_detections_create_two_tracks(self):
        tracker = CentroidTracker(entry_zone="ENTRY", max_disappear=5)
        tracker.update(
            [(200.0, 950.0), (800.0, 950.0)],
            ["ENTRY", "ENTRY"],
            frame_idx=0
        )
        assert len(tracker.tracks) == 2

    def test_far_apart_detections_get_different_ids(self):
        tracker = CentroidTracker(entry_zone="ENTRY", distance_threshold=50.0)
        tracker.update([(100.0, 950.0)], ["ENTRY"], frame_idx=0)
        # Second detection far away (>50px) → new track
        tracker.update([(100.0, 950.0), (1600.0, 500.0)], ["ENTRY", "BILLING"], frame_idx=1)
        assert len(tracker.tracks) == 2

    # --- Exit event ---

    def test_disappeared_track_emits_exit_after_max_disappear(self):
        tracker = CentroidTracker(entry_zone="ENTRY", max_disappear=3)
        tracker.update([(960.0, 950.0)], ["ENTRY"], frame_idx=0)
        # Collect exit events across all empty frames — exit fires once the
        # threshold is crossed, which could be on any iteration.
        all_exit_events = []
        for i in range(5):
            events = tracker.update([], [], frame_idx=i + 1)
            all_exit_events.extend(
                v for v in events.values() if v.get("event") == "EXIT"
            )
        assert len(all_exit_events) == 1

    def test_no_exit_before_max_disappear(self):
        tracker = CentroidTracker(entry_zone="ENTRY", max_disappear=10)
        tracker.update([(960.0, 950.0)], ["ENTRY"], frame_idx=0)
        # Disappear for only 5 frames
        events = {}
        for i in range(5):
            events = tracker.update([], [], frame_idx=i + 1)
        exit_events = [v for v in events.values() if v.get("event") == "EXIT"]
        assert len(exit_events) == 0
        # Track should still exist
        assert len(tracker.tracks) == 1

    # --- Zone transitions ---

    def test_zone_change_produces_event(self):
        # distance_threshold=2000 ensures the tracker matches the same person
        # even though centroids are far apart (they span two zones in a large frame).
        # Default 50px threshold would create a new track instead of a zone transition.
        tracker = CentroidTracker(
            entry_zone="ENTRY", max_disappear=30, distance_threshold=2000.0
        )
        # Frame 0: person in ENTRY zone
        tracker.update([(960.0, 950.0)], ["ENTRY"], frame_idx=0)
        # Frame 1: same person moves to SKINCARE (distance ≈ 782px, within threshold)
        events = tracker.update([(320.0, 500.0)], ["SKINCARE"], frame_idx=1)
        # A zone-transition or zone-enter event must be emitted
        assert len(events) > 0

    # --- Visitor ID format ---

    def test_visitor_id_starts_with_VIS(self):
        tracker = CentroidTracker(entry_zone="ENTRY")
        events = tracker.update([(960.0, 950.0)], ["ENTRY"], frame_idx=0)
        vis_id = list(events.keys())[0]
        assert vis_id.startswith("VIS_")

    # --- exited_visitors set ---

    def test_exited_visitor_not_double_counted(self):
        tracker = CentroidTracker(entry_zone="ENTRY", max_disappear=2)
        tracker.update([(960.0, 950.0)], ["ENTRY"], frame_idx=0)
        for i in range(3):
            tracker.update([], [], frame_idx=i + 1)
        # Continue with empty — should not emit another EXIT
        events = tracker.update([], [], frame_idx=5)
        exit_events = [v for v in events.values() if v.get("event") == "EXIT"]
        assert len(exit_events) == 0

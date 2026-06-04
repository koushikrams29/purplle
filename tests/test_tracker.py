# PROMPT:
# I have SimpleTracker in pipeline/tracker.py:
#
# SimpleTracker(max_disappear_frames=30, distance_threshold=50.0)
#   .update(detections: List[(x1,y1,x2,y2,confidence)])
#     -> List[(visitor_id, x1, y1, x2, y2, confidence)]
#   .get_active_persons() -> Dict[visitor_id, TrackedPerson]
#   .get_stats() -> Dict
#
# TrackedPerson has: visitor_id, last_centroid, bbox, confidence,
#   frames_since_detection, status ('active'|'lost'|'exited')
#
# Write unit tests. The tracker is pure Python + numpy — no video or YOLO needed.
# Cover: empty detections, new person registration, centroid matching across frames,
# status transitions (active→lost→exited), distance threshold boundary,
# get_stats counts, visitor_id format.
#
# CHANGES MADE:
# - Discovered update() returns List (not Dict), adjusted all assertions
# - Confirmed visitor_id format is "VIS_" + 6 hex chars
# - Status transitions: active→lost on first missed frame; lost→exited after
#   frames_since_detection > max_disappear_frames
# - Distance matching: detection within threshold → same visitor_id; outside → new id

import pytest
from pipeline.tracker import SimpleTracker


class TestSimpleTrackerInit:
    def test_default_construction(self):
        t = SimpleTracker()
        assert t.max_disappear_frames == 30
        assert t.distance_threshold == 50.0

    def test_custom_params(self):
        t = SimpleTracker(max_disappear_frames=10, distance_threshold=100.0)
        assert t.max_disappear_frames == 10
        assert t.distance_threshold == 100.0

    def test_invalid_max_disappear_raises(self):
        with pytest.raises(ValueError):
            SimpleTracker(max_disappear_frames=0)

    def test_invalid_distance_raises(self):
        with pytest.raises(ValueError):
            SimpleTracker(distance_threshold=0.5)

    def test_initially_empty(self):
        t = SimpleTracker()
        assert len(t.get_active_persons()) == 0


class TestSimpleTrackerUpdate:
    """Core update() behaviour."""

    def test_empty_detections_returns_empty(self):
        t = SimpleTracker()
        result = t.update([])
        assert result == []

    def test_single_detection_creates_new_person(self):
        t = SimpleTracker()
        result = t.update([(100.0, 100.0, 200.0, 200.0, 0.9)])
        assert len(result) == 1
        visitor_id, x1, y1, x2, y2, conf = result[0]
        assert visitor_id.startswith("VIS_")
        assert abs(x1 - 100.0) < 1e-9

    def test_two_detections_create_two_persons(self):
        t = SimpleTracker()
        result = t.update([
            (100.0, 100.0, 200.0, 200.0, 0.9),
            (800.0, 100.0, 900.0, 200.0, 0.8),
        ])
        assert len(result) == 2
        ids = {r[0] for r in result}
        assert len(ids) == 2  # distinct visitor IDs

    def test_same_person_same_id_across_frames(self):
        t = SimpleTracker(distance_threshold=100.0)
        result1 = t.update([(100.0, 100.0, 200.0, 200.0, 0.9)])
        id1 = result1[0][0]
        # Slightly moved centroid — still within threshold
        result2 = t.update([(105.0, 102.0, 205.0, 202.0, 0.88)])
        id2 = result2[0][0]
        assert id1 == id2

    def test_far_detection_gets_new_id(self):
        t = SimpleTracker(distance_threshold=50.0)
        result1 = t.update([(100.0, 100.0, 200.0, 200.0, 0.9)])
        id1 = result1[0][0]
        # Move far away (>50px centroid distance)
        result2 = t.update([(100.0, 100.0, 200.0, 200.0, 0.9),
                             (1500.0, 800.0, 1600.0, 900.0, 0.85)])
        ids = [r[0] for r in result2]
        assert id1 in ids  # original still matched
        assert len(set(ids)) == 2  # second is new

    def test_visitor_id_format(self):
        t = SimpleTracker()
        result = t.update([(100.0, 100.0, 200.0, 200.0, 0.9)])
        visitor_id = result[0][0]
        assert visitor_id.startswith("VIS_")
        suffix = visitor_id[4:]
        assert len(suffix) == 6
        assert all(c in "0123456789abcdef" for c in suffix)

    def test_confidence_preserved_in_output(self):
        t = SimpleTracker()
        result = t.update([(100.0, 100.0, 200.0, 200.0, 0.76)])
        assert abs(result[0][5] - 0.76) < 1e-6

    def test_bbox_preserved_in_output(self):
        t = SimpleTracker()
        result = t.update([(50.0, 60.0, 150.0, 160.0, 0.9)])
        _, x1, y1, x2, y2, _ = result[0]
        assert x1 == 50.0
        assert y1 == 60.0
        assert x2 == 150.0
        assert y2 == 160.0


class TestSimpleTrackerStatusTransitions:
    """Status transitions: active → lost → exited."""

    def test_active_after_detection(self):
        t = SimpleTracker(max_disappear_frames=5)
        t.update([(100.0, 100.0, 200.0, 200.0, 0.9)])
        person = list(t.get_active_persons().values())[0]
        assert person.status == "active"

    def test_status_lost_after_one_missed_frame(self):
        t = SimpleTracker(max_disappear_frames=5)
        result = t.update([(100.0, 100.0, 200.0, 200.0, 0.9)])
        vis_id = result[0][0]
        t.update([])  # missed frame
        person = t.persons[vis_id]
        assert person.status == "lost"

    def test_status_exited_after_max_disappear(self):
        t = SimpleTracker(max_disappear_frames=3)
        result = t.update([(100.0, 100.0, 200.0, 200.0, 0.9)])
        vis_id = result[0][0]
        for _ in range(4):
            t.update([])
        person = t.persons[vis_id]
        assert person.status == "exited"

    def test_exited_person_not_matched(self):
        t = SimpleTracker(max_disappear_frames=2, distance_threshold=200.0)
        result = t.update([(100.0, 100.0, 200.0, 200.0, 0.9)])
        vis_id = result[0][0]
        for _ in range(3):
            t.update([])
        # Detection at same location — should NOT reuse exited visitor's ID
        result2 = t.update([(100.0, 100.0, 200.0, 200.0, 0.9)])
        new_id = result2[0][0]
        assert new_id != vis_id

    def test_active_resets_after_redetection(self):
        t = SimpleTracker(max_disappear_frames=10, distance_threshold=100.0)
        result = t.update([(100.0, 100.0, 200.0, 200.0, 0.9)])
        vis_id = result[0][0]
        t.update([])  # lost
        t.update([(102.0, 102.0, 202.0, 202.0, 0.88)])  # re-detected
        person = t.persons[vis_id]
        assert person.status == "active"
        assert person.frames_since_detection == 0


class TestSimpleTrackerGetMethods:
    """get_active_persons(), get_stats(), get_all_persons()."""

    def test_get_active_persons_returns_active(self):
        t = SimpleTracker(max_disappear_frames=5)
        t.update([(100.0, 100.0, 200.0, 200.0, 0.9)])
        active = t.get_active_persons()
        assert len(active) == 1

    def test_get_active_excludes_lost(self):
        t = SimpleTracker(max_disappear_frames=5)
        t.update([(100.0, 100.0, 200.0, 200.0, 0.9)])
        t.update([])  # makes it lost
        active = t.get_active_persons()
        assert len(active) == 0

    def test_get_stats_structure(self):
        t = SimpleTracker()
        stats = t.get_stats()
        assert isinstance(stats, dict)
        assert "active" in stats or len(stats) > 0

    def test_get_all_persons_includes_lost(self):
        t = SimpleTracker(max_disappear_frames=5)
        t.update([(100.0, 100.0, 200.0, 200.0, 0.9)])
        t.update([])  # makes it lost
        all_persons = t.get_all_persons()
        assert len(all_persons) == 1

    def test_reset_clears_state(self):
        t = SimpleTracker()
        t.update([(100.0, 100.0, 200.0, 200.0, 0.9)])
        t.reset()
        assert len(t.persons) == 0
        assert len(t.get_active_persons()) == 0

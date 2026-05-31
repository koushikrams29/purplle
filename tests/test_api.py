# PROMPT:
# Create comprehensive tests for the Store Intelligence API.
#
# CHANGES MADE:
# - Added ingestion endpoint tests
# - Added metrics endpoint tests
# - Added funnel endpoint tests
# - Added anomaly endpoint tests
# - Added health endpoint tests

from datetime import datetime

from app.db_models import EventRecord


class TestEventIngestion:
    """
    Tests for POST /events/ingest
    """

    def test_ingest_valid_events(
        self,
        client,
        sample_event
    ):
        response = client.post(
            "/events/ingest",
            json={
                "events": [sample_event]
            }
        )

        assert response.status_code == 200

        data = response.json()

        assert data["status"] == "ok"
        assert data["ingested"] == 1
        assert data["failed"] == 0

    def test_ingest_invalid_schema(
        self,
        client
    ):
        response = client.post(
            "/events/ingest",
            json={
                "events": [
                    {
                        "bad_field": "invalid"
                    }
                ]
            }
        )

        assert response.status_code in [200, 422]

    def test_ingest_duplicate_idempotency(
        self,
        client,
        sample_event
    ):
        client.post(
            "/events/ingest",
            json={
                "events": [sample_event]
            }
        )

        response = client.post(
            "/events/ingest",
            json={
                "events": [sample_event]
            }
        )

        data = response.json()

        assert data["ingested"] == 0

    def test_ingest_malformed_batch(
        self,
        client
    ):
        response = client.post(
            "/events/ingest",
            json={
                "wrong_key": []
            }
        )

        assert response.status_code in [400, 422]

    def test_ingest_partial_success(
        self,
        client,
        sample_event
    ):
        response = client.post(
            "/events/ingest",
            json={
                "events": [
                    sample_event,
                    {
                        "bad": "event"
                    }
                ]
            }
        )

        assert response.status_code == 200

        data = response.json()

        assert data["ingested"] >= 1
        assert data["failed"] >= 1


class TestMetrics:
    """
    Tests for metrics endpoint
    """

    def test_metrics_empty_store(
        self,
        client
    ):
        response = client.get(
            "/stores/EMPTY_STORE/metrics"
        )

        assert response.status_code == 200

        data = response.json()

        assert data["unique_visitors"] == 0

    def test_metrics_with_visitors(
        self,
        client,
        sample_event
    ):
        client.post(
            "/events/ingest",
            json={
                "events": [sample_event]
            }
        )

        response = client.get(
            "/stores/STORE_BLR_002/metrics"
        )

        assert response.status_code == 200

        data = response.json()

        assert data["unique_visitors"] >= 1

    def test_metrics_zero_purchases(
        self,
        client,
        sample_event
    ):
        client.post(
            "/events/ingest",
            json={
                "events": [sample_event]
            }
        )

        response = client.get(
            "/stores/STORE_BLR_002/metrics"
        )

        data = response.json()

        assert data["conversion_rate"] == 0

    def test_metrics_zero_division(
        self,
        client
    ):
        response = client.get(
            "/stores/NO_DATA/metrics"
        )

        data = response.json()

        assert data["conversion_rate"] == 0


class TestFunnel:
    """
    Tests for funnel endpoint
    """

    def test_funnel_single_visitor(
        self,
        client,
        sample_event
    ):
        client.post(
            "/events/ingest",
            json={
                "events": [sample_event]
            }
        )

        response = client.get(
            "/stores/STORE_BLR_002/funnel"
        )

        assert response.status_code == 200

        data = response.json()

        assert data["funnel"][0]["count"] >= 1

    def test_funnel_multiple_visitors(
        self,
        client,
        sample_event
    ):
        event2 = sample_event.copy()
        event2["event_id"] = "event-002"
        event2["visitor_id"] = "VIS_002"

        client.post(
            "/events/ingest",
            json={
                "events": [
                    sample_event,
                    event2
                ]
            }
        )

        response = client.get(
            "/stores/STORE_BLR_002/funnel"
        )

        data = response.json()

        assert data["funnel"][0]["count"] >= 2

    def test_funnel_no_conversion(
        self,
        client,
        sample_event
    ):
        client.post(
            "/events/ingest",
            json={
                "events": [sample_event]
            }
        )

        response = client.get(
            "/stores/STORE_BLR_002/funnel"
        )

        data = response.json()

        assert data["total_conversion_rate"] == 0

    def test_funnel_drop_off_calc(
        self,
        client,
        sample_event
    ):
        client.post(
            "/events/ingest",
            json={
                "events": [sample_event]
            }
        )

        response = client.get(
            "/stores/STORE_BLR_002/funnel"
        )

        data = response.json()

        assert "drop_off_pct" in data["funnel"][0]


class TestAnomalies:
    """
    Tests for anomaly endpoints
    """

    def test_anomaly_none(
        self,
        client
    ):
        response = client.get(
            "/stores/EMPTY_STORE/anomalies"
        )

        assert response.status_code == 200

        data = response.json()

        assert "anomalies" in data


class TestHealth:
    """
    Tests for health endpoint
    """

    def test_health_endpoint_normal(
        self,
        client
    ):
        response = client.get(
            "/health"
        )

        assert response.status_code == 200

        data = response.json()

        assert data["status"] == "healthy"

    def test_health_stale_feed_warning(
        self,
        client,
        db
    ):
        old_event = EventRecord(
            event_id="old-event",
            store_id="STORE_BLR_002",
            camera_id="CAM_ENTRY_01",
            visitor_id="VIS_OLD",
            event_type="ENTRY",
            timestamp=datetime(2025, 1, 1),
            zone_id=None,
            dwell_ms=0,
            is_staff=False,
            confidence=1.0,
            metadata_json={}
        )

        db.add(old_event)
        db.commit()

        response = client.get(
            "/health"
        )

        assert response.status_code == 200

        data = response.json()

        assert "warnings" in data
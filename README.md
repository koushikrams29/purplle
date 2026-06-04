# Store Intelligence API

Real-time retail analytics pipeline — from raw CCTV footage to a queryable store intelligence API.

## Quick Start (5 commands)

```bash
git clone <repo-url> && cd store-intelligence

cp -r /path/to/resources/cctv_clips data/cctv/          # place video files

docker compose up --build -d                             # starts API on :8000

python pipeline/run_detection.py \
  --video_dir data/cctv/ \
  --store_id STORE_BLR_002 \
  --store_layout scripts/brigade_store_layout.json \
  --output events/events.jsonl                           # run detection → events

python scripts/feed_events.py events/events.jsonl        # feed events into API
```

API is live at http://localhost:8000 · Docs at http://localhost:8000/docs

---

## Architecture

```
CCTV Clips
    │
    ▼
pipeline/run_detection.py          # YOLOv8 person detection + centroid tracking
    │  (emits structured JSONL events)
    ▼
POST /events/ingest                # FastAPI ingestion endpoint
    │  (validates, deduplicates, stores, correlates with POS)
    ▼
SQLite (store.db)
    │
    ├─ GET /stores/{id}/metrics    # unique visitors, conversion rate, dwell, queue
    ├─ GET /stores/{id}/funnel     # 4-stage funnel with drop-off %
    ├─ GET /stores/{id}/heatmap    # zone frequency + dwell, normalised 0-100
    ├─ GET /stores/{id}/anomalies  # queue spike, conversion drop, dead zones
    └─ GET /health                 # service status, lag, STALE_FEED warning
```

---

## Running the Detection Pipeline

### Process a single video clip

```bash
python pipeline/run_detection.py \
  --video path/to/clip.mp4 \
  --store_id STORE_BLR_002 \
  --camera_id CAM_ENTRY_01 \
  --store_layout scripts/brigade_store_layout.json \
  --output events/entry_events.jsonl \
  --model yolov8m.pt \
  --confidence 0.5
```

### Process all clips in a directory

```bash
bash pipeline/run.sh
```

### Feed events into the API

```bash
# Feed a JSONL file (one event JSON per line)
python scripts/feed_events.py events/events.jsonl

# Or POST directly
curl -X POST http://localhost:8000/events/ingest \
  -H "Content-Type: application/json" \
  -d '{"events": [...]}'
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/events/ingest` | Batch ingest up to 500 events. Idempotent by event_id. |
| GET | `/stores/{id}/metrics` | Unique visitors, conversion rate, avg dwell, queue depth |
| GET | `/stores/{id}/funnel` | 4-stage conversion funnel with drop-off percentages |
| GET | `/stores/{id}/heatmap` | Zone visit frequency + dwell, normalised 0–100 |
| GET | `/stores/{id}/anomalies` | Queue spike, conversion drop, dead zones, abandonment |
| GET | `/health` | Service health, last event timestamp, STALE_FEED warning |

### Example: Ingest events

```bash
curl -X POST http://localhost:8000/events/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "events": [{
      "event_id": "550e8400-e29b-41d4-a716-446655440000",
      "store_id": "STORE_BLR_002",
      "camera_id": "CAM_ENTRY_01",
      "visitor_id": "VIS_c8a2f1",
      "event_type": "ENTRY",
      "timestamp": "2026-03-03T14:22:10Z",
      "zone_id": null,
      "dwell_ms": 0,
      "is_staff": false,
      "confidence": 0.91,
      "metadata": {"queue_depth": null, "sku_zone": null, "session_seq": 1}
    }]
  }'
```

### Example: Get store metrics

```bash
curl http://localhost:8000/stores/STORE_BLR_002/metrics
```

---

## Event Schema

```json
{
  "event_id":   "uuid-v4",
  "store_id":   "STORE_BLR_002",
  "camera_id":  "CAM_ENTRY_01",
  "visitor_id": "VIS_c8a2f1",
  "event_type": "ZONE_DWELL",
  "timestamp":  "2026-03-03T14:22:10Z",
  "zone_id":    "SKINCARE",
  "dwell_ms":   8400,
  "is_staff":   false,
  "confidence": 0.91,
  "metadata": {
    "queue_depth": null,
    "sku_zone":    "MOISTURISER",
    "session_seq": 5
  }
}
```

**Supported event types:** `ENTRY`, `EXIT`, `ZONE_ENTER`, `ZONE_EXIT`, `ZONE_DWELL`, `BILLING_QUEUE_JOIN`, `BILLING_QUEUE_ABANDON`, `REENTRY`

---

## Live Dashboard

```bash
python scripts/dashboard.py --store STORE_BLR_002
```

Displays a Rich terminal dashboard with real-time metrics, anomalies, and data freshness. Refreshes every 3 seconds.

For the bonus live experience, run the dashboard while the detection pipeline is processing clips in parallel.

---

## Configuration

Environment variables (set in `docker-compose.yml` or `.env`):

| Variable | Default | Description |
|----------|---------|-------------|
| `DATABASE_URL` | `sqlite:///./store.db` | SQLAlchemy DB connection string |
| `YOLO_MODEL` | `yolov8m` | YOLOv8 model variant |
| `LOG_LEVEL` | `INFO` | Logging level |
| `POS_DATA_PATH` | _(auto-detected)_ | Path to POS transactions CSV |
| `ENVIRONMENT` | `production` | Environment tag |

---

## Running Tests

```bash
pip install -r requirements.txt
pytest tests/ -v --tb=short
```

For coverage report:
```bash
pytest tests/ --cov=app --cov=pipeline --cov-report=term-missing
```

---

## POS Data Format

Place your POS CSV at `data/pos_transactions.csv`. The loader supports:

```
order_id, order_date, order_time, store_id, product_id, brand_name, total_amount
1, 10-04-2026, 12:15:05, STORE_BLR_002, 399945, Faces Canada, 302.33
```

The API automatically correlates billing zone events with POS transactions in a 5-minute window to compute conversion rates.

---

## Project Structure

```
store-intelligence/
├── pipeline/
│   ├── detect.py          # YOLOv8 + centroid tracking
│   ├── tracker.py         # Multi-person tracker
│   ├── emit.py            # Event generation
│   ├── run_detection.py   # CLI entry point
│   └── run.sh             # One-command batch runner
├── app/
│   ├── main.py            # FastAPI app + all endpoints
│   ├── models.py          # Pydantic event schema
│   ├── ingestion.py       # Ingest, dedup, POS correlation
│   ├── metrics.py         # Real-time metric computation
│   ├── funnel.py          # Funnel + session logic
│   ├── heatmap.py         # Zone heatmap analytics
│   ├── anomalies.py       # Anomaly detection
│   ├── db_models.py       # SQLAlchemy ORM models
│   ├── database.py        # DB setup and session
│   ├── config.py          # Settings management
│   └── logging_config.py  # Structured JSON logging
├── scripts/
│   ├── dashboard.py       # Rich live dashboard
│   ├── load_pos.py        # POS CSV normaliser
│   └── brigade_store_layout.json
├── tests/
│   ├── conftest.py
│   ├── test_api.py
│   ├── test_pipeline.py
│   ├── test_metrics.py
│   └── test_anomalies.py
├── docs/
│   ├── DESIGN.md
│   └── CHOICES.md
├── docker-compose.yml
├── Dockerfile
└── requirements.txt
```

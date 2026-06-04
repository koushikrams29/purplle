# System Design — Store Intelligence API

## Overview

The Store Intelligence system converts raw CCTV footage into real-time retail analytics. It consists of four stages:

```
Raw CCTV Clips
      │
      ▼
Detection Layer (pipeline/)
  YOLOv8m person detection @ 15fps
  Centroid-based multi-person tracking
  Zone polygon classification (ray-casting)
  Event generation: ENTRY / EXIT / ZONE_DWELL / QUEUE / REENTRY
      │  (JSONL file or HTTP POST)
      ▼
Event Ingest (POST /events/ingest)
  Pydantic validation + deduplication by event_id
  Visitor session lifecycle management
  POS transaction correlation (5-minute billing window)
      │
      ▼
SQLite (store.db)
  EventRecord — every raw event, indexed by store_id + timestamp + visitor_id
  VisitorSession — aggregated per-visit sessions with zone journeys
  ConversionRecord — POS-matched purchases
      │
      ▼
Intelligence API (app/)
  GET /metrics   — unique visitors, conversion rate, dwell, queue
  GET /funnel    — 4-stage funnel with drop-off %
  GET /heatmap   — zone heat scores 0-100
  GET /anomalies — spike / drop / dead-zone detection
  GET /health    — lag check, STALE_FEED warning
      │
      ▼
Live Dashboard (scripts/dashboard.py)
  Rich terminal UI, 3-second refresh
```

---

## Detection Layer Design

### Person Detection
YOLOv8m (medium variant) was chosen for the person detection backbone. It runs on CPU in inference-only mode, processing sampled frames (every Nth frame, configurable) to balance accuracy vs. throughput. Each detection produces a bounding box and confidence score for class 0 (person).

### Tracking
A centroid-based tracker maintains persistent visitor IDs (`VIS_XXXXXX`) across frames. Each track is matched to the nearest unassigned detection within a configurable distance threshold (default 50px). Visitors are declared "exited" after `max_disappear_frames` (default 30) consecutive frames of absence, at which point an EXIT event is emitted and the track is purged.

### Zone Classification
Each frame's person centroids are classified into store zones using a ray-casting point-in-polygon algorithm. Zone polygons are defined in `scripts/brigade_store_layout.json` as pixel-space coordinates at 1920×1080 resolution. Zones are mutually exclusive and non-overlapping (ENTRY at bottom strip, SKINCARE/MAKEUP/BILLING across the mid-section, BACKROOM at the top).

### Event Emission Rules
- **ENTRY**: First frame a new track appears
- **ZONE_ENTER / ZONE_EXIT**: Zone boundary crossing detected frame-to-frame
- **ZONE_DWELL**: Emitted every 30 seconds (fps × 30 frames) of continuous zone presence
- **EXIT**: Track disappears for > `max_disappear_frames` frames
- **REENTRY**: Same visitor_id re-detected after a prior EXIT (Re-ID via centroid trajectory)
- **BILLING_QUEUE_JOIN**: Centroid enters BILLING zone while queue_depth > 0
- **BILLING_QUEUE_ABANDON**: Visitor leaves BILLING zone without a subsequent POS transaction (post-processing correlation)

---

## API Design

### Storage: SQLite
SQLite was chosen for zero-infrastructure setup (no separate DB container). Three tables: `event_records`, `visitor_sessions`, `conversion_records`. Composite indexes on `(store_id, timestamp)` and `(visitor_id, timestamp)` support all query patterns efficiently for single-store analytics loads.

For production at 40 stores with continuous event streams, SQLite would be replaced by PostgreSQL with time-partitioned `event_records` and a TSDB-backed metrics cache layer.

### Idempotency
`POST /events/ingest` checks for existing `event_id` before inserting. Duplicate events are silently skipped and not counted in `ingested`. The same payload can be sent twice with identical results — safe for at-least-once delivery from the pipeline.

### Session Logic
Each `ENTRY` event opens a `VisitorSession`. `ZONE_DWELL` and `ZONE_ENTER` events append to `zones_visited`. `EXIT` closes the session with `exit_time`. `REENTRY` closes any lingering open session and opens a new one with `is_reentry=True`. This prevents re-entry from inflating unique visitor counts.

### POS Correlation
POS transactions are loaded from CSV at API startup. After each ingest batch, sessions with BILLING zone visits are correlated against POS records within a ±5-minute window by `(store_id, transaction_time)`. A matched session has `converted=True`, driving the conversion rate metric.

### Anomaly Detection
Four checks run per request to `/anomalies` (no background jobs, no caching):
1. **Queue spike** — peak `queue_depth` in `metadata_json` of `BILLING_QUEUE_JOIN` events
2. **Conversion drop** — today's rate vs. 7-day baseline; skip if baseline is zero
3. **Dead zone** — zone with no `ZONE_DWELL` in the past 30 minutes
4. **Abandonment** — queue abandon rate > 15%

Results are sorted CRITICAL → WARN → INFO. Each anomaly includes a `suggested_action` string.

---

## AI-Assisted Decisions

### 1. Zone Classification Approach
I asked Claude to evaluate three approaches: (a) pixel polygon rules, (b) homography-based real-world coordinate mapping, (c) VLM-based zone prompting per frame. Claude recommended approach (a) as the most reliable for fixed-camera setups where the store layout is known. I agreed — the store layout JSON already defines pixel polygons, and ray-casting is deterministic and fast. I rejected the VLM option for runtime zone classification because it would require an API call per frame (prohibitively slow) and because polygons are more auditable.

### 2. Event Schema — `zone_id` Nullability
Claude initially generated a schema with `zone_id: str` as a required field. I overrode this — ENTRY and EXIT events don't have a meaningful zone (they represent crossing the threshold), and forcing a non-null zone_id would require a sentinel value like `"UNKNOWN"` which pollutes zone analytics. The final schema has `zone_id: Optional[str]`. Claude's first suggestion would have broken the funnel query that checks `zone_id == "BILLING"`.

### 3. REENTRY Session Handling
I described the re-entry problem to Claude: same person exits, comes back, should count as one unique visitor but flag the re-visit. Claude's first suggestion was to set `is_reentry=True` on the existing active session. I rejected this — a visitor who re-enters has no active session (it was closed by EXIT). The correct behaviour is to create a new session with `is_reentry=True`, and deduplicate visitor counts by visitor_id (not session count). Claude revised its suggestion after I explained this, and the revised logic was correct.

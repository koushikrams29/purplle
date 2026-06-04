# Architectural Choices

Three decisions that shaped the system, with the full reasoning behind each.

---

## Decision 1: Detection Model — YOLOv8m

### Options Considered
1. **YOLOv8n (nano)** — fastest, but detection quality degrades on partially occluded people and in low-light conditions present in the retail footage
2. **YOLOv8m (medium)** — 2× slower than nano but substantially better on crowded scenes and partial occlusion
3. **YOLOv9 / RT-DETR** — marginally better accuracy, but significantly larger, harder to containerise, and overkill for 15fps retail CCTV
4. **MediaPipe Pose** — fast but tuned for visible full-body pose, not bounding-box counting; degrades on head/shoulder-only visibility in billing queues

### What AI Suggested
Claude suggested YOLOv8n for speed and YOLOv8m as a balanced option, noting that YOLOv8m is the community-accepted starting point for crowd counting in retail surveillance. Claude flagged that the partial occlusion edge case (people behind displays) would be better handled by the medium variant's larger receptive field.

### What I Chose and Why
**YOLOv8m.** The retail footage includes the specific challenges the problem statement lists: group entry (3–4 people), billing queue buildup, and partial occlusion by displays. Nano's detection drops to ~60% recall in those conditions based on published benchmarks. Medium trades some speed for recall, and since the pipeline can batch-process clips (not strict real-time), the extra compute is acceptable. The model weights (`yolov8m.pt`) are included in the repo for reproducibility.

**If I used a VLM:** I tested GPT-4V for staff detection — prompting it with a cropped bounding box and asking "Is this person wearing a store uniform or staff badge?" The prompt was: *"Look at this person in a retail store setting. Are they likely store staff? Indicators: lanyard/badge, uniform shirt, staff vest. Answer: staff / customer / uncertain."* It worked on clear cases (~85% accuracy on the debug frames) but was too slow (2–3s per crop) and too expensive for per-frame use. I moved staff detection to a rule-based heuristic (BACKROOM zone presence > 50% of session time → likely staff) instead.

---

## Decision 2: Event Schema Design

### Options Considered
1. **Flat event per type** — separate tables/schemas per event type (EntryEvent, DwellEvent, etc.)
2. **Single unified event schema** — one schema with optional fields, a type discriminator, and a `metadata` bag for type-specific fields
3. **Partially typed schema** — typed events at ingestion, flattened to single table at storage

### What AI Suggested
Claude recommended option 2 (unified schema) for this use case — it simplifies the ingest endpoint (one validator, one table), makes funnel and metrics queries straightforward SQL, and the `metadata: {queue_depth, sku_zone, session_seq}` bag handles the few type-specific fields. Claude explicitly warned against option 1 as it would require N separate ingest endpoints and make cross-event queries (e.g., "how long between ZONE_ENTER and EXIT") complex joins.

### What I Chose and Why
**Unified schema (option 2), with one critical override of Claude's suggestion.**

Claude's initial schema had `zone_id` as required with `min_length=1`. I overrode this because:
- ENTRY and EXIT events represent threshold crossing — there is no meaningful zone
- Forcing `zone_id="UNKNOWN"` or `zone_id="ENTRY"` (a zone name, not the event's zone) confuses zone analytics
- The problem statement example explicitly shows `"zone_id": null` for ENTRY events

I made `zone_id: Optional[str]` and updated all callers. This was the right call — the `/funnel` endpoint queries `zone_id == "BILLING"` specifically, and a non-null sentinel would have polluted that filter.

The `metadata` bag uses `extra="forbid"` on `EventMetadata` to prevent silent field expansion — fields must be explicitly added to the schema, not smuggled in as arbitrary JSON.

---

## Decision 3: API Architecture — SQLite vs. PostgreSQL, Synchronous vs. Async Metrics

### Options Considered
1. **SQLite + synchronous metric queries** — simple, zero infrastructure, runs in-container
2. **PostgreSQL + synchronous queries** — production-grade, but requires a separate container, migrations, connection pooling
3. **SQLite + pre-aggregated metrics (background worker)** — cache metrics at ingest time, serve from a pre-computed table; faster reads, stale on ingest failure
4. **SQLite + synchronous queries + time-windowed caching** — compute on request, cache for N seconds; simple to implement

### What AI Suggested
Claude suggested PostgreSQL for production and SQLite for the challenge context, with a note that pre-aggregated metrics (option 3) would be necessary at 40 stores × real-time event volume. Claude specifically recommended against serving raw SQL aggregations at request time in production because at high event rates, GROUP BY queries on an unpartitioned event table would slow to seconds.

### What I Chose and Why
**SQLite + synchronous metric queries (option 1)** for this submission, with explicit acknowledgment of the production limitation.

Reasons:
1. The acceptance gate requires `docker compose up` with no manual steps. PostgreSQL adds a container, a migration step, and a connection string — all of which are failure points in a fresh-clone evaluation.
2. The event volume from 5 stores × 3 cameras × 20-minute clips is modest enough that SQL aggregations return in milliseconds on SQLite.
3. The problem FAQ explicitly permits SQLite: *"SQLite is fine."*

**What would break at 40 live stores in production:** The `GET /metrics` handler runs five `GROUP BY` aggregations synchronously on `event_records`. At 40 stores × 15fps × 3 cameras = 1,800 events/second, the table would have ~6.5M rows per hour. Without time-partitioning and a pre-aggregated metrics table updated at ingest time, query latency would grow to seconds within an hour. The fix: add a `store_metrics_cache` table updated transactionally with each ingest batch, served as the primary response, with the live SQL aggregation as a fallback.

**One thing I overrode AI on:** Claude suggested adding a Redis cache layer for metrics in front of SQLite. I rejected this — it adds infrastructure complexity, requires cache invalidation logic, and introduces potential stale-read bugs (serving yesterday's conversion rate). For a challenge submission, clarity and correctness of the on-request computation is more valuable than the performance optimisation. If I were building for production, the Redis suggestion would be revisited.

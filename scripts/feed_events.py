"""
Feed a JSONL events file into the Store Intelligence API.

Usage:
    python scripts/feed_events.py events/events.jsonl
    python scripts/feed_events.py events/events.jsonl --api http://localhost:8000 --batch 200
"""

import argparse
import json
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path


def feed_events(jsonl_path: str, api_url: str, batch_size: int) -> None:
    path = Path(jsonl_path)
    if not path.exists():
        print(f"File not found: {jsonl_path}", file=sys.stderr)
        sys.exit(1)

    lines = path.read_text(encoding="utf-8").splitlines()
    events = []
    for line in lines:
        line = line.strip()
        if line:
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"Skipping invalid JSON line: {e}", file=sys.stderr)

    if not events:
        print("No events found in file.")
        return

    total = len(events)
    ingested = 0
    failed = 0
    batches = (total + batch_size - 1) // batch_size

    print(f"Feeding {total} events to {api_url}/events/ingest in {batches} batches...")

    for i in range(0, total, batch_size):
        batch = events[i:i + batch_size]
        payload = json.dumps({"events": batch}).encode("utf-8")

        try:
            req = urllib.request.Request(
                f"{api_url}/events/ingest",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read())
                ingested += data.get("ingested", 0)
                failed += data.get("failed", 0)
                print(
                    f"Batch {i // batch_size + 1}/{batches}: "
                    f"ingested={data.get('ingested', 0)}, "
                    f"failed={data.get('failed', 0)}"
                )
        except urllib.error.URLError as e:
            print(f"Batch {i // batch_size + 1} failed: {e}", file=sys.stderr)
            failed += len(batch)

        time.sleep(0.05)  # gentle rate limiting

    print(f"\nDone. Total ingested: {ingested}, Total failed: {failed}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Feed JSONL events into the API")
    parser.add_argument("jsonl_file", help="Path to events JSONL file")
    parser.add_argument("--api", default="http://localhost:8000", help="API base URL")
    parser.add_argument("--batch", type=int, default=200, help="Batch size (max 500)")
    args = parser.parse_args()

    feed_events(args.jsonl_file, args.api, min(args.batch, 500))

#!/usr/bin/env bash
# One-command script to process all CCTV clips through the detection pipeline
# and write events to events/events.jsonl
#
# Usage:
#   bash pipeline/run.sh [--video_dir data/cctv] [--store_id STORE_BLR_002] [--output events/events.jsonl]
#
# Requires: Python 3.11+, YOLOv8, OpenCV (see requirements.txt)

set -euo pipefail

VIDEO_DIR="${1:-data/cctv}"
STORE_ID="${2:-STORE_BLR_002}"
STORE_LAYOUT="scripts/brigade_store_layout.json"
OUTPUT="${3:-events/events.jsonl}"
MODEL="${4:-yolov8m.pt}"
CONFIDENCE="${5:-0.5}"

mkdir -p "$(dirname "$OUTPUT")"

echo "=== Store Intelligence — Detection Pipeline ==="
echo "Video directory : $VIDEO_DIR"
echo "Store ID        : $STORE_ID"
echo "Output          : $OUTPUT"
echo "Model           : $MODEL"
echo ""

# Clear output file for fresh run
> "$OUTPUT"

# Camera ID mapping — explicit ordered if-elif (most specific patterns FIRST)
# Confirmed camera-to-zone assignments:
#   CAM 1 = SKINCARE  | CAM 2 = MAKEUP  | CAM 3 = ENTRY
#   CAM 4 = BACKROOM  | CAM 5 = BILLING
# NOTE: bash associative arrays iterate in random order, so we use if-elif
# to guarantee the more specific pattern (zone1) is checked before the
# substring it contains (zone). Order matters here.

get_camera_id() {
    local f="$1"
    if   [[ "$f" == *"zone1"*    ]]; then echo "CAM_SKINCARE_01"
    elif [[ "$f" == *"cam1"*     ]]; then echo "CAM_SKINCARE_01"
    elif [[ "$f" == *"zone2"*    ]]; then echo "CAM_MAKEUP_02"
    elif [[ "$f" == *"cam2"*     ]]; then echo "CAM_MAKEUP_02"
    elif [[ "$f" == *"zone"*     ]]; then echo "CAM_MAKEUP_02"
    elif [[ "$f" == *"entry"*    ]]; then echo "CAM_ENTRY_01"
    elif [[ "$f" == *"cam3"*     ]]; then echo "CAM_ENTRY_01"
    elif [[ "$f" == *"backroom"* ]]; then echo "CAM_BACKROOM_04"
    elif [[ "$f" == *"cam4"*     ]]; then echo "CAM_BACKROOM_04"
    elif [[ "$f" == *"billing"*  ]]; then echo "CAM_BILLING_03"
    elif [[ "$f" == *"cam5"*     ]]; then echo "CAM_BILLING_03"
    else echo "CAM_ENTRY_01"  # safe default
    fi
}

CLIPS_PROCESSED=0

for video_file in "$VIDEO_DIR"/*.mp4 "$VIDEO_DIR"/*.avi "$VIDEO_DIR"/*.mov; do
    [ -f "$video_file" ] || continue

    filename=$(basename "$video_file" | tr '[:upper:]' '[:lower:]')
    camera_id=$(get_camera_id "$filename")

    echo "[$(date -u +%H:%M:%S)] Processing: $video_file → camera: $camera_id"

    python pipeline/run_detection.py \
        --video "$video_file" \
        --store_id "$STORE_ID" \
        --camera_id "$camera_id" \
        --store_layout "$STORE_LAYOUT" \
        --output "$OUTPUT" \
        --append \
        --model "$MODEL" \
        --confidence "$CONFIDENCE"

    CLIPS_PROCESSED=$((CLIPS_PROCESSED + 1))
done

if [ "$CLIPS_PROCESSED" -eq 0 ]; then
    echo "WARNING: No video files found in $VIDEO_DIR"
    echo "Place .mp4/.avi/.mov files in $VIDEO_DIR and re-run."
    exit 1
fi

EVENT_COUNT=$(wc -l < "$OUTPUT" 2>/dev/null || echo 0)
echo ""
echo "=== Done ==="
echo "Clips processed : $CLIPS_PROCESSED"
echo "Events written  : $EVENT_COUNT"
echo "Output file     : $OUTPUT"
echo ""
echo "Next step — feed events into the API:"
echo "  python scripts/feed_events.py $OUTPUT"

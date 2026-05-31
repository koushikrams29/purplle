"""
CLI script for running the detection pipeline on CCTV video clips.

Processes video files, generates structured visitor events, and exports to JSONL format.

Example usage:
    python pipeline/run_detection.py \
        --video_path data/cctv/clip1.mp4 \
        --store_id STORE_BLR_002 \
        --camera_id CAM_ENTRY_01 \
        --store_layout scripts/sample_store_layout.json \
        --output events.jsonl
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from datetime import datetime, timezone
from typing import List, Dict, Optional

from pipeline.detect import StoreDetector
from app.models import VisitorEvent


# ============================================================================
# LOGGING SETUP
# ============================================================================

def setup_logging(level: str = "INFO") -> logging.Logger:
    """Configure logging for the detection pipeline."""
    logging.basicConfig(
        level=getattr(logging, level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('pipeline.log')
        ]
    )
    return logging.getLogger(__name__)


logger = setup_logging()


# ============================================================================
# FILE LOADING UTILITIES
# ============================================================================

def load_store_layout(layout_path: str) -> Dict:
    """
    Load store layout from JSON file.
    
    Args:
        layout_path: Path to store_layout.json
        
    Returns:
        Dict with zones and camera mappings
        
    Raises:
        FileNotFoundError: If file doesn't exist
        json.JSONDecodeError: If JSON is invalid
    """
    layout_path = Path(layout_path)
    
    if not layout_path.exists():
        raise FileNotFoundError(f"Store layout file not found: {layout_path}")
    
    logger.info(f"Loading store layout from: {layout_path}")
    
    with open(layout_path, 'r') as f:
        layout = json.load(f)
    
    logger.info(f"Loaded layout for store: {layout.get('store_id', 'UNKNOWN')}")
    logger.info(f"Zones: {list(layout.get('zones', {}).keys())}")
    
    return layout

# ============================================================================
# EVENT VALIDATION & EXPORT
# ============================================================================

def validate_events(events: List[Dict]) -> tuple[List[VisitorEvent], List[Dict]]:
    """
    Validate events against Pydantic schema.
    
    Args:
        events: List of event dictionaries
        
    Returns:
        Tuple of (valid_events, invalid_events)
    """
    valid_events = []
    invalid_events = []
    
    logger.info(f"Validating {len(events)} events...")
    
    for i, event_dict in enumerate(events):
        try:
            # Convert to VisitorEvent and back to validate
            event = VisitorEvent(**event_dict)
            valid_events.append(event)
        except Exception as e:
            logger.warning(f"Event {i} validation failed: {str(e)}")
            invalid_events.append({
                'index': i,
                'data': event_dict,
                'error': str(e)
            })
    
    logger.info(f"Valid: {len(valid_events)}, Invalid: {len(invalid_events)}")
    return valid_events, invalid_events


def write_events_to_jsonl(
    events: List[VisitorEvent],
    output_path: str
) -> None:
    """
    Write events to JSONL file (one JSON object per line).
    
    Args:
        events: List of VisitorEvent objects
        output_path: Path to output JSONL file
        
    Raises:
        IOError: If write fails
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Writing {len(events)} events to: {output_path}")
    
    try:
        with open(output_path, 'w') as f:
            for event in events:
                f.write(event.model_dump_json() + '\n')
        
        logger.info(f"Successfully wrote events to {output_path}")
    except IOError as e:
        logger.error(f"Failed to write output file: {str(e)}")
        raise


# ============================================================================
# STATISTICS & REPORTING
# ============================================================================

def compute_statistics(events: List[VisitorEvent]) -> Dict:
    """
    Compute summary statistics from events.

    Args:
        events: List of VisitorEvent objects

    Returns:
        Dict with statistics
    """
    if not events:
        return {
            "total_events": 0,
            "unique_visitors": 0,
            "entry_count": 0,
            "exit_count": 0,
            "zone_dwell_count": 0,
            "zones": {},
            "event_types": {}
        }

    from app.models import EventType

    unique_visitors = set(e.visitor_id for e in events)

    entry_count = sum(
        1 for e in events
        if e.event_type == EventType.ENTRY
    )

    exit_count = sum(
        1 for e in events
        if e.event_type == EventType.EXIT
    )

    zone_dwell_count = sum(
        1 for e in events
        if e.event_type == EventType.ZONE_DWELL
    )

    zones = {}

    for event in events:
        zone = event.zone_id

        if zone not in zones:
            zones[zone] = 0

        zones[zone] += 1

    return {
        "total_events": len(events),
        "unique_visitors": len(unique_visitors),
        "entry_count": entry_count,
        "exit_count": exit_count,
        "zone_dwell_count": zone_dwell_count,
        "zones": zones,
        "event_types": {
            "ENTRY": entry_count,
            "EXIT": exit_count,
            "ZONE_ENTER": sum(
                1 for e in events
                if e.event_type == EventType.ZONE_ENTER
            ),
            "ZONE_EXIT": sum(
                1 for e in events
                if e.event_type == EventType.ZONE_EXIT
            ),
            "ZONE_DWELL": zone_dwell_count,
            "BILLING_QUEUE_JOIN": sum(
                1 for e in events
                if e.event_type == EventType.BILLING_QUEUE_JOIN
            ),
            "BILLING_QUEUE_ABANDON": sum(
                1 for e in events
                if e.event_type == EventType.BILLING_QUEUE_ABANDON
            ),
            "REENTRY": sum(
                1 for e in events
                if e.event_type == EventType.REENTRY
            )
        }
    }

def print_summary(stats: Dict, output_path: str) -> None:
    """Print summary statistics to console."""
    print("\n" + "="*70)
    print("DETECTION PIPELINE SUMMARY")
    print("="*70)
    print(f"\nOutput file: {output_path}")
    print(f"\nTotal Events: {stats['total_events']}")
    print(f"Unique Visitors: {stats['unique_visitors']}")
    print(f"Entry Count: {stats['entry_count']}")
    print(f"Exit Count: {stats['exit_count']}")
    print(f"Zone Dwell Count: {stats['zone_dwell_count']}")
    
    print(f"\nEvent Types:")
    for event_type, count in stats['event_types'].items():
        print(f"  {event_type}: {count}")
    
    print(f"\nZones Visited:")
    for zone, count in stats['zones'].items():
        print(f"  {zone}: {count}")
    
    print("\n" + "="*70 + "\n")


# ============================================================================
# MAIN CLI
# ============================================================================

def main():
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        description='Detection Pipeline CLI - Process CCTV footage and generate events',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python pipeline/run_detection.py \\
    --video_path data/cctv/clip1.mp4 \\
    --store_id STORE_BLR_002 \\
    --camera_id CAM_ENTRY_01 \\
    --store_layout scripts/sample_store_layout.json \\
    --output events.jsonl
        '''
    )
    
    parser.add_argument(
        '--video_path',
        required=True,
        help='Path to MP4 video file'
    )
    parser.add_argument(
        '--store_id',
        required=True,
        help='Store identifier (e.g., STORE_BLR_002)'
    )
    parser.add_argument(
        '--camera_id',
        required=True,
        help='Camera identifier (e.g., CAM_ENTRY_01)'
    )
    parser.add_argument(
        '--store_layout',
        required=True,
        help='Path to store_layout.json'
    )
    parser.add_argument(
        '--output',
        required=True,
        help='Output JSONL file path'
    )
    parser.add_argument(
        '--start_timestamp',
        default=None,
        help='Video start timestamp (ISO-8601 UTC, optional)'
    )
    parser.add_argument(
        '--yolo_model',
        default='yolov8m',
        help='YOLO model to use (default: yolov8m)'
    )
    parser.add_argument(
        '--log_level',
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    logger.info("Starting detection pipeline...")
    
    try:
        # Load store layout
        store_layout = load_store_layout(args.store_layout)
        
        # Parse start timestamp if provided
        start_timestamp = None
        if args.start_timestamp:
            try:
                start_timestamp = datetime.fromisoformat(
                    args.start_timestamp.replace('Z', '+00:00')
                )
            except ValueError as e:
                logger.warning(f"Invalid timestamp format: {e}, using current time")
        
        # Process video
        logger.info("Initializing StoreDetector...")

        detector = StoreDetector(
            model_name=args.yolo_model,
            device="cpu"
        )

        events = detector.process_clip(
            video_path=args.video_path,
            store_id=args.store_id,
            camera_id=args.camera_id,
            store_layout=store_layout
        )
        
        # Validate events
        valid_events = []
        invalid_events = []

        for i, event in enumerate(events):
            if isinstance(event, VisitorEvent):
                valid_events.append(event)
            else:
                invalid_events.append({
                    "index": i,
                    "error": f"Expected VisitorEvent, got {type(event).__name__}"
                })

        logger.info(
            f"Valid events: {len(valid_events)}, "
            f"Invalid events: {len(invalid_events)}"
        )
        
        if invalid_events:
            logger.warning(f"{len(invalid_events)} events failed validation")
        
        # Write to JSONL
        write_events_to_jsonl(valid_events, args.output)
        
        # Compute and print statistics
        stats = compute_statistics(valid_events)
        print_summary(stats, args.output)
        
        logger.info("Detection pipeline completed successfully!")
        return 0
    
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}", exc_info=True)
        print(f"\n❌ ERROR: {str(e)}\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
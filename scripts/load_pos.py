"""
POS transaction data loader.

Normalises the actual Purplle CSV format into the internal dict format
expected by ingestion.correlate_with_pos:

    {"store_id", "transaction_id", "transaction_time": datetime, "transaction_amount": float}

Actual CSV columns:
    order_id, order_date, order_time, store_id, product_id, brand_name, total_amount

Usage:
    from scripts.load_pos import load_pos_transactions
    pos_data = load_pos_transactions("path/to/pos.csv")
"""

import csv
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict

logger = logging.getLogger(__name__)

_DATE_FORMATS = [
    "%d-%m-%Y",
    "%Y-%m-%d",
    "%m/%d/%Y",
]

_TIME_FORMATS = [
    "%H:%M:%S",
    "%H:%M",
]


def _parse_date(date_str: str) -> datetime:
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(date_str.strip(), fmt)
        except ValueError:
            continue
    raise ValueError(f"Cannot parse date: {date_str!r}")


def _parse_time(time_str: str) -> tuple:
    for fmt in _TIME_FORMATS:
        try:
            t = datetime.strptime(time_str.strip(), fmt)
            return t.hour, t.minute, t.second
        except ValueError:
            continue
    raise ValueError(f"Cannot parse time: {time_str!r}")


def load_pos_transactions(csv_path: str) -> List[Dict]:
    """
    Load and normalise POS transactions from CSV.

    Args:
        csv_path: Path to pos_transactions CSV file

    Returns:
        List of normalised transaction dicts with keys:
            store_id, transaction_id, transaction_time, transaction_amount
    """
    path = Path(csv_path)
    if not path.exists():
        logger.warning(f"POS file not found: {csv_path}")
        return []

    transactions = []

    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                order_id = row.get("order_id", "").strip()
                order_date = row.get("order_date", "").strip()
                order_time = row.get("order_time", "").strip()
                store_id = row.get("store_id", "").strip()
                total_amount = row.get("total_amount", "0").strip()

                if not order_date or not store_id:
                    continue

                date_dt = _parse_date(order_date)
                h, m, s = _parse_time(order_time) if order_time else (0, 0, 0)
                txn_time = date_dt.replace(hour=h, minute=m, second=s)

                transactions.append({
                    "store_id": store_id,
                    "transaction_id": f"TXN_{order_id}",
                    "transaction_time": txn_time,
                    "transaction_amount": float(total_amount) if total_amount else 0.0,
                })

            except Exception as exc:
                logger.warning(f"Skipped POS row: {exc}")
                continue

    logger.info(f"Loaded {len(transactions)} POS transactions from {csv_path}")
    return transactions


def group_by_store(transactions: List[Dict]) -> Dict[str, List[Dict]]:
    """Group normalised transactions by store_id for fast lookup."""
    grouped: Dict[str, List[Dict]] = {}
    for txn in transactions:
        sid = txn["store_id"]
        grouped.setdefault(sid, []).append(txn)
    return grouped

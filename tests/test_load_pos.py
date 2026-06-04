# PROMPT:
# I have scripts/load_pos.py that parses a POS transactions CSV.
#
# Actual CSV format (not what the problem statement says):
#   order_id, order_date, order_time, store_id, product_id, brand_name, total_amount
#   1, 10-04-2026, 12:15:05, ST1008, 399945, Faces Canada, 302.33
#
# load_pos_transactions(csv_path) -> List[Dict] where each dict has:
#   store_id, transaction_id ("TXN_{order_id}"), transaction_time (datetime), transaction_amount (float)
#
# group_by_store(transactions) -> Dict[store_id, List[Dict]]
#
# Write unit tests using tmp_path to create temp CSV files.
# Cover: valid CSV, missing file, zero amount, invalid date (skipped), group_by_store.
#
# CHANGES MADE:
# - Used tmp_path fixture (pytest built-in) instead of tempfile to match project style
# - Discovered date format in resources is DD-MM-YYYY, not YYYY-MM-DD — added test for that
# - Confirmed missing-file returns [] with a warning (no exception raised)
# - transaction_id is "TXN_" + order_id (not UUID), confirmed from load_pos.py source

import pytest
from datetime import datetime
from pathlib import Path
import csv

from scripts.load_pos import load_pos_transactions, group_by_store


def _write_csv(path: Path, rows: list, header=None) -> Path:
    if header is None:
        header = ["order_id", "order_date", "order_time", "store_id",
                  "product_id", "brand_name", "total_amount"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    return path


# ---------------------------------------------------------------------------
# load_pos_transactions
# ---------------------------------------------------------------------------

class TestLoadPosTransactions:

    def test_missing_file_returns_empty_list(self):
        result = load_pos_transactions("/nonexistent/path/pos.csv")
        assert result == []

    def test_valid_single_row(self, tmp_path):
        csv_file = tmp_path / "pos.csv"
        _write_csv(csv_file, [
            ["1", "10-04-2026", "12:15:05", "ST1008", "399945", "Faces Canada", "302.33"]
        ])
        result = load_pos_transactions(str(csv_file))
        assert len(result) == 1
        row = result[0]
        assert row["store_id"] == "ST1008"
        assert row["transaction_id"] == "TXN_1"
        assert isinstance(row["transaction_time"], datetime)
        assert abs(row["transaction_amount"] - 302.33) < 0.01

    def test_date_parsed_correctly(self, tmp_path):
        csv_file = tmp_path / "pos.csv"
        _write_csv(csv_file, [
            ["5", "10-04-2026", "14:30:00", "ST1008", "111", "BrandX", "500.00"]
        ])
        result = load_pos_transactions(str(csv_file))
        ts = result[0]["transaction_time"]
        assert ts.hour == 14
        assert ts.minute == 30
        assert ts.day == 10
        assert ts.month == 4
        assert ts.year == 2026

    def test_zero_amount_row_included(self, tmp_path):
        csv_file = tmp_path / "pos.csv"
        _write_csv(csv_file, [
            ["2", "10-04-2026", "12:42:18", "ST1008", "123", "Brand", "0"]
        ])
        result = load_pos_transactions(str(csv_file))
        assert len(result) == 1
        assert result[0]["transaction_amount"] == 0.0

    def test_multiple_rows_all_loaded(self, tmp_path):
        csv_file = tmp_path / "pos.csv"
        _write_csv(csv_file, [
            ["1", "10-04-2026", "12:00:00", "ST1008", "111", "BrandA", "100.00"],
            ["2", "10-04-2026", "12:05:00", "ST1008", "222", "BrandB", "200.00"],
            ["3", "10-04-2026", "12:10:00", "ST1009", "333", "BrandC", "300.00"],
        ])
        result = load_pos_transactions(str(csv_file))
        assert len(result) == 3

    def test_invalid_date_row_skipped(self, tmp_path):
        csv_file = tmp_path / "pos.csv"
        _write_csv(csv_file, [
            ["1", "INVALID_DATE", "12:00:00", "ST1008", "111", "B", "100.00"],
            ["2", "10-04-2026", "12:05:00", "ST1008", "222", "B", "200.00"],
        ])
        result = load_pos_transactions(str(csv_file))
        # Invalid row skipped; only valid row loaded
        assert len(result) == 1
        assert result[0]["transaction_id"] == "TXN_2"

    def test_transaction_id_format(self, tmp_path):
        csv_file = tmp_path / "pos.csv"
        _write_csv(csv_file, [
            ["42", "10-04-2026", "10:00:00", "ST1008", "999", "Brand", "150.00"]
        ])
        result = load_pos_transactions(str(csv_file))
        assert result[0]["transaction_id"] == "TXN_42"

    def test_empty_csv_returns_empty_list(self, tmp_path):
        csv_file = tmp_path / "empty.csv"
        _write_csv(csv_file, [])  # header only
        result = load_pos_transactions(str(csv_file))
        assert result == []

    def test_all_required_keys_present(self, tmp_path):
        csv_file = tmp_path / "pos.csv"
        _write_csv(csv_file, [
            ["1", "10-04-2026", "12:00:00", "ST1008", "111", "Brand", "100.00"]
        ])
        result = load_pos_transactions(str(csv_file))
        row = result[0]
        assert "store_id" in row
        assert "transaction_id" in row
        assert "transaction_time" in row
        assert "transaction_amount" in row


# ---------------------------------------------------------------------------
# group_by_store
# ---------------------------------------------------------------------------

class TestGroupByStore:

    def test_empty_list_returns_empty_dict(self):
        assert group_by_store([]) == {}

    def test_single_store_grouped(self):
        txns = [
            {"store_id": "ST1008", "transaction_id": "TXN_1",
             "transaction_time": datetime(2026, 4, 10, 12, 0), "transaction_amount": 100.0},
        ]
        result = group_by_store(txns)
        assert "ST1008" in result
        assert len(result["ST1008"]) == 1

    def test_two_stores_separated(self):
        txns = [
            {"store_id": "ST1008", "transaction_id": "TXN_1",
             "transaction_time": datetime(2026, 4, 10, 12, 0), "transaction_amount": 100.0},
            {"store_id": "ST1009", "transaction_id": "TXN_2",
             "transaction_time": datetime(2026, 4, 10, 13, 0), "transaction_amount": 200.0},
            {"store_id": "ST1008", "transaction_id": "TXN_3",
             "transaction_time": datetime(2026, 4, 10, 14, 0), "transaction_amount": 50.0},
        ]
        result = group_by_store(txns)
        assert len(result) == 2
        assert len(result["ST1008"]) == 2
        assert len(result["ST1009"]) == 1

    def test_preserves_transaction_data(self):
        txns = [{"store_id": "ST1008", "transaction_id": "TXN_5",
                 "transaction_time": datetime(2026, 4, 10, 15, 0), "transaction_amount": 999.0}]
        result = group_by_store(txns)
        row = result["ST1008"][0]
        assert row["transaction_id"] == "TXN_5"
        assert row["transaction_amount"] == 999.0

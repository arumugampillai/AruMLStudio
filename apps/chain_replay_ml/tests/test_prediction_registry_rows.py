"""Registry sources[].rows → Dataset rows (no parquet scan)."""

from __future__ import annotations

import json
import os
import tempfile
import unittest

from chain_replay_ml.model_lab.prediction_io import (
    dataset_row_counts_from_meta,
    list_trading_days_from_meta,
    load_parent_dataset_row_counts,
)


class TestDatasetRowCountsFromMeta(unittest.TestCase):
    def test_sources_rows_preferred(self) -> None:
        meta = {
            "days": [
                {"trading_day": "2026-05-26", "market": "NIFTY"},
                {"trading_day": "2026-05-27", "row_count": 1},
            ],
            "sources": [
                {
                    "trading_day": "2026-05-26",
                    "market": "NIFTY",
                    "status": "loaded",
                    "rows": 73939,
                },
                {
                    "trading_day": "2026-05-27",
                    "rows": 100,
                },
            ],
        }
        counts = dataset_row_counts_from_meta(meta)
        self.assertEqual(counts["2026-05-26"], 73939)
        self.assertEqual(counts["2026-05-27"], 100)

    def test_days_row_count_fallback(self) -> None:
        meta = {
            "days": [
                {"trading_day": "2026-01-02", "row_count": 42},
            ],
        }
        self.assertEqual(dataset_row_counts_from_meta(meta)["2026-01-02"], 42)

    def test_list_days_from_sources(self) -> None:
        meta = {
            "sources": [
                {"trading_day": "2026-05-27", "rows": 1},
                {"trading_day": "2026-05-26", "rows": 2},
            ],
        }
        self.assertEqual(
            list_trading_days_from_meta(meta),
            ["2026-05-26", "2026-05-27"],
        )

    def test_load_from_registry_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = tmp
            ds_dir = os.path.join(data_dir, "datasets")
            os.makedirs(ds_dir, exist_ok=True)
            path = os.path.join(ds_dir, "MS_test.json")
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(
                    {
                        "dataset_name": "MS_test",
                        "sources": [
                            {"trading_day": "2026-05-26", "rows": 12},
                        ],
                    },
                    fh,
                )
            counts = load_parent_dataset_row_counts(data_dir, "MS_test")
            self.assertEqual(counts, {"2026-05-26": 12})


if __name__ == "__main__":
    unittest.main()

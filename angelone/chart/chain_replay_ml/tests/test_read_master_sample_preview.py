"""Tests for lightweight master sample preview query."""

from __future__ import annotations

import os
import tempfile
import unittest

from chain_replay_ml.dataset_builder.master_status import read_master_sample_preview


class ReadMasterSamplePreviewTests(unittest.TestCase):
    def test_preview_counts_filtered_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            datasets_dir = os.path.join(tmp, "datasets")
            os.makedirs(datasets_dir)
            db_path = os.path.join(datasets_dir, "master_dataset_nifty_10s.db")
            from chain_replay_ml.dataset_builder.master_store import MasterStore

            store = MasterStore(db_path)
            store.open()
            try:
                cols = [
                    "trading_day", "timestamp", "token", "symbol", "ltp", "spot",
                    "strike_distance_from_atm",
                ]
                store.begin_day("2026-07-01", cols)
                store.insert_rows([
                    {
                        "trading_day": "2026-07-01",
                        "timestamp": 1000.0,
                        "token": "A",
                        "symbol": "SYM-A",
                        "ltp": 25.0,
                        "spot": 24000.0,
                        "strike_distance_from_atm": 0,
                    },
                    {
                        "trading_day": "2026-07-01",
                        "timestamp": 1003.0,
                        "token": "B",
                        "symbol": "SYM-B",
                        "ltp": 5.0,
                        "spot": 24000.0,
                        "strike_distance_from_atm": 2,
                    },
                ])
                store.commit_day("2026-07-01")
            finally:
                store.close()

            preview = read_master_sample_preview(
                tmp,
                market="NIFTY",
                interval_sec=10,
                selected_days=["2026-07-01"],
                premium_min=20.0,
                premium_max=100.0,
            )
            assert preview is not None
            self.assertEqual(preview["match_count"], 1)
            self.assertEqual(len(preview["rows"]), 1)
            self.assertEqual(preview["rows"][0]["token"], "A")


if __name__ == "__main__":
    unittest.main()

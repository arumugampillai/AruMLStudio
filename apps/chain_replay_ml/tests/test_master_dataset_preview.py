"""Tests for metadata-driven master dataset preview."""

from __future__ import annotations

import os
import tempfile
import unittest

from chain_replay_ml.dataset_builder.master_dataset_preview import (
    MasterDatasetPreviewService,
    PreviewFilters,
)


class TestMasterDatasetPreview(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp()
        self.datasets_dir = os.path.join(self.tmp, "datasets")
        os.makedirs(self.datasets_dir, exist_ok=True)
        self.db_path = os.path.join(self.datasets_dir, "master_dataset_nifty_10s.db")

    def _seed_db(self) -> None:
        from chain_replay_ml.dataset_builder.master_store import MasterStore

        store = MasterStore(self.db_path)
        store.open()
        try:
            cols = [
                "trading_day", "timestamp", "token", "ltp",
                "strike_distance_from_atm", "abs_delta",
            ]
            store.begin_day("2026-05-27", cols)
            store.insert_rows([
                {
                    "trading_day": "2026-05-27",
                    "timestamp": 1000.0,
                    "token": "A",
                    "ltp": 25.0,
                    "strike_distance_from_atm": 0,
                    "abs_delta": 0.15,
                },
                {
                    "trading_day": "2026-05-27",
                    "timestamp": 1003.0,
                    "token": "B",
                    "ltp": 5.0,
                    "strike_distance_from_atm": 2,
                    "abs_delta": 0.25,
                },
            ])
            store.commit_day("2026-05-27")
            store.update_build_identity({
                "market": "NIFTY",
                "sampling_interval_sec": 10,
                "builder_version": "1.4.2",
                "created_from": "test",
                "feature_registry_version": "203-v5",
                "feature_hash": "ABCD1234",
                "target_hash": "EFGH5678",
                "schema_hash": "9395CE8C",
                "feature_count": 100,
                "target_count": 5,
                "dataset_fingerprint": {
                    "sampling_interval": 10,
                    "features": 100,
                    "targets": 5,
                },
            })
        finally:
            store.close()

    def test_preview_filters_premium(self) -> None:
        self._seed_db()
        svc = MasterDatasetPreviewService(self.db_path)
        out = svc.preview(
            PreviewFilters(
                selected_days=["2026-05-27"],
                premium_range={"min": 20, "max": 100},
            )
        )
        self.assertEqual(out["estimated_days"], 1)
        self.assertEqual(out["estimated_rows"], 1)
        self.assertEqual(out["accuracy"], "estimated")

    def test_preview_from_body(self) -> None:
        self._seed_db()
        out = MasterDatasetPreviewService.preview_from_body(
            self.tmp,
            {
                "market": "NIFTY",
                "master_dataset": "master_dataset_nifty_10s.db",
                "selected_days": ["2026-05-27"],
                "atm_band": 1,
                "premium_range": None,
                "delta_range": None,
            },
        )
        self.assertGreaterEqual(out["estimated_rows"], 1)
        self.assertGreater(out["metadata_version"], 0)


if __name__ == "__main__":
    unittest.main()

"""Tests for Model Builder Dataset Engine load path (no full training)."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from chain_replay_ml.training.config import TrainingConfig
from chain_replay_ml.training.dataset_loader import (
    apply_config_premium_filter,
    compare_training_load_backends,
    load_dataset_frame,
    load_training_xy,
)
from chain_replay_ml.training.load_backend import resolve_training_load_backend
from chain_replay_ml.tests.dataset_engine._fixtures import HAS_DUCKDB, require_duckdb


def _write_mini_dataset(data_dir: str, name: str = "mini_train") -> TrainingConfig:
    from chain_replay_ml.dataset_builder.writer import datasets_dir, _safe_filename

    out = datasets_dir(data_dir)
    os.makedirs(out, exist_ok=True)
    safe = _safe_filename(name)
    path = os.path.join(out, f"{safe}.parquet")
    meta = os.path.join(out, f"{safe}.json")
    df = pd.DataFrame(
        {
            "trading_day": ["2026-07-23"] * 6,
            "timestamp": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "token": ["A", "B", "C", "D", "E", "F"],
            "strike": [100.0] * 6,
            "option_type": ["CE"] * 6,
            "spot": [100.0] * 6,
            "ltp": [10.0, 20.0, 40.0, 80.0, 120.0, 5.0],
            "symbol": ["X"] * 6,
            "market": ["NIFTY"] * 6,
            "expiry": ["2026-07-28"] * 6,
            "feat_a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "future_ltp_60": [11.0, 21.0, 41.0, 81.0, 121.0, 6.0],
        }
    )
    df.to_parquet(path, index=False)
    with open(meta, "w", encoding="utf-8") as fh:
        json.dump({"dataset_name": name, "row_count": len(df)}, fh)
    return TrainingConfig(
        dataset=name,
        features=["feat_a"],
        target="future_ltp_60",
        premium_selection_enabled=True,
        premium_min=15.0,
        premium_max=100.0,
    )


class TestResolveLoadBackend(unittest.TestCase):
    def test_off(self) -> None:
        self.assertEqual(resolve_training_load_backend("off"), "pandas")

    def test_on(self) -> None:
        self.assertEqual(resolve_training_load_backend("on"), "dataset_engine")

    def test_auto_without_duckdb(self) -> None:
        with patch.dict(os.environ, {"ARUNEO_DATASET_ENGINE": "auto"}, clear=False):
            with patch(
                "chain_replay_ml.training.load_backend.resolve_training_load_backend"
            ):
                pass
        # Direct: when duckdb missing, auto → pandas
        if not HAS_DUCKDB:
            self.assertEqual(resolve_training_load_backend("auto"), "pandas")


class TestTrainingLoadPandas(unittest.TestCase):
    def test_premium_filter_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _write_mini_dataset(tmp)
            with patch.dict(os.environ, {"ARUNEO_DATASET_ENGINE": "off"}):
                X, y, feats, meta, _exp, _ctx = load_training_xy(tmp, cfg)
            self.assertEqual(len(X), 3)  # ltp 20,40,80
            self.assertEqual(meta["dataset_load"]["backend"], "pandas")
            self.assertIn("load_time_sec", meta["dataset_load"])
            self.assertEqual(meta["dataset_load"]["rows_returned"], 3)
            self.assertIsNotNone(meta.get("premium_selection"))


class TestTrainingLoadEngine(unittest.TestCase):
    def setUp(self) -> None:
        require_duckdb()

    def test_engine_matches_pandas_matrices(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _write_mini_dataset(tmp)
            report = compare_training_load_backends(tmp, cfg)
            self.assertTrue(report["rows_match"])
            self.assertTrue(report["matrices_equal"])
            self.assertEqual(report["pandas"]["rows_returned"], 3)
            self.assertEqual(report["dataset_engine"]["rows_returned"], 3)
            self.assertIn("load_time_sec", report["pandas"])
            self.assertIn("load_time_sec", report["dataset_engine"])
            self.assertIn("partitions_scanned", report["dataset_engine"])

    def test_load_training_xy_via_engine(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _write_mini_dataset(tmp)
            with patch.dict(os.environ, {"ARUNEO_DATASET_ENGINE": "on"}):
                X, y, feats, meta, _exp, _ctx = load_training_xy(tmp, cfg)
            self.assertEqual(len(X), 3)
            self.assertEqual(meta["dataset_load"]["backend"], "dataset_engine")
            self.assertIn("partitions_scanned", meta["dataset_load"])
            self.assertEqual(meta["premium_selection"].get("applied_via"), "dataset_engine")

            with patch.dict(os.environ, {"ARUNEO_DATASET_ENGINE": "off"}):
                Xp, yp, *_ = load_training_xy(tmp, cfg)
            import numpy as np

            self.assertTrue(
                np.allclose(X.to_numpy(dtype=float), Xp.to_numpy(dtype=float), equal_nan=True)
            )
            self.assertTrue(
                np.allclose(y.to_numpy(dtype=float), yp.to_numpy(dtype=float), equal_nan=True)
            )

    def test_engine_failure_falls_back_to_pandas(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _write_mini_dataset(tmp)
            with patch.dict(os.environ, {"ARUNEO_DATASET_ENGINE": "on"}):
                with patch(
                    "chain_replay_ml.training.dataset_loader.load_dataset_frame_via_engine",
                    side_effect=RuntimeError("boom"),
                ):
                    X, y, feats, meta, _exp, _ctx = load_training_xy(tmp, cfg)
            self.assertEqual(len(X), 3)
            self.assertEqual(meta["dataset_load"]["backend"], "pandas")
            self.assertTrue(meta["dataset_load"].get("engine_fallback"))
            self.assertIn("boom", str(meta["dataset_load"].get("engine_fallback_reason") or ""))


if __name__ == "__main__":
    unittest.main()

"""Tests for incomplete prediction-horizon exclusion."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import pandas as pd

from chain_replay_ml.dataset_builder.tick_coverage import clipped_grid_bounds
from chain_replay_ml.model_lab.prediction_parallel import _exclude_incomplete_horizon_rows
from chain_replay_ml.model_lab.prediction_schema import (
    has_complete_prediction_horizon,
    horizon_sec_from_target,
    prediction_horizon_cutoff_ts,
)


class HorizonHelpersTests(unittest.TestCase):
    def test_horizon_from_target(self) -> None:
        self.assertEqual(horizon_sec_from_target("future_ltp_3m"), 180.0)
        self.assertEqual(horizon_sec_from_target("future_ltp_5m"), 300.0)
        self.assertEqual(horizon_sec_from_target("future_ltp_10m"), 600.0)

    def test_complete_horizon_gate(self) -> None:
        # 5m horizon: sample at T needs data through T+300
        self.assertTrue(
            has_complete_prediction_horizon(
                timestamp=1000.0, data_end_ts=1300.0, horizon_sec=300.0
            )
        )
        self.assertFalse(
            has_complete_prediction_horizon(
                timestamp=1001.0, data_end_ts=1300.0, horizon_sec=300.0
            )
        )
        self.assertEqual(prediction_horizon_cutoff_ts(1300.0, 300.0), 1000.0)


class ClippedGridBoundsTests(unittest.TestCase):
    def test_excludes_final_horizon_from_last_tick_not_close(self) -> None:
        # Spot data ends at 2000; session close is later (2500). Horizon 300s.
        # Samples must end at 1700 (= last_tick - horizon), not at close-horizon.
        index_tl = SimpleNamespace(timestamps=[1000.0, 1500.0, 2000.0])
        ctx = SimpleNamespace(open_ts=900.0, close_ts=2500.0, index_tl=index_tl)
        bounds = clipped_grid_bounds(ctx, max_horizon_sec=300)
        self.assertIsNotNone(bounds)
        assert bounds is not None
        grid_start, grid_end = bounds
        self.assertEqual(grid_end, 1700.0)
        self.assertGreater(grid_start, 0)


class ExcludeIncompleteHorizonRowsTests(unittest.TestCase):
    def test_drops_rows_past_token_last_tick_minus_horizon(self) -> None:
        day_df = pd.DataFrame(
            [
                {"timestamp": 1000.0, "token": "A", "future_ltp_5m": 1.0},
                {"timestamp": 1100.0, "token": "A", "future_ltp_5m": 1.0},
                {"timestamp": 1000.0, "token": "B", "future_ltp_5m": 1.0},
            ]
        )
        timelines = {
            "A": SimpleNamespace(timestamps=[900.0, 1300.0]),  # cutoff 1000
            "B": SimpleNamespace(timestamps=[900.0, 1400.0]),  # cutoff 1100
        }
        out, dropped = _exclude_incomplete_horizon_rows(
            day_df, horizon_sec=300.0, timelines=timelines
        )
        self.assertEqual(dropped, 1)
        self.assertEqual(len(out), 2)
        self.assertTrue((out["timestamp"] <= 1000.0).all() or set(out["token"]) == {"A", "B"})
        # Row at 1100/A must be gone; 1000/A and 1000/B remain
        self.assertFalse(((out["token"] == "A") & (out["timestamp"] == 1100.0)).any())
        self.assertEqual(len(out[out["token"] == "A"]), 1)
        self.assertEqual(len(out[out["token"] == "B"]), 1)

    def test_noop_without_timelines(self) -> None:
        day_df = pd.DataFrame(
            [{"timestamp": 1000.0, "token": "A"}, {"timestamp": 2000.0, "token": "A"}]
        )
        out, dropped = _exclude_incomplete_horizon_rows(
            day_df, horizon_sec=300.0, timelines=None
        )
        self.assertEqual(dropped, 0)
        self.assertEqual(len(out), 2)


if __name__ == "__main__":
    unittest.main()

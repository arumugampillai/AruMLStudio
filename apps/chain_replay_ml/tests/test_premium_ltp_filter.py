"""Unit tests for post No-Null LTP premium filter."""

from __future__ import annotations

import unittest

import pandas as pd

from chain_replay_ml.dataset_builder.premium_ltp_filter import (
    apply_premium_ltp_filter_frame,
    normalize_premium_bounds,
)


class PremiumLtpFilterTests(unittest.TestCase):
    def test_normalize_swaps_inverted_bounds(self) -> None:
        self.assertEqual(normalize_premium_bounds(40, 15), (15.0, 40.0))
        self.assertIsNone(normalize_premium_bounds(15, None))

    def test_filters_ltp_band_inclusive(self) -> None:
        frame = pd.DataFrame({
            "trading_day": ["2026-07-24"] * 5,
            "ltp": [10.0, 15.0, 25.0, 40.0, 50.0],
            "token": ["a", "b", "c", "d", "e"],
        })
        result = apply_premium_ltp_filter_frame(frame, premium_min=15, premium_max=40)
        out = result["frame"]
        self.assertEqual(out["ltp"].tolist(), [15.0, 25.0, 40.0])
        self.assertEqual(result["report"]["rows_before"], 5)
        self.assertEqual(result["report"]["rows_after"], 3)
        self.assertEqual(result["report"]["stage"], "post_no_null")

    def test_drops_null_ltp(self) -> None:
        frame = pd.DataFrame({"ltp": [20.0, None, 30.0]})
        result = apply_premium_ltp_filter_frame(frame, premium_min=15, premium_max=40)
        self.assertEqual(len(result["frame"]), 2)

    def test_missing_ltp_column_raises(self) -> None:
        with self.assertRaises(ValueError):
            apply_premium_ltp_filter_frame(pd.DataFrame({"x": [1]}), premium_min=1, premium_max=2)

    def test_training_config_premium_filter_helper(self) -> None:
        from chain_replay_ml.training.config import TrainingConfig
        from chain_replay_ml.training.dataset_loader import apply_config_premium_filter

        frame = pd.DataFrame({"ltp": [10.0, 20.0, 50.0, 120.0]})
        off = TrainingConfig(dataset="d", target="t", premium_selection_enabled=False)
        out, report = apply_config_premium_filter(frame, off)
        self.assertIs(out, frame)
        self.assertIsNone(report)

        on = TrainingConfig(
            dataset="d",
            target="t",
            premium_selection_enabled=True,
            premium_min=15.0,
            premium_max=100.0,
        )
        filtered, report = apply_config_premium_filter(frame, on)
        self.assertEqual(filtered["ltp"].tolist(), [20.0, 50.0])
        self.assertEqual(report["stage"], "training_premium_selection")
        self.assertEqual(report["rows_after"], 2)


if __name__ == "__main__":
    unittest.main()

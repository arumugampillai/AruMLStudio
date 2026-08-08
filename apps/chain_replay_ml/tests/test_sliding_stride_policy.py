"""Tests for sliding stride validation and dataset configuration."""

from __future__ import annotations

import unittest

from chain_replay_ml.dataset_builder.lookback_policy import build_dataset_configuration
from chain_replay_ml.dataset_builder.sliding_stride_policy import (
    resolve_feature_window_sec,
    resolve_sliding_stride_sec,
    validate_sliding_stride,
)
from chain_replay_ml.dataset_builder.tick_coverage import list_clipped_grid_timestamps


class TestSlidingStridePolicy(unittest.TestCase):
    def test_default_stride_equals_window(self) -> None:
        sampling = {"trainingIntervalSec": 6}
        self.assertEqual(resolve_sliding_stride_sec(sampling), 6)
        self.assertIsNone(validate_sliding_stride(6, 6))

    def test_explicit_stride(self) -> None:
        sampling = {"trainingIntervalSec": 6, "slidingStrideSec": 2}
        self.assertEqual(resolve_sliding_stride_sec(sampling), 2)
        self.assertEqual(resolve_feature_window_sec(sampling), 6)
        self.assertIsNone(validate_sliding_stride(6, 2))

    def test_invalid_divisor(self) -> None:
        err = validate_sliding_stride(6, 4)
        self.assertIsNotNone(err)
        self.assertIn("divisible", err or "")

    def test_dataset_configuration_grid_step(self) -> None:
        cfg = build_dataset_configuration(
            sampling={"trainingIntervalSec": 6, "slidingStrideSec": 2},
            horizons_sec=[10],
        )
        self.assertEqual(cfg["sampling_interval_sec"], 6)
        self.assertEqual(cfg["feature_grid_step_sec"], 2)
        self.assertEqual(cfg["sliding_stride_sec"], 2)


class TestSlidingStrideTimestamps(unittest.TestCase):
    def test_stride_produces_more_grid_points(self) -> None:
        class _Ctx:
            open_ts = 0.0
            close_ts = 100_000.0
            index_tl = type("TL", (), {"timestamps": [0.0, 100_000.0]})()

        ctx = _Ctx()
        max_hor = 0
        coarse = list_clipped_grid_timestamps(ctx, step_sec=6, max_horizon_sec=max_hor)
        fine = list_clipped_grid_timestamps(ctx, step_sec=2, max_horizon_sec=max_hor)
        self.assertGreater(len(fine), len(coarse))
        self.assertEqual(len(fine) / len(coarse), 3.0)


if __name__ == "__main__":
    unittest.main()

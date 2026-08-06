"""Tests for gap policy helpers."""

from __future__ import annotations

import unittest

from chain_replay_ml.dataset_builder.extended_features import _ema_series_from_prices
from chain_replay_ml.dataset_builder.gap_policy import (
    default_gap_policy,
    gap_max_sec_from_policy,
    gap_summary_label,
    normalize_gap_policy,
)
import numpy as np


class GapPolicyTests(unittest.TestCase):
    def test_default_is_20_sec(self) -> None:
        cfg = default_gap_policy()
        self.assertEqual(cfg["preset"], "20")
        self.assertTrue(cfg["enabled"])
        self.assertEqual(gap_max_sec_from_policy(cfg), 20.0)

    def test_presets_include_longer_thresholds(self) -> None:
        for sec in (60, 90, 120, 180):
            cfg = normalize_gap_policy({"preset": str(sec)})
            self.assertEqual(cfg["preset"], str(sec))
            self.assertEqual(gap_max_sec_from_policy(cfg), float(sec))

    def test_disabled_returns_zero_gap(self) -> None:
        cfg = normalize_gap_policy({"enabled": False, "preset": "20"})
        self.assertFalse(cfg["enabled"])
        self.assertEqual(gap_max_sec_from_policy(cfg), 0.0)
        self.assertIn("off", gap_summary_label(cfg))

    def test_custom_preset(self) -> None:
        cfg = normalize_gap_policy({"preset": "custom", "gapMaxSec": 45})
        self.assertEqual(cfg["preset"], "custom")
        self.assertEqual(gap_max_sec_from_policy(cfg), 45.0)
        self.assertIn("custom", gap_summary_label(cfg))

    def test_legacy_preset_still_accepted(self) -> None:
        cfg = normalize_gap_policy({"preset": "30"})
        self.assertEqual(cfg["preset"], "30")
        self.assertEqual(gap_max_sec_from_policy(cfg), 30.0)

    def test_ema_resets_on_tick_gap(self) -> None:
        prices = np.array([100.0, 100.0, 100.0, 110.0], dtype=float)
        tick_ts = np.array([0.0, 10.0, 50.0, 60.0], dtype=float)
        ema = _ema_series_from_prices(prices, 9, last_tick_ts=tick_ts, gap_max_sec=20.0)
        self.assertEqual(float(ema[2]), 100.0)
        self.assertGreater(float(ema[3]), float(ema[2]))
        self.assertLess(float(ema[3]), 110.0)


if __name__ == "__main__":
    unittest.main()

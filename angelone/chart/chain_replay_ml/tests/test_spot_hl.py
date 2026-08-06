"""Tests for Spot HL (high/low EMA channel) features."""

from __future__ import annotations

import unittest

from chain_replay_ml.dataset_builder.spot_hl import (
    CHANNEL_EPS,
    _ema_series_from_prices,
    _ltp_to_channel_width,
    _safe_ratio,
    _weighted_blend,
    _weighted_to_ltp_ratio,
)
from chain_replay_ml.ticks import TickTimeline


class TestSpotHl(unittest.TestCase):
    def test_high_low_series_on_grid(self) -> None:
        tl = TickTimeline()
        tl.append(1000.0, 10000, 0, 0, 0)
        tl.append(1005.0, 10200, 0, 0, 0)
        tl.append(1010.0, 9800, 0, 0, 0)
        tl.append(1020.0, 10100, 0, 0, 0)
        highs, lows = tl.high_low_rupees_series_on_grid(1000.0, 1020.0, 10.0)
        self.assertEqual(len(highs), 3)
        self.assertAlmostEqual(highs[1], 102.0, places=4)
        self.assertAlmostEqual(lows[1], 98.0, places=4)

    def test_weighted_blend_and_ltp_ratio(self) -> None:
        blend = _weighted_blend(80.0, 70.0, 60.0, 50.0)
        self.assertAlmostEqual(blend, 80 * 4 + 70 * 3 + 60 * 2 + 50, places=6)
        val = _weighted_to_ltp_ratio(80.0, 70.0, 60.0, 50.0, 100.0)
        self.assertAlmostEqual(val, blend / 1000.0, places=6)

    def test_safe_ratio(self) -> None:
        self.assertAlmostEqual(_safe_ratio(200.0, 100.0), 2.0)
        self.assertEqual(_safe_ratio(200.0, 0.0), 0.0)

    def test_channel_width_ratio(self) -> None:
        val = _ltp_to_channel_width(210.0, 24010.0, 23990.0)
        self.assertAlmostEqual(val, 210.0 / 20.0, places=5)
        self.assertEqual(_ltp_to_channel_width(210.0, 100.0, 100.0), 210.0 / CHANNEL_EPS)

    def test_ema_series(self) -> None:
        import numpy as np

        out = _ema_series_from_prices(np.array([100.0, 110.0, 105.0]), 20)
        self.assertEqual(len(out), 3)
        self.assertGreater(out[1], out[0])


if __name__ == "__main__":
    unittest.main()

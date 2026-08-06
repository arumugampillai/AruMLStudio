"""Tests for warm-up simulator duration-limited tick loading."""

from __future__ import annotations

import unittest
from datetime import datetime
from zoneinfo import ZoneInfo

from chain_replay_ml.dataset_builder.day_context import simulator_tick_load_bounds

IST = ZoneInfo("Asia/Kolkata")


class SimulatorTickLoadBoundsTests(unittest.TestCase):
    def test_window_from_session_open(self) -> None:
        day = "2026-07-01"
        open_ts = datetime(2026, 7, 1, 9, 15, tzinfo=IST).timestamp()
        tick_start, tick_end = simulator_tick_load_bounds(
            day,
            duration_minutes=20,
            first_spot_ts=open_ts + 30,
            pad_before_sec=60,
        )
        self.assertAlmostEqual(tick_start, open_ts - 60, places=3)
        self.assertAlmostEqual(tick_end, open_ts + 20 * 60, places=3)

    def test_window_shifted_when_spot_starts_late(self) -> None:
        day = "2026-07-01"
        open_ts = datetime(2026, 7, 1, 9, 15, tzinfo=IST).timestamp()
        late_spot = open_ts + 45 * 60
        tick_start, tick_end = simulator_tick_load_bounds(
            day,
            duration_minutes=20,
            first_spot_ts=late_spot,
            pad_before_sec=60,
        )
        self.assertAlmostEqual(tick_start, open_ts - 60, places=3)
        self.assertAlmostEqual(tick_end, late_spot + 20 * 60, places=3)

    def test_window_extends_for_prediction_targets(self) -> None:
        day = "2026-07-01"
        open_ts = datetime(2026, 7, 1, 9, 15, tzinfo=IST).timestamp()
        tick_start, tick_end = simulator_tick_load_bounds(
            day,
            duration_minutes=5,
            first_spot_ts=open_ts + 30,
            pad_before_sec=60,
            max_horizon_sec=300,
        )
        self.assertAlmostEqual(tick_end, open_ts + 5 * 60 + 300, places=3)


if __name__ == "__main__":
    unittest.main()

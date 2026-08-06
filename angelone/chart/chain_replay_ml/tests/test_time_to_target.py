"""Tests for Time To Target vs direction-aware MFE / MAE."""

from __future__ import annotations

import unittest

from chain_replay_ml.prediction_meta.outcomes import (
    compute_path_outcomes,
    compute_time_to_target,
    first_target_reached_ts,
    move_trend,
)
from chain_replay_ml.ticks import TickTimeline


def _tl(stamps: list[float], ltps: list[float]) -> TickTimeline:
    return TickTimeline(timestamps=stamps, ltps_paise=[int(round(x * 100)) for x in ltps])


class TimeToTargetTests(unittest.TestCase):
    def test_move_trend(self) -> None:
        self.assertEqual(move_trend(4.5), "UP")
        self.assertEqual(move_trend(-6.4), "DOWN")
        self.assertEqual(move_trend(0.0), "FLAT")

    def test_up_target_first_touch(self) -> None:
        t0 = 1_000.0
        sec = compute_time_to_target(
            [t0, t0 + 10, t0 + 20, t0 + 30],
            [100.0, 102.0, 105.0, 108.0],
            entry_ts=t0,
            entry_ltp=100.0,
            predicted_ltp=105.0,
        )
        self.assertEqual(sec, 20.0)
        self.assertEqual(
            first_target_reached_ts(
                [t0, t0 + 10, t0 + 20, t0 + 30],
                [100.0, 102.0, 105.0, 108.0],
                entry_ts=t0,
                entry_ltp=100.0,
                predicted_ltp=105.0,
            ),
            t0 + 20,
        )

    def test_down_target_first_touch(self) -> None:
        t0 = 1_000.0
        sec = compute_time_to_target(
            [t0, t0 + 5, t0 + 15],
            [100.0, 98.0, 95.0],
            entry_ts=t0,
            entry_ltp=100.0,
            predicted_ltp=96.0,
        )
        self.assertEqual(sec, 15.0)

    def test_miss_returns_minus_one_and_null_abs(self) -> None:
        t0 = 1_000.0
        seg_ts = [t0, t0 + 10, t0 + 20]
        seg_ltp = [100.0, 101.0, 102.0]
        sec = compute_time_to_target(
            seg_ts, seg_ltp, entry_ts=t0, entry_ltp=100.0, predicted_ltp=110.0,
        )
        self.assertEqual(sec, -1.0)
        self.assertIsNone(
            first_target_reached_ts(
                seg_ts, seg_ltp, entry_ts=t0, entry_ltp=100.0, predicted_ltp=110.0,
            )
        )

    def test_flat_is_zero(self) -> None:
        self.assertEqual(
            compute_time_to_target(
                [1.0, 2.0],
                [50.0, 51.0],
                entry_ts=1.0,
                entry_ltp=50.0,
                predicted_ltp=50.0,
            ),
            0.0,
        )

    def test_up_mfe_mae(self) -> None:
        """UP: MFE = high-entry; MAE = entry-low."""
        t0 = 10_000.0
        # 100 → 101 → 108 → 103 → 106  (entry 100, pred UP to 105)
        tl = _tl(
            [t0, t0 + 10, t0 + 40, t0 + 70, t0 + 100],
            [100.0, 101.0, 108.0, 103.0, 106.0],
        )
        out = compute_path_outcomes(
            tl, ts=t0, entry_ltp=100.0, horizon_sec=300.0, predicted_ltp=105.0,
        )
        self.assertEqual(out["actual_max_profit"], 8.0)
        self.assertEqual(out["actual_max_drawdown"], 0.0)  # never below entry
        self.assertEqual(out["actual_max_profit_5m"], 8.0)
        self.assertEqual(out["actual_max_drawdown_5m"], 0.0)
        self.assertEqual(out["horizon_sec"], 300.0)
        self.assertEqual(out["max_profit_at"], t0 + 40)
        self.assertEqual(out["time_to_target"], 40.0)  # 108 >= 105 first at high? 101 then 108
        # First >= 105 is 108 at t+40
        self.assertEqual(out["target_reached_at"], t0 + 40)
        self.assertEqual(out["exit_at"], t0 + 300)

    def test_down_mfe_mae(self) -> None:
        """DOWN: MFE = entry-low; MAE = high-entry."""
        t0 = 10_000.0
        # 100 → 99 → 97 → 95 → 98 → 96  (entry 100, pred DOWN to 96)
        tl = _tl(
            [t0, t0 + 5, t0 + 10, t0 + 20, t0 + 30, t0 + 40],
            [100.0, 99.0, 97.0, 95.0, 98.0, 96.0],
        )
        out = compute_path_outcomes(
            tl, ts=t0, entry_ltp=100.0, horizon_sec=300.0, predicted_ltp=96.0,
        )
        self.assertEqual(out["actual_max_profit"], 5.0)  # 100-95
        self.assertEqual(out["actual_max_drawdown"], 0.0)  # never above entry
        self.assertEqual(out["actual_max_profit_5m"], 5.0)
        self.assertEqual(out["actual_max_drawdown_5m"], 0.0)
        self.assertEqual(out["max_profit_at"], t0 + 20)  # low at 95
        self.assertEqual(out["time_to_target"], 20.0)  # first <= 96 is 95 at t+20
        self.assertEqual(out["target_reached_at"], t0 + 20)
        # No rally above entry before target → dd_before_target = 0
        self.assertEqual(out["dd_before_target"], 0.0)
        self.assertEqual(out["time_to_dd_before_target"], 0.0)


class DdBeforeTargetTests(unittest.TestCase):
    def test_no_adverse_up_forces_time_zero(self) -> None:
        """Spec example: 20 → 20.10 → 20.30 → 20.80 → 21 (target)."""
        t0 = 1_000.0
        tl = _tl(
            [t0, t0 + 3, t0 + 6, t0 + 9, t0 + 12],
            [20.0, 20.10, 20.30, 20.80, 21.0],
        )
        out = compute_path_outcomes(
            tl, ts=t0, entry_ltp=20.0, horizon_sec=300.0, predicted_ltp=21.0,
        )
        self.assertEqual(out["dd_before_target"], 0.0)
        self.assertEqual(out["time_to_dd_before_target"], 0.0)

    def test_flat_revisit_entry_still_zero_time(self) -> None:
        """Min equals entry twice; must force time=0 not later revisit."""
        t0 = 1_000.0
        tl = _tl(
            [t0, t0 + 3, t0 + 6, t0 + 9],
            [20.0, 20.10, 20.0, 21.0],
        )
        out = compute_path_outcomes(
            tl, ts=t0, entry_ltp=20.0, horizon_sec=300.0, predicted_ltp=21.0,
        )
        self.assertEqual(out["dd_before_target"], 0.0)
        self.assertEqual(out["time_to_dd_before_target"], 0.0)

    def test_adverse_before_target_up(self) -> None:
        t0 = 1_000.0
        # 20 → 19.30 → 19.50 → 21
        tl = _tl(
            [t0, t0 + 5, t0 + 10, t0 + 20],
            [20.0, 19.30, 19.50, 21.0],
        )
        out = compute_path_outcomes(
            tl, ts=t0, entry_ltp=20.0, horizon_sec=300.0, predicted_ltp=21.0,
        )
        self.assertAlmostEqual(out["dd_before_target"], 0.70, places=6)
        self.assertEqual(out["time_to_dd_before_target"], 5.0)
        self.assertGreater(out["dd_before_target"], 0.0)

    def test_never_reached_falls_back_to_max_dd(self) -> None:
        t0 = 1_000.0
        # Never hits 25; dips to 19 then recovers
        tl = _tl(
            [t0, t0 + 10, t0 + 20, t0 + 30],
            [20.0, 19.0, 20.5, 21.0],
        )
        out = compute_path_outcomes(
            tl, ts=t0, entry_ltp=20.0, horizon_sec=300.0, predicted_ltp=25.0,
        )
        self.assertEqual(out["time_to_target"], -1.0)
        self.assertIsNone(out["target_reached_at"])
        self.assertEqual(out["dd_before_target"], out["actual_max_drawdown"])
        self.assertEqual(out["dd_before_target"], out["actual_max_drawdown_5m"])
        self.assertEqual(out["time_to_dd_before_target"], out["time_to_max_drawdown"])
        self.assertEqual(out["dd_before_target"], 1.0)
        self.assertEqual(out["time_to_dd_before_target"], 10.0)

    def test_down_adverse_before_target(self) -> None:
        t0 = 1_000.0
        # DOWN to 95; adverse rally 100 → 101 → 99 → 95
        tl = _tl(
            [t0, t0 + 4, t0 + 8, t0 + 12],
            [100.0, 101.0, 99.0, 95.0],
        )
        out = compute_path_outcomes(
            tl, ts=t0, entry_ltp=100.0, horizon_sec=300.0, predicted_ltp=95.0,
        )
        self.assertEqual(out["dd_before_target"], 1.0)
        self.assertEqual(out["time_to_dd_before_target"], 4.0)


if __name__ == "__main__":
    unittest.main()

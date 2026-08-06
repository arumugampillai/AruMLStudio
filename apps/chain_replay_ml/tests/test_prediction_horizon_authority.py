"""Authority of configured regression target horizon across Research Lab."""

from __future__ import annotations

import inspect
import unittest

from chain_replay_ml.model_lab.prediction_schema import (
    actual_ltp_column_from_target,
    horizon_label_from_target,
    horizon_sec_from_target,
)
from chain_replay_ml.prediction_meta.outcomes import compute_path_outcomes
from chain_replay_ml.ticks import TickTimeline


def _tl(stamps: list[float], ltps: list[float]) -> TickTimeline:
    return TickTimeline(
        timestamps=stamps,
        ltps_paise=[int(round(x * 100)) for x in ltps],
    )


class PredictionHorizonAuthorityTests(unittest.TestCase):
    def test_horizon_sec_required_no_default_300(self) -> None:
        sig = inspect.signature(compute_path_outcomes)
        param = sig.parameters["horizon_sec"]
        self.assertIs(param.default, inspect.Parameter.empty)

    def test_horizon_from_target_no_silent_5m_fallback(self) -> None:
        self.assertEqual(horizon_sec_from_target("future_ltp_3m"), 180.0)
        self.assertEqual(horizon_sec_from_target("future_ltp_5m"), 300.0)
        self.assertEqual(horizon_sec_from_target("future_ltp_10m"), 600.0)
        with self.assertRaises(ValueError):
            horizon_sec_from_target(None)
        with self.assertRaises(ValueError):
            horizon_sec_from_target("")
        with self.assertRaises(ValueError):
            horizon_sec_from_target("not_a_target")

    def test_actual_ltp_column_from_target(self) -> None:
        self.assertEqual(actual_ltp_column_from_target("future_ltp_3m"), "actual_3m_ltp")
        self.assertEqual(actual_ltp_column_from_target("future_ltp_5m"), "actual_5m_ltp")
        self.assertEqual(actual_ltp_column_from_target("future_ltp_30s"), "actual_30s_ltp")
        self.assertEqual(horizon_label_from_target("future_ltp_10m"), "10m")
        with self.assertRaises(ValueError):
            actual_ltp_column_from_target(None)

    def test_path_outcomes_honor_configured_horizon(self) -> None:
        t0 = 1_000_000.0
        stamps = [t0 + i for i in range(0, 181, 10)]
        ltps = [100.0 + (i * 0.1) for i in range(len(stamps))]
        tl = _tl(stamps, ltps)

        out_3m = compute_path_outcomes(
            tl, ts=t0, entry_ltp=100.0, horizon_sec=180.0, predicted_ltp=105.0
        )
        out_10m = compute_path_outcomes(
            tl, ts=t0, entry_ltp=100.0, horizon_sec=600.0, predicted_ltp=105.0
        )
        self.assertEqual(out_3m["exit_at"], t0 + 180.0)
        self.assertEqual(out_10m["exit_at"], t0 + 600.0)
        self.assertEqual(out_3m["horizon_sec"], 180.0)
        self.assertEqual(out_10m["horizon_sec"], 600.0)
        self.assertEqual(out_3m["actual_max_profit"], out_3m["actual_max_profit_5m"])


if __name__ == "__main__":
    unittest.main()

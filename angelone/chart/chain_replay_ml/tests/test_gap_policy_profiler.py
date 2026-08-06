"""Tests for gap policy profiler and O(1) row gap checks."""

from __future__ import annotations

import unittest

from chain_replay_ml.dataset_builder.extended_features import _ema_series_from_prices
from chain_replay_ml.dataset_builder.gap_policy_profiler import (
    row_gap_exceeds,
    start_gap_policy_profiler,
    stop_gap_policy_profiler,
)
import numpy as np


class GapPolicyProfilerTests(unittest.TestCase):
    def test_row_gap_o1_no_policy(self) -> None:
        self.assertFalse(row_gap_exceeds(100.0, 80.0, None))
        self.assertFalse(row_gap_exceeds(100.0, 80.0, 0.0))

    def test_row_gap_detects_only_when_delta_exceeds(self) -> None:
        self.assertFalse(row_gap_exceeds(30.0, 10.0, 20.0))
        self.assertTrue(row_gap_exceeds(31.0, 10.0, 20.0))

    def test_profiler_counts_checks_and_resets(self) -> None:
        start_gap_policy_profiler(gap_max_sec=20.0, use_cprofile=False)
        from chain_replay_ml.dataset_builder.gap_policy_instrumentation import record_gap_check, record_gap_reset

        record_gap_check(is_gap=False)
        record_gap_check(is_gap=True)
        record_gap_reset()
        stats = stop_gap_policy_profiler()
        self.assertIsNotNone(stats)
        assert stats is not None
        self.assertEqual(stats.gap_checks, 2)
        self.assertEqual(stats.gaps_detected, 1)
        self.assertEqual(stats.reset_count, 1)

    def test_ema_gap_path_unchanged(self) -> None:
        prices = np.array([100.0, 100.0, 100.0, 110.0], dtype=float)
        tick_ts = np.array([0.0, 10.0, 50.0, 60.0], dtype=float)
        ema = _ema_series_from_prices(prices, 9, last_tick_ts=tick_ts, gap_max_sec=20.0)
        self.assertEqual(float(ema[2]), 100.0)
        self.assertGreater(float(ema[3]), float(ema[2]))


    def test_diff_cprofile_totals_sorts_by_delta(self) -> None:
        from chain_replay_ml.dataset_builder.gap_policy_profiler import diff_cprofile_totals

        off = {"a": 1.0, "b": 0.5, "c": 2.0}
        on = {"a": 1.1, "b": 2.0, "c": 2.0}
        rows = diff_cprofile_totals(off, on, limit=5)
        self.assertEqual(rows[0]["function"], "b")
        self.assertGreater(float(rows[0]["delta_sec"]), 0.0)


    def test_start_profiler_cleans_stale_session(self) -> None:
        start_gap_policy_profiler(gap_max_sec=20.0, use_cprofile=True)
        # Simulate a leaked session: stop without disabling cProfile reference.
        stats = start_gap_policy_profiler(gap_max_sec=None, use_cprofile=True)
        self.assertIsNotNone(stats)
        stopped = stop_gap_policy_profiler()
        self.assertIsNotNone(stopped)

    def test_nested_start_does_not_raise(self) -> None:
        start_gap_policy_profiler(gap_max_sec=20.0, use_cprofile=True)
        start_gap_policy_profiler(gap_max_sec=None, use_cprofile=True)
        stats = stop_gap_policy_profiler()
        self.assertIsNotNone(stats)
        self.assertIsNone(stop_gap_policy_profiler())

    def test_build_gap_pass_comparison_doc_pins_rows(self) -> None:
        from chain_replay_ml.dataset_builder.gap_policy_profiler import build_gap_pass_comparison_doc

        off = {
            "C:\\proj\\feature_enrichment.py:37(build_feature_raw_for_row)": 9.0,
            "C:\\proj\\features_atm_band.py:254(extract_timeline_features)": 4.0,
            "C:\\proj\\ticks.py:37(ltp_rupees_at)": 1.0,
        }
        on = {
            "C:\\proj\\feature_enrichment.py:37(build_feature_raw_for_row)": 14.0,
            "C:\\proj\\features_atm_band.py:254(extract_timeline_features)": 4.2,
            "C:\\proj\\ticks.py:37(ltp_rupees_at)": 1.0,
        }
        off_calls = {
            "C:\\proj\\feature_enrichment.py:37(build_feature_raw_for_row)": 4239,
            "C:\\proj\\features_atm_band.py:254(extract_timeline_features)": 4239,
            "C:\\proj\\ticks.py:37(ltp_rupees_at)": 100000,
        }
        on_calls = {
            "C:\\proj\\feature_enrichment.py:37(build_feature_raw_for_row)": 4239,
            "C:\\proj\\features_atm_band.py:254(extract_timeline_features)": 8478,
            "C:\\proj\\ticks.py:37(ltp_rupees_at)": 100000,
        }
        doc = build_gap_pass_comparison_doc(
            off_totals=off,
            on_totals=on,
            off_calls=off_calls,
            on_calls=on_calls,
            off_wall_sec=9.0,
            on_wall_sec=14.0,
            gap_on_max_sec=20.0,
        )
        self.assertAlmostEqual(float(doc["delta_wall_sec"]), 5.0)
        labels = [str(r["label"]) for r in doc["by_function"]]
        self.assertIn("build_feature_raw_for_row", labels)
        self.assertIn("extract_timeline_features", labels)
        self.assertNotIn("ltp_rupees_at", labels)
        top = doc["by_function"][0]
        self.assertEqual(top["label"], "build_feature_raw_for_row")
        self.assertAlmostEqual(float(top["delta_sec"]), 5.0)
        self.assertEqual(doc["dominant_function"], "build_feature_raw_for_row")
        self.assertAlmostEqual(float(doc["dominant_delta_sec"]), 5.0)
        timeline = next(r for r in doc["by_function"] if r["label"] == "extract_timeline_features")
        self.assertEqual(int(timeline["calls_off"]), 4239)
        self.assertEqual(int(timeline["calls_on"]), 8478)


    def test_compare_gap_profiles_runs_two_builds(self) -> None:
        from unittest.mock import patch

        from chain_replay_ml.dataset_builder.gap_policy_profiler import compare_gap_profiles

        class _Ctx:
            feature_grid_step_sec = 10
            feature_grid_gap_max_sec = 0.0

        ctx = _Ctx()
        kwargs = dict(
            step_sec=10,
            strike_selection={"mode": "atm_band", "band": 10},
            horizons_sec=[5, 10],
            enabled_groups=["core"],
            group_labels={"core": "Core"},
            implemented_features=["ltp"],
            per_group_features={"core": ["ltp"]},
            gap_on_max_sec=20.0,
        )
        with patch("chain_replay_ml.dataset_builder.stages.build_day_rows", return_value=([], {})) as mock_build:
            doc = compare_gap_profiles(ctx, **kwargs)
        self.assertEqual(mock_build.call_count, 2)
        self.assertEqual(mock_build.call_args_list[0].kwargs.get("gap_max_sec"), None)
        self.assertEqual(mock_build.call_args_list[1].kwargs.get("gap_max_sec"), 20.0)
        self.assertIn("by_function", doc)
        self.assertIn("gap_off_wall_sec", doc)


if __name__ == "__main__":
    unittest.main()

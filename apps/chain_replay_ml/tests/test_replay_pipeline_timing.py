"""Tests for replay pipeline stage timing helpers."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from chain_replay_ml.feature_policy.replay_pipeline_timing import (
    benchmark_build_day_rows_cold_warm,
    finalize_pipeline_stages,
    merge_frame_timing,
    pipeline_total,
)
from master_dataset_tk import warmup_simulator_format as sim_fmt


class ReplayPipelineTimingTests(unittest.TestCase):
    def test_merge_frame_timing(self) -> None:
        stats = {"timing_sec": {"load_day_context": 0.0, "build_day_rows": 9.08, "to_dataframe": 0.12}}
        merged = merge_frame_timing(stats)
        self.assertAlmostEqual(merged["build_day_rows_sec"], 9.08)
        self.assertAlmostEqual(merged["to_dataframe_sec"], 0.12)

    def test_pipeline_total(self) -> None:
        stages = finalize_pipeline_stages({
            "load_day_context_sec": 0.0,
            "build_day_rows_sec": 9.08,
            "to_dataframe_sec": 0.12,
            "serialize_replay_rows_sec": 2.41,
            "replay_statistics_sec": 0.61,
            "build_replay_lookup_sec": 2.03,
        })
        self.assertAlmostEqual(pipeline_total(stages), 14.25)
        self.assertAlmostEqual(float(stages["total_sec"]), 14.25)

    def test_format_replay_pipeline(self) -> None:
        text = sim_fmt.format_replay_pipeline({
            "replay_pipeline": {
                "build_day_rows_sec": 9.08,
                "serialize_replay_rows_sec": 2.41,
                "build_replay_lookup_sec": 2.03,
                "total_sec": 14.25,
            },
        })
        self.assertIn("Replay Pipeline", text)
        self.assertIn("build_day_rows", text)
        self.assertIn("serialize_replay_rows", text)
        self.assertIn("14.250", text)

    def test_format_replay_context_benchmark(self) -> None:
        text = sim_fmt.format_replay_context_benchmark({
            "replay_context_benchmark": {
                "cold_build_day_rows_sec": 14.4,
                "warm_build_day_rows_sec": 9.1,
                "cache_savings_sec": 5.3,
            },
        })
        self.assertIn("Cold build", text)
        self.assertIn("14.400", text)
        self.assertIn("9.100", text)

    def test_benchmark_build_day_rows_cold_warm(self) -> None:
        from chain_replay_ml.feature_policy.performance_debug import (
            PerformanceDebugConfig,
            PerformanceDebugLevel,
        )

        class _Ctx:
            feature_grid_step_sec = 10
            feature_grid_gap_max_sec = 0.0

        ctx = _Ctx()
        build_kw = {
            "step_sec": 10,
            "strike_selection": {"mode": "atm_band"},
            "horizons_sec": [],
            "enabled_groups": ["core"],
            "group_labels": {"core": "Core"},
            "implemented_features": ["ltp"],
            "per_group_features": {"core": ["ltp"]},
            "lookback_policy_doc": None,
            "trim_target_rows": False,
            "active_features": None,
            "gap_max_sec": None,
            "performance_debug": PerformanceDebugConfig(level=PerformanceDebugLevel.FULL),
        }
        with patch("chain_replay_ml.dataset_builder.stages.build_day_rows", return_value=([], {})):
            doc = benchmark_build_day_rows_cold_warm(
                ctx,
                build_kwargs=build_kw,
                performance_debug=PerformanceDebugConfig(level=PerformanceDebugLevel.FULL),
            )
        self.assertIn("cold_build_day_rows_sec", doc)
        self.assertIn("warm_build_day_rows_sec", doc)


if __name__ == "__main__":
    unittest.main()

"""Tests and benchmark for PerformanceDebugLevel / PerformanceDebugConfig gating."""

from __future__ import annotations

import sys
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

_CHART = Path(__file__).resolve().parents[2]
if str(_CHART) not in sys.path:
    sys.path.insert(0, str(_CHART))

from chain_replay_ml.dataset_builder.gap_policy_instrumentation import profiler_active
from chain_replay_ml.feature_policy.performance_debug import (
    PerformanceDebugConfig,
    PerformanceDebugLevel,
)
from master_dataset_tk import warmup_simulator_format as sim_fmt


def _sample_rows() -> list[dict]:
    return [
        {
            "trading_day": "2026-01-02",
            "timestamp": 1000.0,
            "strike": 24000,
            "option_type": "CE",
            "token": "1",
            "ltp": 100.5,
        },
    ]


def _strip_profiler_stats(stats: dict) -> dict:
    out = dict(stats)
    for key in (
        "gap_policy_profiler",
        "readiness_profiler",
        "feature_readiness",
    ):
        out.pop(key, None)
    return out


class PerformanceDebugConfigTests(unittest.TestCase):
    def test_resolve_defaults_off(self) -> None:
        perf = PerformanceDebugConfig.resolve(None)
        self.assertEqual(perf.level, PerformanceDebugLevel.OFF)
        self.assertFalse(perf.collect_gap_profile())
        self.assertFalse(perf.collect_cprofile())
        self.assertFalse(perf.collect_pipeline_timings())

    def test_basic_pipeline_only(self) -> None:
        perf = PerformanceDebugConfig.resolve(PerformanceDebugLevel.BASIC)
        self.assertTrue(perf.collect_pipeline_timings())
        self.assertFalse(perf.collect_gap_profile())
        self.assertFalse(perf.collect_cprofile())
        self.assertFalse(perf.run_cache_benchmark())

    def test_full_enables_all_collectors(self) -> None:
        perf = PerformanceDebugConfig.resolve(PerformanceDebugLevel.FULL)
        self.assertTrue(perf.collect_gap_profile())
        self.assertTrue(perf.collect_cprofile())
        self.assertTrue(perf.collect_readiness_profile())
        self.assertTrue(perf.run_cache_benchmark())
        self.assertTrue(perf.run_gap_pass_comparison(gap_parity=True))

    def test_legacy_readiness_bumps_to_full(self) -> None:
        perf = PerformanceDebugConfig.resolve(
            PerformanceDebugLevel.OFF,
            readiness_profile=True,
        )
        self.assertEqual(perf.level, PerformanceDebugLevel.FULL)


class BuildDayRowsProfilerGatingTests(unittest.TestCase):
    def _run_stages_off_empty_grid(self, perf: PerformanceDebugConfig) -> None:
        from chain_replay_ml.dataset_builder.stages import build_day_rows

        ctx = MagicMock()
        with patch(
            "chain_replay_ml.dataset_builder.tick_coverage.list_clipped_grid_timestamps",
            return_value=[],
        ), patch(
            "chain_replay_ml.dataset_builder.tick_coverage.build_clipped_sample_timestamps",
            return_value=([], {}),
        ), patch(
            "chain_replay_ml.dataset_builder.stages.sync_feature_grid_step",
        ), patch(
            "chain_replay_ml.dataset_builder.gap_policy_profiler.start_gap_policy_profiler",
        ) as mock_start, patch(
            "chain_replay_ml.dataset_builder.gap_policy_profiler.stop_gap_policy_profiler",
        ) as mock_stop:
            build_day_rows(
                ctx,
                step_sec=10,
                strike_selection={"mode": "atm_band", "band": 10},
                horizons_sec=[],
                enabled_groups=["core"],
                group_labels={"core": "Core"},
                implemented_features=["ltp"],
                per_group_features={"core": ["ltp"]},
                gap_max_sec=20.0,
                performance_debug=perf,
            )
        mock_start.assert_not_called()
        mock_stop.assert_not_called()

    def test_off_skips_gap_profiler(self) -> None:
        self._run_stages_off_empty_grid(PerformanceDebugConfig(level=PerformanceDebugLevel.OFF))

    def test_basic_skips_gap_profiler(self) -> None:
        self._run_stages_off_empty_grid(PerformanceDebugConfig(level=PerformanceDebugLevel.BASIC))

    def test_full_starts_gap_profiler_with_cprofile(self) -> None:
        from chain_replay_ml.dataset_builder.stages import build_day_rows

        ctx = MagicMock()
        with patch(
            "chain_replay_ml.dataset_builder.tick_coverage.list_clipped_grid_timestamps",
            return_value=[],
        ), patch(
            "chain_replay_ml.dataset_builder.tick_coverage.build_clipped_sample_timestamps",
            return_value=([], {}),
        ), patch(
            "chain_replay_ml.dataset_builder.stages.sync_feature_grid_step",
        ), patch(
            "chain_replay_ml.dataset_builder.gap_policy_profiler.start_gap_policy_profiler",
        ) as mock_start, patch(
            "chain_replay_ml.dataset_builder.gap_policy_profiler.stop_gap_policy_profiler",
        ):
            build_day_rows(
                ctx,
                step_sec=10,
                strike_selection={"mode": "atm_band", "band": 10},
                horizons_sec=[],
                enabled_groups=["core"],
                group_labels={"core": "Core"},
                implemented_features=["ltp"],
                per_group_features={"core": ["ltp"]},
                gap_max_sec=20.0,
                performance_debug=PerformanceDebugConfig(level=PerformanceDebugLevel.FULL),
            )
        mock_start.assert_called_once()
        self.assertTrue(mock_start.call_args.kwargs.get("use_cprofile"))


class DatasetParityTests(unittest.TestCase):
    def test_off_and_full_empty_grid_identical(self) -> None:
        from chain_replay_ml.dataset_builder.stages import build_day_rows

        ctx = MagicMock()
        shared = dict(
            step_sec=10,
            strike_selection={"mode": "atm_band", "band": 10},
            horizons_sec=[],
            enabled_groups=["core"],
            group_labels={"core": "Core"},
            implemented_features=["ltp"],
            per_group_features={"core": ["ltp"]},
            gap_max_sec=20.0,
        )
        with patch(
            "chain_replay_ml.dataset_builder.tick_coverage.list_clipped_grid_timestamps",
            return_value=[],
        ), patch(
            "chain_replay_ml.dataset_builder.tick_coverage.build_clipped_sample_timestamps",
            return_value=([], {}),
        ), patch(
            "chain_replay_ml.dataset_builder.stages.sync_feature_grid_step",
        ):
            off_rows, off_stats = build_day_rows(
                ctx,
                performance_debug=PerformanceDebugConfig(level=PerformanceDebugLevel.OFF),
                **shared,
            )
            full_rows, full_stats = build_day_rows(
                ctx,
                performance_debug=PerformanceDebugConfig(level=PerformanceDebugLevel.FULL),
                **shared,
            )

        self.assertEqual(off_rows, full_rows)
        self.assertEqual(_strip_profiler_stats(off_stats), _strip_profiler_stats(full_stats))


class HotPathImportTests(unittest.TestCase):
    def test_instrumentation_module_has_no_cprofile(self) -> None:
        import chain_replay_ml.dataset_builder.gap_policy_instrumentation as inst

        self.assertNotIn("cProfile", inst.__dict__)
        self.assertFalse(profiler_active())

    def test_feature_enrichment_imports_instrumentation_only(self) -> None:
        import chain_replay_ml.dataset_builder.feature_enrichment as fe

        self.assertIn("gap_policy_instrumentation", fe.gap_policy_profile_block.__module__)


class CacheBenchmarkGatingTests(unittest.TestCase):
    def test_off_skips_cache_benchmark(self) -> None:
        from chain_replay_ml.feature_policy.replay_pipeline_timing import benchmark_build_day_rows_cold_warm

        ctx = MagicMock()
        out = benchmark_build_day_rows_cold_warm(
            ctx,
            build_kwargs={"step_sec": 10},
            performance_debug=PerformanceDebugConfig(level=PerformanceDebugLevel.OFF),
        )
        self.assertEqual(out, {})


class FormatLevelTests(unittest.TestCase):
    def test_off_shows_production_summary_only(self) -> None:
        from chain_replay_ml.feature_policy.warmup_simulator import WarmupSimulationResult

        result = WarmupSimulationResult(
            timing={
                "performance_debug_level": "off",
                "load_ticks_sec": 1.0,
                "build_grid_sec": 0.1,
                "policy_engine_sec": 0.2,
                "feature_calc_sec": 3.0,
                "total_sec": 4.5,
                "gap_policy_profiler": {"gap_checks": 99},
                "replay_pipeline": {"total_sec": 2.5},
            },
        )
        text = sim_fmt.format_timing_summary(result)
        self.assertIn("Fetch ticks", text)
        self.assertNotIn("Replay Pipeline", text)
        self.assertNotIn("Gap Policy", text)

    def test_basic_shows_pipeline_not_gap_profiler(self) -> None:
        from chain_replay_ml.feature_policy.warmup_simulator import WarmupSimulationResult

        result = WarmupSimulationResult(
            timing={
                "performance_debug_level": "basic",
                "load_ticks_sec": 1.0,
                "build_grid_sec": 0.1,
                "policy_engine_sec": 0.2,
                "feature_calc_sec": 3.0,
                "total_sec": 4.5,
                "gap_policy_profiler": {"gap_checks": 99},
                "replay_pipeline": {
                    "build_day_rows_sec": 2.0,
                    "total_sec": 2.5,
                },
            },
        )
        text = sim_fmt.format_timing_summary(result)
        self.assertIn("Replay Pipeline", text)
        self.assertNotIn("Gap Policy", text)


class PerformanceDebugBenchmarkTests(unittest.TestCase):
    """Measure profiler start/stop overhead across debug modes (empty grid path)."""

    def _timed_empty_build(self, perf: PerformanceDebugConfig, *, repeats: int = 20) -> float:
        from chain_replay_ml.dataset_builder.stages import build_day_rows

        ctx = MagicMock()
        shared = dict(
            step_sec=10,
            strike_selection={"mode": "atm_band", "band": 10},
            horizons_sec=[],
            enabled_groups=["core"],
            group_labels={"core": "Core"},
            implemented_features=["ltp"],
            per_group_features={"core": ["ltp"]},
            gap_max_sec=20.0,
        )
        timings: list[float] = []
        with patch(
            "chain_replay_ml.dataset_builder.tick_coverage.list_clipped_grid_timestamps",
            return_value=[],
        ), patch(
            "chain_replay_ml.dataset_builder.tick_coverage.build_clipped_sample_timestamps",
            return_value=([], {}),
        ), patch(
            "chain_replay_ml.dataset_builder.stages.sync_feature_grid_step",
        ):
            for _ in range(repeats):
                t0 = time.perf_counter()
                build_day_rows(ctx, performance_debug=perf, **shared)
                timings.append(time.perf_counter() - t0)
        return round(sum(timings) / len(timings), 6)

    def test_benchmark_table(self) -> None:
        modes = {
            "OFF": PerformanceDebugConfig(level=PerformanceDebugLevel.OFF),
            "BASIC": PerformanceDebugConfig(level=PerformanceDebugLevel.BASIC),
            "FULL": PerformanceDebugConfig(level=PerformanceDebugLevel.FULL),
        }
        results = {name: self._timed_empty_build(perf) for name, perf in modes.items()}
        off_total = results["OFF"]
        lines = [
            "",
            "Performance debug benchmark (empty-grid build_day_rows)",
            "| Mode | Total Time | build_day_rows | Extra Overhead |",
            "|------|------------|----------------|----------------|",
        ]
        for name in ("OFF", "BASIC", "FULL"):
            total = results[name]
            extra = round(total - off_total, 6)
            lines.append(
                f"| {name} | {total:.6f}s | {total:.6f}s | {extra:+.6f}s |",
            )
        print("\n".join(lines))

        self.assertLessEqual(results["OFF"], results["BASIC"] + 0.01)
        self.assertGreater(results["FULL"], results["OFF"])


if __name__ == "__main__":
    unittest.main()

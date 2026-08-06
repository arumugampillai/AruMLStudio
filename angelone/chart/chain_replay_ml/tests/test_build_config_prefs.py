"""Tests for build config persistence."""

from __future__ import annotations

import json
import os
import tempfile
import unittest

from chain_replay_ml.feature_policy.warmup_simulator import WarmupSimulationResult
from master_dataset_tk import warmup_simulator_format as sim_fmt
from master_dataset_tk.build_config_prefs import (
    infer_simulator_duration_preset,
    load_build_config_prefs,
    resolve_enabled_groups,
    resolve_simulator_duration_minutes,
    save_build_config_prefs,
    simulator_duration_prefs_for_save,
    simulator_duration_preset_for_save,
)


class BuildConfigPrefsTests(unittest.TestCase):
    def test_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            chart_dir = os.path.join(tmp, "chart")
            os.makedirs(os.path.join(chart_dir, "data"))
            save_build_config_prefs(chart_dir, {
                "build": {
                    "enabled_groups": ["greeks", "volume"],
                    "horizons_sec": [10, 60],
                    "sampling_interval_sec": 10,
                },
            })
            doc = load_build_config_prefs(chart_dir)
            self.assertIsNotNone(doc)
            build = doc.get("build")
            self.assertEqual(build.get("enabled_groups"), ["greeks", "volume"])

    def test_resolve_enabled_groups_partial(self) -> None:
        states = resolve_enabled_groups(
            ["greeks"],
            ["greeks", "volume", "oi"],
            default_all=True,
        )
        self.assertTrue(states["greeks"])
        self.assertFalse(states["volume"])
        self.assertFalse(states["oi"])

    def test_resolve_enabled_groups_none(self) -> None:
        states = resolve_enabled_groups([], ["greeks", "volume"], default_all=True)
        self.assertFalse(states["greeks"])
        self.assertFalse(states["volume"])

    def test_simulation_trace_csv(self) -> None:
        result = WarmupSimulationResult(
            ok=True,
            feature_name="ltp_ema20_to_ltp_ratio",
            trading_day="2026-05-27",
            controller_label="EMA20",
            dependency_labels=["LTP", "EMA20"],
            full_trace=[
                {
                    "samples": 1,
                    "time": "09:15:00",
                    "ts": 100.0,
                    "ctrl_ready": False,
                    "ctrl_samples": 1,
                    "feature_ready": False,
                    "output_display": "NULL",
                    "deps": {"LTP": True, "EMA20": False},
                },
            ],
        )
        csv_text = sim_fmt.simulation_trace_csv(result)
        self.assertIn("feature,trading_day,sample", csv_text)
        self.assertIn("ltp_ema20_to_ltp_ratio", csv_text)
        self.assertIn("NULL", csv_text)

    def test_format_timing_summary(self) -> None:
        result = WarmupSimulationResult(
            ok=True,
            feature_name="ltp_ema20_to_ltp_ratio",
            trading_day="2026-05-27",
            samples_processed=120,
            timing={
                "load_ticks_sec": 2.34,
                "spot_ticks": 12_345,
                "chain_ticks": 2_450_000,
                "source_ticks": 2_462_345,
                "build_grid_sec": 0.12,
                "policy_engine_sec": 0.56,
                "feature_calc_sec": 45.2,
                "maturity_replay_sec": 38.1,
                "total_sec": 86.32,
                "run_completed_at": "2026-07-11 12:38:05",
                "performance_debug_level": "full",
            },
            calc_debug={"ok": True, "rows": [{}] * 12},
            maturity_replay_lookup={1: {}},
        )
        text = sim_fmt.format_timing_summary(result)
        self.assertIn("Time Taken (2026-07-11 12:38:05)", text)
        self.assertEqual(
            sim_fmt.timing_tab_title(result),
            "Time Taken (2026-07-11 12:38:05)",
        )
        self.assertIn("Fetching ticks", text)
        self.assertIn("12,345 spot", text)
        self.assertIn("2,450,000 chain", text)
        self.assertIn("Feature calculation", text)
        self.assertIn("2.34 s", text)
        self.assertIn("45.20 s", text)

    def test_all_features_csv(self) -> None:
        result = WarmupSimulationResult(
            ok=True,
            feature_name="ltp_ema20_to_ltp_ratio",
            trading_day="2026-05-27",
            sampling_interval_sec=10.0,
            maturity_feature_names=["ltp_ema20_to_ltp_ratio", "ltp"],
            all_features_lookup={
                100: {"ltp_ema20_to_ltp_ratio": 1.002, "ltp": 24500.0},
            },
            all_features_rows=[
                {
                    "timestamp": 100.0,
                    "token": "111",
                    "symbol": "NIFTY CE",
                    "strike": 24500,
                    "option_type": "CE",
                    "ltp_ema20_to_ltp_ratio": 1.002,
                    "ltp": 100.5,
                },
                {
                    "timestamp": 100.0,
                    "token": "222",
                    "symbol": "NIFTY PE",
                    "strike": 24500,
                    "option_type": "PE",
                    "ltp_ema20_to_ltp_ratio": 0.998,
                    "ltp": 99.5,
                },
            ],
            full_trace=[
                {"samples": 1, "time": "09:15:00", "ts": 100.0},
            ],
        )
        csv_text = sim_fmt.all_features_csv(result)
        self.assertIn("token,symbol,strike,option_type", csv_text)
        self.assertIn("111", csv_text)
        self.assertIn("222", csv_text)
        self.assertIn("1.002", csv_text)
        self.assertIn("0.998", csv_text)

    def test_all_features_csv_includes_targets(self) -> None:
        result = WarmupSimulationResult(
            ok=True,
            feature_name="ltp",
            trading_day="2026-05-27",
            sampling_interval_sec=10.0,
            horizons_sec=[60, 300],
            target_columns=["future_ltp_1m", "future_ltp_5m"],
            all_features_rows=[
                {
                    "timestamp": 100.0,
                    "token": "111",
                    "strike": 24500,
                    "option_type": "CE",
                    "ltp": 100.5,
                    "future_ltp_1m": 101.0,
                    "future_ltp_5m": 102.0,
                },
            ],
            full_trace=[{"samples": 1, "time": "09:15:00", "ts": 100.0}],
        )
        csv_text = sim_fmt.all_features_csv(result)
        self.assertIn("future_ltp_1m", csv_text)
        self.assertIn("future_ltp_5m", csv_text)
        self.assertIn("101.0", csv_text)

    def test_all_features_csv_uses_maturity_lookup_fallback(self) -> None:
        result = WarmupSimulationResult(
            ok=True,
            feature_name="weighted_ltp_ema_to_ltp_ratio",
            trading_day="2026-07-01",
            sampling_interval_sec=3.0,
            maturity_feature_names=["weighted_ltp_ema_to_ltp_ratio", "ltp", "spot"],
            maturity_replay_lookup={
                1782811773: {
                    "ltp": 20.35,
                    "spot": 23981.75,
                    "weighted_ltp_ema_to_ltp_ratio": 1.0,
                    "token": "999",
                },
            },
            full_trace=[
                {"samples": 1, "time": "09:59:33", "ts": 1782811773.0},
            ],
        )
        status = sim_fmt.all_features_export_status(result)
        self.assertTrue(status["ok"])
        csv_text = sim_fmt.all_features_csv(result)
        self.assertIn("weighted_ltp_ema_to_ltp_ratio", csv_text)
        self.assertIn("20.35", csv_text)

    def test_format_timing_summary_chain_counts(self) -> None:
        result = WarmupSimulationResult(
            ok=True,
            feature_name="ltp_ema20_to_ltp_ratio",
            trading_day="2026-05-27",
            samples_processed=101,
            timing={
                "load_ticks_sec": 0.32,
                "feature_calc_sec": 8.36,
                "all_features_calc": True,
                "total_sec": 9.99,
                "policy_grid_samples": 101,
                "chain_rows_total": 4239,
                "chain_grid_timestamps": 101,
                "strikes_in_band": 21,
                "rows_per_timestamp": 42,
                "expected_chain_rows": 4242,
                "avg_rows_per_timestamp": 42.0,
                "unique_tokens": 44,
                "strike_selection_label": "ATM ±10",
                "performance_debug_level": "full",
            },
        )
        text = sim_fmt.format_timing_summary(result)
        self.assertIn("Sample counts", text)
        self.assertIn("4,239", text)
        self.assertIn("21 × CE/PE", text)
        self.assertIn("4,242", text)
        self.assertIn("ATM ±10", text)

    def test_format_timing_summary_all_features(self) -> None:
        result = WarmupSimulationResult(
            ok=True,
            feature_name="ltp_ema20_to_ltp_ratio",
            trading_day="2026-05-27",
            samples_processed=120,
            timing={
                "load_ticks_sec": 2.34,
                "feature_calc_sec": 45.2,
                "all_features_calc": True,
                "maturity_replay_sec": 0.0,
                "maturity_replay_shared": True,
                "total_sec": 50.0,
                "performance_debug_level": "full",
            },
            all_features_lookup={100: {"ltp": 1.0}},
            maturity_feature_names=["ltp"],
        )
        text = sim_fmt.format_timing_summary(result)
        self.assertIn("All features calculation", text)
        self.assertIn("shared with all-features calc", text)
        self.assertIn("All features lookup", text)

    def test_format_timing_summary_lookback_benchmark_skipped(self) -> None:
        result = WarmupSimulationResult(
            ok=True,
            feature_name="ltp",
            trading_day="2026-05-27",
            samples_processed=101,
            timing={
                "load_ticks_sec": 0.5,
                "feature_calc_sec": 8.68,
                "all_features_calc": True,
                "lookback_nearest_snapshot": True,
                "lookback_policy_method": "nearest_snapshot",
                "lookback_benchmark_skipped": "dual-pass lookback benchmark disabled — single nearest_snapshot pass",
                "match_build_dataset_selection": False,
                "match_build_gap_parity": False,
                "total_sec": 10.0,
                "performance_debug_level": "full",
            },
        )
        text = sim_fmt.format_timing_summary(result)
        self.assertIn("Lookback benchmark                    skipped", text)
        self.assertNotIn("exact_timestamp baseline", text)
        self.assertIn("Lookback nearest_snapshot             ON", text)

    def test_format_timing_summary_lookback_benchmark(self) -> None:
        result = WarmupSimulationResult(
            ok=True,
            feature_name="ltp",
            trading_day="2026-05-27",
            samples_processed=101,
            timing={
                "load_ticks_sec": 0.5,
                "feature_calc_sec": 29.29,
                "feature_calc_without_lookback_sec": 8.94,
                "lookback_nearest_snapshot_sec": 20.35,
                "all_features_calc_wall_sec": 38.23,
                "all_features_calc": True,
                "lookback_nearest_snapshot": True,
                "lookback_policy_method": "nearest_snapshot",
                "chain_rows_total": 100,
                "total_sec": 40.0,
                "performance_debug_level": "full",
            },
        )
        text = sim_fmt.format_timing_summary(result)
        self.assertIn("exact_timestamp baseline", text)
        self.assertIn("Lookback nearest_snapshot overhead", text)
        self.assertIn("20.35 s", text)
        self.assertIn("Lookback nearest_snapshot             ON", text)

    def test_format_timing_summary_temp_build_io(self) -> None:
        result = WarmupSimulationResult(
            ok=True,
            feature_name="ltp",
            trading_day="2026-05-27",
            samples_processed=101,
            timing={
                "load_ticks_sec": 0.5,
                "feature_calc_sec": 8.0,
                "all_features_calc": True,
                "temp_sqlite_insert_sec": 1.23,
                "temp_sqlite_rows": 4239,
                "temp_sqlite_bytes": 512_000,
                "temp_parquet_export_sec": 0.87,
                "temp_parquet_rows": 4239,
                "temp_parquet_bytes": 256_000,
                "total_sec": 10.6,
                "performance_debug_level": "full",
            },
        )
        text = sim_fmt.format_timing_summary(result)
        self.assertIn("Temp SQLite insert", text)
        self.assertIn("Temp Parquet export", text)
        self.assertIn("4,239", text)
        self.assertIn("500.0 KB", text)

    def test_run_temp_build_io(self) -> None:
        from chain_replay_ml.feature_policy.warmup_simulator import _run_temp_build_io

        rows = [
            {
                "trading_day": "2026-05-27",
                "timestamp": 100.0,
                "token": "111",
                "strike": 24500,
                "option_type": "CE",
                "ltp": 100.5,
            },
            {
                "trading_day": "2026-05-27",
                "timestamp": 100.0,
                "token": "222",
                "strike": 24500,
                "option_type": "PE",
                "ltp": 99.5,
            },
        ]
        result = WarmupSimulationResult(ok=True, trading_day="2026-05-27")
        _run_temp_build_io(rows, trading_day="2026-05-27", result=result, on_progress=None)
        self.assertEqual(result.timing.get("temp_sqlite_rows"), 2)
        self.assertEqual(result.timing.get("temp_parquet_rows"), 2)
        self.assertGreater(result.timing.get("temp_sqlite_insert_sec", 0), 0)
        self.assertGreater(result.timing.get("temp_parquet_export_sec", 0), 0)
        self.assertNotIn("temp_build_io_error", result.timing)

    def test_simulator_custom_duration_preset_save(self) -> None:
        self.assertEqual(simulator_duration_preset_for_save(0), 0)
        self.assertEqual(simulator_duration_preset_for_save(15), 15)

    def test_resolve_simulator_duration_minutes_custom(self) -> None:
        self.assertEqual(
            resolve_simulator_duration_minutes(preset_minutes=0, custom_minutes="60"),
            60,
        )
        self.assertEqual(
            resolve_simulator_duration_minutes(preset_minutes=15, custom_minutes="60"),
            15,
        )

    def test_infer_simulator_duration_preset_recovers_custom(self) -> None:
        self.assertEqual(infer_simulator_duration_preset(0, "60"), 0)
        self.assertEqual(infer_simulator_duration_preset(15, "15"), 15)
        self.assertEqual(infer_simulator_duration_preset(10, "60"), 10)
        self.assertEqual(infer_simulator_duration_preset(15, "60"), 15)

    def test_format_timing_summary_gap_pass_comparison_error_full_only(self) -> None:
        result = WarmupSimulationResult(
            ok=True,
            feature_name="ltp",
            trading_day="2026-05-27",
            samples_processed=101,
            timing={
                "load_ticks_sec": 0.5,
                "feature_calc_sec": 8.0,
                "total_sec": 10.0,
                "performance_debug_level": "full",
                "gap_pass_comparison_error": "Another profiling tool is already active",
            },
        )
        text = sim_fmt.format_timing_summary(result)
        self.assertIn("Gap pass compare error", text)
        self.assertIn("Another profiling tool is already active", text)

        result.timing["performance_debug_level"] = "off"
        off_text = sim_fmt.format_timing_summary(result)
        self.assertNotIn("Gap pass compare error", off_text)

    def test_simulator_duration_prefs_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            chart_dir = os.path.join(tmp, "chart")
            os.makedirs(os.path.join(chart_dir, "data"))
            save_build_config_prefs(chart_dir, {
                "simulator": {
                    **simulator_duration_prefs_for_save(preset_minutes=10, custom_minutes="60"),
                    "trading_day": "2026-07-01",
                },
            })
            doc = load_build_config_prefs(chart_dir) or {}
            sim = doc.get("simulator") or {}
            self.assertEqual(sim.get("duration_minutes"), 10)
            self.assertEqual(sim.get("custom_duration"), "")
            self.assertEqual(
                infer_simulator_duration_preset(sim.get("duration_minutes"), sim.get("custom_duration")),
                10,
            )
            save_build_config_prefs(chart_dir, {
                "simulator": {
                    **simulator_duration_prefs_for_save(preset_minutes=0, custom_minutes="60"),
                },
            })
            doc = load_build_config_prefs(chart_dir) or {}
            sim = doc.get("simulator") or {}
            self.assertEqual(sim.get("duration_minutes"), 0)
            self.assertEqual(sim.get("custom_duration"), "60")
            self.assertEqual(sim.get("trading_day"), "2026-07-01")


if __name__ == "__main__":
    unittest.main()

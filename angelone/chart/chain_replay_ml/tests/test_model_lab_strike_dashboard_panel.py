"""Unit + smoke tests for Strike Prediction Dashboard."""

from __future__ import annotations

import inspect
import math
import unittest


class ConfidenceColumnResolutionTests(unittest.TestCase):
    def test_prefers_pred_prob_ladder(self) -> None:
        from chain_replay_ml.model_lab.strike_prediction_dashboard import (
            resolve_confidence_column,
        )

        col = resolve_confidence_column(
            ["current_ltp", "pred_prob_up_4pct_5m", "confidence_target_hit_pred"]
        )
        self.assertEqual(col, "pred_prob_up_4pct_5m")

    def test_prefers_2pct_when_multiple_probs(self) -> None:
        from chain_replay_ml.model_lab.strike_prediction_dashboard import (
            resolve_confidence_column,
        )

        col = resolve_confidence_column(
            ["pred_prob_up_6pct_5m", "pred_prob_up_2pct_5m", "tb_pred_probability"]
        )
        self.assertEqual(col, "pred_prob_up_2pct_5m")

    def test_falls_back_to_tb_then_binary(self) -> None:
        from chain_replay_ml.model_lab.strike_prediction_dashboard import (
            resolve_confidence_column,
        )

        self.assertEqual(
            resolve_confidence_column(["tb_pred_probability", "confidence_rr_1_1_pred"]),
            "tb_pred_probability",
        )
        self.assertEqual(
            resolve_confidence_column(["confidence_trade_winner_pred", "foo"]),
            "confidence_trade_winner_pred",
        )
        self.assertIsNone(resolve_confidence_column(["current_ltp", "strike"]))

    def test_honors_preferred(self) -> None:
        from chain_replay_ml.model_lab.strike_prediction_dashboard import (
            resolve_confidence_column,
        )

        self.assertEqual(
            resolve_confidence_column(
                ["pred_prob_up_2pct_5m", "confidence_target_hit_pred"],
                preferred="confidence_target_hit_pred",
            ),
            "confidence_target_hit_pred",
        )


class ErrorSeriesAndEmaTests(unittest.TestCase):
    def test_prediction_error_series(self) -> None:
        from chain_replay_ml.model_lab.strike_prediction_dashboard import (
            prediction_error_series,
        )

        err = prediction_error_series([110.0, 105.0, None], [100.0, None, 90.0])
        self.assertEqual(err[0], 10.0)
        self.assertIsNone(err[1])
        self.assertIsNone(err[2])

    def test_prediction_gap_series(self) -> None:
        from chain_replay_ml.model_lab.strike_prediction_dashboard import (
            prediction_gap_series,
        )

        gap = prediction_gap_series([108.0, None, 95.0], [100.0, 101.0, 90.0])
        self.assertEqual(gap[0], 8.0)
        self.assertIsNone(gap[1])
        self.assertEqual(gap[2], 5.0)

    def test_ema_slope_series(self) -> None:
        from chain_replay_ml.model_lab.strike_prediction_dashboard import (
            ema_slope_series,
        )

        slope = ema_slope_series([1.0, 1.5, 2.0, None, 3.0], window=1)
        self.assertIsNone(slope[0])
        self.assertAlmostEqual(slope[1], 0.5, places=6)
        self.assertAlmostEqual(slope[2], 0.5, places=6)
        self.assertIsNone(slope[3])
        self.assertIsNone(slope[4])  # prev is None

    def test_apply_ema_span_5(self) -> None:
        from chain_replay_ml.model_lab.strike_prediction_dashboard import apply_ema

        vals = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        ema5 = apply_ema(vals, 5)
        self.assertEqual(len(ema5), len(vals))
        self.assertAlmostEqual(ema5[0], 1.0, places=6)
        # ewm(span=5, adjust=False): alpha=2/(5+1)=1/3
        # y1 = 1; y2 = (1/3)*2 + (2/3)*1 = 4/3
        self.assertAlmostEqual(ema5[1], 4.0 / 3.0, places=6)
        ema10 = apply_ema(vals, 10)
        self.assertNotEqual(ema5[-1], ema10[-1])

    def test_build_bundle_ema_overlay(self) -> None:
        from chain_replay_ml.model_lab.strike_prediction_dashboard import (
            build_strike_chart_bundle,
        )

        cols = [
            "prediction_id",
            "timestamp",
            "current_ltp",
            "predicted_future_ltp",
            "pred_prob_up_2pct_5m",
        ]
        rows = [
            ("p1", 1.0, 100.0, 110.0, 0.4),
            ("p2", 2.0, 101.0, 112.0, 0.5),
            ("p3", 3.0, 102.0, 108.0, 0.6),
            ("p4", 4.0, 103.0, 115.0, 0.55),
            ("p5", 5.0, 104.0, 111.0, 0.7),
        ]
        bundle = build_strike_chart_bundle(cols, rows, ema_span=5)
        self.assertEqual(bundle["confidence_column"], "pred_prob_up_2pct_5m")
        self.assertEqual(bundle["error"][0], 10.0)
        self.assertEqual(len(bundle["predicted_ema"]), 5)
        self.assertEqual(len(bundle["confidence_ema"]), 5)
        self.assertEqual(len(bundle["confidence_ema_5"]), 5)
        self.assertEqual(len(bundle["confidence_ema_10"]), 5)
        self.assertEqual(len(bundle["gap"]), 5)
        self.assertEqual(len(bundle["regression_ema_slope"]), 5)
        self.assertIsNone(bundle["regression_ema_slope"][0])
        self.assertAlmostEqual(
            bundle["gap"][0],
            bundle["predicted_ema"][0] - 100.0,
            places=6,
        )
        self.assertFalse(bundle["has_trade_data"])
        self.assertEqual(bundle["position_size"], [0.0] * 5)
        self.assertTrue(all(v is None for v in bundle["pnl"]))
        self.assertIn("error_summary", bundle)
        self.assertEqual(bundle["error_summary"]["n"], 5)
        self.assertAlmostEqual(
            bundle["error_summary"]["mean_error"],
            sum(bundle["error"]) / 5,
            places=6,
        )
        self.assertIn("ltp_summary", bundle)
        self.assertEqual(bundle["ltp_summary"]["actual"]["n"], 5)
        self.assertAlmostEqual(bundle["ltp_summary"]["actual"]["min"], 100.0, places=6)
        self.assertAlmostEqual(bundle["ltp_summary"]["actual"]["max"], 104.0, places=6)
        self.assertAlmostEqual(
            bundle["ltp_summary"]["premium_range"]["spread"], 4.0, places=6
        )
        self.assertEqual(len(bundle["error_recent_rows"]), 5)
        self.assertIn("error_downsampled", bundle)
        recent0 = bundle["error_recent_rows"][0]
        self.assertEqual(recent0["current_ltp"], 100.0)
        self.assertEqual(recent0["future_actual"], 100.0)  # fallback: no actual_future
        self.assertEqual(recent0["future_pred"], 110.0)
        self.assertAlmostEqual(recent0["actual_delta"], 0.0, places=6)
        self.assertAlmostEqual(recent0["pred_delta"], 10.0, places=6)
        self.assertAlmostEqual(recent0["error"], 10.0, places=6)
        self.assertIn("actual_future_ltp", bundle)
        self.assertIn("future_actual_ltp", bundle)
        self.assertIn("current_ltp", bundle)

    def test_build_bundle_uses_actual_future_for_error(self) -> None:
        from chain_replay_ml.model_lab.strike_prediction_dashboard import (
            build_strike_chart_bundle,
        )

        cols = [
            "prediction_id",
            "timestamp",
            "current_ltp",
            "predicted_future_ltp",
            "actual_future_ltp",
        ]
        # Example: Current 305.40, Future Actual 309.00, Future Pred 308.59
        rows = [
            ("p1", 1.0, 305.40, 308.59, 309.00),
            ("p2", 2.0, 306.00, 310.00, 307.50),
        ]
        bundle = build_strike_chart_bundle(cols, rows, ema_span=5)
        self.assertAlmostEqual(bundle["error"][0], 308.59 - 309.00, places=6)
        self.assertAlmostEqual(bundle["error"][1], 310.00 - 307.50, places=6)
        # LTP chart still uses current LTP
        self.assertEqual(bundle["actual_ltp"][0], 305.40)
        row0 = bundle["error_recent_rows"][0]
        self.assertAlmostEqual(row0["current_ltp"], 305.40, places=6)
        self.assertAlmostEqual(row0["future_actual"], 309.00, places=6)
        self.assertAlmostEqual(row0["future_pred"], 308.59, places=6)
        self.assertAlmostEqual(row0["actual_delta"], 3.60, places=6)
        self.assertAlmostEqual(row0["pred_delta"], 3.19, places=6)
        self.assertAlmostEqual(row0["error"], -0.41, places=6)


class LtpVsPredictionStatsTests(unittest.TestCase):
    def test_series_ltp_stats_odd_even_and_none(self) -> None:
        from chain_replay_ml.model_lab.strike_prediction_dashboard import (
            series_ltp_stats,
        )

        odd = series_ltp_stats([10.0, 30.0, 20.0, None])
        self.assertEqual(odd["n"], 3)
        self.assertEqual(odd["min"], 10.0)
        self.assertEqual(odd["max"], 30.0)
        self.assertAlmostEqual(odd["mean"], 20.0, places=6)
        self.assertEqual(odd["median"], 20.0)

        even = series_ltp_stats([1.0, 2.0, 3.0, 4.0])
        self.assertEqual(even["n"], 4)
        self.assertAlmostEqual(even["median"], 2.5, places=6)
        self.assertAlmostEqual(even["mean"], 2.5, places=6)

        empty = series_ltp_stats([None, float("nan")])
        self.assertEqual(empty["n"], 0)
        self.assertIsNone(empty["min"])
        self.assertIsNone(empty["max"])
        self.assertIsNone(empty["mean"])
        self.assertIsNone(empty["median"])

    def test_ltp_vs_prediction_summary_premium_from_actual(self) -> None:
        from chain_replay_ml.model_lab.strike_prediction_dashboard import (
            ltp_vs_prediction_summary,
        )

        s = ltp_vs_prediction_summary(
            [100.0, 110.0, 90.0, None],
            [105.0, 120.0, None, 95.0],
        )
        self.assertEqual(s["actual"]["n"], 3)
        self.assertAlmostEqual(s["actual"]["min"], 90.0, places=6)
        self.assertAlmostEqual(s["actual"]["max"], 110.0, places=6)
        self.assertAlmostEqual(s["actual"]["mean"], 100.0, places=6)
        self.assertEqual(s["actual"]["median"], 100.0)

        self.assertEqual(s["predicted"]["n"], 3)
        self.assertAlmostEqual(s["predicted"]["min"], 95.0, places=6)
        self.assertAlmostEqual(s["predicted"]["max"], 120.0, places=6)

        # Premium range/spread come from actual LTP, not predicted.
        self.assertAlmostEqual(s["premium_range"]["min"], 90.0, places=6)
        self.assertAlmostEqual(s["premium_range"]["max"], 110.0, places=6)
        self.assertAlmostEqual(s["premium_range"]["spread"], 20.0, places=6)
        self.assertEqual(s["premium_range"]["n"], 3)

        empty = ltp_vs_prediction_summary([], [])
        self.assertEqual(empty["actual"]["n"], 0)
        self.assertIsNone(empty["premium_range"]["spread"])


class PredictionErrorStatsTests(unittest.TestCase):
    def test_mae_rmse(self) -> None:
        from chain_replay_ml.model_lab.strike_prediction_dashboard import (
            prediction_error_mae,
            prediction_error_rmse,
        )

        errs = [2.0, -2.0, 4.0, None]
        self.assertAlmostEqual(prediction_error_mae(errs), 8.0 / 3.0, places=6)
        self.assertAlmostEqual(
            prediction_error_rmse(errs), (24.0 / 3.0) ** 0.5, places=6
        )
        self.assertIsNone(prediction_error_mae([]))
        self.assertIsNone(prediction_error_rmse([None, None]))

    def test_quantiles(self) -> None:
        from chain_replay_ml.model_lab.strike_prediction_dashboard import (
            ERROR_QUANTILE_PROBS,
            prediction_error_quantiles,
        )

        self.assertEqual(len(ERROR_QUANTILE_PROBS), 99)
        self.assertAlmostEqual(ERROR_QUANTILE_PROBS[0], 0.01, places=9)
        self.assertAlmostEqual(ERROR_QUANTILE_PROBS[-1], 0.99, places=9)

        # Odd count: P50 is middle value after sort.
        q = prediction_error_quantiles([10.0, -5.0, 0.0, 2.0, 8.0])
        self.assertEqual(len(q), 99)
        self.assertEqual(list(q.keys()), [f"p{i:02d}" for i in range(1, 100)])
        self.assertAlmostEqual(q["p50"]["error"], 2.0, places=6)
        self.assertAlmostEqual(
            q["p10"]["error"], -5.0 + 0.4 * (0.0 - (-5.0)), places=6
        )
        self.assertAlmostEqual(
            q["p01"]["error"], -5.0 + 0.04 * (0.0 - (-5.0)), places=6
        )
        self.assertLess(q["p01"]["error"], q["p25"]["error"])
        self.assertLess(q["p25"]["error"], q["p50"]["error"])
        self.assertLess(q["p50"]["error"], q["p75"]["error"])
        self.assertLess(q["p75"]["error"], q["p90"]["error"])
        self.assertLess(q["p90"]["error"], q["p99"]["error"])
        # Samples: count in (P{i-1}, Pi]; values above P99 are outside the bins.
        vals = [-5.0, 0.0, 2.0, 8.0, 10.0]
        total_in_bins = sum(int(q[k]["samples"]) for k in q)
        above_p99 = sum(1 for v in vals if v > float(q["p99"]["error"]))
        self.assertEqual(total_in_bins + above_p99, len(vals))
        self.assertGreater(int(q["p01"]["samples"]), 0)
        empty = prediction_error_quantiles([None, None])
        self.assertEqual(len(empty), 99)
        self.assertIsNone(empty["p01"]["error"])
        self.assertIsNone(empty["p50"]["error"])
        self.assertIsNone(empty["p99"]["error"])
        self.assertEqual(empty["p01"]["samples"], 0)
        self.assertEqual(empty["p99"]["samples"], 0)

    def test_quantile_samples_bins(self) -> None:
        from chain_replay_ml.model_lab.strike_prediction_dashboard import (
            prediction_error_quantiles,
        )

        # 2400 evenly spaced errors → ~24 per percentile bin for large n.
        errs = [float(i) for i in range(2400)]
        q = prediction_error_quantiles(errs)
        samples = [int(q[f"p{i:02d}"]["samples"]) for i in range(1, 100)]
        self.assertEqual(sum(samples), 2376)  # all but the top ~1% above P99
        for s in samples:
            self.assertAlmostEqual(s, 24, delta=2)

    def test_trend_flags_expanding_optimistic(self) -> None:
        from chain_replay_ml.model_lab.strike_prediction_dashboard import (
            prediction_error_trend_flags,
        )

        # First half small |err|, second half large positive → expanding + optimistic.
        errs = [0.1, -0.1, 0.2, -0.2, 5.0, 6.0, 5.5, 6.5]
        flags = prediction_error_trend_flags(errs)
        self.assertEqual(flags["magnitude"], "expanding")
        self.assertEqual(flags["bias"], "more_optimistic")
        self.assertIn("expanding", flags["magnitude_label"].lower())

    def test_trend_flags_contracting_pessimistic(self) -> None:
        from chain_replay_ml.model_lab.strike_prediction_dashboard import (
            prediction_error_trend_flags,
        )

        errs = [5.0, 6.0, 5.5, 6.5, -0.1, 0.1, -0.2, 0.05]
        flags = prediction_error_trend_flags(errs)
        self.assertEqual(flags["magnitude"], "contracting")
        self.assertEqual(flags["bias"], "more_pessimistic")

    def test_trend_flags_recent_n(self) -> None:
        from chain_replay_ml.model_lab.strike_prediction_dashboard import (
            prediction_error_trend_flags,
        )

        errs = [0.0] * 10 + [10.0] * 10
        flags = prediction_error_trend_flags(errs, recent_n=5)
        self.assertEqual(flags["mode"], "recent_5")
        self.assertEqual(flags["n_later"], 5)
        # Both recent windows are all 10s → stable magnitude and bias.
        self.assertEqual(flags["magnitude"], "stable")
        self.assertEqual(flags["bias"], "stable")

        errs2 = [0.0] * 10 + [0.0] * 5 + [8.0] * 5
        flags2 = prediction_error_trend_flags(errs2, recent_n=5)
        self.assertEqual(flags2["magnitude"], "expanding")
        self.assertEqual(flags2["bias"], "more_optimistic")

    def test_summary_optimistic_pct_and_latest(self) -> None:
        from chain_replay_ml.model_lab.strike_prediction_dashboard import (
            prediction_error_summary,
        )

        errs = [2.0, -1.0, 3.0, 0.0]
        ema = [2.0, 1.0, 1.5, 2.0]
        s = prediction_error_summary(errs, ema)
        self.assertEqual(s["n"], 4)
        self.assertAlmostEqual(s["pct_optimistic"], 50.0, places=6)  # 2 of 4 > 0
        self.assertAlmostEqual(s["pct_pessimistic"], 25.0, places=6)  # 1 of 4 < 0
        self.assertEqual(s["latest_error"], 0.0)
        self.assertEqual(s["latest_error_ema"], 2.0)
        self.assertEqual(len(s["quantiles"]), 99)
        self.assertIn("p01", s["quantiles"])
        self.assertIn("p50", s["quantiles"])
        self.assertIn("p99", s["quantiles"])
        self.assertIn("error", s["quantiles"]["p50"])
        self.assertIn("samples", s["quantiles"]["p50"])
        self.assertIn("magnitude", s["trends"])

    def test_downsample_bucket_mean(self) -> None:
        from chain_replay_ml.model_lab.strike_prediction_dashboard import (
            downsample_series,
        )

        vals = list(range(1000))
        ds = downsample_series(vals, target=100, timestamps=list(range(1000)))
        self.assertEqual(ds["n_downsampled"], 100)
        self.assertEqual(ds["n_source"], 1000)
        self.assertEqual(len(ds["source_indices"]), 100)
        self.assertEqual(len(ds["values"]), 100)
        # First bucket mean of 0..9
        self.assertAlmostEqual(ds["values"][0], 4.5, places=6)

    def test_recent_rows(self) -> None:
        from chain_replay_ml.model_lab.strike_prediction_dashboard import (
            recent_prediction_error_rows,
        )

        rows = recent_prediction_error_rows(
            [1, 2, 3, 4, 5],
            [100.0, 101.0, 102.0, 103.0, 104.0],  # current
            [103.0, 104.0, 105.0, 106.0, 107.0],  # future actual
            [102.5, 103.5, 104.5, 105.5, 106.5],  # future pred
            [-0.5, -0.5, -0.5, -0.5, -0.5],  # error = pred − actual
            [-0.5, -0.5, -0.5, -0.5, -0.5],
            confidence=[0.4, 0.5, 0.6, 0.7, 0.8],
            confidence_ema=[0.4, 0.45, 0.5, 0.55, 0.6],
            n=3,
        )
        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[0]["index"], 2)
        self.assertEqual(rows[-1]["timestamp"], 5)
        self.assertEqual(rows[0]["current_ltp"], 102.0)
        self.assertEqual(rows[0]["future_actual"], 105.0)
        self.assertEqual(rows[0]["future_pred"], 104.5)
        self.assertAlmostEqual(rows[0]["actual_delta"], 3.0, places=6)
        self.assertAlmostEqual(rows[0]["pred_delta"], 2.5, places=6)
        self.assertAlmostEqual(rows[0]["error"], -0.5, places=6)
        self.assertAlmostEqual(rows[0]["confidence"], 0.6, places=6)
        self.assertAlmostEqual(rows[0]["confidence_ema"], 0.5, places=6)
        self.assertAlmostEqual(rows[-1]["confidence"], 0.8, places=6)
        self.assertAlmostEqual(rows[-1]["confidence_ema"], 0.6, places=6)

    def test_resolve_future_actual_fallback(self) -> None:
        from chain_replay_ml.model_lab.strike_prediction_dashboard import (
            resolve_future_actual_series,
        )

        current = [100.0, 101.0, 102.0]
        self.assertEqual(
            resolve_future_actual_series(current, [None, None, None]),
            current,
        )
        self.assertEqual(
            resolve_future_actual_series(current, None),
            current,
        )
        resolved = resolve_future_actual_series(current, [110.0, None, 112.0])
        self.assertEqual(resolved[0], 110.0)
        self.assertIsNone(resolved[1])
        self.assertEqual(resolved[2], 112.0)


class CrosshairLookupTests(unittest.TestCase):
    def test_series_index_from_x(self) -> None:
        from chain_replay_ml.model_lab.strike_prediction_dashboard import (
            series_index_from_x,
        )

        self.assertEqual(
            series_index_from_x(28.0, pad=28.0, inner_w=100.0, n_points=5), 0
        )
        self.assertEqual(
            series_index_from_x(128.0, pad=28.0, inner_w=100.0, n_points=5), 4
        )
        self.assertEqual(
            series_index_from_x(78.0, pad=28.0, inner_w=100.0, n_points=5), 2
        )
        self.assertIsNone(
            series_index_from_x(50.0, pad=28.0, inner_w=100.0, n_points=0)
        )

    def test_index_for_timestamp(self) -> None:
        from chain_replay_ml.model_lab.strike_prediction_dashboard import (
            index_for_timestamp,
        )

        ts = [1.0, 2.0, 3.0, 4.0]
        self.assertEqual(index_for_timestamp(ts, 2.0), 1)
        self.assertEqual(index_for_timestamp(ts, 2.4), 1)
        self.assertEqual(index_for_timestamp(ts, 2.6), 2)
        self.assertIsNone(index_for_timestamp([], 1.0))

    def test_crosshair_detail_at_index(self) -> None:
        from chain_replay_ml.model_lab.strike_prediction_dashboard import (
            build_strike_chart_bundle,
            crosshair_detail_at_index,
        )

        cols = [
            "prediction_id",
            "timestamp",
            "current_ltp",
            "predicted_future_ltp",
            "pred_prob_up_2pct_5m",
        ]
        rows = [
            ("p1", 10.0, 100.0, 110.0, 0.4),
            ("p2", 20.0, 101.0, 112.0, 0.5),
            ("p3", 30.0, 102.0, 108.0, 0.6),
        ]
        bundle = build_strike_chart_bundle(cols, rows, ema_span=5)
        detail = crosshair_detail_at_index(bundle, 1)
        self.assertEqual(detail["index"], 1)
        self.assertEqual(detail["timestamp"], 20.0)
        self.assertEqual(detail["actual_ltp"], 101.0)
        self.assertEqual(detail["predicted_ltp"], 112.0)
        self.assertIsNotNone(detail["predicted_ema"])
        self.assertIsNotNone(detail["gap"])
        self.assertIsNotNone(detail["confidence_ema_5"])
        self.assertIsNotNone(detail["confidence_ema_10"])


class TradeAlignTests(unittest.TestCase):
    def test_no_trades_honest_empty(self) -> None:
        from chain_replay_ml.model_lab.strike_prediction_dashboard import (
            align_trades_to_timestamps,
        )

        sizes, pnl, has = align_trades_to_timestamps(
            [1.0, 2.0, 3.0],
            ["a", "b", "c"],
            [],
            strike=24500.0,
            option_type="CE",
        )
        self.assertFalse(has)
        self.assertEqual(sizes, [0.0, 0.0, 0.0])
        self.assertEqual(pnl, [None, None, None])

    def test_join_by_entry_prediction_id(self) -> None:
        from chain_replay_ml.model_lab.strike_prediction_dashboard import (
            align_trades_to_timestamps,
        )

        trades = [
            {
                "entry_prediction_id": "b",
                "entry_ts": 2.0,
                "exit_ts": 3.0,
                "strike": 24500.0,
                "option_type": "CE",
                "trading_day": "2024-01-02",
                "qty": 65,
                "net_pnl": 150.0,
                "lots": 2,
            }
        ]
        sizes, pnl, has = align_trades_to_timestamps(
            [1.0, 2.0, 3.0, 4.0],
            ["a", "b", "c", "d"],
            trades,
            strike=24500.0,
            option_type="CE",
            trading_day="2024-01-02",
        )
        self.assertTrue(has)
        self.assertEqual(sizes[0], 0.0)
        self.assertEqual(sizes[1], 2.0)
        self.assertEqual(sizes[2], 2.0)
        self.assertEqual(sizes[3], 0.0)
        self.assertIsNone(pnl[0])
        self.assertIsNone(pnl[1])
        self.assertEqual(pnl[2], 150.0)
        self.assertEqual(pnl[3], 150.0)


class PanelSmokeTests(unittest.TestCase):
    def test_panel_exposes_lab_refresh(self) -> None:
        from master_dataset_tk.strike_prediction_dashboard_panel import (
            ModelLabStrikeDashboardPanel,
        )

        self.assertTrue(hasattr(ModelLabStrikeDashboardPanel, "refresh_for_lab"))

    def test_panel_builds_chart_notebook_tabs(self) -> None:
        import tkinter as tk

        from master_dataset_tk.strike_prediction_dashboard_panel import (
            TAB_SPECS,
            ModelLabStrikeDashboardPanel,
        )

        root = tk.Tk()
        root.withdraw()
        try:
            panel = ModelLabStrikeDashboardPanel(root, chart_dir=".")
            nb = panel._chart_notebook
            tabs = [nb.tab(i, "text") for i in range(nb.index("end"))]
            expected = [title for _key, title in TAB_SPECS]
            self.assertEqual(tabs, expected)
            self.assertEqual(
                set(panel._canvases),
                {"ltp", "conf", "err", "gap", "regr", "conf_pred"},
            )
            self.assertTrue(hasattr(panel, "_cursor_index"))
            self.assertTrue(hasattr(panel, "_cursor_ts"))
            self.assertTrue(hasattr(panel, "_detail_var"))
            self.assertTrue(hasattr(panel, "_err_summary_vars"))
            self.assertTrue(hasattr(panel, "_ltp_actual_vars"))
            self.assertTrue(hasattr(panel, "_ltp_predicted_vars"))
            self.assertTrue(hasattr(panel, "_ltp_premium_vars"))
            self.assertTrue(hasattr(panel, "_ltp_strip_actual_var"))
            self.assertTrue(hasattr(panel, "_ltp_strip_predicted_var"))
            self.assertTrue(hasattr(panel, "_ltp_strip_premium_var"))
            self.assertTrue(hasattr(panel, "_err_q_tree"))
            self.assertTrue(hasattr(panel, "_err_recent_tree"))
            self.assertFalse(panel._show_err_chart_var.get())
            build_src = inspect.getsource(ModelLabStrikeDashboardPanel._build_ui)
            self.assertIn("ttk.Notebook", build_src)
            self.assertIn("_build_global_ltp_strip", build_src)
            self.assertNotIn("grid.rowconfigure", build_src)
            strip_src = inspect.getsource(
                ModelLabStrikeDashboardPanel._build_global_ltp_strip
            )
            self.assertIn("Actual LTP", strip_src)
            self.assertIn("Predicted LTP", strip_src)
            self.assertIn("Premium Range", strip_src)
            ltp_src = inspect.getsource(ModelLabStrikeDashboardPanel._build_ltp_tab)
            self.assertIn("Actual LTP", ltp_src)
            self.assertIn("Predicted LTP", ltp_src)
            self.assertIn("Premium Range", ltp_src)
            self.assertIn('uniform="ltp_sum"', ltp_src)
            err_src = inspect.getsource(ModelLabStrikeDashboardPanel._build_error_tab)
            self.assertIn("Show chart (downsampled)", err_src)
            self.assertIn("Error summary", err_src)
            self.assertIn("Error Quantiles", err_src)
            self.assertIn("Recent Samples", err_src)
            self.assertTrue(hasattr(panel, "_err_sub_notebook"))
            self.assertEqual(
                tuple(panel._err_q_tree["columns"]), ("q", "v", "n")
            )
            self.assertEqual(panel._err_q_tree.heading("n", "text"), "Samples")
            self.assertEqual(
                tuple(panel._err_recent_tree["columns"]),
                (
                    "ts",
                    "cur",
                    "fact",
                    "fpred",
                    "adelta",
                    "pdelta",
                    "err",
                    "ema",
                    "conf",
                    "conf_ema",
                ),
            )
            self.assertEqual(
                panel._err_recent_tree.heading("cur", "text"), "Current LTP"
            )
            self.assertEqual(
                panel._err_recent_tree.heading("adelta", "text"), "Actual Δ₹"
            )
            self.assertEqual(
                panel._err_recent_tree.heading("pdelta", "text"), "Pred Δ₹"
            )
            self.assertEqual(
                panel._err_recent_tree.heading("conf", "text"), "Conf"
            )
            self.assertEqual(
                panel._err_recent_tree.heading("conf_ema", "text"), "Conf EMA"
            )
            self.assertEqual(panel._signed_rupee(3.6), "+3.60")
            self.assertEqual(panel._signed_rupee(-0.41), "−0.41")
            self.assertEqual(panel._signed_rupee(0.0), "0.00")
            panel._last_bundle = {
                "error_summary": {
                    "n": 1,
                    "mean_error": -0.41,
                    "mae": 0.41,
                    "rmse": 0.41,
                    "pct_optimistic": 0.0,
                    "pct_pessimistic": 100.0,
                    "std_error": 0.0,
                    "latest_error": -0.41,
                    "latest_error_ema": -0.41,
                    "quantiles": {},
                    "trends": {},
                },
                "ema_span": 5,
                "error_recent_rows": [
                    {
                        "index": 0,
                        "timestamp": 1.0,
                        "current_ltp": 305.40,
                        "future_actual": 309.00,
                        "future_pred": 308.59,
                        "actual_delta": 3.60,
                        "pred_delta": 3.19,
                        "error": -0.41,
                        "error_ema": -0.41,
                        "confidence": 0.55,
                        "confidence_ema": 0.55,
                    }
                ],
                "error_downsampled": {"n_source": 1, "n_downsampled": 1},
                "ltp_summary": {
                    "actual": {
                        "n": 2,
                        "min": 10.0,
                        "max": 20.0,
                        "mean": 15.0,
                        "median": 15.0,
                    },
                    "predicted": {
                        "n": 2,
                        "min": 11.0,
                        "max": 22.0,
                        "mean": 16.5,
                        "median": 16.5,
                    },
                    "premium_range": {
                        "min": 10.0,
                        "max": 20.0,
                        "spread": 10.0,
                        "n": 2,
                    },
                },
            }
            panel._refresh_error_stats_ui()
            kids = panel._err_recent_tree.get_children()
            self.assertEqual(len(kids), 1)
            vals = panel._err_recent_tree.item(kids[0], "values")
            self.assertEqual(vals[1], "305.40")
            self.assertEqual(vals[2], "309.00")
            self.assertEqual(vals[3], "308.59")
            self.assertEqual(vals[4], "+3.60")
            self.assertEqual(vals[5], "+3.19")
            self.assertEqual(vals[6], "−0.41")
            self.assertEqual(vals[8], "0.550")
            self.assertEqual(vals[9], "0.550")
            panel._last_bundle = {
                "ltp_summary": {
                    "actual": {
                        "n": 2,
                        "min": 10.0,
                        "max": 20.0,
                        "mean": 15.0,
                        "median": 15.0,
                    },
                    "predicted": {
                        "n": 2,
                        "min": 11.0,
                        "max": 22.0,
                        "mean": 16.5,
                        "median": 16.5,
                    },
                    "premium_range": {
                        "min": 10.0,
                        "max": 20.0,
                        "spread": 10.0,
                        "n": 2,
                    },
                }
            }
            panel._refresh_ltp_stats_ui()
            self.assertEqual(panel._ltp_actual_vars["min"].get(), "Min: 10.00")
            self.assertEqual(panel._ltp_actual_vars["max"].get(), "Max: 20.00")
            self.assertIn("₹10.00 → ₹20.00", panel._ltp_premium_vars["range"].get())
            self.assertEqual(panel._ltp_premium_vars["spread"].get(), "Spread: ₹10.00")
            self.assertIn("Min 10.00", panel._ltp_strip_actual_var.get())
            self.assertIn("Mean 16.50", panel._ltp_strip_predicted_var.get())
            self.assertIn("₹10.00 → ₹20.00", panel._ltp_strip_premium_var.get())
            self.assertIn("Spread ₹10.00", panel._ltp_strip_premium_var.get())
        finally:
            root.destroy()

    def test_model_lab_window_wires_strike_tab(self) -> None:
        from master_dataset_tk.model_lab_window import ModelLabWindow

        self.assertTrue(hasattr(ModelLabWindow, "_build_strike_dashboard_tab"))
        self.assertTrue(hasattr(ModelLabWindow, "_refresh_strike_dashboard_tab"))
        self.assertTrue(hasattr(ModelLabWindow, "select_strike_dashboard_tab"))
        src = inspect.getsource(ModelLabWindow._build_strike_dashboard_tab)
        self.assertIn("ModelLabStrikeDashboardPanel", src)
        reset_src = inspect.getsource(ModelLabWindow._reset_tabs)
        self.assertIn("Strike Dashboard", reset_src)

    def test_draw_line_chart_multi_series_import(self) -> None:
        from master_dataset_tk.fold_replay_widgets import draw_line_chart

        self.assertTrue(callable(draw_line_chart))
        sig = inspect.signature(draw_line_chart)
        self.assertIn("series", sig.parameters)
        self.assertIn("secondary_series", sig.parameters)
        self.assertIn("cursor_index", sig.parameters)


class EmaFiniteCheck(unittest.TestCase):
    def test_ema_handles_none(self) -> None:
        from chain_replay_ml.model_lab.strike_prediction_dashboard import apply_ema

        out = apply_ema([1.0, None, 3.0], 5)
        self.assertEqual(len(out), 3)
        self.assertTrue(out[0] is not None and math.isfinite(out[0]))


if __name__ == "__main__":
    unittest.main()

"""Premium-normalized regression metrics + band performance."""

from __future__ import annotations

import math
import unittest

import numpy as np
import pandas as pd

from chain_replay_ml.training.evaluator import (
    PREMIUM_METRIC_BANDS,
    aggregate_premium_band_performance,
    evaluate_regression,
    premium_band_performance,
    premium_mae_pct,
    premium_rmse_pct,
    resolve_ltp_baseline,
    resolve_ltp_baseline_from_frames,
)
from chain_replay_ml.training.registry import _build_production_metrics, _resolve_authoritative_metrics
from chain_replay_ml.training.walk_forward_runner import (
    aggregate_fold_metrics,
    validation_metrics_from_wf_aggregate,
)


class PremiumMetricFormulasTests(unittest.TestCase):
    def test_premium_mae_rmse_formulas(self) -> None:
        y_true = np.array([10.0, 20.0, 50.0], dtype=float)
        y_pred = np.array([11.0, 18.0, 55.0], dtype=float)
        # |1|/10 + |2|/20 + |5|/50 = 0.1 + 0.1 + 0.1 → mean 0.1 → 10%
        self.assertAlmostEqual(premium_mae_pct(y_true, y_pred), 10.0, places=6)
        # ((0.1)^2 + (-0.1)^2 + (0.1)^2) / 3 = 0.01 → sqrt = 0.1 → 10%
        self.assertAlmostEqual(premium_rmse_pct(y_true, y_pred), 10.0, places=6)

    def test_evaluate_regression_includes_premium_fields(self) -> None:
        y_true = np.array([8.0, 25.0, 40.0, 80.0, 150.0, 250.0], dtype=float)
        y_pred = np.array([9.0, 22.0, 44.0, 70.0, 160.0, 240.0], dtype=float)
        baseline = np.array([7.0, 24.0, 38.0, 85.0, 140.0, 260.0], dtype=float)
        metrics = evaluate_regression(y_true, y_pred, baseline=baseline)
        self.assertIn("premium_mae_pct", metrics)
        self.assertIn("premium_rmse_pct", metrics)
        self.assertIsInstance(metrics["premium_mae_pct"], float)
        self.assertIsInstance(metrics["premium_rmse_pct"], float)
        bands = metrics["premium_band_performance"]
        self.assertEqual(len(bands), len(PREMIUM_METRIC_BANDS))
        labels = [b["band"] for b in bands]
        self.assertEqual(labels, [b[0] for b in PREMIUM_METRIC_BANDS])
        filled = [b for b in bands if b["samples"] > 0]
        self.assertGreaterEqual(len(filled), 4)
        for b in filled:
            self.assertIsNotNone(b["mae"])
            self.assertIsNotNone(b["premium_mae_pct"])

    def test_band_boundaries(self) -> None:
        y_true = np.array([0.0, 14.9, 15.0, 29.9, 30.0, 49.9, 50.0, 99.9, 100.0, 199.9, 200.0], dtype=float)
        y_pred = y_true + 1.0
        bands = {b["band"]: b["samples"] for b in premium_band_performance(y_true, y_pred)}
        self.assertEqual(bands["0-15"], 2)
        self.assertEqual(bands["15-30"], 2)
        self.assertEqual(bands["30-50"], 2)
        self.assertEqual(bands["50-100"], 2)
        self.assertEqual(bands["100-200"], 2)
        self.assertEqual(bands["200+"], 1)

    def test_aggregate_band_performance_weighted(self) -> None:
        fold_a = premium_band_performance(np.array([10.0, 10.0]), np.array([12.0, 8.0]))
        fold_b = premium_band_performance(np.array([20.0]), np.array([22.0]))
        merged = aggregate_premium_band_performance([fold_a, fold_b])
        by_band = {r["band"]: r for r in merged}
        self.assertEqual(by_band["0-15"]["samples"], 2)
        self.assertEqual(by_band["15-30"]["samples"], 1)
        self.assertAlmostEqual(by_band["0-15"]["mae"], 2.0, places=6)

    def test_walk_forward_aggregate_propagates_premium(self) -> None:
        m1 = evaluate_regression(np.array([10.0, 25.0]), np.array([11.0, 23.0]))
        m2 = evaluate_regression(np.array([40.0, 80.0]), np.array([44.0, 70.0]))
        folds = [
            {"metrics": {"rmse": m1["rmse"], "mae": m1["mae"], "premium_mae_pct": m1["premium_mae_pct"],
                         "premium_rmse_pct": m1["premium_rmse_pct"],
                         "premium_band_performance": m1["premium_band_performance"],
                         "directional_accuracy_pct": 70.0}},
            {"metrics": {"rmse": m2["rmse"], "mae": m2["mae"], "premium_mae_pct": m2["premium_mae_pct"],
                         "premium_rmse_pct": m2["premium_rmse_pct"],
                         "premium_band_performance": m2["premium_band_performance"],
                         "directional_accuracy_pct": 80.0}},
        ]
        agg = aggregate_fold_metrics(folds)
        self.assertIsNotNone(agg.get("mean_premium_mae_pct"))
        self.assertIsNotNone(agg.get("mean_premium_rmse_pct"))
        self.assertTrue(agg.get("premium_band_performance"))
        flat = validation_metrics_from_wf_aggregate(agg)
        self.assertEqual(flat["premium_mae_pct"], agg["mean_premium_mae_pct"])
        self.assertTrue(flat["premium_band_performance"])

    def test_risk_metrics_medae_p95_bias(self) -> None:
        y_true = np.array([10.0, 20.0, 30.0, 40.0, 50.0], dtype=float)
        y_pred = np.array([11.0, 18.0, 33.0, 40.0, 55.0], dtype=float)
        # abs errors: 1, 2, 3, 0, 5 → median 2, p95 near 5, bias mean = (1-2+3+0+5)/5 = 1.4
        metrics = evaluate_regression(y_true, y_pred)
        self.assertAlmostEqual(metrics["medae"], 2.0, places=6)
        self.assertAlmostEqual(metrics["median_error"], 2.0, places=6)
        self.assertAlmostEqual(metrics["p95_error"], 4.6, places=5)  # np.percentile linear for n=5
        self.assertAlmostEqual(metrics["prediction_bias"], 1.4, places=6)
        # relative: 0.1 + (-0.1) + 0.1 + 0.0 + 0.1 = 0.2 → mean 0.04 → 4%
        self.assertAlmostEqual(metrics["prediction_bias_pct"], 4.0, places=6)

    def test_production_metrics_includes_risk_fields(self) -> None:
        prod = _build_production_metrics(
            stage_key="test",
            stage_label="Test Metrics",
            source_file="metrics.json",
            source_path="$.test",
            raw_metrics={
                "mae": 2.5,
                "rmse": 3.1,
                "directional_accuracy_pct": 78.0,
                "composite_score": 0.45,
                "medae": 1.8,
                "p95_error": 4.8,
                "prediction_bias": 0.18,
                "prediction_bias_pct": 0.42,
                "premium_mae_pct": 12.5,
                "premium_rmse_pct": 15.0,
                "premium_band_performance": [{"band": "0-15", "samples": 10, "mae": 1.0}],
            },
        )
        self.assertEqual(prod["premium_mae_pct"], 12.5)
        self.assertEqual(prod["premium_rmse_pct"], 15.0)
        self.assertEqual(len(prod["premium_band_performance"]), 1)
        self.assertEqual(prod["medae"], 1.8)
        self.assertEqual(prod["p95_error"], 4.8)
        self.assertEqual(prod["prediction_bias"], 0.18)
        self.assertEqual(prod["prediction_bias_pct"], 0.42)


    def test_resolve_authoritative_from_production_wf(self) -> None:
        metrics_doc = {
            "production_walk_forward": {
                "n_folds": 3,
                "mean_mae": 2.0,
                "mean_rmse": 3.0,
                "mean_directional_accuracy_pct": 75.0,
                "mean_premium_mae_pct": 11.0,
                "mean_premium_rmse_pct": 14.0,
                "mean_composite_score": 0.5,
                "premium_band_performance": [{"band": "15-30", "samples": 5, "mae": 2.1}],
            },
            "composite_scores": {"production_composite": {"score": 0.5, "source_file": "metrics.json"}},
        }
        resolved = _resolve_authoritative_metrics(
            strategy={"key": "walk_forward", "label": "Walk Forward"},
            metrics_doc=metrics_doc,
            summary_doc={},
            wf_summary_doc=None,
        )
        self.assertEqual(resolved["premium_mae_pct"], 11.0)
        self.assertEqual(resolved["premium_rmse_pct"], 14.0)
        self.assertEqual(resolved["premium_band_performance"][0]["band"], "15-30")

    def test_zero_actual_skipped_in_premium(self) -> None:
        y_true = np.array([0.0, 10.0], dtype=float)
        y_pred = np.array([1.0, 11.0], dtype=float)
        mae = premium_mae_pct(y_true, y_pred)
        self.assertAlmostEqual(mae, 10.0, places=6)
        self.assertFalse(math.isnan(mae))


class ResolveLtpBaselineTests(unittest.TestCase):
    def test_prefers_explicit_ltp(self) -> None:
        df = pd.DataFrame({"ltp": [100.0, 200.0], "spot": [25000.0, 25100.0], "ltp_to_spot_ratio": [0.004, 0.008]})
        baseline = resolve_ltp_baseline(df)
        self.assertIsNotNone(baseline)
        self.assertAlmostEqual(float(baseline.iloc[0]), 100.0)

    def test_derives_from_ratio_and_spot(self) -> None:
        df = pd.DataFrame({"spot": [25000.0, 25100.0], "ltp_to_spot_ratio": [0.004, 0.008]})
        baseline = resolve_ltp_baseline(df)
        self.assertIsNotNone(baseline)
        self.assertAlmostEqual(float(baseline.iloc[0]), 100.0)
        self.assertAlmostEqual(float(baseline.iloc[1]), 200.8)

    def test_directional_accuracy_with_derived_ltp(self) -> None:
        df = pd.DataFrame({"spot": [25000.0], "ltp_to_spot_ratio": [0.004]})
        baseline = resolve_ltp_baseline(df)
        y_true = np.array([105.0])
        y_pred = np.array([110.0])
        metrics = evaluate_regression(y_true, y_pred, baseline=baseline)
        self.assertEqual(metrics["directional_accuracy_pct"], 100.0)

    def test_returns_none_without_sources(self) -> None:
        self.assertIsNone(resolve_ltp_baseline(pd.DataFrame({"spot": [25000.0]})))
        self.assertIsNone(resolve_ltp_baseline(None))

    def test_merges_ratio_from_features_with_spot_from_context(self) -> None:
        features = pd.DataFrame({"ltp_to_spot_ratio": [0.004, 0.008]})
        context = pd.DataFrame({"spot": [25000.0, 25100.0]})
        baseline = resolve_ltp_baseline_from_frames(features, context)
        self.assertIsNotNone(baseline)
        self.assertAlmostEqual(float(baseline.iloc[0]), 100.0)
        self.assertAlmostEqual(float(baseline.iloc[1]), 200.8)

    def test_mismatched_index_frames_align_by_position(self) -> None:
        n = 1000
        features = pd.DataFrame(
            {"ltp_to_spot_ratio": np.linspace(0.004, 0.008, n)},
            index=range(500_000, 500_000 + n),
        )
        context = pd.DataFrame(
            {"spot": np.full(n, 25000.0)},
            index=range(n),
        )
        baseline = resolve_ltp_baseline_from_frames(features, context)
        self.assertIsNotNone(baseline)
        self.assertEqual(len(baseline), n)

        y_true = np.linspace(100.0, 110.0, n)
        y_pred = y_true + 1.0
        metrics = evaluate_regression(y_true, y_pred, baseline=baseline)
        self.assertIn("directional_accuracy_pct", metrics)
        self.assertEqual(len(metrics["premium_band_performance"]), len(PREMIUM_METRIC_BANDS))

    def test_unequal_length_frames_use_shortest(self) -> None:
        features = pd.DataFrame({"ltp": [100.0, 200.0, 300.0]})
        context = pd.DataFrame({"spot": [25000.0, 25100.0]})
        baseline = resolve_ltp_baseline_from_frames(features, context)
        self.assertIsNotNone(baseline)
        self.assertEqual(len(baseline), 2)
        self.assertAlmostEqual(float(baseline.iloc[0]), 100.0)
        self.assertAlmostEqual(float(baseline.iloc[1]), 200.0)

    def test_fold_directional_accuracy_uses_context_ltp_not_features(self) -> None:
        """LTP can be absent from model features but present in context for DA."""
        from chain_replay_ml.training.walk_forward_runner import evaluate_walk_forward_folds

        rng = np.random.default_rng(7)
        n = 300
        X = pd.DataFrame({"f1": rng.normal(size=n), "f2": rng.normal(size=n)})
        ltp = np.abs(rng.normal(100, 15, size=n))
        y = pd.Series(ltp + rng.normal(0, 1.5, size=n))
        ctx = pd.DataFrame({"ltp": ltp})
        fold_defs = [{
            "fold": 1,
            "train": {"start": 0, "stop": 180},
            "validation": {"start": 180, "stop": 240},
        }]
        params = {
            "n_estimators": 20,
            "max_depth": 3,
            "learning_rate": 0.1,
            "early_stopping_rounds": 5,
        }
        without = evaluate_walk_forward_folds(
            X=X, y=y, features=["f1", "f2"], parameters=params,
            fold_defs=fold_defs, compute_shap=False, algorithm="xgboost",
        )
        self.assertIsNone(without["fold_results"][0]["metrics"].get("directional_accuracy_pct"))

        with_ctx = evaluate_walk_forward_folds(
            X=X, y=y, features=["f1", "f2"], parameters=params,
            fold_defs=fold_defs, compute_shap=False, algorithm="xgboost",
            context_df=ctx,
        )
        da = with_ctx["fold_results"][0]["metrics"].get("directional_accuracy_pct")
        self.assertIsInstance(da, float)
        self.assertGreaterEqual(da, 0.0)
        self.assertLessEqual(da, 100.0)


if __name__ == "__main__":
    unittest.main()

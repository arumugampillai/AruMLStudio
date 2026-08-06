"""Holdout performance analysis — unit tests."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from chain_replay_ml.training.holdout_performance import (
    build_holdout_overview_csv,
    build_holdout_performance_csv,
    build_holdout_performance_report,
    build_region_comparison_rows,
    build_feature_drift_ranking,
    build_prediction_error_change_row,
    composite_drift_risk_components,
    composite_drift_risk_label,
    composite_drift_risk_score,
    compute_drift_scores,
    compute_similarity_score,
    diagnose_degradation,
    distribution_summary,
    extract_saved_prediction_metrics,
    feature_drift_risk,
    feature_drift_score,
    ks_drift_metrics,
    normalized_mean_shift,
    premium_band_pct,
    resolve_holdout_slice,
    series_null_pct,
    wasserstein_drift_metrics,
)


class DistributionSummaryTests(unittest.TestCase):
    def test_distribution_summary_basic(self) -> None:
        s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        stats = distribution_summary(s)
        self.assertEqual(stats["count"], 5)
        self.assertAlmostEqual(float(stats["mean"]), 3.0, places=6)
        self.assertAlmostEqual(float(stats["p50"]), 3.0, places=6)

    def test_normalized_mean_shift(self) -> None:
        self.assertAlmostEqual(normalized_mean_shift(10.0, 12.0, 2.0), 1.0, places=6)
        self.assertIsNone(normalized_mean_shift(None, 12.0, 2.0))


class PremiumBandPctTests(unittest.TestCase):
    def test_premium_band_pct(self) -> None:
        baseline = np.array([10.0, 20.0, 40.0, 80.0, 150.0, 250.0], dtype=float)
        bands = premium_band_pct(baseline)
        self.assertAlmostEqual(bands["0-15"], 1 / 6 * 100.0, places=4)
        self.assertAlmostEqual(bands["15-30"], 1 / 6 * 100.0, places=4)
        self.assertGreater(bands["200+"], 0)


class RegionComparisonTests(unittest.TestCase):
    def test_build_region_comparison_rows(self) -> None:
        n = 100
        wf = pd.DataFrame({
            "target": np.linspace(0, 1, n),
            "feat_a": np.random.default_rng(1).normal(0, 1, n),
            "ltp": np.full(n, 25.0),
            "trading_day": ["2024-01-01"] * 50 + ["2024-01-02"] * 50,
        })
        ho = pd.DataFrame({
            "target": np.linspace(0, 2, 40),
            "feat_a": np.random.default_rng(2).normal(0.5, 1, 40),
            "ltp": np.full(40, 30.0),
            "trading_day": ["2024-01-03"] * 40,
        })
        rows = build_region_comparison_rows(
            target_wf=wf["target"],
            target_holdout=ho["target"],
            baseline_wf=wf["ltp"],
            baseline_holdout=ho["ltp"],
            vol_wf=None,
            vol_holdout=None,
            trading_day_wf=wf["trading_day"],
            trading_day_holdout=ho["trading_day"],
            feature_frames=[("feat_a", wf["feat_a"], ho["feat_a"])],
        )
        cats = {r["category"] for r in rows}
        self.assertIn("Target distribution", cats)
        self.assertIn("Premium bands", cats)
        self.assertIn("Trading days", cats)
        mean_row = next(r for r in rows if r["category"] == "Target distribution" and r["metric"] == "Mean")
        self.assertIsNotNone(mean_row["wf"])
        self.assertIsNotNone(mean_row["holdout"])


class DriftScoreTests(unittest.TestCase):
    def test_compute_drift_scores_and_similarity(self) -> None:
        wf = pd.Series(np.linspace(0, 1, 100))
        ho = pd.Series(np.linspace(0, 2, 40))
        ranking = [{"feature": "f1", "drift": 0.6}, {"feature": "f2", "drift": 0.4}]
        scores = compute_drift_scores(
            target_wf=wf,
            target_holdout=ho,
            baseline_wf=pd.Series(np.full(100, 25.0)),
            baseline_holdout=pd.Series(np.full(40, 30.0)),
            vol_wf=pd.Series(np.random.default_rng(1).normal(0.2, 0.05, 100)),
            vol_holdout=pd.Series(np.random.default_rng(2).normal(0.25, 0.08, 40)),
            feature_ranking=ranking,
        )
        self.assertIn("target", scores)
        self.assertIn("feature", scores)
        sim = compute_similarity_score(scores)
        self.assertGreaterEqual(sim, 0.0)
        self.assertLessEqual(sim, 100.0)

    def test_feature_drift_score_detects_shift(self) -> None:
        wf = pd.Series(np.random.default_rng(1).normal(0, 1, 200))
        ho = pd.Series(np.random.default_rng(2).normal(1.5, 1, 80))
        score = feature_drift_score(wf, ho)
        self.assertGreater(score, 0.2)

    def test_feature_drift_ranking_includes_means_and_risk(self) -> None:
        wf_df = pd.DataFrame({
            "f_high": np.random.default_rng(1).normal(0.18, 0.04, 50),
            "f_low": np.random.default_rng(3).normal(1.0, 0.1, 50),
        })
        ho_df = pd.DataFrame({
            "f_high": np.random.default_rng(2).normal(0.26, 0.04, 20),
            "f_low": np.random.default_rng(4).normal(1.0, 0.1, 20),
        })
        ranking = build_feature_drift_ranking(
            wf_df,
            ho_df,
            ["f_high", "f_low"],
            importance_map={"f_high": 0.18, "f_low": 0.02},
        )
        self.assertEqual(len(ranking), 2)
        top = ranking[0]
        self.assertEqual(top["feature"], "f_high")
        self.assertIsNotNone(top["wf_mean"])
        self.assertIsNotNone(top["holdout_mean"])
        self.assertGreater(float(top["drift_pct"]), 0)
        self.assertIsNotNone(top["ks_statistic"])
        self.assertIsNotNone(top["wasserstein_distance"])
        self.assertIsNotNone(top["wasserstein_normalized"])
        self.assertGreaterEqual(float(top["risk_score"]), 0.0)
        self.assertLessEqual(float(top["risk_score"]), 100.0)
        self.assertIn(top["risk"], ("high", "medium", "low"))
        self.assertEqual(top["risk"], "high")

    def test_feature_drift_risk_levels(self) -> None:
        self.assertEqual(feature_drift_risk(0.59, 0.18), "high")
        self.assertEqual(feature_drift_risk(0.59, 0.03), "medium")
        self.assertEqual(feature_drift_risk(0.20, 0.02), "low")

    def test_ks_detects_variance_shift_same_mean(self) -> None:
        rng = np.random.default_rng(7)
        wf = pd.Series(rng.normal(0.0, 1.0, 400))
        ho = pd.Series(rng.normal(0.0, 3.0, 200))
        ks, p = ks_drift_metrics(wf, ho)
        self.assertIsNotNone(ks)
        self.assertGreater(float(ks), 0.15)
        self.assertLess(float(p), 0.05)

    def test_wasserstein_shift_and_normalization(self) -> None:
        wf = pd.Series([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
        ho = pd.Series([2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
        raw, norm = wasserstein_drift_metrics(wf, ho, wf_std=float(np.std(wf.to_numpy())))
        self.assertIsNotNone(raw)
        self.assertAlmostEqual(float(raw), 2.0, places=5)
        self.assertGreater(float(norm), 0.0)
        # Identical samples → ~0 distance
        z_raw, z_norm = wasserstein_drift_metrics(wf, wf)
        self.assertAlmostEqual(float(z_raw or 0), 0.0, places=6)
        self.assertAlmostEqual(float(z_norm or 0), 0.0, places=6)

    def test_composite_risk_uses_normalized_wasserstein(self) -> None:
        # Same mean-drift / KS / null / importance; larger W_norm → higher risk
        low = composite_drift_risk_score(
            mean_drift=0.4,
            ks_statistic=0.2,
            wasserstein_normalized=0.2,
            null_drift_pp=0.0,
            importance=0.05,
        )
        high = composite_drift_risk_score(
            mean_drift=0.4,
            ks_statistic=0.2,
            wasserstein_normalized=4.0,
            null_drift_pp=0.0,
            importance=0.05,
        )
        self.assertGreater(high, low)
        self.assertGreaterEqual(low, 0.0)
        self.assertLessEqual(high, 100.0)
        self.assertEqual(composite_drift_risk_label(60.0), "high")
        self.assertEqual(composite_drift_risk_label(35.0), "medium")
        self.assertEqual(composite_drift_risk_label(10.0), "low")

    def test_composite_risk_contribution_shares(self) -> None:
        breakdown = composite_drift_risk_components(
            mean_drift=0.5,
            ks_statistic=1.0,
            wasserstein_normalized=2.0,  # → clamp01(1.0)
            null_drift_pp=0.0,
            importance=0.0,
        )
        self.assertAlmostEqual(breakdown["risk_score"], 50.0, places=2)
        shares = breakdown["shares_pct"]
        self.assertAlmostEqual(sum(shares.values()), 100.0, places=1)
        # KS and W each contribute a full 1.0 term; mean 0.5 → shares 20/40/40
        self.assertAlmostEqual(shares["mean_drift"], 20.0, places=1)
        self.assertAlmostEqual(shares["ks"], 40.0, places=1)
        self.assertAlmostEqual(shares["wasserstein_normalized"], 40.0, places=1)
        self.assertAlmostEqual(shares["null_drift"], 0.0, places=1)
        self.assertAlmostEqual(shares["importance"], 0.0, places=1)
        # Score helper stays consistent with breakdown
        self.assertEqual(
            composite_drift_risk_score(
                mean_drift=0.5,
                ks_statistic=1.0,
                wasserstein_normalized=2.0,
                null_drift_pp=0.0,
                importance=0.0,
            ),
            breakdown["risk_score"],
        )

    def test_series_null_pct(self) -> None:
        s = pd.Series([1.0, np.nan, 3.0, np.inf, 5.0])
        self.assertAlmostEqual(series_null_pct(s), 40.0, places=4)

class DiagnosisHeuristicTests(unittest.TestCase):
    def _rows_with_shift(self, shift: float) -> list[dict]:
        return [
            {"category": "Target distribution", "metric": "Mean", "shift": shift},
            {"category": "Feature distributions", "metric": "f1", "shift": shift},
        ]

    def test_overfitting_diagnosis(self) -> None:
        out = diagnose_degradation(
            wf_validation_mae=1.0,
            holdout_mae=1.5,
            comparison_rows=self._rows_with_shift(0.05),
            wf_target_std=1.0,
            holdout_target_std=1.0,
        )
        self.assertEqual(out["primary_cause"], "overfitting")

    def test_data_drift_diagnosis(self) -> None:
        out = diagnose_degradation(
            wf_validation_mae=1.0,
            holdout_mae=1.1,
            comparison_rows=self._rows_with_shift(0.4),
            wf_target_std=1.0,
            holdout_target_std=1.0,
            drift_scores={"target": 40.0, "feature": 50.0, "premium": 20.0, "volatility": 30.0},
            feature_ranking=[{"feature": "feat_a", "drift": 0.7}],
            wf_tgt_stats={"p50": 1.0},
            ho_tgt_stats={"p50": 1.2},
            premium_pct_change=14.0,
            holdout_unique_days=2,
            wf_unique_days=40,
        )
        self.assertEqual(out["primary_cause"], "data_drift")
        self.assertGreaterEqual(out["confidence_pct"], 70)
        self.assertTrue(any("Premium" in e for e in out["evidence"]))
        self.assertTrue(any("trading day" in e.lower() for e in out["evidence"]))
        self.assertIn("likely_reason", out)

    def test_difficult_market_diagnosis(self) -> None:
        out = diagnose_degradation(
            wf_validation_mae=1.0,
            holdout_mae=1.15,
            comparison_rows=self._rows_with_shift(0.1),
            wf_target_std=1.0,
            holdout_target_std=1.5,
        )
        self.assertEqual(out["primary_cause"], "difficult_market")

    def test_stable_diagnosis(self) -> None:
        out = diagnose_degradation(
            wf_validation_mae=1.0,
            holdout_mae=1.05,
            comparison_rows=self._rows_with_shift(0.05),
            wf_target_std=1.0,
            holdout_target_std=1.0,
        )
        self.assertEqual(out["primary_cause"], "stable")


class ResolveHoldoutSliceTests(unittest.TestCase):
    def test_from_summary_test_holdout(self) -> None:
        doc = {
            "walk_forward": {
                "summary": {
                    "data": {
                        "test_holdout": {"start": 900, "stop": 1000, "rows": 100},
                    }
                }
            }
        }
        start, stop = resolve_holdout_slice(doc, 1000)
        self.assertEqual((start, stop), (900, 1000))

    def test_reconstruct_from_split_config(self) -> None:
        doc = {
            "config": {
                "split": {
                    "strategy": "walk_forward",
                    "test": 20,
                    "walk_forward": {
                        "n_folds": 2,
                        "train_window_size": 50,
                        "validation_window_size": 20,
                    },
                }
            }
        }
        start, stop = resolve_holdout_slice(doc, 200)
        self.assertEqual(stop, 200)
        self.assertEqual(stop - start, 40)  # 20% of 200


class SavedPredictionMetricsTests(unittest.TestCase):
    def test_extract_matches_validation_tab_sources(self) -> None:
        doc = {
            "production_metrics": {
                "mae": 2.96,
                "rmse": 3.85,
                "premium_mae_pct": 2.4,
                "premium_rmse_pct": 3.02,
                "directional_accuracy_pct": 76.96,
            },
            "metrics": {
                "validation": {"mae": 2.96, "rmse": 3.85},
                "test": {
                    "mae": 9.97,
                    "rmse": 16.09,
                    "premium_mae_pct": 10.47,
                    "premium_rmse_pct": 80.02,
                    "directional_accuracy_pct": 50.37,
                },
            },
        }
        saved = extract_saved_prediction_metrics(doc)
        self.assertAlmostEqual(float(saved["production_wf"]["mae"]), 2.96)
        self.assertAlmostEqual(float(saved["holdout_test"]["mae"]), 9.97)

    def test_prediction_error_change_row(self) -> None:
        change = build_prediction_error_change_row(
            {"mae": 2.96, "rmse": 3.85, "premium_mae_pct": 2.4, "premium_rmse_pct": 3.02, "directional_accuracy_pct": 76.96},
            {"mae": 9.97, "rmse": 16.09, "premium_mae_pct": 10.47, "premium_rmse_pct": 80.02, "directional_accuracy_pct": 50.37},
        )
        self.assertEqual(change["label"], "Change")
        self.assertAlmostEqual(float(change["mae_pct_change"]), 236.8, delta=1.0)
        self.assertAlmostEqual(float(change["direction_pts_change"]), -26.59, places=1)


class HoldoutPerformanceCsvTests(unittest.TestCase):
    def test_build_holdout_performance_csv_includes_all_tabs(self) -> None:
        report = {
            "ok": True,
            "similarity_pct": 90.0,
            "drift_scores": {"target": 0.1, "feature": 0.2, "premium": 0.3, "volatility": 0.4},
            "prediction_errors": {
                "production_wf": {"mae": 2.0, "rmse": 3.0, "premium_mae_pct": 2.0, "premium_rmse_pct": 3.0, "directional_accuracy_pct": 70.0},
                "holdout_test": {"mae": 4.0, "rmse": 5.0, "premium_mae_pct": 6.0, "premium_rmse_pct": 7.0, "directional_accuracy_pct": 50.0},
                "change": {"mae_pct_change": 100.0, "rmse_pct_change": 66.7, "premium_mae_pct_change": 200.0, "premium_rmse_pct_change": 133.3, "direction_pts_change": -20.0},
            },
            "diagnosis": {"label": "Outlier-driven errors", "confidence_pct": 85.0, "evidence": ["Low target drift"]},
            "feature_drift_ranking": [{"feature": "gamma", "wf_mean": 0.01, "holdout_mean": 0.02, "drift_pct": 100.0, "drift": 0.5, "importance": 0.1, "risk_label": "high"}],
            "overview": {"wf_rows": 100, "holdout_rows": 20, "holdout_start": 80, "holdout_stop": 100, "wf_day_start": "2024-01-01", "wf_day_end": "2024-01-10", "holdout_day_start": "2024-01-11", "holdout_day_end": "2024-01-12", "volatility_column": "implied_vol"},
            "region_comparison": [{"category": "Target distribution", "metric": "Mean", "wf": 1.0, "holdout": 1.2, "shift": 0.2}],
            "holdout_by_premium_band": [{"band_label": "₹15-30", "samples": 10, "mae": 1.2, "rmse": 1.5, "premium_mae_pct": 12.0, "directional_accuracy_pct": 55.0}],
            "holdout_by_trading_day": [{"trading_day": "2024-01-11", "rows": 20, "mae": 1.1, "rmse": 1.4, "premium_mae_pct": 11.0, "directional_accuracy_pct": 54.0}],
            "premium_analysis": {
                "model_summary": {"overall_quality": "Good", "main_weakness": "Expiry gamma"},
                "root_cause": {"primary_cause": "Outliers", "conclusion": "Test"},
                "band_breakdown": [{"band_label": "₹15-30", "rows": 10, "premium_rmse_pct": 80.0, "contribution_pct": 40.0}],
                "top1_analysis": {
                    "ok": True,
                    "executive_summary": {"title": "Top 1% Error Investigation", "rows_analyzed": 10, "avg_premium_error_pct": 200.0, "patterns": ["100% are Expiry Day"]},
                    "metric_comparison": [{"metric": "Gamma", "top1_mean": 0.01, "rest_mean": 0.002, "difference_pct": 400.0}],
                    "distribution_comparison": [],
                    "conclusion": {"confidence_pct": 80.0, "root_causes": ["Expiry Gamma"], "recommendation": "Train specialist", "finding_text": "Weak on expiry"},
                },
            },
        }
        overview_csv = build_holdout_overview_csv(report)
        self.assertIn("Similarity Score", overview_csv)
        self.assertIn("Feature Drift Ranking", overview_csv)
        self.assertIn("Region Comparison", overview_csv)

        full_csv = build_holdout_performance_csv(report)
        self.assertIn("TAB: Overview", full_csv)
        self.assertIn("TAB: Premium Analysis", full_csv)
        self.assertIn("TAB: Top 1% Error Analysis", full_csv)
        self.assertIn("Model Summary", full_csv)
        self.assertIn("Executive Summary", full_csv)


class NonWalkForwardReportTests(unittest.TestCase):
    def test_non_wf_returns_error(self) -> None:
        report = build_holdout_performance_report("/tmp", {"is_walk_forward": False, "model_name": "m"})
        self.assertFalse(report.get("ok"))
        self.assertIn("walk-forward", str(report.get("error")).lower())


if __name__ == "__main__":
    unittest.main()

"""Tests for holdout premium RMSE analysis."""

from __future__ import annotations

import unittest

import numpy as np

from chain_replay_ml.training.holdout_premium_analysis import (
    build_model_summary,
    build_premium_analysis_csv,
    build_premium_root_cause_summary,
    outlier_contribution_to_premium_rmse,
    premium_rmse_band_breakdown,
    premium_rmse_excluding_top_pct,
    relative_error_percentiles,
)


class PremiumRmseBandBreakdownTests(unittest.TestCase):
    def test_contributions_sum_near_100(self) -> None:
        n = 200
        y = np.concatenate([np.full(100, 10.0), np.full(100, 100.0)])
        pred = y.copy()
        pred[0] = 50.0  # one huge outlier in cheap band
        rows = premium_rmse_band_breakdown(y, pred, baseline=y)
        total_contrib = sum(float(r.get("contribution_pct") or 0) for r in rows)
        self.assertAlmostEqual(total_contrib, 100.0, delta=0.5)
        self.assertGreater(float(rows[0]["contribution_pct"]), 30.0)


class OutlierContributionTests(unittest.TestCase):
    def test_top1_less_than_top10(self) -> None:
        y = np.full(1000, 20.0)
        pred = y.copy()
        pred[0] = 200.0
        pred[1] = 180.0
        rows = outlier_contribution_to_premium_rmse(y, pred)
        by_pct = {int(r["top_pct"]): float(r["contribution_pct"]) for r in rows}
        self.assertLessEqual(by_pct[1], by_pct[10])
        self.assertGreater(by_pct[1], 10.0)

    def test_uses_squared_not_absolute_relative_error(self) -> None:
        """Outlier share must be computed on rel^2 (RMSE basis), not |rel| (MAE basis)."""
        n = 5000
        y = np.full(n, 20.0)
        pred = y + 0.5  # small uniform relative error on every row
        pred[0] = 200.0
        pred[1] = 180.0
        pred[2] = 160.0

        rel_sq = np.square((pred - y) / np.abs(y))
        rel_abs = np.abs((pred - y) / np.abs(y))
        total_sq = float(np.sum(rel_sq))
        total_abs = float(np.sum(rel_abs))
        k = max(1, int(np.ceil(n * 1 / 100.0)))
        expected_rmse = round(float(np.sum(np.sort(rel_sq)[::-1][:k]) / total_sq * 100.0), 1)
        expected_mae = round(float(np.sum(np.sort(rel_abs)[::-1][:k]) / total_abs * 100.0), 1)

        rows = outlier_contribution_to_premium_rmse(y, pred)
        top1 = next(float(r["contribution_pct"]) for r in rows if r.get("top_pct") == 1)
        self.assertAlmostEqual(top1, expected_rmse, places=1)
        self.assertGreater(top1, expected_mae)


class PredictionQualityTests(unittest.TestCase):
    def test_rmse_excluding_top1_much_lower_than_full(self) -> None:
        n = 1000
        y = np.full(n, 20.0)
        pred = y.copy()
        pred[0] = 200.0
        pred[1] = 180.0
        full = relative_error_percentiles(y, pred)
        self.assertIsNotNone(full["median"])
        self.assertLess(float(full["median"]), 5.0)
        excl = premium_rmse_excluding_top_pct(y, pred, top_pct=1)
        from chain_replay_ml.training.evaluator import premium_rmse_pct
        full_rmse = premium_rmse_pct(y, pred)
        self.assertIsNotNone(excl)
        self.assertIsNotNone(full_rmse)
        self.assertLess(excl, full_rmse * 0.5)

    def test_outlier_impact_extreme_status(self) -> None:
        from chain_replay_ml.training.holdout_premium_analysis import build_outlier_impact
        n = 2000
        y = np.full(n, 20.0)
        pred = y.copy()
        pred[0] = 500.0
        impact = build_outlier_impact(y, pred)
        self.assertGreaterEqual(float(impact["contribution_pct"]), 90.0)
        self.assertEqual(impact["status"], "Extreme")
        self.assertEqual(impact["row_count"], 20)


class ModelSummaryTests(unittest.TestCase):
    def test_typical_good_outlier_poor_four_stars(self) -> None:
        summary = build_model_summary(
            root_cause={"primary_cause": "Extreme prediction errors concentrated in low-premium options"},
            quality_summary={
                "relative_error": {"median": 4.5, "p90": 22.8},
                "premium_rmse_excl_top_pct": 18.0,
            },
            outlier_impact={"contribution_pct": 99.4, "status": "Extreme"},
            band_breakdown=[{"band_label": "₹15-30", "rows": 100, "contribution_pct": 71.0}],
            top_error_samples=[{"reason": "Expiry gamma"}] * 5,
        )
        self.assertEqual(summary["overall_stars"], 4)
        self.assertEqual(summary["overall_quality"], "Good")
        self.assertIn("★★★★", summary["overall_stars_display"])
        self.assertEqual(summary["typical_prediction_quality"], "Good")
        self.assertEqual(summary["extreme_outlier_handling"], "Poor")
        self.assertEqual(summary["main_weakness"], "Low-premium expiry gamma options")


class PremiumRootCauseTests(unittest.TestCase):
    def test_high_similarity_outliers_conclusion(self) -> None:
        summary = build_premium_root_cause_summary(
            drift_scores={"volatility": 19.0, "feature": 11.0, "target": 7.0, "premium": 8.0},
            similarity_pct=90.0,
            production_premium_rmse=3.02,
            holdout_premium_rmse=80.02,
            outlier_rows=[
                {"top_pct": 1, "contribution_pct": 99.4},
                {"top_pct": 10, "contribution_pct": 99.9},
            ],
            band_breakdown=[
                {"band_label": "₹15-30", "rows": 17012, "contribution_pct": 71.3},
            ],
        )
        self.assertIn("similar", summary["conclusion"].lower())
        checklist_text = " ".join(item["text"] for item in summary["checklist"])
        self.assertIn("90% similar", checklist_text)
        self.assertIn("Target drift: Low (7%)", checklist_text)
        self.assertIn("Volatility drift: Moderate (19%)", checklist_text)
        self.assertTrue(any("dominated" in w.lower() for w in summary["warnings"]))
        self.assertTrue(any("99.4%" in b for b in summary["detail_bullets"]))
        self.assertTrue(any("71%" in b and "₹15-30" in b for b in summary["detail_bullets"]))
        self.assertEqual(
            summary["primary_cause"],
            "Extreme prediction errors concentrated in low-premium options",
        )


class PremiumAnalysisCsvTests(unittest.TestCase):
    def test_build_premium_analysis_csv_includes_sections(self) -> None:
        csv_text = build_premium_analysis_csv({
            "model_summary": {
                "overall_stars_display": "★★★★☆",
                "overall_quality": "Good",
                "typical_prediction_quality": "Good",
                "extreme_outlier_handling": "Poor",
                "main_weakness": "Low-premium expiry gamma options",
            },
            "root_cause": {
                "primary_cause": "Extreme prediction errors concentrated in low-premium options",
                "checklist": [{"status": "ok", "text": "Training vs Holdout: 90% similar"}],
                "warnings": ["Premium RMSE is dominated by a very small number of predictions."],
                "detail_bullets": [
                    "Top 1% of rows contribute 99.4% of total squared relative error "
                    "(Premium RMSE numerator, before √mean).",
                ],
                "bullets": ["Low target drift (+7%)"],
                "conclusion": "Test",
            },
            "band_breakdown": [{"band_label": "₹0-15", "rows": 10, "premium_rmse_pct": 420.0, "contribution_pct": 48.0}],
            "quality_summary": {
                "relative_error": {"median": 4.5, "p90": 22.8, "p95": 32.6, "p99": 54.5},
                "premium_rmse_pct": 166.0,
                "premium_rmse_excl_top_pct": 18.0,
                "exclude_top_pct": 1,
            },
            "outlier_impact": {
                "label": "Top 1% rows",
                "row_count": 1125,
                "contribution_pct": 99.4,
                "status": "Extreme",
            },
            "outlier_contribution": [{"label": "Top 1% rows", "contribution_pct": 42.0}],
            "error_distribution": {"wf": {"median": 1.2}, "holdout": {"median": 1.6}},
            "worst_trading_days": [{"trading_day": "2024-07-11", "mae": 18.6, "directional_accuracy_pct": 41.0}],
            "top_error_samples": [{"time": "10:21:18", "strike": "25100 CE", "actual": 18, "predicted": 52, "error": 34, "reason": "High IV spike"}],
        })
        self.assertIn("Model Summary", csv_text)
        self.assertIn("Root Cause Summary", csv_text)
        self.assertIn("Premium RMSE Breakdown", csv_text)
        self.assertIn("Prediction Quality", csv_text)
        self.assertIn("Outlier Impact", csv_text)
        self.assertIn("Top Error Samples", csv_text)
        self.assertIn("25100 CE", csv_text)


if __name__ == "__main__":
    unittest.main()

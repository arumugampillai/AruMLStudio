"""Unit tests for fold comparison (metrics, days, diagnosis)."""

from __future__ import annotations

import csv
import json
import os
import tempfile
import unittest
from unittest import mock

from chain_replay_ml.training.fold_comparison import (
    build_absolute_error_histogram,
    build_feature_importance_delta,
    build_fold_comparison,
    build_fold_comparison_csv,
    build_fold_diagnosis,
    list_fold_ids_on_disk,
    model_fold_metrics_table,
)


class AbsoluteErrorHistogramTests(unittest.TestCase):
    def test_bucket_counts_sum_to_total(self) -> None:
        errors = [0.5, 1.2, 2.5, 4.0, 6.0, 8.0]
        hist = build_absolute_error_histogram(errors)
        self.assertEqual(hist["total"], 6)
        self.assertEqual(sum(r["count"] for r in hist["rows"]), 6)
        tail = next(r for r in hist["rows"] if r["bucket"] == ">5")
        self.assertEqual(tail["count"], 2)


class FoldDiagnosisTests(unittest.TestCase):
    def test_headline_and_trading_day_reason(self) -> None:
        diagnosis = build_fold_diagnosis(
            label_a="Fold 5",
            label_b="Fold 10",
            metrics_a={"mae": 1.2, "rmse": 2.0, "directional_accuracy_pct": 60.0, "composite_score": 0.5},
            metrics_b={"mae": 2.6, "rmse": 3.5, "directional_accuracy_pct": 32.0, "composite_score": 0.2},
            days_a=["2026-06-18"],
            days_b=["2026-07-01"],
            premium_bands=[{
                "band": "15-30",
                "band_label": "₹15-30",
                "mae_a": 0.5,
                "mae_b": 1.8,
            }],
            importance={
                "available": True,
                "rows": [{"feature": "spot_high_ema50_to_ltp_ratio", "delta": 18.0}],
            },
        )
        self.assertIn("Fold 5 outperformed", diagnosis["headline"])
        self.assertEqual(diagnosis["winner"], "Fold 5")
        joined = " ".join(diagnosis["reasons"])
        self.assertIn("Direction", joined)
        self.assertIn("MAE lower", joined)
        self.assertIn("Validation days differ", joined)
        self.assertIn("spot_high_ema50_to_ltp_ratio", joined)


class FeatureImportanceDeltaTests(unittest.TestCase):
    def test_not_saved_when_missing(self) -> None:
        out = build_feature_importance_delta(
            {"available": False, "rows": []},
            {"available": True, "rows": [{"feature": "x", "importance_pct": 1.0}]},
            label_a="Fold 5",
            label_b="Fold 10",
        )
        self.assertFalse(out["available"])
        self.assertIn("Fold 5", out["message"] or "")

    def test_delta_sorted(self) -> None:
        out = build_feature_importance_delta(
            {"available": True, "rows": [
                {"feature": "a", "importance_pct": 10.0},
                {"feature": "b", "importance_pct": 5.0},
            ]},
            {"available": True, "rows": [
                {"feature": "a", "importance_pct": 12.0},
                {"feature": "b", "importance_pct": 20.0},
            ]},
            label_a="Fold 5",
            label_b="Fold 10",
        )
        self.assertTrue(out["available"])
        self.assertEqual(out["rows"][0]["feature"], "b")
        self.assertAlmostEqual(float(out["rows"][0]["delta"]), 15.0)
        self.assertEqual(out["rows"][0]["signal"], "fold_b")
        self.assertEqual(out["rows"][0]["signal_emoji"], "🔴")
        self.assertTrue(out["largest_shifts"])
        self.assertEqual(out["largest_shifts"][0]["feature"], "b")

    def test_fold_a_signal_on_negative_delta(self) -> None:
        out = build_feature_importance_delta(
            {"available": True, "rows": [
                {"feature": "spot_low_ema300_to_ltp_ratio", "importance_pct": 35.63},
            ]},
            {"available": True, "rows": [
                {"feature": "spot_low_ema300_to_ltp_ratio", "importance_pct": 11.30},
            ]},
            label_a="Fold 5",
            label_b="Fold 10",
        )
        row = out["rows"][0]
        self.assertEqual(row["signal"], "fold_a")
        self.assertEqual(row["signal_emoji"], "🟢")
        self.assertEqual(row["arrow"], "↓")
        self.assertIn("Fold 5", row["signal_label"])


class FoldComparisonIntegrationTests(unittest.TestCase):
    def test_build_from_temp_package(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model = "Demo_WF_Model"
            wf = os.path.join(tmp, "models", model, "walk_forward")
            for fold, days_unused in ((5, None), (10, None)):
                d = os.path.join(wf, f"fold_{fold:02d}")
                os.makedirs(d, exist_ok=True)
                with open(os.path.join(d, "fold.json"), "w", encoding="utf-8") as fh:
                    json.dump({
                        "fold": fold,
                        "train": {"start": 0, "stop": 100, "rows": 100},
                        "validation": {"start": 100 if fold == 5 else 200, "stop": 150 if fold == 5 else 250, "rows": 50},
                    }, fh)
                with open(os.path.join(d, "metrics.json"), "w", encoding="utf-8") as fh:
                    json.dump({
                        "mae": 1.0 if fold == 5 else 3.0,
                        "rmse": 1.5 if fold == 5 else 4.0,
                        "directional_accuracy_pct": 70.0 if fold == 5 else 40.0,
                        "composite_score": 0.6 if fold == 5 else 0.2,
                        "prediction_bias": 0.1,
                        "p95_error": 2.0,
                        "premium_band_performance": [
                            {"band": "15-30", "band_label": "₹15-30", "samples": 10, "mae": 0.5 if fold == 5 else 1.5, "directional_accuracy_pct": 60.0},
                        ],
                    }, fh)
                with open(os.path.join(d, "feature_importance.csv"), "w", encoding="utf-8", newline="") as fh:
                    w = csv.DictWriter(fh, fieldnames=["feature", "importance", "importance_pct"])
                    w.writeheader()
                    w.writerow({"feature": "feat_x", "importance": 1, "importance_pct": 10 if fold == 5 else 25})

            self.assertEqual(list_fold_ids_on_disk(tmp, model), [5, 10])

            def _load(path: str) -> dict:
                with open(path, encoding="utf-8") as fh:
                    return json.load(fh)

            doc = {
                "model_name": model,
                "is_walk_forward": True,
                "config": {"dataset": "unused"},
                "walk_forward": {
                    "champion_aggregate": {
                        "data": {
                            "fold_results": [
                                {
                                    "fold": 5,
                                    "fold_def": _load(os.path.join(wf, "fold_05", "fold.json")),
                                    "metrics": _load(os.path.join(wf, "fold_05", "metrics.json")),
                                },
                                {
                                    "fold": 10,
                                    "fold_def": _load(os.path.join(wf, "fold_10", "fold.json")),
                                    "metrics": _load(os.path.join(wf, "fold_10", "metrics.json")),
                                },
                            ]
                        }
                    }
                },
            }

            with mock.patch(
                "chain_replay_ml.training.fold_comparison.resolve_fold_trading_days",
                side_effect=lambda *_a, **_k: ["2026-06-18"] if _a[2].get("validation", {}).get("start") == 100 else ["2026-07-01"],
            ):
                report = build_fold_comparison(tmp, doc, 5, 10)

            self.assertTrue(report["ok"], report.get("error"))
            self.assertEqual(report["validation_window"]["fold_a"]["trading_days"], ["2026-06-18"])
            self.assertEqual(report["validation_window"]["fold_b"]["trading_days"], ["2026-07-01"])
            self.assertEqual(report["diagnosis"]["winner"], "Fold 5")
            self.assertTrue(report["feature_importance"]["available"])
            self.assertFalse(report["prediction_metrics"]["available"])
            self.assertIn("Prediction data not available", report["prediction_metrics"]["message"])

            table = model_fold_metrics_table(doc)
            self.assertEqual(len(table["rows"]), 2)
            self.assertEqual(table["rows"][0]["fold"], 5)
            self.assertEqual(table["fold_ids"], [5, 10])
            self.assertIn("champion", table["source_label"].lower())
            self.assertEqual(table["rows"][0]["validation_rows"], 50)
            eh = report.get("error_histograms") if isinstance(report.get("error_histograms"), dict) else {}
            self.assertIn("available", eh)
            self.assertEqual(table["rows"][1]["validation_rows"], 50)


class FoldComparisonCsvTests(unittest.TestCase):
    def test_csv_includes_all_tab_sections(self) -> None:
        report = {
            "ok": True,
            "model_name": "Demo_Model",
            "dataset": "MS_test",
            "fold_a": 6,
            "fold_b": 10,
            "label_a": "Fold 6",
            "label_b": "Fold 10",
            "source": "champion",
            "diagnosis": {
                "headline": "Fold 10 differs from Fold 6 mainly because:",
                "worse_label": "Fold 10",
                "better_label": "Fold 6",
                "winner": "Fold 6",
                "reasons": ["IV average +82%", "Spot volatility +53%"],
                "metric_cards": [{
                    "key": "mae",
                    "title": "MAE",
                    "fold_label": "Fold 10",
                    "value_display": "₹2.75 ↑",
                    "better_value": 1.1,
                    "worse_value": 2.75,
                    "why": ["IV average +82%"],
                }],
                "what_is_unique": {
                    "available": True,
                    "fold_label": "Fold 10",
                    "rows": [{"feature": "iv_zscore_5m", "display": "+3.2σ", "z_score": 3.2}],
                },
            },
            "summary_metrics": [
                ("MAE", 1.1, 2.75, "+1.65", "Fold 6"),
                ("Direction", 96.0, 78.0, "−18.00 pts", "Fold 6"),
            ],
            "error_metrics": [
                ("MAE", 1.1, 2.75, "+1.65", "Fold 6"),
                ("Bias", 0.1, 0.4, "+0.30", "Fold 6"),
            ],
            "error_histograms": {
                "available": True,
                "unit": "Rs",
                "insight": "Fold 10 has many >5 misses",
                "fold_a": {
                    "trading_days": ["2026-06-01"],
                    "total": 100,
                    "mean_abs_error": 1.2,
                    "median_abs_error": 0.8,
                    "max_abs_error": 6.0,
                    "tail_gt5_pct": 16.0,
                    "rows": [
                        {"bucket": "0-1", "count": 50, "pct": 50.0},
                        {"bucket": ">5", "count": 16, "pct": 16.0},
                    ],
                },
                "fold_b": {
                    "trading_days": ["2026-06-25"],
                    "total": 100,
                    "mean_abs_error": 4.5,
                    "median_abs_error": 5.2,
                    "max_abs_error": 20.0,
                    "tail_gt5_pct": 58.0,
                    "rows": [
                        {"bucket": "0-1", "count": 10, "pct": 10.0},
                        {"bucket": ">5", "count": 58, "pct": 58.0},
                    ],
                },
            },
            "premium_bands": [{
                "band": "15-30",
                "band_label": "₹15-30",
                "samples_a": 40,
                "samples_b": 55,
                "mae_a": 0.8,
                "mae_b": 2.1,
                "mae_winner": "Fold 6",
                "dir_a": 90.0,
                "dir_b": 70.0,
                "dir_winner": "Fold 6",
            }],
            "feature_importance": {
                "available": True,
                "rows": [{
                    "feature": "ltp",
                    "fold_a": 10.0,
                    "fold_b": 25.0,
                    "delta": 15.0,
                    "delta_display": "+15.00%",
                    "signal": "fold_b",
                    "signal_label": "Fold 10 relies much more",
                }],
                "largest_shifts": [{
                    "feature": "ltp",
                    "delta": 15.0,
                    "delta_display": "+15.00%",
                    "arrow": "↑",
                    "signal_label": "Fold 10 relies much more",
                }],
            },
            "prediction_metrics": {"available": False, "message": "Prediction data not available", "rows": []},
            "diagnostics": {
                "available": True,
                "context_a": {
                    "available": True,
                    "market_regime": "Sideways · Low Volatility",
                    "rows": [{"label": "Validation day(s)", "value": "2026-06-01"}],
                },
                "context_b": {
                    "available": True,
                    "market_regime": "Trending Up · High Volatility",
                    "rows": [{"label": "Validation day(s)", "value": "2026-06-25"}],
                },
                "distribution_shift": {
                    "available": True,
                    "rows": [{
                        "feature": "iv_zscore_5m",
                        "fold_a": 0.3,
                        "fold_b": 2.8,
                        "delta": 2.5,
                        "display_delta": "+833%",
                        "pct_change": 833.0,
                        "z_diff": 3.1,
                        "severity": "huge",
                    }],
                },
                "outliers_a": {"available": False, "message": "No strong outliers"},
                "outliers_b": {
                    "available": True,
                    "rows": [{
                        "feature": "iv_zscore_5m",
                        "fold_mean": 2.8,
                        "train_mean": 0.3,
                        "difference": 2.5,
                        "z_score": 3.2,
                        "display": "+3.2σ",
                        "percentile": 99.0,
                    }],
                },
            },
            "validation_window": {
                "fold_a": {
                    "fold": 6,
                    "trading_days": ["2026-06-01"],
                    "validation_rows": 100,
                    "train_rows": 1000,
                    "validation_start": 100,
                    "validation_stop": 200,
                },
                "fold_b": {
                    "fold": 10,
                    "trading_days": ["2026-06-25"],
                    "validation_rows": 120,
                    "train_rows": 2000,
                    "validation_start": 300,
                    "validation_stop": 420,
                },
            },
        }
        text = build_fold_comparison_csv(report)
        for needle in (
            "Fold Comparison",
            "Summary — Why?",
            "Summary — Fold Context: Fold 6",
            "Summary — Fold Context: Fold 10",
            "Summary — Summary Metrics",
            "Distributions — Feature Distribution Shift",
            "Distributions — Unusual vs Training: Fold 10",
            "Prediction Errors — Error Histogram",
            "Prediction Errors — Fold 6",
            "Prediction Errors — Fold 10",
            "Premium Bands",
            "Feature Importance",
            "Error Metrics",
            "Prediction Metrics",
            "IV average +82%",
            "iv_zscore_5m",
        ):
            self.assertIn(needle, text, f"missing section/content: {needle}")


if __name__ == "__main__":
    unittest.main()

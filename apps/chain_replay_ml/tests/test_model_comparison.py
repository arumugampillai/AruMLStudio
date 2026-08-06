"""Unit tests for model comparison row builders."""

from __future__ import annotations

import unittest

from master_dataset_tk.model_comparison import (
    _numeric_delta,
    build_core_model_metrics_comparison,
    build_feature_set_comparison,
    build_metric_comparison,
    build_premium_model_metrics_rows,
    build_premium_band_comparison,
    premium_metrics_available,
    build_summary_comparison,
    build_validation_comparison,
    build_walk_forward_comparison,
    model_display_label,
)
from master_dataset_tk.model_comparison_panel import ModelComparisonPanel


def _mock_doc(
    name: str,
    *,
    mae: float = 10.0,
    rmse: float = 12.0,
    direction: float = 55.0,
    composite: float = 0.45,
    is_wf: bool = False,
) -> dict:
    return {
        "model_name": name,
        "is_walk_forward": is_wf,
        "table_row": {"dataset": "DS_A", "target": "future_ltp_5m", "status": "ready"},
        "metadata": {"data": {"algorithm": "xgboost", "dataset": "DS_A", "target": "future_ltp_5m", "row_count": 1000}},
        "config": {"algorithm_label": "XGBoost", "dataset": "DS_A", "target": "future_ltp_5m"},
        "training_summary": {"rows": 1000, "features": 50, "validation_strategy_label": "Walk Forward"},
        "production_metrics": {
            "mae": mae,
            "rmse": rmse,
            "directional_accuracy_pct": direction,
            "composite_score": composite,
            "source_path": "walk_forward/champion_aggregate.json",
        },
        "metrics": {
            "validation": {"mae": mae - 1, "rmse": rmse - 1, "r2": 0.3, "directional_accuracy_pct": direction - 2},
            "test": {"mae": mae + 1, "rmse": rmse + 1, "r2": 0.2, "directional_accuracy_pct": direction - 5},
        },
        "walk_forward": {
            "display": {
                "n_folds": 5,
                "window_mode": "expanding",
                "train_window_size": 5000,
                "validation_window_size": 1000,
            },
            "champion_aggregate": {
                "data": {
                    "fold_results": [
                        {"fold": 1, "metrics": {"rmse": rmse, "mae": mae, "r2": 0.3, "directional_accuracy_pct": direction, "composite_score": composite}},
                        {"fold": 2, "metrics": {"rmse": rmse + 0.5, "mae": mae + 0.5, "r2": 0.28, "directional_accuracy_pct": direction - 1, "composite_score": composite - 0.01}},
                    ],
                },
            },
        },
    }


class TestModelComparison(unittest.TestCase):
    def test_model_display_label(self) -> None:
        self.assertEqual(model_display_label({"model_name": "Model_X"}), "Model_X")
        self.assertEqual(model_display_label({}), "—")

    def test_numeric_delta_lower_better(self) -> None:
        delta, winner = _numeric_delta(12.0, 10.0, higher_better=False)
        self.assertEqual(winner, "B")
        self.assertTrue(delta.startswith("−"))

    def test_numeric_delta_higher_better(self) -> None:
        delta, winner = _numeric_delta(50.0, 55.0, higher_better=True)
        self.assertEqual(winner, "B")
        self.assertTrue(delta.startswith("+"))

    def test_numeric_delta_tie(self) -> None:
        delta, winner = _numeric_delta(10.0, 10.0)
        self.assertEqual(winner, "Tie")
        self.assertEqual(delta, "0")

    def test_build_summary_comparison(self) -> None:
        doc_a = _mock_doc("A")
        doc_b = _mock_doc("B", is_wf=True)
        groups = build_summary_comparison(doc_a, doc_b)
        self.assertEqual(groups[0][0], "Validation Strategy")
        validation_rows = groups[0][1]
        self.assertTrue(any(r[0] == "Validation strategy" for r in validation_rows))
        self.assertTrue(any(r[0] == "Is walk forward" for r in validation_rows))
        self.assertEqual(groups[1][0], "Dataset")
        dataset_rows = groups[1][1]
        self.assertTrue(any(r[0] == "Dataset name" for r in dataset_rows))
        self.assertTrue(any(r[0] == "Rows count" for r in dataset_rows))
        self.assertTrue(any(r[0] == "Sampling" for r in dataset_rows))
        self.assertTrue(any(r[0] == "Trading day filter" for r in dataset_rows))
        self.assertTrue(any(r[0] == "Excluded expiry dates" for r in dataset_rows))
        general_rows = groups[2][1]
        self.assertTrue(any(r[0] == "Algorithm" for r in general_rows))
        self.assertFalse(any(r[0] == "Dataset" for r in general_rows))

    def test_dataset_trading_day_filter_rows(self) -> None:
        doc_a = _mock_doc("A")
        doc_a["dataset_build_snapshot"] = {
            "dataset_name": "MS_A",
            "trading_days": 9,
            "trading_day_labels": "2026-01-02, 2026-01-09",
            "filter_summary": [
                {"label": "Trading day filter", "value": "Exclude expiry days (2/4 days)"},
                {"label": "Excluded expiry dates", "value": "2026-01-08, 2026-01-15"},
                {"label": "LTP / Premium", "value": "15–40"},
            ],
            "sampling_label": "10s",
            "strike_selection_label": "±10",
        }
        doc_b = _mock_doc("B")
        doc_b["dataset_build_snapshot"] = {
            "dataset_name": "MS_B",
            "trading_days": 4,
            "trading_day_filter": {
                "mode": "all",
                "selected_days": 4,
                "exported_days": 4,
            },
            "filter_summary": [
                {"label": "Trading day filter", "value": "All selected days (4/4 days)"},
            ],
            "sampling_label": "10s",
            "strike_selection_label": "±10",
        }
        groups = build_summary_comparison(doc_a, doc_b)
        dataset_rows = dict((r[0], (r[1], r[2])) for r in groups[1][1])
        self.assertEqual(dataset_rows["Trading day filter"][0], "Exclude expiry days (2/4 days)")
        self.assertEqual(dataset_rows["Excluded expiry dates"][0], "2026-01-08, 2026-01-15")
        self.assertEqual(dataset_rows["Trading day filter"][1], "All selected days (4/4 days)")

    def test_dataset_trading_day_filter_from_registry_json(self) -> None:
        import json
        import os
        import tempfile

        tmp = tempfile.mkdtemp()
        ds_dir = os.path.join(tmp, "datasets")
        os.makedirs(ds_dir, exist_ok=True)
        with open(os.path.join(ds_dir, "MS_239f_3s_2223.json"), "w", encoding="utf-8") as fh:
            json.dump({
                "dataset_name": "MS_239f_3s_2223",
                "trading_days": 9,
                "days": [
                    {"trading_day": "2026-05-27"},
                    {"trading_day": "2026-05-29"},
                ],
                "trading_day_filter": {
                    "mode": "exclude_expiry",
                    "selected_days": 13,
                    "exported_days": 9,
                    "excluded_dates": ["2026-06-04", "2026-06-11", "2026-06-18", "2026-07-09"],
                    "expiry_dates": ["2026-06-04", "2026-06-11", "2026-06-18", "2026-07-09"],
                },
            }, fh)

        doc_a = _mock_doc("A")
        doc_a["_data_dir"] = tmp
        doc_a["dataset_build_snapshot"] = {
            "dataset_name": "MS_239f_3s_2223",
            "trading_days": 9,
            "filter_summary": [
                {"label": "LTP / Premium", "value": "15–500"},
            ],
            "sampling_label": "3s",
            "strike_selection_label": "±10",
        }
        doc_a["table_row"]["dataset"] = "MS_239f_3s_2223"
        doc_a["config"]["dataset"] = "MS_239f_3s_2223"
        doc_a["metadata"]["data"]["dataset"] = "MS_239f_3s_2223"
        doc_b = _mock_doc("B")
        doc_b["_data_dir"] = tmp
        groups = build_summary_comparison(doc_a, doc_b)
        dataset_rows = dict((r[0], (r[1], r[2])) for r in groups[1][1])
        self.assertIn("Exclude expiry days", str(dataset_rows["Trading day filter"][0]))
        self.assertIn("2026-06-04", str(dataset_rows["Excluded expiry dates"][0]))

    def test_build_core_model_metrics_comparison(self) -> None:
        doc_a = _mock_doc("A", mae=12.0, rmse=14.0, direction=50.0, composite=0.40)
        doc_b = _mock_doc("B", mae=10.0, rmse=11.0, direction=58.0, composite=0.48)
        rows = build_core_model_metrics_comparison(doc_a, doc_b)
        self.assertEqual([r[0] for r in rows], ["MAE", "RMSE", "Direction Accuracy", "Composite Score"])
        mae_row = next(r for r in rows if r[0] == "MAE")
        self.assertEqual(mae_row[4], "B")

    def test_build_metric_comparison_winner(self) -> None:
        doc_a = _mock_doc("A", mae=12.0, rmse=14.0, direction=50.0, composite=0.40)
        doc_b = _mock_doc("B", mae=10.0, rmse=11.0, direction=58.0, composite=0.48)
        rows = build_metric_comparison(doc_a, doc_b, "model")
        mae_row = next(r for r in rows if r[0] == "MAE (₹)")
        self.assertEqual(mae_row[4], "B")
        dir_row = next(r for r in rows if "Direction" in r[0])
        self.assertEqual(dir_row[4], "B")

    def test_build_validation_comparison_sections(self) -> None:
        doc_a = _mock_doc("A")
        doc_b = _mock_doc("B")
        sections = build_validation_comparison(doc_a, doc_b)
        self.assertIn("training_validation", sections)
        self.assertIn("production", sections)
        self.assertIn("holdout_test", sections)
        self.assertEqual(len(sections["production"]), 6)

    def test_build_walk_forward_comparison_folds(self) -> None:
        doc_a = _mock_doc("A", is_wf=True)
        doc_b = _mock_doc("B", is_wf=True, rmse=11.0)
        wf = build_walk_forward_comparison(doc_a, doc_b)
        self.assertTrue(wf["config"])
        self.assertEqual(len(wf["folds"]), 2)
        fold1 = wf["folds"][0]
        self.assertEqual(fold1["fold"], "1")
        rmse_row = next(r for r in fold1["metrics"] if r[0] == "RMSE")
        self.assertEqual(rmse_row[4], "B")

    def test_premium_metrics_available_detects_saved_fields(self) -> None:
        doc = _mock_doc("A")
        doc["production_metrics"]["premium_mae_pct"] = 10.0
        self.assertTrue(premium_metrics_available(doc))
        self.assertFalse(premium_metrics_available(_mock_doc("B")))

    def test_build_premium_model_metrics_rows(self) -> None:
        doc_a = _mock_doc("A", mae=12.0)
        doc_a["production_metrics"]["premium_mae_pct"] = 11.5
        doc_a["production_metrics"]["premium_rmse_pct"] = 13.2
        doc_b = _mock_doc("B", mae=10.0)
        doc_b["production_metrics"]["premium_mae_pct"] = 9.8
        rows = build_premium_model_metrics_rows(doc_a, doc_b)
        labels = [r[0] for r in rows]
        self.assertEqual(labels[0], "Premium MAE")
        self.assertEqual(labels[1], "Premium RMSE")
        self.assertEqual(rows[0][1], 11.5)

    def test_build_premium_band_comparison(self) -> None:
        doc_a = _mock_doc("A")
        doc_a["production_metrics"]["premium_band_performance"] = [
            {"band": "0-15", "band_label": "₹0-15", "mae": 8.0, "rmse": 10.0, "premium_mae_pct": 12.0, "premium_rmse_pct": 14.0, "directional_accuracy_pct": 55.0},
        ]
        doc_b = _mock_doc("B", mae=11.0)
        doc_b["production_metrics"]["premium_band_performance"] = [
            {"band": "0-15", "band_label": "₹0-15", "mae": 7.0, "rmse": 9.0, "premium_mae_pct": 10.0, "premium_rmse_pct": 11.0, "directional_accuracy_pct": 58.0},
        ]
        rows = build_premium_band_comparison(doc_a, doc_b)
        self.assertEqual(len(rows), 1)
        band, mae_pair, rmse_pair, mae_pct_pair, rmse_pct_pair, dir_pair, winner = rows[0]
        self.assertEqual(band, "₹0-15")
        self.assertEqual(mae_pair, "8.00 / 7.00")
        self.assertEqual(winner, "B")

    def test_summary_status_same_and_different(self) -> None:
        self.assertEqual(ModelComparisonPanel._summary_status("xgboost", "xgboost"), "Same")
        self.assertEqual(ModelComparisonPanel._summary_status(10, 10.0), "Same")
        self.assertEqual(ModelComparisonPanel._summary_status(None, ""), "Same")
        self.assertEqual(ModelComparisonPanel._summary_status("A", "B"), "Different")
        self.assertEqual(ModelComparisonPanel._summary_status(10, 11), "Different")

    def test_feature_set_comparison(self) -> None:
        doc_a = {
            "model_name": "A",
            "selected_features": ["f1", "f2", "f3"],
            "metadata": {},
            "config": {},
            "table_row": {},
        }
        doc_b = {
            "model_name": "B",
            "selected_features": ["f2", "f3", "f4"],
            "metadata": {},
            "config": {},
            "table_row": {},
        }
        cmp = build_feature_set_comparison(doc_a, doc_b)
        self.assertEqual(cmp["common"], ["f2", "f3"])
        self.assertEqual(cmp["only_a"], ["f1"])
        self.assertEqual(cmp["only_b"], ["f4"])
        self.assertEqual(cmp["common_count"], 2)
        self.assertEqual(cmp["only_a_count"], 1)
        self.assertEqual(cmp["only_b_count"], 1)


if __name__ == "__main__":
    unittest.main()

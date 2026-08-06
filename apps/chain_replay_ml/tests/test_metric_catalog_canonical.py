"""Canonical Model Quality / Trading Outcome metric catalog + evaluator APIs."""

from __future__ import annotations

import unittest
from unittest import mock

import numpy as np

from chain_replay_ml.prediction_meta.outcomes import compute_prediction_quality
from chain_replay_ml.training.evaluator import (
    direction_correct_flag,
    directional_accuracy_pct,
    endpoint_hit_rate_pct,
    evaluate_regression,
)
from chain_replay_ml.training.metric_catalog import (
    MODEL_QUALITY,
    TRADING_OUTCOME,
    UI_ENDPOINT_HIT,
    UI_PATH_TOUCH,
    assert_no_shared_ui_labels,
)


class MetricCatalogTests(unittest.TestCase):
    def test_no_shared_ui_labels(self) -> None:
        assert_no_shared_ui_labels()
        self.assertEqual(UI_ENDPOINT_HIT, "Endpoint Hit %")
        self.assertEqual(UI_PATH_TOUCH, "Path Touch Rate")
        mq = {v["ui_label"] for v in MODEL_QUALITY.values()}
        to = {v["ui_label"] for v in TRADING_OUTCOME.values()}
        self.assertFalse(mq & to)


class EndpointHitTests(unittest.TestCase):
    def test_formula(self) -> None:
        y = np.array([100.0, 100.0])
        pred = np.array([104.0, 120.0])
        self.assertAlmostEqual(endpoint_hit_rate_pct(y, pred), 50.0, places=2)

    def test_in_evaluate_regression(self) -> None:
        y = np.array([100.0, 50.0, 20.0])
        pred = np.array([102.0, 51.0, 20.5])
        base = np.array([99.0, 49.0, 21.0])
        m = evaluate_regression(y, pred, baseline=base)
        self.assertEqual(m["endpoint_hit_pct"], 100.0)
        self.assertEqual(m["hit_rate_pct"], 100.0)


class DirectionCanonicalTests(unittest.TestCase):
    def test_excludes_flat_actual(self) -> None:
        y = np.array([100.0, 110.0])
        pred = np.array([105.0, 120.0])
        base = np.array([100.0, 100.0])  # first actual flat
        self.assertEqual(directional_accuracy_pct(y, pred, base), 100.0)
        self.assertIsNone(direction_correct_flag(105.0, 100.0, 100.0))
        self.assertEqual(direction_correct_flag(120.0, 110.0, 100.0), 1)

    def test_prediction_quality_uses_canonical(self) -> None:
        q = compute_prediction_quality(
            ensemble_mean=105.0, entry_ltp=100.0, actual_ltp=100.0
        )
        self.assertIsNone(q["direction_correct"])  # flat actual excluded
        q2 = compute_prediction_quality(
            ensemble_mean=105.0, entry_ltp=100.0, actual_ltp=110.0
        )
        self.assertEqual(q2["direction_correct"], 1.0)


class FoldMetricsNoChampionRescoreTests(unittest.TestCase):
    def test_endpoint_hit_from_fold_metrics_only(self) -> None:
        from chain_replay_ml.training.fold_comparison import model_fold_metrics_table

        doc = {
            "model_name": "Demo",
            "is_walk_forward": True,
            "config": {"dataset": "x"},
            "walk_forward": {
                "summary": {
                    "data": {
                        "fold_results": [
                            {
                                "fold": 1,
                                "fold_def": {
                                    "validation": {"start": 0, "stop": 10, "rows": 10},
                                },
                                "metrics": {
                                    "mae": 1.0,
                                    "rmse": 1.5,
                                    "directional_accuracy_pct": 80.0,
                                    "endpoint_hit_pct": 55.0,
                                    "composite_score": 0.5,
                                },
                            }
                        ]
                    }
                }
            },
        }
        out = model_fold_metrics_table(doc, data_dir=None)
        self.assertEqual(out["rows"][0]["endpoint_hit_pct"], 55.0)
        self.assertIn("same fold model", (out.get("hit_note") or "").lower())

    def test_missing_endpoint_hit_not_champion_filled(self) -> None:
        from chain_replay_ml.training import fold_comparison as fc

        doc = {
            "model_name": "Demo",
            "is_walk_forward": True,
            "config": {"dataset": "x"},
            "walk_forward": {
                "summary": {
                    "data": {
                        "fold_results": [
                            {
                                "fold": 1,
                                "fold_def": {
                                    "validation": {"start": 0, "stop": 10, "rows": 10},
                                },
                                "metrics": {
                                    "mae": 1.0,
                                    "rmse": 1.5,
                                    "directional_accuracy_pct": 80.0,
                                    "composite_score": 0.5,
                                },
                            }
                        ]
                    }
                }
            },
        }
        with mock.patch.object(
            fc, "_compute_fold_target_hit_map", side_effect=AssertionError("must not call")
        ):
            out = fc.model_fold_metrics_table(doc, data_dir=None)
        self.assertIsNone(out["rows"][0]["endpoint_hit_pct"])
        self.assertIn("Retrain", out.get("hit_note") or "")


if __name__ == "__main__":
    unittest.main()

"""Tests for Feature Research laboratory."""

from __future__ import annotations

import os
import tempfile
import unittest

from chain_replay_ml.model_lab.feature_research import (
    analyze_feature,
    build_research_conclusion,
    filter_label_from_spec,
    filter_sql_from_spec,
    list_research_features,
)
from chain_replay_ml.model_lab.store import ModelLabStore
from chain_replay_ml.tests.test_research_dashboard import _seed_rows


class FeatureResearchTests(unittest.TestCase):
    def test_list_and_analyze(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "lab.db")
            # Need >= 9 rows for tertiles — extend seed
            _seed_rows(path)
            with ModelLabStore(path) as store:
                store.ensure_feature_columns(["sf_demo_feat"])
                extra = []
                for i in range(10):
                    extra.append(
                        {
                            "lab_uuid": "u2",
                            "prediction_id": f"px{i}",
                            "trading_day": "2026-01-04",
                            "timestamp": 10.0 + i,
                            "current_ltp": 25.0,
                            "expected_move": 1.0,
                            "actual_move": 1.0,
                            "predicted_trend": "UP",
                            "actual_trend": "UP",
                            "direction_correct": 1 if i % 2 == 0 else 0,
                            "target_reached": 1 if i < 7 else 0,
                            "time_to_target": 5.0 if i < 7 else -1.0,
                            "dd_before_target": 0.3,
                            "maximum_profit": 1.0,
                            "maximum_drawdown": 0.5,
                            "absolute_error": 1.0,
                            "prediction_error": 1.0,
                            "premium_error_pct": 4.0,
                            "sf_demo_feat": float(i) * 0.1 + 1.0,
                        }
                    )
                store.insert_prediction_rows(extra, feature_columns=["sf_demo_feat"])

            catalog = list_research_features(path)
            self.assertTrue(catalog["available"])
            names = [f["feature"] for f in catalog["features"]]
            self.assertIn("demo_feat", names)

            analysis = analyze_feature(path, "demo_feat")
            self.assertTrue(analysis.get("available"), analysis.get("error"))
            self.assertIn("stats", analysis)
            self.assertIsNotNone(analysis["stats"]["median"])
            self.assertIn("low", analysis["tertiles"])
            self.assertTrue(analysis.get("histogram"))
            self.assertGreater(int(analysis.get("rows") or 0), 0)
            self.assertEqual(analysis.get("rows_analyzed"), analysis.get("rows"))
            self.assertIsNotNone(analysis.get("missing_values"))
            self.assertEqual(analysis.get("research_rank"), 1)
            self.assertEqual(
                analysis.get("model_rank"),
                analysis.get("feature_rank"),
            )
            # Filters compile
            sql, args = filter_sql_from_spec(analysis["filters"]["low"])
            self.assertIn("sf_demo_feat", sql)
            self.assertEqual(len(args), 1)
            label = filter_label_from_spec(
                analysis["filters"]["best"] or analysis["filters"]["low"],
                feature="demo_feat",
            )
            self.assertIn("demo_feat", label)
            conclusion = analysis.get("conclusion") or {}
            self.assertTrue(conclusion.get("text"))
            self.assertIn("score_stars", conclusion)

    def test_research_conclusion_narrative(self) -> None:
        compare = {
            "low": {
                "hit_rate": 0.50,
                "direction_accuracy": 0.70,
                "mae": 7.95,
                "avg_dd_before_target": 4.71,
                "rows": 100,
            },
            "high": {
                "hit_rate": 0.548,
                "direction_accuracy": 0.499,
                "mae": 3.62,
                "avg_dd_before_target": 1.40,
                "rows": 100,
            },
            "delta_hit_rate": 0.048,
            "delta_dir": -0.201,
            "delta_mae": 3.62 - 7.95,
            "delta_dd": 1.40 - 4.71,
        }
        doc = build_research_conclusion("atm6_total_to_ltp_ratio", compare)
        self.assertEqual(doc["preferred"], "high")
        text = doc["text"]
        self.assertIn("improves Hit Rate", text)
        self.assertIn("reduce MAE", text)
        self.assertIn("reduce Drawdown", text)
        self.assertIn("Direction accuracy decreases", text)
        self.assertIn("★", doc["score_stars"])
        self.assertGreaterEqual(int(doc["score"]), 1)
        self.assertLessEqual(int(doc["score"]), 5)


if __name__ == "__main__":
    unittest.main()

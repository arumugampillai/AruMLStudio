"""Tests for Research Dashboard overall statistics."""

from __future__ import annotations

import os
import tempfile
import unittest

from chain_replay_ml.model_lab.research_dashboard import (
    OVERALL_STAT_ROWS,
    QUALITY_ROWS,
    compute_overall_statistics,
    compute_research_dashboard,
    compute_research_dashboard_for_day,
)
from chain_replay_ml.model_lab.store import ModelLabStore


def _seed_rows(path: str) -> None:
    with ModelLabStore(path) as store:
        store.ensure_prediction_schema()
        store.ensure_feature_columns(["sf_demo_feat"])
        store.write_prediction_summary(
            lab_uuid="u2",
            status="ready",
            row_count=4,
            trading_days=2,
            feature_columns_json='{"demo_feat": "sf_demo_feat"}',
            selected_feature_count=1,
        )
        store.insert_prediction_rows(
            [
                {
                    "lab_uuid": "u2",
                    "prediction_id": "p1",
                    "trading_day": "2026-01-02",
                    "timestamp": 1.0,
                    "current_ltp": 20.0,
                    "expected_move": 1.0,
                    "actual_move": 1.0,
                    "predicted_trend": "UP",
                    "actual_trend": "UP",
                    "direction_correct": 1,
                    "target_reached": 1,
                    "time_to_target": 10.0,
                    "dd_before_target": 0.5,
                    "maximum_profit": 2.0,
                    "maximum_drawdown": 1.0,
                    "absolute_error": 1.0,
                    "prediction_error": 1.0,
                    "premium_error_pct": 5.0,
                    "sf_demo_feat": 1.0,
                },
                {
                    "lab_uuid": "u2",
                    "prediction_id": "p2",
                    "trading_day": "2026-01-02",
                    "timestamp": 2.0,
                    "current_ltp": 40.0,
                    "expected_move": -1.0,
                    "actual_move": 1.0,
                    "predicted_trend": "DOWN",
                    "actual_trend": "UP",
                    "direction_correct": 0,
                    "target_reached": 0,
                    "time_to_target": -1.0,
                    "dd_before_target": 1.5,
                    "maximum_profit": 0.0,
                    "maximum_drawdown": 3.0,
                    "absolute_error": 3.0,
                    "prediction_error": -3.0,
                    "premium_error_pct": 15.0,
                    "sf_demo_feat": 2.0,
                },
                {
                    "lab_uuid": "u2",
                    "prediction_id": "p3",
                    "trading_day": "2026-01-03",
                    "timestamp": 3.0,
                    "current_ltp": 60.0,
                    "expected_move": 2.0,
                    "actual_move": 2.0,
                    "predicted_trend": "UP",
                    "actual_trend": "UP",
                    "direction_correct": 1,
                    "target_reached": 1,
                    "time_to_target": 20.0,
                    "dd_before_target": 0.2,
                    "maximum_profit": 4.0,
                    "maximum_drawdown": 0.5,
                    "absolute_error": 0.5,
                    "prediction_error": 0.5,
                    "premium_error_pct": 2.0,
                    "sf_demo_feat": 3.0,
                },
                {
                    "lab_uuid": "u2",
                    "prediction_id": "p4",
                    "trading_day": "2026-01-03",
                    "timestamp": 4.0,
                    "current_ltp": 220.0,
                    "expected_move": -2.0,
                    "actual_move": -1.0,
                    "predicted_trend": "DOWN",
                    "actual_trend": "DOWN",
                    "direction_correct": 1,
                    "target_reached": 1,
                    "time_to_target": 30.0,
                    "dd_before_target": 0.8,
                    "maximum_profit": 1.0,
                    "maximum_drawdown": 1.2,
                    "absolute_error": 2.0,
                    "prediction_error": -2.0,
                    "premium_error_pct": 8.0,
                    "sf_demo_feat": 10.0,
                },
            ],
            feature_columns=["sf_demo_feat"],
        )


class ResearchDashboardOverallStatsTests(unittest.TestCase):
    def test_empty_lab(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "lab.db")
            with ModelLabStore(path) as store:
                store.ensure_prediction_schema()
            stats = compute_overall_statistics(path)
            self.assertEqual(stats["total_predictions"], 0)
            self.assertFalse(stats["available"])
            self.assertGreaterEqual(len(OVERALL_STAT_ROWS), 11)
            self.assertEqual(len(QUALITY_ROWS), 7)

    def test_metrics_from_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "lab.db")
            _seed_rows(path)

            stats = compute_overall_statistics(path)
            self.assertTrue(stats["available"])
            self.assertEqual(stats["total_predictions"], 4)
            self.assertAlmostEqual(stats["direction_accuracy"], 0.75)
            self.assertAlmostEqual(stats["target_hit_rate"], 0.75)
            self.assertAlmostEqual(stats["target_miss_rate"], 0.25)
            self.assertAlmostEqual(stats["average_time_to_target"], 20.0)
            self.assertIsNotNone(stats["median_time_to_target"])
            self.assertIsNotNone(stats["p95_time_to_target"])
            self.assertIsNotNone(stats["mean_prediction_error"])

            dash = compute_research_dashboard(path)
            self.assertTrue(dash["available"])
            self.assertIn("target_hit_rate", dash["kpi"])
            # Second load must be cached (no rebuild)
            dash2 = compute_research_dashboard(path)
            self.assertTrue(dash2.get("cached"))
            self.assertFalse(dash2.get("rebuilt"))
            dist = dash["distribution"]
            self.assertEqual(dist["target_hits"], 3)
            self.assertEqual(dist["target_misses"], 1)
            self.assertAlmostEqual(dist["predicted_up_rate"], 0.5)
            bands = {b["band"]: b for b in dash["premium_bands"]}
            self.assertEqual(bands["₹15–30"]["rows"], 1)
            self.assertEqual(bands["₹200+"]["rows"], 1)
            self.assertEqual(len(dash["trading_days"]), 2)
            self.assertEqual(dash["features"], [])

            from chain_replay_ml.model_lab.research_dashboard import (
                refresh_research_dashboard_cache,
                research_dashboard_cache_is_fresh,
            )

            # Feature Research is a separate workload — rebuild must not populate it.
            rebuilt = refresh_research_dashboard_cache(path, force=True)
            self.assertTrue(rebuilt.get("rebuilt"))
            self.assertEqual(rebuilt.get("features") or [], [])
            self.assertTrue(research_dashboard_cache_is_fresh(path))

    def test_day_filter_scopes_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "lab.db")
            _seed_rows(path)
            all_dash = compute_research_dashboard(path)
            day_dash = compute_research_dashboard_for_day(path, "2026-01-02")
            self.assertTrue(day_dash["available"])
            self.assertEqual(day_dash.get("filter_trading_day"), "2026-01-02")
            self.assertEqual(int(day_dash["total_predictions"]), 2)
            self.assertEqual(int(all_dash["total_predictions"]), 4)
            self.assertAlmostEqual(
                float(day_dash["kpi"]["direction_accuracy"]),
                0.5,
            )
            self.assertEqual(len(day_dash["trading_days"]), 1)
            self.assertEqual(day_dash["trading_days"][0]["trading_day"], "2026-01-02")

            empty_day = compute_research_dashboard_for_day(path, "2026-01-99")
            self.assertFalse(empty_day["available"])
            self.assertEqual(empty_day.get("filter_trading_day"), "2026-01-99")


if __name__ == "__main__":
    unittest.main()

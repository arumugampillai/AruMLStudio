"""Tests for RR Validation report (read-only)."""

from __future__ import annotations

import os
import tempfile
import unittest

from chain_replay_ml.model_lab.prediction_schema import compute_rr_hit_labels
from chain_replay_ml.model_lab.rr_validation import load_rr_validation_report
from chain_replay_ml.model_lab.store import ModelLabStore


def _seed_labeled_rows(path: str) -> None:
    with ModelLabStore(path) as store:
        store.ensure_prediction_schema()
        store.write_prediction_summary(
            lab_uuid="u1",
            status="ready",
            row_count=4,
            trading_days=2,
            target_column="future_ltp_5m",
        )
        rows = []
        # (day, ts, exit_at, hit, profit, dd, mfe_at, mae_at, target_at, ltp)
        specs = [
            ("2026-07-01", 1.0, 301.0, 1, 12.0, 3.0, 250.0, 120.0, 280.0, 20.0),  # RR 4
            ("2026-07-01", 2.0, 302.0, 1, 8.0, 3.0, 260.0, 130.0, 290.0, 40.0),   # RR 2
            ("2026-07-02", 3.0, 303.0, 0, 15.0, 2.0, 210.0, 105.0, None, 80.0),  # miss
            ("2026-07-02", 4.0, 304.0, 1, 9.0, 3.0, 270.0, 140.0, 290.0, 150.0), # RR 3
        ]
        for i, (day, ts, exit_at, hit, profit, dd, mfe_at, mae_at, target_at, ltp) in enumerate(
            specs, start=1
        ):
            rr = compute_rr_hit_labels(
                target_reached=hit,
                maximum_profit=profit,
                maximum_drawdown=dd,
            )
            rows.append(
                {
                    "lab_uuid": "u1",
                    "prediction_id": f"p{i}",
                    "trading_day": day,
                    "timestamp": ts,
                    "exit_at": exit_at,
                    "current_ltp": ltp,
                    "target_reached": hit,
                    "target_reached_at": target_at,
                    "maximum_profit": profit,
                    "maximum_drawdown": dd,
                    "max_profit_at": mfe_at,
                    "max_drawdown_at": mae_at,
                    **rr,
                }
            )
        store.insert_prediction_rows(rows)


class RrValidationReportTests(unittest.TestCase):
    def test_empty_lab(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "lab.db")
            with ModelLabStore(path) as store:
                store.ensure_prediction_schema()
            report = load_rr_validation_report(path)
            self.assertFalse(report["available"])

    def test_summary_and_consistency(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "lab.db")
            _seed_labeled_rows(path)
            report = load_rr_validation_report(path)
            self.assertTrue(report["available"])
            self.assertEqual(report["total_rows"], 4)
            self.assertEqual(report["labeled_rows"], 4)
            self.assertTrue(report["consistency"]["ok"])
            self.assertTrue(report["outcome_window"]["ok"])

            by_metric = {r["metric"]: r for r in report["summary"]}
            self.assertEqual(by_metric["Target Hit"]["count"], 3)
            self.assertEqual(by_metric["RR 1:2 Hit"]["count"], 3)
            self.assertEqual(by_metric["RR 1:3 Hit"]["count"], 2)
            self.assertEqual(by_metric["RR 1:4 Hit"]["count"], 1)

            balance = {r["label"]: r for r in report["class_balance"]}
            self.assertEqual(balance["RR 1:2"]["positive"], 3)
            self.assertEqual(balance["RR 1:2"]["negative"], 1)
            self.assertAlmostEqual(balance["RR 1:2"]["positive_pct"], 75.0, places=2)
            self.assertAlmostEqual(balance["RR 1:3"]["positive_pct"], 50.0, places=2)
            self.assertAlmostEqual(balance["RR 1:4"]["positive_pct"], 25.0, places=2)

            # Ratios: 12/3=4, 8/3≈2.667, 15/2=7.5, 9/3=3
            rr = report["reward_risk"]
            self.assertEqual(rr["n"], 4)
            self.assertAlmostEqual(rr["avg"], (4.0 + 8.0 / 3.0 + 7.5 + 3.0) / 4.0, places=4)
            self.assertIsNotNone(rr["median"])
            self.assertIsNotNone(rr["p95"])

            bands = {b["band"]: b for b in report["premium_bands"]}
            self.assertIn("₹15–30", bands)
            self.assertAlmostEqual(bands["₹15–30"]["rr_1_4_pct"], 100.0, places=2)
            self.assertIn("₹100–200", bands)
            self.assertAlmostEqual(bands["₹100–200"]["rr_1_3_pct"], 100.0, places=2)

            days = {d["trading_day"]: d for d in report["trading_days"]}
            self.assertEqual(set(days), {"2026-07-01", "2026-07-02"})
            self.assertAlmostEqual(days["2026-07-01"]["target_hit_pct"], 100.0, places=2)
            self.assertAlmostEqual(days["2026-07-02"]["target_hit_pct"], 50.0, places=2)

    def test_consistency_failure_detected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "lab.db")
            with ModelLabStore(path) as store:
                store.ensure_prediction_schema()
                store.insert_prediction_rows(
                    [
                        {
                            "lab_uuid": "u1",
                            "prediction_id": "bad",
                            "trading_day": "2026-01-02",
                            "timestamp": 1.0,
                            "exit_at": 301.0,
                            "target_reached": 1,
                            "maximum_profit": 10.0,
                            "maximum_drawdown": 2.0,
                            "rr_1_2_hit": 0,
                            "rr_1_3_hit": 1,
                            "rr_1_4_hit": 0,
                        }
                    ]
                )
            report = load_rr_validation_report(path)
            self.assertFalse(report["consistency"]["ok"])
            self.assertIn("RR 1:3 exceeds RR 1:2", report["consistency"]["message"])


if __name__ == "__main__":
    unittest.main()

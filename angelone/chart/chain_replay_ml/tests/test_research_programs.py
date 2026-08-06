"""Tests for Research Programs."""

from __future__ import annotations

import os
import tempfile
import unittest

from chain_replay_ml.model_lab.research_programs import (
    build_program_conclusions,
    classify_evidence,
    format_feature_difference,
    list_research_programs,
    run_research_program,
)
from chain_replay_ml.model_lab.store import ModelLabStore
from chain_replay_ml.tests.test_research_dashboard import _seed_rows


class ResearchProgramsTests(unittest.TestCase):
    def test_catalog_and_run(self) -> None:
        programs = list_research_programs()
        self.assertGreaterEqual(len(programs), 8)
        ids = {p["id"] for p in programs}
        self.assertIn("top_success", ids)
        self.assertIn("premium_bands", ids)

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "lab.db")
            _seed_rows(path)
            with ModelLabStore(path) as store:
                store.ensure_feature_columns(["sf_demo_feat"])
                extra = []
                for i in range(30):
                    extra.append(
                        {
                            "lab_uuid": "u2",
                            "prediction_id": f"rp{i}",
                            "trading_day": "2026-01-05",
                            "timestamp": 20.0 + i,
                            "current_ltp": 20.0 + (i % 5) * 10.0,
                            "expected_move": 1.0,
                            "actual_move": 1.0,
                            "predicted_trend": "UP",
                            "actual_trend": "UP",
                            "direction_correct": 1 if i % 3 else 0,
                            "target_reached": 1 if i % 2 == 0 else 0,
                            "time_to_target": float(i * 10),
                            "dd_before_target": 0.0 if i < 5 else float(i),
                            "maximum_profit": 1.0,
                            "maximum_drawdown": 0.5,
                            "absolute_error": float(i) * 0.2,
                            "prediction_error": 1.0,
                            "premium_error_pct": 4.0,
                            "sf_demo_feat": float(i) * 0.1,
                        }
                    )
                store.insert_prediction_rows(extra, feature_columns=["sf_demo_feat"])

            top = run_research_program(path, "top_success", limit=5, compare_to="dataset")
            self.assertTrue(top.get("available"), top.get("error"))
            self.assertEqual(top.get("answer"), "Why did they succeed?")
            primary = top.get("primary") or {}
            self.assertGreaterEqual(int((primary.get("metrics") or {}).get("rows") or 0), 1)
            self.assertTrue(primary.get("where_sql"))
            self.assertTrue(primary.get("why_rows") or primary.get("feature_profile"))
            self.assertIn("conclusions", primary)
            self.assertTrue((primary.get("conclusions") or {}).get("text"))

            vs_worst = run_research_program(path, "top_success", limit=5, compare_to="worst")
            self.assertTrue(vs_worst.get("available"), vs_worst.get("error"))
            self.assertEqual(vs_worst.get("compare_to"), "worst")

            bands = run_research_program(path, "premium_bands")
            self.assertTrue(bands.get("available"), bands.get("error"))
            self.assertEqual(len(bands.get("cohorts") or []), 3)

            dream = run_research_program(path, "no_drawdown")
            self.assertTrue(dream.get("available"), dream.get("error"))
            self.assertGreaterEqual(
                int(((dream.get("primary") or {}).get("metrics") or {}).get("rows") or 0),
                1,
            )

    def test_difference_format_and_conclusions(self) -> None:
        d = format_feature_difference(0.000004, 0.000007)
        self.assertIn("%", d["display"])
        strong = classify_evidence(
            effect_pct=80.0,
            effect_abs=None,
            rows_affected=842,
            cohort_rows=1000,
            total_rows=10000,
        )
        self.assertEqual(strong["evidence"], "Strong")
        self.assertEqual(strong["effect"], "Strong")
        self.assertEqual(strong["confidence"], "High")
        weak = classify_evidence(
            effect_pct=80.0,
            effect_abs=None,
            rows_affected=42,
            cohort_rows=50,
            total_rows=10000,
        )
        self.assertEqual(weak["evidence"], "Weak")
        self.assertEqual(weak["confidence"], "Low")
        conc = build_program_conclusions(
            metrics={
                "hit_rate": 0.99,
                "mae": 0.01,
                "avg_dd_before_target": 1.2,
                "avg_time_to_target": 40.0,
                "avg_current_ltp": 35.0,
                "rows": 1000,
                "total_rows": 10000,
            },
            compare_metrics={
                "hit_rate": 0.80,
                "mae": 0.05,
                "avg_dd_before_target": 4.0,
                "avg_time_to_target": 90.0,
            },
            why_rows=[
                {
                    "feature": "gamma_ltp_to_spot_ratio",
                    "pct": -43.0,
                    "delta": -0.000003,
                    "difference": "↓43%",
                    "evidence": "Strong",
                    "rows_affected": 842,
                }
            ],
            program_id="top_success",
        )
        self.assertIn("Strong evidence", conc["text"])
        self.assertIn("Rows affected", conc["text"])


if __name__ == "__main__":
    unittest.main()

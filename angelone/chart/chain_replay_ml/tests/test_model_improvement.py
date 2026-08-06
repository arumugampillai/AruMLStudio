"""Tests for Model Improvement Lab."""

from __future__ import annotations

import os
import tempfile
import unittest

from chain_replay_ml.model_lab.model_improvement import (
    REC_CANDIDATE_REMOVE,
    REC_PROMOTE,
    REC_REVIEW,
    REC_STRONG_PROMOTE,
    REC_WATCH,
    classify_suggestion_evidence,
    compute_model_improvement,
    recommend_action,
    research_score_from_spread,
)
from chain_replay_ml.model_lab.store import ModelLabStore
from chain_replay_ml.tests.test_research_dashboard import _seed_rows


class ModelImprovementTests(unittest.TestCase):
    def test_recommend_rules(self) -> None:
        self.assertEqual(
            recommend_action(
                research_score=95, model_rank=25, n_features=60, unstable=False, in_model=True
            ),
            REC_WATCH,
        )
        self.assertEqual(
            recommend_action(
                research_score=91, model_rank=59, n_features=60, unstable=False, in_model=True
            ),
            REC_STRONG_PROMOTE,
        )
        self.assertEqual(
            recommend_action(
                research_score=75, model_rank=59, n_features=60, unstable=False, in_model=True
            ),
            REC_PROMOTE,
        )
        self.assertEqual(
            recommend_action(
                research_score=18, model_rank=10, n_features=60, unstable=False, in_model=True
            ),
            REC_CANDIDATE_REMOVE,
        )
        self.assertEqual(
            recommend_action(
                research_score=80, model_rank=5, n_features=60, unstable=True, in_model=True
            ),
            REC_REVIEW,
        )
        self.assertAlmostEqual(research_score_from_spread(0.05, max_spread=0.05), 100.0)

    def test_evidence_levels(self) -> None:
        high = classify_suggestion_evidence(
            total_rows=20_000,
            tertile={"low": {"rows": 5000}, "high": {"rows": 5000}},
        )
        self.assertEqual(high["evidence"], "High")
        low = classify_suggestion_evidence(
            total_rows=200,
            tertile={"low": {"rows": 10}, "high": {"rows": 10}},
        )
        self.assertEqual(low["evidence"], "Low")

    def test_compute_payload(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "lab.db")
            _seed_rows(path)
            with ModelLabStore(path) as store:
                store.ensure_feature_columns(["sf_demo_feat", "sf_weak_feat"])
                extra = []
                for i in range(40):
                    extra.append(
                        {
                            "lab_uuid": "u2",
                            "prediction_id": f"mi{i}",
                            "trading_day": "2026-01-06",
                            "timestamp": 30.0 + i,
                            "current_ltp": 25.0,
                            "expected_move": 1.0,
                            "actual_move": 1.0,
                            "predicted_trend": "UP",
                            "actual_trend": "UP",
                            "direction_correct": 1,
                            "target_reached": 1 if (i < 25 and i > 5) or i >= 30 else 0,
                            "time_to_target": 10.0,
                            "dd_before_target": 0.2,
                            "maximum_profit": 1.0,
                            "maximum_drawdown": 0.5,
                            "absolute_error": 0.1,
                            "prediction_error": 0.1,
                            "premium_error_pct": 2.0,
                            "sf_demo_feat": float(i),
                            "sf_weak_feat": 1.0,
                        }
                    )
                store.insert_prediction_rows(
                    extra, feature_columns=["sf_demo_feat", "sf_weak_feat"]
                )
                store.write_prediction_summary(
                    lab_uuid="u2",
                    status="ready",
                    row_count=44,
                    trading_days=3,
                    feature_columns_json='{"demo_feat":"sf_demo_feat","weak_feat":"sf_weak_feat"}',
                    selected_feature_count=2,
                )

            out = compute_model_improvement(path)
            self.assertTrue(out.get("available"), out.get("error"))
            row = (out.get("features") or [{}])[0]
            self.assertIn("evidence", row)
            self.assertIn("structured_evidence", row)
            self.assertIn("Research Score", row.get("structured_evidence") or {})
            self.assertIn("candidate_remove", out.get("answers") or {})


if __name__ == "__main__":
    unittest.main()

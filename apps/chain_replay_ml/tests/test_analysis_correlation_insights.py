"""Tests for Correlation Insights recommendations."""

from __future__ import annotations

import os
import tempfile
import unittest

import numpy as np
import pandas as pd

from chain_replay_ml.dataset_builder.analysis_correlation import run_correlation_analysis
from chain_replay_ml.dataset_builder.analysis_correlation_insights import (
    REC_DUPLICATE,
    REC_KEEP,
    REC_REVIEW,
    build_correlation_insights,
    load_correlation_insights,
    recommend_for_cluster,
)
from chain_replay_ml.dataset_builder.analysis_lab_store import (
    ensure_analysis_run,
    register_dataset,
)


class CorrelationInsightsRulesTests(unittest.TestCase):
    def test_duplicate_candidate_two_feature_same_family(self) -> None:
        decision = recommend_for_cluster(
            members=["volume_change_1m", "volume_return_frac_1m"],
            max_corr=1.0,
            avg_corr=1.0,
            family="Volume",
            same_family=True,
            intra_pairs=[("volume_change_1m", "volume_return_frac_1m", 1.0)],
        )
        self.assertEqual(decision["recommendation"], REC_DUPLICATE)
        self.assertTrue(decision["flags"]["possible_mathematical_duplicates"])
        self.assertTrue(decision["flags"]["investigate_duplicate_implementation"])
        self.assertIn("mathematically equivalent", decision["reason"])
        self.assertNotIn("delete", decision["reason"].lower().split("not ")[0])

    def test_large_cluster_review_after_discovery(self) -> None:
        members = [f"spot_ema{i}" for i in range(12)]
        decision = recommend_for_cluster(
            members=members,
            max_corr=1.0,
            avg_corr=0.99,
            family="Price",
            same_family=True,
            intra_pairs=[],
        )
        self.assertEqual(decision["recommendation"], REC_REVIEW)
        self.assertTrue(decision["flags"]["large_feature_family"])
        self.assertTrue(decision["flags"]["review_after_discovery"])
        self.assertIn("Discovery Rating", decision["reason"])
        self.assertNotIn("SHAP", decision["reason"])
        self.assertIn("not a feature selection tool", decision["reason"])

    def test_keep_small_moderate_cluster(self) -> None:
        decision = recommend_for_cluster(
            members=["spot", "spot_ema9", "spot_ema20"],
            max_corr=0.98,
            avg_corr=0.97,
            family="Price",
            same_family=True,
            intra_pairs=[("spot", "spot_ema9", 0.98)],
        )
        self.assertEqual(decision["recommendation"], REC_KEEP)
        self.assertFalse(decision["flags"]["possible_mathematical_duplicates"])


class CorrelationInsightsPipelineTests(unittest.TestCase):
    def test_run_correlation_persists_insights(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            n = 150
            spot = pd.Series(np.linspace(100, 200, n))
            # Near-perfect volume duplicates
            vol = pd.Series(np.linspace(1, 50, n))
            df = pd.DataFrame(
                {
                    "spot": spot,
                    "spot_ema9": spot * 1.0001,
                    "spot_ema20": spot * 1.0002,
                    "volume_change_1m": vol,
                    "volume_return_frac_1m": vol * 1.0000001,
                    "noise": np.random.default_rng(1).normal(size=n),
                }
            )
            # Pad a large price family (>10) with near-collinear copies
            for i in range(12):
                df[f"spot_level_{i}"] = spot * (1.0 + i * 1e-6)

            path = os.path.join(tmp, "insights_demo.parquet")
            df.to_parquet(path, index=False)
            register_dataset(tmp, path, name="insights_demo")
            run = ensure_analysis_run(tmp, "insights_demo")
            summary = run_correlation_analysis(
                tmp, run["run_id"], {"path": path, "dataset_id": "insights_demo"}
            )
            self.assertGreaterEqual(int(summary.get("insights_count") or 0), 1)

            insights = load_correlation_insights(tmp, run["run_id"])
            self.assertTrue(insights)
            recs = {str(i["recommendation"]) for i in insights}
            self.assertTrue(REC_DUPLICATE in recs or REC_REVIEW in recs)

            # Built insights from clusters also covers duplicate detection
            built = build_correlation_insights(
                clusters=[
                    {
                        "cluster": "Volume Family #7",
                        "members": ["volume_change_1m", "volume_return_frac_1m"],
                        "representative": "volume_change_1m",
                        "highest_correlation": 1.0,
                        "size": 2,
                    }
                ],
                pairs=[("volume_change_1m", "volume_return_frac_1m", 1.0)],
            )
            self.assertEqual(built[0]["recommendation"], REC_DUPLICATE)


if __name__ == "__main__":
    unittest.main()

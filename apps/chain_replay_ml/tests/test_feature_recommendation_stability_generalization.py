"""Unit tests for Phase 2B: Score Stability, Volatility, Risk Badges, and Level-1 Context Generalization."""

from __future__ import annotations

import hashlib
import os
import tempfile
import unittest
from datetime import datetime, timezone, timedelta

from chain_replay_ml.production_validation.api import (
    BasePipelinePolicy,
    ExperimentalLifecyclePolicy,
    FeatureRegistryPolicy,
    RecommendationPolicy,
    ScoringPolicy,
    build_dataset_context,
    compute_context_generalization,
    compute_evidence_confidence,
    compute_evidence_score,
    compute_model_consensus,
    compute_recency_staleness,
    compute_score_volatility,
    derive_risk_badges,
    get_population_recommendations,
)
from chain_replay_ml.production_validation.evidence_store import (
    append_validation_evidence,
    get_connection,
)


class TestFeatureRecommendationStabilityAndGeneralization(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp()
        self.chart_data = os.path.join(self.tmp, "data")
        os.makedirs(self.chart_data, exist_ok=True)
        self.policy = RecommendationPolicy()
        self.now_utc = datetime.now(timezone.utc)

    def test_score_volatility_insufficient_runs_n1_n2(self) -> None:
        # N = 1
        rows_n1 = [{
            "evidence_id": "ev_01",
            "run_timestamp": self.now_utc.isoformat(),
            "recommendation": "KEEP",
        }]
        res_n1 = compute_score_volatility(rows_n1, policy=self.policy)
        self.assertEqual(res_n1["total_observations"], 1)
        self.assertIsNone(res_n1["volatility_score"])
        self.assertIsNone(res_n1["score_range"])
        self.assertIsNone(res_n1["direction_flips"])
        self.assertEqual(res_n1["stability_label"], "Insufficient Data")
        self.assertIn("N/A (< 3 runs)", res_n1["display_text"])

        # N = 2
        rows_n2 = [
            {"evidence_id": "ev_01", "run_timestamp": (self.now_utc - timedelta(days=2)).isoformat(), "recommendation": "KEEP"},
            {"evidence_id": "ev_02", "run_timestamp": self.now_utc.isoformat(), "recommendation": "KEEP"},
        ]
        res_n2 = compute_score_volatility(rows_n2, policy=self.policy)
        self.assertEqual(res_n2["total_observations"], 2)
        self.assertIsNone(res_n2["volatility_score"])
        self.assertIsNone(res_n2["score_range"])
        self.assertIsNone(res_n2["direction_flips"])
        self.assertEqual(res_n2["stability_label"], "Insufficient Data")
        self.assertIn("N/A (< 3 runs)", res_n2["display_text"])

    def test_score_volatility_identical_scores_sigma_zero(self) -> None:
        # Custom policy where score is capped or identical
        custom_scoring = ScoringPolicy(weight_keep=0.0, weight_remove=0.0, weight_watch=0.0, bonus_consecutive_keep=0.0)
        rows_n3 = [
            {"evidence_id": "ev_01", "run_timestamp": (self.now_utc - timedelta(days=3)).isoformat(), "recommendation": "KEEP"},
            {"evidence_id": "ev_02", "run_timestamp": (self.now_utc - timedelta(days=2)).isoformat(), "recommendation": "KEEP"},
            {"evidence_id": "ev_03", "run_timestamp": self.now_utc.isoformat(), "recommendation": "KEEP"},
        ]
        res_n3 = compute_score_volatility(rows_n3, policy=custom_scoring)
        self.assertEqual(res_n3["total_observations"], 3)
        self.assertEqual(res_n3["volatility_score"], 0.0)
        self.assertEqual(res_n3["score_range"], 0.0)
        self.assertEqual(res_n3["direction_flips"], 0)
        self.assertEqual(res_n3["stability_label"], "Stable")
        self.assertIn("Stable (σ=0.0)", res_n3["display_text"])

    def test_score_volatility_alternating_scores_volatile(self) -> None:
        # N = 4: KEEP -> REMOVE -> KEEP -> REMOVE (extreme flip-flop)
        rows_n4 = [
            {"evidence_id": "ev_01", "run_timestamp": (self.now_utc - timedelta(days=4)).isoformat(), "recommendation": "KEEP"},
            {"evidence_id": "ev_02", "run_timestamp": (self.now_utc - timedelta(days=3)).isoformat(), "recommendation": "REMOVE"},
            {"evidence_id": "ev_03", "run_timestamp": (self.now_utc - timedelta(days=2)).isoformat(), "recommendation": "KEEP"},
            {"evidence_id": "ev_04", "run_timestamp": self.now_utc.isoformat(), "recommendation": "REMOVE"},
        ]
        res_n4 = compute_score_volatility(rows_n4, policy=self.policy)
        self.assertEqual(res_n4["total_observations"], 4)
        self.assertIsNotNone(res_n4["volatility_score"])
        self.assertGreaterEqual(res_n4["volatility_score"], 35.0)
        self.assertEqual(res_n4["stability_label"], "Volatile")
        self.assertGreater(res_n4["score_range"], 40.0)
        self.assertEqual(res_n4["direction_flips"], 2)  # up -> down -> up -> down => 2 direction reversals

    def test_level1_context_generalization_k1_single_context(self) -> None:
        # Only 1 context present
        summaries = {
            "ctx_nifty_3s": {
                "feat_alpha": {"dominant_recommendation": "KEEP", "evidence_score": 80.0}
            }
        }
        res = compute_context_generalization(
            primary_context_id="ctx_nifty_3s",
            feature_name="feat_alpha",
            context_summaries_by_cid=summaries,
            level1_comparable_context_ids=[],
        )
        self.assertEqual(res["comparable_context_count"], 1)
        self.assertIsNone(res["generalization_score"])
        self.assertIsNone(res["agreement_ratio"])
        self.assertEqual(res["generalization_label"], "Single Context")
        self.assertIn("Single Context", res["display_text"])

    def test_level1_context_generalization_high_g_universal(self) -> None:
        # K = 3 comparable contexts (e.g. NIFTY 3s, NIFTY 6s, NIFTY 10s)
        # All 3 have dominant KEEP, with close scores (80, 85, 75 => Delta = 10)
        summaries = {
            "ctx_nifty_3s": {"feat_alpha": {"dominant_recommendation": "KEEP", "evidence_score": 80.0}},
            "ctx_nifty_6s": {"feat_alpha": {"dominant_recommendation": "KEEP", "evidence_score": 85.0}},
            "ctx_nifty_10s": {"feat_alpha": {"dominant_recommendation": "KEEP", "evidence_score": 75.0}},
        }
        level1_cids = ["ctx_nifty_6s", "ctx_nifty_10s"]
        res = compute_context_generalization(
            primary_context_id="ctx_nifty_3s",
            feature_name="feat_alpha",
            context_summaries_by_cid=summaries,
            level1_comparable_context_ids=level1_cids,
        )
        self.assertEqual(res["comparable_context_count"], 3)
        self.assertEqual(res["agreement_ratio"], 1.0)
        self.assertEqual(res["score_spread"], 10.0)
        # G = 1.0 * (1 - 10 / 100) = 0.90
        self.assertEqual(res["generalization_score"], 0.90)
        self.assertEqual(res["generalization_label"], "Universal")
        self.assertIn("Universal (G=0.90)", res["display_text"])

    def test_level1_context_generalization_opposite_recs_scale_specific(self) -> None:
        # K = 2 comparable contexts (NIFTY 3s has KEEP (+60), NIFTY 6s has REMOVE (-40) => Delta = 100)
        summaries = {
            "ctx_nifty_3s": {"feat_beta": {"dominant_recommendation": "KEEP", "evidence_score": 60.0}},
            "ctx_nifty_6s": {"feat_beta": {"dominant_recommendation": "REMOVE", "evidence_score": -40.0}},
        }
        level1_cids = ["ctx_nifty_6s"]
        res = compute_context_generalization(
            primary_context_id="ctx_nifty_3s",
            feature_name="feat_beta",
            context_summaries_by_cid=summaries,
            level1_comparable_context_ids=level1_cids,
        )
        self.assertEqual(res["comparable_context_count"], 2)
        self.assertEqual(res["agreement_ratio"], 0.5)
        self.assertEqual(res["score_spread"], 100.0)
        # G = 0.5 * (1 - min(1, 100/100)) = 0.0
        self.assertEqual(res["generalization_score"], 0.0)
        self.assertEqual(res["generalization_label"], "Scale-Specific")
        self.assertIn("Specific (G=0.00)", res["display_text"])

    def test_comparability_hierarchy_isolation(self) -> None:
        # Level 1: same market, same window, same project, diff interval
        ctx_nifty_3s = build_dataset_context(market="NIFTY", sampling_interval_sec=3, sliding_window="standard", feature_project_id="all")
        ctx_nifty_6s = build_dataset_context(market="NIFTY", sampling_interval_sec=6, sliding_window="standard", feature_project_id="all")
        # Level 2: same market, same interval, diff window
        ctx_nifty_3s_rolling = build_dataset_context(market="NIFTY", sampling_interval_sec=3, sliding_window="rolling", feature_project_id="all")
        # Level 3: diff market
        ctx_sensex_1s = build_dataset_context(market="SENSEX", sampling_interval_sec=1, sliding_window="standard", feature_project_id="all")

        conn = get_connection(self.chart_data)
        try:
            append_validation_evidence(
                conn,
                context=ctx_nifty_3s,
                evidence_rows=[{
                    "evidence_id": "ev_n3_1", "feature_name": "f1", "feature_source": "base_pipeline",
                    "validation_run_id": "run_1", "model_name": "M1", "run_timestamp": self.now_utc.isoformat(), "recommendation": "KEEP"
                }]
            )
            append_validation_evidence(
                conn,
                context=ctx_nifty_6s,
                evidence_rows=[{
                    "evidence_id": "ev_n6_1", "feature_name": "f1", "feature_source": "base_pipeline",
                    "validation_run_id": "run_2", "model_name": "M1", "run_timestamp": self.now_utc.isoformat(), "recommendation": "KEEP"
                }]
            )
            append_validation_evidence(
                conn,
                context=ctx_nifty_3s_rolling,
                evidence_rows=[{
                    "evidence_id": "ev_n3r_1", "feature_name": "f1", "feature_source": "base_pipeline",
                    "validation_run_id": "run_3", "model_name": "M1", "run_timestamp": self.now_utc.isoformat(), "recommendation": "REMOVE"
                }]
            )
            append_validation_evidence(
                conn,
                context=ctx_sensex_1s,
                evidence_rows=[{
                    "evidence_id": "ev_sx_1", "feature_name": "f1", "feature_source": "base_pipeline",
                    "validation_run_id": "run_4", "model_name": "M1", "run_timestamp": self.now_utc.isoformat(), "recommendation": "REMOVE"
                }]
            )
        finally:
            conn.close()

        # Query NIFTY 3s population: it should only match Level 1 with NIFTY 6s (K = 2), ignoring rolling window and SENSEX
        rows_3s = get_population_recommendations(self.chart_data, population="base_pipeline", context_id=ctx_nifty_3s.context_id)
        self.assertEqual(len(rows_3s), 1)
        r = rows_3s[0]
        self.assertEqual(r["comparable_context_count"], 2)
        self.assertEqual(r["generalization_label"], "Universal")  # 3s and 6s are both KEEP

    def test_explicit_risk_badges_derivation(self) -> None:
        # Clean feature
        badges_clean = derive_risk_badges(evidence_score=50.0, is_consensus_tie=False, freshness_label="Fresh", stability_label="Stable")
        self.assertEqual(badges_clean, [])

        # Multiple risk factors
        badges_risky = derive_risk_badges(evidence_score=-50.0, is_consensus_tie=True, freshness_label="Stale", stability_label="Volatile")
        self.assertIn("DEGRADED", badges_risky)
        self.assertIn("SPLIT", badges_risky)
        self.assertIn("STALE", badges_risky)
        self.assertIn("UNSTABLE", badges_risky)

    def test_evidence_db_immutability_and_schema_preservation(self) -> None:
        ctx = build_dataset_context(market="NIFTY", sampling_interval_sec=3, sliding_window="standard", feature_project_id="all")
        conn = get_connection(self.chart_data)
        try:
            append_validation_evidence(
                conn,
                context=ctx,
                evidence_rows=[
                    {"evidence_id": f"ev_{i}", "feature_name": f"f_{i}", "feature_source": "registry", "validation_run_id": "r1", "model_name": "m1", "run_timestamp": self.now_utc.isoformat(), "recommendation": "KEEP"}
                    for i in range(10)
                ]
            )
            cur = conn.execute("SELECT evidence_id, feature_name, recommendation FROM recommendation_evidence ORDER BY evidence_id;")
            raw_before = [tuple(r) for r in cur.fetchall()]
            chk_before = hashlib.sha256(str(raw_before).encode()).hexdigest()
            schema_before = {r["name"]: r["sql"] for r in conn.execute("SELECT name, sql FROM sqlite_master WHERE type='table';").fetchall()}
        finally:
            conn.close()

        # Run population queries with Phase 2B enrichment
        get_population_recommendations(self.chart_data, population="registry", context_id=ctx.context_id)
        get_population_recommendations(self.chart_data, population="base_pipeline", context_id=ctx.context_id)
        get_population_recommendations(self.chart_data, population="experimental", context_id=ctx.context_id)

        conn = get_connection(self.chart_data)
        try:
            cur = conn.execute("SELECT evidence_id, feature_name, recommendation FROM recommendation_evidence ORDER BY evidence_id;")
            raw_after = [tuple(r) for r in cur.fetchall()]
            chk_after = hashlib.sha256(str(raw_after).encode()).hexdigest()
            schema_after = {r["name"]: r["sql"] for r in conn.execute("SELECT name, sql FROM sqlite_master WHERE type='table';").fetchall()}
        finally:
            conn.close()

        self.assertEqual(chk_before, chk_after, "Raw recommendation_evidence mutated!")
        self.assertEqual(schema_before, schema_after, "SQLite schema mutated!")


if __name__ == "__main__":
    unittest.main()

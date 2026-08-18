"""Comprehensive unit test suite for Phase 2A Feature Recommendation Evidence Intelligence."""

from __future__ import annotations

import os
import tempfile
import unittest
from datetime import datetime, timedelta, timezone

from chain_replay_ml.production_validation.api import (
    BasePipelinePolicy,
    ExperimentalLifecyclePolicy,
    FeatureRegistryPolicy,
    RecommendationPolicy,
    ScoringPolicy,
    build_dataset_context,
    compute_evidence_confidence,
    compute_evidence_score,
    compute_model_consensus,
    compute_recency_staleness,
    get_population_recommendations,
)
from chain_replay_ml.production_validation.evidence_store import (
    append_validation_evidence,
    get_connection,
)


class TestFeatureRecommendationIntelligence(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp()
        self.ctx = build_dataset_context(
            market="NIFTY",
            sampling_interval_sec=3,
            sliding_window="standard",
            feature_project_id="all",
        )

    def test_evidence_confidence_calculation(self) -> None:
        # Reference Case 1: N=0, M=0 -> C = 0.000
        self.assertEqual(compute_evidence_confidence(0, 0), 0.0)
        self.assertEqual(compute_evidence_confidence(0, 2), 0.0)
        self.assertEqual(compute_evidence_confidence(2, 0), 0.0)

        # Reference Case 2: N=1, M=1 -> C ≈ 0.334
        c_1_1 = compute_evidence_confidence(1, 1)
        self.assertAlmostEqual(c_1_1, 0.334, places=3)
        self.assertEqual(round(c_1_1, 3), 0.334)

        # Reference Case 3: N=2, M=2 -> C ≈ 0.555
        c_2_2 = compute_evidence_confidence(2, 2)
        self.assertAlmostEqual(c_2_2, 0.555, places=3)
        self.assertEqual(round(c_2_2, 3), 0.555)

        # Reference Case 4: N=3, M=2 -> C ≈ 0.632
        c_3_2 = compute_evidence_confidence(3, 2)
        self.assertAlmostEqual(c_3_2, 0.632, places=3)
        self.assertEqual(round(c_3_2, 3), 0.632)

        # Reference Case 5: N=5, M=3 -> C ≈ 0.794
        c_5_3 = compute_evidence_confidence(5, 3)
        self.assertAlmostEqual(c_5_3, 0.794, places=3)
        self.assertEqual(round(c_5_3, 3), 0.794)

        # Reference Case 6: N=10, M=4 -> C ≈ 0.913
        c_10_4 = compute_evidence_confidence(10, 4)
        self.assertAlmostEqual(c_10_4, 0.913, places=3)
        self.assertEqual(round(c_10_4, 3), 0.913)

    def test_configurable_saturation_parameters(self) -> None:
        # Default policy (k_runs=3.0, k_models=2.0)
        pol_default = ScoringPolicy()
        c_def = compute_evidence_confidence(3, 2, policy=pol_default)
        self.assertAlmostEqual(c_def, 0.6321, places=3)

        # Slower saturation policy (k_runs=6.0, k_models=4.0)
        pol_slow = ScoringPolicy(confidence_runs_saturation=6.0, confidence_models_saturation=4.0)
        c_slow = compute_evidence_confidence(3, 2, policy=pol_slow)
        self.assertLess(c_slow, c_def)
        self.assertAlmostEqual(c_slow, 0.3935, places=3)

    def test_recency_staleness_and_freshness_bands(self) -> None:
        now = datetime(2026, 8, 18, 12, 0, 0, tzinfo=timezone.utc)

        # 1. Fresh (< 24h)
        ts_fresh = (now - timedelta(hours=4)).isoformat()
        res_fresh = compute_recency_staleness(ts_fresh, now_utc=now)
        self.assertEqual(res_fresh["freshness_label"], "Fresh")
        self.assertIn("Fresh (4h ago)", res_fresh["display_text"])

        # 2. Recent (1d - 7d)
        ts_recent = (now - timedelta(days=3)).isoformat()
        res_recent = compute_recency_staleness(ts_recent, now_utc=now)
        self.assertEqual(res_recent["freshness_label"], "Recent")
        self.assertIn("Recent (3d ago)", res_recent["display_text"])

        # 3. Aging (7d - 30d)
        ts_aging = (now - timedelta(days=15)).isoformat()
        res_aging = compute_recency_staleness(ts_aging, now_utc=now)
        self.assertEqual(res_aging["freshness_label"], "Aging")
        self.assertIn("Aging (15d ago)", res_aging["display_text"])

        # 4. Stale (>= 30d)
        ts_stale = (now - timedelta(days=45)).isoformat()
        res_stale = compute_recency_staleness(ts_stale, now_utc=now)
        self.assertEqual(res_stale["freshness_label"], "Stale")
        self.assertIn("Stale (45d ago)", res_stale["display_text"])

        # 5. Missing / None timestamp
        res_none = compute_recency_staleness(None, now_utc=now)
        self.assertEqual(res_none["freshness_label"], "Unvalidated")
        self.assertIsNone(res_none["staleness_seconds"])

        # 6. Corrupted string
        res_corrupt = compute_recency_staleness("not-a-timestamp", now_utc=now)
        self.assertEqual(res_corrupt["freshness_label"], "Unvalidated")

    def test_model_consensus_unanimous_and_split(self) -> None:
        # Unanimous 2 models
        rows = [
            {"model_name": "ModelA", "recommendation": "KEEP", "run_timestamp": "2026-08-16T10:00:00Z"},
            {"model_name": "ModelB", "recommendation": "KEEP", "run_timestamp": "2026-08-16T11:00:00Z"},
        ]
        res = compute_model_consensus(rows)
        self.assertEqual(res["total_models"], 2)
        self.assertEqual(res["dominant_recommendation"], "KEEP")
        self.assertEqual(res["consensus_ratio"], 1.0)
        self.assertFalse(res["is_tie"])
        self.assertIn("100.0% KEEP (2/2)", res["display_text"])

        # Majority 2-to-1
        rows_split = [
            {"model_name": "ModelA", "recommendation": "KEEP", "run_timestamp": "2026-08-16T10:00:00Z"},
            {"model_name": "ModelB", "recommendation": "KEEP", "run_timestamp": "2026-08-16T11:00:00Z"},
            {"model_name": "ModelC", "recommendation": "REMOVE", "run_timestamp": "2026-08-16T12:00:00Z"},
        ]
        res_split = compute_model_consensus(rows_split)
        self.assertEqual(res_split["total_models"], 3)
        self.assertEqual(res_split["dominant_recommendation"], "KEEP")
        self.assertAlmostEqual(res_split["consensus_ratio"], 0.6667, places=3)
        self.assertFalse(res_split["is_tie"])
        self.assertIn("66.7% KEEP (2/3)", res_split["display_text"])

    def test_model_consensus_vote_change_across_runs(self) -> None:
        # ModelA voted WATCH in Run 1, then voted KEEP in Run 2
        # ModelB voted KEEP in Run 1
        # Model Consensus must evaluate latest recommendation per model -> both ModelA and ModelB are KEEP!
        rows = [
            {"model_name": "ModelA", "recommendation": "WATCH", "run_timestamp": "2026-08-16T09:00:00Z"},
            {"model_name": "ModelB", "recommendation": "KEEP", "run_timestamp": "2026-08-16T10:00:00Z"},
            {"model_name": "ModelA", "recommendation": "KEEP", "run_timestamp": "2026-08-16T11:00:00Z"},
        ]
        res = compute_model_consensus(rows)
        self.assertEqual(res["total_models"], 2)
        self.assertEqual(res["vote_distribution"], {"KEEP": 2, "WATCH": 0, "REMOVE": 0})
        self.assertEqual(res["dominant_recommendation"], "KEEP")
        self.assertEqual(res["consensus_ratio"], 1.0)
        self.assertFalse(res["is_tie"])

    def test_model_consensus_strict_ties(self) -> None:
        # 50/50 exact tie: ModelA = KEEP, ModelB = REMOVE
        rows_50_50 = [
            {"model_name": "ModelA", "recommendation": "KEEP", "run_timestamp": "2026-08-16T10:00:00Z"},
            {"model_name": "ModelB", "recommendation": "REMOVE", "run_timestamp": "2026-08-16T11:00:00Z"},
        ]
        res_50_50 = compute_model_consensus(rows_50_50)
        self.assertEqual(res_50_50["total_models"], 2)
        self.assertTrue(res_50_50["is_tie"])
        self.assertEqual(res_50_50["dominant_recommendation"], "SPLIT (KEEP/REMOVE)")
        self.assertEqual(res_50_50["consensus_ratio"], 0.50)
        self.assertIn("50.0% SPLIT (1 KEEP / 1 REMOVE)", res_50_50["display_text"])

        # 3-way tie: ModelA = KEEP, ModelB = WATCH, ModelC = REMOVE
        rows_3way = [
            {"model_name": "ModelA", "recommendation": "KEEP", "run_timestamp": "2026-08-16T10:00:00Z"},
            {"model_name": "ModelB", "recommendation": "WATCH", "run_timestamp": "2026-08-16T11:00:00Z"},
            {"model_name": "ModelC", "recommendation": "REMOVE", "run_timestamp": "2026-08-16T12:00:00Z"},
        ]
        res_3way = compute_model_consensus(rows_3way)
        self.assertEqual(res_3way["total_models"], 3)
        self.assertTrue(res_3way["is_tie"])
        self.assertEqual(res_3way["dominant_recommendation"], "SPLIT (3-WAY)")
        self.assertAlmostEqual(res_3way["consensus_ratio"], 0.3333, places=3)
        self.assertIn("33.3% SPLIT (1 KEEP / 1 WATCH / 1 REMOVE)", res_3way["display_text"])

    def test_model_consensus_single_model(self) -> None:
        rows_single = [
            {"model_name": "ModelA", "recommendation": "KEEP", "run_timestamp": "2026-08-16T10:00:00Z"},
            {"model_name": "ModelA", "recommendation": "KEEP", "run_timestamp": "2026-08-16T11:00:00Z"},
        ]
        res_single = compute_model_consensus(rows_single)
        self.assertEqual(res_single["total_models"], 1)
        self.assertFalse(res_single["is_tie"])
        self.assertEqual(res_single["dominant_recommendation"], "KEEP")
        self.assertEqual(res_single["consensus_ratio"], 1.0)
        self.assertIn("1/1 model - Single", res_single["display_text"])

    def test_base_pipeline_dual_ranking(self) -> None:
        conn = get_connection(self.tmp)
        try:
            # Feature 1: High raw score (+85) from 1 run on 1 model -> Low confidence (33.4%) -> Op Score ~ +28.4
            # Feature 2: Moderate raw score (+70) from 6 runs on 3 models -> High confidence (83.8%) -> Op Score ~ +58.7
            rows = [
                {"evidence_id": "ev_f1", "feature_name": "feat_untested_flash", "feature_source": "base_pipeline", "recommendation": "KEEP", "validation_run_id": "r1", "model_name": "ModelA", "run_timestamp": "2026-08-16T10:00:00Z"},
                {"evidence_id": "ev_f2_1", "feature_name": "feat_proven_workhorse", "feature_source": "base_pipeline", "recommendation": "KEEP", "validation_run_id": "r1", "model_name": "ModelA", "run_timestamp": "2026-08-16T10:00:00Z"},
                {"evidence_id": "ev_f2_2", "feature_name": "feat_proven_workhorse", "feature_source": "base_pipeline", "recommendation": "KEEP", "validation_run_id": "r2", "model_name": "ModelB", "run_timestamp": "2026-08-16T11:00:00Z"},
                {"evidence_id": "ev_f2_3", "feature_name": "feat_proven_workhorse", "feature_source": "base_pipeline", "recommendation": "KEEP", "validation_run_id": "r3", "model_name": "ModelC", "run_timestamp": "2026-08-16T12:00:00Z"},
            ]
            append_validation_evidence(conn, context=self.ctx, evidence_rows=rows)
        finally:
            conn.close()

        base_rows = get_population_recommendations(self.tmp, population="base_pipeline", context_id=self.ctx.context_id)
        self.assertEqual(len(base_rows), 2)
        by_name = {r["feature_name"]: r for r in base_rows}

        untested = by_name["feat_untested_flash"]
        proven = by_name["feat_proven_workhorse"]

        # 1. Phase 1 priority_rank: feat_proven_workhorse (3 models, score=100.0) is rank 1, untested (1 model, score=40.0) is rank 2
        self.assertEqual(proven["priority_rank"], 1)
        self.assertEqual(untested["priority_rank"], 2)

        # 2. Evidence Confidence
        self.assertGreater(proven["evidence_confidence"], untested["evidence_confidence"])

        # 3. Operational Priority Score (score * conf)
        self.assertGreater(proven["operational_priority_score"], untested["operational_priority_score"])

        # 4. Advisory Rank
        self.assertEqual(proven["advisory_rank"], 1)
        self.assertEqual(untested["advisory_rank"], 2)

    def test_dataset_context_isolation_in_intelligence(self) -> None:
        ctx_sensex = build_dataset_context(market="SENSEX", sampling_interval_sec=1, sliding_window="standard", feature_project_id="all")
        conn = get_connection(self.tmp)
        try:
            rows_nifty = [
                {"evidence_id": "n_1", "feature_name": "feat_vol", "feature_source": "registry", "recommendation": "KEEP", "validation_run_id": "rn1", "model_name": "ModelA", "run_timestamp": "2026-08-16T10:00:00Z"},
                {"evidence_id": "n_2", "feature_name": "feat_vol", "feature_source": "registry", "recommendation": "KEEP", "validation_run_id": "rn2", "model_name": "ModelB", "run_timestamp": "2026-08-16T11:00:00Z"},
            ]
            rows_sensex = [
                {"evidence_id": "s_1", "feature_name": "feat_vol", "feature_source": "registry", "recommendation": "REMOVE", "validation_run_id": "rs1", "model_name": "ModelA", "run_timestamp": "2026-08-16T10:00:00Z"},
            ]
            append_validation_evidence(conn, context=self.ctx, evidence_rows=rows_nifty)
            append_validation_evidence(conn, context=ctx_sensex, evidence_rows=rows_sensex)
        finally:
            conn.close()

        res_nifty = get_population_recommendations(self.tmp, population="registry", context_id=self.ctx.context_id)
        res_sensex = get_population_recommendations(self.tmp, population="registry", context_id=ctx_sensex.context_id)

        self.assertEqual(res_nifty[0]["dominant_recommendation"], "KEEP")
        self.assertEqual(res_nifty[0]["consensus_ratio"], 1.0)
        self.assertAlmostEqual(res_nifty[0]["evidence_confidence"], 0.5546, places=3)

        self.assertEqual(res_sensex[0]["dominant_recommendation"], "REMOVE")
        self.assertEqual(res_sensex[0]["consensus_ratio"], 1.0)
        self.assertAlmostEqual(res_sensex[0]["evidence_confidence"], 0.3338, places=3)

    def test_evidence_db_immutability_invariant(self) -> None:
        conn = get_connection(self.tmp)
        try:
            rows = [
                {"evidence_id": "ev_imm_1", "feature_name": "feat_imm", "feature_source": "registry", "recommendation": "KEEP", "validation_run_id": "r1", "model_name": "ModelA", "run_timestamp": "2026-08-16T10:00:00Z"},
            ]
            append_validation_evidence(conn, context=self.ctx, evidence_rows=rows)
            count_before = conn.execute("SELECT count(*) FROM recommendation_evidence;").fetchone()[0]
        finally:
            conn.close()

        # Query recommendations with intelligence calculations multiple times
        for _ in range(5):
            get_population_recommendations(self.tmp, population="registry", context_id=self.ctx.context_id)
            get_population_recommendations(self.tmp, population="base_pipeline", context_id=self.ctx.context_id)
            get_population_recommendations(self.tmp, population="experimental", context_id=self.ctx.context_id)

        conn2 = get_connection(self.tmp)
        try:
            count_after = conn2.execute("SELECT count(*) FROM recommendation_evidence;").fetchone()[0]
            self.assertEqual(count_before, count_after)
        finally:
            conn2.close()

    def test_ui_viewer_tabs_with_intelligence_columns(self) -> None:
        import tkinter as tk
        from master_dataset_tk.feature_recommendation_viewer import open_feature_recommendation_viewer

        root = tk.Tk()
        root.withdraw()
        try:
            dlg = open_feature_recommendation_viewer(
                root,
                chart_dir=self.tmp,
                initial_market="NIFTY",
                initial_interval_sec=3,
                initial_sliding_window="standard",
                initial_feature_project_id="all",
            )
            # Verify treeviews have the Phase 2A columns
            reg_cols = dlg._reg_tree["columns"]
            self.assertIn("confidence", reg_cols)
            self.assertIn("consensus", reg_cols)
            self.assertIn("freshness", reg_cols)

            base_cols = dlg._base_tree["columns"]
            self.assertIn("rank", base_cols)
            self.assertIn("adj_score", base_cols)
            self.assertIn("advisory_rank", base_cols)
            self.assertIn("confidence", base_cols)
            self.assertIn("consensus", base_cols)
            self.assertIn("freshness", base_cols)

            exp_cols = dlg._exp_tree["columns"]
            self.assertIn("confidence", exp_cols)
            self.assertIn("consensus", exp_cols)
            self.assertIn("freshness", exp_cols)

            dlg.destroy()
        finally:
            root.destroy()


if __name__ == "__main__":
    unittest.main()

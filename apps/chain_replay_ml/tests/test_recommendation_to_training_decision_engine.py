"""Unit tests for Phase 3A Recommendation-to-Training Decision Engine Core."""

from __future__ import annotations

import hashlib
import os
import sqlite3
import tempfile
import unittest

from chain_replay_ml.production_validation.dataset_context import DatasetContext, build_dataset_context
from chain_replay_ml.production_validation.evidence_store import (
    append_validation_evidence,
    get_connection,
)
from chain_replay_ml.production_validation.recommendation_policy import (
    RecommendationPolicy,
    ScoringPolicy,
    TrainingDecisionPolicy,
    load_policy_store,
    load_recommendation_policy,
    restore_policy_version,
    save_recommendation_policy,
    validate_recommendation_policy,
)
from chain_replay_ml.production_validation.training_decision_engine import (
    TrainingDecisionResult,
    TrainingDecisionState,
    evaluate_candidate_training_eligibility,
    evaluate_population_training_decisions,
    evaluate_training_decision,
)


def _file_sha256(filepath: str) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as fh:
        while chunk := fh.read(8192):
            h.update(chunk)
    return h.hexdigest()


class TestTrainingDecisionEngineCore(unittest.TestCase):
    """Tests for Phase 3A Decision Engine Core logic and contracts."""

    def test_four_boolean_contract_exact_mapping(self) -> None:
        """Verify the 4-state boolean contract matches exact specifications."""
        pol = RecommendationPolicy()

        # 1. TRAIN_CANDIDATE
        res_cand = evaluate_training_decision(
            feature_name="feat_a",
            context_id="ctx_1",
            feature_source="experimental",
            total_runs=3,
            unique_models_count=2,
            evidence_score=50.0,
            evidence_confidence=0.60,
            dominant_recommendation="KEEP",
            policy=pol,
        )
        self.assertEqual(res_cand.decision, TrainingDecisionState.TRAIN_CANDIDATE)
        self.assertFalse(res_cand.is_excluded)
        self.assertTrue(res_cand.is_candidate_generation_allowed)
        self.assertTrue(res_cand.is_training_candidate)
        self.assertFalse(res_cand.requires_review)

        # 2. REVIEW
        res_rev = evaluate_training_decision(
            feature_name="feat_b",
            context_id="ctx_1",
            feature_source="experimental",
            total_runs=2,
            unique_models_count=1,
            evidence_score=10.0,  # Below candidate min (+20.0)
            evidence_confidence=0.50,
            dominant_recommendation="KEEP",
            policy=pol,
        )
        self.assertEqual(res_rev.decision, TrainingDecisionState.REVIEW)
        self.assertFalse(res_rev.is_excluded)
        self.assertTrue(res_rev.is_candidate_generation_allowed)
        self.assertFalse(res_rev.is_training_candidate)
        self.assertTrue(res_rev.requires_review)

        # 3. NEW_UNSEEN (total_runs == 0)
        res_unseen = evaluate_training_decision(
            feature_name="feat_c",
            context_id="ctx_1",
            feature_source="experimental",
            total_runs=0,
            policy=pol,
        )
        self.assertEqual(res_unseen.decision, TrainingDecisionState.NEW_UNSEEN)
        self.assertFalse(res_unseen.is_excluded)
        self.assertTrue(res_unseen.is_candidate_generation_allowed)
        self.assertFalse(res_unseen.is_training_candidate)
        self.assertFalse(res_unseen.requires_review)

        # 4. EXCLUDE
        res_excl = evaluate_training_decision(
            feature_name="feat_d",
            context_id="ctx_1",
            feature_source="experimental",
            total_runs=4,
            remove_runs=4,  # Breaches total remove threshold
            evidence_score=-50.0,
            policy=pol,
        )
        self.assertEqual(res_excl.decision, TrainingDecisionState.EXCLUDE)
        self.assertTrue(res_excl.is_excluded)
        self.assertFalse(res_excl.is_candidate_generation_allowed)
        self.assertFalse(res_excl.is_training_candidate)
        self.assertFalse(res_excl.requires_review)

    def test_zero_runs_invariant(self) -> None:
        """NEW_UNSEEN is returned strictly and only when total_runs == 0."""
        pol = RecommendationPolicy()

        # total_runs == 0 returns NEW_UNSEEN
        res0 = evaluate_training_decision(
            feature_name="f_new",
            context_id="ctx_1",
            feature_source="experimental",
            total_runs=0,
            policy=pol,
        )
        self.assertEqual(res0.decision, TrainingDecisionState.NEW_UNSEEN)

        # total_runs == 1 never returns NEW_UNSEEN
        res1 = evaluate_training_decision(
            feature_name="f_new",
            context_id="ctx_1",
            feature_source="experimental",
            total_runs=1,
            unique_models_count=1,
            evidence_score=5.0,  # Below min candidate score
            evidence_confidence=0.15,
            dominant_recommendation="KEEP",
            policy=pol,
        )
        self.assertNotEqual(res1.decision, TrainingDecisionState.NEW_UNSEEN)
        self.assertEqual(res1.decision, TrainingDecisionState.REVIEW)

    def test_population_immunities(self) -> None:
        """Feature Registry and Base Pipeline features are immune from hard exclusion."""
        pol = RecommendationPolicy()

        # Severe score on Feature Registry routes to REVIEW [ALERT], not EXCLUDE
        res_reg = evaluate_training_decision(
            feature_name="reg_feature",
            context_id="ctx_1",
            feature_source="registry",
            total_runs=5,
            unique_models_count=3,
            evidence_score=-80.0,
            consecutive_remove_count=3,
            remove_runs=5,
            dominant_recommendation="REMOVE",
            policy=pol,
        )
        self.assertEqual(res_reg.decision, TrainingDecisionState.REVIEW)
        self.assertFalse(res_reg.is_excluded)
        self.assertTrue(res_reg.requires_review)
        self.assertIn("[ALERT]", res_reg.reason_badges)
        self.assertEqual(res_reg.primary_reason, "HEALTH_ALERT")

        # Severe score on Base Pipeline routes to REVIEW [ALERT], not EXCLUDE
        res_base = evaluate_training_decision(
            feature_name="base_feature",
            context_id="ctx_1",
            feature_source="base_pipeline",
            total_runs=5,
            unique_models_count=2,
            evidence_score=-65.0,
            consecutive_remove_count=2,
            dominant_recommendation="REMOVE",
            policy=pol,
        )
        self.assertEqual(res_base.decision, TrainingDecisionState.REVIEW)
        self.assertFalse(res_base.is_excluded)
        self.assertTrue(res_base.requires_review)
        self.assertIn("[ALERT]", res_base.reason_badges)

        # Same evidence on Experimental feature yields EXCLUDE
        res_exp = evaluate_training_decision(
            feature_name="exp_feature",
            context_id="ctx_1",
            feature_source="experimental",
            total_runs=5,
            unique_models_count=2,
            evidence_score=-65.0,
            consecutive_remove_count=2,
            dominant_recommendation="REMOVE",
            policy=pol,
        )
        self.assertEqual(res_exp.decision, TrainingDecisionState.EXCLUDE)
        self.assertTrue(res_exp.is_excluded)
        self.assertFalse(res_exp.is_candidate_generation_allowed)

    def test_n_less_than_3_volatility_na_invariant(self) -> None:
        """When N < 3, volatility is N/A and lack of volatility data must NOT cause REVIEW."""
        pol = RecommendationPolicy()

        # N = 2 runs with strong score and confidence -> TRAIN_CANDIDATE (not REVIEW due to lack of volatility)
        res = evaluate_training_decision(
            feature_name="feat_n2",
            context_id="ctx_1",
            feature_source="experimental",
            total_runs=2,
            unique_models_count=2,
            evidence_score=40.0,
            evidence_confidence=0.40,
            dominant_recommendation="KEEP",
            score_volatility=None,  # N/A
            policy=pol,
        )
        self.assertEqual(res.decision, TrainingDecisionState.TRAIN_CANDIDATE)
        self.assertTrue(res.is_training_candidate)
        self.assertFalse(res.requires_review)

    def test_k_less_than_2_generalization_na_invariant(self) -> None:
        """When K < 2, generalization is Single Context and lack of cross-context data must NOT cause REVIEW."""
        pol = RecommendationPolicy()

        # Single context feature -> TRAIN_CANDIDATE (not REVIEW due to single context)
        res = evaluate_training_decision(
            feature_name="feat_k1",
            context_id="ctx_1",
            feature_source="experimental",
            total_runs=4,
            unique_models_count=2,
            evidence_score=50.0,
            evidence_confidence=0.65,
            dominant_recommendation="KEEP",
            score_volatility=12.0,  # Stable
            generalization_score=None,  # Single context N/A
            policy=pol,
        )
        self.assertEqual(res.decision, TrainingDecisionState.TRAIN_CANDIDATE)
        self.assertTrue(res.is_training_candidate)

    def test_review_rules_precedence_and_triggers(self) -> None:
        """Verify individual review rules trigger correctly."""
        pol = RecommendationPolicy()

        # R2: Consensus Split
        res_split = evaluate_training_decision(
            feature_name="f_split",
            context_id="ctx_1",
            feature_source="experimental",
            total_runs=4,
            unique_models_count=2,
            evidence_score=35.0,
            evidence_confidence=0.50,
            is_consensus_tie=True,
            policy=pol,
        )
        self.assertEqual(res_split.decision, TrainingDecisionState.REVIEW)
        self.assertEqual(res_split.primary_reason, "CONSENSUS_SPLIT")
        self.assertIn("[SPLIT]", res_split.reason_badges)

        # R3: Dominant WATCH
        res_watch = evaluate_training_decision(
            feature_name="f_watch",
            context_id="ctx_1",
            feature_source="experimental",
            total_runs=4,
            unique_models_count=2,
            evidence_score=30.0,
            evidence_confidence=0.50,
            dominant_recommendation="WATCH",
            policy=pol,
        )
        self.assertEqual(res_watch.decision, TrainingDecisionState.REVIEW)
        self.assertEqual(res_watch.primary_reason, "DOMINANT_WATCH")

        # R4: Consensus REMOVE disagreement without hard block breach
        res_rem_disagree = evaluate_training_decision(
            feature_name="f_rem_disagree",
            context_id="ctx_1",
            feature_source="experimental",
            total_runs=2,
            unique_models_count=2,
            evidence_score=10.0,
            evidence_confidence=0.40,
            consecutive_remove_count=1,
            remove_runs=1,
            dominant_recommendation="REMOVE",
            policy=pol,
        )
        self.assertEqual(res_rem_disagree.decision, TrainingDecisionState.REVIEW)
        self.assertEqual(res_rem_disagree.primary_reason, "MODEL_DISAGREEMENT_REMOVE")

        # R5: Stale Evidence (> 30 days)
        res_stale = evaluate_training_decision(
            feature_name="f_stale",
            context_id="ctx_1",
            feature_source="experimental",
            total_runs=4,
            unique_models_count=2,
            evidence_score=45.0,
            evidence_confidence=0.60,
            dominant_recommendation="KEEP",
            freshness_label="Stale",
            policy=pol,
        )
        self.assertEqual(res_stale.decision, TrainingDecisionState.REVIEW)
        self.assertEqual(res_stale.primary_reason, "STALE_EVIDENCE")
        self.assertIn("[STALE]", res_stale.reason_badges)

        # R6: High Volatility (σ >= 35.0)
        res_vol = evaluate_training_decision(
            feature_name="f_vol",
            context_id="ctx_1",
            feature_source="experimental",
            total_runs=5,
            unique_models_count=2,
            evidence_score=40.0,
            evidence_confidence=0.60,
            dominant_recommendation="KEEP",
            score_volatility=42.5,
            policy=pol,
        )
        self.assertEqual(res_vol.decision, TrainingDecisionState.REVIEW)
        self.assertEqual(res_vol.primary_reason, "HIGH_VOLATILITY")
        self.assertIn("[UNSTABLE]", res_vol.reason_badges)

        # R7: Scale Specific Generalization (G < 0.25)
        res_gen = evaluate_training_decision(
            feature_name="f_gen",
            context_id="ctx_1",
            feature_source="experimental",
            total_runs=4,
            unique_models_count=2,
            evidence_score=40.0,
            evidence_confidence=0.60,
            dominant_recommendation="KEEP",
            score_volatility=15.0,
            generalization_score=0.18,
            policy=pol,
        )
        self.assertEqual(res_gen.decision, TrainingDecisionState.REVIEW)
        self.assertEqual(res_gen.primary_reason, "SCALE_SPECIFIC_DIVERGENCE")

    def test_promotion_candidate_handling(self) -> None:
        """Promotion candidate is non-mandatory for TRAIN_CANDIDATE, but receives top priority when qualified."""
        pol = RecommendationPolicy()

        # Non-promotion candidate feature with healthy evidence -> TRAIN_CANDIDATE
        res_regular = evaluate_training_decision(
            feature_name="feat_regular",
            context_id="ctx_1",
            feature_source="experimental",
            total_runs=3,
            unique_models_count=2,
            evidence_score=35.0,
            evidence_confidence=0.45,
            dominant_recommendation="KEEP",
            is_promotion_candidate=False,
            policy=pol,
        )
        self.assertEqual(res_regular.decision, TrainingDecisionState.TRAIN_CANDIDATE)
        self.assertEqual(res_regular.primary_reason, "TRAINING_CANDIDATE_ELIGIBLE")

        # Promotion candidate feature with healthy evidence -> TRAIN_CANDIDATE with PROMOTION badge
        res_promo = evaluate_training_decision(
            feature_name="feat_promo",
            context_id="ctx_1",
            feature_source="experimental",
            total_runs=5,
            unique_models_count=3,
            evidence_score=85.0,
            evidence_confidence=0.80,
            consecutive_keep_count=4,
            dominant_recommendation="KEEP",
            is_promotion_candidate=True,
            policy=pol,
        )
        self.assertEqual(res_promo.decision, TrainingDecisionState.TRAIN_CANDIDATE)
        self.assertEqual(res_promo.primary_reason, "PROMOTION_CANDIDATE_QUALIFIED")
        self.assertIn("[PROMOTION]", res_promo.reason_badges)

        # Promotion candidate with Stale flag routes to REVIEW (gated safety check)
        res_promo_stale = evaluate_training_decision(
            feature_name="feat_promo_stale",
            context_id="ctx_1",
            feature_source="experimental",
            total_runs=5,
            unique_models_count=3,
            evidence_score=85.0,
            evidence_confidence=0.80,
            consecutive_keep_count=4,
            dominant_recommendation="KEEP",
            freshness_label="Stale",
            is_promotion_candidate=True,
            policy=pol,
        )
        self.assertEqual(res_promo_stale.decision, TrainingDecisionState.REVIEW)
        self.assertEqual(res_promo_stale.primary_reason, "STALE_EVIDENCE")

    def test_policy_threshold_reactivity(self) -> None:
        """Policy threshold changes dynamically update training decisions."""
        # Policy A: Default min score +20.0
        pol_a = RecommendationPolicy()
        res_a = evaluate_training_decision(
            feature_name="feat_react",
            context_id="ctx_1",
            feature_source="experimental",
            total_runs=3,
            unique_models_count=2,
            evidence_score=35.0,
            evidence_confidence=0.50,
            dominant_recommendation="KEEP",
            policy=pol_a,
        )
        self.assertEqual(res_a.decision, TrainingDecisionState.TRAIN_CANDIDATE)

        # Policy B: Strict min score +50.0
        pol_b = RecommendationPolicy(
            training_decision=TrainingDecisionPolicy(train_candidate_min_score=50.0)
        )
        res_b = evaluate_training_decision(
            feature_name="feat_react",
            context_id="ctx_1",
            feature_source="experimental",
            total_runs=3,
            unique_models_count=2,
            evidence_score=35.0,
            evidence_confidence=0.50,
            dominant_recommendation="KEEP",
            policy=pol_b,
        )
        self.assertEqual(res_b.decision, TrainingDecisionState.REVIEW)
        self.assertEqual(res_b.primary_reason, "SCORE_BELOW_CANDIDATE_MIN")

    def test_context_isolation(self) -> None:
        """Decisions in Context A do not bleed into Context B."""
        pol = RecommendationPolicy()

        # In Context NIFTY 3s: Clean Candidate
        res_nifty = evaluate_training_decision(
            feature_name="shared_feat",
            context_id="ctx_nifty_3s",
            feature_source="experimental",
            total_runs=4,
            unique_models_count=2,
            evidence_score=60.0,
            evidence_confidence=0.70,
            dominant_recommendation="KEEP",
            policy=pol,
        )
        self.assertEqual(res_nifty.decision, TrainingDecisionState.TRAIN_CANDIDATE)

        # In Context SENSEX 1s: Blocked Exclude
        res_sensex = evaluate_training_decision(
            feature_name="shared_feat",
            context_id="ctx_sensex_1s",
            feature_source="experimental",
            total_runs=3,
            consecutive_remove_count=2,
            context_status="BLOCKED",
            evidence_score=-50.0,
            policy=pol,
        )
        self.assertEqual(res_sensex.decision, TrainingDecisionState.EXCLUDE)
        self.assertTrue(res_sensex.is_excluded)


class TestTrainingDecisionEngineBatchAndPersistence(unittest.TestCase):
    """Batch evaluation, SQLite Evidence DB integration, and immutability tests."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.data_dir = self.temp_dir.name
        get_connection(self.data_dir).close()

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_batch_candidate_evaluation_and_evidence_immutability(self) -> None:
        """Verify evaluate_candidate_training_eligibility and Evidence DB immutability."""
        ctx = build_dataset_context(
            market="NIFTY",
            sampling_interval_sec=3,
            sliding_window="standard",
            feature_project_id="fp_test",
        )

        # Record evidence for feat_clean (KEEP) and feat_blocked (2 REMOVEs)
        items = [
            {
                "feature_name": "feat_clean",
                "feature_source": "experimental",
                "pipeline_id": "PL_0001",
                "pipeline_snapshot_id": "snp_01",
                "model_name": "model_lgbm",
                "market": "NIFTY",
                "sampling_interval_sec": 3,
                "sliding_window": "standard",
                "feature_project_id": "fp_test",
                "recommendation": "KEEP",
                "permutation_importance": 0.05,
                "importance_drop_ratio": 0.0,
                "stability_score": 0.90,
                "run_timestamp": "2026-08-18T10:00:00Z",
            },
            {
                "feature_name": "feat_clean",
                "feature_source": "experimental",
                "pipeline_id": "PL_0001",
                "pipeline_snapshot_id": "snp_01",
                "model_name": "model_xgb",
                "market": "NIFTY",
                "sampling_interval_sec": 3,
                "sliding_window": "standard",
                "feature_project_id": "fp_test",
                "recommendation": "KEEP",
                "permutation_importance": 0.04,
                "importance_drop_ratio": 0.0,
                "stability_score": 0.85,
                "run_timestamp": "2026-08-18T11:00:00Z",
            },
            {
                "feature_name": "feat_blocked",
                "feature_source": "experimental",
                "pipeline_id": "PL_0002",
                "pipeline_snapshot_id": "snp_02",
                "model_name": "model_lgbm",
                "market": "NIFTY",
                "sampling_interval_sec": 3,
                "sliding_window": "standard",
                "feature_project_id": "fp_test",
                "recommendation": "REMOVE",
                "permutation_importance": -0.01,
                "importance_drop_ratio": 0.8,
                "stability_score": 0.20,
                "run_timestamp": "2026-08-18T10:00:00Z",
            },
            {
                "feature_name": "feat_blocked",
                "feature_source": "experimental",
                "pipeline_id": "PL_0002",
                "pipeline_snapshot_id": "snp_02",
                "model_name": "model_xgb",
                "market": "NIFTY",
                "sampling_interval_sec": 3,
                "sliding_window": "standard",
                "feature_project_id": "fp_test",
                "recommendation": "REMOVE",
                "permutation_importance": -0.02,
                "importance_drop_ratio": 0.9,
                "stability_score": 0.15,
                "run_timestamp": "2026-08-18T11:00:00Z",
            },
        ]
        conn = get_connection(self.data_dir)
        try:
            append_validation_evidence(conn, context=ctx, evidence_rows=items)
        finally:
            conn.close()

        db_path = os.path.join(self.data_dir, "feature_recommendation_evidence.db")
        sha_before = _file_sha256(db_path)

        # Batch evaluate candidate names: feat_clean, feat_blocked, and feat_unseen (N=0)
        candidates = ["feat_clean", "feat_blocked", "feat_unseen"]
        decisions = evaluate_candidate_training_eligibility(
            data_dir_or_conn=self.data_dir,
            context_id=ctx.context_id,
            candidate_names=candidates,
        )

        self.assertEqual(len(decisions), 3)

        # feat_clean -> TRAIN_CANDIDATE (2 runs, 2 models, score +50.0)
        self.assertEqual(decisions["feat_clean"].decision, TrainingDecisionState.TRAIN_CANDIDATE)
        self.assertTrue(decisions["feat_clean"].is_candidate_generation_allowed)
        self.assertTrue(decisions["feat_clean"].is_training_candidate)

        # feat_blocked -> EXCLUDE (2 consecutive REMOVEs -> score <= -40.0)
        self.assertEqual(decisions["feat_blocked"].decision, TrainingDecisionState.EXCLUDE)
        self.assertFalse(decisions["feat_blocked"].is_candidate_generation_allowed)
        self.assertTrue(decisions["feat_blocked"].is_excluded)

        # feat_unseen -> NEW_UNSEEN (0 runs)
        self.assertEqual(decisions["feat_unseen"].decision, TrainingDecisionState.NEW_UNSEEN)
        self.assertTrue(decisions["feat_unseen"].is_candidate_generation_allowed)
        self.assertFalse(decisions["feat_unseen"].is_training_candidate)

        # SHA-256 Checksum must be 100% identical (Zero Evidence DB Mutations)
        sha_after = _file_sha256(db_path)
        self.assertEqual(sha_before, sha_after, "Evidence DB was mutated during decision query!")

    def test_policy_storage_and_version_restoration(self) -> None:
        """Verify TrainingDecisionPolicy stores inside RecommendationPolicy without a secondary store."""
        pol1 = RecommendationPolicy(
            description="Base Policy v2",
            training_decision=TrainingDecisionPolicy(train_candidate_min_score=25.0),
        )
        saved1 = save_recommendation_policy(self.data_dir, pol1)
        self.assertEqual(saved1.policy_version, 2)
        self.assertEqual(saved1.training_decision.train_candidate_min_score, 25.0)

        # Save version 3 with updated training decision threshold
        pol2 = RecommendationPolicy(
            description="Policy v3 with higher threshold",
            training_decision=TrainingDecisionPolicy(train_candidate_min_score=45.0),
        )
        saved2 = save_recommendation_policy(self.data_dir, pol2)
        self.assertEqual(saved2.policy_version, 3)
        self.assertEqual(saved2.training_decision.train_candidate_min_score, 45.0)

        # Restore version 2 as version 4
        restored = restore_policy_version(self.data_dir, target_version=2)
        self.assertEqual(restored.policy_version, 4)
        self.assertEqual(restored.restored_from_version, 2)
        self.assertEqual(restored.training_decision.train_candidate_min_score, 25.0)


if __name__ == "__main__":
    unittest.main()

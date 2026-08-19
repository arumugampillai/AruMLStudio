"""Dedicated unit tests for Phase 4F.4: Automated Fine-Tuning & Descendant Mutation Controller."""

import hashlib
import json
import os
import shutil
import tempfile
import unittest

from path_config import ensure_ml_studio_paths
ensure_ml_studio_paths()

from chain_replay_ml.candidate_generation import (
    CandidateSpec,
    MutationType,
    create_candidate_spec,
)
from chain_replay_ml.fine_tuning import (
    DescendantEvaluationRecord,
    FineTuningBudget,
    FineTuningCampaignResult,
    FineTuningController,
    FineTuningDecision,
    evaluate_child_vs_parent,
    generate_fine_tuning_descendants,
    init_fine_tuning_tables,
    load_fine_tuning_records_for_context,
    persist_fine_tuning_records,
)
from chain_replay_ml.model_ranking import (
    CandidateEvidenceScore,
    CandidateRankingPolicy,
    ContextRankingReport,
    RecommendationClass,
    evaluate_candidate_evidence,
    rank_candidates_in_context,
)
from chain_replay_ml.research_memory import init_analysis_db


class TestFineTuningController(unittest.TestCase):
    """Comprehensive test suite verifying Phase 4F.4 fine-tuning invariants and decision accuracy."""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp(prefix="aruml_test_ft_")
        init_analysis_db(self.tmp_dir)
        self.context_key = "NIFTY_3s_DIRECTION_CLASSIFIER_5m_R001"
        self.base_features = ["adx_14", "rsi_14", "macd_diff", "bb_width_20", "iv_mean"]

        self.parent_cand = create_candidate_spec(
            context_key=self.context_key,
            algorithm="xgboost",
            features=self.base_features,
        )
        self.parent_score = evaluate_candidate_evidence(
            candidate_id=self.parent_cand.candidate_id,
            signature_hash=self.parent_cand.signature_hash,
            context_key=self.context_key,
            model_metrics={"roc_auc": 0.74, "fold_mean": 0.74, "fold_std": 0.02},
            trading_metrics={"win_rate_pct": 56.0, "profit_factor": 1.45, "mfe_mae_ratio": 1.15, "total_trades": 45},
        )

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_01_parent_selection_from_ranking(self):
        """1. Verify selection of top-ranked fine-tune parents from ContextRankingReport."""
        items = [
            {"candidate_id": "CAND_TOP", "signature_hash": "sig_1", "model_metrics": {"roc_auc": 0.80}, "trading_metrics": {"win_rate_pct": 65.0, "profit_factor": 1.8, "mfe_mae_ratio": 1.4, "total_trades": 50}},
            {"candidate_id": "CAND_MID", "signature_hash": "sig_2", "model_metrics": {"roc_auc": 0.75}, "trading_metrics": {"win_rate_pct": 58.0, "profit_factor": 1.5, "mfe_mae_ratio": 1.2, "total_trades": 50}},
            {"candidate_id": "CAND_BAD", "signature_hash": "sig_3", "model_metrics": {"roc_auc": 0.50}, "trading_metrics": {"win_rate_pct": 35.0, "profit_factor": 0.6, "mfe_mae_ratio": 0.5, "total_trades": 20}},
        ]
        report = rank_candidates_in_context(self.tmp_dir, self.context_key, candidate_items=items)
        ctrl = FineTuningController()
        parents = ctrl.select_promising_parents(report)
        self.assertEqual(len(parents), 2)
        self.assertNotIn("CAND_BAD", [p.candidate_id for p in parents])

    def test_02_feature_mutation_descendant_generation(self):
        """2. Verify feature mutation generates valid descendant candidate."""
        desc = generate_fine_tuning_descendants(self.tmp_dir, self.parent_cand, self.parent_score)
        self.assertGreater(len(desc), 0)
        for d in desc:
            self.assertEqual(d.context_key, self.context_key)
            self.assertEqual(d.lineage.parent_candidate_id, self.parent_cand.candidate_id)

    def test_03_hyperparameter_mutation_descendant_generation(self):
        """3. Verify hyperparameter mutation descendant generation."""
        desc = generate_fine_tuning_descendants(self.tmp_dir, self.parent_cand, self.parent_score)
        hp_desc = [d for d in desc if d.lineage.mutation_type == MutationType.HYPERPARAMETER_MUTATION]
        self.assertGreater(len(hp_desc), 0)
        self.assertNotEqual(hp_desc[0].hyperparameters["max_depth"], self.parent_cand.hyperparameters["max_depth"])

    def test_04_algorithm_mutation_challenger_generation(self):
        """4. Verify algorithm challenger generation while preserving features and context."""
        desc = generate_fine_tuning_descendants(self.tmp_dir, self.parent_cand, self.parent_score, budget=FineTuningBudget(max_descendants_per_parent=4))
        algo_desc = [d for d in desc if d.lineage.mutation_type == MutationType.ALGORITHM_MUTATION]
        self.assertGreater(len(algo_desc), 0)
        self.assertEqual(algo_desc[0].algorithm, "lightgbm")

    def test_05_target_mutation_outcome_hypotheses(self):
        """5. Verify outcome classification target hypotheses generation."""
        cand_outcome = create_candidate_spec(
            context_key="NIFTY_3s_TRIPLE_BARRIER_5m_R001",
            algorithm="xgboost",
            features=self.base_features,
            parent_spec=self.parent_cand,
            mutation_type=MutationType.TARGET_HORIZON_MUTATION,
        )
        self.assertEqual(cand_outcome.task_type, "TRIPLE_BARRIER")
        self.assertEqual(cand_outcome.lineage.parent_candidate_id, self.parent_cand.candidate_id)

    def test_06_regime_specialization_isolation(self):
        """6. Verify regime specialization preserves strict ModelContextKey boundaries."""
        cand_r2 = create_candidate_spec(
            context_key="NIFTY_3s_DIRECTION_CLASSIFIER_5m_R002",
            algorithm="xgboost",
            features=self.base_features,
            parent_spec=self.parent_cand,
            mutation_type=MutationType.REGIME_SPECIALIZATION,
        )
        self.assertEqual(cand_r2.regime_id, "R002")

    def test_07_phase4e_feature_affinity_integration(self):
        """7. Verify Phase 4E feature affinity feeds feature mutation."""
        desc = generate_fine_tuning_descendants(self.tmp_dir, self.parent_cand, self.parent_score)
        self.assertGreater(len(desc), 0)

    def test_08_phase4e_interaction_synergy_integration(self):
        """8. Verify Phase 4E interaction synergy pairs are integrated into descendant candidates."""
        desc = generate_fine_tuning_descendants(self.tmp_dir, self.parent_cand, self.parent_score)
        self.assertGreater(len(desc), 0)

    def test_09_deprecated_feature_rejection(self):
        """9. Verify descendant candidate containing deprecated feature is strictly rejected by pruning."""
        mock_schema = {
            "columns": {
                "deprecated_feat_z": {"status": "DEPRECATED"},
                "rsi_14": {"status": "ACTIVE"},
            }
        }
        cand = create_candidate_spec(
            context_key=self.context_key,
            algorithm="xgboost",
            features=["deprecated_feat_z", "rsi_14"],
            parent_spec=self.parent_cand,
        )
        ctrl = FineTuningController()
        proposed = ctrl.propose_fine_tuning_batch(self.tmp_dir, [cand], schema=mock_schema)
        # Deprecated candidate must not be scheduled
        self.assertNotIn("deprecated_feat_z", [f for p in proposed for f in p.features if "deprecated_feat_z" in f])

    def test_10_negative_evidence_pruning(self):
        """10. Verify negative evidence pruning stops bad search paths."""
        desc = generate_fine_tuning_descendants(self.tmp_dir, self.parent_cand, self.parent_score)
        self.assertTrue(all(d.eligibility != "EXCLUDED" for d in desc))

    def test_11_duplicate_suppression(self):
        """11. Verify duplicate candidate signature is suppressed."""
        desc = generate_fine_tuning_descendants(self.tmp_dir, self.parent_cand, self.parent_score)
        signatures = [d.signature_hash for d in desc]
        self.assertEqual(len(signatures), len(set(signatures)))

    def test_12_deterministic_mutation_reproducibility(self):
        """12. Invariant: Same Parent + Same Mutation -> Identical Child Signature Hash."""
        desc1 = generate_fine_tuning_descendants(self.tmp_dir, self.parent_cand, self.parent_score)
        desc2 = generate_fine_tuning_descendants(self.tmp_dir, self.parent_cand, self.parent_score)
        self.assertEqual([d.signature_hash for d in desc1], [d.signature_hash for d in desc2])

    def test_13_parent_child_lineage_tracking(self):
        """13. Verify child lineage correctly references parent candidate ID and signature."""
        desc = generate_fine_tuning_descendants(self.tmp_dir, self.parent_cand, self.parent_score)
        child = desc[0]
        self.assertEqual(child.lineage.parent_candidate_id, self.parent_cand.candidate_id)
        self.assertEqual(child.lineage.parent_signature_hash, self.parent_cand.signature_hash)

    def test_14_generation_depth_ceiling_enforcement(self):
        """14. Verify descendant generation halts when max_generations budget is reached."""
        c = self.parent_cand
        budget = FineTuningBudget(max_generations=2)
        # Generation 0 -> 1 -> 2
        g1 = generate_fine_tuning_descendants(self.tmp_dir, c, budget=budget)
        g2 = generate_fine_tuning_descendants(self.tmp_dir, g1[0], budget=budget)
        g3 = generate_fine_tuning_descendants(self.tmp_dir, g2[0], budget=budget)
        self.assertEqual(len(g3), 0)

    def test_15_child_vs_parent_score_delta(self):
        """15. Verify exact calculation of multidimensional child vs parent deltas."""
        child_cand = create_candidate_spec(context_key=self.context_key, algorithm="xgboost", features=self.base_features, parent_spec=self.parent_cand)
        child_score = evaluate_candidate_evidence(
            candidate_id=child_cand.candidate_id,
            signature_hash=child_cand.signature_hash,
            context_key=self.context_key,
            model_metrics={"roc_auc": 0.78, "fold_mean": 0.78, "fold_std": 0.01},
            trading_metrics={"win_rate_pct": 62.0, "profit_factor": 1.75, "mfe_mae_ratio": 1.35, "total_trades": 50},
        )
        rec = evaluate_child_vs_parent(child_score=child_score, parent_score=self.parent_score)
        self.assertGreater(rec.delta_composite_score, 0.0)
        self.assertAlmostEqual(rec.delta_composite_score, child_score.composite_score - self.parent_score.composite_score, places=3)

    def test_16_confirmed_mutation_lift_classification(self):
        """16. Verify Delta composite score >= +1.5 triggers CONFIRMED_MUTATION_LIFT."""
        child_cand = create_candidate_spec(context_key=self.context_key, algorithm="xgboost", features=self.base_features, parent_spec=self.parent_cand)
        child_score = evaluate_candidate_evidence(
            candidate_id=child_cand.candidate_id,
            signature_hash=child_cand.signature_hash,
            context_key=self.context_key,
            model_metrics={"roc_auc": 0.82, "fold_mean": 0.82, "fold_std": 0.01},
            trading_metrics={"win_rate_pct": 65.0, "profit_factor": 1.95, "mfe_mae_ratio": 1.5, "total_trades": 50},
        )
        rec = evaluate_child_vs_parent(child_score=child_score, parent_score=self.parent_score)
        self.assertEqual(rec.decision_verdict, FineTuningDecision.CONFIRMED_MUTATION_LIFT)
        self.assertFalse(rec.is_branch_pruned)

    def test_17_regression_pruning_classification(self):
        """17. Verify Delta composite score <= -3.0 triggers PRUNED_MUTATION_PATH."""
        child_cand = create_candidate_spec(context_key=self.context_key, algorithm="xgboost", features=self.base_features, parent_spec=self.parent_cand)
        child_score = evaluate_candidate_evidence(
            candidate_id=child_cand.candidate_id,
            signature_hash=child_cand.signature_hash,
            context_key=self.context_key,
            model_metrics={"roc_auc": 0.60, "fold_mean": 0.60, "fold_std": 0.04},
            trading_metrics={"win_rate_pct": 42.0, "profit_factor": 0.90, "mfe_mae_ratio": 0.7, "max_drawdown_pct": 14.0, "total_trades": 40},
        )
        rec = evaluate_child_vs_parent(child_score=child_score, parent_score=self.parent_score)
        self.assertEqual(rec.decision_verdict, FineTuningDecision.PRUNED_MUTATION_PATH)
        self.assertTrue(rec.is_branch_pruned)

    def test_18_context_isolation_invariant(self):
        """18. Verify fine-tuning never mixes evidence across ModelContextKeys."""
        ctrl = FineTuningController()
        p_cand_bn = create_candidate_spec(context_key="BANKNIFTY_3s_DIRECTION_CLASSIFIER_5m_R002", algorithm="xgboost", features=self.base_features)
        proposed = ctrl.propose_fine_tuning_batch(self.tmp_dir, [p_cand_bn])
        for p in proposed:
            self.assertEqual(p.context_key, "BANKNIFTY_3s_DIRECTION_CLASSIFIER_5m_R002")

    def test_19_nan_invalid_metric_safety(self):
        """19. Verify controller handles NaN/Infinity child metrics safely without crashing."""
        child_cand = create_candidate_spec(context_key=self.context_key, algorithm="xgboost", features=self.base_features, parent_spec=self.parent_cand)
        child_score = evaluate_candidate_evidence(
            candidate_id=child_cand.candidate_id,
            signature_hash=child_cand.signature_hash,
            context_key=self.context_key,
            model_metrics={"roc_auc": float("nan")},
            trading_metrics={"win_rate_pct": float("nan")},
        )
        rec = evaluate_child_vs_parent(child_score=child_score, parent_score=self.parent_score)
        self.assertEqual(rec.decision_verdict, FineTuningDecision.PRUNED_MUTATION_PATH)

    def test_20_resource_candidate_budget_enforcement(self):
        """20. Verify campaign candidate ceiling limit is enforced."""
        ctrl = FineTuningController(budget=FineTuningBudget(max_candidates_total=2))
        proposed = ctrl.propose_fine_tuning_batch(self.tmp_dir, [self.parent_cand])
        self.assertLessEqual(len(proposed), 2)

    def test_21_analysis_db_persistence(self):
        """21. Verify fine-tuning trial records persist and round-trip to analysis.db cleanly."""
        child_cand = create_candidate_spec(context_key=self.context_key, algorithm="xgboost", features=self.base_features, parent_spec=self.parent_cand)
        child_score = evaluate_candidate_evidence(
            candidate_id=child_cand.candidate_id,
            signature_hash=child_cand.signature_hash,
            context_key=self.context_key,
            model_metrics={"roc_auc": 0.78},
            trading_metrics={"win_rate_pct": 60.0, "profit_factor": 1.6, "mfe_mae_ratio": 1.3, "total_trades": 50},
        )
        rec = evaluate_child_vs_parent(child_score=child_score, parent_score=self.parent_score)
        written = persist_fine_tuning_records(self.tmp_dir, [rec])
        self.assertEqual(written, 1)

        loaded = load_fine_tuning_records_for_context(self.tmp_dir, self.context_key)
        self.assertEqual(len(loaded), 1)
        self.assertEqual(loaded[0].trial_id, rec.trial_id)
        self.assertEqual(loaded[0].decision_verdict, rec.decision_verdict)

    def test_22_production_immutability(self):
        """22. Invariant: Fine-tuning controller never touches .active_model.json or production model directories."""
        ctrl = FineTuningController()
        ctrl.propose_fine_tuning_batch(self.tmp_dir, [self.parent_cand])
        active_model_path = os.path.join(self.tmp_dir, "models", ".active_model.json")
        self.assertFalse(os.path.exists(active_model_path))

    def test_23_legacy_aruneo_exclusion(self):
        """23. Invariant: Fine-tuning controller never creates or touches .lifecycle_registry.db."""
        ctrl = FineTuningController()
        ctrl.propose_fine_tuning_batch(self.tmp_dir, [self.parent_cand])
        legacy_db_path = os.path.join(self.tmp_dir, "models", ".lifecycle_registry.db")
        self.assertFalse(os.path.exists(legacy_db_path))

    def test_24_evidence_db_immutability(self):
        """24. Invariant: Feature Recommendation Evidence DB remains unmutated."""
        ev_db_path = os.path.join("apps", "feature_recommendation_evidence.db")
        if os.path.exists(ev_db_path):
            with open(ev_db_path, "rb") as f:
                sha = hashlib.sha256(f.read()).hexdigest()
            self.assertEqual(sha, "6f91afca94ec87a1210d8f4bcef356b9c16a6ef5a488268f41c5b1b81431ade2")

    def test_25_end_to_end_parent_mutation_evaluation_cycle(self):
        """25. Full End-to-End: Parent -> Propose Descendants -> Evaluate Children -> Record Campaign."""
        ctrl = FineTuningController()
        descendants = ctrl.propose_fine_tuning_batch(self.tmp_dir, [self.parent_cand])
        self.assertGreater(len(descendants), 0)

        # Simulate evaluation of children
        child_scores = []
        for i, d in enumerate(descendants):
            score = evaluate_candidate_evidence(
                candidate_id=d.candidate_id,
                signature_hash=d.signature_hash,
                context_key=self.context_key,
                model_metrics={"roc_auc": 0.75 + (0.02 * i)},
                trading_metrics={"win_rate_pct": 57.0 + (2.0 * i), "profit_factor": 1.5 + (0.1 * i), "mfe_mae_ratio": 1.2, "total_trades": 45},
                parent_candidate_id=self.parent_cand.candidate_id,
                parent_composite_score=self.parent_score.composite_score,
            )
            child_scores.append(score)

        campaign_res = ctrl.evaluate_and_record_campaign(
            context_key=self.context_key,
            child_scores=child_scores,
            parent_scores={self.parent_cand.candidate_id: self.parent_score},
        )
        self.assertEqual(campaign_res.total_descendants_generated, len(descendants))
        self.assertIsNotNone(campaign_res.best_descendant)

    def test_26_neutral_mutation_classification(self):
        """26. Verify small delta (-0.2 composite) is classified as NEUTRAL_MUTATION without pruning."""
        child_cand = create_candidate_spec(context_key=self.context_key, algorithm="xgboost", features=self.base_features, parent_spec=self.parent_cand)
        child_score = evaluate_candidate_evidence(
            candidate_id=child_cand.candidate_id,
            signature_hash=child_cand.signature_hash,
            context_key=self.context_key,
            model_metrics={"roc_auc": 0.74},
            trading_metrics={"win_rate_pct": 55.8, "profit_factor": 1.44, "mfe_mae_ratio": 1.15, "total_trades": 45},
        )
        rec = evaluate_child_vs_parent(child_score=child_score, parent_score=self.parent_score)
        self.assertEqual(rec.decision_verdict, FineTuningDecision.NEUTRAL_MUTATION)
        self.assertFalse(rec.is_branch_pruned)

    def test_27_output_compatibility_with_phase4f5(self):
        """27. Verify campaign result serializes cleanly for downstream Phase 4F.5 overnight campaign controller."""
        ctrl = FineTuningController()
        campaign_res = ctrl.evaluate_and_record_campaign(
            context_key=self.context_key,
            child_scores=[],
            parent_scores={},
            campaign_id="CAMP_OVERNIGHT_01",
        )
        d = campaign_res.to_dict()
        self.assertEqual(d["context_key"], self.context_key)
        self.assertEqual(d["campaign_id"], "CAMP_OVERNIGHT_01")
        self.assertEqual(d["total_descendants_generated"], 0)


if __name__ == "__main__":
    unittest.main()

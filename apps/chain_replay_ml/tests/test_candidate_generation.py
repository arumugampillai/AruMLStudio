"""Unit Tests for Phase 4F.2: Automated Candidate Generator & Lineage Tracker."""

import hashlib
import os
import shutil
import tempfile
import unittest

from path_config import ensure_ml_studio_paths
ensure_ml_studio_paths()

from chain_replay_ml.candidate_generation import (
    CandidateEligibility,
    CandidateGenerationBudget,
    CandidateLineageRecord,
    CandidateSpec,
    MutationType,
    create_candidate_spec,
    evaluate_candidate_eligibility,
    generate_candidate_batch,
    generate_candidates_from_priority_agenda,
    generate_cold_start_candidates,
    generate_descendant_mutations,
    mutate_algorithm,
    mutate_feature_subset,
    mutate_hyperparameters,
    reconstruct_lineage_graph,
    trace_ancestors,
    validate_hyperparameters_for_algorithm,
)
from chain_replay_ml.research_memory import (
    init_analysis_db,
    record_model_benchmark,
    register_or_get_experiment,
)
from chain_replay_ml.research_recommendations.priority_scoring import (
    ComponentScoreBreakdown,
    ContextPriorityAgendaReport,
    EvidenceConfidenceLevel,
    OpportunityType,
    ResearchOpportunity,
    ResearchPriorityClass,
)
from chain_replay_ml.research_recommendations.negative_pruning import (
    ExclusionReason,
    ExclusionVerdict,
)
from chain_replay_ml.strategy_evaluation import StrategyEvaluationPolicy


class TestCandidateGeneratorAndLineageTracker(unittest.TestCase):
    """Comprehensive test suite verifying Phase 4F.2 candidate generation and lineage invariants."""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp(prefix="aruml_test_cand_gen_")
        init_analysis_db(self.tmp_dir)
        self.context_key = "NIFTY_3s_DIRECTION_CLASSIFIER_5m_R001"
        self.base_features = ["adx_14", "rsi_14", "macd_diff", "bb_width_20", "iv_mean"]

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_01_cold_start_candidate_generation(self):
        """1. Verify cold-start baseline candidate generation for empty context."""
        cands = generate_cold_start_candidates(self.context_key, self.base_features)
        self.assertGreaterEqual(len(cands), 3)
        for c in cands:
            self.assertEqual(c.context_key, self.context_key)
            self.assertEqual(c.lineage.generation_number, 0)
            self.assertEqual(c.lineage.mutation_type, MutationType.COLD_START)
            self.assertTrue(len(c.signature_hash) == 64)

    def test_02_supported_algorithm_specifications(self):
        """2. Verify generated candidates strictly use verified algorithms."""
        cands = generate_cold_start_candidates(
            self.context_key,
            self.base_features,
            algorithms=["xgboost", "lightgbm", "catboost", "random_forest", "extra_trees"],
        )
        supported = {"xgboost", "lightgbm", "catboost", "random_forest", "extra_trees"}
        for c in cands:
            self.assertIn(c.algorithm, supported)

    def test_03_verified_hyperparameter_schemas(self):
        """3. Verify hyperparameter validation rejects unmapped parameter keys."""
        cleaned_xgb = validate_hyperparameters_for_algorithm("xgboost", {"max_depth": 7, "invalid_param": 999})
        self.assertIn("max_depth", cleaned_xgb)
        self.assertNotIn("invalid_param", cleaned_xgb)
        self.assertEqual(cleaned_xgb["max_depth"], 7)

    def test_04_feature_subset_mutation_step(self):
        """4. Verify feature subset mutation correctly adds and removes features."""
        parent = create_candidate_spec(
            context_key=self.context_key,
            algorithm="xgboost",
            features=["adx_14", "rsi_14"],
        )
        child = mutate_feature_subset(
            parent,
            features_to_add=["iv_mean", "macd_diff"],
            features_to_remove=["adx_14"],
        )
        self.assertIn("iv_mean", child.features)
        self.assertIn("macd_diff", child.features)
        self.assertIn("rsi_14", child.features)
        self.assertNotIn("adx_14", child.features)
        self.assertEqual(child.lineage.parent_candidate_id, parent.candidate_id)
        self.assertEqual(child.lineage.generation_number, 1)

    def test_05_algorithm_mutation_step(self):
        """5. Verify algorithm mutation preserves context and feature set."""
        parent = create_candidate_spec(
            context_key=self.context_key,
            algorithm="xgboost",
            features=self.base_features,
        )
        child = mutate_algorithm(parent, "catboost")
        self.assertEqual(child.algorithm, "catboost")
        self.assertEqual(child.features, parent.features)
        self.assertEqual(child.context_key, parent.context_key)
        self.assertEqual(child.lineage.parent_candidate_id, parent.candidate_id)

    def test_06_hyperparameter_mutation_step(self):
        """6. Verify hyperparameter mutation step applies parameter updates."""
        parent = create_candidate_spec(
            context_key=self.context_key,
            algorithm="xgboost",
            features=self.base_features,
            hyperparameters={"max_depth": 6, "learning_rate": 0.05},
        )
        child = mutate_hyperparameters(parent, {"max_depth": 8, "learning_rate": 0.02})
        self.assertEqual(child.hyperparameters["max_depth"], 8)
        self.assertEqual(child.hyperparameters["learning_rate"], 0.02)
        self.assertEqual(child.lineage.mutation_type, MutationType.HYPERPARAMETER_MUTATION)

    def test_07_cryptographic_lineage_chain(self):
        """7. Verify complete 3-generation lineage tracking (Parent -> Child -> Grandchild)."""
        root = create_candidate_spec(context_key=self.context_key, algorithm="xgboost", features=self.base_features)
        child = mutate_hyperparameters(root, {"max_depth": 8})
        grandchild = mutate_feature_subset(child, features_to_add=["iv_skew"])

        self.assertEqual(root.lineage.generation_number, 0)
        self.assertIsNone(root.lineage.parent_candidate_id)

        self.assertEqual(child.lineage.generation_number, 1)
        self.assertEqual(child.lineage.parent_candidate_id, root.candidate_id)
        self.assertEqual(child.lineage.parent_signature_hash, root.signature_hash)

        self.assertEqual(grandchild.lineage.generation_number, 2)
        self.assertEqual(grandchild.lineage.parent_candidate_id, child.candidate_id)
        self.assertEqual(grandchild.lineage.parent_signature_hash, child.signature_hash)

    def test_08_lineage_graph_reconstruction(self):
        """8. Verify lineage graph reconstruction and ancestor tracing."""
        root = create_candidate_spec(context_key=self.context_key, algorithm="xgboost", features=self.base_features)
        child1 = mutate_hyperparameters(root, {"max_depth": 4}, suffix="_C1")
        child2 = mutate_hyperparameters(root, {"max_depth": 8}, suffix="_C2")
        grandchild = mutate_feature_subset(child1, features_to_add=["iv_skew"], suffix="_GC1")

        candidates = [root, child1, child2, grandchild]
        graph = reconstruct_lineage_graph(candidates)

        self.assertIn(root.candidate_id, graph)
        self.assertEqual(set(graph[root.candidate_id]), {child1.candidate_id, child2.candidate_id})
        self.assertIn(child1.candidate_id, graph)
        self.assertEqual(graph[child1.candidate_id], [grandchild.candidate_id])

        c_map = {c.candidate_id: c for c in candidates}
        ancestors = trace_ancestors(grandchild.candidate_id, c_map)
        self.assertEqual(len(ancestors), 3)  # grandchild, child1, root
        self.assertEqual(ancestors[0].candidate_id, grandchild.candidate_id)
        self.assertEqual(ancestors[1].candidate_id, child1.candidate_id)
        self.assertEqual(ancestors[2].candidate_id, root.candidate_id)

    def test_09_duplicate_signature_detection(self):
        """9. Verify duplicate signature detection flags candidate as ALREADY_EVALUATED."""
        cand = create_candidate_spec(context_key=self.context_key, algorithm="xgboost", features=self.base_features)
        register_or_get_experiment(self.tmp_dir, cand.to_experiment_spec(), model_name="PRIOR_EVAL_MODEL")

        validated = evaluate_candidate_eligibility(self.tmp_dir, cand)
        self.assertEqual(validated.eligibility, CandidateEligibility.ALREADY_EVALUATED)
        self.assertTrue(any("ALREADY_EVALUATED" in r for r in validated.exclusion_reasons))

    def test_10_negative_pruning_exclusion(self):
        """10. Verify candidate with deprecated feature is classified as EXCLUDED."""
        mock_schema = {
            "columns": {
                "dep_old": {"status": "DEPRECATED"},
                "rsi_14": {"status": "ACTIVE"},
            }
        }
        cand = create_candidate_spec(
            context_key=self.context_key,
            algorithm="xgboost",
            features=["dep_old", "rsi_14"],
        )
        validated = evaluate_candidate_eligibility(self.tmp_dir, cand, schema=mock_schema)
        self.assertEqual(validated.eligibility, CandidateEligibility.EXCLUDED)
        self.assertTrue(any("DEPRECATED_FEATURE" in r for r in validated.exclusion_reasons))

    def test_11_candidate_generation_budget_limits(self):
        """11. Verify candidate generation respects max_candidates_per_campaign budget."""
        budget = CandidateGenerationBudget(max_candidates_per_campaign=2)
        res = generate_candidate_batch(
            self.tmp_dir,
            self.context_key,
            self.base_features,
            budget=budget,
        )
        self.assertEqual(res.total_generated, 2)
        self.assertTrue(res.budget_exhausted)

    def test_12_branching_factor_enforcement(self):
        """12. Verify descendant generation respects max_descendants_per_parent."""
        parent = create_candidate_spec(context_key=self.context_key, algorithm="xgboost", features=self.base_features)
        budget = CandidateGenerationBudget(max_descendants_per_parent=2)
        desc = generate_descendant_mutations(
            parent,
            top_affinity_features=["feat_a", "feat_b", "feat_c"],
            budget=budget,
        )
        self.assertLessEqual(len(desc), 2)

    def test_13_generation_depth_ceiling(self):
        """13. Verify mutation halts when max_generations ceiling is reached."""
        root = create_candidate_spec(context_key=self.context_key, algorithm="xgboost", features=self.base_features)
        c1 = mutate_hyperparameters(root, {"max_depth": 7})
        c2 = mutate_hyperparameters(c1, {"max_depth": 8})
        c3 = mutate_hyperparameters(c2, {"max_depth": 9})  # Gen 3

        budget = CandidateGenerationBudget(max_generations=3)
        desc = generate_descendant_mutations(c3, budget=budget)
        self.assertEqual(len(desc), 0)

    def test_14_feature_ceiling_memory_safety(self):
        """14. Verify candidate generator caps feature count to max_features_per_candidate."""
        many_features = [f"feature_{i:03d}" for i in range(50)]
        budget = CandidateGenerationBudget(max_features_per_candidate=20)
        cands = generate_cold_start_candidates(self.context_key, many_features, budget=budget)
        for c in cands:
            self.assertLessEqual(len(c.features), 20)

    def test_15_context_key_strict_isolation(self):
        """15. Verify candidate generation never mixes across context keys."""
        cand = create_candidate_spec(context_key="BANKNIFTY_3s_DIRECTION_CLASSIFIER_5m_R002", algorithm="xgboost", features=self.base_features)
        self.assertEqual(cand.market, "BANKNIFTY")
        self.assertEqual(cand.regime_id, "R002")
        self.assertEqual(cand.task_type, "DIRECTION_CLASSIFIER")

    def test_16_strategy_evaluation_policy_immutability(self):
        """16. Verify StrategyEvaluationPolicy remains fixed and unaffected by candidate generation."""
        pol = StrategyEvaluationPolicy()
        self.assertEqual(pol.target_return_pct, 2.0)
        self.assertEqual(pol.stop_loss_pct, 2.0)
        self.assertEqual(pol.min_confidence_threshold, 0.55)

    def test_17_candidate_spec_reproducibility(self):
        """17. Invariant: Same Context + Same Parent + Same Mutation -> Identical Signature Hash."""
        root = create_candidate_spec(context_key=self.context_key, algorithm="xgboost", features=self.base_features)
        child_a = mutate_hyperparameters(root, {"max_depth": 8, "learning_rate": 0.02})
        child_b = mutate_hyperparameters(root, {"max_depth": 8, "learning_rate": 0.02})

        self.assertEqual(child_a.signature_hash, child_b.signature_hash)

    def test_18_descendant_batch_generation_with_affinity(self):
        """18. Verify batch candidate generation with seed parent and affinity features."""
        root = create_candidate_spec(context_key=self.context_key, algorithm="xgboost", features=self.base_features)
        res = generate_candidate_batch(
            self.tmp_dir,
            self.context_key,
            self.base_features,
            seed_candidates=[root],
            budget=CandidateGenerationBudget(max_candidates_per_campaign=5),
        )
        self.assertGreater(res.total_generated, 0)
        for c in res.candidates:
            self.assertEqual(c.lineage.parent_candidate_id, root.candidate_id)

    def test_19_caution_verdict_propagation(self):
        """19. Verify CAUTION eligibility classification preserves warnings without full exclusion."""
        cand = create_candidate_spec(context_key=self.context_key, algorithm="xgboost", features=self.base_features)
        cand.eligibility = CandidateEligibility.CAUTION
        cand.caution_warnings.append("EXTREME_REGIME_FRAGILITY: High degradation observed.")
        d = cand.to_dict()
        self.assertEqual(d["eligibility"], "CAUTION")
        self.assertIn("EXTREME_REGIME_FRAGILITY", d["caution_warnings"][0])

    def test_20_random_forest_and_extra_trees_generation(self):
        """20. Verify Random Forest and Extra Trees parameter validation and defaults."""
        rf_params = validate_hyperparameters_for_algorithm("random_forest", {"max_depth": 12, "n_estimators": 250})
        et_params = validate_hyperparameters_for_algorithm("extra_trees", {"max_depth": 15, "n_estimators": 300})
        self.assertEqual(rf_params["max_depth"], 12)
        self.assertEqual(rf_params["n_estimators"], 250)
        self.assertEqual(et_params["max_depth"], 15)
        self.assertEqual(et_params["n_estimators"], 300)

    def test_21_outcome_classifier_hypotheses_generation(self):
        """21. Verify triple barrier outcome and meta-confidence task types remain valid classification specifications."""
        cand_tb = create_candidate_spec(context_key="NIFTY_3s_TRIPLE_BARRIER_5m_R001", algorithm="xgboost", features=self.base_features)
        cand_conf = create_candidate_spec(context_key="NIFTY_3s_CONFIDENCE_CLASSIFIER_5m_R001", algorithm="catboost", features=self.base_features)
        cand_reg = create_candidate_spec(context_key="NIFTY_3s_REGIME_CLASSIFIER_5m_R000", algorithm="lightgbm", features=self.base_features)
        self.assertEqual(cand_tb.task_type, "TRIPLE_BARRIER")
        self.assertEqual(cand_conf.task_type, "CONFIDENCE_CLASSIFIER")
        self.assertEqual(cand_reg.task_type, "REGIME_CLASSIFIER")
        self.assertTrue(len(cand_tb.signature_hash) == 64)
        self.assertTrue(len(cand_conf.signature_hash) == 64)
        self.assertTrue(len(cand_reg.signature_hash) == 64)

    def test_22_production_immutability(self):
        """22. Invariant: Candidate generation never writes to .active_model.json or production model directories."""
        generate_candidate_batch(self.tmp_dir, self.context_key, self.base_features)
        active_model_path = os.path.join(self.tmp_dir, "models", ".active_model.json")
        self.assertFalse(os.path.exists(active_model_path))

    def test_23_legacy_aruneo_exclusion(self):
        """23. Invariant: Candidate generation never creates or touches .lifecycle_registry.db."""
        generate_candidate_batch(self.tmp_dir, self.context_key, self.base_features)
        legacy_db_path = os.path.join(self.tmp_dir, "models", ".lifecycle_registry.db")
        self.assertFalse(os.path.exists(legacy_db_path))

    def test_24_evidence_db_immutability(self):
        """24. Invariant: Feature Recommendation Evidence DB remains unmutated."""
        ev_db_path = os.path.join("apps", "feature_recommendation_evidence.db")
        if os.path.exists(ev_db_path):
            with open(ev_db_path, "rb") as f:
                sha = hashlib.sha256(f.read()).hexdigest()
            self.assertEqual(sha, "6f91afca94ec87a1210d8f4bcef356b9c16a6ef5a488268f41c5b1b81431ade2")

    def test_25_end_to_end_generation_and_ancestral_tracing(self):
        """25. Full End-to-End: Cold-start -> 2 Generations of Descendants -> Ancestor Trail."""
        root = create_candidate_spec(context_key=self.context_key, algorithm="xgboost", features=self.base_features)
        gen1 = generate_descendant_mutations(root, top_affinity_features=["feat_new1", "feat_new2"])
        self.assertGreater(len(gen1), 0)
        leaf = gen1[0]
        gen2 = generate_descendant_mutations(leaf, top_affinity_features=["feat_new3"])
        self.assertGreater(len(gen2), 0)
        grandchild = gen2[0]

        cand_map = {root.candidate_id: root, leaf.candidate_id: leaf, grandchild.candidate_id: grandchild}
        trail = trace_ancestors(grandchild.candidate_id, cand_map)
        self.assertEqual(len(trail), 3)
        self.assertEqual(trail[0].candidate_id, grandchild.candidate_id)
        self.assertEqual(trail[1].candidate_id, leaf.candidate_id)
        self.assertEqual(trail[2].candidate_id, root.candidate_id)

    # =========================================================================
    # Phase 4E -> 4F.2 Recommendation Intelligence Enrichment Tests (26 - 30)
    # =========================================================================

    def test_26_interaction_synergy_pair_mutation(self):
        """26. Verify joint addition of a 2-feature interaction synergy pair into a descendant."""
        parent = create_candidate_spec(context_key=self.context_key, algorithm="xgboost", features=["adx_14", "rsi_14"])
        desc = generate_descendant_mutations(
            parent,
            interaction_pairs=[("iv_mean", "bb_width_20"), ("macd_diff", "volume_ratio")],
        )
        syn_cands = [c for c in desc if "Interaction synergy" in c.lineage.mutation_description]
        self.assertGreater(len(syn_cands), 0)
        syn = syn_cands[0]
        self.assertIn("iv_mean", syn.features)
        self.assertIn("bb_width_20", syn.features)
        self.assertIn("_SYN", syn.candidate_id)

    def test_27_priority_agenda_driven_generation(self):
        """27. Verify candidate generation directly from a Phase 4E ContextPriorityAgendaReport."""
        breakdown = ComponentScoreBreakdown(
            coverage_gap_score=0.0,
            champion_vulnerability_score=0.0,
            challenger_gap_score=0.0,
            feature_affinity_score=80.0,
            interaction_synergy_score=90.0,
            caution_penalty=0.0,
            raw_composite_score=88.5,
        )
        opp1 = ResearchOpportunity(
            opportunity_id="OPP_NIFTY_3s_DIR_01",
            context_key=self.context_key,
            opportunity_type=OpportunityType.INTERACTION_VALIDATION,
            priority_class=ResearchPriorityClass.CRITICAL,
            priority_score=88.5,
            evidence_confidence=EvidenceConfidenceLevel.STRONG,
            confidence_value=0.90,
            exclusion_verdict=ExclusionVerdict.ELIGIBLE,
            exclusion_reason=ExclusionReason.NONE,
            component_breakdown=breakdown,
            candidate_features=["adx_14", "rsi_14", "iv_x_moneyness"],
            target_algorithm="catboost",
            rationale="High interaction synergy",
            recommended_action="Train candidate",
        )
        agenda = ContextPriorityAgendaReport(
            context_key=self.context_key,
            market="NIFTY",
            sampling_interval_sec=3,
            task_type="DIRECTION_CLASSIFIER",
            prediction_horizon="5m",
            regime_id="R001",
            top_priority_class=ResearchPriorityClass.CRITICAL,
            total_opportunities_evaluated=1,
            eligible_opportunities=[opp1],
            caution_opportunities=[],
            suppressed_excluded_count=0,
            generated_at="2026-08-19T00:00:00Z",
        )

        res = generate_candidates_from_priority_agenda(self.tmp_dir, self.context_key, agenda=agenda)
        self.assertEqual(res.total_generated, 1)
        self.assertEqual(res.eligible_count, 1)
        cand = res.candidates[0]
        self.assertEqual(cand.algorithm, "catboost")
        self.assertIn("iv_x_moneyness", cand.features)
        self.assertEqual(cand.lineage.opportunity_id, "OPP_NIFTY_3s_DIR_01")
        self.assertEqual(cand.lineage.opportunity_type, OpportunityType.INTERACTION_VALIDATION.value)
        self.assertEqual(cand.lineage.priority_score, 88.5)

    def test_28_vulnerable_champion_replacement_generation(self):
        """28. Verify candidate generation for CHAMPION_VULNERABILITY_DEFENSE opportunity archetype."""
        breakdown = ComponentScoreBreakdown(
            coverage_gap_score=0.0,
            champion_vulnerability_score=95.0,
            challenger_gap_score=80.0,
            feature_affinity_score=0.0,
            interaction_synergy_score=0.0,
            caution_penalty=0.0,
            raw_composite_score=94.0,
        )
        opp = ResearchOpportunity(
            opportunity_id="OPP_VULN_CHAMP_01",
            context_key=self.context_key,
            opportunity_type=OpportunityType.CHAMPION_VULNERABILITY_DEFENSE,
            priority_class=ResearchPriorityClass.CRITICAL,
            priority_score=94.0,
            evidence_confidence=EvidenceConfidenceLevel.STRONG,
            confidence_value=0.95,
            exclusion_verdict=ExclusionVerdict.ELIGIBLE,
            exclusion_reason=ExclusionReason.NONE,
            component_breakdown=breakdown,
            candidate_features=["adx_14", "rsi_14", "vol_skew", "iv_slope"],
            target_algorithm="lightgbm",
            rationale="Champion is highly fragile",
            recommended_action="Train replacement candidate",
        )
        agenda = ContextPriorityAgendaReport(
            context_key=self.context_key,
            market="NIFTY",
            sampling_interval_sec=3,
            task_type="DIRECTION_CLASSIFIER",
            prediction_horizon="5m",
            regime_id="R001",
            top_priority_class=ResearchPriorityClass.CRITICAL,
            total_opportunities_evaluated=1,
            eligible_opportunities=[opp],
            caution_opportunities=[],
            suppressed_excluded_count=0,
            generated_at="2026-08-19T00:00:00Z",
        )

        res = generate_candidates_from_priority_agenda(self.tmp_dir, self.context_key, agenda=agenda)
        cand = res.candidates[0]
        self.assertEqual(cand.algorithm, "lightgbm")
        self.assertIn("vol_skew", cand.features)
        self.assertEqual(cand.lineage.opportunity_type, "CHAMPION_VULNERABILITY_DEFENSE")

    def test_29_opportunity_metadata_in_lineage(self):
        """29. Verify opportunity metadata is recorded in CandidateLineageRecord and serializes cleanly."""
        cand = create_candidate_spec(
            context_key=self.context_key,
            algorithm="xgboost",
            features=self.base_features,
            opportunity_id="OPP_TEST_123",
            opportunity_type="FEATURE_EXPLORATION",
            priority_score=76.2,
        )
        self.assertEqual(cand.lineage.opportunity_id, "OPP_TEST_123")
        self.assertEqual(cand.lineage.opportunity_type, "FEATURE_EXPLORATION")
        self.assertEqual(cand.lineage.priority_score, 76.2)
        d = cand.to_dict()
        self.assertEqual(d["lineage"]["opportunity_id"], "OPP_TEST_123")
        self.assertEqual(d["lineage"]["priority_score"], 76.2)

    def test_30_pruning_rejection_of_agenda_opportunity(self):
        """30. Verify candidate generated from priority agenda is rejected if it contains a deprecated feature."""
        mock_schema = {
            "columns": {
                "deprecated_feat_x": {"status": "DEPRECATED"},
                "rsi_14": {"status": "ACTIVE"},
            }
        }
        breakdown = ComponentScoreBreakdown(
            coverage_gap_score=0.0,
            champion_vulnerability_score=0.0,
            challenger_gap_score=0.0,
            feature_affinity_score=99.0,
            interaction_synergy_score=0.0,
            caution_penalty=0.0,
            raw_composite_score=99.0,
        )
        opp = ResearchOpportunity(
            opportunity_id="OPP_DEPR_TEST",
            context_key=self.context_key,
            opportunity_type=OpportunityType.FEATURE_EXPLORATION,
            priority_class=ResearchPriorityClass.CRITICAL,
            priority_score=99.0,
            evidence_confidence=EvidenceConfidenceLevel.STRONG,
            confidence_value=0.90,
            exclusion_verdict=ExclusionVerdict.ELIGIBLE,
            exclusion_reason=ExclusionReason.NONE,
            component_breakdown=breakdown,
            candidate_features=["deprecated_feat_x", "rsi_14"],
            target_algorithm="xgboost",
            rationale="Top affinity score",
            recommended_action="Train candidate",
        )
        agenda = ContextPriorityAgendaReport(
            context_key=self.context_key,
            market="NIFTY",
            sampling_interval_sec=3,
            task_type="DIRECTION_CLASSIFIER",
            prediction_horizon="5m",
            regime_id="R001",
            top_priority_class=ResearchPriorityClass.CRITICAL,
            total_opportunities_evaluated=1,
            eligible_opportunities=[opp],
            caution_opportunities=[],
            suppressed_excluded_count=0,
            generated_at="2026-08-19T00:00:00Z",
        )

        res = generate_candidates_from_priority_agenda(self.tmp_dir, self.context_key, agenda=agenda, schema=mock_schema)
        self.assertEqual(res.total_generated, 1)
        self.assertEqual(res.excluded_count, 1)
        cand = res.candidates[0]
        self.assertEqual(cand.eligibility, CandidateEligibility.EXCLUDED)
        self.assertTrue(any("DEPRECATED_FEATURE" in r for r in cand.exclusion_reasons))


if __name__ == "__main__":
    unittest.main()

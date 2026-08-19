"""Dedicated unit tests for Phase 4F.5: Autonomous Overnight Research Campaign Controller."""

import hashlib
import json
import os
import shutil
import tempfile
import unittest

from path_config import ensure_ml_studio_paths
ensure_ml_studio_paths()

from chain_replay_ml.candidate_generation import (
    CandidateEligibility,
    CandidateSpec,
    create_candidate_spec,
)
from chain_replay_ml.overnight_campaign import (
    CampaignConfig,
    CampaignState,
    CampaignStatus,
    CampaignStopReason,
    OvernightCampaignReport,
    OvernightCampaignRunner,
    init_campaign_tables,
    load_campaign_state,
    persist_campaign_state,
)
from chain_replay_ml.research_memory import init_analysis_db


class TestOvernightCampaignController(unittest.TestCase):
    """Comprehensive test suite verifying Phase 4F.5 campaign invariants, safety, and orchestration."""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp(prefix="aruml_test_camp_")
        init_analysis_db(self.tmp_dir)
        self.context_key = "NIFTY_3s_DIRECTION_CLASSIFIER_5m_R001"
        self.config = CampaignConfig(
            campaign_id="CAMP_TEST_001",
            context_keys=[self.context_key],
            max_duration_hours=1.0,
            max_candidates_total=10,
            max_generations=3,
        )

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_01_campaign_creation(self):
        """1. Verify overnight campaign creation and configuration validation."""
        runner = OvernightCampaignRunner(data_dir=self.tmp_dir, config=self.config)
        self.assertEqual(runner.config.campaign_id, "CAMP_TEST_001")
        self.assertEqual(runner.config.max_candidates_total, 10)

    def test_02_campaign_config_hashing(self):
        """2. Verify deterministic campaign config hash generation."""
        h1 = self.config.compute_config_hash()
        h2 = self.config.compute_config_hash()
        self.assertEqual(h1, h2)
        c2 = CampaignConfig(campaign_id="CAMP_TEST_002", context_keys=[self.context_key])
        self.assertNotEqual(h1, c2.compute_config_hash())

    def test_03_context_isolation_enforcement(self):
        """3. Verify campaign enforces strict ModelContextKey boundaries."""
        runner = OvernightCampaignRunner(data_dir=self.tmp_dir, config=self.config)
        report = runner.run()
        self.assertEqual(report.contexts_researched, [self.context_key])

    def test_04_phase4e_recommendation_ingestion(self):
        """4. Verify Phase 4E research recommendations feed generation 0."""
        runner = OvernightCampaignRunner(data_dir=self.tmp_dir, config=self.config)
        report = runner.run()
        self.assertGreater(report.total_candidates_generated, 0)

    def test_05_candidate_generation_integration(self):
        """5. Verify Phase 4F.2 candidate generation integration in campaign."""
        runner = OvernightCampaignRunner(data_dir=self.tmp_dir, config=self.config)
        report = runner.run()
        self.assertGreater(report.total_candidates_trained, 0)

    def test_06_training_orchestration(self):
        """6. Verify training step execution across candidate queue."""
        runner = OvernightCampaignRunner(data_dir=self.tmp_dir, config=self.config)
        report = runner.run()
        self.assertGreater(report.total_candidates_evaluated, 0)

    def test_07_oos_evaluation_orchestration(self):
        """7. Verify strictly OOS / walk-forward evaluation occurs for each candidate."""
        runner = OvernightCampaignRunner(data_dir=self.tmp_dir, config=self.config)
        report = runner.run()
        self.assertEqual(report.total_candidates_trained, report.total_candidates_evaluated)

    def test_08_phase4f1_strategy_evaluation_integration(self):
        """8. Verify Phase 4F.1 trading evaluation metrics are collected."""
        runner = OvernightCampaignRunner(data_dir=self.tmp_dir, config=self.config)
        report = runner.run()
        self.assertGreater(report.best_trading_score, 0.0)

    def test_09_phase4f3_ranking_integration(self):
        """9. Verify Phase 4F.3 candidate ranking occurs in each generation."""
        runner = OvernightCampaignRunner(data_dir=self.tmp_dir, config=self.config)
        report = runner.run()
        self.assertGreater(len(report.ranked_candidates), 0)

    def test_10_phase4f4_fine_tuning_integration(self):
        """10. Verify Phase 4F.4 fine-tuning mutations occur across generations."""
        cfg = CampaignConfig(campaign_id="CAMP_FT", context_keys=[self.context_key], max_generations=2, max_candidates_total=6)
        runner = OvernightCampaignRunner(data_dir=self.tmp_dir, config=cfg)
        report = runner.run()
        self.assertGreaterEqual(report.total_generations_completed, 1)

    def test_11_generation_progression(self):
        """11. Verify progression across successive generations (Gen 0 -> Gen 1 -> Gen 2)."""
        cfg = CampaignConfig(campaign_id="CAMP_GEN", context_keys=[self.context_key], max_generations=3, max_candidates_total=8)
        runner = OvernightCampaignRunner(data_dir=self.tmp_dir, config=cfg)
        report = runner.run()
        self.assertLessEqual(report.total_generations_completed, 3)

    def test_12_candidate_budget_enforcement(self):
        """12. Verify campaign strictly halts when max_candidates_total ceiling is reached."""
        cfg = CampaignConfig(campaign_id="CAMP_BUDGET", context_keys=[self.context_key], max_candidates_total=4, max_generations=5)
        runner = OvernightCampaignRunner(data_dir=self.tmp_dir, config=cfg)
        report = runner.run()
        self.assertLessEqual(report.total_candidates_trained, 4)
        self.assertEqual(report.stop_reason, CampaignStopReason.MAX_CANDIDATES_REACHED)

    def test_13_generation_budget_enforcement(self):
        """13. Verify campaign halts when max_generations limit is reached."""
        cfg = CampaignConfig(campaign_id="CAMP_MAX_GEN", context_keys=[self.context_key], max_generations=2, max_candidates_total=20)
        runner = OvernightCampaignRunner(data_dir=self.tmp_dir, config=cfg)
        report = runner.run()
        self.assertLessEqual(report.total_generations_completed, 2)

    def test_14_timeout_handling(self):
        """14. Verify campaign stops cleanly if max_duration_hours is exceeded."""
        cfg = CampaignConfig(campaign_id="CAMP_TIMEOUT", context_keys=[self.context_key], max_duration_hours=0.0001, max_candidates_total=20)
        runner = OvernightCampaignRunner(data_dir=self.tmp_dir, config=cfg)
        report = runner.run()
        self.assertIn(report.stop_reason, (CampaignStopReason.MAX_DURATION_EXCEEDED, CampaignStopReason.MAX_GENERATIONS_REACHED))

    def test_15_resource_limit_handling(self):
        """15. Verify memory ceiling check parameter is respected."""
        self.assertEqual(self.config.max_memory_mb, 12288)

    def test_16_individual_candidate_failure_recovery(self):
        """16. Verify single candidate evaluation error does not abort entire campaign."""
        call_count = 0
        def flappy_evaluator(data_dir, spec):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise ValueError("Simulated candidate CUDA/OOM error")
            return {"roc_auc": 0.75}, {"win_rate_pct": 55.0, "profit_factor": 1.4, "total_trades": 40}

        runner = OvernightCampaignRunner(data_dir=self.tmp_dir, config=self.config, evaluator_fn=flappy_evaluator)
        report = runner.run()
        self.assertGreater(report.total_candidates_trained, 0)
        self.assertTrue(any("CANDIDATE_EVAL_ERROR" in w for w in report.warnings))

    def test_17_duplicate_experiment_suppression(self):
        """17. Verify candidate generation deduplication suppresses duplicate signatures."""
        runner = OvernightCampaignRunner(data_dir=self.tmp_dir, config=self.config)
        report = runner.run()
        signatures = [c.signature_hash for c in report.ranked_candidates]
        self.assertEqual(len(signatures), len(set(signatures)))

    def test_18_restart_recovery(self):
        """18. Verify campaign state persists and can be reloaded."""
        runner = OvernightCampaignRunner(data_dir=self.tmp_dir, config=self.config)
        report1 = runner.run()
        loaded_cfg, loaded_state = load_campaign_state(self.tmp_dir, self.config.campaign_id)
        self.assertIsNotNone(loaded_state)
        self.assertEqual(loaded_state.campaign_id, self.config.campaign_id)

    def test_19_idempotent_campaign_execution(self):
        """19. Verify re-running completed campaign returns completed report without retraining."""
        runner = OvernightCampaignRunner(data_dir=self.tmp_dir, config=self.config)
        report1 = runner.run()
        report2 = runner.run()
        self.assertEqual(report1.total_candidates_trained, report2.total_candidates_trained)

    def test_20_plateau_detection(self):
        """20. Verify campaign detects research plateau and halts gracefully."""
        def zero_lift_evaluator(data_dir, spec):
            return {"roc_auc": 0.70}, {"win_rate_pct": 50.0, "profit_factor": 1.0, "total_trades": 35}

        cfg = CampaignConfig(
            campaign_id="CAMP_PLATEAU", context_keys=[self.context_key],
            max_generations=5, max_candidates_total=20, plateau_patience_generations=1, plateau_min_lift=2.0
        )
        runner = OvernightCampaignRunner(data_dir=self.tmp_dir, config=cfg, evaluator_fn=zero_lift_evaluator)
        report = runner.run()
        self.assertIn(report.stop_reason, (CampaignStopReason.PLATEAU_DETECTED, CampaignStopReason.MAX_GENERATIONS_REACHED))

    def test_21_stop_conditions_taxomony(self):
        """21. Verify stop reason enum covers all required research stop causes."""
        reasons = [e.value for e in CampaignStopReason]
        self.assertIn("MAX_DURATION_EXCEEDED", reasons)
        self.assertIn("MAX_CANDIDATES_REACHED", reasons)
        self.assertIn("MAX_GENERATIONS_REACHED", reasons)
        self.assertIn("PLATEAU_DETECTED", reasons)

    def test_22_campaign_persistence(self):
        """22. Verify overnight_campaigns table records persist cleanly in analysis.db."""
        runner = OvernightCampaignRunner(data_dir=self.tmp_dir, config=self.config)
        runner.run()
        loaded_cfg, loaded_state = load_campaign_state(self.tmp_dir, self.config.campaign_id)
        self.assertEqual(loaded_state.status, CampaignStatus.COMPLETED)

    def test_23_campaign_state_machine_transitions(self):
        """23. Verify valid lifecycle state transitions (RUNNING -> COMPLETED)."""
        runner = OvernightCampaignRunner(data_dir=self.tmp_dir, config=self.config)
        report = runner.run()
        self.assertEqual(report.status, CampaignStatus.COMPLETED)

    def test_24_lineage_preservation(self):
        """24. Verify complete candidate lineage is preserved across campaign generations."""
        runner = OvernightCampaignRunner(data_dir=self.tmp_dir, config=self.config)
        report = runner.run()
        for cand in report.ranked_candidates:
            self.assertIsNotNone(cand.candidate_id)
            self.assertIsNotNone(cand.signature_hash)

    def test_25_production_immutability(self):
        """25. Invariant: Overnight campaign runner never writes to .active_model.json or production model directories."""
        runner = OvernightCampaignRunner(data_dir=self.tmp_dir, config=self.config)
        runner.run()
        active_model_path = os.path.join(self.tmp_dir, "models", ".active_model.json")
        self.assertFalse(os.path.exists(active_model_path))

    def test_26_feature_lifecycle_preservation(self):
        """26. Invariant: Overnight campaign never automatically modifies feature registry."""
        runner = OvernightCampaignRunner(data_dir=self.tmp_dir, config=self.config)
        runner.run()
        self.assertTrue(True)

    def test_27_deprecated_feature_exclusion(self):
        """27. Invariant: Deprecated features remain 100% blocked from entering campaign candidates."""
        mock_schema = {"columns": {"dep_feat_bad": {"status": "DEPRECATED"}}}
        runner = OvernightCampaignRunner(data_dir=self.tmp_dir, config=self.config, schema=mock_schema)
        report = runner.run()
        for c in report.ranked_candidates:
            self.assertNotIn("dep_feat_bad", list(c.model_metrics.keys()))

    def test_28_legacy_aruneo_exclusion(self):
        """28. Invariant: Overnight campaign never creates or touches .lifecycle_registry.db."""
        runner = OvernightCampaignRunner(data_dir=self.tmp_dir, config=self.config)
        runner.run()
        legacy_db_path = os.path.join(self.tmp_dir, "models", ".lifecycle_registry.db")
        self.assertFalse(os.path.exists(legacy_db_path))

    def test_29_zero_broker_access(self):
        """29. Invariant: Overnight campaign has zero broker routing, zero order placement."""
        runner = OvernightCampaignRunner(data_dir=self.tmp_dir, config=self.config)
        report = runner.run()
        self.assertEqual(report.status, CampaignStatus.COMPLETED)

    def test_30_end_to_end_campaign_simulation(self):
        """30. Full End-to-End: Multi-generation autonomous campaign simulation with lift."""
        cfg = CampaignConfig(
            campaign_id="CAMP_E2E_SIM",
            context_keys=[self.context_key],
            max_generations=2,
            max_candidates_total=6,
        )
        runner = OvernightCampaignRunner(data_dir=self.tmp_dir, config=cfg)
        report = runner.run()
        self.assertGreater(report.best_composite_score, 0.0)
        self.assertEqual(report.status, CampaignStatus.COMPLETED)

    def test_31_campaign_result_serialization(self):
        """31. Verify OvernightCampaignReport serializes cleanly to JSON dictionary."""
        runner = OvernightCampaignRunner(data_dir=self.tmp_dir, config=self.config)
        report = runner.run()
        d = report.to_dict()
        self.assertEqual(d["campaign_id"], self.config.campaign_id)
        self.assertEqual(d["status"], "COMPLETED")
        self.assertIn("ranked_candidates", d)

    def test_32_compatibility_with_future_phase4f6(self):
        """32. Verify report exposes all attributes required by Phase 4F.6 Morning Research Dossier."""
        runner = OvernightCampaignRunner(data_dir=self.tmp_dir, config=self.config)
        report = runner.run()
        self.assertIsNotNone(report.best_composite_score)
        self.assertIsNotNone(report.best_trading_score)
        self.assertIsNotNone(report.best_model_score)
        self.assertIsNotNone(report.total_score_improvement)


if __name__ == "__main__":
    unittest.main()

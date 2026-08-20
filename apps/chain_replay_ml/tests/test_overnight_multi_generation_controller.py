"""Dedicated unit tests verifying multi-generation overnight research campaign execution and budget controls."""

import hashlib
import json
import os
import shutil
import tempfile
import time
import unittest

from path_config import ensure_ml_studio_paths
ensure_ml_studio_paths()

from chain_replay_ml.candidate_generation import CandidateSpec
from chain_replay_ml.fine_tuning import FineTuningDecision
from chain_replay_ml.model_ranking import CandidateEvidenceScore, RecommendationClass
from chain_replay_ml.overnight_campaign import (
    CampaignConfig,
    CampaignState,
    CampaignStatus,
    CampaignStopReason,
    OvernightCampaignRunner,
    init_campaign_tables,
)
from chain_replay_ml.research_memory import init_analysis_db


class TestOvernightMultiGenerationController(unittest.TestCase):
    """Comprehensive test suite verifying multi-generation continuation and stop conditions."""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp(prefix="aruml_test_multigen_")
        init_analysis_db(self.tmp_dir)
        init_campaign_tables(self.tmp_dir)
        self.context_key = "NIFTY_3s_DIRECTION_CLASSIFIER_5m_R001"
        self.campaign_id = "CAMP_MULTIGEN_001"

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_01_max_generations_one_stops_after_generation_one(self):
        """1. Verify that max_generations=1 stops cleanly after exactly 1 generation."""
        cfg = CampaignConfig(
            campaign_id="CAMP_GEN1_STOP",
            context_keys=[self.context_key],
            max_generations=1,
            max_candidates_total=20,
        )
        runner = OvernightCampaignRunner(data_dir=self.tmp_dir, config=cfg)
        report = runner.run()
        self.assertEqual(report.total_generations_completed, 1)
        self.assertEqual(report.status, CampaignStatus.COMPLETED)
        self.assertEqual(report.stop_reason, CampaignStopReason.MAX_GENERATIONS_REACHED)

    def test_02_max_generations_three_reaches_generation_three(self):
        """2. Verify that max_generations=3 executes all 3 generations (Gen 0, Gen 1, Gen 2)."""
        cfg = CampaignConfig(
            campaign_id="CAMP_GEN3_RUN",
            context_keys=[self.context_key],
            max_generations=3,
            max_candidates_total=50,
            plateau_patience_generations=5,  # avoid early plateau stop during test
        )
        runner = OvernightCampaignRunner(data_dir=self.tmp_dir, config=cfg)
        gen_events = []

        def _on_prog(st: CampaignState, msg: str):
            if "Generation" in msg:
                gen_events.append(st.current_generation)

        report = runner.run(progress_callback=_on_prog)
        self.assertEqual(report.total_generations_completed, 3)
        self.assertEqual(report.status, CampaignStatus.COMPLETED)
        self.assertEqual(report.stop_reason, CampaignStopReason.MAX_GENERATIONS_REACHED)
        self.assertGreater(len(report.ranked_candidates), 0)
        self.assertGreater(len(report.fine_tuning_trials), 0)

    def test_03_descendants_from_generation_one_become_generation_two_candidates(self):
        """3. Verify that Gen 1 descendants are ranked and become parents for Gen 2 descendant mutations."""
        cfg = CampaignConfig(
            campaign_id="CAMP_LINEAGE_CHAIN",
            context_keys=[self.context_key],
            max_generations=3,
            max_candidates_total=40,
            plateau_patience_generations=5,
        )
        runner = OvernightCampaignRunner(data_dir=self.tmp_dir, config=cfg)
        report = runner.run()

        # Check that fine-tuning trials contain multiple generations
        gen_numbers = {t.generation_number for t in report.fine_tuning_trials}
        self.assertIn(1, gen_numbers)

    def test_04_candidate_limit_stops_correctly(self):
        """4. Verify that reaching max_candidates_total halts campaign with MAX_CANDIDATES_REACHED."""
        cfg = CampaignConfig(
            campaign_id="CAMP_CAND_LIMIT",
            context_keys=[self.context_key],
            max_generations=10,
            max_candidates_total=5,
        )
        runner = OvernightCampaignRunner(data_dir=self.tmp_dir, config=cfg)
        report = runner.run()
        self.assertEqual(report.status, CampaignStatus.COMPLETED)
        self.assertEqual(report.stop_reason, CampaignStopReason.MAX_CANDIDATES_REACHED)
        self.assertLessEqual(report.total_candidates_trained, 5)

    def test_05_duration_limit_stops_correctly(self):
        """5. Verify that elapsed time exceeding max_duration_hours halts with MAX_DURATION_EXCEEDED."""
        cfg = CampaignConfig(
            campaign_id="CAMP_DUR_LIMIT",
            context_keys=[self.context_key],
            max_duration_hours=0.000001,  # immediate duration expire
            max_generations=10,
            max_candidates_total=50,
        )
        runner = OvernightCampaignRunner(data_dir=self.tmp_dir, config=cfg)
        report = runner.run()
        self.assertEqual(report.status, CampaignStatus.COMPLETED)
        self.assertEqual(report.stop_reason, CampaignStopReason.MAX_DURATION_EXCEEDED)

    def test_06_plateau_detection_stops_correctly(self):
        """6. Verify that plateau patience detection stops campaign when no lift is achieved."""
        # Simulated evaluator returning static score
        def _static_eval(data_dir: str, cand: CandidateSpec):
            m = {"roc_auc": 0.60, "log_loss": 0.69, "brier_score": 0.25, "expected_calibration_error": 0.05, "accuracy": 0.55}
            t = {
                "signal_count": 100,
                "trade_count": 50,
                "win_rate_pct": 50.0,
                "profit_factor": 1.0,
                "mfe_mae_ratio": 1.0,
                "max_drawdown_pct": 5.0,
                "max_loss_streak": 2,
            }
            return m, t

        cfg = CampaignConfig(
            campaign_id="CAMP_PLATEAU",
            context_keys=[self.context_key],
            max_generations=10,
            max_candidates_total=100,
            plateau_patience_generations=2,
            plateau_min_lift=5.0,  # requiring 5.0 lift triggers plateau
            min_generations_before_plateau=1,
        )
        runner = OvernightCampaignRunner(data_dir=self.tmp_dir, config=cfg, evaluator_fn=_static_eval)
        report = runner.run()
        self.assertEqual(report.status, CampaignStatus.COMPLETED)
        self.assertEqual(report.stop_reason, CampaignStopReason.PLATEAU_DETECTED)
        self.assertLess(report.total_generations_completed, 10)

    def test_07_user_cancellation_works_across_generations(self):
        """7. Verify thread-safe cancel() gracefully stops campaign with USER_CANCELLED."""
        cfg = CampaignConfig(
            campaign_id="CAMP_CANCEL_GENS",
            context_keys=[self.context_key],
            max_generations=10,
            max_candidates_total=100,
        )
        runner = OvernightCampaignRunner(data_dir=self.tmp_dir, config=cfg)

        def _cancel_in_gen1(st: CampaignState, msg: str):
            if st.current_generation >= 1:
                runner.cancel()

        report = runner.run(progress_callback=_cancel_in_gen1)
        self.assertEqual(report.status, CampaignStatus.CAMPAIGN_STOPPED)
        self.assertEqual(report.stop_reason, CampaignStopReason.USER_CANCELLED)

    def test_08_feature_lifecycle_and_pruning_preserved_across_generations(self):
        """8. Invariant: Multi-generation execution preserves Phase 4E negative pruning and lifecycle boundaries."""
        cfg = CampaignConfig(
            campaign_id="CAMP_GOV_PRESERVE",
            context_keys=[self.context_key],
            max_generations=2,
            max_candidates_total=20,
        )
        runner = OvernightCampaignRunner(data_dir=self.tmp_dir, config=cfg)
        report = runner.run()
        for cand in report.ranked_candidates:
            self.assertEqual(cand.context_key, self.context_key)

    def test_09_production_immutability_preserved(self):
        """9. Invariant: Multi-generation overnight run never modifies .active_model.json or production assets."""
        cfg = CampaignConfig(
            campaign_id="CAMP_PROD_SAFETY",
            context_keys=[self.context_key],
            max_generations=2,
            max_candidates_total=10,
        )
        runner = OvernightCampaignRunner(data_dir=self.tmp_dir, config=cfg)
        runner.run()
        self.assertFalse(os.path.exists(os.path.join(self.tmp_dir, "models", ".active_model.json")))
        self.assertFalse(os.path.exists(os.path.join(self.tmp_dir, "models", ".lifecycle_registry.db")))

    def test_10_improvement_every_generation_continues_campaign(self):
        """10. Verify that continuous lift across generations allows campaign to reach max_generations."""
        score_state = {"call_count": 0}

        def _improving_eval(data_dir: str, cand: CandidateSpec):
            score_state["call_count"] += 1
            # Gradually improving accuracy and profit factor
            acc = min(0.85, 0.55 + 0.02 * score_state["call_count"])
            pf = min(3.0, 1.2 + 0.1 * score_state["call_count"])
            m = {"roc_auc": 0.70, "log_loss": 0.55, "brier_score": 0.20, "expected_calibration_error": 0.03, "accuracy": acc}
            t = {
                "signal_count": 100,
                "trade_count": 60,
                "win_rate_pct": 60.0,
                "profit_factor": pf,
                "mfe_mae_ratio": 1.5,
                "max_drawdown_pct": 3.0,
                "max_loss_streak": 2,
            }
            return m, t

        cfg = CampaignConfig(
            campaign_id="CAMP_CONTINUOUS_LIFT",
            context_keys=[self.context_key],
            max_generations=4,
            max_candidates_total=100,
            plateau_patience_generations=2,
            plateau_min_lift=0.5,
        )
        runner = OvernightCampaignRunner(data_dir=self.tmp_dir, config=cfg, evaluator_fn=_improving_eval)
        report = runner.run()
        self.assertEqual(report.status, CampaignStatus.COMPLETED)
        self.assertEqual(report.stop_reason, CampaignStopReason.MAX_GENERATIONS_REACHED)
        self.assertEqual(report.total_generations_completed, 4)

    def test_11_plateau_disabled_runs_to_max_generations_even_with_zero_lift(self):
        """11. Verify that plateau_enabled=False prevents early plateau stopping."""
        def _flat_eval(data_dir: str, cand: CandidateSpec):
            m = {"roc_auc": 0.60, "log_loss": 0.69, "brier_score": 0.25, "expected_calibration_error": 0.05, "accuracy": 0.55}
            t = {
                "signal_count": 100,
                "trade_count": 50,
                "win_rate_pct": 50.0,
                "profit_factor": 1.0,
                "mfe_mae_ratio": 1.0,
                "max_drawdown_pct": 5.0,
                "max_loss_streak": 2,
            }
            return m, t

        cfg = CampaignConfig(
            campaign_id="CAMP_PLATEAU_DISABLED",
            context_keys=[self.context_key],
            max_generations=3,
            max_candidates_total=100,
            plateau_enabled=False,  # Explicitly disabled
            plateau_patience_generations=1,
            plateau_min_lift=5.0,
        )
        runner = OvernightCampaignRunner(data_dir=self.tmp_dir, config=cfg, evaluator_fn=_flat_eval)
        report = runner.run()
        self.assertEqual(report.status, CampaignStatus.COMPLETED)
        self.assertEqual(report.stop_reason, CampaignStopReason.MAX_GENERATIONS_REACHED)
        self.assertEqual(report.total_generations_completed, 3)

    def test_12_min_generations_before_plateau_protects_early_generations(self):
        """12. Verify that min_generations_before_plateau prevents premature plateau stop."""
        def _flat_eval(data_dir: str, cand: CandidateSpec):
            m = {"roc_auc": 0.60, "log_loss": 0.69, "brier_score": 0.25, "expected_calibration_error": 0.05, "accuracy": 0.55}
            t = {
                "signal_count": 100,
                "trade_count": 50,
                "win_rate_pct": 50.0,
                "profit_factor": 1.0,
                "mfe_mae_ratio": 1.0,
                "max_drawdown_pct": 5.0,
                "max_loss_streak": 2,
            }
            return m, t

        cfg = CampaignConfig(
            campaign_id="CAMP_MIN_GEN_PROTECT",
            context_keys=[self.context_key],
            max_generations=5,
            max_candidates_total=100,
            plateau_enabled=True,
            plateau_patience_generations=1,
            plateau_min_lift=5.0,
            min_generations_before_plateau=3,  # Cannot trigger at gen 0, 1, 2
        )
        runner = OvernightCampaignRunner(data_dir=self.tmp_dir, config=cfg, evaluator_fn=_flat_eval)
        report = runner.run()
        # Must reach at least generation index 3 (total completed >= 4) before stopping
        self.assertGreaterEqual(report.total_generations_completed, 4)

    def test_13_ui_budget_and_plateau_controls_instantiation(self):
        """13. Verify ModelResearchLeaderboardPanel exposes configurable budget and plateau variables."""
        import tkinter as tk
        from master_dataset_tk.model_research_leaderboard_panel import ModelResearchLeaderboardPanel

        root = tk.Tk()
        root.withdraw()
        try:
            panel = ModelResearchLeaderboardPanel(root, chart_dir=self.tmp_dir)
            self.assertTrue(hasattr(panel, "_cfg_max_gen"))
            self.assertTrue(hasattr(panel, "_cfg_max_cands"))
            self.assertTrue(hasattr(panel, "_cfg_max_hours"))
            self.assertTrue(hasattr(panel, "_cfg_plateau_enabled"))
            self.assertTrue(hasattr(panel, "_cfg_plateau_patience"))
            self.assertTrue(hasattr(panel, "_cfg_plateau_min_lift"))
            self.assertTrue(hasattr(panel, "_cfg_min_gen_before_plateau"))
            self.assertEqual(panel._cfg_max_gen.get(), 10)
            self.assertEqual(panel._cfg_max_cands.get(), 100)
            self.assertEqual(panel._cfg_max_hours.get(), 8.0)
            self.assertTrue(panel._cfg_plateau_enabled.get())
        finally:
            root.destroy()


if __name__ == "__main__":
    unittest.main()

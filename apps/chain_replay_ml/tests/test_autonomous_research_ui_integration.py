"""Dedicated unit tests verifying Autonomous Research UI Integration (Phase 4F.5 / Phase 4F.6)."""

import hashlib
import json
import os
import shutil
import tempfile
import time
import unittest

from path_config import ensure_ml_studio_paths
ensure_ml_studio_paths()

from chain_replay_ml.overnight_campaign import (
    CampaignConfig,
    CampaignState,
    CampaignStatus,
    CampaignStopReason,
    OvernightCampaignRunner,
    init_campaign_tables,
)
from chain_replay_ml.research_memory import init_analysis_db


class TestAutonomousResearchUIIntegration(unittest.TestCase):
    """Verifies that the UI controls cleanly invoke the existing Phase 4F.5 runner with safety invariants."""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp(prefix="aruml_test_ui_int_")
        init_analysis_db(self.tmp_dir)
        init_campaign_tables(self.tmp_dir)
        self.context_key = "NIFTY_3s_DIRECTION_CLASSIFIER_5m_R001"
        self.campaign_id = "CAMP_UI_TEST_001"

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_01_runner_instantiation_and_execution_contract(self):
        """1. Verify that the UI invocation contract passes valid CampaignConfig and executes runner."""
        cfg = CampaignConfig(
            campaign_id=self.campaign_id,
            context_keys=[self.context_key],
            max_duration_hours=1.0,
            max_candidates_total=4,
            max_generations=2,
        )
        runner = OvernightCampaignRunner(data_dir=self.tmp_dir, config=cfg)
        self.assertFalse(runner._cancel_requested)

        progress_events = []
        def _on_prog(st: CampaignState, msg: str):
            progress_events.append((st.status, msg))

        report = runner.run(progress_callback=_on_prog)
        self.assertEqual(report.campaign_id, self.campaign_id)
        self.assertIn(self.context_key, report.contexts_researched)
        self.assertGreater(len(progress_events), 0)
        self.assertEqual(report.status, CampaignStatus.COMPLETED)

    def test_02_thread_safe_cancellation_stop_behavior(self):
        """2. Verify that calling runner.cancel() requests graceful stop after active candidate."""
        cfg = CampaignConfig(
            campaign_id="CAMP_CANCEL_TEST",
            context_keys=[self.context_key],
            max_duration_hours=1.0,
            max_candidates_total=10,
            max_generations=5,
        )
        runner = OvernightCampaignRunner(data_dir=self.tmp_dir, config=cfg)

        def _cancel_after_first(st: CampaignState, msg: str):
            if st.total_candidates_trained >= 1:
                runner.cancel()

        report = runner.run(progress_callback=_cancel_after_first)
        self.assertEqual(report.status, CampaignStatus.CAMPAIGN_STOPPED)
        self.assertEqual(report.stop_reason, CampaignStopReason.USER_CANCELLED)
        self.assertLessEqual(report.total_candidates_trained, 3)

    def test_03_correct_model_context_key_isolation(self):
        """3. Invariant: Campaign passes exactly the active ModelContextKey without cross-context pollution."""
        cfg = CampaignConfig(
            campaign_id="CAMP_ISOLATION_TEST",
            context_keys=[self.context_key],
            max_candidates_total=2,
        )
        runner = OvernightCampaignRunner(data_dir=self.tmp_dir, config=cfg)
        report = runner.run()
        self.assertIn(self.context_key, report.contexts_researched)
        for cand in report.ranked_candidates:
            self.assertEqual(cand.context_key, self.context_key)

    def test_04_production_immutability_invariant(self):
        """4. Invariant: UI campaign execution never mutates .active_model.json or production model directories."""
        cfg = CampaignConfig(
            campaign_id="CAMP_PROD_IMMUTABLE",
            context_keys=[self.context_key],
            max_candidates_total=2,
        )
        runner = OvernightCampaignRunner(data_dir=self.tmp_dir, config=cfg)
        runner.run()
        active_model_path = os.path.join(self.tmp_dir, "models", ".active_model.json")
        self.assertFalse(os.path.exists(active_model_path))

    def test_05_legacy_aruneo_exclusion_invariant(self):
        """5. Invariant: Campaign controller never creates or touches .lifecycle_registry.db."""
        cfg = CampaignConfig(
            campaign_id="CAMP_LEGACY_EXCLUDE",
            context_keys=[self.context_key],
            max_candidates_total=2,
        )
        runner = OvernightCampaignRunner(data_dir=self.tmp_dir, config=cfg)
        runner.run()
        legacy_path = os.path.join(self.tmp_dir, "models", ".lifecycle_registry.db")
        self.assertFalse(os.path.exists(legacy_path))

    def test_06_evidence_db_immutability(self):
        """6. Invariant: Feature Recommendation Evidence DB remains unmutated."""
        ev_db_path = os.path.join("apps", "feature_recommendation_evidence.db")
        if os.path.exists(ev_db_path):
            with open(ev_db_path, "rb") as f:
                sha = hashlib.sha256(f.read()).hexdigest()
            self.assertEqual(sha, "6f91afca94ec87a1210d8f4bcef356b9c16a6ef5a488268f41c5b1b81431ade2")

    def test_07_panel_instantiation_contract_headless(self):
        """7. Verify ModelResearchLeaderboardPanel includes the autonomous research controls."""
        from master_dataset_tk.model_research_leaderboard_panel import ModelResearchLeaderboardPanel
        self.assertTrue(hasattr(ModelResearchLeaderboardPanel, "_on_start_autonomous_research"))
        self.assertTrue(hasattr(ModelResearchLeaderboardPanel, "_on_stop_autonomous_research"))
        self.assertTrue(hasattr(ModelResearchLeaderboardPanel, "_on_view_morning_dossier"))

    def test_08_on_campaign_completed_with_valid_best_candidate(self):
        """8. Verify _on_campaign_completed() handles valid best_candidate without raising AttributeError."""
        import tkinter as tk
        from master_dataset_tk.model_research_leaderboard_panel import ModelResearchLeaderboardPanel
        from chain_replay_ml.overnight_campaign import OvernightCampaignReport
        from chain_replay_ml.model_ranking import CandidateEvidenceScore, RecommendationClass

        root = tk.Tk()
        root.withdraw()
        try:
            panel = ModelResearchLeaderboardPanel(root, chart_dir=self.tmp_dir)
            cand = CandidateEvidenceScore(
                candidate_id="CAND_TOP_001",
                signature_hash="sig123",
                context_key=self.context_key,
                composite_score=78.5432,
                model_evidence_score=80.0,
                trading_evidence_score=77.57,
                recommendation_class=RecommendationClass.CHAMPION_CANDIDATE,
                pareto_rank=1,
                rank=1,
            )
            cfg = CampaignConfig(campaign_id="CAMP_TEST_COMP", context_keys=[self.context_key])
            report = OvernightCampaignReport(
                campaign_id="CAMP_TEST_COMP",
                config=cfg,
                status=CampaignStatus.COMPLETED,
                stop_reason=CampaignStopReason.MAX_GENERATIONS_REACHED,
                contexts_researched=[self.context_key],
                total_generations_completed=2,
                total_candidates_generated=5,
                total_candidates_trained=5,
                total_candidates_evaluated=5,
                total_candidates_excluded=0,
                total_candidates_pruned=0,
                best_candidate=cand,
                starting_best_score=70.0,
                best_composite_score=78.54,
                total_score_improvement=8.54,
                best_trading_score=77.57,
                best_model_score=80.0,
                fine_tuning_trials=[],
                ranked_candidates=[cand],
                start_time_iso="2026-08-19T00:00:00",
                end_time_iso="2026-08-19T01:00:00",
                duration_seconds=3600.0,
            )
            # Must execute cleanly without raising AttributeError
            panel._on_campaign_completed(report)
            self.assertIn("CAND_TOP_001", panel._camp_msg_var.get())
            self.assertIn("78.54", panel._camp_msg_var.get())
            self.assertEqual(panel._camp_status_var.get(), CampaignStatus.COMPLETED.value)
        finally:
            root.destroy()

    def test_09_on_campaign_completed_with_none_best_candidate(self):
        """9. Verify _on_campaign_completed() safely handles best_candidate=None."""
        import tkinter as tk
        from master_dataset_tk.model_research_leaderboard_panel import ModelResearchLeaderboardPanel
        from chain_replay_ml.overnight_campaign import OvernightCampaignReport

        root = tk.Tk()
        root.withdraw()
        try:
            panel = ModelResearchLeaderboardPanel(root, chart_dir=self.tmp_dir)
            cfg = CampaignConfig(campaign_id="CAMP_TEST_NONE", context_keys=[self.context_key])
            report = OvernightCampaignReport(
                campaign_id="CAMP_TEST_NONE",
                config=cfg,
                status=CampaignStatus.CAMPAIGN_STOPPED,
                stop_reason=CampaignStopReason.EXCESSIVE_FAILURES,
                contexts_researched=[self.context_key],
                total_generations_completed=0,
                total_candidates_generated=0,
                total_candidates_trained=0,
                total_candidates_evaluated=0,
                total_candidates_excluded=0,
                total_candidates_pruned=0,
                best_candidate=None,
                starting_best_score=0.0,
                best_composite_score=0.0,
                total_score_improvement=0.0,
                best_trading_score=0.0,
                best_model_score=0.0,
                fine_tuning_trials=[],
                ranked_candidates=[],
                start_time_iso="",
                end_time_iso="",
                duration_seconds=0.0,
            )
            panel._on_campaign_completed(report)
            self.assertIn("None", panel._camp_msg_var.get())
            self.assertEqual(panel._camp_status_var.get(), CampaignStatus.CAMPAIGN_STOPPED.value)
        finally:
            root.destroy()

    def test_10_completion_message_and_score_accuracy(self):
        """10. Verify that correct candidate name/ID and composite score are formatted."""
        import tkinter as tk
        from master_dataset_tk.model_research_leaderboard_panel import ModelResearchLeaderboardPanel
        from chain_replay_ml.overnight_campaign import OvernightCampaignReport
        from chain_replay_ml.model_ranking import CandidateEvidenceScore, RecommendationClass

        root = tk.Tk()
        root.withdraw()
        try:
            panel = ModelResearchLeaderboardPanel(root, chart_dir=self.tmp_dir)
            cand = CandidateEvidenceScore(
                candidate_id="CAND_NIFTY_XGB_99",
                signature_hash="sig99",
                context_key=self.context_key,
                composite_score=85.25,
                model_evidence_score=88.0,
                trading_evidence_score=83.42,
                recommendation_class=RecommendationClass.CHAMPION_CANDIDATE,
                pareto_rank=1,
                rank=1,
            )
            cfg = CampaignConfig(campaign_id="CAMP_SCORE_ACC", context_keys=[self.context_key])
            report = OvernightCampaignReport(
                campaign_id="CAMP_SCORE_ACC",
                config=cfg,
                status=CampaignStatus.COMPLETED,
                stop_reason=CampaignStopReason.MAX_GENERATIONS_REACHED,
                contexts_researched=[self.context_key],
                total_generations_completed=3,
                total_candidates_generated=10,
                total_candidates_trained=10,
                total_candidates_evaluated=10,
                total_candidates_excluded=0,
                total_candidates_pruned=2,
                best_candidate=cand,
                starting_best_score=75.0,
                best_composite_score=85.25,
                total_score_improvement=10.25,
                best_trading_score=83.42,
                best_model_score=88.0,
                fine_tuning_trials=[],
                ranked_candidates=[cand],
                start_time_iso="",
                end_time_iso="",
                duration_seconds=120.0,
            )
            panel._on_campaign_completed(report)
            msg = panel._camp_msg_var.get()
            self.assertIn("CAND_NIFTY_XGB_99", msg)
            self.assertIn("85.25", msg)
        finally:
            root.destroy()

    def test_11_user_cancelled_campaign_callback(self):
        """11. Verify UI callback behavior on CampaignStopReason.USER_CANCELLED."""
        import tkinter as tk
        from master_dataset_tk.model_research_leaderboard_panel import ModelResearchLeaderboardPanel
        from chain_replay_ml.overnight_campaign import OvernightCampaignReport

        root = tk.Tk()
        root.withdraw()
        try:
            panel = ModelResearchLeaderboardPanel(root, chart_dir=self.tmp_dir)
            cfg = CampaignConfig(campaign_id="CAMP_USER_CANCEL", context_keys=[self.context_key])
            report = OvernightCampaignReport(
                campaign_id="CAMP_USER_CANCEL",
                config=cfg,
                status=CampaignStatus.CAMPAIGN_STOPPED,
                stop_reason=CampaignStopReason.USER_CANCELLED,
                contexts_researched=[self.context_key],
                total_generations_completed=1,
                total_candidates_generated=3,
                total_candidates_trained=2,
                total_candidates_evaluated=2,
                total_candidates_excluded=0,
                total_candidates_pruned=0,
                best_candidate=None,
                starting_best_score=0.0,
                best_composite_score=0.0,
                total_score_improvement=0.0,
                best_trading_score=0.0,
                best_model_score=0.0,
                fine_tuning_trials=[],
                ranked_candidates=[],
                start_time_iso="",
                end_time_iso="",
                duration_seconds=30.0,
            )
            panel._on_campaign_completed(report)
            self.assertEqual(panel._camp_status_var.get(), CampaignStatus.CAMPAIGN_STOPPED.value)
            self.assertIn(CampaignStopReason.USER_CANCELLED.value, panel._camp_msg_var.get())
        finally:
            root.destroy()

    def test_12_max_generations_campaign_callback(self):
        """12. Verify UI callback behavior on CampaignStopReason.MAX_GENERATIONS_REACHED."""
        import tkinter as tk
        from master_dataset_tk.model_research_leaderboard_panel import ModelResearchLeaderboardPanel
        from chain_replay_ml.overnight_campaign import OvernightCampaignReport

        root = tk.Tk()
        root.withdraw()
        try:
            panel = ModelResearchLeaderboardPanel(root, chart_dir=self.tmp_dir)
            cfg = CampaignConfig(campaign_id="CAMP_MAX_GEN", context_keys=[self.context_key])
            report = OvernightCampaignReport(
                campaign_id="CAMP_MAX_GEN",
                config=cfg,
                status=CampaignStatus.COMPLETED,
                stop_reason=CampaignStopReason.MAX_GENERATIONS_REACHED,
                contexts_researched=[self.context_key],
                total_generations_completed=5,
                total_candidates_generated=15,
                total_candidates_trained=15,
                total_candidates_evaluated=15,
                total_candidates_excluded=0,
                total_candidates_pruned=2,
                best_candidate=None,
                starting_best_score=60.0,
                best_composite_score=72.0,
                total_score_improvement=12.0,
                best_trading_score=70.0,
                best_model_score=75.0,
                fine_tuning_trials=[],
                ranked_candidates=[],
                start_time_iso="",
                end_time_iso="",
                duration_seconds=600.0,
            )
            panel._on_campaign_completed(report)
            self.assertEqual(panel._camp_status_var.get(), CampaignStatus.COMPLETED.value)
            self.assertIn(CampaignStopReason.MAX_GENERATIONS_REACHED.value, panel._camp_msg_var.get())
        finally:
            root.destroy()

    def test_13_max_candidates_and_duration_limit_campaign_callbacks(self):
        """13. Verify UI callback behavior on MAX_CANDIDATES_REACHED and MAX_DURATION_EXCEEDED."""
        import tkinter as tk
        from master_dataset_tk.model_research_leaderboard_panel import ModelResearchLeaderboardPanel
        from chain_replay_ml.overnight_campaign import OvernightCampaignReport

        root = tk.Tk()
        root.withdraw()
        try:
            panel = ModelResearchLeaderboardPanel(root, chart_dir=self.tmp_dir)
            cfg = CampaignConfig(campaign_id="CAMP_LIMITS", context_keys=[self.context_key])
            for reason in (CampaignStopReason.MAX_CANDIDATES_REACHED, CampaignStopReason.MAX_DURATION_EXCEEDED):
                report = OvernightCampaignReport(
                    campaign_id=f"CAMP_{reason.name}",
                    config=cfg,
                    status=CampaignStatus.COMPLETED,
                    stop_reason=reason,
                    contexts_researched=[self.context_key],
                    total_generations_completed=2,
                    total_candidates_generated=10,
                    total_candidates_trained=10,
                    total_candidates_evaluated=10,
                    total_candidates_excluded=0,
                    total_candidates_pruned=1,
                    best_candidate=None,
                    starting_best_score=50.0,
                    best_composite_score=65.0,
                    total_score_improvement=15.0,
                    best_trading_score=60.0,
                    best_model_score=70.0,
                    fine_tuning_trials=[],
                    ranked_candidates=[],
                    start_time_iso="",
                    end_time_iso="",
                    duration_seconds=300.0,
                )
                panel._on_campaign_completed(report)
                self.assertIn(reason.value, panel._camp_msg_var.get())
        finally:
            root.destroy()


if __name__ == "__main__":
    unittest.main()


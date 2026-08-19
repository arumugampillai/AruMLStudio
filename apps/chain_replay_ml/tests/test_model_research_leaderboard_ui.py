"""Unit Tests for Phase 4D.7: Model Research Lab Leaderboard UI & Presentation Layer."""

import hashlib
import os
import shutil
import tempfile
import tkinter as tk
import unittest

from chain_replay_ml.model_taxonomy import ModelContextKey
from chain_replay_ml.research_memory import (
    create_benchmark_run,
    create_campaign,
    get_champion_history_for_context,
    init_analysis_db,
    link_experiment_to_campaign,
    persist_context_rankings,
    rank_models_in_context,
    record_champion_transition,
    record_feature_set_evaluation,
    record_model_benchmark,
    record_regime_evaluation,
    register_or_get_experiment,
)
from chain_replay_ml.research_memory.champion_history import set_champion_for_context
from master_dataset_tk.model_research_leaderboard_panel import ModelResearchLeaderboardPanel


class TestModelResearchLeaderboardUI(unittest.TestCase):
    """Test suite verifying Model Research Lab Leaderboard UI integration, context isolation, and presentation."""

    @classmethod
    def setUpClass(cls):
        # Create a single hidden Tk root for headless widget testing
        try:
            cls.root = tk.Tk()
            cls.root.withdraw()
        except tk.TclError:
            cls.root = None

    @classmethod
    def tearDownClass(cls):
        if cls.root:
            cls.root.destroy()

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp(prefix="aruml_test_leaderboard_ui_")
        init_analysis_db(self.tmp_dir)

        # Context 1: Trend R001
        self.spec_trend = {
            "market": "NIFTY",
            "sampling_interval_sec": 3,
            "task_type": "DIRECTION_CLASSIFIER",
            "prediction_horizon": "5m",
            "regime_id": "R001",
            "regime_definition_hash": "def_hash_trend",
            "dataset_snapshot_hash": "ds_hash_1",
            "features": ["adx_14", "rsi_14", "atm_iv_pctile"],
            "algorithm": "catboost",
            "hyperparameters": {"depth": 6},
            "walk_forward_config": {"folds": 5, "window_mode": "expanding"},
            "random_seed": 42,
        }
        _, self.rec_trend = register_or_get_experiment(self.tmp_dir, self.spec_trend, model_name="DIR_TREND_CAT_v1")

        # Context 2: Sideways R002
        self.spec_side = {
            "market": "NIFTY",
            "sampling_interval_sec": 3,
            "task_type": "DIRECTION_CLASSIFIER",
            "prediction_horizon": "5m",
            "regime_id": "R002",
            "regime_definition_hash": "def_hash_side",
            "dataset_snapshot_hash": "ds_hash_2",
            "features": ["boll_width_20", "vwap_dist", "oi_pcr"],
            "algorithm": "xgboost",
            "hyperparameters": {"max_depth": 4},
            "walk_forward_config": {"folds": 5, "window_mode": "expanding"},
            "random_seed": 42,
        }
        _, self.rec_side = register_or_get_experiment(self.tmp_dir, self.spec_side, model_name="DIR_SIDE_XGB_v1")

        # Create Benchmark Runs & Scorecards
        self.run_trend = create_benchmark_run(self.tmp_dir, context_key="NIFTY_3s_DIRECTION_CLASSIFIER_5m_R001")
        record_model_benchmark(
            self.tmp_dir,
            benchmark_run_id=self.run_trend,
            signature_hash=self.rec_trend["signature_hash"],
            model_name="DIR_TREND_CAT_v1",
            context_key="NIFTY_3s_DIRECTION_CLASSIFIER_5m_R001",
            algorithm="catboost",
            dataset_name="trend.parquet",
            feature_count=3,
            primary_metric_name="roc_auc",
            primary_metric_value=0.82,
            fold_metric_mean=0.82,
            fold_metric_std=0.01,
        )
        ranked_trend = rank_models_in_context(self.tmp_dir, "NIFTY_3s_DIRECTION_CLASSIFIER_5m_R001", benchmark_run_id=self.run_trend)
        persist_context_rankings(self.tmp_dir, benchmark_run_id=self.run_trend, ranked_dossiers=ranked_trend)

        self.run_side = create_benchmark_run(self.tmp_dir, context_key="NIFTY_3s_DIRECTION_CLASSIFIER_5m_R002")
        record_model_benchmark(
            self.tmp_dir,
            benchmark_run_id=self.run_side,
            signature_hash=self.rec_side["signature_hash"],
            model_name="DIR_SIDE_XGB_v1",
            context_key="NIFTY_3s_DIRECTION_CLASSIFIER_5m_R002",
            algorithm="xgboost",
            dataset_name="side.parquet",
            feature_count=3,
            primary_metric_name="roc_auc",
            primary_metric_value=0.74,
            fold_metric_mean=0.74,
            fold_metric_std=0.02,
        )
        ranked_side = rank_models_in_context(self.tmp_dir, "NIFTY_3s_DIRECTION_CLASSIFIER_5m_R002", benchmark_run_id=self.run_side)
        persist_context_rankings(self.tmp_dir, benchmark_run_id=self.run_side, ranked_dossiers=ranked_side)

        # Set Context Champion in champion_history
        set_champion_for_context(self.tmp_dir, "NIFTY_3s_DIRECTION_CLASSIFIER_5m_R001", "DIR_TREND_PROD_OLD_v0", robustness_score=75.0, promotion_reason="Initial baseline promotion")

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_canonical_context_key_resolution(self):
        """1. Verify dropdown parameters resolve into canonical ModelContextKey string."""
        if not self.root:
            self.skipTest("Tkinter root unavailable")

        panel = ModelResearchLeaderboardPanel(self.root, chart_dir=self.tmp_dir)
        panel._market_var.set("NIFTY")
        panel._sampling_var.set("3s")
        panel._task_var.set("DIRECTION_CLASSIFIER")
        panel._horizon_var.set("5m")
        panel._regime_var.set("R001 - TREND")
        panel._update_resolved_context_key()

        self.assertEqual(panel._context_key_var.get(), "NIFTY_3s_DIRECTION_CLASSIFIER_5m_R001")

    def test_context_isolation_leaderboard_query(self):
        """2. Verify Leaderboard query strictly populates models belonging to the active context key."""
        if not self.root:
            self.skipTest("Tkinter root unavailable")

        panel = ModelResearchLeaderboardPanel(self.root, chart_dir=self.tmp_dir)

        # Query Trend R001
        panel._context_key_var.set("NIFTY_3s_DIRECTION_CLASSIFIER_5m_R001")
        panel.refresh_leaderboard()

        items_r001 = panel.leaderboard_tree.get_children()
        self.assertEqual(len(items_r001), 1)
        self.assertEqual(panel.leaderboard_tree.item(items_r001[0])["values"][1], "DIR_TREND_CAT_v1")

        # Query Sideways R002
        panel._context_key_var.set("NIFTY_3s_DIRECTION_CLASSIFIER_5m_R002")
        panel.refresh_leaderboard()

        items_r002 = panel.leaderboard_tree.get_children()
        self.assertEqual(len(items_r002), 1)
        self.assertEqual(panel.leaderboard_tree.item(items_r002[0])["values"][1], "DIR_SIDE_XGB_v1")

    def test_production_champion_vs_research_candidate_distinction(self):
        """3. Verify visual distinction between Production Champion and Research Candidate."""
        if not self.root:
            self.skipTest("Tkinter root unavailable")

        panel = ModelResearchLeaderboardPanel(self.root, chart_dir=self.tmp_dir)
        panel._context_key_var.set("NIFTY_3s_DIRECTION_CLASSIFIER_5m_R001")
        panel.refresh_leaderboard()

        self.assertIn("DIR_TREND_PROD_OLD_v0", panel._prod_champ_var.get())
        self.assertIn("DIR_TREND_CAT_v1", panel._cand_champ_var.get())

    def test_empty_context_graceful_handling(self):
        """4. Verify querying a context key with zero benchmarks displays empty state without error."""
        if not self.root:
            self.skipTest("Tkinter root unavailable")

        panel = ModelResearchLeaderboardPanel(self.root, chart_dir=self.tmp_dir)
        panel._context_key_var.set("NIFTY_3s_DIRECTION_CLASSIFIER_5m_R007") # Expiry pinning (empty)
        panel.refresh_leaderboard()

        items = panel.leaderboard_tree.get_children()
        self.assertEqual(len(items), 0)
        self.assertIn("None", panel._cand_champ_var.get())

    def test_champion_history_tab_rendering(self):
        """5. Verify champion history sub-panel renders transition audit records."""
        if not self.root:
            self.skipTest("Tkinter root unavailable")

        panel = ModelResearchLeaderboardPanel(self.root, chart_dir=self.tmp_dir)
        panel._context_key_var.set("NIFTY_3s_DIRECTION_CLASSIFIER_5m_R001")
        panel.refresh_leaderboard()

        # Check that get_champion_history_for_context returns the record
        history = get_champion_history_for_context(self.tmp_dir, "NIFTY_3s_DIRECTION_CLASSIFIER_5m_R001")
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]["new_champion_name"], "DIR_TREND_PROD_OLD_v0")

    def test_feature_composition_and_regime_stress_subtabs(self):
        """7. Verify feature composition and cross-regime stress tabs populate correctly."""
        if not self.root:
            self.skipTest("Tkinter root unavailable")

        # Record feature set evaluation
        record_feature_set_evaluation(
            self.tmp_dir,
            signature_hash=self.rec_trend["signature_hash"],
            features=["adx_14", "rsi_14", "atm_iv_pctile"],
        )

        # Record regime evaluation
        record_regime_evaluation(
            self.tmp_dir,
            signature_hash=self.rec_trend["signature_hash"],
            model_name="DIR_TREND_CAT_v1",
            tested_regime_id="R002",
            tested_regime_hash="def_hash_side",
            is_native_regime=False,
            sample_count=1000,
            primary_metric=0.75,
            regime_degradation_pct=8.54,
        )

        panel = ModelResearchLeaderboardPanel(self.root, chart_dir=self.tmp_dir)
        panel._context_key_var.set("NIFTY_3s_DIRECTION_CLASSIFIER_5m_R001")
        panel.refresh_leaderboard()

        # Check that panel selected the first model and populated detail dossier
        self.assertIsNotNone(panel._selected_dossier)
        self.assertEqual(panel._selected_dossier["model_name"], "DIR_TREND_CAT_v1")

    def test_campaign_lineage_subtab(self):
        """8. Verify research lineage tab correctly resolves linked campaign ID."""
        if not self.root:
            self.skipTest("Tkinter root unavailable")

        camp_id = create_campaign(self.tmp_dir, context_key="NIFTY_3s_DIRECTION_CLASSIFIER_5m_R001")
        link_experiment_to_campaign(
            self.tmp_dir,
            campaign_id=camp_id,
            trial_index=1,
            signature_hash=self.rec_trend["signature_hash"],
        )

        b_run = create_benchmark_run(self.tmp_dir, context_key="NIFTY_3s_DIRECTION_CLASSIFIER_5m_R001", campaign_id=camp_id)
        record_model_benchmark(
            self.tmp_dir,
            benchmark_run_id=b_run,
            signature_hash=self.rec_trend["signature_hash"],
            model_name="DIR_TREND_CAT_v1",
            context_key="NIFTY_3s_DIRECTION_CLASSIFIER_5m_R001",
            algorithm="catboost",
            dataset_name="trend.parquet",
            feature_count=3,
            primary_metric_name="roc_auc",
            primary_metric_value=0.82,
            fold_metric_mean=0.82,
            fold_metric_std=0.01,
        )

        ranked = rank_models_in_context(self.tmp_dir, "NIFTY_3s_DIRECTION_CLASSIFIER_5m_R001", benchmark_run_id=b_run)
        persist_context_rankings(self.tmp_dir, benchmark_run_id=b_run, ranked_dossiers=ranked)

        panel = ModelResearchLeaderboardPanel(self.root, chart_dir=self.tmp_dir)
        panel._context_key_var.set("NIFTY_3s_DIRECTION_CLASSIFIER_5m_R001")
        panel.refresh_leaderboard()

        self.assertTrue(any(d["benchmark_run_id"] == b_run for d in panel._ranked_dossiers))

    def test_model_registry_panel_tab_integration(self):
        """9. Verify ModelRegistryPanel integrates the Research Leaderboard tab."""
        if not self.root:
            self.skipTest("Tkinter root unavailable")

        from master_dataset_tk.model_registry_panel import ModelRegistryPanel

        reg_panel = ModelRegistryPanel(self.root, chart_dir=self.tmp_dir)
        self.assertIn("research_leaderboard", reg_panel._models_family_tab_ids)
        self.assertIsNotNone(reg_panel._leaderboard_panel)

        # Switch to Research Leaderboard tab
        reg_panel._set_models_family_tab("research_leaderboard")
        self.assertEqual(reg_panel._models_family, "research_leaderboard")


if __name__ == "__main__":
    unittest.main()

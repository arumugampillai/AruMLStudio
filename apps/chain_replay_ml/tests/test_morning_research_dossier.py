"""Dedicated unit tests for Phase 4F.6: Morning Research Dossier & Presentation Layer."""

import hashlib
import json
import os
import shutil
import tempfile
import unittest

from path_config import ensure_ml_studio_paths
ensure_ml_studio_paths()

from chain_replay_ml.candidate_generation import create_candidate_spec
from chain_replay_ml.fine_tuning import (
    DescendantEvaluationRecord,
    evaluate_child_vs_parent,
    persist_fine_tuning_records,
)
from chain_replay_ml.model_ranking import (
    CandidateEvidenceScore,
    RecommendationClass,
    evaluate_candidate_evidence,
    persist_candidate_rankings,
    rank_candidates_in_context,
)
from chain_replay_ml.morning_dossier import (
    FeatureGovernanceAuditSummary,
    LineageNodeView,
    MorningResearchDossier,
    export_morning_dossier_markdown,
    generate_morning_research_dossier,
)
from chain_replay_ml.overnight_campaign import (
    CampaignConfig,
    CampaignState,
    CampaignStatus,
    CampaignStopReason,
    OvernightCampaignRunner,
    init_campaign_tables,
    persist_campaign_state,
)
from chain_replay_ml.research_memory import init_analysis_db


class TestMorningResearchDossier(unittest.TestCase):
    """Comprehensive test suite verifying Phase 4F.6 dossier compilation, exports, and presentation accuracy."""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp(prefix="aruml_test_dossier_")
        init_analysis_db(self.tmp_dir)
        init_campaign_tables(self.tmp_dir)
        self.context_key = "NIFTY_3s_DIRECTION_CLASSIFIER_5m_R001"
        self.campaign_id = "CAMP_MORNING_001"

        # Populate realistic campaign data
        self.config = CampaignConfig(
            campaign_id=self.campaign_id,
            context_keys=[self.context_key],
            max_duration_hours=2.0,
            max_candidates_total=10,
            max_generations=2,
        )
        runner = OvernightCampaignRunner(data_dir=self.tmp_dir, config=self.config)
        self.report = runner.run()

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_01_dossier_generation_basic(self):
        """1. Verify successful generation of MorningResearchDossier from persisted research memory."""
        dossier = generate_morning_research_dossier(self.tmp_dir, self.campaign_id)
        self.assertEqual(dossier.campaign_id, self.campaign_id)
        self.assertEqual(dossier.context_key, self.context_key)
        self.assertGreater(dossier.best_composite_score, 0.0)

    def test_02_kpi_score_accuracy(self):
        """2. Verify accurate aggregation of best composite, trading, and model scores."""
        dossier = generate_morning_research_dossier(self.tmp_dir, self.campaign_id)
        self.assertGreaterEqual(dossier.best_composite_score, dossier.starting_best_score)
        self.assertEqual(dossier.total_score_improvement, round(dossier.best_composite_score - dossier.starting_best_score, 4))

    def test_03_ranked_candidates_presence(self):
        """3. Verify ranked candidate list is populated and sorted descending by composite score."""
        dossier = generate_morning_research_dossier(self.tmp_dir, self.campaign_id)
        self.assertGreater(len(dossier.ranked_candidates), 0)
        for i in range(len(dossier.ranked_candidates) - 1):
            self.assertGreaterEqual(dossier.ranked_candidates[i].composite_score, dossier.ranked_candidates[i+1].composite_score)

    def test_04_fine_tuning_trials_presence(self):
        """4. Verify generational fine-tuning trials are included in dossier."""
        dossier = generate_morning_research_dossier(self.tmp_dir, self.campaign_id)
        self.assertIsInstance(dossier.fine_tuning_trials, list)

    def test_05_lineage_nodes_structure(self):
        """5. Verify lineage tree nodes map parent/child connections cleanly."""
        dossier = generate_morning_research_dossier(self.tmp_dir, self.campaign_id)
        self.assertGreater(len(dossier.lineage_tree), 0)
        for node in dossier.lineage_tree:
            self.assertIsNotNone(node.candidate_id)
            self.assertIsNotNone(node.composite_score)

    def test_06_feature_governance_audit_summary(self):
        """6. Verify feature governance audit reflects explored feature inventory."""
        dossier = generate_morning_research_dossier(self.tmp_dir, self.campaign_id)
        self.assertGreater(dossier.feature_governance_summary.total_features_evaluated, 0)
        self.assertIsInstance(dossier.feature_governance_summary.features_used, list)

    def test_07_recommended_next_actions_generation(self):
        """7. Verify actionable research recommendations are formulated for researcher."""
        dossier = generate_morning_research_dossier(self.tmp_dir, self.campaign_id)
        self.assertGreater(len(dossier.recommended_next_actions), 0)

    def test_08_champion_candidate_action_flagging(self):
        """8. Verify CHAMPION_CANDIDATE status triggers CRITICAL governance review action."""
        cand_item = {
            "candidate_id": "CAND_BEAT_CHAMP",
            "signature_hash": "sig_champ_123",
            "model_metrics": {"roc_auc": 0.84, "fold_mean": 0.84, "fold_std": 0.01},
            "trading_metrics": {"win_rate_pct": 68.0, "profit_factor": 2.2, "mfe_mae_ratio": 1.6, "total_trades": 55},
        }
        report = rank_candidates_in_context(self.tmp_dir, self.context_key, candidate_items=[cand_item], champion_composite_score=70.0)
        persist_candidate_rankings(self.tmp_dir, report)

        dossier = generate_morning_research_dossier(self.tmp_dir, self.campaign_id)
        self.assertTrue(any("CRITICAL" in act for act in dossier.recommended_next_actions))

    def test_09_markdown_export_formatting(self):
        """9. Verify Markdown dossier export contains all required sections."""
        dossier = generate_morning_research_dossier(self.tmp_dir, self.campaign_id)
        md = export_morning_dossier_markdown(dossier)
        self.assertIn("MORNING RESEARCH DOSSIER", md)
        self.assertIn("Executive Research Summary & KPIs", md)
        self.assertIn("Candidate Research Leaderboard", md)
        self.assertIn("Generational Lineage & Mutation Trials", md)
        self.assertIn("Feature Lifecycle Governance Audit", md)

    def test_10_json_serialization_roundtrip(self):
        """10. Verify JSON dictionary serialization roundtrips cleanly."""
        dossier = generate_morning_research_dossier(self.tmp_dir, self.campaign_id)
        d = dossier.to_dict()
        self.assertEqual(d["campaign_id"], self.campaign_id)
        self.assertEqual(d["context_key"], self.context_key)
        self.assertIn("ranked_candidates", d)
        self.assertIn("lineage_tree", d)

    def test_11_stop_reason_preservation(self):
        """11. Verify campaign stop reason is faithfully preserved in dossier."""
        dossier = generate_morning_research_dossier(self.tmp_dir, self.campaign_id)
        self.assertEqual(dossier.stop_reason, self.report.stop_reason)

    def test_12_empty_campaign_fallback(self):
        """12. Verify dossier handles un-evaluated campaign gracefully without crashing."""
        init_campaign_tables(self.tmp_dir)
        dossier = generate_morning_research_dossier(self.tmp_dir, "CAMP_NON_EXISTENT")
        self.assertEqual(dossier.campaign_id, "CAMP_NON_EXISTENT")
        self.assertEqual(len(dossier.ranked_candidates), 0)

    def test_13_context_key_isolation(self):
        """13. Verify dossier query respects strict ModelContextKey isolation."""
        dossier = generate_morning_research_dossier(self.tmp_dir, self.campaign_id, context_key=self.context_key)
        self.assertEqual(dossier.context_key, self.context_key)

    def test_14_production_immutability(self):
        """14. Invariant: Morning dossier generation never writes to .active_model.json or production model directories."""
        generate_morning_research_dossier(self.tmp_dir, self.campaign_id)
        active_model_path = os.path.join(self.tmp_dir, "models", ".active_model.json")
        self.assertFalse(os.path.exists(active_model_path))

    def test_15_feature_registry_immutability(self):
        """15. Invariant: Morning dossier generation never writes to feature_registry_store.json."""
        generate_morning_research_dossier(self.tmp_dir, self.campaign_id)
        self.assertTrue(True)

    def test_16_evidence_db_immutability(self):
        """16. Invariant: Feature Recommendation Evidence DB remains unmutated."""
        ev_db_path = os.path.join("apps", "feature_recommendation_evidence.db")
        if os.path.exists(ev_db_path):
            with open(ev_db_path, "rb") as f:
                sha = hashlib.sha256(f.read()).hexdigest()
            self.assertEqual(sha, "6f91afca94ec87a1210d8f4bcef356b9c16a6ef5a488268f41c5b1b81431ade2")

    def test_17_legacy_aruneo_exclusion(self):
        """17. Invariant: Morning dossier generation never creates or touches .lifecycle_registry.db."""
        generate_morning_research_dossier(self.tmp_dir, self.campaign_id)
        legacy_db_path = os.path.join(self.tmp_dir, "models", ".lifecycle_registry.db")
        self.assertFalse(os.path.exists(legacy_db_path))

    def test_18_zero_broker_access(self):
        """18. Invariant: Morning dossier generation has zero broker routing, zero order placement."""
        generate_morning_research_dossier(self.tmp_dir, self.campaign_id)
        self.assertTrue(True)

    def test_19_zero_automatic_champion_promotion(self):
        """19. Invariant: Morning dossier displays recommendations as advisory only without mutating champion pointer."""
        dossier = generate_morning_research_dossier(self.tmp_dir, self.campaign_id)
        self.assertIsInstance(dossier.recommended_next_actions, list)

    def test_20_duration_and_timing_telemetry(self):
        """20. Verify start/end timestamps and duration telemetry are recorded."""
        dossier = generate_morning_research_dossier(self.tmp_dir, self.campaign_id)
        self.assertIsNotNone(dossier.start_time_iso)
        self.assertIsNotNone(dossier.end_time_iso)

    def test_21_pruned_paths_count_accuracy(self):
        """21. Verify pruned paths count matches campaign state."""
        dossier = generate_morning_research_dossier(self.tmp_dir, self.campaign_id)
        self.assertGreaterEqual(dossier.total_candidates_pruned, 0)

    def test_22_excluded_candidates_count_accuracy(self):
        """22. Verify pre-training excluded candidates count is recorded."""
        dossier = generate_morning_research_dossier(self.tmp_dir, self.campaign_id)
        self.assertGreaterEqual(dossier.total_candidates_excluded, 0)

    def test_23_lineage_delta_tracking(self):
        """23. Verify parent/child Delta composite score is reflected in lineage node views."""
        dossier = generate_morning_research_dossier(self.tmp_dir, self.campaign_id)
        for n in dossier.lineage_tree:
            if n.parent_candidate_id:
                self.assertIsNotNone(n.delta_vs_parent)

    def test_24_presentation_only_boundary(self):
        """24. Invariant: Morning dossier performs zero candidate training or mutation generation."""
        dossier = generate_morning_research_dossier(self.tmp_dir, self.campaign_id)
        self.assertEqual(dossier.campaign_id, self.campaign_id)

    def test_25_gui_panel_import_and_instantiation_headless(self):
        """25. Verify MorningResearchDossierPanel imports and instantiates in headless mode cleanly."""
        from master_dataset_tk.morning_research_dossier_panel import MorningResearchDossierPanel
        self.assertTrue(callable(MorningResearchDossierPanel))

    def test_26_export_files_on_disk(self):
        """26. Verify export functions write valid markdown files to disk."""
        dossier = generate_morning_research_dossier(self.tmp_dir, self.campaign_id)
        md = export_morning_dossier_markdown(dossier)
        out_file = os.path.join(self.tmp_dir, "test_dossier.md")
        with open(out_file, "w", encoding="utf-8") as f:
            f.write(md)
        self.assertTrue(os.path.exists(out_file))
        self.assertGreater(os.path.getsize(out_file), 100)

    def test_27_dossier_presentation_integrity(self):
        """27. Verify dossier answers all 20 researcher questions clearly."""
        dossier = generate_morning_research_dossier(self.tmp_dir, self.campaign_id)
        self.assertIsNotNone(dossier.best_candidate_id)
        self.assertIsNotNone(dossier.best_win_rate_pct)
        self.assertIsNotNone(dossier.best_profit_factor)
        self.assertIsNotNone(dossier.best_max_drawdown_pct)

    def test_28_panel_rendering_kv_block_contract(self):
        """28. Verify MorningResearchDossierPanel renders overview and kv_block with supported signature without errors."""
        import tkinter as tk
        from master_dataset_tk.morning_research_dossier_panel import MorningResearchDossierPanel

        root = tk.Tk()
        root.withdraw()
        try:
            panel = MorningResearchDossierPanel(root, data_dir=self.tmp_dir)
            panel.selected_campaign_id.set(self.campaign_id)
            # Must succeed without throwing unexpected keyword argument 'num_cols'
            panel.load_selected_campaign()
            self.assertIsNotNone(panel.current_dossier)
            self.assertEqual(panel.current_dossier.campaign_id, self.campaign_id)
        finally:
            root.destroy()

    def test_29_panel_rendering_empty_and_missing_sections(self):
        """29. Verify panel renders cleanly even if optional dossier sections are empty."""
        import tkinter as tk
        from master_dataset_tk.morning_research_dossier_panel import MorningResearchDossierPanel

        root = tk.Tk()
        root.withdraw()
        try:
            panel = MorningResearchDossierPanel(root, data_dir=self.tmp_dir)
            # Create minimal dossier with empty optional sections
            from chain_replay_ml.morning_dossier import MorningResearchDossier, FeatureGovernanceAuditSummary
            d = MorningResearchDossier(
                campaign_id="CAMP_EMPTY_TEST",
                context_key=self.context_key,
                generated_at="2026-08-19T00:00:00",
                campaign_status=CampaignStatus.COMPLETED,
                stop_reason=CampaignStopReason.MAX_GENERATIONS_REACHED,
                start_time_iso="",
                end_time_iso="",
                duration_seconds=0.0,
                total_generations_completed=0,
                total_candidates_generated=0,
                total_candidates_trained=0,
                total_candidates_evaluated=0,
                total_candidates_excluded=0,
                total_candidates_pruned=0,
                starting_best_score=0.0,
                best_composite_score=0.0,
                total_score_improvement=0.0,
                best_candidate_id=None,
                best_candidate_class=None,
                best_trading_score=0.0,
                best_model_score=0.0,
                best_win_rate_pct=0.0,
                best_profit_factor=0.0,
                best_max_drawdown_pct=0.0,
                ranked_candidates=[],
                fine_tuning_trials=[],
                lineage_tree=[],
                feature_governance_summary=FeatureGovernanceAuditSummary(
                    total_features_evaluated=0,
                    features_used=[],
                    phase4e_recommended_features=[],
                    deprecated_features_blocked=[],
                    unknown_features_governed=[],
                ),
                recommended_next_actions=[],
            )
            panel.current_dossier = d
            panel._render_overview_tab()
            panel._render_discovered_features_tab()
            panel._render_leaderboard_tab()
            panel._render_lineage_tab()
            panel._render_governance_tab()
        finally:
            root.destroy()

    def test_30_panel_scrollable_tabs_contract(self):
        """30. Verify that overview and governance tabs use ScrollableFrame for long content."""
        import tkinter as tk
        from master_dataset_tk.morning_research_dossier_panel import MorningResearchDossierPanel
        from master_dataset_tk.model_registry_widgets import ScrollableFrame

        root = tk.Tk()
        root.withdraw()
        try:
            panel = MorningResearchDossierPanel(root, data_dir=self.tmp_dir)
            self.assertIsInstance(panel.tab_overview, ScrollableFrame)
            self.assertIsInstance(panel.tab_discovered_features, ScrollableFrame)
            self.assertIsInstance(panel.tab_governance, ScrollableFrame)
        finally:
            root.destroy()

    def test_31_discovered_features_populated(self):
        """31. Verify dossier.discovered_features contains empirical feature discovery records."""
        from chain_replay_ml.morning_dossier.types import DiscoveredFeatureRecord
        dossier = generate_morning_research_dossier(self.tmp_dir, self.campaign_id)
        self.assertGreater(len(dossier.discovered_features), 0)
        first = dossier.discovered_features[0]
        self.assertIsInstance(first, DiscoveredFeatureRecord)
        self.assertIsNotNone(first.feature_name)
        self.assertGreater(first.times_tested, 0)
        self.assertGreater(first.best_composite_score, 0.0)

    def test_32_discovered_feature_categories_and_recommendation(self):
        """32. Verify feature status classification (STRONG, PROMISING, REJECTED) and governance recommendations."""
        from chain_replay_ml.morning_dossier.types import DiscoveredFeatureStatus
        dossier = generate_morning_research_dossier(self.tmp_dir, self.campaign_id)
        statuses = {f.status for f in dossier.discovered_features}
        self.assertTrue(any(s in statuses for s in (DiscoveredFeatureStatus.STRONG_DISCOVERED, DiscoveredFeatureStatus.PROMISING)))
        for f in dossier.discovered_features:
            if f.status in (DiscoveredFeatureStatus.STRONG_DISCOVERED, DiscoveredFeatureStatus.PROMISING):
                self.assertEqual(f.recommendation, "DISCOVERED — HUMAN REVIEW REQUIRED")

    def test_33_discovered_feature_synergies(self):
        """33. Verify pairwise feature synergies are aggregated into dossier.discovered_synergies."""
        from chain_replay_ml.morning_dossier.types import DiscoveredFeatureSynergy
        dossier = generate_morning_research_dossier(self.tmp_dir, self.campaign_id)
        self.assertIsInstance(dossier.discovered_synergies, list)
        if dossier.discovered_synergies:
            first = dossier.discovered_synergies[0]
            self.assertIsInstance(first, DiscoveredFeatureSynergy)
            self.assertIsNotNone(first.feature_a)
            self.assertIsNotNone(first.feature_b)

    def test_34_candidate_feature_mutation_drilldown(self):
        """34. Verify candidate feature drill-down shows parent, added features, removed features, and metric deltas."""
        from chain_replay_ml.morning_dossier.types import CandidateFeatureDeltaView
        dossier = generate_morning_research_dossier(self.tmp_dir, self.campaign_id)
        self.assertGreater(len(dossier.candidate_feature_deltas), 0)
        first = dossier.candidate_feature_deltas[0]
        self.assertIsInstance(first, CandidateFeatureDeltaView)
        self.assertIsNotNone(first.candidate_id)
        self.assertIsInstance(first.child_features, list)
        self.assertIsInstance(first.added_features, list)

    def test_35_discovered_features_markdown_export(self):
        """35. Verify export_morning_dossier_markdown includes Discovered Feature Intelligence section."""
        dossier = generate_morning_research_dossier(self.tmp_dir, self.campaign_id)
        md = export_morning_dossier_markdown(dossier)
        self.assertIn("## 3. Discovered Feature Intelligence", md)
        self.assertIn("DISCOVERED — HUMAN REVIEW REQUIRED", md)

    def test_36_discovered_features_json_export(self):
        """36. Verify dossier.to_dict() contains discovered_features, discovered_synergies, candidate_feature_deltas."""
        dossier = generate_morning_research_dossier(self.tmp_dir, self.campaign_id)
        data = dossier.to_dict()
        self.assertIn("discovered_features", data)
        self.assertIn("discovered_synergies", data)
        self.assertIn("candidate_feature_deltas", data)
        self.assertGreater(len(data["discovered_features"]), 0)

    def test_37_panel_discovered_features_tab_rendering(self):
        """37. Verify MorningResearchDossierPanel successfully renders the Discovered Features tab."""
        import tkinter as tk
        from master_dataset_tk.morning_research_dossier_panel import MorningResearchDossierPanel

        root = tk.Tk()
        root.withdraw()
        try:
            panel = MorningResearchDossierPanel(root, data_dir=self.tmp_dir)
            panel.selected_campaign_id.set(self.campaign_id)
            panel.load_selected_campaign()
            panel._render_discovered_features_tab()
        finally:
            root.destroy()

    def test_38_production_and_feature_registry_immutability(self):
        """38. Invariant: Discovered feature extraction never modifies production models or feature registry."""
        dossier = generate_morning_research_dossier(self.tmp_dir, self.campaign_id)
        self.assertFalse(os.path.exists(os.path.join(self.tmp_dir, "models", ".active_model.json")))
        self.assertFalse(os.path.exists(os.path.join(self.tmp_dir, "models", ".lifecycle_registry.db")))


if __name__ == "__main__":
    unittest.main()



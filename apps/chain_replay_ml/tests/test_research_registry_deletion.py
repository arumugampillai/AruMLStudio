"""Dedicated unit & smoke tests for Research Registry deletion & multi-selection."""

import os
import shutil
import sqlite3
import tempfile
import tkinter as tk
from unittest.mock import patch
import unittest

from chain_replay_ml.morning_dossier import generate_morning_research_dossier
from chain_replay_ml.overnight_campaign.persistence import init_campaign_tables
from chain_replay_ml.research_memory.db import connect_analysis_db, init_analysis_db
from chain_replay_ml.research_registry.store import (
    ResearchRegistryRecord,
    ResearchStatus,
    delete_research_records,
    get_all_research_records,
    init_research_registry_tables,
    insert_or_update_research_run,
)
from master_dataset_tk.morning_research_dossier_panel import MorningResearchDossierPanel


class TestResearchRegistryDeletion(unittest.TestCase):
    """Verify single and multi-selection deletion, referential integrity, and UI state synchronization."""

    def setUp(self) -> None:
        self.tmp_dir = tempfile.mkdtemp(prefix="aruml_test_del_")
        init_analysis_db(self.tmp_dir)
        init_campaign_tables(self.tmp_dir)
        init_research_registry_tables(self.tmp_dir)

        conn = connect_analysis_db(self.tmp_dir)
        try:
            # Seed 3 research runs: 2 with unique pipelines, 1 with a shared pipeline
            conn.execute("""
                CREATE TABLE IF NOT EXISTS overnight_campaign_events (
                    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    campaign_id TEXT,
                    generation_number INTEGER,
                    event_type TEXT,
                    candidate_id TEXT,
                    message TEXT,
                    event_details_json TEXT,
                    created_at TEXT
                );
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS campaign_candidate_specs (
                    candidate_id TEXT PRIMARY KEY,
                    signature_hash TEXT,
                    context_key TEXT,
                    algorithm TEXT,
                    features_json TEXT,
                    hyperparameters_json TEXT,
                    parent_candidate_id TEXT,
                    mutation_type TEXT,
                    mutation_description TEXT,
                    campaign_id TEXT,
                    created_at TEXT,
                    feature_elimination_strategy TEXT
                );
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS discovery_pipelines (
                    pipeline_id TEXT PRIMARY KEY,
                    campaign_id TEXT,
                    context_key TEXT,
                    dataset_name TEXT,
                    dataset_snapshot_hash TEXT,
                    base_feature_count INTEGER,
                    base_feature_names_json TEXT,
                    base_pipeline_id TEXT,
                    base_pipeline_snapshot_hash TEXT,
                    active_features_count INTEGER,
                    total_generated_count INTEGER,
                    parent_snapshot_hash TEXT,
                    current_snapshot_hash TEXT,
                    current_generation INTEGER,
                    status TEXT,
                    budget_json TEXT,
                    created_at TEXT,
                    updated_at TEXT
                );
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS discovery_pipeline_features (
                    feature_id TEXT PRIMARY KEY,
                    pipeline_id TEXT,
                    feature_name TEXT,
                    formula_expression TEXT,
                    formula_hash TEXT,
                    generator_strategy TEXT,
                    parent_features_json TEXT,
                    generation_discovered INTEGER,
                    lifecycle_status TEXT,
                    evidence_score REAL,
                    total_evaluations INTEGER,
                    holdout_rank INTEGER,
                    relative_imp_drop REAL,
                    drift_severity INTEGER,
                    ks_statistic REAL,
                    ks_pvalue REAL,
                    metadata_json TEXT,
                    created_at TEXT,
                    updated_at TEXT
                );
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS discovery_pipeline_snapshots (
                    snapshot_hash TEXT PRIMARY KEY,
                    pipeline_id TEXT,
                    generation_number INTEGER,
                    active_feature_names_json TEXT,
                    feature_count INTEGER,
                    keep_count INTEGER,
                    watch_count INTEGER,
                    remove_count INTEGER,
                    created_at TEXT
                );
            """)

            # Insert test records
            for i in range(1, 4):
                c_id = f"CAMP_TEST_00{i}"
                r_id = f"RESEARCH_TEST_00{i}"
                dp_id = "DP_SHARED" if i in (2, 3) else f"DP_TEST_00{i}"

                # overnight_campaigns
                conn.execute(
                    """
                    INSERT INTO overnight_campaigns (
                        campaign_id, config_hash, config_json, status, stop_reason,
                        current_generation, total_candidates_generated, total_candidates_trained,
                        total_candidates_evaluated, total_candidates_excluded, total_candidates_pruned,
                        total_failures, best_candidate_id, best_signature_hash,
                        best_composite_score, best_trading_score, best_model_score,
                        starting_best_score, start_time_iso, last_update_iso, end_time_iso,
                        warnings_json, feature_elimination_strategy
                    ) VALUES (
                        ?, ?, ?, ?, ?,
                        1, 5, 5,
                        5, 0, 0,
                        0, ?, 'sig_best',
                        85.5, 80.0, 91.0,
                        70.0, ?, ?, ?,
                        '[]', 'SHAP_AND_EVIDENCE'
                    );
                    """,
                    (c_id, f"hash_{c_id}", "{}", "COMPLETED", "MAX_GENERATIONS_REACHED", f"CAND_{c_id}", "2026-08-20T10:00:00Z", "2026-08-20T10:30:00Z", "2026-08-20T10:30:00Z"),
                )
                # overnight_campaign_events
                conn.execute(
                    "INSERT INTO overnight_campaign_events (campaign_id, generation_number, event_type, message, event_details_json, created_at) VALUES (?, ?, ?, ?, ?, ?);",
                    (c_id, 1, "CANDIDATE_GENERATED", f"Candidate generated in {c_id}", "{}", "2026-08-20T10:05:00Z"),
                )
                # campaign_candidate_specs
                conn.execute(
                    """
                    INSERT INTO campaign_candidate_specs (
                        candidate_id, signature_hash, context_key, algorithm,
                        features_json, hyperparameters_json, parent_candidate_id,
                        mutation_type, mutation_description, campaign_id,
                        created_at, feature_elimination_strategy
                    ) VALUES (
                        ?, ?, ?, ?,
                        '[]', '{}', NULL,
                        'NONE', 'Initial spec', ?,
                        '2026-08-20T10:05:00Z', 'SHAP_AND_EVIDENCE'
                    );
                    """,
                    (f"CAND_{c_id}", f"sig_{c_id}", "NIFTY:5m", "XGBoost", c_id),
                )
                # discovery_pipelines & features
                conn.execute(
                    """
                    INSERT OR REPLACE INTO discovery_pipelines (
                        pipeline_id, campaign_id, context_key, dataset_name, dataset_snapshot_hash,
                        base_feature_count, base_feature_names_json, base_pipeline_id, base_pipeline_snapshot_hash,
                        active_features_count, total_generated_count, parent_snapshot_hash, current_snapshot_hash,
                        current_generation, status, budget_json, created_at, updated_at
                    ) VALUES (
                        ?, ?, 'NIFTY:5m', 'ds_test', 'snap_001',
                        171, '[]', 'PL_0001', 'hash_001',
                        1, 1, 'snap_000', ?,
                        1, 'active', '{}', '2026-08-20T10:00:00Z', '2026-08-20T10:30:00Z'
                    );
                    """,
                    (dp_id, c_id, f"SNAP_{dp_id}"),
                )
                conn.execute(
                    """
                    INSERT OR REPLACE INTO discovery_pipeline_features (
                        feature_id, pipeline_id, feature_name, formula_expression, formula_hash,
                        generator_strategy, parent_features_json, generation_discovered,
                        lifecycle_status, evidence_score, total_evaluations, holdout_rank,
                        relative_imp_drop, drift_severity, ks_statistic, ks_pvalue,
                        metadata_json, created_at, updated_at
                    ) VALUES (
                        ?, ?, ?, 'f(x)', 'hash_f',
                        'ARITHMETIC', '[]', 1,
                        'KEEP', 85.0, 1, 1,
                        0.0, 0, 0.01, 0.99,
                        '{}', '2026-08-20T10:00:00Z', '2026-08-20T10:30:00Z'
                    );
                    """,
                    (f"FEAT_{dp_id}_{i}", dp_id, f"feat_{dp_id}_{i}"),
                )
                conn.execute(
                    """
                    INSERT OR REPLACE INTO discovery_pipeline_snapshots (
                        snapshot_hash, pipeline_id, generation_number,
                        active_feature_names_json, feature_count,
                        keep_count, watch_count, remove_count, created_at
                    ) VALUES (
                        ?, ?, 1,
                        '[]', 1,
                        1, 0, 0, '2026-08-20T10:00:00Z'
                    );
                    """,
                    (f"SNAP_{dp_id}_{i}", dp_id),
                )

            conn.commit()
        finally:
            conn.close()

        # Insert research_registry records
        for i in range(1, 4):
            c_id = f"CAMP_TEST_00{i}"
            r_id = f"RESEARCH_TEST_00{i}"
            dp_id = "DP_SHARED" if i in (2, 3) else f"DP_TEST_00{i}"

            rec = ResearchRegistryRecord(
                research_id=r_id,
                campaign_id=c_id,
                context_key="NIFTY:6:standard:all",
                context_id="ctx_001",
                dataset_name="ds_test",
                dataset_snapshot_hash="snap_001",
                base_pipeline_id="PL_0001",
                base_feature_count=171,
                registry_feature_count=211,
                started_at="2026-08-20T10:00:00Z",
                finished_at="2026-08-20T10:30:00Z",
                duration_seconds=1800.0,
                status=ResearchStatus.COMPLETED,
                stop_reason="MAX_GENERATIONS_REACHED",
                algorithms_used=["XGBoost"],
                elimination_strategy="SHAP_AND_EVIDENCE",
                max_generations_configured=10,
                actual_generations_completed=1,
                max_candidates_configured=50,
                candidates_generated=5,
                candidates_evaluated=5,
                candidates_pruned=0,
                best_candidate_id=f"CAND_{c_id}",
                best_composite_score=85.5,
                best_trading_score=80.0,
                best_model_score=91.0,
                starting_best_score=70.0,
                total_score_lift=15.5,
                discovery_pipeline_id=dp_id,
                final_discovery_snapshot_hash=f"SNAP_{dp_id}",
                total_df_features_created=1,
                unique_formula_count=1,
                keep_count=1,
                watch_count=0,
                remove_count=0,
                active_discovery_pool=1,
                promoted_feature_count=0,
                research_config_json="{}",
                research_outcome_json="{}",
                failure_reason=None,
                architecture_version="2.2.0",
                code_version="1.0.0",
                created_at="2026-08-20T10:00:00Z",
                updated_at="2026-08-20T10:30:00Z",
            )
            insert_or_update_research_run(self.tmp_dir, rec)

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_single_research_deletion(self) -> None:
        """1. Verify deleting a single research run removes uniquely associated records while preserving others."""
        initial_records = get_all_research_records(self.tmp_dir)
        self.assertEqual(len(initial_records), 3)

        del_res = delete_research_records(self.tmp_dir, ["RESEARCH_TEST_001"])
        self.assertEqual(del_res["research_registry"], 1)
        self.assertEqual(del_res["overnight_campaigns"], 1)
        self.assertEqual(del_res["overnight_campaign_events"], 1)
        self.assertEqual(del_res["campaign_candidate_specs"], 1)
        self.assertEqual(del_res["discovery_pipelines"], 1)
        self.assertEqual(del_res["discovery_pipeline_features"], 1)

        after_records = get_all_research_records(self.tmp_dir)
        self.assertEqual(len(after_records), 2)
        remaining_ids = {r["research_id"] for r in after_records}
        self.assertEqual(remaining_ids, {"RESEARCH_TEST_002", "RESEARCH_TEST_003"})

    def test_multi_research_deletion_and_shared_pipeline_protection(self) -> None:
        """2. Verify deleting RESEARCH_TEST_002 does NOT delete DP_SHARED when RESEARCH_TEST_003 still references it."""
        conn = connect_analysis_db(self.tmp_dir)
        try:
            # Delete only run 2
            delete_research_records(self.tmp_dir, ["RESEARCH_TEST_002"])

            # DP_SHARED must still exist because run 3 references it
            dp_cnt = conn.execute("SELECT COUNT(*) as c FROM discovery_pipelines WHERE pipeline_id = 'DP_SHARED';").fetchone()["c"]
            self.assertEqual(dp_cnt, 1)

            # Now delete run 3
            delete_research_records(self.tmp_dir, ["RESEARCH_TEST_003"])

            # DP_SHARED should now be cleaned up
            dp_cnt_after = conn.execute("SELECT COUNT(*) as c FROM discovery_pipelines WHERE pipeline_id = 'DP_SHARED';").fetchone()["c"]
            self.assertEqual(dp_cnt_after, 0)
        finally:
            conn.close()

    def test_ui_selection_and_delete_button_states(self) -> None:
        """3. Verify UI single-selection, Shift/Ctrl multi-selection, and Delete button state transitions."""
        root = tk.Tk()
        root.withdraw()
        try:
            panel = MorningResearchDossierPanel(root, data_dir=self.tmp_dir)
            self.assertEqual(len(panel._all_records), 3)

            # Initially nothing selected -> Delete button disabled
            self.assertEqual(str(panel.delete_btn["state"]), "disabled")
            self.assertEqual(panel.delete_btn.cget("text"), "Delete Research")

            # 1. Single selection
            panel.tree.selection_set(["RESEARCH_TEST_001"])
            panel._on_selection_changed()
            self.assertEqual(str(panel.delete_btn["state"]), "normal")
            self.assertEqual(panel.delete_btn.cget("text"), "🗑️ Delete Selected (1)")

            # 2. Shift / Multi selection (2 items)
            panel.tree.selection_set(["RESEARCH_TEST_001", "RESEARCH_TEST_002"])
            panel._on_selection_changed()
            self.assertEqual(str(panel.delete_btn["state"]), "normal")
            self.assertEqual(panel.delete_btn.cget("text"), "🗑️ Delete Selected (2)")

            # 3. Multi selection (3 items)
            panel.tree.selection_set(["RESEARCH_TEST_001", "RESEARCH_TEST_002", "RESEARCH_TEST_003"])
            panel._on_selection_changed()
            self.assertEqual(str(panel.delete_btn["state"]), "normal")
            self.assertEqual(panel.delete_btn.cget("text"), "🗑️ Delete Selected (3)")

            # 4. Clear selection
            panel.tree.selection_set([])
            panel._on_selection_changed()
            self.assertEqual(str(panel.delete_btn["state"]), "disabled")
            self.assertEqual(panel.delete_btn.cget("text"), "Delete Research")
        finally:
            root.destroy()

    def test_ui_delete_confirmation_flow(self) -> None:
        """4. Verify Delete confirmation dialog: rejection cancels deletion; acceptance deletes & refreshes."""
        root = tk.Tk()
        root.withdraw()
        try:
            panel = MorningResearchDossierPanel(root, data_dir=self.tmp_dir)
            panel.tree.selection_set(["RESEARCH_TEST_001", "RESEARCH_TEST_002"])
            panel._on_selection_changed()

            # A. User clicks 'No' on confirmation dialog
            with patch("tkinter.messagebox.askyesno", return_value=False) as mock_confirm:
                panel._on_delete_clicked()
                mock_confirm.assert_called_once()
                # Records should remain 3
                self.assertEqual(len(get_all_research_records(self.tmp_dir)), 3)

            # B. User clicks 'Yes' on confirmation dialog
            with patch("tkinter.messagebox.askyesno", return_value=True) as mock_confirm, \
                 patch("tkinter.messagebox.showinfo") as mock_info:
                panel._on_delete_clicked()
                mock_confirm.assert_called_once()
                mock_info.assert_called_once()

                # Records should be reduced to 1 (RESEARCH_TEST_003)
                remaining = get_all_research_records(self.tmp_dir)
                self.assertEqual(len(remaining), 1)
                self.assertEqual(remaining[0]["research_id"], "RESEARCH_TEST_003")

                # UI tree & delete button should be refreshed
                self.assertEqual(len(panel._all_records), 1)
                self.assertEqual(len(panel.tree.get_children()), 1)
                self.assertEqual(str(panel.delete_btn["state"]), "disabled")
        finally:
            root.destroy()


if __name__ == "__main__":
    unittest.main()

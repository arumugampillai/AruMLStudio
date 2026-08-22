"""Unit and integration tests for Phase 4F.7: Autonomous Research Registry (Doc 16)."""

from datetime import datetime, timezone
import json
import os
import shutil
import sqlite3
import tempfile
import tkinter as tk
from tkinter import ttk
import unittest

from chain_replay_ml.research_registry.types import (
    FormulaGlobalStatus,
    FormulaMemoryRecord,
    ResearchGenerationLinkage,
    ResearchRegistryRecord,
    ResearchStatus,
)
from chain_replay_ml.research_registry.store import (
    backfill_historical_research_records,
    generate_research_id,
    get_all_research_records,
    get_research_detail,
    init_research_registry_tables,
    insert_or_update_research_run,
    record_generation_linkage,
)
from chain_replay_ml.research_registry.memory import (
    get_blacklisted_formula_hashes,
    update_formula_memory_from_discovery,
)
from chain_replay_ml.overnight_campaign.persistence import (
    init_campaign_tables,
    persist_campaign_state,
)
from chain_replay_ml.overnight_campaign.types import (
    CampaignConfig,
    CampaignState,
    CampaignStatus,
    CampaignStopReason,
)
from chain_replay_ml.research_memory.db import connect_analysis_db


class TestAutonomousResearchRegistryService(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        init_research_registry_tables(self.tmp_dir)

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_research_id_generation(self):
        """1. Verify Research ID generation format and determinism."""
        r_id1 = generate_research_id("NIFTY:6:standard:all", "2026-08-22T00:29:13Z")
        r_id2 = generate_research_id("NIFTY:6:standard:all", "2026-08-22T00:29:13Z")
        self.assertEqual(r_id1, r_id2)
        self.assertTrue(r_id1.startswith("RESEARCH_NIFTY_6_standard_all_20260822_002913_"))

    def test_insert_and_query_research_run(self):
        """2. Verify insert and query of ResearchRegistryRecord."""
        rec = ResearchRegistryRecord(
            research_id="RESEARCH_TEST_001",
            campaign_id="CAMP_TEST_001",
            context_key="NIFTY:6:standard:all",
            context_id="ctx_169e8ab4c718",
            dataset_name="analysis_test.parquet",
            dataset_snapshot_hash="snap_123",
            started_at="2026-08-22T00:00:00Z",
            status=ResearchStatus.RUNNING,
            max_generations_configured=100,
            actual_generations_completed=5,
            best_candidate_id="CAND_005",
            best_composite_score=78.5,
            discovery_pipeline_id="DP_CAMP_TEST_001",
            total_df_features_created=25,
            keep_count=4,
            watch_count=6,
            remove_count=15,
            active_discovery_pool=10,
        )
        insert_or_update_research_run(self.tmp_dir, rec)

        records = get_all_research_records(self.tmp_dir)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["research_id"], "RESEARCH_TEST_001")
        self.assertEqual(records[0]["active_discovery_pool"], 10)
        self.assertEqual(records[0]["best_candidate_id"], "CAND_005")

        # Update record
        updated_rec = ResearchRegistryRecord(
            research_id="RESEARCH_TEST_001",
            campaign_id="CAMP_TEST_001",
            context_key="NIFTY:6:standard:all",
            context_id="ctx_169e8ab4c718",
            dataset_name="analysis_test.parquet",
            dataset_snapshot_hash="snap_123",
            started_at="2026-08-22T00:00:00Z",
            finished_at="2026-08-22T01:00:00Z",
            duration_seconds=3600.0,
            status=ResearchStatus.COMPLETED,
            stop_reason="MAX_GENERATIONS_REACHED",
            actual_generations_completed=100,
            best_candidate_id="CAND_099",
            best_composite_score=85.2,
            discovery_pipeline_id="DP_CAMP_TEST_001",
            total_df_features_created=2840,
            keep_count=43,
            watch_count=87,
            remove_count=2710,
            active_discovery_pool=130,
        )
        insert_or_update_research_run(self.tmp_dir, updated_rec)

        detail = get_research_detail(self.tmp_dir, "RESEARCH_TEST_001")
        self.assertIsNotNone(detail)
        self.assertEqual(detail["status"], "COMPLETED")
        self.assertEqual(detail["active_discovery_pool"], 130)
        self.assertEqual(detail["best_composite_score"], 85.2)

    def test_generational_linkage_recording(self):
        """3. Verify generational snapshot linkage persistence."""
        rec = ResearchRegistryRecord(
            research_id="RESEARCH_TEST_001",
            campaign_id="CAMP_TEST_001",
            context_key="NIFTY:6:standard:all",
            context_id="ctx_169e8ab4c718",
            dataset_name="analysis_test.parquet",
            dataset_snapshot_hash="snap_123",
            started_at="2026-08-22T00:00:00Z",
        )
        insert_or_update_research_run(self.tmp_dir, rec)

        record_generation_linkage(
            self.tmp_dir,
            research_id="RESEARCH_TEST_001",
            campaign_id="CAMP_TEST_001",
            generation_number=1,
            discovery_snapshot_hash="DP_SNAP_abc1",
            candidates_evaluated=10,
            generation_best_score=65.0,
            generation_best_candidate_id="CAND_001",
        )
        record_generation_linkage(
            self.tmp_dir,
            research_id="RESEARCH_TEST_001",
            campaign_id="CAMP_TEST_001",
            generation_number=2,
            discovery_snapshot_hash="DP_SNAP_abc2",
            candidates_evaluated=25,
            generation_best_score=72.4,
            generation_best_candidate_id="CAND_015",
        )

        detail = get_research_detail(self.tmp_dir, "RESEARCH_TEST_001")
        gens = detail.get("generations", [])
        self.assertEqual(len(gens), 2)
        self.assertEqual(gens[0]["discovery_snapshot_hash"], "DP_SNAP_abc1")
        self.assertEqual(gens[1]["discovery_snapshot_hash"], "DP_SNAP_abc2")
        self.assertEqual(gens[1]["generation_best_score"], 72.4)

    def test_formula_memory_and_drift_blacklisting(self):
        """4. Verify formula memory tracking and severe drift blacklist."""
        from chain_replay_ml.discovery_pipeline.persistence import (
            init_discovery_pipeline_tables,
            persist_discovery_pipeline,
            persist_discovered_features,
        )
        from chain_replay_ml.discovery_pipeline.types import (
            DiscoveryPipelineSpec,
            DiscoveredFeatureSpec,
            DiscoveryLifecycleStatus,
            GeneratorStrategy,
        )

        init_discovery_pipeline_tables(self.tmp_dir)
        pipe = DiscoveryPipelineSpec(
            pipeline_id="DP_CAMP_001",
            campaign_id="CAMP_001",
            context_key="NIFTY:6:standard:all",
            dataset_name="test_ds",
            dataset_snapshot_hash="snap_123",
            base_feature_count=171,
        )
        persist_discovery_pipeline(self.tmp_dir, pipe)

        f1 = DiscoveredFeatureSpec(
            feature_id="DF_001",
            pipeline_id="DP_CAMP_001",
            feature_name="DF_A_div_B",
            formula_expression="col(A)/col(B)",
            formula_hash="HASH_KEEP",
            generator_strategy=GeneratorStrategy.RATIO,
            parent_features=["A", "B"],
            generation_discovered=1,
            lifecycle_status=DiscoveryLifecycleStatus.KEEP,
            evidence_score=55.0,
            ks_statistic=0.05,
            drift_severity=0,
            metadata={"delta_auc": 0.002, "governance_rationale": "Strong marginal lift"},
        )
        f2 = DiscoveredFeatureSpec(
            feature_id="DF_002",
            pipeline_id="DP_CAMP_001",
            feature_name="DF_C_mul_D",
            formula_expression="col(C)*col(D)",
            formula_hash="HASH_DRIFT",
            generator_strategy=GeneratorStrategy.INTERACTION,
            parent_features=["C", "D"],
            generation_discovered=1,
            lifecycle_status=DiscoveryLifecycleStatus.REMOVE,
            evidence_score=30.0,
            ks_statistic=0.52,
            drift_severity=2,
            metadata={"delta_auc": -0.001, "governance_rationale": "Severe KS distribution drift"},
        )
        persist_discovered_features(self.tmp_dir, [f1, f2])

        updated = update_formula_memory_from_discovery(
            self.tmp_dir,
            research_id="RESEARCH_001",
            campaign_id="CAMP_001",
            context_key="NIFTY:6:standard:all",
        )
        self.assertEqual(updated, 2)

        # Check blacklist
        bl = get_blacklisted_formula_hashes(self.tmp_dir, "NIFTY:6:standard:all")
        self.assertIn("HASH_DRIFT", bl)
        self.assertNotIn("HASH_KEEP", bl)

    def test_backfill_historical_campaigns(self):
        """5. Verify historical campaigns backfill into research_registry."""
        init_campaign_tables(self.tmp_dir)
        conn = connect_analysis_db(self.tmp_dir)
        conn.execute("""
            INSERT INTO overnight_campaigns (
                campaign_id, config_hash, config_json, status, stop_reason,
                current_generation, total_candidates_generated, total_candidates_trained,
                total_candidates_evaluated, total_candidates_excluded, total_candidates_pruned,
                total_failures, best_candidate_id, best_signature_hash, best_composite_score,
                best_trading_score, best_model_score, starting_best_score, start_time_iso,
                last_update_iso, end_time_iso, warnings_json
            ) VALUES (
                'CAMP_HISTORICAL_001', 'cfg_hash', '{"context_key":"NIFTY:6:standard:all"}', 'completed', 'MAX_GENERATIONS_REACHED',
                10, 50, 50, 50, 0, 10, 0, 'CAND_042', 'sig', 81.2, 75.0, 80.0, 60.0, '2026-08-20T10:00:00Z', '2026-08-20T12:00:00Z', '2026-08-20T12:00:00Z', '[]'
            );
        """)
        conn.commit()
        conn.close()

        count = backfill_historical_research_records(self.tmp_dir)
        self.assertEqual(count, 1)

        records = get_all_research_records(self.tmp_dir)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["campaign_id"], "CAMP_HISTORICAL_001")
        self.assertEqual(records[0]["status"], "COMPLETED")
        self.assertEqual(records[0]["best_composite_score"], 81.2)

    def test_morning_dossier_research_registry_tab_ui(self):
        """6. Verify Morning Research Dossier panel creates and populates Research Registry tab."""
        root = tk.Tk()
        root.withdraw()
        try:
            from apps.master_dataset_tk.morning_research_dossier_panel import MorningResearchDossierPanel
            init_campaign_tables(self.tmp_dir)
            conn = connect_analysis_db(self.tmp_dir)
            conn.execute("""
                INSERT INTO overnight_campaigns (
                    campaign_id, config_hash, config_json, status, stop_reason,
                    current_generation, total_candidates_generated, total_candidates_trained,
                    total_candidates_evaluated, total_candidates_excluded, total_candidates_pruned,
                    total_failures, best_candidate_id, best_signature_hash, best_composite_score,
                    best_trading_score, best_model_score, starting_best_score, start_time_iso,
                    last_update_iso, end_time_iso, warnings_json
                ) VALUES (
                    'CAMP_UI_001', 'cfg_hash', '{"context_key":"NIFTY:6:standard:all"}', 'completed', 'MAX_GENERATIONS_REACHED',
                    5, 25, 25, 25, 0, 5, 0, 'CAND_010', 'sig', 88.0, 80.0, 85.0, 60.0, '2026-08-21T10:00:00Z', '2026-08-21T11:00:00Z', '2026-08-21T11:00:00Z', '[]'
                );
            """)
            conn.commit()
            conn.close()

            panel = MorningResearchDossierPanel(root, data_dir=self.tmp_dir)
            panel.selected_campaign_id.set("CAMP_UI_001")
            panel.load_selected_campaign()

            # Verify tabs in notebook
            tab_names = [panel.notebook.tab(tab_id, "text") for tab_id in panel.notebook.tabs()]
            self.assertTrue(any("Research Registry" in t for t in tab_names), f"Research Registry tab should exist in {tab_names}")
        finally:
            try:
                root.destroy()
            except Exception:
                pass


if __name__ == "__main__":
    unittest.main()

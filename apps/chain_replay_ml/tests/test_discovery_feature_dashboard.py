"""Focused unit and integration verification for Discovery Feature Dashboard (Doc 18)."""

from datetime import datetime, timezone
import json
import os
import shutil
import sqlite3
import tempfile
import tkinter as tk
import unittest

from chain_replay_ml.discovery_dashboard.types import (
    CrossPipelineSelectionBasket,
    PipelineCreationRequest,
    PipelineCreationResult,
    SelectedDiscoveryFeatureRef,
)
from chain_replay_ml.discovery_dashboard.service import (
    create_candidate_discovery_pipeline,
    list_discovery_features,
    list_discovery_pipelines,
    validate_cross_pipeline_selection,
)
from chain_replay_ml.discovery_pipeline.persistence import (
    init_discovery_pipeline_tables,
    persist_discovered_features,
    persist_discovery_pipeline,
)
from chain_replay_ml.discovery_pipeline.types import (
    DiscoveredFeatureSpec,
    DiscoveryLifecycleStatus,
    DiscoveryPipelineSpec,
    GeneratorStrategy,
)
from chain_replay_ml.dataset_builder.pipeline_registry_store import (
    create_pipeline,
    get_pipeline,
    load_store as load_pl_store,
    save_store as save_pl_store,
)
from chain_replay_ml.research_memory.db import connect_analysis_db


class TestDiscoveryFeatureDashboard(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        init_discovery_pipeline_tables(self.tmp_dir)

        # Seed pipeline_registry_store.json with authoritative PL_0001 containing 171 features
        self.base_features = [f"base_feat_{i:03d}" for i in range(1, 172)]
        self.assertEqual(len(self.base_features), 171)

        pr_doc = {
            "registry_version": "1.0",
            "next_pipeline_id_seq": 2,
            "next_display_seq": 2,
            "pipelines": {
                "PL_0001": {
                    "pipeline_id": "PL_0001",
                    "name": "Pipeline_001 — Base",
                    "type": "base",
                    "status": "ready",
                    "context_key": "NIFTY:6:standard:all",
                    "candidate_features": list(self.base_features),
                    "registry_feature_ids": [],
                }
            },
            "history": [],
        }
        save_pl_store(self.tmp_dir, pr_doc)

        # Seed 2 Discovery Pipelines in analysis.db
        pipe1 = DiscoveryPipelineSpec(
            pipeline_id="DP_CAMP_001",
            campaign_id="CAMP_001",
            context_key="NIFTY:6:standard:all",
            dataset_name="dataset_1",
            dataset_snapshot_hash="snap_1",
            base_feature_count=171,
        )
        pipe2 = DiscoveryPipelineSpec(
            pipeline_id="DP_CAMP_002",
            campaign_id="CAMP_002",
            context_key="NIFTY:6:standard:all",
            dataset_name="dataset_2",
            dataset_snapshot_hash="snap_2",
            base_feature_count=171,
        )
        persist_discovery_pipeline(self.tmp_dir, pipe1)
        persist_discovery_pipeline(self.tmp_dir, pipe2)

        # Seed features for DP_CAMP_001 (2 KEEP, 1 WATCH, 1 REMOVE)
        feats_p1 = [
            DiscoveredFeatureSpec(
                feature_id="DF_CAMP_001_RATIO_001",
                pipeline_id="DP_CAMP_001",
                feature_name="DF_A_div_B",
                formula_expression="col(A)/col(B)",
                formula_hash="HASH_A_DIV_B",
                generator_strategy=GeneratorStrategy.RATIO,
                parent_features=["A", "B"],
                generation_discovered=1,
                lifecycle_status=DiscoveryLifecycleStatus.KEEP,
                evidence_score=60.0,
                ks_statistic=0.08,
                drift_severity=0,
                metadata={"delta_auc": 0.002, "fold_consistency": 0.8, "governance_rationale": "High marginal lift"},
            ),
            DiscoveredFeatureSpec(
                feature_id="DF_CAMP_001_INTERACTION_002",
                pipeline_id="DP_CAMP_001",
                feature_name="DF_C_mul_D",
                formula_expression="col(C)*col(D)",
                formula_hash="HASH_C_MUL_D",
                generator_strategy=GeneratorStrategy.INTERACTION,
                parent_features=["C", "D"],
                generation_discovered=1,
                lifecycle_status=DiscoveryLifecycleStatus.KEEP,
                evidence_score=55.0,
                ks_statistic=0.12,
                drift_severity=0,
                metadata={"delta_auc": 0.0015, "fold_consistency": 0.7, "governance_rationale": "Low drift"},
            ),
            DiscoveredFeatureSpec(
                feature_id="DF_CAMP_001_NONLINEAR_003",
                pipeline_id="DP_CAMP_001",
                feature_name="DF_LOG_E",
                formula_expression="log1p(abs(col(E)))",
                formula_hash="HASH_LOG_E",
                generator_strategy=GeneratorStrategy.NONLINEAR,
                parent_features=["E"],
                generation_discovered=2,
                lifecycle_status=DiscoveryLifecycleStatus.WATCH,
                evidence_score=48.0,
                ks_statistic=0.25,
                drift_severity=1,
                metadata={"delta_auc": 0.0005, "fold_consistency": 0.5, "governance_rationale": "Moderate drift"},
            ),
            DiscoveredFeatureSpec(
                feature_id="DF_CAMP_001_RATIO_004",
                pipeline_id="DP_CAMP_001",
                feature_name="DF_F_div_G",
                formula_expression="col(F)/col(G)",
                formula_hash="HASH_F_DIV_G",
                generator_strategy=GeneratorStrategy.RATIO,
                parent_features=["F", "G"],
                generation_discovered=2,
                lifecycle_status=DiscoveryLifecycleStatus.REMOVE,
                evidence_score=25.0,
                ks_statistic=0.45,
                drift_severity=2,
                metadata={"delta_auc": -0.002, "fold_consistency": 0.2, "governance_rationale": "Severe KS drift"},
            ),
        ]
        persist_discovered_features(self.tmp_dir, feats_p1)

        # Seed features for DP_CAMP_002 (1 KEEP with duplicate formula HASH_A_DIV_B, 1 unique KEEP)
        feats_p2 = [
            DiscoveredFeatureSpec(
                feature_id="DF_CAMP_002_RATIO_001",
                pipeline_id="DP_CAMP_002",
                feature_name="DF_A_div_B",
                formula_expression="col(A)/col(B)",
                formula_hash="HASH_A_DIV_B", # Duplicate formula across campaigns
                generator_strategy=GeneratorStrategy.RATIO,
                parent_features=["A", "B"],
                generation_discovered=1,
                lifecycle_status=DiscoveryLifecycleStatus.KEEP,
                evidence_score=62.5, # Higher evidence score
                ks_statistic=0.06,
                drift_severity=0,
                metadata={"delta_auc": 0.0025, "fold_consistency": 0.9, "governance_rationale": "Very strong lift"},
            ),
            DiscoveredFeatureSpec(
                feature_id="DF_CAMP_002_SPREAD_002",
                pipeline_id="DP_CAMP_002",
                feature_name="DF_H_sub_I",
                formula_expression="col(H)-col(I)",
                formula_hash="HASH_H_SUB_I",
                generator_strategy=GeneratorStrategy.SPREAD,
                parent_features=["H", "I"],
                generation_discovered=1,
                lifecycle_status=DiscoveryLifecycleStatus.KEEP,
                evidence_score=57.0,
                ks_statistic=0.09,
                drift_severity=0,
                metadata={"delta_auc": 0.0018, "fold_consistency": 0.8, "governance_rationale": "Spread signal"},
            ),
        ]
        persist_discovered_features(self.tmp_dir, feats_p2)

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_list_discovery_pipelines(self):
        """1. Verify Discovery Pipeline querying and aggregated statistics."""
        pipes = list_discovery_pipelines(self.tmp_dir, "NIFTY")
        self.assertEqual(len(pipes), 2)
        p1 = next(p for p in pipes if p["pipeline_id"] == "DP_CAMP_001")
        self.assertEqual(p1["keep_count"], 2)
        self.assertEqual(p1["watch_count"], 1)
        self.assertEqual(p1["remove_count"], 1)
        self.assertEqual(p1["active_discovery_pool"], 3)
        self.assertEqual(p1["total_df_features_created"], 4)

    def test_list_discovery_features_with_filters(self):
        """2. Verify feature listing with governance and strategy filters."""
        # Default: all KEEP & WATCH
        feats = list_discovery_features(self.tmp_dir, "DP_CAMP_001", verdicts=["KEEP", "WATCH"])
        self.assertEqual(len(feats), 3)

        # Only RATIO strategy
        ratio_feats = list_discovery_features(self.tmp_dir, "DP_CAMP_001", verdicts=["KEEP", "WATCH"], strategy="RATIO")
        self.assertEqual(len(ratio_feats), 1)
        self.assertEqual(ratio_feats[0]["feature_id"], "DF_CAMP_001_RATIO_001")

        # Search filter
        search_res = list_discovery_features(self.tmp_dir, "DP_CAMP_001", search_text="log1p")
        self.assertEqual(len(search_res), 1)
        self.assertEqual(search_res[0]["feature_id"], "DF_CAMP_001_NONLINEAR_003")

    def test_remove_feature_lockout(self):
        """3. Verify REMOVE features cannot enter the selection basket."""
        basket = CrossPipelineSelectionBasket()
        rem_ref = SelectedDiscoveryFeatureRef(
            feature_id="DF_CAMP_001_RATIO_004",
            pipeline_id="DP_CAMP_001",
            research_id="RESEARCH_001",
            campaign_id="CAMP_001",
            formula_hash="HASH_F_DIV_G",
            formula_expression="col(F)/col(G)",
            generator_strategy="RATIO",
            discovery_verdict="REMOVE",
        )
        added = basket.add(rem_ref)
        self.assertFalse(added)
        self.assertEqual(basket.total_count, 0)

    def test_cross_pipeline_selection_and_formula_deduplication(self):
        """4. Verify cross-pipeline selection and automatic deduplication of identical formula hashes."""
        basket = CrossPipelineSelectionBasket()

        # Add from DP_CAMP_001
        basket.add(SelectedDiscoveryFeatureRef(
            feature_id="DF_CAMP_001_RATIO_001",
            pipeline_id="DP_CAMP_001",
            research_id="RESEARCH_001",
            campaign_id="CAMP_001",
            formula_hash="HASH_A_DIV_B", # Score 60.0
            formula_expression="col(A)/col(B)",
            generator_strategy="RATIO",
            discovery_verdict="KEEP",
            evidence_score=60.0,
            marginal_delta_auc=0.002,
            context_key="NIFTY:6:standard:all",
        ))
        basket.add(SelectedDiscoveryFeatureRef(
            feature_id="DF_CAMP_001_INTERACTION_002",
            pipeline_id="DP_CAMP_001",
            research_id="RESEARCH_001",
            campaign_id="CAMP_001",
            formula_hash="HASH_C_MUL_D",
            formula_expression="col(C)*col(D)",
            generator_strategy="INTERACTION",
            discovery_verdict="KEEP",
            evidence_score=55.0,
            marginal_delta_auc=0.0015,
            context_key="NIFTY:6:standard:all",
        ))

        # Add from DP_CAMP_002 (including duplicate HASH_A_DIV_B with higher score 62.5)
        basket.add(SelectedDiscoveryFeatureRef(
            feature_id="DF_CAMP_002_RATIO_001",
            pipeline_id="DP_CAMP_002",
            research_id="RESEARCH_002",
            campaign_id="CAMP_002",
            formula_hash="HASH_A_DIV_B", # Score 62.5
            formula_expression="col(A)/col(B)",
            generator_strategy="RATIO",
            discovery_verdict="KEEP",
            evidence_score=62.5,
            marginal_delta_auc=0.0025,
            context_key="NIFTY:6:standard:all",
        ))
        basket.add(SelectedDiscoveryFeatureRef(
            feature_id="DF_CAMP_002_SPREAD_002",
            pipeline_id="DP_CAMP_002",
            research_id="RESEARCH_002",
            campaign_id="CAMP_002",
            formula_hash="HASH_H_SUB_I",
            formula_expression="col(H)-col(I)",
            generator_strategy="SPREAD",
            discovery_verdict="KEEP",
            evidence_score=57.0,
            marginal_delta_auc=0.0018,
            context_key="NIFTY:6:standard:all",
        ))

        self.assertEqual(basket.total_count, 4)
        self.assertEqual(basket.pipeline_count, 2)

        # Validate selection
        is_valid, msg, deduped, co_disc = validate_cross_pipeline_selection(basket, "NIFTY:6:standard:all")
        self.assertTrue(is_valid)
        # 4 items deduplicated to 3 unique formulas
        self.assertEqual(len(deduped), 3)
        self.assertEqual(len(co_disc), 1)

        # Ensure highest evidence score instance was chosen for HASH_A_DIV_B
        a_div_b_item = next(it for it in deduped if it.formula_hash == "HASH_A_DIV_B")
        self.assertEqual(a_div_b_item.feature_id, "DF_CAMP_002_RATIO_001")
        self.assertEqual(a_div_b_item.evidence_score, 62.5)

    def test_create_candidate_discovery_pipeline_end_to_end(self):
        """5. Verify discovery feature pipeline construction with strictly selected features and zero baseline injection."""
        basket = CrossPipelineSelectionBasket()
        basket.add(SelectedDiscoveryFeatureRef(
            feature_id="DF_CAMP_001_RATIO_001",
            display_name="A_to_B_ratio",
            pipeline_id="DP_CAMP_001",
            research_id="RESEARCH_001",
            campaign_id="CAMP_001",
            formula_hash="HASH_A_DIV_B",
            formula_expression="col(A)/col(B)",
            generator_strategy="RATIO",
            parent_features=["A", "B"],
            discovery_verdict="KEEP",
            evidence_score=60.0,
            marginal_delta_auc=0.002,
            context_key="NIFTY:6:standard:all",
        ))
        basket.add(SelectedDiscoveryFeatureRef(
            feature_id="DF_CAMP_002_SPREAD_002",
            display_name="H_minus_I",
            pipeline_id="DP_CAMP_002",
            research_id="RESEARCH_002",
            campaign_id="CAMP_002",
            formula_hash="HASH_H_SUB_I",
            formula_expression="col(H)-col(I)",
            generator_strategy="SPREAD",
            parent_features=["H", "I"],
            discovery_verdict="KEEP",
            evidence_score=57.0,
            marginal_delta_auc=0.0018,
            context_key="NIFTY:6:standard:all",
        ))

        req = PipelineCreationRequest(
            name="Pipeline_002 — Test Discovery Synthesis",
            description="Unit test discovery feature pipeline creation.",
            context_key="NIFTY:6:standard:all",
        )

        res = create_candidate_discovery_pipeline(self.tmp_dir, req, basket)
        self.assertTrue(res.success)
        self.assertEqual(res.pipeline_id, "PL_0002")
        self.assertEqual(res.base_feature_count, 0)
        self.assertEqual(res.discovered_feature_count, 2)
        self.assertEqual(res.total_feature_count, 2)

        # Inspect pipeline_registry_store.json
        pr_doc = load_pl_store(self.tmp_dir)
        pl_0001 = pr_doc["pipelines"]["PL_0001"]
        pl_0002 = pr_doc["pipelines"]["PL_0002"]

        # Invariant 1: PL_0001 remains strictly untouched (171 base features)
        self.assertEqual(len(pl_0001["candidate_features"]), 171)
        self.assertEqual(pl_0001["type"], "base")

        # Invariant 2: PL_0002 has type discovery_experimental and EXACTLY 2 features (zero baseline features injected)
        self.assertEqual(pl_0002["type"], "discovery_experimental")
        self.assertEqual(len(pl_0002["candidate_features"]), 2)
        self.assertEqual(pl_0002["candidate_features"], ["DF_CAMP_001_RATIO_001", "DF_CAMP_002_SPREAD_002"])

        # Invariant 3: Provenance metadata is fully recorded with human-readable display names
        prov = pl_0002.get("provenance_metadata", {})
        self.assertEqual(prov.get("creation_source"), "DISCOVERY_FEATURE_DASHBOARD")
        self.assertEqual(prov.get("creation_mode"), "CROSS_DISCOVERY_PIPELINE_SELECTION")
        self.assertIn("DP_CAMP_001", prov.get("source_discovery_pipelines", []))
        self.assertIn("DP_CAMP_002", prov.get("source_discovery_pipelines", []))
        self.assertEqual(len(prov.get("selected_features_provenance", [])), 2)
        p_items = prov.get("selected_features_provenance", [])
        self.assertEqual(p_items[0]["display_name"], "A_to_B_ratio")
        self.assertEqual(p_items[1]["display_name"], "H_minus_I")

    def test_dashboard_ui_initialization(self):
        """6. Verify DiscoveryFeatureDashboardPanel UI initializes and renders without error."""
        root = tk.Tk()
        root.withdraw()
        try:
            from apps.master_dataset_tk.discovery_feature_dashboard_panel import DiscoveryFeatureDashboardPanel
            panel = DiscoveryFeatureDashboardPanel(root, data_dir=self.tmp_dir)
            self.assertIsNotNone(panel)
            self.assertEqual(len(panel._all_pipelines), 2)
            self.assertGreater(len(panel._current_features), 0)
        finally:
            try:
                root.destroy()
            except Exception:
                pass

    def test_context_incompatibility_rejection(self):
        """7. Verify that selecting features across incompatible contexts is blocked."""
        basket = CrossPipelineSelectionBasket()
        basket.add(SelectedDiscoveryFeatureRef(
            feature_id="DF_CAMP_001_RATIO_001",
            pipeline_id="DP_CAMP_001",
            research_id="RESEARCH_001",
            campaign_id="CAMP_001",
            formula_hash="HASH_A_DIV_B",
            formula_expression="col(A)/col(B)",
            generator_strategy="RATIO",
            discovery_verdict="KEEP",
            context_key="BANKNIFTY:1s:Direction:1m:R001",
        ))

        is_valid, msg, _, _ = validate_cross_pipeline_selection(basket, target_context="NIFTY:6:standard:all")
        self.assertFalse(is_valid)
        self.assertIn("Context mismatch", msg)

    def test_model_registry_panel_navigation_button(self):
        """8. Verify ML Research Studio top navigation contains [🧬 Discovery Features] button."""
        root = tk.Tk()
        root.withdraw()
        try:
            from apps.master_dataset_tk.model_registry_panel import ModelRegistryPanel
            panel = ModelRegistryPanel(root, chart_dir=os.path.join(self.tmp_dir, "charts"))
            self.assertTrue(hasattr(panel, "_discovery_features_btn"))
            btn_text = panel._discovery_features_btn.cget("text")
            self.assertEqual(btn_text, "🧬 Discovery Features")
        finally:
            try:
                root.destroy()
            except Exception:
                pass


if __name__ == "__main__":
    unittest.main()

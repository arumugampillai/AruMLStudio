"""Unit tests for Phase 2: Isolated Discovery Pipeline Storage Layer in analysis.db."""

from __future__ import annotations

import os
import shutil
import tempfile
import unittest

from chain_replay_ml.discovery_pipeline.persistence import (
    get_discovery_pipeline_summary,
    init_discovery_pipeline_tables,
    load_discovered_feature_by_hash,
    load_discovered_features,
    load_discovery_pipeline,
    load_discovery_pipeline_by_campaign,
    load_discovery_snapshot,
    load_discovery_snapshots_for_pipeline,
    persist_discovered_features,
    persist_discovery_pipeline,
    persist_discovery_snapshot,
    update_discovered_feature_status,
)
from chain_replay_ml.discovery_pipeline.types import (
    DiscoveredFeatureSpec,
    DiscoveryLifecycleStatus,
    DiscoveryPipelineBudget,
    DiscoveryPipelineSnapshot,
    DiscoveryPipelineSpec,
    GeneratorStrategy,
    compute_discovery_snapshot_hash,
    compute_formula_hash,
)
from chain_replay_ml.research_memory.db import connect_analysis_db


class TestDiscoveryPipelinePersistence(unittest.TestCase):
    """Test suite for Discovery Pipeline persistence, isolation, snapshots, and deduplication."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_schema_initialization(self):
        """Verify tables and indexes are created in analysis.db."""
        init_discovery_pipeline_tables(self.test_dir)
        conn = connect_analysis_db(self.test_dir)
        try:
            tables = [
                r[0]
                for r in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'discovery_%'"
                ).fetchall()
            ]
            self.assertIn("discovery_pipelines", tables)
            self.assertIn("discovery_pipeline_features", tables)
            self.assertIn("discovery_pipeline_snapshots", tables)
        finally:
            conn.close()

    def test_pipeline_crud(self):
        """Verify DiscoveryPipelineSpec persistence and retrieval."""
        spec = DiscoveryPipelineSpec(
            pipeline_id="DP_CAMP_TEST_001",
            campaign_id="CAMP_TEST_001",
            context_key="NIFTY_6s_DIRECTION_CLASSIFIER_5m_R001",
            dataset_name="analysis_198r_171b_6s_20260820_223630",
            dataset_snapshot_hash="1714b8dddb455a95",
            base_feature_count=382,
            base_feature_names=["f1", "f2", "f3"],
            active_features_count=10,
            total_generated_count=25,
            current_snapshot_hash="DP_SNAP_0001",
            current_generation=1,
            status="active",
        )

        persist_discovery_pipeline(self.test_dir, spec)

        # Load by ID
        loaded = load_discovery_pipeline(self.test_dir, "DP_CAMP_TEST_001")
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded.pipeline_id, "DP_CAMP_TEST_001")
        self.assertEqual(loaded.campaign_id, "CAMP_TEST_001")
        self.assertEqual(loaded.base_feature_count, 382)
        self.assertEqual(loaded.base_feature_names, ["f1", "f2", "f3"])

        # Load by Campaign ID
        loaded_by_camp = load_discovery_pipeline_by_campaign(self.test_dir, "CAMP_TEST_001")
        self.assertIsNotNone(loaded_by_camp)
        self.assertEqual(loaded_by_camp.pipeline_id, "DP_CAMP_TEST_001")

    def test_feature_persistence_and_deduplication(self):
        """Verify discovered feature persistence and duplicate formula collision prevention."""
        pipe_id = "DP_CAMP_TEST_002"
        f1_expr = "col('reiv_skew') / (abs(col('iv_atm')) + 0.001)"
        f1_hash = compute_formula_hash(f1_expr)

        feat1 = DiscoveredFeatureSpec(
            feature_id="DF_CAMP_002_rati_0001",
            pipeline_id=pipe_id,
            feature_name="synth_ratio__reiv_div_iv",
            formula_expression=f1_expr,
            formula_hash=f1_hash,
            generator_strategy=GeneratorStrategy.RATIO,
            parent_features=["reiv_skew", "iv_atm"],
            generation_discovered=1,
            lifecycle_status=DiscoveryLifecycleStatus.CANDIDATE,
            evidence_score=0.0,
        )

        feat2 = DiscoveredFeatureSpec(
            feature_id="DF_CAMP_002_inte_0002",
            pipeline_id=pipe_id,
            feature_name="synth_inter__dgt_x_vol",
            formula_expression="zscore(col('dgt')) * zscore(col('volume'))",
            formula_hash=compute_formula_hash("zscore(col('dgt')) * zscore(col('volume'))"),
            generator_strategy=GeneratorStrategy.INTERACTION,
            parent_features=["dgt", "volume"],
            generation_discovered=1,
            lifecycle_status=DiscoveryLifecycleStatus.KEEP,
            evidence_score=88.0,
        )

        saved = persist_discovered_features(self.test_dir, [feat1, feat2])
        self.assertEqual(saved, 2)

        # Load by hash
        loaded_f1 = load_discovered_feature_by_hash(self.test_dir, pipe_id, f1_hash)
        self.assertIsNotNone(loaded_f1)
        self.assertEqual(loaded_f1.feature_name, "synth_ratio__reiv_div_iv")

        # Duplicate insertion attempt with updated telemetry should UPDATE, not duplicate
        feat1_updated = DiscoveredFeatureSpec(
            feature_id="DF_CAMP_002_rati_0001",
            pipeline_id=pipe_id,
            feature_name="synth_ratio__reiv_div_iv",
            formula_expression=f1_expr,
            formula_hash=f1_hash,
            generator_strategy=GeneratorStrategy.RATIO,
            parent_features=["reiv_skew", "iv_atm"],
            generation_discovered=1,
            lifecycle_status=DiscoveryLifecycleStatus.KEEP,
            evidence_score=92.5,
            total_evaluations=3,
        )
        persist_discovered_features(self.test_dir, [feat1_updated])

        all_feats = load_discovered_features(self.test_dir, pipe_id)
        self.assertEqual(len(all_feats), 2, "Duplicate formula should not create new row")
        
        # Verify updated score
        re_loaded_f1 = load_discovered_feature_by_hash(self.test_dir, pipe_id, f1_hash)
        self.assertEqual(re_loaded_f1.lifecycle_status, DiscoveryLifecycleStatus.KEEP)
        self.assertEqual(re_loaded_f1.evidence_score, 92.5)

    def test_status_transition_and_filtering(self):
        """Verify lifecycle status transitions and status-based query filtering."""
        pipe_id = "DP_CAMP_TEST_003"
        feat1 = DiscoveredFeatureSpec(
            feature_id="DF_1",
            pipeline_id=pipe_id,
            feature_name="f_keep",
            formula_expression="f1 + f2",
            formula_hash=compute_formula_hash("f1 + f2"),
            generator_strategy=GeneratorStrategy.COMPOSITE,
            parent_features=["f1", "f2"],
            generation_discovered=1,
            lifecycle_status=DiscoveryLifecycleStatus.KEEP,
            evidence_score=95.0,
        )
        feat2 = DiscoveredFeatureSpec(
            feature_id="DF_2",
            pipeline_id=pipe_id,
            feature_name="f_watch",
            formula_expression="f1 - f2",
            formula_hash=compute_formula_hash("f1 - f2"),
            generator_strategy=GeneratorStrategy.SPREAD,
            parent_features=["f1", "f2"],
            generation_discovered=1,
            lifecycle_status=DiscoveryLifecycleStatus.WATCH,
            evidence_score=60.0,
        )
        feat3 = DiscoveredFeatureSpec(
            feature_id="DF_3",
            pipeline_id=pipe_id,
            feature_name="f_remove",
            formula_expression="f1 * f2",
            formula_hash=compute_formula_hash("f1 * f2"),
            generator_strategy=GeneratorStrategy.INTERACTION,
            parent_features=["f1", "f2"],
            generation_discovered=1,
            lifecycle_status=DiscoveryLifecycleStatus.REMOVE,
            evidence_score=10.0,
        )

        persist_discovered_features(self.test_dir, [feat1, feat2, feat3])

        # Test filtering
        keeps = load_discovered_features(self.test_dir, pipe_id, status=DiscoveryLifecycleStatus.KEEP)
        self.assertEqual(len(keeps), 1)
        self.assertEqual(keeps[0].feature_name, "f_keep")

        watches = load_discovered_features(self.test_dir, pipe_id, status="WATCH")
        self.assertEqual(len(watches), 1)
        self.assertEqual(watches[0].feature_name, "f_watch")

        # Test status update
        updated = update_discovered_feature_status(
            self.test_dir,
            pipe_id,
            "f_watch",
            DiscoveryLifecycleStatus.KEEP,
            evidence_score=85.0,
        )
        self.assertTrue(updated)

        keeps_after = load_discovered_features(self.test_dir, pipe_id, status=DiscoveryLifecycleStatus.KEEP)
        self.assertEqual(len(keeps_after), 2)

    def test_snapshot_persistence_and_summary(self):
        """Verify generation snapshot persistence and pipeline summary calculation."""
        pipe_id = "DP_CAMP_TEST_004"
        pipe = DiscoveryPipelineSpec(
            pipeline_id=pipe_id,
            campaign_id="CAMP_004",
            context_key="NIFTY_6s_DIRECTION_CLASSIFIER_5m_R001",
            dataset_name="analysis_198r_171b_6s_20260820_223630",
            dataset_snapshot_hash="1714b8dddb455a95",
            base_feature_count=382,
            current_generation=2,
            current_snapshot_hash="DP_SNAP_GEN2",
        )
        persist_discovery_pipeline(self.test_dir, pipe)

        snap1 = DiscoveryPipelineSnapshot(
            snapshot_hash=compute_discovery_snapshot_hash(pipe_id, 1, ["f_keep1"]),
            pipeline_id=pipe_id,
            generation_number=1,
            active_feature_names=["f_keep1"],
            feature_count=1,
            keep_count=1,
        )
        snap2 = DiscoveryPipelineSnapshot(
            snapshot_hash=compute_discovery_snapshot_hash(pipe_id, 2, ["f_keep1", "f_keep2"]),
            pipeline_id=pipe_id,
            generation_number=2,
            active_feature_names=["f_keep1", "f_keep2"],
            feature_count=2,
            keep_count=2,
        )

        persist_discovery_snapshot(self.test_dir, snap1)
        persist_discovery_snapshot(self.test_dir, snap2)

        # Load snapshots
        snaps = load_discovery_snapshots_for_pipeline(self.test_dir, pipe_id)
        self.assertEqual(len(snaps), 2)
        self.assertEqual(snaps[0].generation_number, 1)
        self.assertEqual(snaps[1].generation_number, 2)

        # Summary check
        feat1 = DiscoveredFeatureSpec(
            feature_id="DF_K1",
            pipeline_id=pipe_id,
            feature_name="f_keep1",
            formula_expression="f1",
            formula_hash=compute_formula_hash("f1"),
            generator_strategy=GeneratorStrategy.NONLINEAR,
            parent_features=["f1"],
            generation_discovered=1,
            lifecycle_status=DiscoveryLifecycleStatus.KEEP,
            evidence_score=95.0,
        )
        persist_discovered_features(self.test_dir, [feat1])

        summary = get_discovery_pipeline_summary(self.test_dir, pipe_id)
        self.assertEqual(summary["pipeline_id"], pipe_id)
        self.assertEqual(summary["counts"]["total_generated"], 1)
        self.assertEqual(summary["counts"]["KEEP"], 1)
        self.assertEqual(len(summary["top_keep_features"]), 1)


if __name__ == "__main__":
    unittest.main()

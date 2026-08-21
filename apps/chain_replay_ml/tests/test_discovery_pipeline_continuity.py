"""Unit tests for Phase 9: Next-Day Multi-Session Continuity & Warm-Start Engine."""

from __future__ import annotations

import shutil
import tempfile
import unittest
import numpy as np
import pandas as pd

from chain_replay_ml.discovery_pipeline.continuity import (
    list_available_discovery_snapshots,
    load_discovery_snapshot_bundle,
    warm_start_discovery_pipeline,
)
from chain_replay_ml.discovery_pipeline.persistence import (
    init_discovery_pipeline_tables,
    load_discovered_features,
    load_discovery_pipeline,
    load_discovery_snapshots_for_pipeline,
    persist_discovered_features,
    persist_discovery_pipeline,
    persist_discovery_snapshot,
)
from chain_replay_ml.discovery_pipeline.types import (
    DiscoveredFeatureSpec,
    DiscoveryLifecycleStatus,
    DiscoveryPipelineBudget,
    DiscoveryPipelineSnapshot,
    DiscoveryPipelineSpec,
    GeneratorStrategy,
)


class TestDiscoveryPipelineContinuity(unittest.TestCase):
    """Test suite for Discovery Pipeline warm-start, snapshot loading, and cross-session isolation."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        init_discovery_pipeline_tables(self.test_dir)

        self.camp_day1 = "CAMP_DAY1_001"
        self.pipe_day1 = f"DP_{self.camp_day1}"
        self.camp_day2 = "CAMP_DAY2_002"
        self.pipe_day2 = f"DP_{self.camp_day2}"

        # Create Day 1 pipeline and features
        self.base_features = ["f1", "f2", "f3"]
        pipe1 = DiscoveryPipelineSpec(
            pipeline_id=self.pipe_day1,
            campaign_id=self.camp_day1,
            context_key="NIFTY_6s_DIRECTION_CLASSIFIER_5m_R001",
            dataset_name="day1_data",
            dataset_snapshot_hash="day1_hash",
            base_feature_count=3,
            base_feature_names=self.base_features,
        )
        persist_discovery_pipeline(self.test_dir, pipe1)

        f1 = DiscoveredFeatureSpec(
            feature_id="DF_D1_001",
            pipeline_id=self.pipe_day1,
            feature_name="synth_ratio_f1_f2",
            formula_expression="col('f1') / (abs(col('f2')) + 0.001)",
            formula_hash="h_ratio_12",
            generator_strategy=GeneratorStrategy.RATIO,
            parent_features=["f1", "f2"],
            generation_discovered=1,
            lifecycle_status=DiscoveryLifecycleStatus.KEEP,
            evidence_score=59.0,
            ks_statistic=0.04,
            metadata={"delta_auc": 0.012, "baseline_auc": 0.52, "fold_consistency": 0.80},
        )
        f2 = DiscoveredFeatureSpec(
            feature_id="DF_D1_002",
            pipeline_id=self.pipe_day1,
            feature_name="synth_log1p_f3",
            formula_expression="log1p(abs(col('f3')))",
            formula_hash="h_log1p_3",
            generator_strategy=GeneratorStrategy.NONLINEAR,
            parent_features=["f3"],
            generation_discovered=1,
            lifecycle_status=DiscoveryLifecycleStatus.KEEP,
            evidence_score=55.0,
            ks_statistic=0.05,
            metadata={"delta_auc": 0.006, "baseline_auc": 0.52, "fold_consistency": 0.60},
        )
        persist_discovered_features(self.test_dir, [f1, f2])

        self.snap_hash_day1 = "DP_SNAP_DAY1_FINAL_HASH"
        snap1 = DiscoveryPipelineSnapshot(
            snapshot_hash=self.snap_hash_day1,
            pipeline_id=self.pipe_day1,
            generation_number=1,
            active_feature_names=["synth_ratio_f1_f2", "synth_log1p_f3"],
            feature_count=2,
            keep_count=2,
            watch_count=0,
            remove_count=0,
        )
        persist_discovery_snapshot(self.test_dir, snap1)

        # Simulated Day 2 dataset
        np.random.seed(99)
        n = 600
        sig = np.random.normal(0, 1, n)
        self.day2_df = pd.DataFrame({
            "f1": sig + np.random.normal(0, 0.2, n),
            "f2": np.random.normal(0, 1, n),
            "f3": np.random.exponential(1.0, n) + 0.1,
            "label_up_5pct_5m": np.random.choice([0, 1], n, p=[0.5, 0.5]),
        })

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_list_and_load_snapshot_bundle(self):
        """Verify snapshot listing and bundle loading."""
        snaps = list_available_discovery_snapshots(self.test_dir)
        self.assertEqual(len(snaps), 1)
        self.assertEqual(snaps[0]["snapshot_hash"], self.snap_hash_day1)

        bundle = load_discovery_snapshot_bundle(self.test_dir, self.snap_hash_day1)
        self.assertIsNotNone(bundle)
        self.assertEqual(len(bundle["active_features"]), 2)
        self.assertEqual(bundle["pipeline"].pipeline_id, self.pipe_day1)

    def test_warm_start_pipeline_initialization(self):
        """Verify Day 2 warm start from Day 1 snapshot, revalidation, and isolation."""
        res = warm_start_discovery_pipeline(
            self.day2_df,
            data_dir=self.test_dir,
            source_snapshot_hash=self.snap_hash_day1,
            new_campaign_id=self.camp_day2,
            base_features=self.base_features,
            dataset_name="day2_market_data",
            dataset_snapshot_hash="day2_hash",
            revalidate_features=True,
        )

        self.assertEqual(res["pipeline_id"], self.pipe_day2)
        self.assertEqual(res["source_snapshot_hash"], self.snap_hash_day1)
        self.assertEqual(res["imported_features_count"], 2)
        self.assertTrue(res["initial_snapshot_hash"].startswith("DP_SNAP_"))

        # Verify Day 2 pipeline spec persisted
        pipe2 = load_discovery_pipeline(self.test_dir, self.pipe_day2)
        self.assertIsNotNone(pipe2)
        self.assertEqual(pipe2.parent_snapshot_hash, self.snap_hash_day1)

        # Verify Day 2 cloned features persisted
        feats2 = load_discovered_features(self.test_dir, self.pipe_day2)
        self.assertEqual(len(feats2), 2)
        for f in feats2:
            self.assertEqual(f.pipeline_id, self.pipe_day2)
            self.assertIn("warm_started_from_snapshot", f.metadata)

        # CRITICAL ISOLATION PROOF: Day 1 records remain 100% intact
        feats1 = load_discovered_features(self.test_dir, self.pipe_day1)
        self.assertEqual(len(feats1), 2)
        self.assertEqual(feats1[0].pipeline_id, self.pipe_day1)


if __name__ == "__main__":
    unittest.main()

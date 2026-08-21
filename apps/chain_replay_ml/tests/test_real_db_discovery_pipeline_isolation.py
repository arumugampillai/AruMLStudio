"""Real Database Verification Test for Phase 2: Discovery Pipeline Storage Isolation.

Verifies against the live workspace environment:
1. Real analysis.db table initialization.
2. Campaign isolation (DP_CAMP_REAL_A vs DP_CAMP_REAL_B).
3. Snapshot generation and cryptographic hash reproducibility.
4. Duplicate formula prevention on live database.
5. Invariant: ZERO modification to feature_registry_store.json.
6. Invariant: ZERO modification to pipeline_registry_store.json.
7. Clean teardown of temporary test records from live analysis.db.
"""

from __future__ import annotations

import hashlib
import json
import os
import unittest

from chain_replay_ml.discovery_pipeline.persistence import (
    get_discovery_pipeline_summary,
    init_discovery_pipeline_tables,
    load_discovered_feature_by_hash,
    load_discovered_features,
    load_discovery_pipeline,
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


def _sha256_file(path: str) -> str | None:
    if not os.path.isfile(path):
        return None
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


class TestRealDatabaseDiscoveryIsolation(unittest.TestCase):
    """Rigorous verification on the actual workspace real data directory."""

    @classmethod
    def setUpClass(cls):
        cls.data_dir = "data"
        cls.feat_store_path = os.path.join(cls.data_dir, "feature_registry_store.json")
        cls.pipe_store_path = os.path.join(cls.data_dir, "pipeline_registry_store.json")

        cls.feat_store_hash_before = _sha256_file(cls.feat_store_path)
        cls.pipe_store_hash_before = _sha256_file(cls.pipe_store_path)

        cls.camp_a_id = "CAMP_REAL_TEST_A_9991"
        cls.camp_b_id = "CAMP_REAL_TEST_B_9992"
        cls.pipe_a_id = f"DP_{cls.camp_a_id}"
        cls.pipe_b_id = f"DP_{cls.camp_b_id}"

    @classmethod
    def tearDownClass(cls):
        # Clean up test rows from real analysis.db
        conn = connect_analysis_db(cls.data_dir)
        try:
            with conn:
                conn.execute("DELETE FROM discovery_pipeline_snapshots WHERE pipeline_id IN (?, ?)", (cls.pipe_a_id, cls.pipe_b_id))
                conn.execute("DELETE FROM discovery_pipeline_features WHERE pipeline_id IN (?, ?)", (cls.pipe_a_id, cls.pipe_b_id))
                conn.execute("DELETE FROM discovery_pipelines WHERE pipeline_id IN (?, ?)", (cls.pipe_a_id, cls.pipe_b_id))
        finally:
            conn.close()

    def test_real_db_isolation_and_integrity(self):
        """Execute full isolated lifecycle on live analysis.db and verify zero registry mutation."""
        # 1. Initialize schema
        init_discovery_pipeline_tables(self.data_dir)

        # 2. Persist two distinct campaign discovery pipelines
        pipe_a = DiscoveryPipelineSpec(
            pipeline_id=self.pipe_a_id,
            campaign_id=self.camp_a_id,
            context_key="NIFTY_6s_DIRECTION_CLASSIFIER_5m_R001",
            dataset_name="analysis_198r_171b_6s_20260820_223630",
            dataset_snapshot_hash="1714b8dddb455a95",
            base_feature_count=382,
            current_generation=1,
        )
        pipe_b = DiscoveryPipelineSpec(
            pipeline_id=self.pipe_b_id,
            campaign_id=self.camp_b_id,
            context_key="NIFTY_6s_DIRECTION_CLASSIFIER_5m_R001",
            dataset_name="analysis_198r_171b_6s_20260820_223630",
            dataset_snapshot_hash="1714b8dddb455a95",
            base_feature_count=382,
            current_generation=1,
        )
        persist_discovery_pipeline(self.data_dir, pipe_a)
        persist_discovery_pipeline(self.data_dir, pipe_b)

        # 3. Add distinct features to Campaign A and Campaign B
        f_expr = "col('reiv_skew') / (abs(col('iv_atm')) + 0.001)"
        feat_a = DiscoveredFeatureSpec(
            feature_id=f"DF_{self.camp_a_id}_01",
            pipeline_id=self.pipe_a_id,
            feature_name="synth_ratio_reiv_iv_a",
            formula_expression=f_expr,
            formula_hash=compute_formula_hash(f_expr),
            generator_strategy=GeneratorStrategy.RATIO,
            parent_features=["reiv_skew", "iv_atm"],
            generation_discovered=1,
            lifecycle_status=DiscoveryLifecycleStatus.KEEP,
            evidence_score=91.0,
        )
        feat_b = DiscoveredFeatureSpec(
            feature_id=f"DF_{self.camp_b_id}_01",
            pipeline_id=self.pipe_b_id,
            feature_name="synth_inter_dgt_vol_b",
            formula_expression="zscore(col('dgt')) * zscore(col('vol'))",
            formula_hash=compute_formula_hash("zscore(col('dgt')) * zscore(col('vol'))"),
            generator_strategy=GeneratorStrategy.INTERACTION,
            parent_features=["dgt", "vol"],
            generation_discovered=1,
            lifecycle_status=DiscoveryLifecycleStatus.WATCH,
            evidence_score=62.0,
        )

        persist_discovered_features(self.data_dir, [feat_a])
        persist_discovered_features(self.data_dir, [feat_b])

        # 4. Verify isolation
        feats_a = load_discovered_features(self.data_dir, self.pipe_a_id)
        feats_b = load_discovered_features(self.data_dir, self.pipe_b_id)
        self.assertEqual(len(feats_a), 1)
        self.assertEqual(len(feats_b), 1)
        self.assertEqual(feats_a[0].feature_name, "synth_ratio_reiv_iv_a")
        self.assertEqual(feats_b[0].feature_name, "synth_inter_dgt_vol_b")

        # 5. Verify snapshot persistence and hash reproducibility
        snap_a = DiscoveryPipelineSnapshot(
            snapshot_hash=compute_discovery_snapshot_hash(self.pipe_a_id, 1, ["synth_ratio_reiv_iv_a"]),
            pipeline_id=self.pipe_a_id,
            generation_number=1,
            active_feature_names=["synth_ratio_reiv_iv_a"],
            feature_count=1,
            keep_count=1,
        )
        persist_discovery_snapshot(self.data_dir, snap_a)

        loaded_snap = load_discovery_snapshot(self.data_dir, snap_a.snapshot_hash)
        self.assertIsNotNone(loaded_snap)
        self.assertEqual(loaded_snap.feature_count, 1)

        # 6. Verify duplicate formula collision handling on real DB
        # Re-inserting feat_a with updated score should update, not duplicate
        feat_a_up = DiscoveredFeatureSpec(
            feature_id=f"DF_{self.camp_a_id}_01",
            pipeline_id=self.pipe_a_id,
            feature_name="synth_ratio_reiv_iv_a",
            formula_expression=f_expr,
            formula_hash=compute_formula_hash(f_expr),
            generator_strategy=GeneratorStrategy.RATIO,
            parent_features=["reiv_skew", "iv_atm"],
            generation_discovered=1,
            lifecycle_status=DiscoveryLifecycleStatus.KEEP,
            evidence_score=96.5,
            total_evaluations=2,
        )
        persist_discovered_features(self.data_dir, [feat_a_up])
        feats_a_reloaded = load_discovered_features(self.data_dir, self.pipe_a_id)
        self.assertEqual(len(feats_a_reloaded), 1, "Duplicate formula must not insert new row")
        self.assertEqual(feats_a_reloaded[0].evidence_score, 96.5)

        # 7. Verify Invariant: ZERO modification to Feature Registry store
        feat_store_hash_after = _sha256_file(self.feat_store_path)
        self.assertEqual(
            self.feat_store_hash_before,
            feat_store_hash_after,
            "CRITICAL INVARIANT VIOLATION: feature_registry_store.json was modified during discovery persistence!",
        )

        # 8. Verify Invariant: ZERO modification to Pipeline Registry store
        pipe_store_hash_after = _sha256_file(self.pipe_store_path)
        self.assertEqual(
            self.pipe_store_hash_before,
            pipe_store_hash_after,
            "CRITICAL INVARIANT VIOLATION: pipeline_registry_store.json was modified during discovery persistence!",
        )


if __name__ == "__main__":
    unittest.main()

"""Authoritative test suite verifying the Base Pipeline Architecture Fix across all subsystems."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import unittest
import numpy as np
import pandas as pd

from chain_replay_ml.dataset_builder.pipeline_registry_store import (
    ensure_default_existing_pipeline,
    get_base_pipeline_for_context,
    load_store as load_pipeline_store,
)
from chain_replay_ml.discovery_pipeline.governance import (
    evaluate_discovery_governance_decision,
    run_discovery_pipeline_governance,
)
from chain_replay_ml.discovery_pipeline.loop import (
    run_autonomous_discovery_loop,
    run_discovery_generation,
)
from chain_replay_ml.discovery_pipeline.persistence import (
    init_discovery_pipeline_tables,
    load_discovered_features,
    load_discovery_pipeline,
    persist_discovered_features,
    persist_discovery_pipeline,
)
from chain_replay_ml.discovery_pipeline.types import (
    DiscoveredFeatureSpec,
    DiscoveryLifecycleStatus,
    DiscoveryPipelineBudget,
    DiscoveryPipelineSpec,
    GeneratorStrategy,
)


class TestBasePipelineArchitectureFix(unittest.TestCase):
    """Test suite proving Base Pipeline immutability, context binding, Gen 0 resolution, and governance protection."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        init_discovery_pipeline_tables(self.test_dir)

        # Context key under test
        self.context_key = "NIFTY_6s_DIRECTION_CLASSIFIER_5m_R001"

        # Seed PL_0001 in test_dir
        self.doc = ensure_default_existing_pipeline(self.test_dir, context_key=self.context_key)
        self.base_pipe = get_base_pipeline_for_context(self.test_dir, self.context_key)

        # Synthetic test DataFrame with base features
        np.random.seed(42)
        n = 500
        self.base_features = ["reiv_skew", "iv_atm", "iv_call_otm", "volume_flow", "delta_oi", "spot_ema_ratio"]
        self.df = pd.DataFrame({
            "reiv_skew": np.random.normal(0.05, 0.12, n),
            "iv_atm": np.random.uniform(0.12, 0.28, n),
            "iv_call_otm": np.random.uniform(0.14, 0.32, n),
            "volume_flow": np.random.exponential(50000.0, n),
            "delta_oi": np.random.normal(1200.0, 450.0, n),
            "spot_ema_ratio": np.random.normal(1.002, 0.005, n),
            "label_up_5pct_5m": np.random.choice([0, 1], n, p=[0.5, 0.5]),
            "timestamp": pd.date_range("2026-08-20 09:15:00", periods=n, freq="6s"),
            "token": [26000] * n,
            "symbol": ["NIFTY"] * n,
        })

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_base_pipeline_seeding_and_context_indexing(self):
        """Proof 1: PL_0001 is seeded on disk, marked type='base', and indexed by context key."""
        self.assertIsNotNone(self.base_pipe)
        self.assertEqual(self.base_pipe["pipeline_id"], "PL_0001")
        self.assertEqual(self.base_pipe["type"], "base")
        self.assertEqual(self.base_pipe["status"], "ready")
        self.assertEqual(self.base_pipe["context_key"], self.context_key)
        self.assertGreaterEqual(len(self.base_pipe["candidate_features"]), 382)
        self.assertEqual(self.base_pipe["pipeline_snapshot_id"], "1714b8dddb455a95")

    def test_discovery_gen0_base_pipeline_resolution(self):
        """Proof 2: Discovery Pipeline Gen 0 automatically resolves and binds PL_0001 as its base anchor."""
        camp_id = "CAMP_TEST_BASE_GEN0"
        pipe_id = f"DP_{camp_id}"

        # Run Generation 1
        res = run_discovery_generation(
            self.df,
            data_dir=self.test_dir,
            pipeline_id=pipe_id,
            campaign_id=camp_id,
            generation_number=1,
            base_features=self.base_features,
            target_column="label_up_5pct_5m",
            context_key=self.context_key,
            budget=DiscoveryPipelineBudget(max_new_features_per_gen=3),
        )
        self.assertEqual(res["generation_number"], 1)

        # Verify pipeline spec in analysis.db recorded base_pipeline_id = PL_0001
        pipe_spec = load_discovery_pipeline(self.test_dir, pipe_id)
        self.assertIsNotNone(pipe_spec)
        self.assertEqual(pipe_spec.base_pipeline_id, "PL_0001")
        self.assertEqual(pipe_spec.base_pipeline_snapshot_hash, "1714b8dddb455a95")

    def test_base_pipeline_immutability_under_governance(self):
        """Proof 3: Discovery governance REMOVE verdicts can NEVER mutate or delete Base Pipeline features."""
        # Capture base pipeline candidate features before
        pipe_before = get_base_pipeline_for_context(self.test_dir, self.context_key)
        base_features_before = list(pipe_before["candidate_features"])

        camp_id = "CAMP_TEST_GOV_IMMUTABLE"
        pipe_id = f"DP_{camp_id}"

        # Create a degrading experimental feature
        degraded_feat = DiscoveredFeatureSpec(
            feature_id="DF_TEST_DEGRADED_001",
            pipeline_id=pipe_id,
            feature_name="synth_degraded_ratio",
            formula_expression="col('reiv_skew') / col('iv_atm')",
            formula_hash="h_deg_001",
            generator_strategy=GeneratorStrategy.RATIO,
            parent_features=["reiv_skew", "iv_atm"],
            generation_discovered=1,
            lifecycle_status=DiscoveryLifecycleStatus.CANDIDATE,
            evidence_score=20.0,
            ks_statistic=0.45,  # High drift -> triggers REMOVE
            drift_severity=2,
            metadata={"delta_auc": -0.010},
        )
        persist_discovered_features(self.test_dir, [degraded_feat])

        # Run governance
        gov_res = run_discovery_pipeline_governance(
            self.test_dir,
            pipeline_id=pipe_id,
            campaign_id=camp_id,
            context_key=self.context_key,
        )
        self.assertEqual(gov_res["remove_count"], 1)

        # Verify degraded feature received REMOVE status in discovery DB
        feats = load_discovered_features(self.test_dir, pipe_id)
        self.assertEqual(feats[0].lifecycle_status, DiscoveryLifecycleStatus.REMOVE)

        # CRITICAL PROOF: Base Pipeline candidate_features in pipeline_registry_store.json is 100% UNCHANGED
        pipe_after = get_base_pipeline_for_context(self.test_dir, self.context_key)
        self.assertEqual(base_features_before, pipe_after["candidate_features"])
        self.assertEqual(pipe_after["type"], "base")


if __name__ == "__main__":
    unittest.main()

"""Unit tests for Phase 1: Autonomous Research Discovery Pipeline Types, Invariants & Schemas."""

from __future__ import annotations

import json
import unittest

from chain_replay_ml.discovery_pipeline.types import (
    DiscoveredFeatureSpec,
    DiscoveryLifecycleStatus,
    DiscoveryPipelineBudget,
    DiscoveryPipelineSnapshot,
    DiscoveryPipelineSpec,
    GeneratorStrategy,
    compute_discovery_snapshot_hash,
    compute_formula_hash,
    format_discovered_feature_id,
    format_discovery_pipeline_id,
    normalize_formula_expression,
)


class TestDiscoveryPipelineTypes(unittest.TestCase):
    """Test suite for Discovery Pipeline data structures and identity primitives."""

    def test_discovery_pipeline_id_formatting(self):
        """Verify campaign-scoped ID formatting."""
        self.assertEqual(format_discovery_pipeline_id("CAMP_20260821_180000"), "DP_CAMP_20260821_180000")
        self.assertEqual(format_discovery_pipeline_id("DP_CAMP_20260821_180000"), "DP_CAMP_20260821_180000")
        self.assertEqual(format_discovery_pipeline_id("20260821_180000"), "DP_CAMP_20260821_180000")

    def test_discovered_feature_id_formatting(self):
        """Verify unique deterministic feature ID formatting."""
        fid = format_discovered_feature_id("DP_CAMP_1a2b", "RATIO", 1)
        self.assertEqual(fid, "DF_CAMP_1a2b_rati_0001")

        fid2 = format_discovered_feature_id("DP_CAMP_1a2b", "INTERACTION", 42)
        self.assertEqual(fid2, "DF_CAMP_1a2b_inte_0042")

    def test_formula_normalization_and_hashing(self):
        """Verify formula normalization and deterministic MD5 hashing."""
        f1 = "col('reiv_skew')   /   (abs(col('iv_atm')) + 0.001)"
        f2 = "col('reiv_skew') / (abs(col('iv_atm')) + 0.001)"
        
        self.assertEqual(normalize_formula_expression(f1), normalize_formula_expression(f2))
        h1 = compute_formula_hash(f1)
        h2 = compute_formula_hash(f2)
        self.assertEqual(h1, h2)
        self.assertEqual(len(h1), 16)

    def test_discovery_snapshot_hash(self):
        """Verify snapshot hashing is invariant to feature order."""
        snap1 = compute_discovery_snapshot_hash("DP_CAMP_01", 1, ["feat_b", "feat_a", "feat_c"])
        snap2 = compute_discovery_snapshot_hash("DP_CAMP_01", 1, ["feat_a", "feat_c", "feat_b"])
        
        self.assertEqual(snap1, snap2)
        self.assertTrue(snap1.startswith("DP_SNAP_"))
        self.assertEqual(len(snap1), 8 + 16)  # "DP_SNAP_" + 16 hex chars

    def test_discovered_feature_spec_roundtrip(self):
        """Verify DiscoveredFeatureSpec serialization and deserialization."""
        spec = DiscoveredFeatureSpec(
            feature_id="DF_CAMP_01_rati_0001",
            pipeline_id="DP_CAMP_01",
            feature_name="synth_ratio__reiv_div_iv",
            formula_expression="col('reiv_skew') / (abs(col('iv_atm')) + 0.001)",
            formula_hash=compute_formula_hash("col('reiv_skew') / (abs(col('iv_atm')) + 0.001)"),
            generator_strategy=GeneratorStrategy.RATIO,
            parent_features=["reiv_skew", "iv_atm"],
            generation_discovered=1,
            lifecycle_status=DiscoveryLifecycleStatus.KEEP,
            evidence_score=85.5,
            total_evaluations=5,
            holdout_rank=3,
            relative_imp_drop=0.08,
            drift_severity=1,
            ks_statistic=0.12,
            ks_pvalue=0.65,
            metadata={"source": "test"},
        )

        d = spec.to_dict()
        self.assertEqual(d["generator_strategy"], "RATIO")
        self.assertEqual(d["lifecycle_status"], "KEEP")
        self.assertEqual(d["evidence_score"], 85.5)

        spec_restored = DiscoveredFeatureSpec.from_dict(d)
        self.assertEqual(spec_restored.feature_id, spec.feature_id)
        self.assertEqual(spec_restored.generator_strategy, GeneratorStrategy.RATIO)
        self.assertEqual(spec_restored.lifecycle_status, DiscoveryLifecycleStatus.KEEP)
        self.assertEqual(spec_restored.parent_features, ["reiv_skew", "iv_atm"])
        self.assertEqual(spec_restored.ks_statistic, 0.12)

    def test_discovery_pipeline_snapshot_roundtrip(self):
        """Verify DiscoveryPipelineSnapshot serialization and deserialization."""
        snap = DiscoveryPipelineSnapshot(
            snapshot_hash="DP_SNAP_1234567890abcdef",
            pipeline_id="DP_CAMP_01",
            generation_number=2,
            active_feature_names=["f1", "f2", "f3"],
            feature_count=3,
            keep_count=2,
            watch_count=1,
            remove_count=0,
        )

        d = snap.to_dict()
        self.assertEqual(d["snapshot_hash"], "DP_SNAP_1234567890abcdef")
        self.assertEqual(d["feature_count"], 3)

        snap_restored = DiscoveryPipelineSnapshot.from_dict(d)
        self.assertEqual(snap_restored.snapshot_hash, snap.snapshot_hash)
        self.assertEqual(snap_restored.active_feature_names, ["f1", "f2", "f3"])

    def test_discovery_pipeline_spec_roundtrip(self):
        """Verify DiscoveryPipelineSpec serialization and deserialization."""
        budget = DiscoveryPipelineBudget(
            max_new_features_per_gen=25,
            max_active_discovery_features=150,
            max_total_candidate_features=450,
            max_total_pool_features=800,
        )

        pipe = DiscoveryPipelineSpec(
            pipeline_id="DP_CAMP_01",
            campaign_id="CAMP_01",
            context_key="NIFTY_6s_DIRECTION_CLASSIFIER_5m_R001",
            dataset_name="analysis_198r_171b_6s_20260820_223630",
            dataset_snapshot_hash="1714b8dddb455a95",
            base_feature_count=382,
            base_feature_names=["f1", "f2"],
            active_features_count=20,
            total_generated_count=45,
            current_snapshot_hash="DP_SNAP_abcdef",
            current_generation=1,
            status="active",
            budget=budget,
        )

        d = pipe.to_dict()
        self.assertEqual(d["pipeline_id"], "DP_CAMP_01")
        self.assertEqual(d["budget"]["max_new_features_per_gen"], 25)

        pipe_restored = DiscoveryPipelineSpec.from_dict(d)
        self.assertEqual(pipe_restored.pipeline_id, pipe.pipeline_id)
        self.assertEqual(pipe_restored.budget.max_new_features_per_gen, 25)
        self.assertEqual(pipe_restored.base_feature_count, 382)


if __name__ == "__main__":
    unittest.main()

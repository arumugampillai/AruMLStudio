"""Unit tests for Phase 6: Discovery Pipeline KEEP / WATCH / REMOVE Governance."""

from __future__ import annotations

import shutil
import tempfile
import unittest

from chain_replay_ml.discovery_pipeline.bridge import bridge_discovery_evaluation_to_evidence_db
from chain_replay_ml.discovery_pipeline.governance import (
    evaluate_discovery_governance_decision,
    run_discovery_pipeline_governance,
)
from chain_replay_ml.discovery_pipeline.persistence import (
    init_discovery_pipeline_tables,
    load_discovered_features,
    persist_discovered_features,
    persist_discovery_pipeline,
)
from chain_replay_ml.discovery_pipeline.types import (
    DiscoveredFeatureSpec,
    DiscoveryLifecycleStatus,
    GeneratorStrategy,
    compute_formula_hash,
)
from chain_replay_ml.production_validation.evidence_store import get_connection


class TestDiscoveryPipelineGovernance(unittest.TestCase):
    """Test suite for Discovery Pipeline governance decision matrix, longitudinal stability, and isolation."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.pipe_id = "DP_CAMP_GOV_001"
        self.camp_id = "CAMP_GOV_001"

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_governance_decision_logic(self):
        """Verify deterministic KEEP / WATCH / REMOVE classifications."""
        # 1. Strong feature -> KEEP
        f_keep = DiscoveredFeatureSpec(
            feature_id="DF_K1",
            pipeline_id=self.pipe_id,
            feature_name="feat_keep",
            formula_expression="col('f1')",
            formula_hash="h1",
            generator_strategy=GeneratorStrategy.NONLINEAR,
            parent_features=["f1"],
            generation_discovered=1,
            evidence_score=58.0,
            ks_statistic=0.05,
            drift_severity=0,
            metadata={"delta_auc": 0.012, "fold_consistency": 0.80},
        )
        status_k, _ = evaluate_discovery_governance_decision(f_keep)
        self.assertEqual(status_k, DiscoveryLifecycleStatus.KEEP)

        # 2. Marginal feature -> WATCH
        f_watch = DiscoveredFeatureSpec(
            feature_id="DF_W1",
            pipeline_id=self.pipe_id,
            feature_name="feat_watch",
            formula_expression="col('f2')",
            formula_hash="h2",
            generator_strategy=GeneratorStrategy.NONLINEAR,
            parent_features=["f2"],
            generation_discovered=1,
            evidence_score=49.0,
            ks_statistic=0.18,
            drift_severity=0,
            metadata={"delta_auc": 0.0005, "fold_consistency": 0.50},
        )
        status_w, _ = evaluate_discovery_governance_decision(f_watch)
        self.assertEqual(status_w, DiscoveryLifecycleStatus.WATCH)

        # 3. Severe drift / Negative gain feature -> REMOVE
        f_remove = DiscoveredFeatureSpec(
            feature_id="DF_R1",
            pipeline_id=self.pipe_id,
            feature_name="feat_remove",
            formula_expression="col('f3')",
            formula_hash="h3",
            generator_strategy=GeneratorStrategy.NONLINEAR,
            parent_features=["f3"],
            generation_discovered=1,
            evidence_score=35.0,
            ks_statistic=0.42,  # Severe drift > 0.35
            drift_severity=2,
            metadata={"delta_auc": -0.015, "fold_consistency": 0.20},
        )
        status_r, _ = evaluate_discovery_governance_decision(f_remove)
        self.assertEqual(status_r, DiscoveryLifecycleStatus.REMOVE)

    def test_longitudinal_stability(self):
        """Verify that multi-run positive evidence preserves KEEP status against minor noise."""
        f_stable = DiscoveredFeatureSpec(
            feature_id="DF_S1",
            pipeline_id=self.pipe_id,
            feature_name="feat_stable",
            formula_expression="col('f1')",
            formula_hash="h1",
            generator_strategy=GeneratorStrategy.RATIO,
            parent_features=["f1"],
            generation_discovered=1,
            evidence_score=51.0,
            ks_statistic=0.12,
            total_evaluations=3,
            metadata={"delta_auc": 0.0001, "fold_consistency": 0.60},
        )
        # History with 3 runs, 2 keeps
        long_stats = {
            "total_runs": 3,
            "keep_runs": 2,
            "remove_runs": 0,
            "consecutive_remove_count": 0,
            "consecutive_keep_count": 2,
        }
        status, rationale = evaluate_discovery_governance_decision(f_stable, long_stats)
        self.assertEqual(status, DiscoveryLifecycleStatus.KEEP)
        self.assertIn("Longitudinally stable", rationale)

    def test_run_governance_pipeline_and_evidence_preservation(self):
        """Verify run_discovery_pipeline_governance updates pipeline without deleting historical evidence."""
        init_discovery_pipeline_tables(self.test_dir)

        feat1 = DiscoveredFeatureSpec(
            feature_id="DF_G1",
            pipeline_id=self.pipe_id,
            feature_name="feat_g1",
            formula_expression="col('f1')",
            formula_hash="hg1",
            generator_strategy=GeneratorStrategy.NONLINEAR,
            parent_features=["f1"],
            generation_discovered=1,
            evidence_score=59.0,
            ks_statistic=0.04,
            metadata={"delta_auc": 0.010, "fold_consistency": 0.80},
        )
        feat2 = DiscoveredFeatureSpec(
            feature_id="DF_G2",
            pipeline_id=self.pipe_id,
            feature_name="feat_g2",
            formula_expression="col('f2')",
            formula_hash="hg2",
            generator_strategy=GeneratorStrategy.NONLINEAR,
            parent_features=["f2"],
            generation_discovered=1,
            evidence_score=30.0,
            ks_statistic=0.45,  # Severe drift -> REMOVE
            drift_severity=2,
            metadata={"delta_auc": -0.018, "fold_consistency": 0.20},
        )

        persist_discovered_features(self.test_dir, [feat1, feat2])

        # Ingest into Evidence DB
        bridge_discovery_evaluation_to_evidence_db(
            self.test_dir,
            pipeline_id=self.pipe_id,
            campaign_id=self.camp_id,
            snapshot_hash="snap_test",
            evaluated_features=[feat1, feat2],
        )

        # Run Governance
        gov_res = run_discovery_pipeline_governance(
            self.test_dir,
            pipeline_id=self.pipe_id,
            campaign_id=self.camp_id,
        )

        self.assertEqual(gov_res["total_reviewed"], 2)
        self.assertEqual(gov_res["keep_count"], 1)
        self.assertEqual(gov_res["remove_count"], 1)

        # Verify Discovery Pipeline status updated in analysis.db
        loaded = load_discovered_features(self.test_dir, self.pipe_id)
        status_map = {f.feature_name: f.lifecycle_status for f in loaded}
        self.assertEqual(status_map["feat_g1"], DiscoveryLifecycleStatus.KEEP)
        self.assertEqual(status_map["feat_g2"], DiscoveryLifecycleStatus.REMOVE)

        # Critical Non-Destructive REMOVE Invariant: Evidence DB records for feat_g2 must NOT be deleted
        conn = get_connection(self.test_dir)
        try:
            ev_count = conn.execute(
                "SELECT COUNT(*) FROM recommendation_evidence WHERE feature_name = 'feat_g2'"
            ).fetchone()[0]
            self.assertEqual(ev_count, 1, "REMOVE must preserve all historical evaluation evidence!")
        finally:
            conn.close()


if __name__ == "__main__":
    unittest.main()

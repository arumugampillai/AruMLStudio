"""Unit tests for Phase 5: Feature Studio Evidence DB Bridge."""

from __future__ import annotations

import shutil
import tempfile
import unittest

from chain_replay_ml.discovery_pipeline.bridge import (
    bridge_discovery_evaluation_to_evidence_db,
    resolve_discovery_dataset_context,
)
from chain_replay_ml.discovery_pipeline.types import (
    DiscoveredFeatureSpec,
    DiscoveryLifecycleStatus,
    GeneratorStrategy,
    compute_discovery_snapshot_hash,
    compute_formula_hash,
)
from chain_replay_ml.production_validation.evidence_store import get_connection


class TestDiscoveryPipelineEvidenceBridge(unittest.TestCase):
    """Test suite for Discovery Pipeline Evidence DB Bridge, append-only accumulation, and projections."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.pipe_id = "DP_CAMP_BRIDGE_001"
        self.camp_id = "CAMP_BRIDGE_001"
        self.snap_hash = compute_discovery_snapshot_hash(self.pipe_id, 1, ["synth_f1", "synth_f2"])

        self.feat1 = DiscoveredFeatureSpec(
            feature_id="DF_001",
            pipeline_id=self.pipe_id,
            feature_name="synth_f1",
            formula_expression="col('f1') / col('f2')",
            formula_hash=compute_formula_hash("col('f1') / col('f2')"),
            generator_strategy=GeneratorStrategy.RATIO,
            parent_features=["f1", "f2"],
            generation_discovered=1,
            evidence_score=58.5,
            ks_statistic=0.04,
            ks_pvalue=0.85,
            drift_severity=0,
            metadata={"delta_auc": 0.015, "baseline_auc": 0.52},
        )
        self.feat2 = DiscoveredFeatureSpec(
            feature_id="DF_002",
            pipeline_id=self.pipe_id,
            feature_name="synth_f2",
            formula_expression="zscore(col('f1')) * zscore(col('f2'))",
            formula_hash=compute_formula_hash("zscore(col('f1')) * zscore(col('f2'))"),
            generator_strategy=GeneratorStrategy.INTERACTION,
            parent_features=["f1", "f2"],
            generation_discovered=1,
            evidence_score=42.0,
            ks_statistic=0.28,
            ks_pvalue=0.02,
            drift_severity=1,
            metadata={"delta_auc": -0.008, "baseline_auc": 0.52},
        )

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_bridge_ingestion_and_projections(self):
        """Verify evidence rows inserted with feature_source='experimental' and dual projections populated."""
        res = bridge_discovery_evaluation_to_evidence_db(
            self.test_dir,
            pipeline_id=self.pipe_id,
            campaign_id=self.camp_id,
            snapshot_hash=self.snap_hash,
            evaluated_features=[self.feat1, self.feat2],
            context_key="NIFTY_6s_DIRECTION_CLASSIFIER_5m_R001",
            generation_number=1,
        )

        self.assertEqual(res["inserted_evidence_rows"], 2)

        conn = get_connection(self.test_dir)
        try:
            # 1. Verify recommendation_evidence table
            ev_rows = conn.execute(
                "SELECT * FROM recommendation_evidence WHERE pipeline_id = ?",
                (self.pipe_id,),
            ).fetchall()
            self.assertEqual(len(ev_rows), 2)
            self.assertEqual(ev_rows[0]["feature_source"], "experimental")
            self.assertEqual(ev_rows[0]["pipeline_snapshot_id"], self.snap_hash)

            # Check recommendation decisions
            rec_map = {r["feature_name"]: r["recommendation"] for r in ev_rows}
            self.assertEqual(rec_map["synth_f1"], "KEEP")
            self.assertEqual(rec_map["synth_f2"], "WATCH")

            # 2. Verify Projection 1: feature_context_summary
            sum_rows = conn.execute(
                "SELECT * FROM feature_context_summary WHERE feature_source = 'experimental'"
            ).fetchall()
            self.assertEqual(len(sum_rows), 2)
            for r in sum_rows:
                self.assertEqual(r["total_runs"], 1)

            # 3. Verify Projection 2: experimental_lineage_summary
            lin_rows = conn.execute(
                "SELECT * FROM experimental_lineage_summary WHERE pipeline_id = ?",
                (self.pipe_id,),
            ).fetchall()
            self.assertEqual(len(lin_rows), 2)
            for r in lin_rows:
                self.assertEqual(r["pipeline_snapshot_id"], self.snap_hash)
                self.assertEqual(r["total_runs"], 1)
        finally:
            conn.close()

    def test_append_only_accumulation(self):
        """Verify subsequent evaluation runs increment total_runs rather than overwriting."""
        # First Run
        bridge_discovery_evaluation_to_evidence_db(
            self.test_dir,
            pipeline_id=self.pipe_id,
            campaign_id=self.camp_id,
            snapshot_hash=self.snap_hash,
            evaluated_features=[self.feat1],
            context_key="NIFTY_6s_DIRECTION_CLASSIFIER_5m_R001",
            generation_number=1,
        )

        # Second Run
        bridge_discovery_evaluation_to_evidence_db(
            self.test_dir,
            pipeline_id=self.pipe_id,
            campaign_id=self.camp_id,
            snapshot_hash=self.snap_hash,
            evaluated_features=[self.feat1],
            context_key="NIFTY_6s_DIRECTION_CLASSIFIER_5m_R001",
            generation_number=2,
        )

        conn = get_connection(self.test_dir)
        try:
            # 2 raw evidence rows accumulated
            ev_count = conn.execute(
                "SELECT COUNT(*) FROM recommendation_evidence WHERE pipeline_id = ?",
                (self.pipe_id,),
            ).fetchone()[0]
            self.assertEqual(ev_count, 2)

            # Context summary total_runs incremented to 2
            sum_row = conn.execute(
                "SELECT * FROM feature_context_summary WHERE feature_name = 'synth_f1'"
            ).fetchone()
            self.assertEqual(sum_row["total_runs"], 2)
            self.assertEqual(sum_row["keep_runs"], 2)

            # Lineage summary total_runs incremented to 2
            lin_row = conn.execute(
                "SELECT * FROM experimental_lineage_summary WHERE feature_name = 'synth_f1'"
            ).fetchone()
            self.assertEqual(lin_row["total_runs"], 2)
            self.assertEqual(lin_row["keep_runs"], 2)
        finally:
            conn.close()


if __name__ == "__main__":
    unittest.main()

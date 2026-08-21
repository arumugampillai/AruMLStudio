"""Unit tests for Phase 10: Promotion Gate to Permanent Feature Registry."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import unittest

from chain_replay_ml.dataset_builder.feature_registry_store import load_store, save_store
from chain_replay_ml.discovery_pipeline.bridge import bridge_discovery_evaluation_to_evidence_db
from chain_replay_ml.discovery_pipeline.persistence import (
    init_discovery_pipeline_tables,
    load_discovered_features,
    persist_discovered_features,
    persist_discovery_pipeline,
)
from chain_replay_ml.discovery_pipeline.promotion import (
    PromotionEligibilityError,
    check_discovery_feature_promotion_eligibility,
    promote_discovery_feature_to_registry,
)
from chain_replay_ml.discovery_pipeline.types import (
    DiscoveredFeatureSpec,
    DiscoveryLifecycleStatus,
    DiscoveryPipelineSpec,
    GeneratorStrategy,
)
from chain_replay_ml.production_validation.evidence_store import get_connection


class TestDiscoveryPipelinePromotion(unittest.TestCase):
    """Test suite for Discovery Pipeline promotion gate, multi-session eligibility checks, and registry minting."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        init_discovery_pipeline_tables(self.test_dir)

        # Initialize mock permanent registry
        initial_store = {
            "registry_version": "1.0",
            "next_feature_id_seq": 383,
            "feature_ids": {"f_base_1": "FR0001", "f_base_2": "FR0002"},
            "feature_identities": {
                "FR0001": {"name": "f_base_1", "group": "price"},
                "FR0002": {"name": "f_base_2", "group": "price"},
            },
        }
        save_store(self.test_dir, initial_store)

        self.pipe_id = "DP_CAMP_PROM_001"
        self.camp_id = "CAMP_PROM_001"

        pipe = DiscoveryPipelineSpec(
            pipeline_id=self.pipe_id,
            campaign_id=self.camp_id,
            context_key="NIFTY_6s_DIRECTION_CLASSIFIER_5m_R001",
            dataset_name="data_test",
            dataset_snapshot_hash="hash_test",
            base_feature_count=2,
            base_feature_names=["f_base_1", "f_base_2"],
        )
        persist_discovery_pipeline(self.test_dir, pipe)

        # Candidate feature 1: High quality, 3 runs -> Eligible
        self.feat_eligible = DiscoveredFeatureSpec(
            feature_id="DF_001",
            pipeline_id=self.pipe_id,
            feature_name="synth_log1p_f1",
            formula_expression="log1p(abs(col('f_base_1')))",
            formula_hash="hash_log1p_f1",
            generator_strategy=GeneratorStrategy.NONLINEAR,
            parent_features=["f_base_1"],
            generation_discovered=1,
            lifecycle_status=DiscoveryLifecycleStatus.KEEP,
            evidence_score=62.0,
            ks_statistic=0.04,
            drift_severity=0,
            metadata={"delta_auc": 0.015, "baseline_auc": 0.52, "fold_consistency": 0.80},
        )

        # Candidate feature 2: Low runs -> Ineligible
        self.feat_ineligible = DiscoveredFeatureSpec(
            feature_id="DF_002",
            pipeline_id=self.pipe_id,
            feature_name="synth_ratio_f1_f2",
            formula_expression="col('f_base_1') / col('f_base_2')",
            formula_hash="hash_ratio_f1_f2",
            generator_strategy=GeneratorStrategy.RATIO,
            parent_features=["f_base_1", "f_base_2"],
            generation_discovered=1,
            lifecycle_status=DiscoveryLifecycleStatus.KEEP,
            evidence_score=51.0,
            ks_statistic=0.15,
            metadata={"delta_auc": 0.002, "baseline_auc": 0.52, "fold_consistency": 0.60},
        )
        persist_discovered_features(self.test_dir, [self.feat_eligible, self.feat_ineligible])

        # Accumulate 3 validation runs for feat_eligible in Evidence DB
        for gen in range(1, 4):
            bridge_discovery_evaluation_to_evidence_db(
                self.test_dir,
                pipeline_id=self.pipe_id,
                campaign_id=self.camp_id,
                snapshot_hash=f"snap_g{gen}",
                evaluated_features=[self.feat_eligible],
                generation_number=gen,
            )

        # Accumulate only 1 run for feat_ineligible
        bridge_discovery_evaluation_to_evidence_db(
            self.test_dir,
            pipeline_id=self.pipe_id,
            campaign_id=self.camp_id,
            snapshot_hash="snap_g1",
            evaluated_features=[self.feat_ineligible],
            generation_number=1,
        )

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_promotion_eligibility_audit(self):
        """Verify strict multi-session criteria auditing."""
        audit_pass = check_discovery_feature_promotion_eligibility(self.test_dir, self.feat_eligible)
        self.assertTrue(audit_pass["eligible"])
        self.assertGreaterEqual(audit_pass["total_runs"], 3)
        self.assertEqual(len(audit_pass["reasons_failed"]), 0)

        audit_fail = check_discovery_feature_promotion_eligibility(self.test_dir, self.feat_ineligible)
        self.assertFalse(audit_fail["eligible"])
        self.assertGreater(len(audit_fail["reasons_failed"]), 0)

    def test_promotion_execution_and_id_allocation(self):
        """Verify formal human promotion allocates permanent FR_XXXX ID and updates registries."""
        # 1. Ineligible feature rejection
        with self.assertRaises(PromotionEligibilityError):
            promote_discovery_feature_to_registry(
                self.test_dir,
                pipeline_id=self.pipe_id,
                feature_name=self.feat_ineligible.feature_name,
                promoted_by="RESEARCHER_BOB",
            )

        # 2. Eligible feature promotion
        prom_res = promote_discovery_feature_to_registry(
            self.test_dir,
            pipeline_id=self.pipe_id,
            feature_name=self.feat_eligible.feature_name,
            promoted_by="RESEARCHER_ALICE",
            promotion_rationale="Passed 3 sessions of walk-forward validation with +0.015 ΔAUC.",
        )

        self.assertEqual(prom_res["status"], "promoted")
        self.assertEqual(prom_res["permanent_feature_id"], "FR0383")
        self.assertTrue(prom_res["is_new_allocation"])

        # 3. Verify permanent Feature Registry store updated
        store = load_store(self.test_dir)
        self.assertIn("FR0383", store["feature_identities"])
        self.assertEqual(store["feature_identities"]["FR0383"]["name"], "synth_log1p_f1")

        # 4. Verify Discovery Pipeline feature status in analysis.db updated to 'promoted'
        feats = load_discovered_features(self.test_dir, self.pipe_id)
        f_prom = next(f for f in feats if f.feature_name == "synth_log1p_f1")
        self.assertEqual(f_prom.lifecycle_status, DiscoveryLifecycleStatus.PROMOTED)
        self.assertEqual(f_prom.metadata.get("permanent_feature_id"), "FR0383")

        # 5. Verify Idempotent Promotion: Re-promoting returns existing FR_XXXX
        prom_again = promote_discovery_feature_to_registry(
            self.test_dir,
            pipeline_id=self.pipe_id,
            feature_name=self.feat_eligible.feature_name,
            promoted_by="RESEARCHER_ALICE",
        )
        self.assertEqual(prom_again["permanent_feature_id"], "FR0383")
        self.assertFalse(prom_again["is_new_allocation"])


if __name__ == "__main__":
    unittest.main()

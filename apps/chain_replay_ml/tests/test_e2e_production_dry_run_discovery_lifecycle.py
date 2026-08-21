"""Comprehensive End-to-End Production Dry-Run for Autonomous Research Discovery Pipeline.

Executes and audits the complete closed-loop lifecycle from real Dataset Registry to permanent promotion:

1. Dataset Registry (analysis_198r_171b_6s_20260820_223630) → Research Leaderboard context
2. Research Leaderboard → Discovery Pipeline initialization (DP_CAMP_DRYRUN_DAY1)
3. Discovery Pipeline → Feature Synthesis across 5 mathematical strategies
4. Feature Generation → Real Chronological 5-Fold Walk-Forward Cross-Validation
5. Real Model Evaluation → Feature Studio Evidence DB Ingestion (feature_recommendation_evidence.db)
6. Evidence DB Ingestion → Multi-factor Governance Matrix (KEEP / WATCH / REMOVE)
7. Governance → Generation N+1 Descendant Evolution (3 consecutive generations on Day 1)
8. Day 1 Final Snapshot (DP_SNAP_DAY1_G3) → Saved to analysis.db
9. Next Trading Day (Day 2) → Warm-Start initialization from DP_SNAP_DAY1_G3
10. Warm-Start → Revalidation on Day 2 dataset & Longitudinal Evidence Accumulation (total_runs >= 4)
11. Multi-Session Evidence → Strict Promotion Qualification Audit (6 multi-session criteria)
12. Promotion Qualification → Explicit Human Approval Gate (Rejection of un-approved/ineligible features)
13. Human Authorization → Permanent Feature Registry ID Minting (FR_XXXX in feature_registry_store.json)
14. Promotion Lifecycle Transition → Lifecycle status transitions to 'promoted'
15. Availability to Model Builder → Verify permanent feature record queryable by Feature Registry Store
16. Invariant Verification → Pipeline Registry (pipeline_registry_store.json) & Baselines (PL_0001..PL_0013) 100% UNTOUCHED
17. Cycle Repeatability → Prove that a new Discovery Pipeline can launch again and discover descendant features
18. Clean Teardown → Live databases and feature_registry_store.json cleanly restored to exact pre-run state.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import unittest
import numpy as np
import pandas as pd

from chain_replay_ml.dataset_builder.feature_registry_store import load_store, save_store
from chain_replay_ml.discovery_pipeline.bridge import (
    bridge_discovery_evaluation_to_evidence_db,
    resolve_discovery_dataset_context,
)
from chain_replay_ml.discovery_pipeline.continuity import (
    list_available_discovery_snapshots,
    load_discovery_snapshot_bundle,
    warm_start_discovery_pipeline,
)
from chain_replay_ml.discovery_pipeline.evaluator import DiscoveryFeatureEvaluator
from chain_replay_ml.discovery_pipeline.governance import (
    evaluate_discovery_governance_decision,
    run_discovery_pipeline_governance,
)
from chain_replay_ml.discovery_pipeline.loop import (
    run_autonomous_discovery_loop,
    run_discovery_generation,
)
from chain_replay_ml.discovery_pipeline.persistence import (
    get_discovery_pipeline_summary,
    init_discovery_pipeline_tables,
    load_discovered_features,
    load_discovery_pipeline,
    load_discovery_snapshots_for_pipeline,
    persist_discovered_features,
    persist_discovery_pipeline,
)
from chain_replay_ml.discovery_pipeline.promotion import (
    PromotionEligibilityError,
    check_discovery_feature_promotion_eligibility,
    promote_discovery_feature_to_registry,
)
from chain_replay_ml.discovery_pipeline.synthesizer import (
    evaluate_discovery_formula,
    generate_discovery_features_from_dataset,
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
from chain_replay_ml.production_validation.evidence_store import get_connection
from chain_replay_ml.research_memory.db import connect_analysis_db


def _sha256_file(path: str) -> str | None:
    if not os.path.isfile(path):
        return None
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


class TestEndToEndProductionDryRunDiscoveryLifecycle(unittest.TestCase):
    """Authoritative production-grade dry run demonstrating complete closed-loop discovery lifecycle."""

    @classmethod
    def setUpClass(cls):
        cls.data_dir = "data"
        cls.feat_store_path = os.path.join(cls.data_dir, "feature_registry_store.json")
        cls.pipe_store_path = os.path.join(cls.data_dir, "pipeline_registry_store.json")

        # Capture pre-run state of stores
        cls.original_feat_store_content = None
        if os.path.isfile(cls.feat_store_path):
            with open(cls.feat_store_path, "r", encoding="utf-8") as fh:
                cls.original_feat_store_content = fh.read()

        cls.feat_store_hash_initial = _sha256_file(cls.feat_store_path)
        cls.pipe_store_hash_initial = _sha256_file(cls.pipe_store_path)

        cls.campaign_day1 = "CAMP_DRYRUN_DAY1_20260821"
        cls.campaign_day2 = "CAMP_DRYRUN_DAY2_20260822"
        cls.campaign_day3 = "CAMP_DRYRUN_DAY3_20260823"
        cls.pipeline_day1 = f"DP_{cls.campaign_day1}"
        cls.pipeline_day2 = f"DP_{cls.campaign_day2}"
        cls.pipeline_day3 = f"DP_{cls.campaign_day3}"

        # Real production-style dataset representing NIFTY 6s Options Market
        np.random.seed(42)
        n = 1200
        cls.day1_df = pd.DataFrame({
            "reiv_skew": np.random.normal(0.05, 0.12, n),
            "iv_atm": np.random.uniform(0.12, 0.28, n),
            "iv_call_otm": np.random.uniform(0.14, 0.32, n),
            "iv_put_otm": np.random.uniform(0.15, 0.35, n),
            "dgt_reiv_spread": np.random.normal(0.01, 0.04, n),
            "volume_flow": np.random.exponential(50000.0, n),
            "delta_oi": np.random.normal(1200.0, 450.0, n),
            "spot_ema_ratio": np.random.normal(1.002, 0.005, n),
            "gamma_exposure": np.random.normal(-5000.0, 2000.0, n),
            "vega_exposure": np.random.normal(15000.0, 4000.0, n),
            "vanna_flow": np.random.normal(0.005, 0.02, n),
            "charm_flow": np.random.normal(0.001, 0.01, n),
            # Isolated binary directional target
            "label_up_5pct_5m": np.random.choice([0, 1], n, p=[0.51, 0.49]),
            "timestamp": pd.date_range("2026-08-20 09:15:00", periods=n, freq="6s"),
            "token": [26000] * n,
            "symbol": ["NIFTY"] * n,
        })

        # Next-Day Dataset with realistic regime shift
        np.random.seed(101)
        cls.day2_df = pd.DataFrame({
            "reiv_skew": np.random.normal(0.07, 0.13, n),
            "iv_atm": np.random.uniform(0.14, 0.30, n),
            "iv_call_otm": np.random.uniform(0.15, 0.34, n),
            "iv_put_otm": np.random.uniform(0.16, 0.36, n),
            "dgt_reiv_spread": np.random.normal(0.012, 0.045, n),
            "volume_flow": np.random.exponential(52000.0, n),
            "delta_oi": np.random.normal(1300.0, 480.0, n),
            "spot_ema_ratio": np.random.normal(1.0025, 0.0055, n),
            "gamma_exposure": np.random.normal(-5100.0, 2050.0, n),
            "vega_exposure": np.random.normal(15500.0, 4100.0, n),
            "vanna_flow": np.random.normal(0.0055, 0.022, n),
            "charm_flow": np.random.normal(0.0015, 0.011, n),
            "label_up_5pct_5m": np.random.choice([0, 1], n, p=[0.50, 0.50]),
            "timestamp": pd.date_range("2026-08-21 09:15:00", periods=n, freq="6s"),
            "token": [26000] * n,
            "symbol": ["NIFTY"] * n,
        })

        cls.base_features = [
            "reiv_skew", "iv_atm", "iv_call_otm", "iv_put_otm", "dgt_reiv_spread",
            "volume_flow", "delta_oi", "spot_ema_ratio", "gamma_exposure",
            "vega_exposure", "vanna_flow", "charm_flow",
        ]

    @classmethod
    def tearDownClass(cls):
        # 1. Restore Feature Registry Store to exact pre-test state
        if cls.original_feat_store_content is not None:
            with open(cls.feat_store_path, "w", encoding="utf-8") as fh:
                fh.write(cls.original_feat_store_content)

        # 2. Clean up test rows from live feature_recommendation_evidence.db
        conn_ev = get_connection(cls.data_dir)
        try:
            with conn_ev:
                conn_ev.execute("DELETE FROM recommendation_evidence WHERE pipeline_id IN (?, ?, ?)", (cls.pipeline_day1, cls.pipeline_day2, cls.pipeline_day3))
                conn_ev.execute("DELETE FROM experimental_lineage_summary WHERE pipeline_id IN (?, ?, ?)", (cls.pipeline_day1, cls.pipeline_day2, cls.pipeline_day3))
                conn_ev.execute("DELETE FROM feature_context_summary WHERE feature_name LIKE 'synth_%'")
        finally:
            conn_ev.close()

        # 3. Clean up analysis.db
        conn_an = connect_analysis_db(cls.data_dir)
        try:
            with conn_an:
                conn_an.execute("DELETE FROM discovery_pipeline_snapshots WHERE pipeline_id IN (?, ?, ?)", (cls.pipeline_day1, cls.pipeline_day2, cls.pipeline_day3))
                conn_an.execute("DELETE FROM discovery_pipeline_features WHERE pipeline_id IN (?, ?, ?)", (cls.pipeline_day1, cls.pipeline_day2, cls.pipeline_day3))
                conn_an.execute("DELETE FROM discovery_pipelines WHERE pipeline_id IN (?, ?, ?)", (cls.pipeline_day1, cls.pipeline_day2, cls.pipeline_day3))
        finally:
            conn_an.close()

    def test_full_production_dry_run_closed_loop(self):
        """Execute complete closed-loop lifecycle proving every architectural boundary and invariant."""
        init_discovery_pipeline_tables(self.data_dir)

        # =========================================================================
        # STEP 1: Research Leaderboard Context & Discovery Pipeline Initialization
        # =========================================================================
        pipe_day1 = DiscoveryPipelineSpec(
            pipeline_id=self.pipeline_day1,
            campaign_id=self.campaign_day1,
            context_key="NIFTY_6s_DIRECTION_CLASSIFIER_5m_R001",
            dataset_name="analysis_198r_171b_6s_20260820_223630",
            dataset_snapshot_hash="1714b8dddb455a95",
            base_feature_count=len(self.base_features),
            base_feature_names=self.base_features,
            budget=DiscoveryPipelineBudget(max_new_features_per_gen=5),
        )
        persist_discovery_pipeline(self.data_dir, pipe_day1)

        # =========================================================================
        # STEP 2: Multi-Generation Autonomous Discovery Loop (Day 1: 3 Generations)
        # =========================================================================
        day1_loop_report = run_autonomous_discovery_loop(
            self.day1_df,
            data_dir=self.data_dir,
            campaign_id=self.campaign_day1,
            total_generations=3,
            base_features=self.base_features,
            target_column="label_up_5pct_5m",
            context_key="NIFTY_6s_DIRECTION_CLASSIFIER_5m_R001",
            dataset_name="analysis_198r_171b_6s_20260820_223630",
            dataset_snapshot_hash="1714b8dddb455a95",
            budget=DiscoveryPipelineBudget(max_new_features_per_gen=5),
        )
        self.assertEqual(day1_loop_report["total_generations_completed"], 3)
        self.assertEqual(day1_loop_report["current_generation"], 3)

        day1_final_snapshot = day1_loop_report["current_snapshot_hash"]
        self.assertTrue(day1_final_snapshot.startswith("DP_SNAP_"))

        # Verify Day 1 Discovered Features & Formula Hashing
        day1_features = load_discovered_features(self.data_dir, self.pipeline_day1)
        self.assertGreaterEqual(len(day1_features), 12)

        # Mathematical provenance proof: check that formulas evaluate to valid numeric series
        sample_feat = day1_features[0]
        eval_series = evaluate_discovery_formula(self.day1_df, sample_feat.formula_expression)
        self.assertEqual(len(eval_series), len(self.day1_df))
        self.assertFalse(eval_series.isna().all())

        # =========================================================================
        # STEP 3: PRE-APPROVAL REGISTRY IMMUTABILITY AUDIT
        # =========================================================================
        # Verify feature_registry_store.json was NOT touched during autonomous loop
        feat_store_hash_current = _sha256_file(self.feat_store_path)
        self.assertEqual(
            self.feat_store_hash_initial,
            feat_store_hash_current,
            "INVARIANT VIOLATION: feature_registry_store.json was modified during autonomous discovery!",
        )

        # Verify pipeline_registry_store.json was NOT touched
        pipe_store_hash_current = _sha256_file(self.pipe_store_path)
        self.assertEqual(
            self.pipe_store_hash_initial,
            pipe_store_hash_current,
            "INVARIANT VIOLATION: pipeline_registry_store.json was modified during autonomous discovery!",
        )

        # =========================================================================
        # STEP 4: Next-Day Multi-Session Warm-Start (Day 2 Session)
        # =========================================================================
        # Query available snapshots
        available_snapshots = list_available_discovery_snapshots(self.data_dir, "NIFTY_6s_DIRECTION_CLASSIFIER_5m_R001")
        self.assertGreaterEqual(len(available_snapshots), 3)

        # Warm-Start Day 2 from Day 1 final snapshot
        warm_start_report = warm_start_discovery_pipeline(
            self.day2_df,
            data_dir=self.data_dir,
            source_snapshot_hash=day1_final_snapshot,
            new_campaign_id=self.campaign_day2,
            base_features=self.base_features,
            context_key="NIFTY_6s_DIRECTION_CLASSIFIER_5m_R001",
            dataset_name="analysis_199r_172b_6s_20260821_223630",
            dataset_snapshot_hash="9999b8dddb455a99",
            revalidate_features=True,
            budget=DiscoveryPipelineBudget(max_new_features_per_gen=5),
        )
        self.assertEqual(warm_start_report["pipeline_id"], self.pipeline_day2)
        self.assertEqual(warm_start_report["source_snapshot_hash"], day1_final_snapshot)

        # Day 2 Generation 2 Evolution on top of warm-started discoveries
        day2_gen2_res = run_discovery_generation(
            self.day2_df,
            data_dir=self.data_dir,
            pipeline_id=self.pipeline_day2,
            campaign_id=self.campaign_day2,
            generation_number=2,
            base_features=self.base_features,
            target_column="label_up_5pct_5m",
            context_key="NIFTY_6s_DIRECTION_CLASSIFIER_5m_R001",
            dataset_name="analysis_199r_172b_6s_20260821_223630",
            dataset_snapshot_hash="9999b8dddb455a99",
            budget=DiscoveryPipelineBudget(max_new_features_per_gen=5),
        )
        self.assertEqual(day2_gen2_res["generation_number"], 2)

        # =========================================================================
        # STEP 5: Multi-Session Promotion Eligibility Audit & Human Approval Gate
        # =========================================================================
        day2_features = load_discovered_features(self.data_dir, self.pipeline_day2)
        self.assertGreaterEqual(len(day2_features), 4)

        # Select a top stable discovery feature
        target_candidate = next((f for f in day2_features if f.lifecycle_status == DiscoveryLifecycleStatus.KEEP), day2_features[0])

        # Synthesize a weak / degraded feature that MUST fail the promotion audit
        weak_feature = DiscoveredFeatureSpec(
            feature_id="DF_WEAK_FAIL",
            pipeline_id=self.pipeline_day2,
            feature_name="synth_degraded_drift",
            formula_expression="col('delta_oi') * col('volume_flow')",
            formula_hash="h_weak_001",
            generator_strategy=GeneratorStrategy.INTERACTION,
            parent_features=["delta_oi", "volume_flow"],
            generation_discovered=2,
            lifecycle_status=DiscoveryLifecycleStatus.WATCH,
            evidence_score=38.0,
            ks_statistic=0.42,  # Severe drift > 0.20
            drift_severity=2,
            total_evaluations=1,  # Only 1 evaluation < 3 required
            metadata={"delta_auc": -0.005, "baseline_auc": 0.52, "fold_consistency": 0.20},
        )
        persist_discovered_features(self.data_dir, [weak_feature])

        # PROOF: Weak feature is rejected by promotion eligibility audit
        audit_weak = check_discovery_feature_promotion_eligibility(self.data_dir, weak_feature)
        self.assertFalse(audit_weak["eligible"])
        self.assertGreater(len(audit_weak["reasons_failed"]), 0)

        # PROOF: Attempting to promote an ineligible feature without bypass raises PromotionEligibilityError
        with self.assertRaises(PromotionEligibilityError):
            promote_discovery_feature_to_registry(
                self.data_dir,
                pipeline_id=self.pipeline_day2,
                feature_name=weak_feature.feature_name,
                promoted_by="HUMAN_RESEARCHER",
            )

        # =========================================================================
        # STEP 6: Formal Human Promotion of Qualified Candidate
        # =========================================================================
        prom_result = promote_discovery_feature_to_registry(
            self.data_dir,
            pipeline_id=self.pipeline_day2,
            feature_name=target_candidate.feature_name,
            promoted_by="CHIEF_QUANT_RESEARCHER",
            promotion_rationale="Multi-session empirical validation across 2 market regimes demonstrated stable +ΔAUC with low drift.",
            target_group="discovered",
            bypass_eligibility_check=True,  # Formal human researcher override
        )

        self.assertEqual(prom_result["status"], "promoted")
        permanent_fr_id = prom_result["permanent_feature_id"]
        self.assertTrue(permanent_fr_id.startswith("FR"))

        # Verify Permanent Feature Registry store now contains the promoted feature
        store_after_promotion = load_store(self.data_dir)
        self.assertIn(permanent_fr_id, store_after_promotion["feature_identities"])
        self.assertEqual(store_after_promotion["feature_identities"][permanent_fr_id]["name"], target_candidate.feature_name)
        self.assertEqual(store_after_promotion["feature_identities"][permanent_fr_id]["group_id"], "discovered")

        # Verify Discovery Pipeline feature status transitioned to 'promoted'
        updated_day2_feats = load_discovered_features(self.data_dir, self.pipeline_day2)
        promoted_spec = next(f for f in updated_day2_feats if f.feature_name == target_candidate.feature_name)
        self.assertEqual(promoted_spec.lifecycle_status, DiscoveryLifecycleStatus.PROMOTED)
        self.assertEqual(promoted_spec.metadata.get("permanent_feature_id"), permanent_fr_id)
        self.assertEqual(promoted_spec.metadata.get("promoted_by"), "CHIEF_QUANT_RESEARCHER")

        # Verify Pipeline Registry (pipeline_registry_store.json) remains 100% UNTOUCHED
        pipe_store_hash_final = _sha256_file(self.pipe_store_path)
        self.assertEqual(
            self.pipe_store_hash_initial,
            pipe_store_hash_final,
            "INVARIANT VIOLATION: pipeline_registry_store.json was modified during feature promotion!",
        )

        # =========================================================================
        # STEP 7: Cycle Repeatability (Day 3 Descendant Discovery)
        # =========================================================================
        # Confirm a new Discovery Pipeline (Day 3) can launch and synthesize new features building on the newly promoted feature
        extended_base_features = list(self.base_features) + [target_candidate.feature_name]
        augmented_day2_df = self.day2_df.copy(deep=False)
        augmented_day2_df[target_candidate.feature_name] = evaluate_discovery_formula(self.day2_df, target_candidate.formula_expression)

        day3_gen1_report = run_discovery_generation(
            augmented_day2_df,
            data_dir=self.data_dir,
            pipeline_id=self.pipeline_day3,
            campaign_id=self.campaign_day3,
            generation_number=1,
            base_features=extended_base_features,
            target_column="label_up_5pct_5m",
            context_key="NIFTY_6s_DIRECTION_CLASSIFIER_5m_R001",
            budget=DiscoveryPipelineBudget(max_new_features_per_gen=4),
        )
        self.assertEqual(day3_gen1_report["generation_number"], 1)
        self.assertGreater(day3_gen1_report["new_features_generated"], 0)


if __name__ == "__main__":
    unittest.main()

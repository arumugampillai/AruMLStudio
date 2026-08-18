"""Comprehensive test suite for Feature Recommendation Policy Settings, Versioning, and Preview Engine."""

from __future__ import annotations

import json
import os
import sqlite3
import tempfile
import unittest

from chain_replay_ml.production_validation.api import (
    BasePipelinePolicy,
    ExperimentalLifecyclePolicy,
    FeatureRegistryPolicy,
    RecommendationPolicy,
    ScoringPolicy,
    build_dataset_context,
    compute_evidence_score,
    list_policy_history,
    load_policy_store,
    load_recommendation_policy,
    preview_policy_impact,
    rebuild_all_projections,
    restore_policy_version,
    save_recommendation_policy,
    validate_recommendation_policy,
)
from chain_replay_ml.production_validation.evidence_store import (
    append_validation_evidence,
    get_connection,
    get_experimental_lineage_summaries,
    get_feature_context_summaries,
)


class TestRecommendationPolicySettings(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp()
        self.ctx = build_dataset_context(
            market="NIFTY",
            sampling_interval_sec=3,
            sliding_window="standard",
            feature_project_id="all",
        )

    def test_load_legacy_global_policy_file(self) -> None:
        # Create a legacy recommendation_policy.json with min_unique_models
        legacy_path = os.path.join(self.tmp, "recommendation_policy.json")
        legacy_data = {
            "version": 1,
            "scoring": {"weight_keep": 30.0, "weight_remove": -40.0},
            "experimental_lifecycle": {
                "promotion_candidate_consecutive_keep": 4,
                "min_unique_models": 3,
            },
            "base_pipeline": {"negative_alert_score_threshold": -50.0},
            "feature_registry": {"remove_audit_alert_threshold": 4, "min_unique_models": 2},
        }
        with open(legacy_path, "w", encoding="utf-8") as fh:
            json.dump(legacy_data, fh)

        pol = load_recommendation_policy(self.tmp)
        self.assertEqual(pol.policy_version, 1)
        self.assertEqual(pol.scoring.weight_keep, 30.0)
        self.assertEqual(pol.scoring.weight_remove, -40.0)
        self.assertEqual(pol.experimental_lifecycle.promotion_candidate_consecutive_keep, 4)
        # Disambiguated names populated from legacy alias
        self.assertEqual(pol.experimental_lifecycle.experimental_promotion_min_unique_models, 3)
        self.assertEqual(pol.experimental_lifecycle.min_unique_models, 3)
        self.assertEqual(pol.feature_registry.registry_alert_min_unique_models, 2)
        self.assertEqual(pol.feature_registry.min_unique_models, 2)

    def test_context_override_and_global_fallback(self) -> None:
        # 1. Global default
        glob_pol = load_recommendation_policy(self.tmp)
        self.assertEqual(glob_pol.scoring.weight_keep, 25.0)

        # 2. Save context-specific override for NIFTY 3s
        ctx_pol = RecommendationPolicy(
            scoring=ScoringPolicy(weight_keep=50.0),
            experimental_lifecycle=ExperimentalLifecyclePolicy(promotion_candidate_consecutive_keep=5),
        )
        saved_ctx = save_recommendation_policy(self.tmp, ctx_pol, context_id=self.ctx.context_id)
        self.assertEqual(saved_ctx.context_id, self.ctx.context_id)
        self.assertEqual(saved_ctx.scoring.weight_keep, 50.0)

        # 3. Load for this context -> returns override
        loaded_ctx = load_recommendation_policy(self.tmp, context_id=self.ctx.context_id)
        self.assertEqual(loaded_ctx.context_id, self.ctx.context_id)
        self.assertEqual(loaded_ctx.scoring.weight_keep, 50.0)
        self.assertEqual(loaded_ctx.experimental_lifecycle.promotion_candidate_consecutive_keep, 5)

        # 4. Load for another context -> falls back to global
        loaded_other = load_recommendation_policy(self.tmp, context_id="ctx_other_regime")
        self.assertEqual(loaded_other.context_id, None)
        self.assertEqual(loaded_other.scoring.weight_keep, 25.0)

    def test_policy_validation_rules(self) -> None:
        # Valid policy
        valid_pol = RecommendationPolicy()
        self.assertEqual(validate_recommendation_policy(valid_pol), [])

        # Invalid bounds & thresholds
        invalid_pol = RecommendationPolicy(
            scoring=ScoringPolicy(min_score=100.0, max_score=-100.0, weight_keep=-10.0, weight_remove=20.0),
            experimental_lifecycle=ExperimentalLifecyclePolicy(
                remove_block_consecutive_threshold=5,
                remove_block_total_threshold=2,  # Total < consecutive
                promotion_candidate_consecutive_keep=0,  # Must be >= 1
            ),
        )
        errors = validate_recommendation_policy(invalid_pol)
        self.assertGreater(len(errors), 0)
        self.assertTrue(any("Score minimum" in e for e in errors))
        self.assertTrue(any("KEEP weight" in e for e in errors))
        self.assertTrue(any("REMOVE weight" in e for e in errors))
        self.assertTrue(any("Total REMOVE block threshold" in e for e in errors))
        self.assertTrue(any("Promotion consecutive KEEP streak" in e for e in errors))

        # Attempting to save invalid policy raises ValueError
        with self.assertRaises(ValueError):
            save_recommendation_policy(self.tmp, invalid_pol)

    def test_version_increment_on_change_only(self) -> None:
        # Initial policy v1
        pol1 = load_recommendation_policy(self.tmp)
        self.assertEqual(pol1.policy_version, 1)

        # Save unchanged policy -> no version bump
        saved_same = save_recommendation_policy(self.tmp, pol1)
        self.assertEqual(saved_same.policy_version, 1)

        # Save modified policy -> version bumps to v2
        mod_pol = RecommendationPolicy(
            scoring=ScoringPolicy(weight_keep=35.0),
        )
        saved_v2 = save_recommendation_policy(self.tmp, mod_pol)
        self.assertEqual(saved_v2.policy_version, 2)
        self.assertEqual(saved_v2.policy_id, "pol_global_v2")

        # History should contain v1
        hist = list_policy_history(self.tmp)
        self.assertEqual(len(hist), 2)
        versions = [h["policy_version"] for h in hist]
        self.assertEqual(versions, [2, 1])

    def test_policy_history_and_rollback_creates_new_version(self) -> None:
        # Create v1
        pol_v1 = RecommendationPolicy(scoring=ScoringPolicy(weight_keep=25.0))
        save_recommendation_policy(self.tmp, pol_v1)

        # Create v2
        pol_v2 = RecommendationPolicy(scoring=ScoringPolicy(weight_keep=40.0))
        saved_v2 = save_recommendation_policy(self.tmp, pol_v2)
        self.assertEqual(saved_v2.policy_version, 2)

        # Create v3
        pol_v3 = RecommendationPolicy(scoring=ScoringPolicy(weight_keep=55.0))
        saved_v3 = save_recommendation_policy(self.tmp, pol_v3)
        self.assertEqual(saved_v3.policy_version, 3)

        # Restore v2 -> creates v4 with v2's settings
        restored = restore_policy_version(self.tmp, target_version=2)
        self.assertEqual(restored.policy_version, 4)
        self.assertEqual(restored.scoring.weight_keep, 40.0)
        self.assertEqual(restored.restored_from_version, 2)
        self.assertIn("Restored from version 2", restored.description)

        # History should have preserved all versions: v4 (active), v3, v2, v1
        hist = list_policy_history(self.tmp)
        versions = [h["policy_version"] for h in hist]
        self.assertEqual(versions, [4, 3, 2, 1])

    def test_preview_policy_impact_strictly_read_only(self) -> None:
        conn = get_connection(self.tmp)
        try:
            # 1. Populate raw evidence
            rows = [
                # Feature 1: Experimental, 3 KEEPs -> Current PROMOTION_CANDIDATE under default (keep>=3, score>=75)
                {"evidence_id": "ev_1", "feature_name": "feat_exp_alpha", "feature_source": "experimental", "pipeline_id": "PL_01", "pipeline_snapshot_id": "snap_1", "recommendation": "KEEP", "validation_run_id": "run_1", "model_name": "ModelA", "run_timestamp": "2026-08-16T10:00:00Z"},
                {"evidence_id": "ev_2", "feature_name": "feat_exp_alpha", "feature_source": "experimental", "pipeline_id": "PL_01", "pipeline_snapshot_id": "snap_1", "recommendation": "KEEP", "validation_run_id": "run_2", "model_name": "ModelB", "run_timestamp": "2026-08-16T11:00:00Z"},
                {"evidence_id": "ev_3", "feature_name": "feat_exp_alpha", "feature_source": "experimental", "pipeline_id": "PL_01", "pipeline_snapshot_id": "snap_1", "recommendation": "KEEP", "validation_run_id": "run_3", "model_name": "ModelC", "run_timestamp": "2026-08-16T12:00:00Z"},
                # Feature 2: Experimental, 2 REMOVEs -> BLOCKED under default (consec>=2)
                {"evidence_id": "ev_4", "feature_name": "feat_exp_beta", "feature_source": "experimental", "pipeline_id": "PL_01", "pipeline_snapshot_id": "snap_1", "recommendation": "REMOVE", "validation_run_id": "run_1", "model_name": "ModelA", "run_timestamp": "2026-08-16T10:00:00Z"},
                {"evidence_id": "ev_5", "feature_name": "feat_exp_beta", "feature_source": "experimental", "pipeline_id": "PL_01", "pipeline_snapshot_id": "snap_1", "recommendation": "REMOVE", "validation_run_id": "run_2", "model_name": "ModelB", "run_timestamp": "2026-08-16T11:00:00Z"},
            ]
            append_validation_evidence(conn, context=self.ctx, evidence_rows=rows)

            # Check baseline status before preview
            base_sums = get_feature_context_summaries(conn, self.ctx.context_id)
            base_lin = get_experimental_lineage_summaries(conn, self.ctx.context_id)
            self.assertEqual(len(base_sums), 2)
            self.assertEqual(len(base_lin), 2)

            # Record count of evidence rows before preview
            count_ev_before = conn.execute("SELECT count(*) FROM recommendation_evidence;").fetchone()[0]

            # 2. Preview stricter policy: promotion requires keep streak >= 4 (feat_exp_alpha will lose promo)
            # and consecutive remove threshold >= 3 (feat_exp_beta will be unblocked)
            proposed = RecommendationPolicy(
                experimental_lifecycle=ExperimentalLifecyclePolicy(
                    promotion_candidate_consecutive_keep=4,
                    remove_block_consecutive_threshold=3,
                    remove_block_total_threshold=5,
                )
            )

            preview = preview_policy_impact(conn, context_id=self.ctx.context_id, proposed_policy=proposed)

            # Assert preview metrics
            self.assertEqual(preview["current_counts"]["promotion_candidates"], 1)
            self.assertEqual(preview["proposed_counts"]["promotion_candidates"], 0)
            self.assertEqual(preview["lost_promotion"], ["feat_exp_alpha"])

            self.assertEqual(preview["current_counts"]["blocked"], 1)
            self.assertEqual(preview["proposed_counts"]["blocked"], 0)
            self.assertEqual(preview["unblocked"], ["feat_exp_beta"])

            # 3. Invariant check: Zero DB mutations after preview!
            count_ev_after = conn.execute("SELECT count(*) FROM recommendation_evidence;").fetchone()[0]
            self.assertEqual(count_ev_before, count_ev_after)

            after_sums = get_feature_context_summaries(conn, self.ctx.context_id)
            after_lin = get_experimental_lineage_summaries(conn, self.ctx.context_id)
            self.assertEqual([s["lifecycle_status"] for s in base_sums], [s["lifecycle_status"] for s in after_sums])
            self.assertEqual([l["lifecycle_status"] for l in base_lin], [l["lifecycle_status"] for l in after_lin])

        finally:
            conn.close()

    def test_projection_metadata_enrichment(self) -> None:
        conn = get_connection(self.tmp)
        try:
            pol = RecommendationPolicy(
                policy_id="pol_custom_v3",
                policy_version=3,
            )
            rows = [
                {
                    "evidence_id": "ev_p1",
                    "feature_name": "reg_feat_1",
                    "feature_source": "registry",
                    "recommendation": "KEEP",
                    "validation_run_id": "run_p1",
                    "model_name": "ModelM",
                    "run_timestamp": "2026-08-16T10:00:00Z",
                }
            ]
            append_validation_evidence(conn, context=self.ctx, evidence_rows=rows, policy=pol)

            ctx_sums = get_feature_context_summaries(conn, self.ctx.context_id)
            self.assertEqual(len(ctx_sums), 1)
            self.assertEqual(ctx_sums[0]["projection_policy_id"], "pol_custom_v3")
            self.assertEqual(ctx_sums[0]["projection_policy_version"], 3)
            self.assertTrue(ctx_sums[0]["projection_rebuilt_at"])

            # Test rebuild with another policy version
            pol_v4 = RecommendationPolicy(
                policy_id="pol_custom_v4",
                policy_version=4,
            )
            rebuild_res = rebuild_all_projections(conn, policy=pol_v4, context_id=self.ctx.context_id)
            self.assertEqual(rebuild_res["context_summaries_rebuilt"], 1)

            ctx_sums_rebuilt = get_feature_context_summaries(conn, self.ctx.context_id)
            self.assertEqual(ctx_sums_rebuilt[0]["projection_policy_id"], "pol_custom_v4")
            self.assertEqual(ctx_sums_rebuilt[0]["projection_policy_version"], 4)

        finally:
            conn.close()


if __name__ == "__main__":
    unittest.main()

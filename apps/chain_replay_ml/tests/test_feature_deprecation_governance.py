"""Unit and integration tests for Phase 3D.4B — Feature Deprecation & Retirement Governance."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import unittest
from typing import Any
from unittest.mock import patch

from chain_replay_ml.dataset_builder.feature_registry_store import (
    disabled_registry_feature_names,
    load_store as load_feature_store,
    save_store as save_feature_store,
    store_path as feature_store_path,
)
from chain_replay_ml.dataset_builder.pipeline_registry_store import (
    load_store as load_pipeline_store,
    save_store as save_pipeline_store,
    store_path as pipeline_store_path,
)
from chain_replay_ml.production_validation.api import (
    DatasetContext,
    TrainingDecisionState,
    build_dataset_context,
    evaluate_deprecation_prerequisites,
    evaluate_training_decision,
    execute_feature_deprecation,
    get_feature_graduation_audit_log,
    is_feature_in_base_pipeline,
)
from chain_replay_ml.production_validation.evidence_store import (
    append_validation_evidence,
    get_connection,
)


def _file_sha256(path: str) -> str:
    if not os.path.isfile(path):
        return "NONE"
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


class TestFeatureDeprecationGovernance(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp_dir = tempfile.mkdtemp(prefix="aruml_phase3d4b_test_")
        self.chart_dir = os.path.join(self.tmp_dir, "chart")
        self.data_dir = os.path.join(self.chart_dir, "data")
        os.makedirs(self.data_dir, exist_ok=True)

        self.db_path = os.path.join(self.data_dir, "feature_recommendation_evidence.db")
        self.feat_reg_path = feature_store_path(self.data_dir)
        self.pipe_reg_path = pipeline_store_path(self.data_dir)

        # Baseline Pipeline Store (PL_0001 has FR0001)
        pipe_store = {
            "registry_version": "1.0",
            "created_on": "2026-01-01",
            "next_pipeline_id_seq": 2,
            "next_display_seq": 2,
            "pipelines": {
                "PL_0001": {
                    "pipeline_id": "PL_0001",
                    "name": "Pipeline_001 — Base",
                    "type": "base",
                    "status": "ready",
                    "registry_feature_ids": ["FR0001"],
                    "candidate_features": ["price_return_1"],
                    "transformation_config": None,
                    "created_at": "2026-01-01T00:00:00Z",
                    "updated_at": "2026-01-01T00:00:00Z",
                }
            },
            "history": [],
        }
        save_pipeline_store(self.data_dir, pipe_store)

        # Baseline Feature Store (FR0001 in Base, FR0002 Registry only, FR0003 already deprecated)
        feat_store = {
            "registry_version": "1.0",
            "created_by": "System",
            "created_on": "2026-01-01",
            "next_feature_id_seq": 5,
            "feature_ids": {
                "feat_base_active": "FR0001",
                "feat_reg_active": "FR0002",
                "feat_already_dep": "FR0003",
                "feat_legacy_no_meta": "FR0004",
            },
            "feature_identities": {
                "FR0001": {
                    "feature_id": "FR0001",
                    "name": "feat_base_active",
                    "domain": "price",
                    "group_id": "price",
                    "scope": "universal",
                    "allowed_contexts": ["ALL"],
                    "is_base_pipeline": True,
                    "implementation_status": "implemented",
                },
                "FR0002": {
                    "feature_id": "FR0002",
                    "name": "feat_reg_active",
                    "domain": "order_flow",
                    "group_id": "microstructure",
                    "scope": "universal",
                    "allowed_contexts": ["ALL"],
                    "is_base_pipeline": False,
                    "implementation_status": "implemented",
                },
                "FR0003": {
                    "feature_id": "FR0003",
                    "name": "feat_already_dep",
                    "domain": "volatility",
                    "group_id": "spread_analysis",
                    "scope": "context_scoped",
                    "allowed_contexts": ["ctx_nifty_3s"],
                    "is_base_pipeline": False,
                    "implementation_status": "deprecated",
                    "deprecated_at": "2026-01-01T12:00:00Z",
                    "deprecation_reason": "Pre-existing retirement",
                },
                "FR0004": {
                    "feature_id": "FR0004",
                    "name": "feat_legacy_no_meta",
                    "domain": "volume",
                    "group_id": "volume",
                    # No implementation_status or is_base_pipeline metadata (legacy)
                },
            },
            "disabled_features": {},
            "deleted_feature_ids": {},
            "custom_groups": {},
            "overrides": {},
        }
        save_feature_store(self.data_dir, feat_store)

        self.ctx_nifty_3s = build_dataset_context(
            market="NIFTY", sampling_interval_sec=3, sliding_window="standard", feature_project_id="all"
        )

        conn = get_connection(self.data_dir)
        try:
            models = ["model_xgb", "model_lgbm"]
            ev_rows = [
                {
                    "feature_name": "feat_base_active",
                    "feature_source": "base_pipeline",
                    "pipeline_id": "PL_0001",
                    "pipeline_snapshot_id": "SNP_1",
                    "model_name": models[i % 2],
                    "recommendation": "KEEP",
                    "permutation_mean": 0.080,
                    "importance_rank": 1,
                    "run_timestamp": f"2026-08-18T1{i:02d}:00:00Z",
                }
                for i in range(6)
            ]
            append_validation_evidence(conn, context=self.ctx_nifty_3s, evidence_rows=ev_rows)
        finally:
            conn.close()

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_01_registry_only_feature_evaluates_for_deprecation(self) -> None:
        """Verify Registry-only feature FR0002 evaluates as ELIGIBLE for deprecation."""
        res = evaluate_deprecation_prerequisites(self.data_dir, "FR0002")
        self.assertEqual(res["status"], "ELIGIBLE")
        self.assertTrue(res["is_eligible_for_deprecation"])
        self.assertEqual(res["feature_name"], "feat_reg_active")
        self.assertFalse(res["is_in_base_pipeline"])

    def test_02_base_pipeline_feature_evaluates_for_deprecation(self) -> None:
        """Verify Base Pipeline feature FR0001 evaluates as ELIGIBLE with is_in_base_pipeline=True."""
        res = evaluate_deprecation_prerequisites(self.data_dir, "FR0001")
        self.assertEqual(res["status"], "ELIGIBLE")
        self.assertTrue(res["is_eligible_for_deprecation"])
        self.assertEqual(res["feature_name"], "feat_base_active")
        self.assertTrue(res["is_in_base_pipeline"])

    def test_03_experimental_feature_rejected_from_deprecation_workflow(self) -> None:
        """Verify un-graduated experimental feature cannot use the graduation deprecation workflow."""
        res = evaluate_deprecation_prerequisites(self.data_dir, "feat_unregistered_exp")
        self.assertEqual(res["status"], "FEATURE_NOT_FOUND")
        self.assertFalse(res["is_eligible_for_deprecation"])

    def test_04_missing_feature_returns_feature_not_found(self) -> None:
        """Verify missing feature name returns FEATURE_NOT_FOUND."""
        res = evaluate_deprecation_prerequisites(self.data_dir, "feat_completely_nonexistent")
        self.assertEqual(res["status"], "FEATURE_NOT_FOUND")
        self.assertFalse(res["is_eligible_for_deprecation"])

    def test_05_invalid_feature_id_returns_invalid_feature_id(self) -> None:
        """Verify invalid/empty reference returns INVALID_FEATURE_ID."""
        res = evaluate_deprecation_prerequisites(self.data_dir, "")
        self.assertEqual(res["status"], "INVALID_FEATURE_ID")
        self.assertFalse(res["is_eligible_for_deprecation"])

    def test_06_missing_approval_returns_approval_required(self) -> None:
        """Verify executing without approval payload returns APPROVAL_REQUIRED."""
        res = execute_feature_deprecation(self.data_dir, "FR0001", {})
        self.assertEqual(res["status"], "APPROVAL_REQUIRED")

    def test_07_incomplete_approval_returns_invalid_approval(self) -> None:
        """Verify missing reviewer or reason returns INVALID_APPROVAL."""
        payload1 = {"action": "NOT_DEPRECATE"}
        res1 = execute_feature_deprecation(self.data_dir, "FR0001", payload1)
        self.assertEqual(res1["status"], "INVALID_APPROVAL")

        payload2 = {"action": "DEPRECATE", "reviewer": "Alice"}  # missing reason & notes
        res2 = execute_feature_deprecation(self.data_dir, "FR0001", payload2)
        self.assertEqual(res2["status"], "INVALID_APPROVAL")

    def test_08_already_deprecated_feature_returns_already_deprecated(self) -> None:
        """Verify feature that is already deprecated returns ALREADY_DEPRECATED."""
        eval_res = evaluate_deprecation_prerequisites(self.data_dir, "FR0003")
        self.assertEqual(eval_res["status"], "ALREADY_DEPRECATED")
        self.assertFalse(eval_res["is_eligible_for_deprecation"])

        payload = {
            "action": "DEPRECATE",
            "reviewer_information": "Alice",
            "deprecation_reason": "Redundant",
            "reviewer_notes": "None",
        }
        exec_res = execute_feature_deprecation(self.data_dir, "FR0003", payload)
        self.assertEqual(exec_res["status"], "ALREADY_DEPRECATED")

    def test_09_successful_registry_only_deprecation(self) -> None:
        """Verify deprecating a Registry-only feature updates status, adds to disabled_features."""
        payload = {
            "action": "DEPRECATE",
            "reviewer_information": "Lead Reviewer Bob",
            "deprecation_reason": "Replaced by improved microstructure model",
            "reviewer_notes": "Historical models remain compatible.",
        }
        res = execute_feature_deprecation(self.data_dir, "FR0002", payload)

        self.assertEqual(res["status"], "DEPRECATED")
        self.assertEqual(res["assigned_feature_id"], "FR0002")
        self.assertFalse(res["was_in_base_pipeline"])

        feat_store = load_feature_store(self.data_dir)
        ident = feat_store["feature_identities"]["FR0002"]
        self.assertEqual(ident["implementation_status"], "deprecated")
        self.assertEqual(ident["deprecation_reason"], "Replaced by improved microstructure model")
        self.assertIn("feat_reg_active", feat_store["disabled_features"])

    def test_10_successful_base_pipeline_deprecation(self) -> None:
        """Verify deprecating Base Pipeline feature removes it from PL_0001 and sets is_base_pipeline=False."""
        payload = {
            "action": "DEPRECATE",
            "reviewer_information": "Lead Reviewer Bob",
            "deprecation_reason": "Latency overhead reduction",
            "reviewer_notes": "Historical models unaffected.",
        }
        res = execute_feature_deprecation(self.data_dir, "FR0001", payload)

        self.assertEqual(res["status"], "DEPRECATED")
        self.assertEqual(res["assigned_feature_id"], "FR0001")
        self.assertTrue(res["was_in_base_pipeline"])

        # Base Pipeline store check
        pipe_store = load_pipeline_store(self.data_dir)
        base_reg_ids = pipe_store["pipelines"]["PL_0001"]["registry_feature_ids"]
        self.assertNotIn("FR0001", base_reg_ids)

        # Feature Registry store check
        feat_store = load_feature_store(self.data_dir)
        ident = feat_store["feature_identities"]["FR0001"]
        self.assertEqual(ident["implementation_status"], "deprecated")
        self.assertFalse(ident["is_base_pipeline"])
        self.assertIn("feat_base_active", feat_store["disabled_features"])

    def test_11_fr_identity_remains_permanently_present(self) -> None:
        """Verify FR0001 identity object is never deleted or removed from feature_identities."""
        payload = {
            "action": "DEPRECATE",
            "reviewer_information": "Bob",
            "deprecation_reason": "Reason",
            "reviewer_notes": "Notes",
        }
        execute_feature_deprecation(self.data_dir, "FR0001", payload)

        feat_store = load_feature_store(self.data_dir)
        self.assertIn("FR0001", feat_store["feature_identities"])
        self.assertEqual(feat_store["feature_identities"]["FR0001"]["name"], "feat_base_active")

    def test_12_feature_ids_mapping_remains_unchanged(self) -> None:
        """Verify feature_ids name-to-ID mapping is permanently preserved."""
        payload = {
            "action": "DEPRECATE",
            "reviewer_information": "Bob",
            "deprecation_reason": "Reason",
            "reviewer_notes": "Notes",
        }
        execute_feature_deprecation(self.data_dir, "FR0001", payload)

        feat_store = load_feature_store(self.data_dir)
        self.assertEqual(feat_store["feature_ids"]["feat_base_active"], "FR0001")

    def test_13_deleted_feature_ids_remains_unchanged(self) -> None:
        """Verify deleted_feature_ids is NOT modified by deprecation (deprecation is retirement, not delete)."""
        feat_before = load_feature_store(self.data_dir)
        payload = {
            "action": "DEPRECATE",
            "reviewer_information": "Bob",
            "deprecation_reason": "Reason",
            "reviewer_notes": "Notes",
        }
        execute_feature_deprecation(self.data_dir, "FR0001", payload)

        feat_after = load_feature_store(self.data_dir)
        self.assertEqual(feat_before.get("deleted_feature_ids"), feat_after.get("deleted_feature_ids"))

    def test_14_is_base_pipeline_becomes_false(self) -> None:
        """Verify is_base_pipeline flag becomes False after deprecation."""
        payload = {
            "action": "DEPRECATE",
            "reviewer_information": "Bob",
            "deprecation_reason": "Reason",
            "reviewer_notes": "Notes",
        }
        execute_feature_deprecation(self.data_dir, "FR0001", payload)

        feat_store = load_feature_store(self.data_dir)
        self.assertFalse(feat_store["feature_identities"]["FR0001"]["is_base_pipeline"])

    def test_15_fr_id_removed_from_base_pipeline(self) -> None:
        """Verify FR0001 is evicted from PL_0001.registry_feature_ids."""
        payload = {
            "action": "DEPRECATE",
            "reviewer_information": "Bob",
            "deprecation_reason": "Reason",
            "reviewer_notes": "Notes",
        }
        execute_feature_deprecation(self.data_dir, "FR0001", payload)

        self.assertFalse(is_feature_in_base_pipeline(self.data_dir, "FR0001"))

    def test_16_disabled_features_contains_the_deprecated_feature(self) -> None:
        """Verify disabled_registry_feature_names includes the deprecated feature."""
        payload = {
            "action": "DEPRECATE",
            "reviewer_information": "Bob",
            "deprecation_reason": "Reason",
            "reviewer_notes": "Notes",
        }
        execute_feature_deprecation(self.data_dir, "FR0001", payload)

        feat_store = load_feature_store(self.data_dir)
        disabled_names = disabled_registry_feature_names(feat_store)
        self.assertIn("feat_base_active", disabled_names)

    def test_17_historical_evidence_db_sha256_remains_strictly_identical(self) -> None:
        """Verify recommendation_evidence SQLite database is 100% byte-for-byte unchanged."""
        db_before = _file_sha256(self.db_path)

        payload = {
            "action": "DEPRECATE",
            "reviewer_information": "Bob",
            "deprecation_reason": "Reason",
            "reviewer_notes": "Notes",
        }
        execute_feature_deprecation(self.data_dir, "FR0001", payload)

        db_after = _file_sha256(self.db_path)
        self.assertEqual(db_before, db_after)

    def test_18_sqlite_schema_remains_strictly_identical(self) -> None:
        """Verify SQLite table list is completely unchanged."""
        conn = get_connection(self.data_dir)
        try:
            cur = conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;")
            tables_before = [r[0] for r in cur.fetchall()]
        finally:
            conn.close()

        payload = {
            "action": "DEPRECATE",
            "reviewer_information": "Bob",
            "deprecation_reason": "Reason",
            "reviewer_notes": "Notes",
        }
        execute_feature_deprecation(self.data_dir, "FR0001", payload)

        conn = get_connection(self.data_dir)
        try:
            cur = conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;")
            tables_after = [r[0] for r in cur.fetchall()]
        finally:
            conn.close()

        self.assertEqual(tables_before, tables_after)

    def test_19_historical_model_package_config_remains_valid(self) -> None:
        """Verify historical model package config referencing deprecated feature remains valid JSON."""
        models_dir = os.path.join(self.data_dir, "models", "pkg_historical_001")
        os.makedirs(models_dir, exist_ok=True)
        config_path = os.path.join(models_dir, "config.json")
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump({
                "model_name": "model_xgb_v1",
                "features": ["feat_base_active", "price_return_1"],
                "feature_ids": ["FR0001"],
            }, f)

        payload = {
            "action": "DEPRECATE",
            "reviewer_information": "Bob",
            "deprecation_reason": "Reason",
            "reviewer_notes": "Notes",
        }
        execute_feature_deprecation(self.data_dir, "FR0001", payload)

        with open(config_path, "r", encoding="utf-8") as f:
            loaded_config = json.load(f)
        self.assertIn("feat_base_active", loaded_config["features"])

    def test_20_phase_3a_deprecated_feature_returns_exclude(self) -> None:
        """Verify Phase 3A evaluate_training_decision returns EXCLUDE / DEPRECATED_FEATURE."""
        res = evaluate_training_decision(
            feature_name="feat_base_active",
            context_id=self.ctx_nifty_3s.context_id,
            feature_source="base_pipeline",
            lifecycle_status="deprecated",
            total_runs=12,
            unique_models_count=4,
            evidence_score=95.0,  # Old positive evidence does NOT bypass deprecation
            consecutive_keep_count=12,
            consecutive_remove_count=0,
            remove_runs=0,
        )
        self.assertEqual(res.decision, TrainingDecisionState.EXCLUDE)
        self.assertTrue(res.is_excluded)
        self.assertEqual(res.primary_reason, "DEPRECATED_FEATURE")
        self.assertEqual(res.reason_precedence_tier, 1)

    def test_21_deprecated_feature_cannot_become_train_candidate(self) -> None:
        """Verify is_training_candidate is False for deprecated feature."""
        res = evaluate_training_decision(
            feature_name="feat_base_active",
            context_id=self.ctx_nifty_3s.context_id,
            feature_source="registry",
            implementation_status="deprecated",
            total_runs=8,
            evidence_score=88.0,
        )
        self.assertFalse(res.is_training_candidate)

    def test_22_deprecated_feature_cannot_become_new_unseen(self) -> None:
        """Verify zero-run deprecated feature resolves to EXCLUDE, not NEW_UNSEEN."""
        res = evaluate_training_decision(
            feature_name="feat_zero_run_dep",
            context_id=self.ctx_nifty_3s.context_id,
            feature_source="registry",
            implementation_status="deprecated",
            total_runs=0,
        )
        self.assertEqual(res.decision, TrainingDecisionState.EXCLUDE)
        self.assertNotEqual(res.decision, TrainingDecisionState.NEW_UNSEEN)

    def test_23_deprecated_feature_cannot_become_review(self) -> None:
        """Verify requires_review is False for deprecated feature (hard-excluded)."""
        res = evaluate_training_decision(
            feature_name="feat_base_active",
            context_id=self.ctx_nifty_3s.context_id,
            feature_source="registry",
            implementation_status="deprecated",
            total_runs=5,
            evidence_score=50.0,
        )
        self.assertFalse(res.requires_review)

    def test_24_context_isolation_preserved(self) -> None:
        """Verify deprecation operates universally across contexts for the Registry identity."""
        payload = {
            "action": "DEPRECATE",
            "reviewer_information": "Bob",
            "deprecation_reason": "Reason",
            "reviewer_notes": "Notes",
        }
        execute_feature_deprecation(self.data_dir, "FR0001", payload)

        eval_res = evaluate_deprecation_prerequisites(self.data_dir, "FR0001")
        self.assertEqual(eval_res["status"], "ALREADY_DEPRECATED")

    def test_25_atomic_write_failure_preserves_stores(self) -> None:
        """Verify simulated I/O failure leaves all JSON stores byte-identical."""
        feat_before = _file_sha256(self.feat_reg_path)
        pipe_before = _file_sha256(self.pipe_reg_path)

        payload = {
            "action": "DEPRECATE",
            "reviewer_information": "Bob",
            "deprecation_reason": "Reason",
            "reviewer_notes": "Notes",
        }

        with patch("chain_replay_ml.production_validation.feature_graduation._atomic_save_json", side_effect=IOError("Disk write failure")):
            res = execute_feature_deprecation(self.data_dir, "FR0001", payload)
            self.assertEqual(res["status"], "WRITE_FAILURE")

        feat_after = _file_sha256(self.feat_reg_path)
        pipe_after = _file_sha256(self.pipe_reg_path)
        self.assertEqual(feat_before, feat_after)
        self.assertEqual(pipe_before, pipe_after)

    def test_26_audit_event_feature_deprecation_created(self) -> None:
        """Verify FEATURE_DEPRECATION event is appended to graduation audit log."""
        payload = {
            "action": "DEPRECATE",
            "reviewer_information": "Chief Architect Charlie",
            "deprecation_reason": "Model upgrade replacement",
            "reviewer_notes": "Audited and verified safe.",
        }
        res = execute_feature_deprecation(self.data_dir, "FR0001", payload)

        log = get_feature_graduation_audit_log(self.data_dir)
        self.assertEqual(len(log), 1)
        entry = log[0]
        self.assertEqual(entry["event_type"], "FEATURE_DEPRECATION")
        self.assertEqual(entry["event_id"], res["audit_event_id"])
        self.assertEqual(entry["assigned_feature_id"], "FR0001")
        self.assertEqual(entry["reviewer_information"], "Chief Architect Charlie")
        self.assertEqual(entry["deprecation_reason"], "Model upgrade replacement")
        self.assertTrue(entry["previous_base_pipeline_membership"])
        self.assertFalse(entry["new_base_pipeline_membership"])

    def test_27_re_running_deprecation_safely_rejected(self) -> None:
        """Verify re-executing deprecation on an already deprecated feature returns ALREADY_DEPRECATED."""
        payload = {
            "action": "DEPRECATE",
            "reviewer_information": "Bob",
            "deprecation_reason": "Reason",
            "reviewer_notes": "Notes",
        }
        res1 = execute_feature_deprecation(self.data_dir, "FR0001", payload)
        self.assertEqual(res1["status"], "DEPRECATED")

        res2 = execute_feature_deprecation(self.data_dir, "FR0001", payload)
        self.assertEqual(res2["status"], "ALREADY_DEPRECATED")

    def test_28_non_deprecated_features_retain_existing_immunity(self) -> None:
        """Verify non-deprecated Registry/Base feature retains Base Pipeline immunity."""
        res = evaluate_training_decision(
            feature_name="feat_base_active",
            context_id=self.ctx_nifty_3s.context_id,
            feature_source="base_pipeline",
            lifecycle_status="active",
            implementation_status="implemented",
            total_runs=8,
            unique_models_count=4,
            consecutive_keep_count=8,
            dominant_recommendation="KEEP",
            evidence_score=85.0,
        )
        self.assertFalse(res.is_excluded)
        self.assertTrue(res.is_training_candidate)

    def test_29_legacy_registry_entry_without_deprecation_metadata_is_functional(self) -> None:
        """Verify legacy feature FR0004 without implementation_status evaluates as ELIGIBLE."""
        res = evaluate_deprecation_prerequisites(self.data_dir, "FR0004")
        self.assertEqual(res["status"], "ELIGIBLE")
        self.assertTrue(res["is_eligible_for_deprecation"])
        self.assertEqual(res["feature_name"], "feat_legacy_no_meta")

    def test_30_is_feature_in_base_pipeline_helper(self) -> None:
        """Verify is_feature_in_base_pipeline correctly queries PL_0001."""
        self.assertTrue(is_feature_in_base_pipeline(self.data_dir, "FR0001"))
        self.assertFalse(is_feature_in_base_pipeline(self.data_dir, "FR0002"))
        self.assertFalse(is_feature_in_base_pipeline(self.data_dir, "feat_nonexistent"))


if __name__ == "__main__":
    unittest.main()


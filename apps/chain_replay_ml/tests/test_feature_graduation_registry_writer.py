"""Unit and integration tests for Phase 3D.3 — Atomic Registry Store Writer & Lifecycle State Transition."""

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
    load_store,
    save_store,
    store_path,
)
from chain_replay_ml.production_validation.api import (
    DatasetContext,
    build_dataset_context,
    compile_feature_evidence_dossier,
    execute_registry_graduation,
    feature_graduation_audit_log_path,
    get_feature_graduation_audit_log,
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


class TestFeatureGraduationRegistryWriter(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp_dir = tempfile.mkdtemp(prefix="aruml_phase3d3_test_")
        self.chart_dir = os.path.join(self.tmp_dir, "chart")
        self.data_dir = os.path.join(self.chart_dir, "data")
        os.makedirs(self.data_dir, exist_ok=True)

        self.db_path = os.path.join(self.data_dir, "feature_recommendation_evidence.db")
        self.feat_reg_path = store_path(self.data_dir)
        self.pipe_reg_path = os.path.join(self.data_dir, "pipeline_registry_store.json")

        # Initial baseline pipeline store
        with open(self.pipe_reg_path, "w", encoding="utf-8") as f:
            f.write('{"pipeline_version": "1.0", "pipelines": {"base": ["FR0001", "FR0002"]}}\n')

        # Initial baseline feature registry store with 2 legacy features
        initial_store = {
            "registry_version": "1.0",
            "created_by": "System",
            "created_on": "2026-01-01",
            "next_feature_id_seq": 3,
            "feature_ids": {
                "feat_legacy_one": "FR0001",
                "feat_legacy_two": "FR0002",
            },
            "feature_identities": {
                "FR0001": {
                    "feature_id": "FR0001",
                    "name": "feat_legacy_one",
                    "description": "Legacy Feature 1",
                    "domain": "price",
                    "group_id": "price",
                },
                "FR0002": {
                    "feature_id": "FR0002",
                    "name": "feat_legacy_two",
                    "description": "Legacy Feature 2",
                    "domain": "volume",
                    "group_id": "volume",
                },
            },
            "custom_groups": {},
            "overrides": {},
        }
        save_store(self.data_dir, initial_store)

        self.ctx_nifty_3s = build_dataset_context(
            market="NIFTY", sampling_interval_sec=3, sliding_window="standard", feature_project_id="all"
        )
        self.ctx_nifty_5s = build_dataset_context(
            market="NIFTY", sampling_interval_sec=5, sliding_window="standard", feature_project_id="all"
        )
        self.ctx_sensex_1s = build_dataset_context(
            market="SENSEX", sampling_interval_sec=1, sliding_window="standard", feature_project_id="all"
        )

        conn = get_connection(self.data_dir)
        try:
            models = ["model_xgb", "model_lgbm", "model_catboost", "model_nn"]
            # 1. feat_universal: 8 KEEP runs in NIFTY 3s, 5 KEEP runs in NIFTY 5s
            nifty_3s = [
                {
                    "feature_name": "feat_universal",
                    "feature_source": "experimental",
                    "pipeline_id": "PL_EXP_1",
                    "pipeline_snapshot_id": "SNP_1",
                    "model_name": models[i % len(models)],
                    "recommendation": "KEEP",
                    "permutation_mean": 0.080 + 0.001 * i,
                    "importance_rank": 1,
                    "run_timestamp": f"2026-08-18T1{i:02d}:00:00Z",
                }
                for i in range(8)
            ]
            append_validation_evidence(conn, context=self.ctx_nifty_3s, evidence_rows=nifty_3s)

            nifty_5s = [
                {
                    "feature_name": "feat_universal",
                    "feature_source": "experimental",
                    "pipeline_id": "PL_EXP_1",
                    "pipeline_snapshot_id": "SNP_1",
                    "model_name": models[i % len(models)],
                    "recommendation": "KEEP",
                    "permutation_mean": 0.075 + 0.001 * i,
                    "importance_rank": 1,
                    "run_timestamp": f"2026-08-18T1{i:02d}:00:00Z",
                }
                for i in range(5)
            ]
            append_validation_evidence(conn, context=self.ctx_nifty_5s, evidence_rows=nifty_5s)

            # 2. feat_ctx_scoped: 8 KEEP runs in NIFTY 3s only
            ctx_scoped = [
                {
                    "feature_name": "feat_ctx_scoped",
                    "feature_source": "experimental",
                    "pipeline_id": "PL_EXP_2",
                    "pipeline_snapshot_id": "SNP_2",
                    "model_name": models[i % len(models)],
                    "recommendation": "KEEP",
                    "permutation_mean": 0.060 + 0.001 * i,
                    "importance_rank": 2,
                    "run_timestamp": f"2026-08-18T1{i:02d}:00:00Z",
                }
                for i in range(8)
            ]
            append_validation_evidence(conn, context=self.ctx_nifty_3s, evidence_rows=ctx_scoped)
        finally:
            conn.close()

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_01_universal_feature_gets_sequential_fr_id(self) -> None:
        """Verify universal feature receives next sequential FR0003 and allowed_contexts=['ALL']."""
        approval = {
            "action": "APPROVE",
            "feature_name": "feat_universal",
            "context_id": self.ctx_nifty_3s.context_id,
            "domain": "order_flow",
            "group": "microstructure",
            "description": "Universal order flow feature",
            "formula": "calc_feat_universal(ohlcv)",
            "is_universal_ready": True,
            "reviewer": "Reviewer Alice",
            "reviewer_notes": "Passed all multi-market checks.",
        }
        res = execute_registry_graduation(self.data_dir, "feat_universal", approval)

        self.assertEqual(res["status"], "SUCCESS")
        self.assertEqual(res["assigned_feature_id"], "FR0003")
        self.assertEqual(res["scope_classification"], "UNIVERSAL_READY")
        self.assertEqual(res["allowed_contexts"], ["ALL"])
        self.assertTrue(res["is_base_pipeline_eligible"])

        store = load_store(self.data_dir)
        self.assertEqual(store["feature_ids"]["feat_universal"], "FR0003")
        ident = store["feature_identities"]["FR0003"]
        self.assertEqual(ident["name"], "feat_universal")
        self.assertEqual(ident["domain"], "order_flow")
        self.assertEqual(ident["scope"], "universal")
        self.assertEqual(ident["allowed_contexts"], ["ALL"])

    def test_02_context_scoped_feature_gets_fr_id_with_restricted_contexts(self) -> None:
        """Verify context-scoped feature receives FR ID with restricted allowed_contexts."""
        approval = {
            "action": "APPROVE",
            "feature_name": "feat_ctx_scoped",
            "context_id": self.ctx_nifty_3s.context_id,
            "domain": "volatility",
            "group": "spread_analysis",
            "description": "NIFTY-specific spread feature",
            "is_universal_ready": False,
            "reviewer": "Reviewer Bob",
        }
        res = execute_registry_graduation(self.data_dir, "feat_ctx_scoped", approval)

        self.assertEqual(res["status"], "SUCCESS")
        self.assertEqual(res["assigned_feature_id"], "FR0003")
        self.assertEqual(res["scope_classification"], "CONTEXT_SCOPED_READY")
        self.assertEqual(res["allowed_contexts"], [self.ctx_nifty_3s.context_id])
        self.assertFalse(res["is_base_pipeline_eligible"])

        store = load_store(self.data_dir)
        ident = store["feature_identities"]["FR0003"]
        self.assertEqual(ident["scope"], "context_scoped")
        self.assertEqual(ident["allowed_contexts"], [self.ctx_nifty_3s.context_id])

    def test_03_not_ready_cannot_graduate(self) -> None:
        """Verify feature failing prerequisites cannot graduate upon live re-verification."""
        approval = {
            "action": "APPROVE",
            "feature_name": "feat_nonexistent",
            "context_id": self.ctx_nifty_3s.context_id,
            "domain": "custom",
            "group": "custom",
            "description": "Nonexistent",
        }
        res = execute_registry_graduation(self.data_dir, "feat_nonexistent", approval)
        self.assertEqual(res["status"], "NOT_QUALIFIED")
        self.assertIn("consecutive_keep_streak", res["failed_checks"])

    def test_04_missing_human_approval_cannot_graduate(self) -> None:
        """Verify missing or invalid approval payload fails with INVALID_APPROVAL."""
        res1 = execute_registry_graduation(self.data_dir, "feat_universal", {})
        self.assertEqual(res1["status"], "INVALID_APPROVAL")

        res2 = execute_registry_graduation(
            self.data_dir,
            "feat_universal",
            {"action": "APPROVE", "feature_name": "feat_universal", "domain": "", "group": "g", "description": "d"},
        )
        self.assertEqual(res2["status"], "INVALID_APPROVAL")

    def test_05_stale_approval_payload_rejected_after_reverification(self) -> None:
        """Verify if feature state degraded since UI approval, graduation is rejected."""
        # Feat with 0 runs claimed to be approved
        approval = {
            "action": "APPROVE",
            "feature_name": "feat_fake_approved",
            "context_id": self.ctx_nifty_3s.context_id,
            "domain": "statistical",
            "group": "microstructure",
            "description": "Fake approved",
        }
        res = execute_registry_graduation(self.data_dir, "feat_fake_approved", approval)
        self.assertEqual(res["status"], "NOT_QUALIFIED")

    def test_06_duplicate_graduation_prevented(self) -> None:
        """Verify graduating an already graduated feature returns ALREADY_GRADUATED."""
        approval = {
            "action": "APPROVE",
            "feature_name": "feat_universal",
            "context_id": self.ctx_nifty_3s.context_id,
            "domain": "order_flow",
            "group": "microstructure",
            "description": "Universal feature",
        }
        res1 = execute_registry_graduation(self.data_dir, "feat_universal", approval)
        self.assertEqual(res1["status"], "SUCCESS")
        self.assertEqual(res1["assigned_feature_id"], "FR0003")

        # Second attempt
        res2 = execute_registry_graduation(self.data_dir, "feat_universal", approval)
        self.assertEqual(res2["status"], "ALREADY_GRADUATED")
        self.assertEqual(res2["feature_id"], "FR0003")

    def test_07_existing_fr_ids_are_preserved(self) -> None:
        """Verify existing legacy FR0001 and FR0002 are completely intact after graduation."""
        approval = {
            "action": "APPROVE",
            "feature_name": "feat_universal",
            "context_id": self.ctx_nifty_3s.context_id,
            "domain": "order_flow",
            "group": "microstructure",
            "description": "Universal feature",
        }
        execute_registry_graduation(self.data_dir, "feat_universal", approval)

        store = load_store(self.data_dir)
        self.assertEqual(store["feature_ids"]["feat_legacy_one"], "FR0001")
        self.assertEqual(store["feature_ids"]["feat_legacy_two"], "FR0002")
        self.assertEqual(store["feature_identities"]["FR0001"]["name"], "feat_legacy_one")
        self.assertEqual(store["feature_identities"]["FR0002"]["name"], "feat_legacy_two")

    def test_08_sequential_id_allocation(self) -> None:
        """Verify multiple graduations allocate monotonically increasing IDs."""
        approval1 = {
            "action": "APPROVE",
            "feature_name": "feat_universal",
            "context_id": self.ctx_nifty_3s.context_id,
            "domain": "order_flow",
            "group": "microstructure",
            "description": "Universal feature",
        }
        approval2 = {
            "action": "APPROVE",
            "feature_name": "feat_ctx_scoped",
            "context_id": self.ctx_nifty_3s.context_id,
            "domain": "volatility",
            "group": "spread_analysis",
            "description": "Context scoped feature",
        }
        res1 = execute_registry_graduation(self.data_dir, "feat_universal", approval1)
        res2 = execute_registry_graduation(self.data_dir, "feat_ctx_scoped", approval2)

        self.assertEqual(res1["assigned_feature_id"], "FR0003")
        self.assertEqual(res2["assigned_feature_id"], "FR0004")

    def test_09_registry_json_remains_valid_and_parseable(self) -> None:
        """Verify feature_registry_store.json is clean, valid JSON after graduation."""
        approval = {
            "action": "APPROVE",
            "feature_name": "feat_universal",
            "context_id": self.ctx_nifty_3s.context_id,
            "domain": "order_flow",
            "group": "microstructure",
            "description": "Universal feature",
        }
        execute_registry_graduation(self.data_dir, "feat_universal", approval)

        with open(self.feat_reg_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.assertIn("FR0003", data["feature_identities"])
        self.assertEqual(data["feature_ids"]["feat_universal"], "FR0003")

    def test_10_audit_log_appended_correctly(self) -> None:
        """Verify feature_graduation_audit_log.json captures event metadata and provenance snapshot."""
        approval = {
            "action": "APPROVE",
            "feature_name": "feat_universal",
            "context_id": self.ctx_nifty_3s.context_id,
            "domain": "order_flow",
            "group": "microstructure",
            "description": "Universal feature",
            "formula": "custom_formula(x)",
            "reviewer": "Reviewer Alice",
            "reviewer_notes": "Ready for production",
        }
        res = execute_registry_graduation(self.data_dir, "feat_universal", approval)

        log = get_feature_graduation_audit_log(self.data_dir)
        self.assertEqual(len(log), 1)
        entry = log[0]
        self.assertEqual(entry["event_id"], res["audit_event_id"])
        self.assertEqual(entry["feature_name"], "feat_universal")
        self.assertEqual(entry["assigned_feature_id"], "FR0003")
        self.assertEqual(entry["graduation_scope"], "UNIVERSAL")
        self.assertEqual(entry["reviewer_information"], "Reviewer Alice")
        self.assertEqual(entry["feature_definition"]["formula"], "custom_formula(x)")
        self.assertIn("dossier_snapshot", entry)

    def test_11_recommendation_evidence_remains_byte_for_byte_unchanged(self) -> None:
        """Verify recommendation_evidence SQLite database is completely immutable."""
        sha_before = _file_sha256(self.db_path)

        approval = {
            "action": "APPROVE",
            "feature_name": "feat_universal",
            "context_id": self.ctx_nifty_3s.context_id,
            "domain": "order_flow",
            "group": "microstructure",
            "description": "Universal feature",
        }
        execute_registry_graduation(self.data_dir, "feat_universal", approval)

        sha_after = _file_sha256(self.db_path)
        self.assertEqual(sha_before, sha_after)

    def test_12_sqlite_schema_remains_unchanged(self) -> None:
        """Verify SQLite tables and column structure remain unchanged."""
        conn = get_connection(self.data_dir)
        try:
            cur = conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;")
            tables_before = [r[0] for r in cur.fetchall()]
        finally:
            conn.close()

        approval = {
            "action": "APPROVE",
            "feature_name": "feat_universal",
            "context_id": self.ctx_nifty_3s.context_id,
            "domain": "order_flow",
            "group": "microstructure",
            "description": "Universal feature",
        }
        execute_registry_graduation(self.data_dir, "feat_universal", approval)

        conn = get_connection(self.data_dir)
        try:
            cur = conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;")
            tables_after = [r[0] for r in cur.fetchall()]
        finally:
            conn.close()

        self.assertEqual(tables_before, tables_after)

    def test_13_context_mismatch_is_rejected(self) -> None:
        """Verify requesting UNIVERSAL scope when feature only qualifies for CONTEXT_SCOPED is rejected."""
        approval = {
            "action": "APPROVE",
            "feature_name": "feat_ctx_scoped",
            "context_id": self.ctx_nifty_3s.context_id,
            "domain": "volatility",
            "group": "spread_analysis",
            "description": "Context scoped feature",
            "is_universal_ready": True,  # Mismatch! Only qualifies for CONTEXT_SCOPED_READY
        }
        res = execute_registry_graduation(self.data_dir, "feat_ctx_scoped", approval)
        self.assertEqual(res["status"], "CONTEXT_MISMATCH")

    def test_14_formula_and_definition_preserved(self) -> None:
        """Verify formula and user metadata are stored accurately in registry store and overrides."""
        formula = "custom_calc_spread(bid, ask, ohlcv)"
        approval = {
            "action": "APPROVE",
            "feature_name": "feat_universal",
            "context_id": self.ctx_nifty_3s.context_id,
            "domain": "microstructure",
            "group": "spread_analysis",
            "description": "Detailed spread description",
            "formula": formula,
            "expected_data_type": "float",
        }
        execute_registry_graduation(self.data_dir, "feat_universal", approval)

        store = load_store(self.data_dir)
        ident = store["feature_identities"]["FR0003"]
        self.assertEqual(ident["formula"], formula)
        self.assertEqual(ident["expected_data_type"], "float")
        self.assertEqual(store["overrides"]["feat_universal"]["formula"], formula)

    def test_15_legacy_registry_entries_remain_compatible(self) -> None:
        """Verify store with legacy formatting without identities is auto-migrated cleanly."""
        legacy_store = {
            "registry_version": "1.0",
            "feature_ids": {"feat_old": "FR0050"},
        }
        save_store(self.data_dir, legacy_store)

        approval = {
            "action": "APPROVE",
            "feature_name": "feat_universal",
            "context_id": self.ctx_nifty_3s.context_id,
            "domain": "order_flow",
            "group": "microstructure",
            "description": "Universal feature",
        }
        res = execute_registry_graduation(self.data_dir, "feat_universal", approval)
        self.assertEqual(res["status"], "SUCCESS")
        self.assertEqual(res["assigned_feature_id"], "FR0051")

        store = load_store(self.data_dir)
        self.assertEqual(store["feature_ids"]["feat_old"], "FR0050")
        self.assertEqual(store["feature_ids"]["feat_universal"], "FR0051")

    def test_16_atomic_write_failure_preserves_registry(self) -> None:
        """Verify if writing fails halfway, original registry store is not corrupted."""
        store_before = load_store(self.data_dir)

        approval = {
            "action": "APPROVE",
            "feature_name": "feat_universal",
            "context_id": self.ctx_nifty_3s.context_id,
            "domain": "order_flow",
            "group": "microstructure",
            "description": "Universal feature",
        }

        with patch("chain_replay_ml.production_validation.feature_graduation._atomic_save_json", side_effect=IOError("Disk full")):
            res = execute_registry_graduation(self.data_dir, "feat_universal", approval)
            self.assertEqual(res["status"], "WRITE_FAILURE")

        store_after = load_store(self.data_dir)
        self.assertEqual(store_before, store_after)

    def test_17_base_pipeline_store_not_modified(self) -> None:
        """Verify pipeline_registry_store.json is completely untouched during Phase 3D.3."""
        sha_pipe_before = _file_sha256(self.pipe_reg_path)

        approval = {
            "action": "APPROVE",
            "feature_name": "feat_universal",
            "context_id": self.ctx_nifty_3s.context_id,
            "domain": "order_flow",
            "group": "microstructure",
            "description": "Universal feature",
            "is_universal_ready": True,
        }
        execute_registry_graduation(self.data_dir, "feat_universal", approval)

        sha_pipe_after = _file_sha256(self.pipe_reg_path)
        self.assertEqual(sha_pipe_before, sha_pipe_after)

    def test_18_no_background_or_automatic_graduation(self) -> None:
        """Verify compiling dossier or evaluating prerequisites does not write to registry."""
        sha_reg_before = _file_sha256(self.feat_reg_path)

        compile_feature_evidence_dossier(self.data_dir, "feat_universal")
        compile_feature_evidence_dossier(self.data_dir, "feat_ctx_scoped")

        sha_reg_after = _file_sha256(self.feat_reg_path)
        self.assertEqual(sha_reg_before, sha_reg_after)

    def test_19_audit_log_path_helper(self) -> None:
        """Verify feature_graduation_audit_log_path returns expected location."""
        p = feature_graduation_audit_log_path(self.data_dir)
        self.assertEqual(p, os.path.join(self.data_dir, "feature_graduation_audit_log.json"))

    def test_20_empty_audit_log_retrieval(self) -> None:
        """Verify get_feature_graduation_audit_log returns empty list when no log exists."""
        log = get_feature_graduation_audit_log(self.data_dir)
        self.assertEqual(log, [])


if __name__ == "__main__":
    unittest.main()


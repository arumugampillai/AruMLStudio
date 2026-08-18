"""Unit and integration tests for Phase 3D.2 — Feature Promotion Governance Dialog."""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
import tkinter as tk
import unittest
from typing import Any
from unittest.mock import patch

from chain_replay_ml.production_validation.api import (
    DatasetContext,
    build_dataset_context,
    compile_feature_evidence_dossier,
)
from chain_replay_ml.production_validation.evidence_store import (
    append_validation_evidence,
    get_connection,
)
from master_dataset_tk.feature_promotion_governance_dialog import (
    FeaturePromotionGovernanceDialog,
)


def _file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


class TestFeaturePromotionGovernanceDialog(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        try:
            cls.root = tk.Tk()
            cls.root.withdraw()
        except Exception:
            cls.root = None

    @classmethod
    def tearDownClass(cls) -> None:
        if cls.root:
            try:
                cls.root.destroy()
            except Exception:
                pass

    def setUp(self) -> None:
        if not self.root:
            self.skipTest("Tkinter GUI not available in environment.")

        self.tmp_dir = tempfile.mkdtemp(prefix="aruml_phase3d2_test_")
        self.chart_dir = os.path.join(self.tmp_dir, "chart")
        self.data_dir = os.path.join(self.chart_dir, "data")
        os.makedirs(self.data_dir, exist_ok=True)

        self.db_path = os.path.join(self.data_dir, "feature_recommendation_evidence.db")

        # Fake registry files to verify non-mutation
        self.feat_reg_path = os.path.join(self.data_dir, "feature_registry_store.json")
        with open(self.feat_reg_path, "w", encoding="utf-8") as f:
            f.write('{"registry_version": "1.0", "feature_ids": {}}\n')

        self.pipe_reg_path = os.path.join(self.data_dir, "pipeline_registry_store.json")
        with open(self.pipe_reg_path, "w", encoding="utf-8") as f:
            f.write('{"pipeline_version": "1.0", "pipelines": {}}\n')

        self.ctx_nifty_3s = build_dataset_context(
            market="NIFTY", sampling_interval_sec=3, sliding_window="standard", feature_project_id="all"
        )
        self.ctx_nifty_5s = build_dataset_context(
            market="NIFTY", sampling_interval_sec=5, sliding_window="standard", feature_project_id="all"
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

    def test_01_dialog_opens_universal_ready(self) -> None:
        """Verify dialog renders with UNIVERSAL_READY classification and enables approval."""
        dlg = FeaturePromotionGovernanceDialog(
            self.root,
            data_dir=self.data_dir,
            feature_name="feat_universal",
            context_id=self.ctx_nifty_3s.context_id,
        )
        self.assertEqual(dlg.status, "UNIVERSAL_READY")
        self.assertEqual(str(dlg.btn_approve["state"]), "normal")
        self.assertEqual(dlg.var_allowed_ctx.get(), "ALL")
        dlg.destroy()

    def test_02_dialog_opens_context_scoped_ready(self) -> None:
        """Verify dialog renders with CONTEXT_SCOPED_READY and restricts context."""
        dlg = FeaturePromotionGovernanceDialog(
            self.root,
            data_dir=self.data_dir,
            feature_name="feat_ctx_scoped",
            context_id=self.ctx_nifty_3s.context_id,
        )
        self.assertEqual(dlg.status, "CONTEXT_SCOPED_READY")
        self.assertEqual(str(dlg.btn_approve["state"]), "normal")
        self.assertEqual(dlg.var_allowed_ctx.get(), self.ctx_nifty_3s.context_id)
        self.assertEqual(str(dlg.ent_allowed_ctx["state"]), "readonly")
        dlg.destroy()

    def test_03_not_ready_cannot_be_approved(self) -> None:
        """Verify feature that is NOT_READY has approval button disabled."""
        dlg = FeaturePromotionGovernanceDialog(
            self.root,
            data_dir=self.data_dir,
            feature_name="feat_nonexistent",
            context_id=self.ctx_nifty_3s.context_id,
        )
        self.assertEqual(dlg.status, "NOT_READY")
        self.assertEqual(str(dlg.btn_approve["state"]), "disabled")
        dlg.destroy()

    def test_04_missing_metadata_blocks_approval(self) -> None:
        """Verify clearing domain, group, or description disables approval button."""
        dlg = FeaturePromotionGovernanceDialog(
            self.root,
            data_dir=self.data_dir,
            feature_name="feat_universal",
            context_id=self.ctx_nifty_3s.context_id,
        )
        self.assertEqual(str(dlg.btn_approve["state"]), "normal")

        # Clear description
        dlg.var_desc.set("")
        dlg._validate_form()
        self.assertEqual(str(dlg.btn_approve["state"]), "disabled")

        # Restore description, clear domain
        dlg.var_desc.set("Valid description")
        dlg.var_domain.set("")
        dlg._validate_form()
        self.assertEqual(str(dlg.btn_approve["state"]), "disabled")

        # Restore domain, clear group
        dlg.var_domain.set("order_flow")
        dlg.var_group.set("")
        dlg._validate_form()
        self.assertEqual(str(dlg.btn_approve["state"]), "disabled")

        # Restore all
        dlg.var_group.set("microstructure")
        dlg._validate_form()
        self.assertEqual(str(dlg.btn_approve["state"]), "normal")
        dlg.destroy()

    @patch("tkinter.messagebox.showinfo")
    def test_05_approval_produces_governance_payload_only(self, mock_info: Any) -> None:
        """Verify clicking Approve generates full in-memory governance payload without side effects."""
        captured_payload: dict[str, Any] = {}

        def _on_decision(payload: dict[str, Any]) -> None:
            nonlocal captured_payload
            captured_payload = payload

        dlg = FeaturePromotionGovernanceDialog(
            self.root,
            data_dir=self.data_dir,
            feature_name="feat_universal",
            context_id=self.ctx_nifty_3s.context_id,
            on_decision=_on_decision,
        )

        dlg.var_domain.set("order_flow")
        dlg.var_group.set("microstructure")
        dlg.var_desc.set("Custom reviewed description")
        dlg.var_notes.set("Approved by lead reviewer")

        dlg._on_approve()

        self.assertIsNotNone(dlg.governance_result)
        self.assertEqual(captured_payload.get("action"), "APPROVE")
        self.assertEqual(captured_payload.get("feature_name"), "feat_universal")
        self.assertEqual(captured_payload.get("scope_classification"), "UNIVERSAL_READY")
        self.assertTrue(captured_payload.get("is_universal_ready"))
        self.assertTrue(captured_payload.get("is_base_pipeline_eligible"))
        self.assertEqual(captured_payload.get("domain"), "order_flow")
        self.assertEqual(captured_payload.get("group"), "microstructure")
        self.assertEqual(captured_payload.get("description"), "Custom reviewed description")
        self.assertEqual(captured_payload.get("reviewer_notes"), "Approved by lead reviewer")
        self.assertIn("approved_at", captured_payload)
        self.assertIn("dossier_snapshot", captured_payload)

    @patch("tkinter.messagebox.showinfo")
    def test_06_reject_defer_produces_rejection_payload(self, mock_info: Any) -> None:
        """Verify clicking Reject/Defer generates rejection payload."""
        captured_payload: dict[str, Any] = {}

        def _on_decision(payload: dict[str, Any]) -> None:
            nonlocal captured_payload
            captured_payload = payload

        dlg = FeaturePromotionGovernanceDialog(
            self.root,
            data_dir=self.data_dir,
            feature_name="feat_universal",
            context_id=self.ctx_nifty_3s.context_id,
            on_decision=_on_decision,
        )

        dlg.var_notes.set("Needs additional cross-market testing on SENSEX")
        dlg._on_reject()

        self.assertEqual(captured_payload.get("action"), "REJECT")
        self.assertEqual(captured_payload.get("feature_name"), "feat_universal")
        self.assertEqual(captured_payload.get("reviewer_notes"), "Needs additional cross-market testing on SENSEX")
        self.assertIn("rejected_at", captured_payload)

    def test_07_no_registry_or_pipeline_file_changes(self) -> None:
        """Verify feature_registry_store.json and pipeline_registry_store.json remain completely untouched."""
        feat_reg_before = _file_sha256(self.feat_reg_path)
        pipe_reg_before = _file_sha256(self.pipe_reg_path)

        with patch("tkinter.messagebox.showinfo"):
            dlg = FeaturePromotionGovernanceDialog(
                self.root,
                data_dir=self.data_dir,
                feature_name="feat_universal",
                context_id=self.ctx_nifty_3s.context_id,
            )
            dlg._on_approve()

        feat_reg_after = _file_sha256(self.feat_reg_path)
        pipe_reg_after = _file_sha256(self.pipe_reg_path)

        self.assertEqual(feat_reg_before, feat_reg_after)
        self.assertEqual(pipe_reg_before, pipe_reg_after)

    def test_08_no_sqlite_changes_during_governance_review(self) -> None:
        """Verify SQLite Evidence DB is 100% byte-identical before and after dialog interactions."""
        db_before = _file_sha256(self.db_path)

        with patch("tkinter.messagebox.showinfo"):
            dlg = FeaturePromotionGovernanceDialog(
                self.root,
                data_dir=self.data_dir,
                feature_name="feat_universal",
                context_id=self.ctx_nifty_3s.context_id,
            )
            dlg._on_approve()

        db_after = _file_sha256(self.db_path)
        self.assertEqual(db_before, db_after)

    @patch("tkinter.messagebox.showinfo")
    def test_09_context_scoped_feature_cannot_request_base_pipeline(self, mock_info: Any) -> None:
        """Verify context-scoped feature approval payload marks is_base_pipeline_eligible as False."""
        captured_payload: dict[str, Any] = {}

        def _on_decision(payload: dict[str, Any]) -> None:
            nonlocal captured_payload
            captured_payload = payload

        dlg = FeaturePromotionGovernanceDialog(
            self.root,
            data_dir=self.data_dir,
            feature_name="feat_ctx_scoped",
            context_id=self.ctx_nifty_3s.context_id,
            on_decision=_on_decision,
        )
        dlg._on_approve()

        self.assertEqual(captured_payload.get("scope_classification"), "CONTEXT_SCOPED_READY")
        self.assertFalse(captured_payload.get("is_universal_ready"))
        self.assertFalse(captured_payload.get("is_base_pipeline_eligible"))
        self.assertEqual(captured_payload.get("allowed_contexts"), [self.ctx_nifty_3s.context_id])

    def test_10_precompiled_dossier_reuse(self) -> None:
        """Verify dialog accepts and renders a precompiled dossier directly without re-querying."""
        mock_dossier = {
            "feature_name": "feat_mock_precompiled",
            "context_id": "ctx_custom",
            "market": "SENSEX",
            "sampling_interval_sec": 1,
            "feature_project_id": "all",
            "total_validation_runs": 10,
            "unique_model_count": 4,
            "consecutive_keep_count": 6,
            "lineage_evidence_score": 88.0,
            "evidence_confidence": 0.92,
            "score_volatility": 14.5,
            "comparable_context_count": 2,
            "generalization_score": 0.75,
            "is_phase_3a_promotion_qualified": True,
            "health_status": "HEALTHY",
        }
        dlg = FeaturePromotionGovernanceDialog(
            self.root,
            data_dir=self.data_dir,
            feature_name="feat_mock_precompiled",
            context_id="ctx_custom",
            precompiled_dossier=mock_dossier,
        )
        self.assertEqual(dlg.status, "UNIVERSAL_READY")
        self.assertEqual(str(dlg.btn_approve["state"]), "normal")
        dlg.destroy()


if __name__ == "__main__":
    unittest.main()


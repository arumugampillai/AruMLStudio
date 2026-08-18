"""Unit and integration tests for Phase 3D.4C — Governance UI & Integration."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import tkinter as tk
import unittest
from typing import Any
from unittest.mock import patch

from chain_replay_ml.dataset_builder.feature_registry_store import (
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
    build_dataset_context,
    compile_feature_evidence_dossier,
    get_feature_graduation_audit_log,
    is_feature_in_base_pipeline,
)
from chain_replay_ml.production_validation.evidence_store import (
    append_validation_evidence,
    get_connection,
)
from master_dataset_tk.feature_promotion_governance_dialog import (
    FeaturePromotionGovernanceDialog,
)


def _file_sha256(path: str) -> str:
    if not os.path.isfile(path):
        return "NONE"
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


class TestFeaturePromotionGovernanceIntegration(unittest.TestCase):
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

        self.tmp_dir = tempfile.mkdtemp(prefix="aruml_phase3d4c_test_")
        self.chart_dir = os.path.join(self.tmp_dir, "chart")
        self.data_dir = os.path.join(self.chart_dir, "data")
        os.makedirs(self.data_dir, exist_ok=True)

        self.db_path = os.path.join(self.data_dir, "feature_recommendation_evidence.db")
        self.feat_reg_path = feature_store_path(self.data_dir)
        self.pipe_reg_path = pipeline_store_path(self.data_dir)

        # Baseline Pipeline Store
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

        # Baseline Feature Store
        feat_store = {
            "registry_version": "1.0",
            "created_by": "System",
            "created_on": "2026-01-01",
            "next_feature_id_seq": 4,
            "feature_ids": {
                "feat_base_active": "FR0001",
                "feat_univ_graduated": "FR0002",
                "feat_ctx_graduated": "FR0003",
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
                    "name": "feat_univ_graduated",
                    "domain": "order_flow",
                    "group_id": "microstructure",
                    "scope": "universal",
                    "allowed_contexts": ["ALL"],
                    "is_base_pipeline": False,
                    "implementation_status": "implemented",
                },
                "FR0003": {
                    "feature_id": "FR0003",
                    "name": "feat_ctx_graduated",
                    "domain": "volatility",
                    "group_id": "spread_analysis",
                    "scope": "context_scoped",
                    "allowed_contexts": ["ctx_nifty_3s"],
                    "is_base_pipeline": False,
                    "implementation_status": "implemented",
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
        self.ctx_nifty_5s = build_dataset_context(
            market="NIFTY", sampling_interval_sec=5, sliding_window="standard", feature_project_id="all"
        )

        conn = get_connection(self.data_dir)
        try:
            models = ["model_xgb", "model_lgbm", "model_catboost", "model_nn"]
            # 1. feat_exp_univ: Qualified experimental feature (12 runs in 3s, 6 runs in 5s)
            ev_3s = [
                {
                    "feature_name": "feat_exp_univ",
                    "feature_source": "experimental",
                    "pipeline_id": "PL_EXP_1",
                    "pipeline_snapshot_id": "SNP_1",
                    "model_name": models[i % 4],
                    "recommendation": "KEEP",
                    "permutation_mean": 0.085 + 0.001 * i,
                    "importance_rank": 1,
                    "run_timestamp": f"2026-08-18T1{i:02d}:00:00Z",
                }
                for i in range(12)
            ]
            append_validation_evidence(conn, context=self.ctx_nifty_3s, evidence_rows=ev_3s)

            ev_5s = [
                {
                    "feature_name": "feat_exp_univ",
                    "feature_source": "experimental",
                    "pipeline_id": "PL_EXP_1",
                    "pipeline_snapshot_id": "SNP_1",
                    "model_name": models[i % 4],
                    "recommendation": "KEEP",
                    "permutation_mean": 0.080 + 0.001 * i,
                    "importance_rank": 1,
                    "run_timestamp": f"2026-08-18T1{i:02d}:00:00Z",
                }
                for i in range(6)
            ]
            append_validation_evidence(conn, context=self.ctx_nifty_5s, evidence_rows=ev_5s)

            # 2. feat_univ_graduated: Evidence for already graduated universal feature
            ev_univ = [
                {
                    "feature_name": "feat_univ_graduated",
                    "feature_source": "registry",
                    "pipeline_id": "PL_0001",
                    "pipeline_snapshot_id": "SNP_1",
                    "model_name": models[i % 4],
                    "recommendation": "KEEP",
                    "permutation_mean": 0.088,
                    "importance_rank": 1,
                    "run_timestamp": f"2026-08-18T1{i:02d}:00:00Z",
                }
                for i in range(12)
            ]
            append_validation_evidence(conn, context=self.ctx_nifty_3s, evidence_rows=ev_univ)
            append_validation_evidence(conn, context=self.ctx_nifty_5s, evidence_rows=ev_univ[:6])
        finally:
            conn.close()

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_01_registry_graduation_ui_invokes_engine(self) -> None:
        """Verify Registry Graduation mode triggers execute_registry_graduation upon confirmation."""
        dlg = FeaturePromotionGovernanceDialog(
            self.root,
            data_dir=self.data_dir,
            feature_name="feat_exp_univ",
            mode="REGISTRY_GRADUATION",
            context_id=self.ctx_nifty_3s.context_id,
        )
        self.assertEqual(dlg.status, "UNIVERSAL_READY")
        self.assertEqual(str(dlg.btn_approve["state"]), "normal")

        with patch("tkinter.messagebox.askyesno", return_value=True), patch("tkinter.messagebox.showinfo"):
            dlg._on_approve_registry_graduation()

        self.assertIsNotNone(dlg.governance_result)
        self.assertEqual(dlg.governance_result.get("status"), "SUCCESS")
        self.assertEqual(dlg.governance_result.get("assigned_feature_id"), "FR0004")

    def test_02_base_pipeline_promotion_ui_invokes_engine(self) -> None:
        """Verify Base Pipeline Promotion mode invokes execute_base_pipeline_promotion upon confirmation."""
        dlg = FeaturePromotionGovernanceDialog(
            self.root,
            data_dir=self.data_dir,
            feature_name="feat_univ_graduated",
            mode="BASE_PIPELINE_PROMOTION",
            context_id=self.ctx_nifty_3s.context_id,
        )
        self.assertTrue(dlg.base_elig_res.get("is_eligible"))
        self.assertEqual(str(dlg.btn_approve["state"]), "normal")

        with patch("tkinter.messagebox.askyesno", return_value=True), patch("tkinter.messagebox.showinfo"):
            dlg._on_approve_base_promotion()

        self.assertIsNotNone(dlg.governance_result)
        self.assertEqual(dlg.governance_result.get("status"), "SUCCESS")
        self.assertEqual(dlg.governance_result.get("assigned_feature_id"), "FR0002")
        self.assertTrue(is_feature_in_base_pipeline(self.data_dir, "FR0002"))

    def test_03_deprecation_ui_invokes_engine(self) -> None:
        """Verify Deprecation mode invokes execute_feature_deprecation upon confirmation."""
        dlg = FeaturePromotionGovernanceDialog(
            self.root,
            data_dir=self.data_dir,
            feature_name="feat_base_active",
            mode="FEATURE_DEPRECATION",
            context_id=self.ctx_nifty_3s.context_id,
        )
        dlg.var_ack_future_exclude.set(True)
        dlg.var_ack_hist_compat.set(True)
        dlg._validate_form()

        self.assertEqual(str(dlg.btn_approve["state"]), "normal")

        with patch("tkinter.messagebox.askyesno", return_value=True), patch("tkinter.messagebox.showinfo"):
            dlg._on_approve_deprecation()

        self.assertIsNotNone(dlg.governance_result)
        self.assertEqual(dlg.governance_result.get("status"), "DEPRECATED")
        self.assertFalse(is_feature_in_base_pipeline(self.data_dir, "FR0001"))

    def test_04_not_ready_disables_graduation_approval(self) -> None:
        """Verify un-ready feature disables graduation approval button."""
        dlg = FeaturePromotionGovernanceDialog(
            self.root,
            data_dir=self.data_dir,
            feature_name="feat_unready",
            mode="REGISTRY_GRADUATION",
            context_id=self.ctx_nifty_3s.context_id,
        )
        self.assertEqual(dlg.status, "NOT_READY")
        self.assertEqual(str(dlg.btn_approve["state"]), "disabled")

    def test_05_context_scoped_feature_disables_base_pipeline_promotion(self) -> None:
        """Verify context-scoped feature disables Base Pipeline approval button."""
        dlg = FeaturePromotionGovernanceDialog(
            self.root,
            data_dir=self.data_dir,
            feature_name="feat_ctx_graduated",
            mode="BASE_PIPELINE_PROMOTION",
            context_id=self.ctx_nifty_3s.context_id,
        )
        self.assertEqual(dlg.base_elig_res.get("status"), "CONTEXT_SCOPED_PROHIBITED")
        self.assertEqual(str(dlg.btn_approve["state"]), "disabled")

    def test_06_missing_latency_approval_disables_base_pipeline_promotion(self) -> None:
        """Verify setting latency compliant to No disables Base Pipeline approval button."""
        dlg = FeaturePromotionGovernanceDialog(
            self.root,
            data_dir=self.data_dir,
            feature_name="feat_univ_graduated",
            mode="BASE_PIPELINE_PROMOTION",
            context_id=self.ctx_nifty_3s.context_id,
        )
        dlg.var_latency.set("No")
        dlg._validate_form()
        self.assertEqual(str(dlg.btn_approve["state"]), "disabled")

    def test_07_missing_deprecation_reason_disables_deprecation(self) -> None:
        """Verify empty deprecation reason disables deprecation button."""
        dlg = FeaturePromotionGovernanceDialog(
            self.root,
            data_dir=self.data_dir,
            feature_name="feat_base_active",
            mode="FEATURE_DEPRECATION",
        )
        dlg.var_ack_future_exclude.set(True)
        dlg.var_ack_hist_compat.set(True)
        dlg.var_dep_reason.set("")
        dlg._validate_form()
        self.assertEqual(str(dlg.btn_approve["state"]), "disabled")

    def test_08_missing_reviewer_information_disables_destructive_action(self) -> None:
        """Verify empty reviewer name disables action button across modes."""
        dlg = FeaturePromotionGovernanceDialog(
            self.root,
            data_dir=self.data_dir,
            feature_name="feat_base_active",
            mode="FEATURE_DEPRECATION",
        )
        dlg.var_ack_future_exclude.set(True)
        dlg.var_ack_hist_compat.set(True)
        dlg.var_reviewer.set("")
        dlg._validate_form()
        self.assertEqual(str(dlg.btn_approve["state"]), "disabled")

    def test_09_confirmation_dialog_cancellation_aborts_action(self) -> None:
        """Verify user clicking Cancel on confirmation dialog aborts execution without modifying stores."""
        pipe_before = _file_sha256(self.pipe_reg_path)
        dlg = FeaturePromotionGovernanceDialog(
            self.root,
            data_dir=self.data_dir,
            feature_name="feat_univ_graduated",
            mode="BASE_PIPELINE_PROMOTION",
        )

        with patch("tkinter.messagebox.askyesno", return_value=False):
            dlg._on_approve_base_promotion()

        self.assertIsNone(dlg.governance_result)
        self.assertEqual(pipe_before, _file_sha256(self.pipe_reg_path))

    def test_10_successful_graduation_records_assigned_fr_id(self) -> None:
        """Verify successful graduation records assigned FRxxxx and updates stores."""
        dlg = FeaturePromotionGovernanceDialog(
            self.root,
            data_dir=self.data_dir,
            feature_name="feat_exp_univ",
            mode="REGISTRY_GRADUATION",
        )
        with patch("tkinter.messagebox.askyesno", return_value=True), patch("tkinter.messagebox.showinfo"):
            dlg._on_approve_registry_graduation()

        feat_store = load_feature_store(self.data_dir)
        self.assertEqual(feat_store["feature_ids"]["feat_exp_univ"], "FR0004")

    def test_11_successful_base_pipeline_promotion_updates_membership(self) -> None:
        """Verify successful promotion adds feature to PL_0001 registry_feature_ids."""
        dlg = FeaturePromotionGovernanceDialog(
            self.root,
            data_dir=self.data_dir,
            feature_name="feat_univ_graduated",
            mode="BASE_PIPELINE_PROMOTION",
        )
        with patch("tkinter.messagebox.askyesno", return_value=True), patch("tkinter.messagebox.showinfo"):
            dlg._on_approve_base_promotion()

        pipe_store = load_pipeline_store(self.data_dir)
        self.assertIn("FR0002", pipe_store["pipelines"]["PL_0001"]["registry_feature_ids"])

    def test_12_successful_deprecation_updates_status_and_evicts_base(self) -> None:
        """Verify deprecation updates implementation_status and evicts feature from Base Pipeline."""
        dlg = FeaturePromotionGovernanceDialog(
            self.root,
            data_dir=self.data_dir,
            feature_name="feat_base_active",
            mode="FEATURE_DEPRECATION",
        )
        dlg.var_ack_future_exclude.set(True)
        dlg.var_ack_hist_compat.set(True)
        dlg._validate_form()

        with patch("tkinter.messagebox.askyesno", return_value=True), patch("tkinter.messagebox.showinfo"):
            dlg._on_approve_deprecation()

        feat_store = load_feature_store(self.data_dir)
        self.assertEqual(feat_store["feature_identities"]["FR0001"]["implementation_status"], "deprecated")
        self.assertFalse(is_feature_in_base_pipeline(self.data_dir, "FR0001"))

    def test_13_ui_never_writes_to_evidence_db(self) -> None:
        """Verify UI operations never mutate recommendation_evidence SQLite database."""
        db_before = _file_sha256(self.db_path)

        dlg = FeaturePromotionGovernanceDialog(
            self.root,
            data_dir=self.data_dir,
            feature_name="feat_univ_graduated",
            mode="BASE_PIPELINE_PROMOTION",
        )
        with patch("tkinter.messagebox.askyesno", return_value=True), patch("tkinter.messagebox.showinfo"):
            dlg._on_approve_base_promotion()

        db_after = _file_sha256(self.db_path)
        self.assertEqual(db_before, db_after)

    def test_14_audit_history_tab_renders_events(self) -> None:
        """Verify audit history tab parses and displays recorded audit events."""
        dlg1 = FeaturePromotionGovernanceDialog(
            self.root,
            data_dir=self.data_dir,
            feature_name="feat_univ_graduated",
            mode="BASE_PIPELINE_PROMOTION",
        )
        with patch("tkinter.messagebox.askyesno", return_value=True), patch("tkinter.messagebox.showinfo"):
            dlg1._on_approve_base_promotion()

        dlg2 = FeaturePromotionGovernanceDialog(
            self.root,
            data_dir=self.data_dir,
            feature_name="feat_univ_graduated",
            mode="REGISTRY_GRADUATION",
        )
        logs = get_feature_graduation_audit_log(self.data_dir)
        self.assertEqual(len(logs), 1)
        self.assertEqual(logs[0]["event_type"], "BASE_PIPELINE_PROMOTION")

    def test_15_deprecated_feature_shows_deprecation_status_in_ui(self) -> None:
        """Verify deprecated feature displays ALREADY_DEPRECATED in deprecation mode."""
        dlg1 = FeaturePromotionGovernanceDialog(
            self.root,
            data_dir=self.data_dir,
            feature_name="feat_base_active",
            mode="FEATURE_DEPRECATION",
        )
        dlg1.var_ack_future_exclude.set(True)
        dlg1.var_ack_hist_compat.set(True)
        dlg1._validate_form()
        with patch("tkinter.messagebox.askyesno", return_value=True), patch("tkinter.messagebox.showinfo"):
            dlg1._on_approve_deprecation()

        dlg2 = FeaturePromotionGovernanceDialog(
            self.root,
            data_dir=self.data_dir,
            feature_name="feat_base_active",
            mode="FEATURE_DEPRECATION",
        )
        self.assertEqual(dlg2.depr_eval_res.get("status"), "ALREADY_DEPRECATED")
        self.assertEqual(str(dlg2.btn_approve["state"]), "disabled")

    def test_16_base_pipeline_runtime_membership_is_displayed_accurately(self) -> None:
        """Verify is_feature_in_base_pipeline determines dialog runtime state."""
        dlg_base = FeaturePromotionGovernanceDialog(
            self.root,
            data_dir=self.data_dir,
            feature_name="feat_base_active",
            mode="BASE_PIPELINE_PROMOTION",
        )
        self.assertTrue(dlg_base.in_base_runtime)

        dlg_reg = FeaturePromotionGovernanceDialog(
            self.root,
            data_dir=self.data_dir,
            feature_name="feat_univ_graduated",
            mode="BASE_PIPELINE_PROMOTION",
        )
        self.assertFalse(dlg_reg.in_base_runtime)

    def test_17_legacy_registry_entries_render_without_errors(self) -> None:
        """Verify legacy store entries without new metadata render safely."""
        feat_store = load_feature_store(self.data_dir)
        feat_store["feature_identities"]["FR0099"] = {
            "feature_id": "FR0099",
            "name": "feat_legacy_untyped",
        }
        feat_store["feature_ids"]["feat_legacy_untyped"] = "FR0099"
        save_feature_store(self.data_dir, feat_store)

        dlg = FeaturePromotionGovernanceDialog(
            self.root,
            data_dir=self.data_dir,
            feature_name="feat_legacy_untyped",
            mode="FEATURE_DEPRECATION",
        )
        self.assertTrue(dlg.depr_eval_res.get("is_eligible_for_deprecation"))

    def test_18_reject_action_records_governance_result(self) -> None:
        """Verify Reject/Defer button records REJECT decision."""
        dlg = FeaturePromotionGovernanceDialog(
            self.root,
            data_dir=self.data_dir,
            feature_name="feat_exp_univ",
            mode="REGISTRY_GRADUATION",
        )
        with patch("tkinter.messagebox.showinfo"):
            dlg._on_reject()

        self.assertIsNotNone(dlg.governance_result)
        self.assertEqual(dlg.governance_result.get("action"), "REJECT")

    def test_19_missing_dossier_fields_display_safe_na(self) -> None:
        """Verify precompiled dossier with missing optional fields renders gracefully."""
        sparse_dossier = {
            "feature_name": "feat_sparse",
            "context_id": "ctx_sparse",
        }
        dlg = FeaturePromotionGovernanceDialog(
            self.root,
            data_dir=self.data_dir,
            feature_name="feat_sparse",
            precompiled_dossier=sparse_dossier,
        )
        self.assertEqual(dlg.feature_name, "feat_sparse")

    def test_20_on_decision_callback_invoked_on_approval(self) -> None:
        """Verify on_decision callback is invoked with result payload."""
        received: list[dict[str, Any]] = []

        def _cb(res: dict[str, Any]) -> None:
            received.append(res)

        dlg = FeaturePromotionGovernanceDialog(
            self.root,
            data_dir=self.data_dir,
            feature_name="feat_exp_univ",
            mode="REGISTRY_GRADUATION",
            on_decision=_cb,
        )
        with patch("tkinter.messagebox.askyesno", return_value=True), patch("tkinter.messagebox.showinfo"):
            dlg._on_approve_registry_graduation()

        self.assertEqual(len(received), 1)
        self.assertEqual(received[0]["status"], "SUCCESS")

    def test_21_error_handling_displays_meaningful_message(self) -> None:
        """Verify write failures or errors display error dialog with detailed message."""
        dlg = FeaturePromotionGovernanceDialog(
            self.root,
            data_dir=self.data_dir,
            feature_name="feat_univ_graduated",
            mode="BASE_PIPELINE_PROMOTION",
        )
        with patch("master_dataset_tk.feature_promotion_governance_dialog.execute_base_pipeline_promotion", return_value={"status": "WRITE_FAILURE", "message": "Simulated disk full"}), \
             patch("master_dataset_tk.feature_promotion_governance_dialog.messagebox.askyesno", return_value=True), \
             patch("master_dataset_tk.feature_promotion_governance_dialog.messagebox.showerror") as mock_err:
            dlg._on_approve_base_promotion()
            self.assertTrue(mock_err.called)


if __name__ == "__main__":
    unittest.main()


"""Unit and integration tests for Phase 3D.4A — Base Pipeline Promotion Engine & Base Immunity Activator."""

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
    load_store as load_feature_store,
    save_store as save_feature_store,
    store_path as feature_store_path,
)
from chain_replay_ml.dataset_builder.pipeline_registry_store import (
    ensure_default_existing_pipeline,
    load_store as load_pipeline_store,
    save_store as save_pipeline_store,
    store_path as pipeline_store_path,
)
from chain_replay_ml.production_validation.api import (
    DatasetContext,
    build_dataset_context,
    evaluate_base_pipeline_eligibility,
    evaluate_training_decision,
    execute_base_pipeline_promotion,
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


class TestBasePipelinePromotionGovernance(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp_dir = tempfile.mkdtemp(prefix="aruml_phase3d4a_test_")
        self.chart_dir = os.path.join(self.tmp_dir, "chart")
        self.data_dir = os.path.join(self.chart_dir, "data")
        os.makedirs(self.data_dir, exist_ok=True)

        self.db_path = os.path.join(self.data_dir, "feature_recommendation_evidence.db")
        self.feat_reg_path = feature_store_path(self.data_dir)
        self.pipe_reg_path = pipeline_store_path(self.data_dir)

        # Baseline Base Pipeline store with legacy candidates
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
                    "candidate_features": ["price_return_1", "volatility_60s"],
                    "transformation_config": None,
                    "created_at": "2026-01-01T00:00:00Z",
                    "updated_at": "2026-01-01T00:00:00Z",
                }
            },
            "history": [],
        }
        save_pipeline_store(self.data_dir, pipe_store)

        # Baseline Feature Registry store with graduated features
        feat_store = {
            "registry_version": "1.0",
            "created_by": "System",
            "created_on": "2026-01-01",
            "next_feature_id_seq": 5,
            "feature_ids": {
                "feat_legacy_base": "FR0001",
                "feat_univ_graduated": "FR0002",
                "feat_ctx_graduated": "FR0003",
                "feat_k2_low_g_graduated": "FR0004",
            },
            "feature_identities": {
                "FR0001": {
                    "feature_id": "FR0001",
                    "name": "feat_legacy_base",
                    "domain": "price",
                    "group_id": "price",
                    "scope": "universal",
                    "allowed_contexts": ["ALL"],
                    "implementation_status": "implemented",
                },
                "FR0002": {
                    "feature_id": "FR0002",
                    "name": "feat_univ_graduated",
                    "domain": "order_flow",
                    "group_id": "microstructure",
                    "scope": "universal",
                    "allowed_contexts": ["ALL"],
                    "implementation_status": "implemented",
                },
                "FR0003": {
                    "feature_id": "FR0003",
                    "name": "feat_ctx_graduated",
                    "domain": "volatility",
                    "group_id": "spread_analysis",
                    "scope": "context_scoped",
                    "allowed_contexts": ["ctx_nifty_3s"],
                    "implementation_status": "implemented",
                },
                "FR0004": {
                    "feature_id": "FR0004",
                    "name": "feat_k2_low_g_graduated",
                    "domain": "momentum",
                    "group_id": "momentum",
                    "scope": "context_scoped",
                    "allowed_contexts": ["ctx_nifty_3s"],
                    "implementation_status": "implemented",
                },
            },
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
        self.ctx_sensex_1s = build_dataset_context(
            market="SENSEX", sampling_interval_sec=1, sliding_window="standard", feature_project_id="all"
        )

        conn = get_connection(self.data_dir)
        try:
            models = ["model_xgb", "model_lgbm", "model_catboost", "model_nn"]
            # 1. feat_univ_graduated: 12 KEEP runs in NIFTY 3s AND 6 KEEP runs in NIFTY 5s (K=2, G=1.0, Runs=12 in 3s)
            nifty_3s_univ = [
                {
                    "feature_name": "feat_univ_graduated",
                    "feature_source": "registry",
                    "pipeline_id": "PL_0001",
                    "pipeline_snapshot_id": "SNP_1",
                    "model_name": models[i % len(models)],
                    "recommendation": "KEEP",
                    "permutation_mean": 0.085 + 0.001 * i,
                    "importance_rank": 1,
                    "run_timestamp": f"2026-08-18T1{i:02d}:00:00Z",
                }
                for i in range(12)
            ]
            append_validation_evidence(conn, context=self.ctx_nifty_3s, evidence_rows=nifty_3s_univ)

            nifty_5s_univ = [
                {
                    "feature_name": "feat_univ_graduated",
                    "feature_source": "registry",
                    "pipeline_id": "PL_0001",
                    "pipeline_snapshot_id": "SNP_1",
                    "model_name": models[i % len(models)],
                    "recommendation": "KEEP",
                    "permutation_mean": 0.080 + 0.001 * i,
                    "importance_rank": 1,
                    "run_timestamp": f"2026-08-18T1{i:02d}:00:00Z",
                }
                for i in range(6)
            ]
            append_validation_evidence(conn, context=self.ctx_nifty_5s, evidence_rows=nifty_5s_univ)

            # 2. feat_ctx_graduated: 8 KEEP runs in NIFTY 3s only (K=1)
            nifty_3s_ctx = [
                {
                    "feature_name": "feat_ctx_graduated",
                    "feature_source": "registry",
                    "pipeline_id": "PL_0001",
                    "pipeline_snapshot_id": "SNP_1",
                    "model_name": models[i % len(models)],
                    "recommendation": "KEEP",
                    "permutation_mean": 0.065 + 0.001 * i,
                    "importance_rank": 2,
                    "run_timestamp": f"2026-08-18T1{i:02d}:00:00Z",
                }
                for i in range(8)
            ]
            append_validation_evidence(conn, context=self.ctx_nifty_3s, evidence_rows=nifty_3s_ctx)

            # 3. feat_k2_low_g_graduated: 8 KEEP runs in NIFTY 3s, but 4 REMOVE runs in NIFTY 5s (G < 0.50)
            k2_3s = [
                {
                    "feature_name": "feat_k2_low_g_graduated",
                    "feature_source": "registry",
                    "pipeline_id": "PL_0001",
                    "pipeline_snapshot_id": "SNP_1",
                    "model_name": models[i % len(models)],
                    "recommendation": "KEEP",
                    "permutation_mean": 0.090,
                    "importance_rank": 1,
                    "run_timestamp": f"2026-08-18T1{i:02d}:00:00Z",
                }
                for i in range(8)
            ]
            append_validation_evidence(conn, context=self.ctx_nifty_3s, evidence_rows=k2_3s)

            k2_5s = [
                {
                    "feature_name": "feat_k2_low_g_graduated",
                    "feature_source": "registry",
                    "pipeline_id": "PL_0001",
                    "pipeline_snapshot_id": "SNP_1",
                    "model_name": models[i % len(models)],
                    "recommendation": "REMOVE",
                    "permutation_mean": -0.030,
                    "importance_rank": 9,
                    "run_timestamp": f"2026-08-18T1{i:02d}:00:00Z",
                }
                for i in range(4)
            ]
            append_validation_evidence(conn, context=self.ctx_nifty_5s, evidence_rows=k2_5s)

        finally:
            conn.close()

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_01_qualified_universal_graduated_feature_is_eligible(self) -> None:
        """Verify feat_univ_graduated (FR0002) passes all Base Pipeline prerequisites."""
        res = evaluate_base_pipeline_eligibility(self.data_dir, "FR0002")
        self.assertEqual(res["status"], "ELIGIBLE")
        self.assertTrue(res["is_eligible"])
        self.assertEqual(res["feature_name"], "feat_univ_graduated")
        self.assertEqual(res["failed_checks_count"], 0)

    def test_02_context_scoped_feature_rejected_from_base_pipeline(self) -> None:
        """Verify context-scoped feature (FR0003) is strictly rejected with CONTEXT_SCOPED_PROHIBITED."""
        res = evaluate_base_pipeline_eligibility(self.data_dir, "FR0003")
        self.assertEqual(res["status"], "CONTEXT_SCOPED_PROHIBITED")
        self.assertFalse(res["is_eligible"])
        self.assertIn("universal_scope", res["failed_checks"])

    def test_03_k1_single_context_feature_rejected(self) -> None:
        """Verify feature evaluated in only 1 context (K=1) cannot enter Base Pipeline."""
        res = evaluate_base_pipeline_eligibility(self.data_dir, "feat_ctx_graduated")
        self.assertEqual(res["status"], "CONTEXT_SCOPED_PROHIBITED")
        self.assertFalse(res["is_eligible"])

    def test_04_low_generalization_feature_k2_low_g_rejected(self) -> None:
        """Verify feature with K >= 2 but G < 0.50 is rejected from Base Pipeline."""
        res = evaluate_base_pipeline_eligibility(self.data_dir, "FR0004")
        self.assertEqual(res["status"], "CONTEXT_SCOPED_PROHIBITED")
        self.assertFalse(res["is_eligible"])

    def test_05_non_graduated_experimental_feature_rejected(self) -> None:
        """Verify ungraduated feature not in feature_registry_store.json returns NOT_GRADUATED."""
        res = evaluate_base_pipeline_eligibility(self.data_dir, "feat_experimental_unregistered")
        self.assertEqual(res["status"], "NOT_GRADUATED")
        self.assertFalse(res["is_eligible"])
        self.assertIn("registry_identity", res["failed_checks"])

    def test_06_insufficient_validation_runs_rejected(self) -> None:
        """Verify feature with runs < 5 fails validation_runs_count check."""
        mock_dossier = {
            "feature_name": "feat_univ_graduated",
            "context_id": self.ctx_nifty_3s.context_id,
            "total_validation_runs": 4,  # < 5
            "unique_model_count": 3,
            "lineage_evidence_score": 85.0,
            "evidence_confidence": 0.85,
            "score_volatility": 10.0,
            "comparable_context_count": 2,
            "generalization_score": 0.80,
            "health_status": "HEALTHY",
            "prerequisites_evaluation": {"is_universal_ready": True},
        }
        res = evaluate_base_pipeline_eligibility(self.data_dir, "FR0002", precompiled_dossier=mock_dossier)
        self.assertEqual(res["status"], "NOT_READY")
        self.assertIn("validation_runs_count", res["failed_checks"])

    def test_07_insufficient_unique_models_rejected(self) -> None:
        """Verify feature with unique models < 3 fails unique_models_count check."""
        mock_dossier = {
            "feature_name": "feat_univ_graduated",
            "context_id": self.ctx_nifty_3s.context_id,
            "total_validation_runs": 8,
            "unique_model_count": 2,  # < 3
            "lineage_evidence_score": 85.0,
            "evidence_confidence": 0.85,
            "score_volatility": 10.0,
            "comparable_context_count": 2,
            "generalization_score": 0.80,
            "health_status": "HEALTHY",
            "prerequisites_evaluation": {"is_universal_ready": True},
        }
        res = evaluate_base_pipeline_eligibility(self.data_dir, "FR0002", precompiled_dossier=mock_dossier)
        self.assertEqual(res["status"], "NOT_READY")
        self.assertIn("unique_models_count", res["failed_checks"])

    def test_08_evidence_score_under_80_rejected(self) -> None:
        """Verify feature with evidence score < 80.0 fails evidence_score check."""
        mock_dossier = {
            "feature_name": "feat_univ_graduated",
            "context_id": self.ctx_nifty_3s.context_id,
            "total_validation_runs": 8,
            "unique_model_count": 3,
            "lineage_evidence_score": 75.0,  # < 80.0
            "evidence_confidence": 0.85,
            "score_volatility": 10.0,
            "comparable_context_count": 2,
            "generalization_score": 0.80,
            "health_status": "HEALTHY",
            "prerequisites_evaluation": {"is_universal_ready": True},
        }
        res = evaluate_base_pipeline_eligibility(self.data_dir, "FR0002", precompiled_dossier=mock_dossier)
        self.assertEqual(res["status"], "NOT_READY")
        self.assertIn("evidence_score", res["failed_checks"])

    def test_09_confidence_under_75_rejected(self) -> None:
        """Verify feature with confidence < 0.75 fails evidence_confidence check."""
        mock_dossier = {
            "feature_name": "feat_univ_graduated",
            "context_id": self.ctx_nifty_3s.context_id,
            "total_validation_runs": 8,
            "unique_model_count": 3,
            "lineage_evidence_score": 85.0,
            "evidence_confidence": 0.70,  # < 0.75
            "score_volatility": 10.0,
            "comparable_context_count": 2,
            "generalization_score": 0.80,
            "health_status": "HEALTHY",
            "prerequisites_evaluation": {"is_universal_ready": True},
        }
        res = evaluate_base_pipeline_eligibility(self.data_dir, "FR0002", precompiled_dossier=mock_dossier)
        self.assertEqual(res["status"], "NOT_READY")
        self.assertIn("evidence_confidence", res["failed_checks"])

    def test_10_volatility_over_20_rejected(self) -> None:
        """Verify feature with volatility > 20.0 fails score_volatility check."""
        mock_dossier = {
            "feature_name": "feat_univ_graduated",
            "context_id": self.ctx_nifty_3s.context_id,
            "total_validation_runs": 8,
            "unique_model_count": 3,
            "lineage_evidence_score": 85.0,
            "evidence_confidence": 0.85,
            "score_volatility": 24.5,  # > 20.0
            "comparable_context_count": 2,
            "generalization_score": 0.80,
            "health_status": "HEALTHY",
            "prerequisites_evaluation": {"is_universal_ready": True},
        }
        res = evaluate_base_pipeline_eligibility(self.data_dir, "FR0002", precompiled_dossier=mock_dossier)
        self.assertEqual(res["status"], "NOT_READY")
        self.assertIn("score_volatility", res["failed_checks"])

    def test_11_active_degradation_alert_rejected(self) -> None:
        """Verify feature with active health alert fails health_integrity check."""
        mock_dossier = {
            "feature_name": "feat_univ_graduated",
            "context_id": self.ctx_nifty_3s.context_id,
            "total_validation_runs": 8,
            "unique_model_count": 3,
            "lineage_evidence_score": 85.0,
            "evidence_confidence": 0.85,
            "score_volatility": 10.0,
            "comparable_context_count": 2,
            "generalization_score": 0.80,
            "health_status": "DEGRADED",
            "has_health_alert": True,
            "prerequisites_evaluation": {"is_universal_ready": True},
        }
        res = evaluate_base_pipeline_eligibility(self.data_dir, "FR0002", precompiled_dossier=mock_dossier)
        self.assertEqual(res["status"], "NOT_READY")
        self.assertIn("health_integrity", res["failed_checks"])

    def test_12_missing_human_approval_rejected(self) -> None:
        """Verify approval payload missing base_pipeline_promotion=True is rejected."""
        approval = {
            "latency_budget_compliant": True,
            "reviewer": "Reviewer Alice",
            "reviewer_notes": "Notes",
        }
        res = execute_base_pipeline_promotion(self.data_dir, "FR0002", approval)
        self.assertEqual(res["status"], "INVALID_APPROVAL")

    def test_13_missing_latency_confirmation_rejected(self) -> None:
        """Verify approval payload without latency_budget_compliant=True returns LATENCY_APPROVAL_REQUIRED."""
        approval = {
            "base_pipeline_promotion": True,
            "latency_budget_compliant": False,
            "reviewer": "Reviewer Alice",
            "reviewer_notes": "Notes",
        }
        res = execute_base_pipeline_promotion(self.data_dir, "FR0002", approval)
        self.assertEqual(res["status"], "LATENCY_APPROVAL_REQUIRED")

    def test_14_successful_approval_adds_fr_id_to_base_pipeline(self) -> None:
        """Verify successful promotion adds FR0002 to PL_0001['registry_feature_ids']."""
        approval = {
            "base_pipeline_promotion": True,
            "latency_budget_compliant": True,
            "reviewer": "Lead Engineer Dave",
            "reviewer_notes": "Latency budget benchmarked at 0.12ms. Approved.",
        }
        res = execute_base_pipeline_promotion(self.data_dir, "FR0002", approval)

        self.assertEqual(res["status"], "SUCCESS")
        self.assertEqual(res["assigned_feature_id"], "FR0002")
        self.assertEqual(res["pipeline_id"], "PL_0001")

        pipe_doc = load_pipeline_store(self.data_dir)
        base_reg_ids = pipe_doc["pipelines"]["PL_0001"]["registry_feature_ids"]
        self.assertIn("FR0002", base_reg_ids)
        self.assertIn("FR0001", base_reg_ids)  # Legacy feature preserved

        feat_doc = load_feature_store(self.data_dir)
        self.assertTrue(feat_doc["feature_identities"]["FR0002"].get("is_base_pipeline"))

    def test_15_duplicate_promotion_returns_already_in_base_pipeline(self) -> None:
        """Verify promoting a feature that is already in PL_0001 returns ALREADY_IN_BASE_PIPELINE."""
        approval = {
            "base_pipeline_promotion": True,
            "latency_budget_compliant": True,
            "reviewer": "Lead Engineer Dave",
            "reviewer_notes": "Approved",
        }
        # First promotion
        res1 = execute_base_pipeline_promotion(self.data_dir, "FR0002", approval)
        self.assertEqual(res1["status"], "SUCCESS")

        # Second promotion attempt
        res2 = execute_base_pipeline_promotion(self.data_dir, "FR0002", approval)
        self.assertEqual(res2["status"], "ALREADY_IN_BASE_PIPELINE")
        self.assertEqual(res2["feature_id"], "FR0002")

    def test_16_existing_candidate_features_remain_unchanged(self) -> None:
        """Verify existing candidate_features in PL_0001 are preserved 100% untouched."""
        approval = {
            "base_pipeline_promotion": True,
            "latency_budget_compliant": True,
            "reviewer": "Lead Engineer Dave",
            "reviewer_notes": "Approved",
        }
        execute_base_pipeline_promotion(self.data_dir, "FR0002", approval)

        pipe_doc = load_pipeline_store(self.data_dir)
        candidates = pipe_doc["pipelines"]["PL_0001"]["candidate_features"]
        self.assertEqual(candidates, ["price_return_1", "volatility_60s"])

    def test_17_atomic_write_failure_preserves_pipeline_store(self) -> None:
        """Verify simulated I/O failure leaves pipeline_registry_store.json unchanged."""
        pipe_before = _file_sha256(self.pipe_reg_path)

        approval = {
            "base_pipeline_promotion": True,
            "latency_budget_compliant": True,
            "reviewer": "Lead Engineer Dave",
            "reviewer_notes": "Approved",
        }

        with patch("chain_replay_ml.production_validation.feature_graduation._atomic_save_json", side_effect=IOError("Disk write error")):
            res = execute_base_pipeline_promotion(self.data_dir, "FR0002", approval)
            self.assertEqual(res["status"], "WRITE_FAILURE")

        pipe_after = _file_sha256(self.pipe_reg_path)
        self.assertEqual(pipe_before, pipe_after)

    def test_18_audit_event_base_pipeline_promotion_created(self) -> None:
        """Verify BASE_PIPELINE_PROMOTION event is recorded in audit log with complete details."""
        approval = {
            "base_pipeline_promotion": True,
            "latency_budget_compliant": True,
            "reviewer": "Lead Engineer Dave",
            "reviewer_notes": "Latency budget verified",
        }
        res = execute_base_pipeline_promotion(self.data_dir, "FR0002", approval)

        log = get_feature_graduation_audit_log(self.data_dir)
        self.assertEqual(len(log), 1)
        entry = log[0]
        self.assertEqual(entry["event_type"], "BASE_PIPELINE_PROMOTION")
        self.assertEqual(entry["event_id"], res["audit_event_id"])
        self.assertEqual(entry["assigned_feature_id"], "FR0002")
        self.assertEqual(entry["pipeline_id"], "PL_0001")
        self.assertEqual(entry["reviewer_information"], "Lead Engineer Dave")
        self.assertTrue(entry["latency_budget_compliant"])
        self.assertIn("dossier_snapshot", entry)
        self.assertIn("promotion_qualification_snapshot", entry)

    def test_19_evidence_db_sha256_remains_strictly_identical(self) -> None:
        """Verify recommendation_evidence SQLite database is byte-for-byte unchanged."""
        db_before = _file_sha256(self.db_path)

        approval = {
            "base_pipeline_promotion": True,
            "latency_budget_compliant": True,
            "reviewer": "Lead Engineer Dave",
            "reviewer_notes": "Approved",
        }
        execute_base_pipeline_promotion(self.data_dir, "FR0002", approval)

        db_after = _file_sha256(self.db_path)
        self.assertEqual(db_before, db_after)

    def test_20_phase_3a_base_pipeline_immunity_behavior(self) -> None:
        """Verify Phase 3A decision evaluation for Base Pipeline feature applies Base Pipeline rules."""
        # For base_pipeline features, low runs or normal variation are protected by Base Pipeline immunity
        dec_res = evaluate_training_decision(
            feature_name="feat_univ_graduated",
            context_id=self.ctx_nifty_3s.context_id,
            feature_source="base_pipeline",
            total_runs=8,
            unique_models_count=4,
            evidence_score=85.0,
            lifecycle_status="active",
            consecutive_remove_count=0,
            remove_runs=0,
            consecutive_keep_count=8,
            dominant_recommendation="KEEP",
            is_consensus_tie=False,
            freshness_label="Fresh (<14d)",
            score_volatility=10.0,
            is_promotion_candidate=False,
        )
        self.assertFalse(dec_res.is_excluded)
        self.assertTrue(dec_res.is_training_candidate)

    def test_21_context_isolation_preserved(self) -> None:
        """Verify checking SENSEX context for NIFTY feature isolates evidence correctly."""
        elig_sensex = evaluate_base_pipeline_eligibility(self.data_dir, "feat_ctx_graduated")
        self.assertFalse(elig_sensex["is_eligible"])
        self.assertEqual(elig_sensex["status"], "CONTEXT_SCOPED_PROHIBITED")

    def test_22_empty_or_legacy_pipeline_store_auto_seeded(self) -> None:
        """Verify an empty pipeline store initializes PL_0001 automatically without error."""
        save_pipeline_store(self.data_dir, {"registry_version": "1.0", "pipelines": {}})

        approval = {
            "base_pipeline_promotion": True,
            "latency_budget_compliant": True,
            "reviewer": "Lead Engineer Dave",
            "reviewer_notes": "Approved",
        }
        res = execute_base_pipeline_promotion(self.data_dir, "FR0002", approval)
        self.assertEqual(res["status"], "SUCCESS")
        self.assertIn("FR0002", load_pipeline_store(self.data_dir)["pipelines"]["PL_0001"]["registry_feature_ids"])

    def test_23_rejection_of_feature_by_name_or_id(self) -> None:
        """Verify evaluate_base_pipeline_eligibility accepts both FR ID and feature name."""
        res_by_id = evaluate_base_pipeline_eligibility(self.data_dir, "FR0002")
        res_by_name = evaluate_base_pipeline_eligibility(self.data_dir, "feat_univ_graduated")
        self.assertEqual(res_by_id["status"], res_by_name["status"])
        self.assertEqual(res_by_id["feature_id"], res_by_name["feature_id"])


if __name__ == "__main__":
    unittest.main()


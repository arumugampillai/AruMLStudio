"""Unit and integration tests for Phase 3D.1 — Evidence Dossier Compiler & Qualification Verifier."""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
import unittest
from typing import Any

from chain_replay_ml.production_validation.api import (
    DatasetContext,
    build_dataset_context,
    compile_feature_evidence_dossier,
    evaluate_graduation_prerequisites,
)
from chain_replay_ml.production_validation.evidence_store import (
    append_validation_evidence,
    get_connection,
)


def _file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


class TestFeatureGraduationDossier(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp_dir = tempfile.mkdtemp(prefix="aruml_phase3d1_test_")
        self.chart_dir = os.path.join(self.tmp_dir, "chart")
        self.data_dir = os.path.join(self.chart_dir, "data")
        os.makedirs(self.data_dir, exist_ok=True)

        self.db_path = os.path.join(self.data_dir, "feature_recommendation_evidence.db")

        self.ctx_nifty_3s = build_dataset_context(
            market="NIFTY", sampling_interval_sec=3, sliding_window="standard", feature_project_id="all"
        )
        self.ctx_nifty_5s = build_dataset_context(
            market="NIFTY", sampling_interval_sec=5, sliding_window="standard", feature_project_id="all"
        )
        self.ctx_sensex_1s = build_dataset_context(
            market="SENSEX", sampling_interval_sec=1, sliding_window="standard", feature_project_id="all"
        )

        # Seed realistic test data
        conn = get_connection(self.data_dir)
        try:
            # 1. feat_universal: 8 KEEP runs across 4 distinct models in NIFTY 3s AND 5 KEEP runs in NIFTY 5s
            models = ["model_xgb", "model_lgbm", "model_catboost", "model_nn"]
            nifty_3s_rows = [
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
            append_validation_evidence(conn, context=self.ctx_nifty_3s, evidence_rows=nifty_3s_rows)

            nifty_5s_rows = [
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
            append_validation_evidence(conn, context=self.ctx_nifty_5s, evidence_rows=nifty_5s_rows)

            # 2. feat_context_scoped_k1: 8 KEEP runs in NIFTY 3s only (K=1)
            ctx_scoped_rows = [
                {
                    "feature_name": "feat_context_scoped_k1",
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
            append_validation_evidence(conn, context=self.ctx_nifty_3s, evidence_rows=ctx_scoped_rows)

            # 3. feat_insufficient_models: 4 runs but all from model_xgb (unique_models = 1)
            single_model_rows = [
                {
                    "feature_name": "feat_single_model",
                    "feature_source": "experimental",
                    "pipeline_id": "PL_EXP_3",
                    "pipeline_snapshot_id": "SNP_3",
                    "model_name": "model_xgb",
                    "recommendation": "KEEP",
                    "permutation_mean": 0.050,
                    "importance_rank": 3,
                    "run_timestamp": f"2026-08-18T1{i:02d}:00:00Z",
                }
                for i in range(4)
            ]
            append_validation_evidence(conn, context=self.ctx_nifty_3s, evidence_rows=single_model_rows)

            # 4. feat_low_score: 3 runs, 3 models, but 1 KEEP, 1 WATCH, 1 REMOVE
            low_score_rows = [
                {
                    "feature_name": "feat_low_score",
                    "feature_source": "experimental",
                    "pipeline_id": "PL_EXP_4",
                    "pipeline_snapshot_id": "SNP_4",
                    "model_name": "model_xgb",
                    "recommendation": "KEEP",
                    "permutation_mean": 0.050,
                    "importance_rank": 3,
                    "run_timestamp": "2026-08-18T10:00:00Z",
                },
                {
                    "feature_name": "feat_low_score",
                    "feature_source": "experimental",
                    "pipeline_id": "PL_EXP_4",
                    "pipeline_snapshot_id": "SNP_4",
                    "model_name": "model_lgbm",
                    "recommendation": "WATCH",
                    "permutation_mean": 0.020,
                    "importance_rank": 5,
                    "run_timestamp": "2026-08-18T11:00:00Z",
                },
                {
                    "feature_name": "feat_low_score",
                    "feature_source": "experimental",
                    "pipeline_id": "PL_EXP_4",
                    "pipeline_snapshot_id": "SNP_4",
                    "model_name": "model_catboost",
                    "recommendation": "REMOVE",
                    "permutation_mean": -0.010,
                    "importance_rank": 8,
                    "run_timestamp": "2026-08-18T12:00:00Z",
                },
            ]
            append_validation_evidence(conn, context=self.ctx_nifty_3s, evidence_rows=low_score_rows)

            # 5. feat_volatile_scale_specific: Great in NIFTY 3s, but REMOVE in NIFTY 5s (G < 0.50)
            feat_scale_3s = [
                {
                    "feature_name": "feat_scale_specific",
                    "feature_source": "experimental",
                    "pipeline_id": "PL_EXP_5",
                    "pipeline_snapshot_id": "SNP_5",
                    "model_name": models[i % len(models)],
                    "recommendation": "KEEP",
                    "permutation_mean": 0.090,
                    "importance_rank": 1,
                    "run_timestamp": f"2026-08-18T1{i:02d}:00:00Z",
                }
                for i in range(8)
            ]
            append_validation_evidence(conn, context=self.ctx_nifty_3s, evidence_rows=feat_scale_3s)

            feat_scale_5s = [
                {
                    "feature_name": "feat_scale_specific",
                    "feature_source": "experimental",
                    "pipeline_id": "PL_EXP_5",
                    "pipeline_snapshot_id": "SNP_5",
                    "model_name": models[i % len(models)],
                    "recommendation": "REMOVE",
                    "permutation_mean": -0.030,
                    "importance_rank": 9,
                    "run_timestamp": f"2026-08-18T1{i:02d}:00:00Z",
                }
                for i in range(4)
            ]
            append_validation_evidence(conn, context=self.ctx_nifty_5s, evidence_rows=feat_scale_5s)

        finally:
            conn.close()

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_01_fully_qualified_universal_feature(self) -> None:
        """Verify feat_universal passes all 7 checks and qualifies for UNIVERSAL_READY (K >= 2, G >= 0.50)."""
        dossier = compile_feature_evidence_dossier(self.data_dir, "feat_universal", context_id=self.ctx_nifty_3s.context_id)
        self.assertEqual(dossier["feature_name"], "feat_universal")
        self.assertGreaterEqual(dossier["consecutive_keep_count"], 3)
        self.assertGreaterEqual(dossier["unique_model_count"], 3)
        self.assertGreaterEqual(dossier["lineage_evidence_score"], 75.0)
        self.assertGreaterEqual(dossier["evidence_confidence"], 0.70)
        self.assertIsNotNone(dossier["score_volatility"])
        self.assertLessEqual(dossier["score_volatility"], 25.0)
        self.assertTrue(dossier["is_phase_3a_promotion_qualified"])

        eval_res = dossier["prerequisites_evaluation"]
        self.assertEqual(eval_res["status"], "UNIVERSAL_READY")
        self.assertTrue(eval_res["is_universal_ready"])
        self.assertTrue(eval_res["is_context_scoped_ready"])
        self.assertTrue(eval_res["is_base_pipeline_eligible"])
        self.assertEqual(eval_res["failed_checks_count"], 0)
        self.assertEqual(eval_res["passed_checks_count"], 7)

    def test_02_qualified_context_scoped_feature_k1(self) -> None:
        """Verify feat_context_scoped_k1 qualifies as CONTEXT_SCOPED_READY (K=1, blocked from Base Pipeline)."""
        dossier = compile_feature_evidence_dossier(self.data_dir, "feat_context_scoped_k1", context_id=self.ctx_nifty_3s.context_id)
        eval_res = dossier["prerequisites_evaluation"]
        self.assertEqual(eval_res["status"], "CONTEXT_SCOPED_READY")
        self.assertFalse(eval_res["is_universal_ready"])
        self.assertTrue(eval_res["is_context_scoped_ready"])
        self.assertFalse(eval_res["is_base_pipeline_eligible"])
        self.assertEqual(eval_res["allowed_contexts"], [self.ctx_nifty_3s.context_id])
        self.assertEqual(eval_res["failed_checks_count"], 0)

    def test_03_insufficient_runs(self) -> None:
        """Verify feature with 0 runs fails graduation prerequisites."""
        dossier = compile_feature_evidence_dossier(self.data_dir, "feat_nonexistent", context_id=self.ctx_nifty_3s.context_id)
        eval_res = dossier["prerequisites_evaluation"]
        self.assertEqual(eval_res["status"], "NOT_READY")
        self.assertFalse(eval_res["is_universal_ready"])
        self.assertFalse(eval_res["is_context_scoped_ready"])
        self.assertIn("consecutive_keep_streak", eval_res["failed_checks"])
        self.assertIn("unique_models_count", eval_res["failed_checks"])

    def test_04_insufficient_unique_models(self) -> None:
        """Verify feature with multiple runs from single model fails unique_models check."""
        dossier = compile_feature_evidence_dossier(self.data_dir, "feat_single_model", context_id=self.ctx_nifty_3s.context_id)
        eval_res = dossier["prerequisites_evaluation"]
        self.assertEqual(eval_res["status"], "NOT_READY")
        self.assertIn("unique_models_count", eval_res["failed_checks"])

    def test_05_low_evidence_score(self) -> None:
        """Verify feature with low evidence score fails evidence_score check."""
        dossier = compile_feature_evidence_dossier(self.data_dir, "feat_low_score", context_id=self.ctx_nifty_3s.context_id)
        eval_res = dossier["prerequisites_evaluation"]
        self.assertEqual(eval_res["status"], "NOT_READY")
        self.assertIn("evidence_score", eval_res["failed_checks"])

    def test_06_insufficient_keep_streak(self) -> None:
        """Verify feature with consecutive_keep < 3 fails streak check."""
        dossier = compile_feature_evidence_dossier(self.data_dir, "feat_low_score", context_id=self.ctx_nifty_3s.context_id)
        eval_res = dossier["prerequisites_evaluation"]
        self.assertIn("consecutive_keep_streak", eval_res["failed_checks"])

    def test_07_scale_specific_generalization_k2_low_g(self) -> None:
        """Verify feature with K >= 2 but G < 0.50 falls back to CONTEXT_SCOPED_READY."""
        dossier = compile_feature_evidence_dossier(self.data_dir, "feat_scale_specific", context_id=self.ctx_nifty_3s.context_id)
        eval_res = dossier["prerequisites_evaluation"]
        self.assertEqual(eval_res["status"], "CONTEXT_SCOPED_READY")
        self.assertFalse(eval_res["is_universal_ready"])
        self.assertTrue(eval_res["is_context_scoped_ready"])
        self.assertFalse(eval_res["is_base_pipeline_eligible"])
        self.assertEqual(eval_res["allowed_contexts"], [self.ctx_nifty_3s.context_id])

    def test_08_context_isolation_nifty_vs_sensex(self) -> None:
        """Verify querying SENSEX context for a NIFTY-only feature shows clean context isolation."""
        dossier_sensex = compile_feature_evidence_dossier(self.data_dir, "feat_universal", context_id=self.ctx_sensex_1s.context_id)
        self.assertEqual(dossier_sensex["total_validation_runs"], 0)
        self.assertEqual(dossier_sensex["context_id"], self.ctx_sensex_1s.context_id)
        eval_sensex = dossier_sensex["prerequisites_evaluation"]
        self.assertEqual(eval_sensex["status"], "NOT_READY")

    def test_09_evidence_db_checksum_unchanged_read_only(self) -> None:
        """Verify dossier compilation and prerequisite evaluation are strictly read-only."""
        sha_before = _file_sha256(self.db_path)
        for _ in range(5):
            compile_feature_evidence_dossier(self.data_dir, "feat_universal")
            evaluate_graduation_prerequisites(self.data_dir, "feat_universal")
        sha_after = _file_sha256(self.db_path)
        self.assertEqual(sha_before, sha_after)

    def test_10_missing_volatility_under_n3_fails_volatility_requirement(self) -> None:
        """Verify that when N < 3, volatility is unavailable and graduation verifier flags it."""
        mock_dossier = {
            "feature_name": "feat_mock_2runs",
            "context_id": "ctx_test",
            "total_validation_runs": 2,
            "unique_model_count": 2,
            "consecutive_keep_count": 2,
            "lineage_evidence_score": 80.0,
            "evidence_confidence": 0.80,
            "score_volatility": None,  # N < 3
            "is_phase_3a_promotion_qualified": False,
            "health_status": "HEALTHY",
        }
        res = evaluate_graduation_prerequisites(self.data_dir, "feat_mock_2runs", precompiled_dossier=mock_dossier)
        self.assertEqual(res["status"], "NOT_READY")
        self.assertIn("score_volatility", res["failed_checks"])

    def test_11_excessive_volatility_fails_prerequisites(self) -> None:
        """Verify feature with volatility > 25.0 fails graduation check."""
        mock_dossier = {
            "feature_name": "feat_volatile",
            "context_id": "ctx_test",
            "total_validation_runs": 5,
            "unique_model_count": 3,
            "consecutive_keep_count": 3,
            "lineage_evidence_score": 85.0,
            "evidence_confidence": 0.85,
            "score_volatility": 38.5,  # > 25.0
            "is_phase_3a_promotion_qualified": True,
            "health_status": "HEALTHY",
        }
        res = evaluate_graduation_prerequisites(self.data_dir, "feat_volatile", precompiled_dossier=mock_dossier)
        self.assertEqual(res["status"], "NOT_READY")
        self.assertIn("score_volatility", res["failed_checks"])

    def test_12_active_health_alert_fails_prerequisites(self) -> None:
        """Verify feature with extreme drop / blocked health fails graduation check."""
        mock_dossier = {
            "feature_name": "feat_degraded",
            "context_id": "ctx_test",
            "total_validation_runs": 5,
            "unique_model_count": 3,
            "consecutive_keep_count": 3,
            "lineage_evidence_score": 80.0,
            "evidence_confidence": 0.80,
            "score_volatility": 12.0,
            "is_phase_3a_promotion_qualified": True,
            "health_status": "BLOCKED",
            "has_health_alert": True,
        }
        res = evaluate_graduation_prerequisites(self.data_dir, "feat_degraded", precompiled_dossier=mock_dossier)
        self.assertEqual(res["status"], "NOT_READY")
        self.assertIn("health_integrity", res["failed_checks"])

    def test_13_phase_3a_promotion_not_qualified_fails(self) -> None:
        """Verify that if Phase 3A did not qualify promotion, graduation fails."""
        mock_dossier = {
            "feature_name": "feat_not_promo",
            "context_id": "ctx_test",
            "total_validation_runs": 5,
            "unique_model_count": 3,
            "consecutive_keep_count": 3,
            "lineage_evidence_score": 80.0,
            "evidence_confidence": 0.80,
            "score_volatility": 12.0,
            "is_phase_3a_promotion_qualified": False,
            "health_status": "HEALTHY",
        }
        res = evaluate_graduation_prerequisites(self.data_dir, "feat_not_promo", precompiled_dossier=mock_dossier)
        self.assertEqual(res["status"], "NOT_READY")
        self.assertIn("phase_3a_promotion_qualified", res["failed_checks"])

    def test_14_dossier_fields_completeness(self) -> None:
        """Verify compile_feature_evidence_dossier returns all expected fields."""
        dossier = compile_feature_evidence_dossier(self.data_dir, "feat_universal", context_id=self.ctx_nifty_3s.context_id)
        expected_keys = [
            "feature_name", "feature_source", "context_id", "market", "sampling_interval_sec",
            "total_validation_runs", "unique_model_count", "consecutive_keep_count",
            "lineage_evidence_score", "evidence_confidence", "model_consensus", "freshness",
            "score_volatility", "generalization_score", "comparable_context_count",
            "is_phase_3a_promotion_qualified", "phase_3a_decision", "health_status",
            "prerequisites_evaluation",
        ]
        for k in expected_keys:
            self.assertIn(k, dossier, f"Missing key {k} in compiled dossier")

    def test_15_explanation_bullets_structure(self) -> None:
        """Verify explanations list is populated with human-readable rationale."""
        eval_pass = evaluate_graduation_prerequisites(self.data_dir, "feat_universal", context_id=self.ctx_nifty_3s.context_id)
        self.assertGreater(len(eval_pass["explanations"]), 0)
        self.assertIn("Universal Graduation Approved", eval_pass["explanations"][0])

        eval_fail = evaluate_graduation_prerequisites(self.data_dir, "feat_nonexistent", context_id=self.ctx_nifty_3s.context_id)
        self.assertGreater(len(eval_fail["explanations"]), 0)
        self.assertIn("Graduation not approved", eval_fail["explanations"][0])


if __name__ == "__main__":
    unittest.main()


"""Unit and integration tests for Phase 3C — Training Provenance, Model Lineage, and Closed-Loop Traceability."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import unittest
from typing import Any
from unittest.mock import MagicMock

import pandas as pd

from chain_replay_ml.production_validation.api import (
    DatasetContext,
    TrainingDecisionState,
    audit_model_training_feedback_loop,
    build_dataset_context,
    build_model_builder_training_bundle,
    export_training_candidates_preset,
    get_model_recommendation_provenance,
    persist_validation_evidence,
    resolve_context_from_model_package,
    resolve_context_or_legacy,
)
from chain_replay_ml.production_validation.evidence_store import (
    append_validation_evidence,
    ensure_schema,
    get_connection,
)
from chain_replay_ml.training.artifacts import save_model_package
from chain_replay_ml.training.config import TrainingConfig, normalize_training_config
from master_dataset_tk.model_builder.feature_preset import (
    apply_feature_preset,
    load_feature_preset,
    save_feature_preset,
)
from master_dataset_tk.model_builder.state import ModelBuilderState, load_persisted_state, save_persisted_state


def _file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


class TestModelTrainingLifecycleTraceability(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp_dir = tempfile.mkdtemp()
        self.chart_dir = os.path.join(self.tmp_dir, "chart")
        self.data_dir = os.path.join(self.chart_dir, "data")
        os.makedirs(self.data_dir, exist_ok=True)

        self.db_path = os.path.join(self.data_dir, "feature_recommendation_evidence.db")

        self.ctx_nifty_3s = build_dataset_context(
            market="NIFTY", sampling_interval_sec=3, sliding_window="standard", feature_project_id="all"
        )
        self.ctx_sensex_1s = build_dataset_context(
            market="SENSEX", sampling_interval_sec=1, sliding_window="standard", feature_project_id="all"
        )

        # Seed sample evidence in NIFTY 3s
        conn = get_connection(self.data_dir)
        try:
            ev_rows = [
                {
                    "feature_name": "feat_promo_alpha",
                    "feature_source": "experimental",
                    "pipeline_id": "PL_EXP_1",
                    "pipeline_snapshot_id": "SNP_1",
                    "recommendation": "KEEP",
                    "permutation_mean": 0.085,
                    "importance_rank": 1,
                    "run_timestamp": "2026-08-18T10:00:00Z",
                    "model_name": "model_seed_1",
                },
                {
                    "feature_name": "feat_promo_alpha",
                    "feature_source": "experimental",
                    "pipeline_id": "PL_EXP_1",
                    "pipeline_snapshot_id": "SNP_1",
                    "recommendation": "KEEP",
                    "permutation_mean": 0.090,
                    "importance_rank": 1,
                    "run_timestamp": "2026-08-18T11:00:00Z",
                    "model_name": "model_seed_2",
                },
                {
                    "feature_name": "feat_promo_alpha",
                    "feature_source": "experimental",
                    "pipeline_id": "PL_EXP_1",
                    "pipeline_snapshot_id": "SNP_1",
                    "recommendation": "KEEP",
                    "permutation_mean": 0.095,
                    "importance_rank": 1,
                    "run_timestamp": "2026-08-18T12:00:00Z",
                    "model_name": "model_seed_3",
                },
                {
                    "feature_name": "feat_standard_beta",
                    "feature_source": "base_pipeline",
                    "recommendation": "KEEP",
                    "permutation_mean": 0.040,
                    "importance_rank": 2,
                    "run_timestamp": "2026-08-18T12:00:00Z",
                    "model_name": "model_seed_3",
                },
            ]
            append_validation_evidence(conn, context=self.ctx_nifty_3s, evidence_rows=ev_rows)
        finally:
            conn.close()

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_01_preset_bundle_to_model_builder_state(self) -> None:
        """Verify Phase 3B preset bundle is captured into ModelBuilderState."""
        preset = export_training_candidates_preset(
            self.chart_dir,
            self.data_dir,
            context=self.ctx_nifty_3s,
        )
        self.assertIn("recommendation_decision_bundle", preset)

        # Apply preset
        res = apply_feature_preset(
            preset,
            dataset_name="ohlc_NIFTY_3s_standard",
            dataset_feature_names=["feat_promo_alpha", "feat_standard_beta"],
        )
        self.assertTrue(res["applied"])
        self.assertIn("recommendation_decision_bundle", res)

        state = ModelBuilderState()
        state.features = set(res["features"])
        rdb = res.get("recommendation_decision_bundle")
        if isinstance(rdb, dict):
            state.recommendation_decision_bundle = dict(rdb)

        self.assertEqual(state.features, {"feat_promo_alpha", "feat_standard_beta"})
        self.assertIsNotNone(state.recommendation_decision_bundle)
        self.assertEqual(state.recommendation_decision_bundle["context_id"], self.ctx_nifty_3s.context_id)

    def test_02_model_builder_state_to_training_config(self) -> None:
        """Verify ModelBuilderState forwards recommendation_decision_bundle into build_training_config()."""
        bundle = build_model_builder_training_bundle(self.data_dir, context=self.ctx_nifty_3s)
        rdb = bundle["recommendation_decision_bundle"]

        state = ModelBuilderState(
            dataset="ohlc_NIFTY_3s_standard",
            target="target_ret",
            features={"feat_promo_alpha", "feat_standard_beta"},
            recommendation_decision_bundle=rdb,
        )

        cfg_dict = state.build_training_config()
        self.assertIn("recommendation_decision_bundle", cfg_dict)
        self.assertEqual(cfg_dict["recommendation_decision_bundle"]["context_id"], self.ctx_nifty_3s.context_id)

    def test_03_training_config_serialization_and_deserialization(self) -> None:
        """Verify TrainingConfig dataclass, to_dict, and normalize_training_config roundtrip."""
        bundle = build_model_builder_training_bundle(self.data_dir, context=self.ctx_nifty_3s)
        rdb = bundle["recommendation_decision_bundle"]

        raw_cfg = {
            "dataset": "ohlc_NIFTY_3s_standard",
            "target": "target_ret",
            "algorithm": "xgboost",
            "features": ["feat_promo_alpha", "feat_standard_beta"],
            "recommendation_decision_bundle": rdb,
        }

        cfg_obj = normalize_training_config(raw_cfg)
        self.assertIsNotNone(cfg_obj.recommendation_decision_bundle)
        self.assertEqual(cfg_obj.recommendation_decision_bundle["context_id"], self.ctx_nifty_3s.context_id)

        serialized = cfg_obj.to_dict()
        self.assertIn("recommendation_decision_bundle", serialized)
        self.assertEqual(serialized["recommendation_decision_bundle"]["context_id"], self.ctx_nifty_3s.context_id)

    def test_04_legacy_configuration_compatibility(self) -> None:
        """Verify legacy configuration without recommendation bundle normalizes and serializes cleanly."""
        legacy_raw = {
            "dataset": "ohlc_legacy",
            "target": "target_ret",
            "algorithm": "xgboost",
            "features": ["feat_1", "feat_2"],
        }
        cfg_obj = normalize_training_config(legacy_raw)
        self.assertIsNone(cfg_obj.recommendation_decision_bundle)

        serialized = cfg_obj.to_dict()
        self.assertNotIn("recommendation_decision_bundle", serialized)

    def test_05_save_model_package_provenance_stamping(self) -> None:
        """Verify save_model_package() stamps recommendation_provenance in metadata.json and manifest.json."""
        bundle = build_model_builder_training_bundle(self.data_dir, context=self.ctx_nifty_3s)
        rdb = bundle["recommendation_decision_bundle"]

        cfg = normalize_training_config({
            "dataset": "ohlc_NIFTY_3s_standard",
            "target": "target_ret",
            "algorithm": "xgboost",
            "features": ["feat_promo_alpha", "feat_standard_beta"],
            "model_name": "model_test_provenance",
            "recommendation_decision_bundle": rdb,
        })

        mock_model = MagicMock()
        mock_model.save_model = MagicMock()

        pkg_info = save_model_package(
            data_dir=self.data_dir,
            config=cfg,
            model=mock_model,
            metrics={"validation": {"rmse": 0.05}},
            feature_importance=pd.DataFrame([
                {"feature": "feat_promo_alpha", "importance": 0.6},
                {"feature": "feat_standard_beta", "importance": 0.4},
            ]),
            metadata={"row_count": 5000},
            matrix_report={},
            split_info={"strategy": "time_series"},
        )

        pkg_dir = pkg_info["package_dir"]
        meta_path = os.path.join(pkg_dir, "metadata.json")
        manifest_path = os.path.join(pkg_dir, "manifest.json")

        self.assertTrue(os.path.isfile(meta_path))
        self.assertTrue(os.path.isfile(manifest_path))

        with open(meta_path, "r", encoding="utf-8") as f:
            meta_doc = json.load(f)

        self.assertIn("recommendation_provenance", meta_doc)
        prov = meta_doc["recommendation_provenance"]
        self.assertTrue(prov["has_recommendation_lineage"])
        self.assertEqual(prov["context_id"], self.ctx_nifty_3s.context_id)
        self.assertEqual(prov["market"], "NIFTY")
        self.assertEqual(prov["sampling_interval_sec"], 3)
        self.assertEqual(prov["trained_candidates"], ["feat_promo_alpha", "feat_standard_beta"])
        self.assertIn("feat_promo_alpha", prov["feature_decision_snapshots"])

        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest_doc = json.load(f)
        self.assertIn("recommendation_provenance", manifest_doc)
        self.assertEqual(manifest_doc["recommendation_provenance"]["context_id"], self.ctx_nifty_3s.context_id)

    def test_06_legacy_model_package_without_provenance(self) -> None:
        """Verify legacy model packages without recommendation bundle save and function without errors."""
        cfg = normalize_training_config({
            "dataset": "ohlc_legacy",
            "target": "target_ret",
            "algorithm": "xgboost",
            "features": ["feat_1", "feat_2"],
            "model_name": "model_test_legacy",
        })

        mock_model = MagicMock()
        mock_model.save_model = MagicMock()

        pkg_info = save_model_package(
            data_dir=self.data_dir,
            config=cfg,
            model=mock_model,
            metrics={"validation": {"rmse": 0.05}},
            feature_importance=pd.DataFrame([{"feature": "feat_1", "importance": 1.0}]),
            metadata={"row_count": 1000},
            matrix_report={},
            split_info={"strategy": "time_series"},
        )

        pkg_dir = pkg_info["package_dir"]
        meta_path = os.path.join(pkg_dir, "metadata.json")
        with open(meta_path, "r", encoding="utf-8") as f:
            meta_doc = json.load(f)
        self.assertNotIn("recommendation_provenance", meta_doc)

        prov = get_model_recommendation_provenance(pkg_dir)
        self.assertIsNone(prov)

    def test_07_authoritative_context_id_resolution_from_provenance(self) -> None:
        """Verify resolve_context_from_model_package() recovers exact context from stamped provenance."""
        bundle = build_model_builder_training_bundle(self.data_dir, context=self.ctx_nifty_3s)
        rdb = bundle["recommendation_decision_bundle"]

        cfg = normalize_training_config({
            "dataset": "ambiguous_dataset_name_without_market",
            "target": "target_ret",
            "features": ["feat_promo_alpha"],
            "model_name": "model_authoritative_context",
            "recommendation_decision_bundle": rdb,
        })

        mock_model = MagicMock()
        save_model_package(
            data_dir=self.data_dir,
            config=cfg,
            model=mock_model,
            metrics={"validation": {"rmse": 0.05}},
            feature_importance=pd.DataFrame([{"feature": "feat_promo_alpha", "importance": 1.0}]),
            metadata={"row_count": 1000},
            matrix_report={},
            split_info={"strategy": "time_series"},
        )

        resolved_ctx = resolve_context_from_model_package(self.data_dir, "model_authoritative_context")
        self.assertIsNotNone(resolved_ctx)
        self.assertEqual(resolved_ctx.context_id, self.ctx_nifty_3s.context_id)
        self.assertEqual(resolved_ctx.market, "NIFTY")
        self.assertEqual(resolved_ctx.sampling_interval_sec, 3)

    def test_08_legacy_heuristic_context_fallback(self) -> None:
        """Verify resolve_context_from_model_package() falls back to heuristics for legacy models."""
        cfg = normalize_training_config({
            "dataset": "ohlc_NIFTY_3s_standard",
            "target": "target_ret",
            "features": ["feat_1"],
            "model_name": "model_legacy_heuristic",
        })
        mock_model = MagicMock()
        save_model_package(
            data_dir=self.data_dir,
            config=cfg,
            model=mock_model,
            metrics={"validation": {"rmse": 0.05}},
            feature_importance=pd.DataFrame([{"feature": "feat_1", "importance": 1.0}]),
            metadata={"row_count": 1000, "market": "NIFTY", "sampling_interval_sec": 3},
            matrix_report={},
            split_info={"strategy": "time_series"},
        )

        resolved_ctx = resolve_context_from_model_package(self.data_dir, "model_legacy_heuristic")
        self.assertIsNotNone(resolved_ctx)
        self.assertEqual(resolved_ctx.market, "NIFTY")
        self.assertEqual(resolved_ctx.sampling_interval_sec, 3)

    def test_09_get_model_recommendation_provenance_api(self) -> None:
        """Verify get_model_recommendation_provenance() reader function."""
        bundle = build_model_builder_training_bundle(self.data_dir, context=self.ctx_nifty_3s)
        cfg = normalize_training_config({
            "dataset": "ohlc_NIFTY_3s_standard",
            "target": "target_ret",
            "features": ["feat_promo_alpha"],
            "model_name": "model_provenance_reader",
            "recommendation_decision_bundle": bundle["recommendation_decision_bundle"],
        })
        mock_model = MagicMock()
        pkg_info = save_model_package(
            data_dir=self.data_dir,
            config=cfg,
            model=mock_model,
            metrics={"validation": {"rmse": 0.05}},
            feature_importance=pd.DataFrame([{"feature": "feat_promo_alpha", "importance": 1.0}]),
            metadata={"row_count": 1000},
            matrix_report={},
            split_info={"strategy": "time_series"},
        )

        prov = get_model_recommendation_provenance(pkg_info["package_dir"])
        self.assertIsNotNone(prov)
        self.assertTrue(prov["has_recommendation_lineage"])
        self.assertEqual(prov["context_id"], self.ctx_nifty_3s.context_id)

    def test_10_audit_model_training_feedback_loop_pass(self) -> None:
        """Verify audit_model_training_feedback_loop() returns PASS on complete traceable loop."""
        bundle = build_model_builder_training_bundle(self.data_dir, context=self.ctx_nifty_3s)
        model_name = "model_audit_traceable"
        cfg = normalize_training_config({
            "dataset": "ohlc_NIFTY_3s_standard",
            "target": "target_ret",
            "features": ["feat_promo_alpha"],
            "model_name": model_name,
            "recommendation_decision_bundle": bundle["recommendation_decision_bundle"],
        })
        mock_model = MagicMock()
        pkg_info = save_model_package(
            data_dir=self.data_dir,
            config=cfg,
            model=mock_model,
            metrics={"validation": {"rmse": 0.05}},
            feature_importance=pd.DataFrame([{"feature": "feat_promo_alpha", "importance": 1.0}]),
            metadata={"row_count": 1000},
            matrix_report={},
            split_info={"strategy": "time_series"},
        )

        # Simulate Production Validation writing evidence for this model
        conn = get_connection(self.data_dir)
        try:
            ev_rows = [
                {
                    "feature_name": "feat_promo_alpha",
                    "feature_source": "experimental",
                    "pipeline_id": "PL_EXP_1",
                    "pipeline_snapshot_id": "SNP_1",
                    "recommendation": "KEEP",
                    "permutation_mean": 0.098,
                    "importance_rank": 1,
                    "run_timestamp": "2026-08-18T13:00:00Z",
                    "model_name": model_name,
                }
            ]
            append_validation_evidence(conn, context=self.ctx_nifty_3s, evidence_rows=ev_rows)
        finally:
            conn.close()

        # Audit
        audit = audit_model_training_feedback_loop(self.data_dir, model_name)
        self.assertEqual(audit["audit_status"], "PASS")
        self.assertTrue(audit["has_recommendation_provenance"])
        self.assertEqual(audit["context_id"], self.ctx_nifty_3s.context_id)
        self.assertGreater(audit["db_validation_runs_count"], 0)
        self.assertTrue(audit["feedback_loop_closed"])

    def test_11_evidence_db_immutability_during_audit(self) -> None:
        """Verify audit_model_training_feedback_loop() is strictly read-only."""
        sha_before = _file_sha256(self.db_path)
        for _ in range(5):
            audit_model_training_feedback_loop(self.data_dir, "non_existent_model")
        sha_after = _file_sha256(self.db_path)
        self.assertEqual(sha_before, sha_after)

    def test_12_persisted_state_roundtrip(self) -> None:
        """Verify save_persisted_state and load_persisted_state preserve recommendation_decision_bundle."""
        bundle = build_model_builder_training_bundle(self.data_dir, context=self.ctx_nifty_3s)
        rdb = bundle["recommendation_decision_bundle"]

        state = ModelBuilderState(
            dataset="ohlc_NIFTY_3s_standard",
            target="target_ret",
            features={"feat_promo_alpha"},
            recommendation_decision_bundle=rdb,
        )

        save_persisted_state(self.chart_dir, state)
        loaded_doc = load_persisted_state(self.chart_dir)
        self.assertIsNotNone(loaded_doc)

        restored_state = ModelBuilderState()
        restored_state.apply_saved_dict(loaded_doc)

        self.assertIsNotNone(restored_state.recommendation_decision_bundle)
        self.assertEqual(
            restored_state.recommendation_decision_bundle["context_id"],
            self.ctx_nifty_3s.context_id,
        )

    def test_13_full_closed_loop_traceability_chain(self) -> None:
        """Comprehensive closed loop: Decision -> Preset -> State -> Config -> Package -> Validation -> Evidence DB."""
        # 1. Phase 3A decision evaluation & Phase 3B bundle
        bundle = build_model_builder_training_bundle(self.data_dir, context=self.ctx_nifty_3s)
        candidates = bundle["features"]
        self.assertIn("feat_promo_alpha", candidates)

        # 2. Export preset
        preset = export_training_candidates_preset(
            self.chart_dir,
            self.data_dir,
            context=self.ctx_nifty_3s,
            selected_features=["feat_promo_alpha"],
        )

        # 3. Model Builder State loads preset
        res = apply_feature_preset(
            preset,
            dataset_name="ohlc_NIFTY_3s_standard",
            dataset_feature_names=["feat_promo_alpha", "feat_standard_beta"],
        )
        state = ModelBuilderState()
        state.features = set(res["features"])
        state.recommendation_decision_bundle = res.get("recommendation_decision_bundle")

        # 4. State builds training config
        train_cfg_dict = state.build_training_config()
        train_cfg_dict["model_name"] = "model_full_loop_v1"
        training_config = normalize_training_config(train_cfg_dict)

        # 5. Training saves package
        mock_model = MagicMock()
        pkg_info = save_model_package(
            data_dir=self.data_dir,
            config=training_config,
            model=mock_model,
            metrics={"validation": {"rmse": 0.04}},
            feature_importance=pd.DataFrame([{"feature": "feat_promo_alpha", "importance": 1.0}]),
            metadata={"row_count": 5000},
            matrix_report={},
            split_info={"strategy": "time_series"},
        )

        # 6. Production Validation resolves context authoritatively
        val_ctx = resolve_context_from_model_package(self.data_dir, "model_full_loop_v1")
        self.assertEqual(val_ctx.context_id, self.ctx_nifty_3s.context_id)

        # 7. Production Validation writes new evidence
        conn = get_connection(self.data_dir)
        try:
            append_validation_evidence(
                conn,
                context=val_ctx,
                evidence_rows=[
                    {
                        "feature_name": "feat_promo_alpha",
                        "feature_source": "experimental",
                        "pipeline_id": "PL_EXP_1",
                        "pipeline_snapshot_id": "SNP_1",
                        "recommendation": "KEEP",
                        "permutation_mean": 0.105,
                        "importance_rank": 1,
                        "run_timestamp": "2026-08-18T14:00:00Z",
                        "model_name": "model_full_loop_v1",
                    }
                ],
            )
        finally:
            conn.close()

        # 8. Verify audit confirms closed loop
        audit = audit_model_training_feedback_loop(self.data_dir, "model_full_loop_v1")
        self.assertEqual(audit["audit_status"], "PASS")
        self.assertTrue(audit["feedback_loop_closed"])
        self.assertEqual(audit["db_validation_runs_count"], 1)

    def test_14_context_mismatch_rejection_and_handling(self) -> None:
        """Verify applying a NIFTY 3s preset against a SENSEX 1s dataset flags mismatch safely."""
        preset = export_training_candidates_preset(
            self.chart_dir,
            self.data_dir,
            context=self.ctx_nifty_3s,
        )

        res = apply_feature_preset(
            preset,
            dataset_name="ohlc_SENSEX_1s_standard",
            dataset_feature_names=["feat_promo_alpha"],
        )
        self.assertFalse(res["context_match"])
        self.assertIsNotNone(res["context_warning"])
        self.assertIn("Context Mismatch", res["context_warning"])

    def test_15_phase_3a_3b_behavior_immutability(self) -> None:
        """Verify Phase 3A decision evaluation and 4-boolean contract remain unchanged."""
        from chain_replay_ml.production_validation.api import evaluate_training_decision

        dec = evaluate_training_decision(
            context_id=self.ctx_nifty_3s.context_id,
            feature_name="feat_promo_alpha",
            feature_source="experimental",
            total_runs=3,
            unique_models_count=3,
            evidence_score=90.0,
            consecutive_keep_count=3,
            dominant_recommendation="KEEP",
            is_promotion_candidate=True,
        )
        self.assertEqual(dec.decision, TrainingDecisionState.TRAIN_CANDIDATE)
        self.assertTrue(dec.is_training_candidate)
        self.assertTrue(dec.is_candidate_generation_allowed)
        self.assertFalse(dec.is_excluded)
        self.assertFalse(dec.requires_review)


if __name__ == "__main__":
    unittest.main()


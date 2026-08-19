"""Comprehensive Unit Tests for Phase 4D.4: Regime & Feature Composition Evaluation."""

import hashlib
import os
import shutil
import sqlite3
import tempfile
import unittest

from chain_replay_ml.research_memory import (
    analyze_feature_set_composition,
    calculate_regime_degradation,
    classify_feature_population,
    get_feature_set_evaluation,
    get_regime_evaluations_for_model,
    get_regime_evaluations_for_regime,
    init_analysis_db,
    record_feature_set_evaluation,
    record_multi_regime_evaluations,
    record_regime_evaluation,
    register_or_get_experiment,
    summarize_regime_feature_affinity,
)


class TestRegimeAndFeatureComposition(unittest.TestCase):
    """Test suite verifying feature population composition, regime degradation, and research persistence."""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp(prefix="aruml_test_regime_comp_")
        init_analysis_db(self.tmp_dir)

        # Mock schema dictionary for deterministic testing
        self.mock_schema = {
            "columns": {
                "adx_14": {"status": "ACTIVE", "is_base": True, "project_id": "PL_0001"},
                "rsi_14": {"status": "ACTIVE", "is_base": True, "project_id": "PL_0001"},
                "atm_iv_pctile": {"status": "ACTIVE", "is_base": False, "project_id": ""},
                "basis": {"status": "ACTIVE", "is_base": False, "project_id": ""},
                "cand_cross_ratio": {"status": "ACTIVE", "is_base": False, "project_id": "PL_0002"},
                "cand_skew_momentum": {"status": "EXPERIMENTAL", "is_base": False, "project_id": "PL_0003"},
                "retired_legacy_vol": {"status": "DEPRECATED", "is_base": False, "project_id": ""},
            }
        }

        self.base_spec = {
            "market": "NIFTY",
            "sampling_interval_sec": 3,
            "task_type": "DIRECTION_CLASSIFIER",
            "prediction_horizon": "5m",
            "regime_id": "R001",
            "regime_definition_hash": "def_hash_trend_v1",
            "dataset_snapshot_hash": "ds_hash_20260819",
            "features": ["adx_14", "rsi_14", "atm_iv_pctile", "cand_cross_ratio", "unknown_custom_feat"],
            "algorithm": "xgboost",
            "hyperparameters": {"max_depth": 6, "learning_rate": 0.05},
            "walk_forward_config": {"folds": 5, "window_mode": "expanding"},
            "random_seed": 42,
        }

        # Pre-register experiment signature in analysis.db
        _, self.rec_exp = register_or_get_experiment(self.tmp_dir, self.base_spec, model_name="DIR_TREND_XGB_v1")

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_classify_feature_population(self):
        """1. Verify individual features classify strictly according to authoritative metadata."""
        self.assertEqual(classify_feature_population("adx_14", schema=self.mock_schema), "BASE")
        self.assertEqual(classify_feature_population("atm_iv_pctile", schema=self.mock_schema), "REGISTRY")
        self.assertEqual(classify_feature_population("cand_cross_ratio", schema=self.mock_schema), "EXPERIMENTAL")
        self.assertEqual(classify_feature_population("cand_skew_momentum", schema=self.mock_schema), "EXPERIMENTAL")
        self.assertEqual(classify_feature_population("retired_legacy_vol", schema=self.mock_schema), "DEPRECATED")
        self.assertEqual(classify_feature_population("non_existent_feature", schema=self.mock_schema), "UNKNOWN")

    def test_analyze_feature_set_composition(self):
        """2. Verify feature set composition calculates counts, dependency ratios, and categorized lists."""
        feats = ["adx_14", "rsi_14", "atm_iv_pctile", "cand_cross_ratio", "retired_legacy_vol", "unknown_x"]
        top_10 = [{"name": "adx_14", "gain": 0.42}, {"name": "rsi_14", "gain": 0.35}]

        comp = analyze_feature_set_composition(feats, top_features=top_10, schema=self.mock_schema)

        self.assertEqual(comp["total_features"], 6)
        self.assertEqual(comp["base_pipeline_count"], 2) # adx_14, rsi_14
        self.assertEqual(comp["registry_feature_count"], 1) # atm_iv_pctile
        self.assertEqual(comp["experimental_feature_count"], 1) # cand_cross_ratio
        self.assertEqual(comp["deprecated_feature_count"], 1) # retired_legacy_vol
        self.assertEqual(comp["unknown_feature_count"], 1) # unknown_x

        # Dependency ratios
        self.assertAlmostEqual(comp["experimental_dependency_ratio"], 1 / 6, places=4)
        self.assertAlmostEqual(comp["base_dependency_ratio"], 2 / 6, places=4)
        self.assertAlmostEqual(comp["registry_dependency_ratio"], 1 / 6, places=4)

        # Inventory check
        inv = comp["categorized_inventory"]
        self.assertIn("adx_14", inv["base_features"])
        self.assertIn("unknown_x", inv["unknown_features"])
        self.assertEqual(len(comp["top_10_features"]), 2)

    def test_record_and_get_feature_set_evaluation(self):
        """3. Verify feature_set_evaluations table persistence and JSON roundtrip."""
        feats = ["adx_14", "rsi_14", "atm_iv_pctile", "cand_cross_ratio"]
        top_10 = [{"name": "adx_14", "importance": 0.55}]

        eval_id = record_feature_set_evaluation(
            self.tmp_dir,
            signature_hash=self.rec_exp["signature_hash"],
            features=feats,
            top_features=top_10,
            schema=self.mock_schema,
        )
        self.assertGreater(eval_id, 0)

        # Retrieve evaluation
        doc = get_feature_set_evaluation(self.tmp_dir, self.rec_exp["signature_hash"])
        self.assertIsNotNone(doc)
        self.assertEqual(doc["total_features"], 4)
        self.assertEqual(doc["base_pipeline_count"], 2)
        self.assertEqual(doc["experimental_feature_count"], 1)
        self.assertIn("base_features", doc["features_inventory"])
        self.assertEqual(len(doc["top_10_features"]), 1)

    def test_calculate_regime_degradation_higher_is_better(self):
        """4. Verify degradation formula for higher-is-better metrics (ROC-AUC, F1, Accuracy)."""
        # 10% drop
        deg1 = calculate_regime_degradation(native_metric=0.70, tested_metric=0.63, higher_is_better=True)
        self.assertEqual(deg1, 10.0)

        # Performance improved outside regime -> 0.0 degradation
        deg2 = calculate_regime_degradation(native_metric=0.70, tested_metric=0.75, higher_is_better=True)
        self.assertEqual(deg2, 0.0)

        # Identical performance -> 0.0 degradation
        deg3 = calculate_regime_degradation(native_metric=0.70, tested_metric=0.70, higher_is_better=True)
        self.assertEqual(deg3, 0.0)

    def test_calculate_regime_degradation_lower_is_better(self):
        """5. Verify degradation formula for lower-is-better metrics (RMSE, MAE, Log-Loss)."""
        # 50% error increase
        deg1 = calculate_regime_degradation(native_metric=0.02, tested_metric=0.03, higher_is_better=False)
        self.assertEqual(deg1, 50.0)

        # Error decreased outside regime -> 0.0 degradation
        deg2 = calculate_regime_degradation(native_metric=0.02, tested_metric=0.015, higher_is_better=False)
        self.assertEqual(deg2, 0.0)

    def test_record_single_regime_evaluation(self):
        """6. Verify single regime evaluation insertion into analysis.db."""
        eval_id = record_regime_evaluation(
            self.tmp_dir,
            model_name="DIR_TREND_XGB_v1",
            signature_hash=self.rec_exp["signature_hash"],
            tested_regime_id="R001",
            tested_regime_hash="def_hash_trend_v1",
            is_native_regime=True,
            sample_count=5000,
            primary_metric=0.72,
            regime_degradation_pct=0.0,
        )
        self.assertGreater(eval_id, 0)

        evals = get_regime_evaluations_for_model(self.tmp_dir, self.rec_exp["signature_hash"])
        self.assertEqual(len(evals), 1)
        self.assertEqual(evals[0]["tested_regime_id"], "R001")
        self.assertEqual(evals[0]["is_native_regime"], 1)
        self.assertEqual(evals[0]["regime_degradation_pct"], 0.0)

    def test_record_multi_regime_evaluations_atomic(self):
        """7. Verify atomic multi-regime evaluation slices and degradation calculation."""
        eval_slices = [
            {"tested_regime_id": "R001", "tested_regime_hash": "def_hash_trend_v1", "sample_count": 4000, "primary_metric": 0.70},
            {"tested_regime_id": "R002", "tested_regime_hash": "def_hash_side_v1", "sample_count": 3500, "primary_metric": 0.56}, # 20% drop
            {"tested_regime_id": "R003", "tested_regime_hash": "def_hash_highvol_v1", "sample_count": 1200, "primary_metric": 0.63}, # 10% drop
            {"tested_regime_id": "R004", "tested_regime_hash": "def_hash_lowvol_v1", "sample_count": 0, "primary_metric": 0.0}, # Missing slice -> skipped
        ]

        ids = record_multi_regime_evaluations(
            self.tmp_dir,
            model_name="DIR_TREND_XGB_v1",
            signature_hash=self.rec_exp["signature_hash"],
            native_regime_id="R001",
            native_metric=0.70,
            evaluations=eval_slices,
            higher_is_better=True,
        )

        # 3 non-empty slices should be inserted (R004 with 0 samples is skipped)
        self.assertEqual(len(ids), 3)

        records = get_regime_evaluations_for_model(self.tmp_dir, self.rec_exp["signature_hash"])
        self.assertEqual(len(records), 3)

        # Native R001
        r001 = next(r for r in records if r["tested_regime_id"] == "R001")
        self.assertEqual(r001["is_native_regime"], 1)
        self.assertEqual(r001["regime_degradation_pct"], 0.0)

        # Non-native R002 (Sideways) -> 20.0% degradation
        r002 = next(r for r in records if r["tested_regime_id"] == "R002")
        self.assertEqual(r002["is_native_regime"], 0)
        self.assertEqual(r002["regime_degradation_pct"], 20.0)

        # Non-native R003 (High Vol) -> 10.0% degradation
        r003 = next(r for r in records if r["tested_regime_id"] == "R003")
        self.assertEqual(r003["is_native_regime"], 0)
        self.assertEqual(r003["regime_degradation_pct"], 10.0)

    def test_summarize_regime_feature_affinity(self):
        """8. Verify descriptive empirical aggregation of feature population ratios by regime."""
        # Record feature set evaluation
        record_feature_set_evaluation(
            self.tmp_dir,
            signature_hash=self.rec_exp["signature_hash"],
            features=["adx_14", "rsi_14", "atm_iv_pctile", "cand_cross_ratio"],
            schema=self.mock_schema,
        )

        # Record regime evaluation under R001
        record_regime_evaluation(
            self.tmp_dir,
            model_name="DIR_TREND_XGB_v1",
            signature_hash=self.rec_exp["signature_hash"],
            tested_regime_id="R001",
            tested_regime_hash="def_hash_trend_v1",
            is_native_regime=True,
            sample_count=4000,
            primary_metric=0.70,
        )

        affinity = summarize_regime_feature_affinity(self.tmp_dir, "R001")
        self.assertEqual(affinity["regime_id"], "R001")
        self.assertEqual(affinity["models_evaluated_count"], 1)
        self.assertEqual(affinity["avg_total_features"], 4.0)
        self.assertEqual(affinity["avg_base_features_count"], 2.0)
        self.assertEqual(affinity["avg_experimental_features_count"], 1.0)
        self.assertAlmostEqual(affinity["avg_experimental_dependency_ratio"], 0.25, places=4)

    def test_foreign_key_lineage_enforcement(self):
        """9. Verify foreign keys reject regime/feature evaluations for non-existent signatures."""
        with self.assertRaises(sqlite3.IntegrityError):
            record_regime_evaluation(
                self.tmp_dir,
                model_name="BAD_MODEL",
                signature_hash="invalid_non_existent_signature_hash",
                tested_regime_id="R001",
                tested_regime_hash="def_hash_trend_v1",
                is_native_regime=True,
                sample_count=100,
                primary_metric=0.5,
            )

        with self.assertRaises(sqlite3.IntegrityError):
            record_feature_set_evaluation(
                self.tmp_dir,
                signature_hash="invalid_non_existent_signature_hash",
                features=["adx_14"],
                schema=self.mock_schema,
            )

    def test_evidence_db_immutability(self):
        """10. Verify Phase 4D.4 operations do NOT touch or mutate feature_recommendation_evidence.db."""
        ev_path = "apps/feature_recommendation_evidence.db"
        self.assertTrue(os.path.isfile(ev_path))
        with open(ev_path, "rb") as fh:
            sha_before = hashlib.sha256(fh.read()).hexdigest()

        # Perform feature comp and regime eval
        record_feature_set_evaluation(
            self.tmp_dir,
            signature_hash=self.rec_exp["signature_hash"],
            features=["adx_14", "rsi_14"],
            schema=self.mock_schema,
        )
        record_regime_evaluation(
            self.tmp_dir,
            model_name="DIR_TREND_XGB_v1",
            signature_hash=self.rec_exp["signature_hash"],
            tested_regime_id="R001",
            tested_regime_hash="def_hash_trend_v1",
            is_native_regime=True,
            sample_count=2000,
            primary_metric=0.68,
        )

        with open(ev_path, "rb") as fh:
            sha_after = hashlib.sha256(fh.read()).hexdigest()

        self.assertEqual(sha_before, sha_after)
        self.assertEqual(sha_after, "6f91afca94ec87a1210d8f4bcef356b9c16a6ef5a488268f41c5b1b81431ade2")


if __name__ == "__main__":
    unittest.main()

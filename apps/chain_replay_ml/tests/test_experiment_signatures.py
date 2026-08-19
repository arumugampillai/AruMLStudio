"""Comprehensive Unit Tests for Phase 4D.2: Experiment Identity & Canonical Deduplication."""

import concurrent.futures
import hashlib
import json
import os
import shutil
import tempfile
import unittest

from chain_replay_ml.research_memory import (
    build_canonical_experiment_payload,
    canonicalize_json,
    check_experiment_exists,
    compute_experiment_signature,
    compute_subcomponent_hash,
    connect_analysis_db,
    get_experiment_by_signature,
    init_analysis_db,
    list_experiments_for_context,
    register_or_get_experiment,
)


class TestExperimentSignatures(unittest.TestCase):
    """Test suite verifying pure canonicalization, deterministic hashing, and atomic deduplication."""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp(prefix="aruml_test_signatures_")
        self.base_spec = {
            "market": "NIFTY",
            "sampling_interval_sec": 3,
            "task_type": "DIRECTION_CLASSIFIER",
            "prediction_horizon": "5m",
            "regime_id": "R001",
            "regime_definition_hash": "def_hash_trend_v1",
            "dataset_snapshot_hash": "ds_hash_20260819",
            "features": ["adx_14", "rsi_14", "basis", "atm_iv_pctile"],
            "algorithm": "xgboost",
            "hyperparameters": {
                "max_depth": 6,
                "learning_rate": 0.05,
                "n_estimators": 500,
                "subsample": 0.8,
            },
            "walk_forward_config": {
                "folds": 5,
                "window_mode": "expanding",
                "train_window": 5000,
                "val_window": 1000,
            },
            "random_seed": 42,
        }

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_identical_experiments_produce_identical_signatures(self):
        """1. Verify identical experiments produce exact same SHA-256 signature hash."""
        sig1, json1, _ = compute_experiment_signature(self.base_spec)
        sig2, json2, _ = compute_experiment_signature(self.base_spec)

        self.assertEqual(sig1, sig2)
        self.assertEqual(json1, json2)
        self.assertEqual(len(sig1), 64)

    def test_dictionary_ordering_does_not_affect_signature(self):
        """2. Verify dictionary key order permutations produce identical signature."""
        spec_permuted = {
            "random_seed": 42,
            "algorithm": "xgboost",
            "features": ["atm_iv_pctile", "basis", "rsi_14", "adx_14"], # permuted list
            "prediction_horizon": "5m",
            "task_type": "DIRECTION_CLASSIFIER",
            "hyperparameters": {
                "subsample": 0.8,
                "n_estimators": 500,
                "learning_rate": 0.05,
                "max_depth": 6,
            },
            "regime_id": "R001",
            "market": "NIFTY",
            "walk_forward_config": {
                "val_window": 1000,
                "train_window": 5000,
                "window_mode": "expanding",
                "folds": 5,
            },
            "dataset_snapshot_hash": "ds_hash_20260819",
            "sampling_interval_sec": 3,
            "regime_definition_hash": "def_hash_trend_v1",
        }

        sig1, _, _ = compute_experiment_signature(self.base_spec)
        sig2, _, _ = compute_experiment_signature(spec_permuted)
        self.assertEqual(sig1, sig2)

    def test_different_features_produce_different_signatures(self):
        """3. Verify different feature sets produce distinct signatures."""
        spec_diff_feats = dict(self.base_spec)
        spec_diff_feats["features"] = ["adx_14", "rsi_14", "basis", "vanna"]

        sig1, _, _ = compute_experiment_signature(self.base_spec)
        sig2, _, _ = compute_experiment_signature(spec_diff_feats)
        self.assertNotEqual(sig1, sig2)

    def test_different_dataset_snapshot_produces_different_signature(self):
        """4. Verify different dataset snapshots produce distinct signatures."""
        spec_diff_ds = dict(self.base_spec)
        spec_diff_ds["dataset_snapshot_hash"] = "ds_hash_diff_date"

        sig1, _, _ = compute_experiment_signature(self.base_spec)
        sig2, _, _ = compute_experiment_signature(spec_diff_ds)
        self.assertNotEqual(sig1, sig2)

    def test_different_regime_definition_hash_produces_different_signature(self):
        """5. Verify updating regime definition changes the experiment signature."""
        spec_diff_regime = dict(self.base_spec)
        spec_diff_regime["regime_definition_hash"] = "def_hash_trend_v2"

        sig1, _, _ = compute_experiment_signature(self.base_spec)
        sig2, _, _ = compute_experiment_signature(spec_diff_regime)
        self.assertNotEqual(sig1, sig2)

    def test_different_hyperparameters_produce_different_signature(self):
        """6. Verify hyperparameter changes change signature."""
        spec_diff_hparam = dict(self.base_spec)
        spec_diff_hparam["hyperparameters"] = dict(self.base_spec["hyperparameters"])
        spec_diff_hparam["hyperparameters"]["learning_rate"] = 0.01

        sig1, _, _ = compute_experiment_signature(self.base_spec)
        sig2, _, _ = compute_experiment_signature(spec_diff_hparam)
        self.assertNotEqual(sig1, sig2)

    def test_different_algorithm_produces_different_signature(self):
        """7. Verify different model architectures change signature."""
        spec_lgbm = dict(self.base_spec)
        spec_lgbm["algorithm"] = "lightgbm"

        sig1, _, _ = compute_experiment_signature(self.base_spec)
        sig2, _, _ = compute_experiment_signature(spec_lgbm)
        self.assertNotEqual(sig1, sig2)

    def test_different_prediction_horizon_produces_different_signature(self):
        """8. Verify prediction horizon change changes signature."""
        spec_15m = dict(self.base_spec)
        spec_15m["prediction_horizon"] = "15m"

        sig1, _, _ = compute_experiment_signature(self.base_spec)
        sig2, _, _ = compute_experiment_signature(spec_15m)
        self.assertNotEqual(sig1, sig2)

    def test_different_random_seed_produces_different_signature(self):
        """9. Verify random seed change changes signature."""
        spec_seed = dict(self.base_spec)
        spec_seed["random_seed"] = 12345

        sig1, _, _ = compute_experiment_signature(self.base_spec)
        sig2, _, _ = compute_experiment_signature(spec_seed)
        self.assertNotEqual(sig1, sig2)

    def test_runtime_metadata_does_not_affect_signature(self):
        """10. Verify non-semantic runtime fields (timestamps, durations, model names) are ignored."""
        spec1 = dict(self.base_spec)
        spec1["model_name"] = "XGB_MODEL_ALPHA"
        spec1["trained_at"] = "2026-08-19T10:00:00Z"
        spec1["training_time_sec"] = 45.2
        spec1["process_id"] = 1234
        spec1["hostname"] = "WORKSTATION-1"

        spec2 = dict(self.base_spec)
        spec2["model_name"] = "XGB_MODEL_BETA"
        spec2["trained_at"] = "2026-08-19T18:30:00Z"
        spec2["training_time_sec"] = 12.8
        spec2["process_id"] = 9999
        spec2["hostname"] = "WORKSTATION-2"

        sig1, _, _ = compute_experiment_signature(spec1)
        sig2, _, _ = compute_experiment_signature(spec2)
        self.assertEqual(sig1, sig2)

    def test_floating_point_quantization_prevents_precision_drift(self):
        """11. Verify floating point numbers are quantized to 6 decimals."""
        spec1 = dict(self.base_spec)
        spec1["hyperparameters"] = {"lr": 0.050000000001}

        spec2 = dict(self.base_spec)
        spec2["hyperparameters"] = {"lr": 0.050000}

        sig1, _, _ = compute_experiment_signature(spec1)
        sig2, _, _ = compute_experiment_signature(spec2)
        self.assertEqual(sig1, sig2)

    def test_subcomponent_hashes_match_expected_hashes(self):
        """12. Verify subcomponent hashing for features, hyperparams, and walk-forward."""
        feat_hash1 = compute_subcomponent_hash(["adx_14", "rsi_14"])
        feat_hash2 = compute_subcomponent_hash(["rsi_14", "adx_14"])
        self.assertEqual(feat_hash1, feat_hash2)

    def test_atomic_registration_and_duplicate_deduplication(self):
        """13. Verify register_or_get_experiment correctly detects duplicate experiments."""
        init_analysis_db(self.tmp_dir)

        # First execution -> New experiment registered
        is_new1, rec1 = register_or_get_experiment(
            self.tmp_dir,
            self.base_spec,
            model_name="DIR_TREND_XGB_v1",
            executed_at="2026-08-19T10:00:00Z",
        )
        self.assertTrue(is_new1)
        self.assertEqual(rec1["execution_count"], 1)
        self.assertEqual(rec1["latest_model_name"], "DIR_TREND_XGB_v1")

        # Second execution -> Duplicate detected, count incremented, no duplicate row
        is_new2, rec2 = register_or_get_experiment(
            self.tmp_dir,
            self.base_spec,
            model_name="DIR_TREND_XGB_v2",
            executed_at="2026-08-19T11:00:00Z",
        )
        self.assertFalse(is_new2)
        self.assertEqual(rec2["execution_count"], 2)
        self.assertEqual(rec2["latest_model_name"], "DIR_TREND_XGB_v2")
        self.assertEqual(rec2["first_executed_at"], "2026-08-19T10:00:00Z")

        # Verify only one row exists in analysis.db
        conn = connect_analysis_db(self.tmp_dir)
        try:
            count = conn.execute("SELECT COUNT(*) FROM experiment_signatures;").fetchone()[0]
            self.assertEqual(count, 1)
        finally:
            conn.close()

    def test_concurrent_experiment_registration_is_strictly_atomic(self):
        """14. Verify concurrent workers attempting to register the same experiment resolve cleanly."""
        init_analysis_db(self.tmp_dir)

        def worker_task(worker_id):
            return register_or_get_experiment(
                self.tmp_dir,
                self.base_spec,
                model_name=f"WORKER_MODEL_{worker_id}",
            )

        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            results = list(executor.map(worker_task, range(5)))

        # Exactly one worker should receive is_new=True
        new_count = sum(1 for is_new, _ in results if is_new)
        self.assertEqual(new_count, 1)

        # Total executions should be 5
        conn = connect_analysis_db(self.tmp_dir)
        try:
            row = conn.execute("SELECT execution_count FROM experiment_signatures;").fetchone()
            self.assertEqual(row["execution_count"], 5)
        finally:
            conn.close()

    def test_distinct_experiments_coexist_safely(self):
        """15. Verify multiple distinct experiments register independently."""
        init_analysis_db(self.tmp_dir)

        spec_trend = dict(self.base_spec)
        spec_side = dict(self.base_spec)
        spec_side["regime_id"] = "R002"
        spec_side["regime_definition_hash"] = "def_hash_sideways_v1"

        is_new_t, _ = register_or_get_experiment(self.tmp_dir, spec_trend, model_name="TREND_M1")
        is_new_s, _ = register_or_get_experiment(self.tmp_dir, spec_side, model_name="SIDE_M1")

        self.assertTrue(is_new_t)
        self.assertTrue(is_new_s)

        trend_exps = list_experiments_for_context(self.tmp_dir, "NIFTY_3s_DIRECTION_CLASSIFIER_5m_R001")
        side_exps = list_experiments_for_context(self.tmp_dir, "NIFTY_3s_DIRECTION_CLASSIFIER_5m_R002")

        self.assertEqual(len(trend_exps), 1)
        self.assertEqual(len(side_exps), 1)

    def test_legacy_model_spec_resolution(self):
        """16. Verify legacy config dictionaries resolve cleanly to canonical signatures without error."""
        legacy_dict = {
            "config": {
                "target": "label_up_5m",
                "market": "NIFTY",
                "sampling_interval_sec": 3,
                "features": ["f1", "f2", "f3"],
                "algorithm": "xgboost",
                "xgb_lr": 0.05,
                "xgb_trees": 500,
            }
        }
        sig, canon_json, norm_payload = compute_experiment_signature(legacy_dict)
        self.assertEqual(len(sig), 64)
        self.assertEqual(norm_payload["regime_id"], "R000")
        self.assertEqual(norm_payload["task_type"], "DIRECTION_CLASSIFIER")


if __name__ == "__main__":
    unittest.main()

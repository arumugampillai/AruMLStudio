"""Comprehensive Unit Tests for Phase 4D.3: Model Benchmark & Metrics Persistence."""

import concurrent.futures
import hashlib
import os
import shutil
import sqlite3
import tempfile
import unittest

from chain_replay_ml.research_memory import (
    create_benchmark_run,
    get_benchmark_metrics,
    get_benchmark_run,
    get_model_benchmark_by_id,
    get_model_benchmarks_for_context,
    init_analysis_db,
    record_benchmark_metrics,
    record_model_benchmark,
    register_or_get_experiment,
)


class TestModelBenchmarksPersistence(unittest.TestCase):
    """Test suite verifying benchmark run creation, scorecard recording, and normalized metrics persistence."""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp(prefix="aruml_test_benchmarks_")
        init_analysis_db(self.tmp_dir)

        self.spec_trend = {
            "market": "NIFTY",
            "sampling_interval_sec": 3,
            "task_type": "DIRECTION_CLASSIFIER",
            "prediction_horizon": "5m",
            "regime_id": "R001",
            "regime_definition_hash": "def_hash_trend_v1",
            "dataset_snapshot_hash": "ds_hash_20260819",
            "features": ["adx_14", "rsi_14", "basis", "atm_iv_pctile"],
            "algorithm": "xgboost",
            "hyperparameters": {"max_depth": 6, "learning_rate": 0.05},
            "walk_forward_config": {"folds": 5, "window_mode": "expanding"},
            "random_seed": 42,
        }

        self.spec_side = {
            "market": "NIFTY",
            "sampling_interval_sec": 3,
            "task_type": "DIRECTION_CLASSIFIER",
            "prediction_horizon": "5m",
            "regime_id": "R002",
            "regime_definition_hash": "def_hash_side_v1",
            "dataset_snapshot_hash": "ds_hash_20260819",
            "features": ["adx_14", "bb_width", "iv_skew_25d"],
            "algorithm": "lightgbm",
            "hyperparameters": {"num_leaves": 31, "learning_rate": 0.03},
            "walk_forward_config": {"folds": 5, "window_mode": "expanding"},
            "random_seed": 42,
        }

        # Pre-register signatures in analysis.db
        _, self.rec_trend = register_or_get_experiment(self.tmp_dir, self.spec_trend, model_name="DIR_TREND_XGB_v1")
        _, self.rec_side = register_or_get_experiment(self.tmp_dir, self.spec_side, model_name="DIR_SIDE_LGB_v1")

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_create_benchmark_run(self):
        """1. Verify benchmark run record creation and field initialization."""
        run_id = create_benchmark_run(
            self.tmp_dir,
            context_key="NIFTY_3s_DIRECTION_CLASSIFIER_5m_R001",
            benchmark_scope="CONTEXT_LEADERBOARD",
            incumbent_champion_name="DIR_TREND_XGB_v0",
        )
        self.assertTrue(run_id.startswith("BM_"))

        run_doc = get_benchmark_run(self.tmp_dir, run_id)
        self.assertIsNotNone(run_doc)
        self.assertEqual(run_doc["context_key"], "NIFTY_3s_DIRECTION_CLASSIFIER_5m_R001")
        self.assertEqual(run_doc["benchmark_scope"], "CONTEXT_LEADERBOARD")
        self.assertEqual(run_doc["models_evaluated_count"], 0)
        self.assertEqual(run_doc["incumbent_champion_name"], "DIR_TREND_XGB_v0")

    def test_record_model_benchmark_with_granular_metrics(self):
        """2. Verify model benchmark scorecard and nested granular metrics persistence."""
        run_id = create_benchmark_run(self.tmp_dir, context_key="NIFTY_3s_DIRECTION_CLASSIFIER_5m_R001")

        granular = [
            {"metric_name": "fold_1_roc_auc", "metric_stage": "WALK_FORWARD_FOLD", "fold_index": 1, "metric_value": 0.68, "metric_type": "SCALAR_FLOAT"},
            {"metric_name": "fold_2_roc_auc", "metric_stage": "WALK_FORWARD_FOLD", "fold_index": 2, "metric_value": 0.72, "metric_type": "SCALAR_FLOAT"},
            {"metric_name": "fold_3_roc_auc", "metric_stage": "WALK_FORWARD_FOLD", "fold_index": 3, "metric_value": 0.70, "metric_type": "SCALAR_FLOAT"},
            {"metric_name": "test_brier_score", "metric_stage": "TEST", "fold_index": None, "metric_value": 0.185, "metric_type": "SCALAR_FLOAT"},
            {"metric_name": "inference_latency_us", "metric_stage": "TEST", "fold_index": None, "metric_value": 142.5, "metric_type": "DURATION_SEC"},
        ]

        bm_id = record_model_benchmark(
            self.tmp_dir,
            benchmark_run_id=run_id,
            signature_hash=self.rec_trend["signature_hash"],
            model_name="DIR_TREND_XGB_v1",
            context_key="NIFTY_3s_DIRECTION_CLASSIFIER_5m_R001",
            algorithm="xgboost",
            dataset_name="analysis_trend_3s_20260819.parquet",
            feature_count=4,
            primary_metric_name="roc_auc",
            primary_metric_value=0.70,
            secondary_metric_value=61.2,
            wf_folds_count=3,
            fold_metric_mean=0.70,
            fold_metric_std=0.02,
            fold_metric_min=0.68,
            fold_metric_max=0.72,
            worst_fold_drawdown=0.04,
            temporal_stability_score=0.95,
            brier_score=0.185,
            log_loss=0.552,
            expected_calibration_error=0.025,
            training_time_sec=42.5,
            inference_latency_us=142.5,
            model_size_bytes=2450000,
            robustness_score=86.5,
            rank_in_context=1,
            recommendation_status="CHAMPION_CANDIDATE",
            granular_metrics=granular,
        )
        self.assertGreater(bm_id, 0)

        # Verify benchmark_runs counter incremented
        run_doc = get_benchmark_run(self.tmp_dir, run_id)
        self.assertEqual(run_doc["models_evaluated_count"], 1)

        # Retrieve scorecard by ID
        bm = get_model_benchmark_by_id(self.tmp_dir, bm_id)
        self.assertIsNotNone(bm)
        self.assertEqual(bm["model_name"], "DIR_TREND_XGB_v1")
        self.assertEqual(bm["primary_metric_name"], "roc_auc")
        self.assertEqual(bm["primary_metric_value"], 0.70)
        self.assertEqual(bm["robustness_score"], 86.5)
        self.assertEqual(bm["recommendation_status"], "CHAMPION_CANDIDATE")

        # Retrieve granular metrics
        metrics = get_benchmark_metrics(self.tmp_dir, bm_id)
        self.assertEqual(len(metrics), 5)
        metric_names = [m["metric_name"] for m in metrics]
        self.assertIn("fold_1_roc_auc", metric_names)
        self.assertIn("test_brier_score", metric_names)
        self.assertIn("inference_latency_us", metric_names)

    def test_batch_record_benchmark_metrics(self):
        """3. Verify standalone record_benchmark_metrics batch insertion."""
        run_id = create_benchmark_run(self.tmp_dir, context_key="NIFTY_3s_DIRECTION_CLASSIFIER_5m_R001")
        bm_id = record_model_benchmark(
            self.tmp_dir,
            benchmark_run_id=run_id,
            signature_hash=self.rec_trend["signature_hash"],
            model_name="DIR_TREND_XGB_v1",
            context_key="NIFTY_3s_DIRECTION_CLASSIFIER_5m_R001",
            algorithm="xgboost",
            dataset_name="analysis_trend_3s.parquet",
            feature_count=4,
            primary_metric_name="roc_auc",
            primary_metric_value=0.70,
        )

        extra_metrics = [
            {"metric_name": "val_precision", "metric_stage": "VAL", "metric_value": 0.65, "metric_type": "PERCENTAGE"},
            {"metric_name": "val_recall", "metric_stage": "VAL", "metric_value": 0.58, "metric_type": "PERCENTAGE"},
        ]
        inserted = record_benchmark_metrics(self.tmp_dir, benchmark_id=bm_id, metrics=extra_metrics)
        self.assertEqual(inserted, 2)

        stored = get_benchmark_metrics(self.tmp_dir, bm_id)
        self.assertEqual(len(stored), 2)

    def test_multi_model_context_leaderboard_retrieval(self):
        """4. Verify multiple models benchmarked in the same context sort properly by robustness score."""
        run_id = create_benchmark_run(self.tmp_dir, context_key="NIFTY_3s_DIRECTION_CLASSIFIER_5m_R001")

        # Record Model A
        record_model_benchmark(
            self.tmp_dir,
            benchmark_run_id=run_id,
            signature_hash=self.rec_trend["signature_hash"],
            model_name="MODEL_A_XGB",
            context_key="NIFTY_3s_DIRECTION_CLASSIFIER_5m_R001",
            algorithm="xgboost",
            dataset_name="analysis_trend_3s.parquet",
            feature_count=4,
            primary_metric_name="roc_auc",
            primary_metric_value=0.68,
            robustness_score=78.0,
            rank_in_context=2,
        )

        # Record Model B (Higher robustness score)
        record_model_benchmark(
            self.tmp_dir,
            benchmark_run_id=run_id,
            signature_hash=self.rec_trend["signature_hash"],
            model_name="MODEL_B_CATBOOST",
            context_key="NIFTY_3s_DIRECTION_CLASSIFIER_5m_R001",
            algorithm="catboost",
            dataset_name="analysis_trend_3s.parquet",
            feature_count=4,
            primary_metric_name="roc_auc",
            primary_metric_value=0.72,
            robustness_score=88.5,
            rank_in_context=1,
        )

        benchmarks = get_model_benchmarks_for_context(self.tmp_dir, "NIFTY_3s_DIRECTION_CLASSIFIER_5m_R001")
        self.assertEqual(len(benchmarks), 2)
        # Verify sorted descending by robustness_score
        self.assertEqual(benchmarks[0]["model_name"], "MODEL_B_CATBOOST")
        self.assertEqual(benchmarks[0]["robustness_score"], 88.5)
        self.assertEqual(benchmarks[1]["model_name"], "MODEL_A_XGB")
        self.assertEqual(benchmarks[1]["robustness_score"], 78.0)

    def test_trend_and_sideways_regime_isolation(self):
        """5. Verify Trend (R001) benchmarks never leak into Sideways (R002) queries."""
        run_trend = create_benchmark_run(self.tmp_dir, context_key="NIFTY_3s_DIRECTION_CLASSIFIER_5m_R001")
        run_side = create_benchmark_run(self.tmp_dir, context_key="NIFTY_3s_DIRECTION_CLASSIFIER_5m_R002")

        record_model_benchmark(
            self.tmp_dir,
            benchmark_run_id=run_trend,
            signature_hash=self.rec_trend["signature_hash"],
            model_name="TREND_CHAMPION",
            context_key="NIFTY_3s_DIRECTION_CLASSIFIER_5m_R001",
            algorithm="xgboost",
            dataset_name="analysis_trend_3s.parquet",
            feature_count=4,
            primary_metric_name="roc_auc",
            primary_metric_value=0.75,
            robustness_score=90.0,
        )

        record_model_benchmark(
            self.tmp_dir,
            benchmark_run_id=run_side,
            signature_hash=self.rec_side["signature_hash"],
            model_name="SIDE_CHAMPION",
            context_key="NIFTY_3s_DIRECTION_CLASSIFIER_5m_R002",
            algorithm="lightgbm",
            dataset_name="analysis_side_3s.parquet",
            feature_count=3,
            primary_metric_name="roc_auc",
            primary_metric_value=0.71,
            robustness_score=85.0,
        )

        trend_list = get_model_benchmarks_for_context(self.tmp_dir, "NIFTY_3s_DIRECTION_CLASSIFIER_5m_R001")
        side_list = get_model_benchmarks_for_context(self.tmp_dir, "NIFTY_3s_DIRECTION_CLASSIFIER_5m_R002")

        self.assertEqual(len(trend_list), 1)
        self.assertEqual(trend_list[0]["model_name"], "TREND_CHAMPION")

        self.assertEqual(len(side_list), 1)
        self.assertEqual(side_list[0]["model_name"], "SIDE_CHAMPION")

    def test_foreign_key_signature_integrity(self):
        """6. Verify foreign key integrity rejects benchmark referencing invalid signature hash."""
        run_id = create_benchmark_run(self.tmp_dir, context_key="NIFTY_3s_DIRECTION_CLASSIFIER_5m_R001")

        with self.assertRaises(sqlite3.IntegrityError):
            record_model_benchmark(
                self.tmp_dir,
                benchmark_run_id=run_id,
                signature_hash="invalid_non_existent_signature_hash",
                model_name="BAD_MODEL",
                context_key="NIFTY_3s_DIRECTION_CLASSIFIER_5m_R001",
                algorithm="xgboost",
                dataset_name="analysis.parquet",
                feature_count=4,
                primary_metric_name="roc_auc",
                primary_metric_value=0.50,
            )

    def test_concurrent_benchmark_recording(self):
        """7. Verify concurrent workers can write benchmark records safely without lock contention."""
        run_id = create_benchmark_run(self.tmp_dir, context_key="NIFTY_3s_DIRECTION_CLASSIFIER_5m_R001")

        def worker_write(idx):
            return record_model_benchmark(
                self.tmp_dir,
                benchmark_run_id=run_id,
                signature_hash=self.rec_trend["signature_hash"],
                model_name=f"WORKER_BENCHMARK_MODEL_{idx}",
                context_key="NIFTY_3s_DIRECTION_CLASSIFIER_5m_R001",
                algorithm="xgboost",
                dataset_name="analysis.parquet",
                feature_count=4,
                primary_metric_name="roc_auc",
                primary_metric_value=0.60 + (idx * 0.01),
                robustness_score=70.0 + idx,
            )

        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            bm_ids = list(executor.map(worker_write, range(5)))

        self.assertEqual(len(bm_ids), 5)
        self.assertTrue(all(bid > 0 for bid in bm_ids))

        run_doc = get_benchmark_run(self.tmp_dir, run_id)
        self.assertEqual(run_doc["models_evaluated_count"], 5)

    def test_evidence_db_immutability(self):
        """8. Verify benchmark operations do NOT touch or mutate feature_recommendation_evidence.db."""
        ev_path = "apps/feature_recommendation_evidence.db"
        self.assertTrue(os.path.isfile(ev_path))
        with open(ev_path, "rb") as fh:
            sha_before = hashlib.sha256(fh.read()).hexdigest()

        run_id = create_benchmark_run(self.tmp_dir, context_key="NIFTY_3s_DIRECTION_CLASSIFIER_5m_R001")
        record_model_benchmark(
            self.tmp_dir,
            benchmark_run_id=run_id,
            signature_hash=self.rec_trend["signature_hash"],
            model_name="IMMUTABILITY_TEST_MODEL",
            context_key="NIFTY_3s_DIRECTION_CLASSIFIER_5m_R001",
            algorithm="xgboost",
            dataset_name="analysis.parquet",
            feature_count=4,
            primary_metric_name="roc_auc",
            primary_metric_value=0.70,
        )

        with open(ev_path, "rb") as fh:
            sha_after = hashlib.sha256(fh.read()).hexdigest()

        self.assertEqual(sha_before, sha_after)
        self.assertEqual(sha_after, "6f91afca94ec87a1210d8f4bcef356b9c16a6ef5a488268f41c5b1b81431ade2")


if __name__ == "__main__":
    unittest.main()

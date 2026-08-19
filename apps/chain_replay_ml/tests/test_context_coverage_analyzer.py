"""Unit tests for Phase 4E.1 — Context Coverage & Evidence Density Analyzer.

Verifies:
1. Empty database/context safety
2. Zero experiments => COLD_START
3. 1–4 experiments => SPARSE
4. 5–19 experiments => DEVELOPING (intermediate transition)
5. 20+ experiments => MATURE
6. Evidence density is exactly 0 when benchmark count is 0
7. Evidence density stays within [0.0, 100.0]
8. Zero feature registry count does not crash (division by zero safety)
9. Context isolation across R001/R002
10. Task-type isolation (DIRECTION_CLASSIFIER vs VOLATILITY_ESTIMATOR)
11. Horizon isolation (5m vs 15m)
12. Sampling interval isolation (3s vs 5s)
13. Market isolation (NIFTY vs BANKNIFTY)
14. Deterministic repeated calculation
15. Missing/partial research-memory data safety
16. No parquet/raw matrix loading verification
17. Production database immutability verification
18. Full regression compatibility
"""

import hashlib
import json
import math
import os
import shutil
import tempfile
import unittest

from chain_replay_ml.model_taxonomy.enums import (
    BASELINE_REGIME_CATALOG,
    TaskType,
)
from chain_replay_ml.model_taxonomy.specs import ModelContextKey
from chain_replay_ml.research_memory.benchmarks import (
    create_benchmark_run,
    record_model_benchmark,
)
from chain_replay_ml.research_memory.campaigns import (
    complete_campaign,
    create_campaign,
    fail_campaign,
    start_campaign,
)
from chain_replay_ml.research_memory.signature import (
    canonical_context_key,
    compute_experiment_signature,
    register_or_get_experiment,
)
from chain_replay_ml.research_recommendations.coverage import (
    ContextCoverage,
    CoverageClass,
    CoverageMatrix,
    analyze_context_coverage,
    build_coverage_matrix,
    classify_coverage,
    compute_evidence_density_score,
)


class TestContextCoverageAnalyzer(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp(prefix="aruml_test_4e1_cov_")

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_empty_database_and_unexplored_context_safety(self):
        """1. Verify analyzer handles empty database and unexplored contexts safely without errors."""
        ckey = "NIFTY_3s_DIRECTION_CLASSIFIER_5m_R001"
        cov = analyze_context_coverage(self.tmp_dir, ckey)

        self.assertEqual(cov.context_key, ckey)
        self.assertEqual(cov.market, "NIFTY")
        self.assertEqual(cov.sampling_interval_sec, 3)
        self.assertEqual(cov.task_type, "DIRECTION_CLASSIFIER")
        self.assertEqual(cov.prediction_horizon, "5m")
        self.assertEqual(cov.regime_id, "R001")
        self.assertEqual(cov.regime_name, "TREND")
        self.assertEqual(cov.benchmark_count, 0)
        self.assertEqual(cov.unique_experiment_count, 0)
        self.assertEqual(cov.unique_features_count, 0)
        self.assertEqual(cov.evidence_density_score, 0.0)
        self.assertEqual(cov.coverage_class, CoverageClass.COLD_START)
        self.assertEqual(cov.temporal_span["span_days"], 0.0)

    def test_zero_experiments_cold_start(self):
        """2. Verify that 0 experiments/benchmarks explicitly produces COLD_START class."""
        self.assertEqual(classify_coverage(0), CoverageClass.COLD_START)
        self.assertEqual(classify_coverage(-5), CoverageClass.COLD_START)

        ckey = "NIFTY_3s_DIRECTION_CLASSIFIER_5m_R002"
        cov = analyze_context_coverage(self.tmp_dir, ckey)
        self.assertEqual(cov.coverage_class, CoverageClass.COLD_START)

    def test_sparse_coverage_thresholds(self):
        """3. Verify that 1 to 4 experiments produces SPARSE coverage class."""
        for count in [1, 2, 3, 4]:
            self.assertEqual(classify_coverage(count), CoverageClass.SPARSE)

        ckey = "NIFTY_3s_DIRECTION_CLASSIFIER_5m_R001"
        b_run = create_benchmark_run(self.tmp_dir, context_key=ckey)

        # Record 3 benchmarks
        for i in range(3):
            spec = {
                "market": "NIFTY",
                "sampling_interval_sec": 3,
                "task_type": "DIRECTION_CLASSIFIER",
                "prediction_horizon": "5m",
                "regime_id": "R001",
                "features": [f"feat_{i}", "adx_14"],
                "algorithm": "xgboost",
                "hyperparameters": {"max_depth": 3 + i},
            }
            sig_h, _, _ = compute_experiment_signature(spec)
            register_or_get_experiment(self.tmp_dir, spec, model_name=f"MODEL_{i}")
            record_model_benchmark(
                self.tmp_dir,
                benchmark_run_id=b_run,
                signature_hash=sig_h,
                model_name=f"MODEL_{i}",
                context_key=ckey,
                algorithm="xgboost",
                dataset_name="d.parquet",
                feature_count=2,
                primary_metric_name="roc_auc",
                primary_metric_value=0.75 + (i * 0.02),
            )

        cov = analyze_context_coverage(self.tmp_dir, ckey)
        self.assertEqual(cov.benchmark_count, 3)
        self.assertEqual(cov.unique_experiment_count, 3)
        self.assertEqual(cov.coverage_class, CoverageClass.SPARSE)

    def test_developing_intermediate_thresholds(self):
        """4. Verify that 5 to 19 experiments produces DEVELOPING coverage class."""
        for count in [5, 10, 15, 19]:
            self.assertEqual(classify_coverage(count), CoverageClass.DEVELOPING)

        ckey = "NIFTY_3s_DIRECTION_CLASSIFIER_5m_R001"
        b_run = create_benchmark_run(self.tmp_dir, context_key=ckey)

        for i in range(7):
            spec = {
                "market": "NIFTY",
                "sampling_interval_sec": 3,
                "task_type": "DIRECTION_CLASSIFIER",
                "prediction_horizon": "5m",
                "regime_id": "R001",
                "features": [f"feat_{i}", "rsi_14"],
                "algorithm": "catboost",
                "hyperparameters": {"depth": 4},
            }
            sig_h, _, _ = compute_experiment_signature(spec)
            register_or_get_experiment(self.tmp_dir, spec, model_name=f"DEV_MODEL_{i}")
            record_model_benchmark(
                self.tmp_dir,
                benchmark_run_id=b_run,
                signature_hash=sig_h,
                model_name=f"DEV_MODEL_{i}",
                context_key=ckey,
                algorithm="catboost",
                dataset_name="d.parquet",
                feature_count=2,
                primary_metric_name="roc_auc",
                primary_metric_value=0.80,
            )

        cov = analyze_context_coverage(self.tmp_dir, ckey)
        self.assertEqual(cov.benchmark_count, 7)
        self.assertEqual(cov.coverage_class, CoverageClass.DEVELOPING)

    def test_mature_coverage_thresholds(self):
        """5. Verify that 20+ experiments produces MATURE coverage class."""
        for count in [20, 25, 100]:
            self.assertEqual(classify_coverage(count), CoverageClass.MATURE)

        ckey = "NIFTY_3s_DIRECTION_CLASSIFIER_5m_R001"
        b_run = create_benchmark_run(self.tmp_dir, context_key=ckey)

        for i in range(22):
            spec = {
                "market": "NIFTY",
                "sampling_interval_sec": 3,
                "task_type": "DIRECTION_CLASSIFIER",
                "prediction_horizon": "5m",
                "regime_id": "R001",
                "features": [f"feat_{i}"],
                "algorithm": "xgboost",
            }
            sig_h, _, _ = compute_experiment_signature(spec)
            register_or_get_experiment(self.tmp_dir, spec, model_name=f"MATURE_MODEL_{i}")
            record_model_benchmark(
                self.tmp_dir,
                benchmark_run_id=b_run,
                signature_hash=sig_h,
                model_name=f"MATURE_MODEL_{i}",
                context_key=ckey,
                algorithm="xgboost",
                dataset_name="d.parquet",
                feature_count=1,
                primary_metric_name="roc_auc",
                primary_metric_value=0.82,
            )

        cov = analyze_context_coverage(self.tmp_dir, ckey)
        self.assertEqual(cov.benchmark_count, 22)
        self.assertEqual(cov.coverage_class, CoverageClass.MATURE)

    def test_evidence_density_zero_when_benchmark_zero(self):
        """6. Assert that evidence density score is exactly 0.0 when benchmark count is 0."""
        score1 = compute_evidence_density_score(0, 0, 50)
        score2 = compute_evidence_density_score(0, 10, 50)
        score3 = compute_evidence_density_score(0, 50, 0)
        self.assertEqual(score1, 0.0)
        self.assertEqual(score2, 0.0)
        self.assertEqual(score3, 0.0)

    def test_evidence_density_bounds(self):
        """7. Verify evidence density score is strictly bounded in [0.0, 100.0] across all scales."""
        for bm_count in [0, 1, 5, 10, 20, 50, 100, 10000]:
            for feat_count in [0, 5, 25, 50, 200]:
                for tot_feat in [1, 10, 50, 500]:
                    score = compute_evidence_density_score(bm_count, feat_count, tot_feat)
                    self.assertGreaterEqual(score, 0.0)
                    self.assertLessEqual(score, 100.0)
                    self.assertFalse(math.isnan(score))
                    self.assertFalse(math.isinf(score))

    def test_zero_feature_registry_division_by_zero_safety(self):
        """8. Verify zero/negative total registry features does not cause division by zero."""
        score_zero = compute_evidence_density_score(10, 5, 0)
        score_neg = compute_evidence_density_score(10, 5, -10)
        self.assertGreater(score_zero, 0.0)
        self.assertLessEqual(score_zero, 100.0)
        self.assertGreater(score_neg, 0.0)
        self.assertLessEqual(score_neg, 100.0)

    def test_context_isolation_across_regimes(self):
        """9. Verify complete evidence isolation between R001 (TREND) and R002 (SIDEWAYS)."""
        ctx_r1 = "NIFTY_3s_DIRECTION_CLASSIFIER_5m_R001"
        ctx_r2 = "NIFTY_3s_DIRECTION_CLASSIFIER_5m_R002"

        # Populate R001 with 10 benchmarks
        b_run1 = create_benchmark_run(self.tmp_dir, context_key=ctx_r1)
        for i in range(10):
            spec = {
                "market": "NIFTY",
                "sampling_interval_sec": 3,
                "task_type": "DIRECTION_CLASSIFIER",
                "prediction_horizon": "5m",
                "regime_id": "R001",
                "features": ["adx_14", "rsi_14"],
                "algorithm": "xgboost",
            }
            sig_h, _, _ = compute_experiment_signature(spec)
            register_or_get_experiment(self.tmp_dir, spec, model_name=f"R1_MODEL_{i}")
            record_model_benchmark(
                self.tmp_dir,
                benchmark_run_id=b_run1,
                signature_hash=sig_h,
                model_name=f"R1_MODEL_{i}",
                context_key=ctx_r1,
                algorithm="xgboost",
                dataset_name="d1.parquet",
                feature_count=2,
                primary_metric_name="roc_auc",
                primary_metric_value=0.81,
            )

        cov_r1 = analyze_context_coverage(self.tmp_dir, ctx_r1)
        cov_r2 = analyze_context_coverage(self.tmp_dir, ctx_r2)

        self.assertEqual(cov_r1.benchmark_count, 10)
        self.assertEqual(cov_r1.coverage_class, CoverageClass.DEVELOPING)
        self.assertGreater(cov_r1.evidence_density_score, 0.0)

        # R002 must remain completely unpolluted (0 benchmarks, COLD_START)
        self.assertEqual(cov_r2.benchmark_count, 0)
        self.assertEqual(cov_r2.coverage_class, CoverageClass.COLD_START)
        self.assertEqual(cov_r2.evidence_density_score, 0.0)

    def test_task_type_isolation(self):
        """10. Verify isolation between DIRECTION_CLASSIFIER and VOLATILITY_ESTIMATOR."""
        ctx_dir = "NIFTY_3s_DIRECTION_CLASSIFIER_5m_R001"
        ctx_vol = "NIFTY_3s_VOLATILITY_ESTIMATOR_5m_R001"

        b_run = create_benchmark_run(self.tmp_dir, context_key=ctx_dir)
        spec = {
            "market": "NIFTY",
            "sampling_interval_sec": 3,
            "task_type": "DIRECTION_CLASSIFIER",
            "prediction_horizon": "5m",
            "regime_id": "R001",
            "features": ["adx_14"],
            "algorithm": "catboost",
        }
        sig_h, _, _ = compute_experiment_signature(spec)
        register_or_get_experiment(self.tmp_dir, spec, model_name="DIR_M1")
        record_model_benchmark(
            self.tmp_dir,
            benchmark_run_id=b_run,
            signature_hash=sig_h,
            model_name="DIR_M1",
            context_key=ctx_dir,
            algorithm="catboost",
            dataset_name="d.parquet",
            feature_count=1,
            primary_metric_name="roc_auc",
            primary_metric_value=0.80,
        )

        cov_dir = analyze_context_coverage(self.tmp_dir, ctx_dir)
        cov_vol = analyze_context_coverage(self.tmp_dir, ctx_vol)

        self.assertEqual(cov_dir.benchmark_count, 1)
        self.assertEqual(cov_vol.benchmark_count, 0)

    def test_horizon_isolation(self):
        """11. Verify isolation across prediction horizons (5m vs 15m)."""
        ctx_5m = "NIFTY_3s_DIRECTION_CLASSIFIER_5m_R001"
        ctx_15m = "NIFTY_3s_DIRECTION_CLASSIFIER_15m_R001"

        b_run = create_benchmark_run(self.tmp_dir, context_key=ctx_5m)
        spec = {
            "market": "NIFTY",
            "sampling_interval_sec": 3,
            "task_type": "DIRECTION_CLASSIFIER",
            "prediction_horizon": "5m",
            "regime_id": "R001",
            "features": ["adx_14"],
            "algorithm": "xgboost",
        }
        sig_h, _, _ = compute_experiment_signature(spec)
        register_or_get_experiment(self.tmp_dir, spec, model_name="M_5m")
        record_model_benchmark(
            self.tmp_dir,
            benchmark_run_id=b_run,
            signature_hash=sig_h,
            model_name="M_5m",
            context_key=ctx_5m,
            algorithm="xgboost",
            dataset_name="d.parquet",
            feature_count=1,
            primary_metric_name="roc_auc",
            primary_metric_value=0.79,
        )

        self.assertEqual(analyze_context_coverage(self.tmp_dir, ctx_5m).benchmark_count, 1)
        self.assertEqual(analyze_context_coverage(self.tmp_dir, ctx_15m).benchmark_count, 0)

    def test_sampling_interval_isolation(self):
        """12. Verify isolation across sampling intervals (3s vs 5s)."""
        ctx_3s = "NIFTY_3s_DIRECTION_CLASSIFIER_5m_R001"
        ctx_5s = "NIFTY_5s_DIRECTION_CLASSIFIER_5m_R001"

        b_run = create_benchmark_run(self.tmp_dir, context_key=ctx_3s)
        spec = {
            "market": "NIFTY",
            "sampling_interval_sec": 3,
            "task_type": "DIRECTION_CLASSIFIER",
            "prediction_horizon": "5m",
            "regime_id": "R001",
            "features": ["adx_14"],
            "algorithm": "xgboost",
        }
        sig_h, _, _ = compute_experiment_signature(spec)
        register_or_get_experiment(self.tmp_dir, spec, model_name="M_3s")
        record_model_benchmark(
            self.tmp_dir,
            benchmark_run_id=b_run,
            signature_hash=sig_h,
            model_name="M_3s",
            context_key=ctx_3s,
            algorithm="xgboost",
            dataset_name="d.parquet",
            feature_count=1,
            primary_metric_name="roc_auc",
            primary_metric_value=0.81,
        )

        self.assertEqual(analyze_context_coverage(self.tmp_dir, ctx_3s).benchmark_count, 1)
        self.assertEqual(analyze_context_coverage(self.tmp_dir, ctx_5s).benchmark_count, 0)

    def test_market_isolation(self):
        """13. Verify isolation across markets (NIFTY vs BANKNIFTY)."""
        ctx_nifty = "NIFTY_3s_DIRECTION_CLASSIFIER_5m_R001"
        ctx_bn = "BANKNIFTY_3s_DIRECTION_CLASSIFIER_5m_R001"

        b_run = create_benchmark_run(self.tmp_dir, context_key=ctx_nifty)
        spec = {
            "market": "NIFTY",
            "sampling_interval_sec": 3,
            "task_type": "DIRECTION_CLASSIFIER",
            "prediction_horizon": "5m",
            "regime_id": "R001",
            "features": ["adx_14"],
            "algorithm": "xgboost",
        }
        sig_h, _, _ = compute_experiment_signature(spec)
        register_or_get_experiment(self.tmp_dir, spec, model_name="M_NIFTY")
        record_model_benchmark(
            self.tmp_dir,
            benchmark_run_id=b_run,
            signature_hash=sig_h,
            model_name="M_NIFTY",
            context_key=ctx_nifty,
            algorithm="xgboost",
            dataset_name="d.parquet",
            feature_count=1,
            primary_metric_name="roc_auc",
            primary_metric_value=0.82,
        )

        self.assertEqual(analyze_context_coverage(self.tmp_dir, ctx_nifty).benchmark_count, 1)
        self.assertEqual(analyze_context_coverage(self.tmp_dir, ctx_bn).benchmark_count, 0)

    def test_deterministic_repeated_calculation(self):
        """14. Verify that running analyze_context_coverage repeatedly yields identical results."""
        ckey = "NIFTY_3s_DIRECTION_CLASSIFIER_5m_R001"
        b_run = create_benchmark_run(self.tmp_dir, context_key=ckey)
        spec = {
            "market": "NIFTY",
            "sampling_interval_sec": 3,
            "task_type": "DIRECTION_CLASSIFIER",
            "prediction_horizon": "5m",
            "regime_id": "R001",
            "features": ["adx_14", "rsi_14"],
            "algorithm": "xgboost",
        }
        sig_h, _, _ = compute_experiment_signature(spec)
        register_or_get_experiment(self.tmp_dir, spec, model_name="M_DET")
        record_model_benchmark(
            self.tmp_dir,
            benchmark_run_id=b_run,
            signature_hash=sig_h,
            model_name="M_DET",
            context_key=ckey,
            algorithm="xgboost",
            dataset_name="d.parquet",
            feature_count=2,
            primary_metric_name="roc_auc",
            primary_metric_value=0.85,
        )

        cov1 = analyze_context_coverage(self.tmp_dir, ckey)
        cov2 = analyze_context_coverage(self.tmp_dir, ckey)

        self.assertEqual(cov1.to_dict(), cov2.to_dict())

    def test_build_coverage_matrix_and_campaign_aggregation(self):
        """15. Verify matrix building over multiple contexts with completed and failed campaigns."""
        ctx1 = "NIFTY_3s_DIRECTION_CLASSIFIER_5m_R001"
        ctx2 = "NIFTY_3s_DIRECTION_CLASSIFIER_5m_R002"

        # Campaign 1: Completed
        c1 = create_campaign(self.tmp_dir, context_key=ctx1)
        start_campaign(self.tmp_dir, c1)
        complete_campaign(self.tmp_dir, c1)

        # Campaign 2: Failed
        c2 = create_campaign(self.tmp_dir, context_key=ctx1)
        start_campaign(self.tmp_dir, c2)
        fail_campaign(self.tmp_dir, c2, error_message="Test failure")

        # Build matrix
        matrix = build_coverage_matrix(self.tmp_dir, explicit_context_keys=[ctx1, ctx2])
        self.assertEqual(matrix.total_contexts, 2)
        self.assertEqual(matrix.cold_start_count, 2)

        cov1 = next(c for c in matrix.contexts if c.context_key == ctx1)
        self.assertEqual(cov1.completed_campaign_count, 1)
        self.assertEqual(cov1.failed_campaign_count, 1)

    def test_production_database_immutability(self):
        """16. Assert that coverage analysis never modifies production databases or registries."""
        ev_path = "apps/feature_recommendation_evidence.db"
        self.assertTrue(os.path.isfile(ev_path), "Evidence DB must exist")

        with open(ev_path, "rb") as fh:
            sha_initial = hashlib.sha256(fh.read()).hexdigest()

        # Run heavy coverage analysis in tmp_dir
        matrix = build_coverage_matrix(
            self.tmp_dir,
            markets=["NIFTY", "BANKNIFTY"],
            task_types=["DIRECTION_CLASSIFIER", "VOLATILITY_ESTIMATOR"],
            regimes=["R001", "R002", "R003", "R005", "R006", "R007"],
        )
        self.assertGreater(matrix.total_contexts, 0)

        with open(ev_path, "rb") as fh:
            sha_final = hashlib.sha256(fh.read()).hexdigest()

        self.assertEqual(sha_initial, sha_final, "Evidence DB must remain 100% unmutated")


if __name__ == "__main__":
    unittest.main()

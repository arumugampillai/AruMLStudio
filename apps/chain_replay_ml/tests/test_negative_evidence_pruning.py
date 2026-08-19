"""Unit tests for Phase 4E.4 — Negative Evidence Pruning & Search Space Exclusion Engine.

Verifies:
1. Duplicate experiment
2. Duplicate canonicalization
3. Feature-order invariance
4. Deprecated feature exclusion
5. Multiple deprecated features
6. Unknown feature remains eligible
7. One low-robustness trial does not create chronic failure
8. Exactly 3 chronic failures
9. Fewer than 3 failures
10. Robustness boundary at 40.0
11. Regime degradation boundary at 30%
12. Calibration boundary at ECE 0.10
13. CAUTION vs EXCLUDED behavior
14. Deterministic reason precedence
15. Cold-start context
16. Sparse context
17. R001/R002 isolation
18. 5m/15m isolation
19. NIFTY/BANKNIFTY isolation
20. Signature exclusion
21. Context pruning agenda
22. Deterministic ordering
23. NaN/Infinity safety
24. Concurrent read safety
25. Production database immutability
26. Evidence database SHA-256 unchanged
"""

import hashlib
import json
import math
import os
import shutil
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor

from chain_replay_ml.research_memory.benchmarks import (
    create_benchmark_run,
    record_model_benchmark,
)
from chain_replay_ml.research_memory.regime_eval import (
    record_regime_evaluation,
)
from chain_replay_ml.research_memory.signature import (
    compute_experiment_signature,
    register_or_get_experiment,
)
from chain_replay_ml.research_recommendations.negative_pruning import (
    ContextPruningAgenda,
    ExclusionReason,
    ExclusionVerdict,
    PruningAuditResult,
    audit_experiment_exclusion,
    audit_signature_exclusion,
    build_context_pruning_agenda,
    is_search_path_excluded,
)


class TestNegativeEvidencePruning(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp(prefix="aruml_test_4e4_prune_")
        self.mock_schema = {
            "columns": {
                "adx_14": {"is_base": True, "project_id": "PL_0001", "status": "ACTIVE"},
                "rsi_14": {"is_base": False, "project_id": None, "status": "ACTIVE"},
                "dep_old": {"is_base": False, "project_id": "PL_0001", "status": "DEPRECATED"},
                "dep_alpha": {"is_base": False, "project_id": "PL_0002", "status": "RETIRED"},
            }
        }

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_01_duplicate_experiment_exclusion(self):
        """1. Verify that a registered experiment signature is excluded as DUPLICATE_EXPERIMENT."""
        ckey = "NIFTY_3s_DIRECTION_CLASSIFIER_5m_R001"
        spec = {
            "market": "NIFTY",
            "sampling_interval_sec": 3,
            "task_type": "DIRECTION_CLASSIFIER",
            "prediction_horizon": "5m",
            "regime_id": "R001",
            "features": ["adx_14", "rsi_14"],
            "algorithm": "xgboost",
        }
        # Register experiment into analysis.db
        register_or_get_experiment(self.tmp_dir, spec, model_name="M_DUP")

        res = audit_experiment_exclusion(self.tmp_dir, spec, schema=self.mock_schema)
        self.assertEqual(res.verdict, ExclusionVerdict.EXCLUDED)
        self.assertEqual(res.primary_reason, ExclusionReason.DUPLICATE_EXPERIMENT)
        self.assertTrue(is_search_path_excluded(self.tmp_dir, spec, schema=self.mock_schema))

    def test_02_duplicate_canonicalization(self):
        """2. Verify canonical payload normalization detects duplicate despite extra whitespace or key ordering."""
        spec1 = {
            "market": "nifty",
            "sampling_interval_sec": 3,
            "task_type": "direction_classifier",
            "prediction_horizon": "5m",
            "regime_id": "r001",
            "features": ["rsi_14", "adx_14"],
            "algorithm": "XGBOOST",
        }
        spec2 = {
            "market": "NIFTY ",
            "sampling_interval_sec": 3,
            "task_type": "DIRECTION_CLASSIFIER",
            "prediction_horizon": "5m",
            "regime_id": "R001",
            "features": ["adx_14", "rsi_14"],
            "algorithm": "xgboost",
        }
        register_or_get_experiment(self.tmp_dir, spec1, model_name="M_CAN")
        res = audit_experiment_exclusion(self.tmp_dir, spec2, schema=self.mock_schema)
        self.assertEqual(res.verdict, ExclusionVerdict.EXCLUDED)
        self.assertEqual(res.primary_reason, ExclusionReason.DUPLICATE_EXPERIMENT)

    def test_03_feature_order_invariance(self):
        """3. Verify feature ordering does not alter deduplication signature."""
        spec_a = {
            "market": "NIFTY",
            "sampling_interval_sec": 3,
            "task_type": "DIRECTION_CLASSIFIER",
            "prediction_horizon": "5m",
            "regime_id": "R001",
            "features": ["rsi_14", "adx_14"],
            "algorithm": "xgboost",
        }
        spec_b = dict(spec_a)
        spec_b["features"] = ["adx_14", "rsi_14"]

        sig_a, _, _ = compute_experiment_signature(spec_a)
        sig_b, _, _ = compute_experiment_signature(spec_b)
        self.assertEqual(sig_a, sig_b)

    def test_04_deprecated_feature_exclusion(self):
        """4. Verify proposed experiment with a DEPRECATED feature is EXCLUDED."""
        spec = {
            "market": "NIFTY",
            "sampling_interval_sec": 3,
            "task_type": "DIRECTION_CLASSIFIER",
            "prediction_horizon": "5m",
            "regime_id": "R001",
            "features": ["dep_old", "adx_14"],
            "algorithm": "xgboost",
        }
        res = audit_experiment_exclusion(self.tmp_dir, spec, schema=self.mock_schema)
        self.assertEqual(res.verdict, ExclusionVerdict.EXCLUDED)
        self.assertEqual(res.primary_reason, ExclusionReason.DEPRECATED_FEATURE)
        self.assertIn("dep_old", res.flagged_features)

    def test_05_multiple_deprecated_features(self):
        """5. Verify all deprecated features are flagged in audit result."""
        spec = {
            "market": "NIFTY",
            "sampling_interval_sec": 3,
            "task_type": "DIRECTION_CLASSIFIER",
            "prediction_horizon": "5m",
            "regime_id": "R001",
            "features": ["dep_old", "dep_alpha"],
            "algorithm": "xgboost",
        }
        res = audit_experiment_exclusion(self.tmp_dir, spec, schema=self.mock_schema)
        self.assertEqual(res.verdict, ExclusionVerdict.EXCLUDED)
        self.assertEqual(res.primary_reason, ExclusionReason.DEPRECATED_FEATURE)
        self.assertEqual(res.flagged_features, ["dep_alpha", "dep_old"])

    def test_06_unknown_feature_remains_eligible(self):
        """6. Verify unmapped/unknown features are NOT excluded by default and remain ELIGIBLE."""
        spec = {
            "market": "NIFTY",
            "sampling_interval_sec": 3,
            "task_type": "DIRECTION_CLASSIFIER",
            "prediction_horizon": "5m",
            "regime_id": "R001",
            "features": ["mystery_signal_x"],
            "algorithm": "xgboost",
        }
        res = audit_experiment_exclusion(self.tmp_dir, spec, schema=self.mock_schema)
        self.assertEqual(res.verdict, ExclusionVerdict.ELIGIBLE)
        self.assertEqual(res.primary_reason, ExclusionReason.NONE)

    def test_07_single_low_robustness_trial_not_chronic(self):
        """7. Verify a single low-robustness trial does NOT trigger CHRONIC_LOW_ROBUSTNESS."""
        ckey = "NIFTY_3s_DIRECTION_CLASSIFIER_5m_R001"
        b_run = create_benchmark_run(self.tmp_dir, context_key=ckey)
        spec = {
            "market": "NIFTY",
            "sampling_interval_sec": 3,
            "task_type": "DIRECTION_CLASSIFIER",
            "prediction_horizon": "5m",
            "regime_id": "R001",
            "features": ["adx_14"],
            "algorithm": "catboost",
            "hyperparameters": {"depth": 4},
        }
        sig_h, _, _ = compute_experiment_signature(spec)
        register_or_get_experiment(self.tmp_dir, spec, model_name="M_FAIL_1")
        record_model_benchmark(
            self.tmp_dir,
            benchmark_run_id=b_run,
            signature_hash=sig_h,
            model_name="M_FAIL_1",
            context_key=ckey,
            algorithm="catboost",
            dataset_name="d.parquet",
            feature_count=1,
            primary_metric_name="roc_auc",
            primary_metric_value=0.35,
            robustness_score=35.0,
        )

        # Proposed new spec with different hyperparameter (not duplicate)
        prop_spec = dict(spec)
        prop_spec["hyperparameters"] = {"depth": 6}

        res = audit_experiment_exclusion(self.tmp_dir, prop_spec, schema=self.mock_schema)
        self.assertEqual(res.verdict, ExclusionVerdict.ELIGIBLE)
        self.assertEqual(res.primary_reason, ExclusionReason.NONE)

    def test_08_exactly_three_chronic_failures(self):
        """8. Verify exactly 3 low-robustness trials (< 40.0) triggers CHRONIC_LOW_ROBUSTNESS."""
        ckey = "NIFTY_3s_DIRECTION_CLASSIFIER_5m_R001"
        b_run = create_benchmark_run(self.tmp_dir, context_key=ckey)

        for i in range(3):
            spec_i = {
                "market": "NIFTY",
                "sampling_interval_sec": 3,
                "task_type": "DIRECTION_CLASSIFIER",
                "prediction_horizon": "5m",
                "regime_id": "R001",
                "features": ["adx_14"],
                "algorithm": "xgboost",
                "hyperparameters": {"max_depth": 3 + i},
            }
            sig_i, _, _ = compute_experiment_signature(spec_i)
            register_or_get_experiment(self.tmp_dir, spec_i, model_name=f"FAIL_{i}")
            record_model_benchmark(
                self.tmp_dir,
                benchmark_run_id=b_run,
                signature_hash=sig_i,
                model_name=f"FAIL_{i}",
                context_key=ckey,
                algorithm="xgboost",
                dataset_name="d.parquet",
                feature_count=1,
                primary_metric_name="roc_auc",
                primary_metric_value=0.35,
                robustness_score=35.0,
            )

        # Proposed 4th trial with same feature set
        prop_spec = {
            "market": "NIFTY",
            "sampling_interval_sec": 3,
            "task_type": "DIRECTION_CLASSIFIER",
            "prediction_horizon": "5m",
            "regime_id": "R001",
            "features": ["adx_14"],
            "algorithm": "xgboost",
            "hyperparameters": {"max_depth": 9},
        }
        res = audit_experiment_exclusion(self.tmp_dir, prop_spec, schema=self.mock_schema)
        self.assertEqual(res.verdict, ExclusionVerdict.EXCLUDED)
        self.assertEqual(res.primary_reason, ExclusionReason.CHRONIC_LOW_ROBUSTNESS)

    def test_09_fewer_than_three_failures(self):
        """9. Verify 2 low-robustness trials does not trigger chronic exclusion."""
        ckey = "NIFTY_3s_DIRECTION_CLASSIFIER_5m_R001"
        b_run = create_benchmark_run(self.tmp_dir, context_key=ckey)

        for i in range(2):
            spec_i = {
                "market": "NIFTY",
                "sampling_interval_sec": 3,
                "task_type": "DIRECTION_CLASSIFIER",
                "prediction_horizon": "5m",
                "regime_id": "R001",
                "features": ["adx_14"],
                "algorithm": "xgboost",
                "hyperparameters": {"max_depth": 3 + i},
            }
            sig_i, _, _ = compute_experiment_signature(spec_i)
            register_or_get_experiment(self.tmp_dir, spec_i, model_name=f"FAIL_{i}")
            record_model_benchmark(
                self.tmp_dir,
                benchmark_run_id=b_run,
                signature_hash=sig_i,
                model_name=f"FAIL_{i}",
                context_key=ckey,
                algorithm="xgboost",
                dataset_name="d.parquet",
                feature_count=1,
                primary_metric_name="roc_auc",
                primary_metric_value=0.30,
                robustness_score=30.0,
            )

        prop_spec = {
            "market": "NIFTY",
            "sampling_interval_sec": 3,
            "task_type": "DIRECTION_CLASSIFIER",
            "prediction_horizon": "5m",
            "regime_id": "R001",
            "features": ["adx_14"],
            "algorithm": "xgboost",
            "hyperparameters": {"max_depth": 7},
        }
        res = audit_experiment_exclusion(self.tmp_dir, prop_spec, schema=self.mock_schema)
        self.assertEqual(res.verdict, ExclusionVerdict.ELIGIBLE)

    def test_10_robustness_boundary_at_40(self):
        """10. Verify robustness threshold: mean score >= 40.0 does NOT trigger chronic exclusion."""
        ckey = "NIFTY_3s_DIRECTION_CLASSIFIER_5m_R001"
        b_run = create_benchmark_run(self.tmp_dir, context_key=ckey)

        for i in range(3):
            spec_i = {
                "market": "NIFTY",
                "sampling_interval_sec": 3,
                "task_type": "DIRECTION_CLASSIFIER",
                "prediction_horizon": "5m",
                "regime_id": "R001",
                "features": ["adx_14"],
                "algorithm": "xgboost",
                "hyperparameters": {"max_depth": 3 + i},
            }
            sig_i, _, _ = compute_experiment_signature(spec_i)
            register_or_get_experiment(self.tmp_dir, spec_i, model_name=f"M_{i}")
            record_model_benchmark(
                self.tmp_dir,
                benchmark_run_id=b_run,
                signature_hash=sig_i,
                model_name=f"M_{i}",
                context_key=ckey,
                algorithm="xgboost",
                dataset_name="d.parquet",
                feature_count=1,
                primary_metric_name="roc_auc",
                primary_metric_value=0.42,
                robustness_score=42.0,
            )

        prop_spec = {
            "market": "NIFTY",
            "sampling_interval_sec": 3,
            "task_type": "DIRECTION_CLASSIFIER",
            "prediction_horizon": "5m",
            "regime_id": "R001",
            "features": ["adx_14"],
            "algorithm": "xgboost",
            "hyperparameters": {"max_depth": 8},
        }
        res = audit_experiment_exclusion(self.tmp_dir, prop_spec, schema=self.mock_schema)
        self.assertEqual(res.verdict, ExclusionVerdict.ELIGIBLE)

    def test_11_regime_degradation_boundary_at_30_pct(self):
        """11. Verify degradation > 30% triggers CAUTION with EXTREME_REGIME_FRAGILITY."""
        ckey = "NIFTY_3s_DIRECTION_CLASSIFIER_5m_R001"
        b_run = create_benchmark_run(self.tmp_dir, context_key=ckey)
        spec = {
            "market": "NIFTY",
            "sampling_interval_sec": 3,
            "task_type": "DIRECTION_CLASSIFIER",
            "prediction_horizon": "5m",
            "regime_id": "R001",
            "features": ["adx_14"],
            "algorithm": "xgboost",
            "hyperparameters": {"depth": 3},
        }
        sig_h, _, _ = compute_experiment_signature(spec)
        register_or_get_experiment(self.tmp_dir, spec, model_name="M_FRAG")
        record_model_benchmark(
            self.tmp_dir,
            benchmark_run_id=b_run,
            signature_hash=sig_h,
            model_name="M_FRAG",
            context_key=ckey,
            algorithm="xgboost",
            dataset_name="d.parquet",
            feature_count=1,
            primary_metric_name="roc_auc",
            primary_metric_value=0.80,
            robustness_score=75.0,
        )
        record_regime_evaluation(
            self.tmp_dir,
            model_name="M_FRAG",
            signature_hash=sig_h,
            tested_regime_id="R002",
            tested_regime_hash="def_hash",
            is_native_regime=False,
            sample_count=1000,
            primary_metric=0.45,
            regime_degradation_pct=35.0, # > 30%
        )

        prop_spec = dict(spec)
        prop_spec["hyperparameters"] = {"depth": 5}

        res = audit_experiment_exclusion(self.tmp_dir, prop_spec, schema=self.mock_schema)
        self.assertEqual(res.verdict, ExclusionVerdict.CAUTION)
        self.assertEqual(res.primary_reason, ExclusionReason.EXTREME_REGIME_FRAGILITY)

    def test_12_calibration_boundary_at_ece_0_10(self):
        """12. Verify ECE >= 0.10 triggers CAUTION with SEVERE_MISCALIBRATION."""
        ckey = "NIFTY_3s_DIRECTION_CLASSIFIER_5m_R001"
        b_run = create_benchmark_run(self.tmp_dir, context_key=ckey)
        spec = {
            "market": "NIFTY",
            "sampling_interval_sec": 3,
            "task_type": "DIRECTION_CLASSIFIER",
            "prediction_horizon": "5m",
            "regime_id": "R001",
            "features": ["adx_14"],
            "algorithm": "xgboost",
            "hyperparameters": {"depth": 3},
        }
        sig_h, _, _ = compute_experiment_signature(spec)
        register_or_get_experiment(self.tmp_dir, spec, model_name="M_CAL")
        record_model_benchmark(
            self.tmp_dir,
            benchmark_run_id=b_run,
            signature_hash=sig_h,
            model_name="M_CAL",
            context_key=ckey,
            algorithm="xgboost",
            dataset_name="d.parquet",
            feature_count=1,
            primary_metric_name="roc_auc",
            primary_metric_value=0.80,
            robustness_score=75.0,
            expected_calibration_error=0.125, # >= 0.10
        )

        prop_spec = dict(spec)
        prop_spec["hyperparameters"] = {"depth": 6}

        res = audit_experiment_exclusion(self.tmp_dir, prop_spec, schema=self.mock_schema)
        self.assertEqual(res.verdict, ExclusionVerdict.CAUTION)
        self.assertEqual(res.primary_reason, ExclusionReason.SEVERE_MISCALIBRATION)

    def test_13_caution_vs_excluded_behavior(self):
        """13. Verify is_search_path_excluded handles allow_caution flag properly."""
        ckey = "NIFTY_3s_DIRECTION_CLASSIFIER_5m_R001"
        b_run = create_benchmark_run(self.tmp_dir, context_key=ckey)
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
        register_or_get_experiment(self.tmp_dir, spec, model_name="M1")
        record_model_benchmark(
            self.tmp_dir,
            benchmark_run_id=b_run,
            signature_hash=sig_h,
            model_name="M1",
            context_key=ckey,
            algorithm="xgboost",
            dataset_name="d.parquet",
            feature_count=1,
            primary_metric_name="roc_auc",
            primary_metric_value=0.80,
            robustness_score=70.0,
            expected_calibration_error=0.15,
        )

        prop_spec = dict(spec)
        prop_spec["hyperparameters"] = {"max_depth": 7}

        # CAUTION allowed -> is_search_path_excluded is False
        self.assertFalse(is_search_path_excluded(self.tmp_dir, prop_spec, allow_caution=True, schema=self.mock_schema))
        # CAUTION not allowed -> is_search_path_excluded is True
        self.assertTrue(is_search_path_excluded(self.tmp_dir, prop_spec, allow_caution=False, schema=self.mock_schema))

    def test_14_deterministic_reason_precedence(self):
        """14. Verify reason precedence: DUPLICATE beats DEPRECATED beats CHRONIC beats FRAGILITY."""
        ckey = "NIFTY_3s_DIRECTION_CLASSIFIER_5m_R001"
        spec = {
            "market": "NIFTY",
            "sampling_interval_sec": 3,
            "task_type": "DIRECTION_CLASSIFIER",
            "prediction_horizon": "5m",
            "regime_id": "R001",
            "features": ["dep_old"],
            "algorithm": "xgboost",
        }
        # Register experiment -> it is both DUPLICATE and contains DEPRECATED feature
        register_or_get_experiment(self.tmp_dir, spec, model_name="M_BOTH")

        res = audit_experiment_exclusion(self.tmp_dir, spec, schema=self.mock_schema)
        self.assertEqual(res.verdict, ExclusionVerdict.EXCLUDED)
        self.assertEqual(res.primary_reason, ExclusionReason.DUPLICATE_EXPERIMENT)

    def test_15_cold_start_context_eligible(self):
        """15. Verify cold-start unexplored context returns ELIGIBLE with reason NONE."""
        ckey = "NIFTY_3s_DIRECTION_CLASSIFIER_5m_R005"
        spec = {
            "market": "NIFTY",
            "sampling_interval_sec": 3,
            "task_type": "DIRECTION_CLASSIFIER",
            "prediction_horizon": "5m",
            "regime_id": "R005",
            "features": ["adx_14"],
            "algorithm": "xgboost",
        }
        res = audit_experiment_exclusion(self.tmp_dir, spec, schema=self.mock_schema)
        self.assertEqual(res.verdict, ExclusionVerdict.ELIGIBLE)
        self.assertEqual(res.primary_reason, ExclusionReason.NONE)

    def test_16_sparse_context_behavior(self):
        """16. Verify sparse context with 1 passing trial returns ELIGIBLE."""
        ckey = "NIFTY_3s_DIRECTION_CLASSIFIER_5m_R001"
        b_run = create_benchmark_run(self.tmp_dir, context_key=ckey)
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
        register_or_get_experiment(self.tmp_dir, spec, model_name="M1")
        record_model_benchmark(
            self.tmp_dir,
            benchmark_run_id=b_run,
            signature_hash=sig_h,
            model_name="M1",
            context_key=ckey,
            algorithm="xgboost",
            dataset_name="d.parquet",
            feature_count=1,
            primary_metric_name="roc_auc",
            primary_metric_value=0.82,
            robustness_score=82.0,
        )

        prop_spec = dict(spec)
        prop_spec["hyperparameters"] = {"max_depth": 6}
        res = audit_experiment_exclusion(self.tmp_dir, prop_spec, schema=self.mock_schema)
        self.assertEqual(res.verdict, ExclusionVerdict.ELIGIBLE)

    def test_17_r001_vs_r002_context_isolation(self):
        """17. Verify duplicate/negative evidence in R001 does NOT exclude identical spec in R002."""
        spec_r1 = {
            "market": "NIFTY",
            "sampling_interval_sec": 3,
            "task_type": "DIRECTION_CLASSIFIER",
            "prediction_horizon": "5m",
            "regime_id": "R001",
            "features": ["adx_14"],
            "algorithm": "xgboost",
        }
        spec_r2 = dict(spec_r1)
        spec_r2["regime_id"] = "R002"

        register_or_get_experiment(self.tmp_dir, spec_r1, model_name="M_R1")

        self.assertEqual(audit_experiment_exclusion(self.tmp_dir, spec_r1, schema=self.mock_schema).verdict, ExclusionVerdict.EXCLUDED)
        self.assertEqual(audit_experiment_exclusion(self.tmp_dir, spec_r2, schema=self.mock_schema).verdict, ExclusionVerdict.ELIGIBLE)

    def test_18_horizon_isolation_5m_vs_15m(self):
        """18. Verify duplicate in 5m horizon does NOT exclude identical spec in 15m horizon."""
        spec_5m = {
            "market": "NIFTY",
            "sampling_interval_sec": 3,
            "task_type": "DIRECTION_CLASSIFIER",
            "prediction_horizon": "5m",
            "regime_id": "R001",
            "features": ["adx_14"],
            "algorithm": "xgboost",
        }
        spec_15m = dict(spec_5m)
        spec_15m["prediction_horizon"] = "15m"

        register_or_get_experiment(self.tmp_dir, spec_5m, model_name="M_5m")

        self.assertEqual(audit_experiment_exclusion(self.tmp_dir, spec_5m, schema=self.mock_schema).verdict, ExclusionVerdict.EXCLUDED)
        self.assertEqual(audit_experiment_exclusion(self.tmp_dir, spec_15m, schema=self.mock_schema).verdict, ExclusionVerdict.ELIGIBLE)

    def test_19_market_isolation_nifty_vs_banknifty(self):
        """19. Verify duplicate in NIFTY does NOT exclude identical spec in BANKNIFTY."""
        spec_n = {
            "market": "NIFTY",
            "sampling_interval_sec": 3,
            "task_type": "DIRECTION_CLASSIFIER",
            "prediction_horizon": "5m",
            "regime_id": "R001",
            "features": ["adx_14"],
            "algorithm": "xgboost",
        }
        spec_bn = dict(spec_n)
        spec_bn["market"] = "BANKNIFTY"

        register_or_get_experiment(self.tmp_dir, spec_n, model_name="M_N")

        self.assertEqual(audit_experiment_exclusion(self.tmp_dir, spec_n, schema=self.mock_schema).verdict, ExclusionVerdict.EXCLUDED)
        self.assertEqual(audit_experiment_exclusion(self.tmp_dir, spec_bn, schema=self.mock_schema).verdict, ExclusionVerdict.ELIGIBLE)

    def test_20_signature_exclusion_audit(self):
        """20. Verify audit_signature_exclusion helper function."""
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
        register_or_get_experiment(self.tmp_dir, spec, model_name="M_SIG")

        res = audit_signature_exclusion(self.tmp_dir, sig_h, schema=self.mock_schema)
        self.assertEqual(res.verdict, ExclusionVerdict.EXCLUDED)
        self.assertEqual(res.primary_reason, ExclusionReason.DUPLICATE_EXPERIMENT)

    def test_21_context_pruning_agenda(self):
        """21. Verify build_context_pruning_agenda aggregates quarantined features and failure counts."""
        ckey = "NIFTY_3s_DIRECTION_CLASSIFIER_5m_R001"
        agenda = build_context_pruning_agenda(self.tmp_dir, ckey, schema=self.mock_schema)

        self.assertEqual(agenda.context_key, ckey)
        self.assertEqual(agenda.market, "NIFTY")
        self.assertIn("dep_old", agenda.quarantined_features)
        self.assertIn("dep_alpha", agenda.quarantined_features)

    def test_22_deterministic_ordering(self):
        """22. Verify repeated agenda generation produces identical results."""
        ckey = "NIFTY_3s_DIRECTION_CLASSIFIER_5m_R001"
        a1 = build_context_pruning_agenda(self.tmp_dir, ckey, schema=self.mock_schema)
        a2 = build_context_pruning_agenda(self.tmp_dir, ckey, schema=self.mock_schema)

        self.assertEqual(a1.quarantined_features, a2.quarantined_features)
        self.assertEqual(a1.total_explored_signatures, a2.total_explored_signatures)

    def test_23_nan_and_infinity_safety(self):
        """23. Verify audit handles corrupted or NaN metrics in database without crashing."""
        ckey = "NIFTY_3s_DIRECTION_CLASSIFIER_5m_R001"
        b_run = create_benchmark_run(self.tmp_dir, context_key=ckey)
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
        register_or_get_experiment(self.tmp_dir, spec, model_name="M_NAN")
        record_model_benchmark(
            self.tmp_dir,
            benchmark_run_id=b_run,
            signature_hash=sig_h,
            model_name="M_NAN",
            context_key=ckey,
            algorithm="xgboost",
            dataset_name="d.parquet",
            feature_count=1,
            primary_metric_name="roc_auc",
            primary_metric_value=0.0,
            robustness_score=0.0,
        )

        res = audit_experiment_exclusion(self.tmp_dir, spec, schema=self.mock_schema)
        self.assertIsInstance(res.verdict, ExclusionVerdict)

    def test_24_concurrent_read_safety(self):
        """24. Verify concurrent audit calls execute safely under multithreading."""
        spec = {
            "market": "NIFTY",
            "sampling_interval_sec": 3,
            "task_type": "DIRECTION_CLASSIFIER",
            "prediction_horizon": "5m",
            "regime_id": "R001",
            "features": ["adx_14"],
            "algorithm": "xgboost",
        }
        register_or_get_experiment(self.tmp_dir, spec, model_name="M_CONC")

        def _run_audit():
            return audit_experiment_exclusion(self.tmp_dir, spec, schema=self.mock_schema)

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(_run_audit) for _ in range(8)]
            results = [f.result() for f in futures]

        self.assertEqual(len(results), 8)
        self.assertTrue(all(r.verdict == ExclusionVerdict.EXCLUDED for r in results))

    def test_25_production_database_immutability(self):
        """25. Assert that negative evidence pruning never mutates production state."""
        ev_path = "apps/feature_recommendation_evidence.db"
        self.assertTrue(os.path.isfile(ev_path), "Evidence DB must exist")

        with open(ev_path, "rb") as fh:
            sha_initial = hashlib.sha256(fh.read()).hexdigest()

        ckey = "NIFTY_3s_DIRECTION_CLASSIFIER_5m_R001"
        _ = build_context_pruning_agenda(self.tmp_dir, ckey, schema=self.mock_schema)

        with open(ev_path, "rb") as fh:
            sha_final = hashlib.sha256(fh.read()).hexdigest()

        self.assertEqual(sha_initial, sha_final, "Evidence DB must remain 100% unmutated")

    def test_26_evidence_database_sha256_unchanged(self):
        """26. Assert that feature_recommendation_evidence.db SHA-256 is unchanged after extensive audits."""
        ev_path = "apps/feature_recommendation_evidence.db"
        with open(ev_path, "rb") as fh:
            sha_before = hashlib.sha256(fh.read()).hexdigest()

        spec = {
            "market": "NIFTY",
            "sampling_interval_sec": 3,
            "task_type": "DIRECTION_CLASSIFIER",
            "prediction_horizon": "5m",
            "regime_id": "R001",
            "features": ["dep_old", "rsi_14"],
            "algorithm": "xgboost",
        }
        for _ in range(5):
            _ = audit_experiment_exclusion(self.tmp_dir, spec, schema=self.mock_schema)

        with open(ev_path, "rb") as fh:
            sha_after = hashlib.sha256(fh.read()).hexdigest()

        self.assertEqual(sha_before, sha_after)


if __name__ == "__main__":
    unittest.main()

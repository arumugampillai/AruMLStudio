"""Comprehensive Unit Tests for Phase 4D.5: Robustness Ranking Policy Engine."""

import hashlib
import os
import shutil
import tempfile
import unittest

from chain_replay_ml.research_memory import (
    ROB_POLICY_v1_0,
    RobustnessRankingPolicy,
    compute_pareto_frontier,
    compute_robustness_score,
    create_benchmark_run,
    get_benchmark_run,
    get_model_benchmark_by_id,
    get_model_benchmarks_for_context,
    init_analysis_db,
    normalize_metric,
    persist_context_rankings,
    rank_models_in_context,
    record_feature_set_evaluation,
    record_model_benchmark,
    record_regime_evaluation,
    register_or_get_experiment,
)


class TestRobustnessRanking(unittest.TestCase):
    """Test suite verifying robustness score calculations, Pareto sorting, context isolation, and ranking dossiers."""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp(prefix="aruml_test_ranking_")
        init_analysis_db(self.tmp_dir)

        self.spec_trend_a = {
            "market": "NIFTY",
            "sampling_interval_sec": 3,
            "task_type": "DIRECTION_CLASSIFIER",
            "prediction_horizon": "5m",
            "regime_id": "R001",
            "regime_definition_hash": "def_hash_trend_v1",
            "dataset_snapshot_hash": "ds_hash_20260819",
            "features": ["adx_14", "rsi_14", "atm_iv_pctile", "feat_a1", "feat_a2"],
            "algorithm": "xgboost",
            "hyperparameters": {"max_depth": 8, "learning_rate": 0.1},
            "walk_forward_config": {"folds": 5, "window_mode": "expanding"},
            "random_seed": 42,
        }

        self.spec_trend_b = {
            "market": "NIFTY",
            "sampling_interval_sec": 3,
            "task_type": "DIRECTION_CLASSIFIER",
            "prediction_horizon": "5m",
            "regime_id": "R001",
            "regime_definition_hash": "def_hash_trend_v1",
            "dataset_snapshot_hash": "ds_hash_20260819",
            "features": ["adx_14", "rsi_14", "atm_iv_pctile"],
            "algorithm": "catboost",
            "hyperparameters": {"depth": 5, "learning_rate": 0.03},
            "walk_forward_config": {"folds": 5, "window_mode": "expanding"},
            "random_seed": 42,
        }

        self.spec_side_c = {
            "market": "NIFTY",
            "sampling_interval_sec": 3,
            "task_type": "DIRECTION_CLASSIFIER",
            "prediction_horizon": "5m",
            "regime_id": "R002",
            "regime_definition_hash": "def_hash_side_v1",
            "dataset_snapshot_hash": "ds_hash_20260819",
            "features": ["adx_14", "bb_width"],
            "algorithm": "lightgbm",
            "hyperparameters": {"num_leaves": 31},
            "walk_forward_config": {"folds": 5, "window_mode": "expanding"},
            "random_seed": 42,
        }

        _, self.rec_trend_a = register_or_get_experiment(self.tmp_dir, self.spec_trend_a, model_name="DIR_TREND_OVERFIT_XGB")
        _, self.rec_trend_b = register_or_get_experiment(self.tmp_dir, self.spec_trend_b, model_name="DIR_TREND_ROBUST_CAT")
        _, self.rec_side_c = register_or_get_experiment(self.tmp_dir, self.spec_side_c, model_name="DIR_SIDE_MODEL_LGB")

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_robustness_scoring_mathematics(self):
        """1. Verify exact calculation of base score and all penalty components."""
        score, breakdown, warnings = compute_robustness_score(
            primary_metric_name="roc_auc",
            primary_metric_value=0.75,
            fold_mean=0.75,
            fold_std=0.03,            # P_sigma = 15.0 * (0.03 / 0.75) = 0.60
            worst_fold_drawdown=0.08, # P_worst = 10.0 * (0.08 - 0.05) = 0.30
            ece=0.05,                 # P_calib = 10.0 * 0.05 = 0.50
            avg_regime_degradation_pct=10.0, # P_deg = 0.15 * 10.0 = 1.50
            total_features=40,        # P_N = 2.0 * ln(40/30) = 2.0 * 0.2877 = 0.5754
            experimental_dependency_ratio=0.25, # P_exp = 12.0 * 0.25 = 3.00
            deprecated_feature_count=0,
            policy=ROB_POLICY_v1_0,
        )

        self.assertEqual(len(warnings), 0)
        self.assertAlmostEqual(breakdown["base_performance_contribution"], 75.0, places=2)
        self.assertAlmostEqual(breakdown["fold_variance_penalty"], -0.60, places=2)
        self.assertAlmostEqual(breakdown["worst_fold_penalty"], -0.30, places=2)
        self.assertAlmostEqual(breakdown["calibration_penalty"], -0.50, places=2)
        self.assertAlmostEqual(breakdown["regime_degradation_penalty"], -1.50, places=2)
        self.assertAlmostEqual(breakdown["experimental_risk_penalty"], -3.00, places=2)
        self.assertAlmostEqual(breakdown["parsimony_penalty"], -0.5754, places=2)

        expected_total = 75.0 - 0.60 - 0.30 - 0.50 - 1.50 - 3.00 - 0.5754
        self.assertAlmostEqual(score, expected_total, places=2)

    def test_robust_candidate_beats_overfit_peak_score(self):
        """2. Verify that a stable, robust model beats a fragile model with higher peak validation score."""
        # Overfitted Model A: Higher peak score (0.82) but high variance, severe degradation, high experimental risk
        score_a, breakdown_a, _ = compute_robustness_score(
            primary_metric_name="roc_auc",
            primary_metric_value=0.82,
            fold_mean=0.82,
            fold_std=0.15,            # High variance: 15.0 * (0.15/0.82) = 2.74
            worst_fold_drawdown=0.25, # Severe worst fold: 10.0 * (0.25-0.05) = 2.00
            ece=0.18,                 # Poor calibration: 10.0 * 0.18 = 1.80
            avg_regime_degradation_pct=35.0, # Severe drop: 0.15 * 35.0 = 5.25
            total_features=80,        # Bloated features: 2.0 * ln(80/30) = 1.96
            experimental_dependency_ratio=0.80, # 80% experimental: 12.0 * 0.8 = 9.60
            policy=ROB_POLICY_v1_0,
        )

        # Robust Model B: Slightly lower peak score (0.78) but rock-solid stability and zero experimental risk
        score_b, breakdown_b, _ = compute_robustness_score(
            primary_metric_name="roc_auc",
            primary_metric_value=0.78,
            fold_mean=0.78,
            fold_std=0.015,           # Low variance: 15.0 * (0.015/0.78) = 0.288
            worst_fold_drawdown=0.04, # Clean worst fold: 0.0 penalty
            ece=0.03,                 # Well-calibrated: 10.0 * 0.03 = 0.30
            avg_regime_degradation_pct=8.0, # Low degradation: 0.15 * 8.0 = 1.20
            total_features=25,        # Compact: 0.0 parsimony penalty
            experimental_dependency_ratio=0.0, # 100% canonical features: 0.0 penalty
            policy=ROB_POLICY_v1_0,
        )

        self.assertGreater(score_b, score_a)
        self.assertGreater(score_b, 75.0)
        self.assertLess(score_a, 60.0)

    def test_pareto_frontier_calculation(self):
        """3. Verify multi-objective non-dominated Pareto ranking."""
        candidates = [
            {
                "model_name": "MODEL_A",
                "score_breakdown": {"base_performance_contribution": 80.0, "fold_variance_penalty": -5.0, "regime_degradation_penalty": -5.0},
                "raw_metrics_summary": {"total_features": 40},
            },
            {
                "model_name": "MODEL_B",
                "score_breakdown": {"base_performance_contribution": 78.0, "fold_variance_penalty": -1.0, "regime_degradation_penalty": -1.0},
                "raw_metrics_summary": {"total_features": 20},
            },
            {
                "model_name": "MODEL_C_DOMINATED",
                "score_breakdown": {"base_performance_contribution": 70.0, "fold_variance_penalty": -6.0, "regime_degradation_penalty": -6.0},
                "raw_metrics_summary": {"total_features": 50},
            },
        ]

        pareto_res = compute_pareto_frontier(candidates)
        self.assertEqual(len(pareto_res), 3)

        cand_a = next(c for c in pareto_res if c["model_name"] == "MODEL_A")
        cand_b = next(c for c in pareto_res if c["model_name"] == "MODEL_B")
        cand_c = next(c for c in pareto_res if c["model_name"] == "MODEL_C_DOMINATED")

        # Models A and B are non-dominated on the frontier
        self.assertEqual(cand_a["pareto_rank"], 1)
        self.assertTrue(cand_a["is_pareto_optimal"])
        self.assertEqual(cand_b["pareto_rank"], 1)
        self.assertTrue(cand_b["is_pareto_optimal"])

        # Model C is strictly dominated by both A and B
        self.assertEqual(cand_c["pareto_rank"], 2)
        self.assertFalse(cand_c["is_pareto_optimal"])

    def test_nan_and_inf_handling(self):
        """4. Verify NaN or Infinity metrics produce 0 score and REJECTED_INVALID_METRICS warning."""
        score_nan, _, warnings_nan = compute_robustness_score(
            primary_metric_name="roc_auc",
            primary_metric_value=float("nan"),
            fold_mean=0.7,
            fold_std=0.05,
        )
        self.assertEqual(score_nan, 0.0)
        self.assertIn("REJECTED_INVALID_METRICS", warnings_nan)

        score_inf, _, warnings_inf = compute_robustness_score(
            primary_metric_name="roc_auc",
            primary_metric_value=0.7,
            fold_mean=float("inf"),
            fold_std=0.05,
        )
        self.assertEqual(score_inf, 0.0)
        self.assertIn("REJECTED_INVALID_METRICS", warnings_inf)

    def test_deprecated_feature_penalty(self):
        """5. Verify severe flat penalty and warning when deprecated features exist."""
        score_clean, _, _ = compute_robustness_score(
            primary_metric_name="roc_auc",
            primary_metric_value=0.70,
            fold_mean=0.70,
            fold_std=0.02,
            deprecated_feature_count=0,
        )

        score_dep, breakdown_dep, warnings_dep = compute_robustness_score(
            primary_metric_name="roc_auc",
            primary_metric_value=0.70,
            fold_mean=0.70,
            fold_std=0.02,
            deprecated_feature_count=2,
        )

        self.assertAlmostEqual(score_clean - score_dep, 25.0, places=2)
        self.assertIn("DEPRECATED_FEATURE_EXPOSURE_COUNT_2", warnings_dep)

    def test_context_isolation_during_ranking(self):
        """6. Verify Trend and Sideways models never enter the same ranking pool."""
        run_trend = create_benchmark_run(self.tmp_dir, context_key="NIFTY_3s_DIRECTION_CLASSIFIER_5m_R001")
        run_side = create_benchmark_run(self.tmp_dir, context_key="NIFTY_3s_DIRECTION_CLASSIFIER_5m_R002")

        # Record Trend Model A (Overfit with high variance and severe worst fold)
        record_model_benchmark(
            self.tmp_dir,
            benchmark_run_id=run_trend,
            signature_hash=self.rec_trend_a["signature_hash"],
            model_name="DIR_TREND_OVERFIT_XGB",
            context_key="NIFTY_3s_DIRECTION_CLASSIFIER_5m_R001",
            algorithm="xgboost",
            dataset_name="trend_3s.parquet",
            feature_count=50,
            primary_metric_name="roc_auc",
            primary_metric_value=0.82,
            fold_metric_mean=0.82,
            fold_metric_std=0.20,
            worst_fold_drawdown=0.35,
            expected_calibration_error=0.15,
        )

        # Record Trend Model B
        record_model_benchmark(
            self.tmp_dir,
            benchmark_run_id=run_trend,
            signature_hash=self.rec_trend_b["signature_hash"],
            model_name="DIR_TREND_ROBUST_CAT",
            context_key="NIFTY_3s_DIRECTION_CLASSIFIER_5m_R001",
            algorithm="catboost",
            dataset_name="trend_3s.parquet",
            feature_count=3,
            primary_metric_name="roc_auc",
            primary_metric_value=0.78,
            fold_metric_mean=0.78,
            fold_metric_std=0.01,
            worst_fold_drawdown=0.02,
        )

        # Record Sideways Model C
        record_model_benchmark(
            self.tmp_dir,
            benchmark_run_id=run_side,
            signature_hash=self.rec_side_c["signature_hash"],
            model_name="DIR_SIDE_MODEL_LGB",
            context_key="NIFTY_3s_DIRECTION_CLASSIFIER_5m_R002",
            algorithm="lightgbm",
            dataset_name="side_3s.parquet",
            feature_count=2,
            primary_metric_name="roc_auc",
            primary_metric_value=0.71,
        )

        # Rank Trend Context
        trend_ranked = rank_models_in_context(self.tmp_dir, "NIFTY_3s_DIRECTION_CLASSIFIER_5m_R001")
        self.assertEqual(len(trend_ranked), 2)
        # Robust Model B should outrank Overfit Model A
        self.assertEqual(trend_ranked[0]["model_name"], "DIR_TREND_ROBUST_CAT")
        self.assertEqual(trend_ranked[0]["rank_in_context"], 1)
        self.assertEqual(trend_ranked[0]["recommendation_status"], "CHAMPION_CANDIDATE")
        self.assertEqual(trend_ranked[1]["model_name"], "DIR_TREND_OVERFIT_XGB")
        self.assertEqual(trend_ranked[1]["rank_in_context"], 2)
        self.assertEqual(trend_ranked[1]["recommendation_status"], "CHALLENGER_CANDIDATE")

        # Rank Sideways Context
        side_ranked = rank_models_in_context(self.tmp_dir, "NIFTY_3s_DIRECTION_CLASSIFIER_5m_R002")
        self.assertEqual(len(side_ranked), 1)
        self.assertEqual(side_ranked[0]["model_name"], "DIR_SIDE_MODEL_LGB")
        self.assertEqual(side_ranked[0]["rank_in_context"], 1)

    def test_persist_and_retrieve_context_rankings(self):
        """7. Verify persist_context_rankings updates database records and benchmark run summary."""
        run_trend = create_benchmark_run(self.tmp_dir, context_key="NIFTY_3s_DIRECTION_CLASSIFIER_5m_R001")

        record_model_benchmark(
            self.tmp_dir,
            benchmark_run_id=run_trend,
            signature_hash=self.rec_trend_b["signature_hash"],
            model_name="DIR_TREND_ROBUST_CAT",
            context_key="NIFTY_3s_DIRECTION_CLASSIFIER_5m_R001",
            algorithm="catboost",
            dataset_name="trend_3s.parquet",
            feature_count=3,
            primary_metric_name="roc_auc",
            primary_metric_value=0.78,
            fold_metric_mean=0.78,
            fold_metric_std=0.01,
        )

        ranked = rank_models_in_context(self.tmp_dir, "NIFTY_3s_DIRECTION_CLASSIFIER_5m_R001", benchmark_run_id=run_trend)
        updated = persist_context_rankings(self.tmp_dir, benchmark_run_id=run_trend, ranked_dossiers=ranked)
        self.assertEqual(updated, 1)

        # Verify benchmark_runs updated
        run_doc = get_benchmark_run(self.tmp_dir, run_trend)
        self.assertEqual(run_doc["top_model_name"], "DIR_TREND_ROBUST_CAT")
        self.assertEqual(run_doc["ranking_policy_version"], "ROB_POLICY_v1.0")

        # Verify model_benchmarks row updated
        bms = get_model_benchmarks_for_context(self.tmp_dir, "NIFTY_3s_DIRECTION_CLASSIFIER_5m_R001")
        self.assertEqual(len(bms), 1)
        self.assertEqual(bms[0]["rank_in_context"], 1)
        self.assertEqual(bms[0]["recommendation_status"], "CHAMPION_CANDIDATE")
        self.assertGreater(bms[0]["robustness_score"], 70.0)

    def test_ranking_policy_versioning_and_hashing(self):
        """8. Verify policy configuration hashing and serializability."""
        policy_v1 = RobustnessRankingPolicy(policy_version="ROB_POLICY_v1.0")
        hash1 = policy_v1.compute_policy_hash()
        self.assertTrue(len(hash1) == 64)

        # Same parameters -> identical hash
        policy_v1_dup = RobustnessRankingPolicy(policy_version="ROB_POLICY_v1.0")
        self.assertEqual(policy_v1_dup.compute_policy_hash(), hash1)

        # Modified parameters -> different hash
        policy_v2 = RobustnessRankingPolicy(policy_version="ROB_POLICY_v2.0", lambda_sigma=25.0)
        self.assertNotEqual(policy_v2.compute_policy_hash(), hash1)

    def test_evidence_db_immutability(self):
        """9. Verify ranking calculations do NOT touch or mutate feature_recommendation_evidence.db."""
        ev_path = "apps/feature_recommendation_evidence.db"
        self.assertTrue(os.path.isfile(ev_path))
        with open(ev_path, "rb") as fh:
            sha_before = hashlib.sha256(fh.read()).hexdigest()

        # Execute ranking workflow
        run_id = create_benchmark_run(self.tmp_dir, context_key="NIFTY_3s_DIRECTION_CLASSIFIER_5m_R001")
        record_model_benchmark(
            self.tmp_dir,
            benchmark_run_id=run_id,
            signature_hash=self.rec_trend_b["signature_hash"],
            model_name="DIR_TREND_ROBUST_CAT",
            context_key="NIFTY_3s_DIRECTION_CLASSIFIER_5m_R001",
            algorithm="catboost",
            dataset_name="trend_3s.parquet",
            feature_count=3,
            primary_metric_name="roc_auc",
            primary_metric_value=0.78,
            fold_metric_mean=0.78,
            fold_metric_std=0.01,
        )
        ranked = rank_models_in_context(self.tmp_dir, "NIFTY_3s_DIRECTION_CLASSIFIER_5m_R001")
        persist_context_rankings(self.tmp_dir, benchmark_run_id=run_id, ranked_dossiers=ranked)

        with open(ev_path, "rb") as fh:
            sha_after = hashlib.sha256(fh.read()).hexdigest()

        self.assertEqual(sha_before, sha_after)
        self.assertEqual(sha_after, "6f91afca94ec87a1210d8f4bcef356b9c16a6ef5a488268f41c5b1b81431ade2")


if __name__ == "__main__":
    unittest.main()

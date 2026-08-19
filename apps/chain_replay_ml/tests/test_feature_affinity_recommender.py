"""Unit tests for Phase 4E.3 — Empirical Feature Affinity & Interaction Recommender.

Verifies:
1. Empty database
2. Single feature
3. Multiple features
4. Feature score bounds
5. Confidence = 0 with zero experiments
6. Confidence scaling
7. Sparse evidence
8. Developing evidence
9. Mature evidence
10. Deprecated feature quarantine
11. Unknown feature handling
12. Base feature population
13. Registry feature population
14. Experimental feature population
15. Context isolation by regime
16. Context isolation by task
17. Context isolation by horizon
18. Context isolation by sampling interval
19. Context isolation by market
20. Pairwise interaction detection
21. Negative interaction lift
22. Sparse pair confidence
23. Deterministic ordering
24. Missing evidence handling
25. NaN/Infinity safety
26. Champion missing-feature opportunity
27. Research candidate feature opportunity
28. Production immutability
29. Evidence DB immutability
30. Full report generation
"""

import hashlib
import json
import math
import os
import shutil
import tempfile
import unittest

from chain_replay_ml.research_memory.benchmarks import (
    create_benchmark_run,
    record_model_benchmark,
)
from chain_replay_ml.research_memory.ranking import (
    persist_context_rankings,
    rank_models_in_context,
)
from chain_replay_ml.research_memory.regime_eval import (
    record_regime_evaluation,
)
from chain_replay_ml.research_memory.signature import (
    compute_experiment_signature,
    register_or_get_experiment,
)
from chain_replay_ml.research_recommendations.feature_affinity import (
    ContextFeatureAffinityReport,
    FeatureAffinityResult,
    FeatureInteractionResult,
    FeatureRecommendationClass,
    analyze_feature_affinity,
    classify_feature_recommendation,
    compute_feature_affinity_score,
    compute_feature_confidence,
    compute_interaction_synergy_score,
    recommend_features_for_context,
)
from chain_replay_ml.research_memory.champion_history import (
    set_champion_for_context,
)


class TestFeatureAffinityRecommender(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp(prefix="aruml_test_4e3_aff_")
        self.mock_schema = {
            "columns": {
                "adx_14": {"is_base": True, "project_id": "PL_0001", "status": "ACTIVE"},
                "rsi_14": {"is_base": False, "project_id": None, "status": "ACTIVE"},
                "exp_alpha": {"is_base": False, "project_id": "PL_0002", "status": "ACTIVE"},
                "dep_feature": {"is_base": False, "project_id": "PL_0001", "status": "DEPRECATED"},
            }
        }

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_01_empty_database_safety(self):
        """1. Verify analyzer handles empty database safely without errors."""
        ckey = "NIFTY_3s_DIRECTION_CLASSIFIER_5m_R001"
        res = analyze_feature_affinity(self.tmp_dir, ckey, "adx_14", schema=self.mock_schema)

        self.assertEqual(res.feature_name, "adx_14")
        self.assertEqual(res.feature_population, "BASE")
        self.assertEqual(res.evidence_count, 0)
        self.assertEqual(res.confidence, 0.0)
        self.assertEqual(res.recommendation_class, FeatureRecommendationClass.EXPLORATORY)
        self.assertGreaterEqual(res.affinity_score, 0.0)
        self.assertLessEqual(res.affinity_score, 100.0)

    def test_02_single_feature_analysis(self):
        """2. Verify single feature affinity analysis with 1 benchmark experiment."""
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
            robustness_score=80.0,
        )

        res = analyze_feature_affinity(self.tmp_dir, ckey, "adx_14", schema=self.mock_schema)
        self.assertEqual(res.evidence_count, 1)
        self.assertEqual(res.robustness_support, 80.0)
        self.assertGreater(res.confidence, 0.0)

    def test_03_multiple_features_in_experiment(self):
        """3. Verify multi-feature experiments accurately distribute evidence to all participating features."""
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
        register_or_get_experiment(self.tmp_dir, spec, model_name="M_MULTI")
        record_model_benchmark(
            self.tmp_dir,
            benchmark_run_id=b_run,
            signature_hash=sig_h,
            model_name="M_MULTI",
            context_key=ckey,
            algorithm="xgboost",
            dataset_name="d.parquet",
            feature_count=2,
            primary_metric_name="roc_auc",
            primary_metric_value=0.85,
            robustness_score=85.0,
        )

        res_adx = analyze_feature_affinity(self.tmp_dir, ckey, "adx_14", schema=self.mock_schema)
        res_rsi = analyze_feature_affinity(self.tmp_dir, ckey, "rsi_14", schema=self.mock_schema)

        self.assertEqual(res_adx.evidence_count, 1)
        self.assertEqual(res_rsi.evidence_count, 1)
        self.assertEqual(res_adx.robustness_support, 85.0)
        self.assertEqual(res_rsi.robustness_support, 85.0)

    def test_04_feature_score_bounds(self):
        """4. Verify feature affinity score is strictly bounded in [0.0, 100.0]."""
        for r in [-100.0, 0.0, 50.0, 100.0, 500.0]:
            for e in [-50.0, 0.0, 50.0, 100.0, 200.0]:
                for s in [0.0, 100.0]:
                    for p in [0.0, 100.0]:
                        score = compute_feature_affinity_score(
                            robustness_support=r,
                            evidence_support=e,
                            stability_support=s,
                            population_support=p,
                        )
                        self.assertGreaterEqual(score, 0.0)
                        self.assertLessEqual(score, 100.0)

    def test_05_confidence_zero_with_zero_experiments(self):
        """5. Verify confidence is exactly 0.0 when experiment count is 0."""
        self.assertEqual(compute_feature_confidence(0), 0.0)
        self.assertEqual(compute_feature_confidence(-5), 0.0)

    def test_06_confidence_scaling(self):
        """6. Verify confidence scales monotonically with experiment count."""
        c1 = compute_feature_confidence(1)
        c5 = compute_feature_confidence(5)
        c15 = compute_feature_confidence(15)

        self.assertLess(c1, c5)
        self.assertLess(c5, c15)
        self.assertGreaterEqual(c1, 0.15)
        self.assertGreaterEqual(c5, 0.60)
        self.assertGreaterEqual(c15, 0.90)

    def test_07_sparse_evidence_class(self):
        """7. Verify sparse evidence (1 experiment) yields EXPLORATORY class even if score is high."""
        aff_score = 85.0
        conf = compute_feature_confidence(1) # ~0.1813
        rec = classify_feature_recommendation(aff_score, conf)
        self.assertEqual(rec, FeatureRecommendationClass.EXPLORATORY)

    def test_08_developing_evidence_class(self):
        """8. Verify developing evidence (2-4 experiments) with high score yields PROMISING class."""
        aff_score = 75.0
        conf = compute_feature_confidence(3) # ~0.4512
        rec = classify_feature_recommendation(aff_score, conf)
        self.assertEqual(rec, FeatureRecommendationClass.PROMISING)

    def test_09_mature_evidence_class(self):
        """9. Verify mature evidence (5+ experiments) with high score yields CONFIRMED class."""
        aff_score = 75.0
        conf = compute_feature_confidence(7) # ~0.7534
        rec = classify_feature_recommendation(aff_score, conf)
        self.assertEqual(rec, FeatureRecommendationClass.CONFIRMED)

    def test_10_deprecated_feature_quarantine(self):
        """10. Verify deprecated features receive affinity score 0.0 and QUARANTINED recommendation."""
        ckey = "NIFTY_3s_DIRECTION_CLASSIFIER_5m_R001"
        res = analyze_feature_affinity(self.tmp_dir, ckey, "dep_feature", schema=self.mock_schema)

        self.assertEqual(res.feature_population, "DEPRECATED")
        self.assertEqual(res.affinity_score, 0.0)
        self.assertEqual(res.recommendation_class, FeatureRecommendationClass.QUARANTINED)

    def test_11_unknown_feature_handling(self):
        """11. Verify unmapped/unknown features receive population UNKNOWN and reduced population support."""
        ckey = "NIFTY_3s_DIRECTION_CLASSIFIER_5m_R001"
        res = analyze_feature_affinity(self.tmp_dir, ckey, "mystery_feat", schema=self.mock_schema)

        self.assertEqual(res.feature_population, "UNKNOWN")
        self.assertEqual(res.population_support, 20.0)

    def test_12_base_feature_population(self):
        """12. Verify Base Pipeline features receive population BASE and population support 100.0."""
        ckey = "NIFTY_3s_DIRECTION_CLASSIFIER_5m_R001"
        res = analyze_feature_affinity(self.tmp_dir, ckey, "adx_14", schema=self.mock_schema)
        self.assertEqual(res.feature_population, "BASE")
        self.assertEqual(res.population_support, 100.0)

    def test_13_registry_feature_population(self):
        """13. Verify Canonical Registry features receive population REGISTRY and population support 85.0."""
        ckey = "NIFTY_3s_DIRECTION_CLASSIFIER_5m_R001"
        res = analyze_feature_affinity(self.tmp_dir, ckey, "rsi_14", schema=self.mock_schema)
        self.assertEqual(res.feature_population, "REGISTRY")
        self.assertEqual(res.population_support, 85.0)

    def test_14_experimental_feature_population(self):
        """14. Verify candidate experimental features receive population EXPERIMENTAL and population support 70.0."""
        ckey = "NIFTY_3s_DIRECTION_CLASSIFIER_5m_R001"
        res = analyze_feature_affinity(self.tmp_dir, ckey, "exp_alpha", schema=self.mock_schema)
        self.assertEqual(res.feature_population, "EXPERIMENTAL")
        self.assertEqual(res.population_support, 70.0)

    def test_15_context_isolation_by_regime(self):
        """15. Verify complete feature evidence isolation between R001 (Trend) and R002 (Sideways)."""
        ctx_r1 = "NIFTY_3s_DIRECTION_CLASSIFIER_5m_R001"
        ctx_r2 = "NIFTY_3s_DIRECTION_CLASSIFIER_5m_R002"

        b_run = create_benchmark_run(self.tmp_dir, context_key=ctx_r1)
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
        register_or_get_experiment(self.tmp_dir, spec, model_name="M_R1")
        record_model_benchmark(
            self.tmp_dir,
            benchmark_run_id=b_run,
            signature_hash=sig_h,
            model_name="M_R1",
            context_key=ctx_r1,
            algorithm="xgboost",
            dataset_name="d.parquet",
            feature_count=1,
            primary_metric_name="roc_auc",
            primary_metric_value=0.90,
            robustness_score=90.0,
        )

        res_r1 = analyze_feature_affinity(self.tmp_dir, ctx_r1, "adx_14", schema=self.mock_schema)
        res_r2 = analyze_feature_affinity(self.tmp_dir, ctx_r2, "adx_14", schema=self.mock_schema)

        self.assertEqual(res_r1.evidence_count, 1)
        self.assertEqual(res_r1.robustness_support, 90.0)

        # R002 must remain 0 experiments
        self.assertEqual(res_r2.evidence_count, 0)
        self.assertEqual(res_r2.confidence, 0.0)

    def test_16_context_isolation_by_task(self):
        """16. Verify isolation between DIRECTION_CLASSIFIER and VOLATILITY_ESTIMATOR."""
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
            "algorithm": "xgboost",
        }
        sig_h, _, _ = compute_experiment_signature(spec)
        register_or_get_experiment(self.tmp_dir, spec, model_name="M_DIR")
        record_model_benchmark(
            self.tmp_dir,
            benchmark_run_id=b_run,
            signature_hash=sig_h,
            model_name="M_DIR",
            context_key=ctx_dir,
            algorithm="xgboost",
            dataset_name="d.parquet",
            feature_count=1,
            primary_metric_name="roc_auc",
            primary_metric_value=0.85,
            robustness_score=85.0,
        )

        self.assertEqual(analyze_feature_affinity(self.tmp_dir, ctx_dir, "adx_14", schema=self.mock_schema).evidence_count, 1)
        self.assertEqual(analyze_feature_affinity(self.tmp_dir, ctx_vol, "adx_14", schema=self.mock_schema).evidence_count, 0)

    def test_17_context_isolation_by_horizon(self):
        """17. Verify isolation across prediction horizons (5m vs 15m)."""
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
            primary_metric_value=0.80,
            robustness_score=80.0,
        )

        self.assertEqual(analyze_feature_affinity(self.tmp_dir, ctx_5m, "adx_14", schema=self.mock_schema).evidence_count, 1)
        self.assertEqual(analyze_feature_affinity(self.tmp_dir, ctx_15m, "adx_14", schema=self.mock_schema).evidence_count, 0)

    def test_18_context_isolation_by_sampling_interval(self):
        """18. Verify isolation across sampling intervals (3s vs 5s)."""
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
            primary_metric_value=0.80,
            robustness_score=80.0,
        )

        self.assertEqual(analyze_feature_affinity(self.tmp_dir, ctx_3s, "adx_14", schema=self.mock_schema).evidence_count, 1)
        self.assertEqual(analyze_feature_affinity(self.tmp_dir, ctx_5s, "adx_14", schema=self.mock_schema).evidence_count, 0)

    def test_19_context_isolation_by_market(self):
        """19. Verify isolation across markets (NIFTY vs BANKNIFTY)."""
        ctx_n = "NIFTY_3s_DIRECTION_CLASSIFIER_5m_R001"
        ctx_bn = "BANKNIFTY_3s_DIRECTION_CLASSIFIER_5m_R001"

        b_run = create_benchmark_run(self.tmp_dir, context_key=ctx_n)
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
        register_or_get_experiment(self.tmp_dir, spec, model_name="M_N")
        record_model_benchmark(
            self.tmp_dir,
            benchmark_run_id=b_run,
            signature_hash=sig_h,
            model_name="M_N",
            context_key=ctx_n,
            algorithm="xgboost",
            dataset_name="d.parquet",
            feature_count=1,
            primary_metric_name="roc_auc",
            primary_metric_value=0.80,
            robustness_score=80.0,
        )

        self.assertEqual(analyze_feature_affinity(self.tmp_dir, ctx_n, "adx_14", schema=self.mock_schema).evidence_count, 1)
        self.assertEqual(analyze_feature_affinity(self.tmp_dir, ctx_bn, "adx_14", schema=self.mock_schema).evidence_count, 0)

    def test_20_pairwise_interaction_detection(self):
        """20. Verify pairwise synergy lift detection when pair outperforms individual features."""
        lift, score, conf = compute_interaction_synergy_score(
            pair_mean_robustness=85.0,
            max_individual_robustness=75.0,
            pair_experiment_count=4,
        )
        self.assertEqual(lift, 10.0)
        self.assertEqual(score, 75.0) # 50 + (10 * 2.5) = 75.0
        self.assertGreater(conf, 0.70)

    def test_21_negative_interaction_lift(self):
        """21. Verify negative synergy lift detection when pair underperforms individual features."""
        lift, score, conf = compute_interaction_synergy_score(
            pair_mean_robustness=60.0,
            max_individual_robustness=75.0,
            pair_experiment_count=3,
        )
        self.assertEqual(lift, -15.0)
        self.assertEqual(score, 12.5) # 50 + (-15 * 2.5) = 12.5

    def test_22_sparse_pair_confidence(self):
        """22. Verify pair interaction confidence is low when pair experiment count is 1."""
        _, _, conf = compute_interaction_synergy_score(80.0, 70.0, 1)
        self.assertLess(conf, 0.30)

    def test_23_deterministic_ordering(self):
        """23. Verify recommendations are sorted deterministically by (-score, -confidence, feature_name)."""
        ckey = "NIFTY_3s_DIRECTION_CLASSIFIER_5m_R001"
        rep1 = recommend_features_for_context(self.tmp_dir, ckey, schema=self.mock_schema)
        rep2 = recommend_features_for_context(self.tmp_dir, ckey, schema=self.mock_schema)

        feats1 = [f.to_dict() for f in rep1.recommended_features]
        feats2 = [f.to_dict() for f in rep2.recommended_features]
        self.assertEqual(feats1, feats2)

    def test_24_missing_evidence_handling(self):
        """24. Verify missing evidence gracefully falls back to neutral scores."""
        score = compute_feature_affinity_score(
            robustness_support=None,
            evidence_support=None,
            stability_support=None,
            population_support=None,
        )
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 100.0)

    def test_25_nan_and_infinity_safety(self):
        """25. Verify compute_feature_affinity_score handles NaN and Infinity safely."""
        s_nan = compute_feature_affinity_score(
            robustness_support=float("nan"),
            evidence_support=float("nan"),
            stability_support=float("nan"),
            population_support=float("nan"),
        )
        self.assertFalse(math.isnan(s_nan))
        self.assertGreaterEqual(s_nan, 0.0)
        self.assertLessEqual(s_nan, 100.0)

        s_inf = compute_feature_affinity_score(
            robustness_support=float("inf"),
            evidence_support=float("inf"),
            stability_support=float("inf"),
            population_support=float("inf"),
        )
        self.assertEqual(s_inf, 100.0)

    def test_26_champion_missing_feature_opportunity(self):
        """26. Verify identification of high-affinity features missing from production champion."""
        ckey = "NIFTY_3s_DIRECTION_CLASSIFIER_5m_R001"
        champ_name = "CHAMP_MINIMAL"
        set_champion_for_context(self.tmp_dir, ckey, champ_name)

        b_run = create_benchmark_run(self.tmp_dir, context_key=ckey)
        # Champ only has adx_14
        spec_c = {
            "market": "NIFTY",
            "sampling_interval_sec": 3,
            "task_type": "DIRECTION_CLASSIFIER",
            "prediction_horizon": "5m",
            "regime_id": "R001",
            "features": ["adx_14"],
            "algorithm": "xgboost",
        }
        sig_c, _, _ = compute_experiment_signature(spec_c)
        register_or_get_experiment(self.tmp_dir, spec_c, model_name=champ_name)
        record_model_benchmark(
            self.tmp_dir,
            benchmark_run_id=b_run,
            signature_hash=sig_c,
            model_name=champ_name,
            context_key=ckey,
            algorithm="xgboost",
            dataset_name="d.parquet",
            feature_count=1,
            primary_metric_name="roc_auc",
            primary_metric_value=0.70,
            robustness_score=70.0,
        )

        # 5 experiments with rsi_14 achieving 88.0 robustness (Matures confidence to CONFIRMED)
        for i in range(5):
            spec_r = {
                "market": "NIFTY",
                "sampling_interval_sec": 3,
                "task_type": "DIRECTION_CLASSIFIER",
                "prediction_horizon": "5m",
                "regime_id": "R001",
                "features": ["rsi_14"],
                "algorithm": "xgboost",
                "hyperparameters": {"depth": 3 + i},
            }
            sig_r, _, _ = compute_experiment_signature(spec_r)
            register_or_get_experiment(self.tmp_dir, spec_r, model_name=f"R_MODEL_{i}")
            record_model_benchmark(
                self.tmp_dir,
                benchmark_run_id=b_run,
                signature_hash=sig_r,
                model_name=f"R_MODEL_{i}",
                context_key=ckey,
                algorithm="xgboost",
                dataset_name="d.parquet",
                feature_count=1,
                primary_metric_name="roc_auc",
                primary_metric_value=0.88,
                robustness_score=88.0,
            )

        report = recommend_features_for_context(self.tmp_dir, ckey, schema=self.mock_schema)
        self.assertIn("rsi_14", report.missing_champion_feature_opportunities)

    def test_27_research_candidate_feature_opportunity(self):
        """27. Verify top research candidates contribute to feature interaction detection."""
        ckey = "NIFTY_3s_DIRECTION_CLASSIFIER_5m_R001"
        b_run = create_benchmark_run(self.tmp_dir, context_key=ckey)
        spec = {
            "market": "NIFTY",
            "sampling_interval_sec": 3,
            "task_type": "DIRECTION_CLASSIFIER",
            "prediction_horizon": "5m",
            "regime_id": "R001",
            "features": ["adx_14", "exp_alpha"],
            "algorithm": "xgboost",
        }
        sig_h, _, _ = compute_experiment_signature(spec)
        register_or_get_experiment(self.tmp_dir, spec, model_name="CAND_TOP")
        record_model_benchmark(
            self.tmp_dir,
            benchmark_run_id=b_run,
            signature_hash=sig_h,
            model_name="CAND_TOP",
            context_key=ckey,
            algorithm="xgboost",
            dataset_name="d.parquet",
            feature_count=2,
            primary_metric_name="roc_auc",
            primary_metric_value=0.92,
            robustness_score=92.0,
        )

        report = recommend_features_for_context(self.tmp_dir, ckey, schema=self.mock_schema)
        pair_sets = [set(i.feature_set) for i in report.interaction_recommendations]
        self.assertIn({"adx_14", "exp_alpha"}, pair_sets)

    def test_28_production_immutability(self):
        """28. Assert that feature affinity recommendations never mutate production state."""
        ev_path = "apps/feature_recommendation_evidence.db"
        self.assertTrue(os.path.isfile(ev_path), "Evidence DB must exist")

        with open(ev_path, "rb") as fh:
            sha_initial = hashlib.sha256(fh.read()).hexdigest()

        # Run feature recommendations
        ckey = "NIFTY_3s_DIRECTION_CLASSIFIER_5m_R001"
        _ = recommend_features_for_context(self.tmp_dir, ckey, schema=self.mock_schema)

        with open(ev_path, "rb") as fh:
            sha_final = hashlib.sha256(fh.read()).hexdigest()

        self.assertEqual(sha_initial, sha_final, "Evidence DB must remain 100% unmutated")

    def test_29_evidence_db_immutability(self):
        """29. Assert that feature affinity queries never insert or modify evidence tables."""
        ev_path = "apps/feature_recommendation_evidence.db"
        with open(ev_path, "rb") as fh:
            sha_before = hashlib.sha256(fh.read()).hexdigest()

        for f in ["adx_14", "rsi_14", "dep_feature", "unknown_xyz"]:
            _ = analyze_feature_affinity(self.tmp_dir, "NIFTY_3s_DIRECTION_CLASSIFIER_5m_R001", f, schema=self.mock_schema)

        with open(ev_path, "rb") as fh:
            sha_after = hashlib.sha256(fh.read()).hexdigest()

        self.assertEqual(sha_before, sha_after)

    def test_30_full_report_generation(self):
        """30. Verify complete ContextFeatureAffinityReport serialization and summary counts."""
        ckey = "NIFTY_3s_DIRECTION_CLASSIFIER_5m_R001"
        report = recommend_features_for_context(self.tmp_dir, ckey, schema=self.mock_schema)

        self.assertEqual(report.context_key, ckey)
        self.assertEqual(report.market, "NIFTY")
        self.assertEqual(report.sampling_interval_sec, 3)
        self.assertEqual(report.task_type, "DIRECTION_CLASSIFIER")
        self.assertEqual(report.prediction_horizon, "5m")
        self.assertEqual(report.regime_id, "R001")
        self.assertGreater(report.total_features_analyzed, 0)
        self.assertIn("dep_feature", report.excluded_features)

        d = report.to_dict()
        self.assertIsInstance(d, dict)
        self.assertEqual(d["context_key"], ckey)


if __name__ == "__main__":
    unittest.main()

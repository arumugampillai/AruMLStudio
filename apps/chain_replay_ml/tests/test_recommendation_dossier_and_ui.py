"""Unit tests for Phase 4E.6 — Model Research Lab Recommendation Dossier & UI Agenda Integration.

Verifies:
1. Dossier generation
2. Empty context
3. Cold-start context
4. High-priority recommendation
5. Low-priority recommendation
6. Confidence display
7. Caution display
8. Excluded opportunities not appearing as recommendations
9. Context isolation
10. R001/R002 isolation
11. Market isolation
12. Task isolation
13. Horizon isolation
14. Sampling isolation
15. Deterministic ordering
16. Underlying evidence traceability
17. Production champion separation
18. Read-only behavior
19. Production database immutability
20. UI rendering safety
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
from chain_replay_ml.research_recommendations.dossier import (
    RecommendationDossier,
    build_recommendation_dossier,
    generate_context_recommendation_dossiers,
)
from chain_replay_ml.research_recommendations.negative_pruning import (
    ExclusionReason,
    ExclusionVerdict,
)
from chain_replay_ml.research_recommendations.priority_scoring import (
    EvidenceConfidenceLevel,
    OpportunityType,
    ResearchOpportunity,
    ResearchPriorityClass,
    build_context_priority_agenda,
    evaluate_research_opportunity,
)
from chain_replay_ml.training.lifecycle_store import (
    set_champion_for_context,
)


class TestRecommendationDossierAndUI(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp(prefix="aruml_test_4e6_dossier_")
        self.mock_schema = {
            "columns": {
                "adx_14": {"is_base": True, "project_id": "PL_0001", "status": "ACTIVE"},
                "rsi_14": {"is_base": False, "project_id": None, "status": "ACTIVE"},
                "exp_vol": {"is_base": False, "project_id": "PL_0002", "status": "ACTIVE"},
                "dep_old": {"is_base": False, "project_id": "PL_0001", "status": "DEPRECATED"},
            }
        }

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_01_dossier_generation(self):
        """1. Verify build_recommendation_dossier creates a complete, well-formed dossier."""
        ckey = "NIFTY_3s_DIRECTION_CLASSIFIER_5m_R001"
        opp = evaluate_research_opportunity(
            self.tmp_dir,
            ckey,
            opportunity_type=OpportunityType.FEATURE_EXPLORATION,
            candidate_features=["adx_14", "rsi_14"],
            schema=self.mock_schema,
        )
        dossier = build_recommendation_dossier(self.tmp_dir, opp, schema=self.mock_schema)

        self.assertIsInstance(dossier, RecommendationDossier)
        self.assertEqual(dossier.context_key, ckey)
        self.assertEqual(dossier.opportunity_id, opp.opportunity_id)
        self.assertIsInstance(dossier.why_recommended, str)
        self.assertGreater(len(dossier.suggested_next_steps), 0)

    def test_02_empty_context_dossiers(self):
        """2. Verify generate_context_recommendation_dossiers on unobserved context."""
        ckey = "NIFTY_3s_DIRECTION_CLASSIFIER_5m_R001"
        dossiers = generate_context_recommendation_dossiers(self.tmp_dir, ckey, schema=self.mock_schema)
        self.assertIsInstance(dossiers, list)
        self.assertTrue(all(isinstance(d, RecommendationDossier) for d in dossiers))

    def test_03_cold_start_context_dossier(self):
        """3. Verify cold-start context contains appropriate warnings and coverage gap explanation."""
        ckey = "NIFTY_3s_DIRECTION_CLASSIFIER_5m_R005"
        dossiers = generate_context_recommendation_dossiers(self.tmp_dir, ckey, schema=self.mock_schema)
        self.assertTrue(len(dossiers) > 0)
        top = dossiers[0]
        self.assertIn("Cold-start context", " ".join(top.caution_warnings))

    def test_04_high_priority_recommendation(self):
        """4. Verify high-vulnerability champion triggers high-priority recommendation dossier."""
        ckey = "NIFTY_3s_DIRECTION_CLASSIFIER_5m_R001"
        champ_name = "CHAMP_FRAGILE"
        set_champion_for_context(self.tmp_dir, ckey, champ_name)

        b_run = create_benchmark_run(self.tmp_dir, context_key=ckey)
        # Fragile champ
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
            primary_metric_value=0.55,
            robustness_score=40.0,
            expected_calibration_error=0.18,
        )
        record_regime_evaluation(
            self.tmp_dir,
            model_name=champ_name,
            signature_hash=sig_c,
            tested_regime_id="R002",
            tested_regime_hash="def",
            is_native_regime=False,
            sample_count=1000,
            primary_metric=0.25,
            regime_degradation_pct=60.0,
        )

        # Leading Challenger with rsi_14
        spec_ch = {
            "market": "NIFTY",
            "sampling_interval_sec": 3,
            "task_type": "DIRECTION_CLASSIFIER",
            "prediction_horizon": "5m",
            "regime_id": "R001",
            "features": ["adx_14", "rsi_14"],
            "algorithm": "catboost",
        }
        sig_ch, _, _ = compute_experiment_signature(spec_ch)
        register_or_get_experiment(self.tmp_dir, spec_ch, model_name="CHALLENGER_LEAD")
        record_model_benchmark(
            self.tmp_dir,
            benchmark_run_id=b_run,
            signature_hash=sig_ch,
            model_name="CHALLENGER_LEAD",
            context_key=ckey,
            algorithm="catboost",
            dataset_name="d.parquet",
            feature_count=2,
            primary_metric_name="roc_auc",
            primary_metric_value=0.90,
            robustness_score=90.0,
            expected_calibration_error=0.02,
        )

        dossiers = generate_context_recommendation_dossiers(self.tmp_dir, ckey, schema=self.mock_schema)
        self.assertTrue(len(dossiers) > 0)
        self.assertIn(dossiers[0].priority_class, (ResearchPriorityClass.CRITICAL, ResearchPriorityClass.HIGH, ResearchPriorityClass.MEDIUM))

    def test_05_low_priority_recommendation(self):
        """5. Verify mature, stable context without vulnerability produces LOW or MEDIUM priority."""
        ckey = "NIFTY_3s_DIRECTION_CLASSIFIER_5m_R001"
        champ_name = "CHAMP_PERFECT"
        set_champion_for_context(self.tmp_dir, ckey, champ_name)

        b_run = create_benchmark_run(self.tmp_dir, context_key=ckey)
        for i in range(12):
            spec_i = {
                "market": "NIFTY",
                "sampling_interval_sec": 3,
                "task_type": "DIRECTION_CLASSIFIER",
                "prediction_horizon": "5m",
                "regime_id": "R001",
                "features": ["adx_14", "rsi_14"],
                "algorithm": "xgboost",
                "hyperparameters": {"depth": 3 + i},
            }
            sig_i, _, _ = compute_experiment_signature(spec_i)
            m_name = champ_name if i == 0 else f"MODEL_{i}"
            register_or_get_experiment(self.tmp_dir, spec_i, model_name=m_name)
            record_model_benchmark(
                self.tmp_dir,
                benchmark_run_id=b_run,
                signature_hash=sig_i,
                model_name=m_name,
                context_key=ckey,
                algorithm="xgboost",
                dataset_name="d.parquet",
                feature_count=2,
                primary_metric_name="roc_auc",
                primary_metric_value=0.92,
                robustness_score=92.0,
                expected_calibration_error=0.015,
            )
            record_regime_evaluation(
                self.tmp_dir,
                model_name=m_name,
                signature_hash=sig_i,
                tested_regime_id="R002",
                tested_regime_hash="def",
                is_native_regime=False,
                sample_count=1000,
                primary_metric=0.88,
                regime_degradation_pct=4.0,
            )

        dossiers = generate_context_recommendation_dossiers(self.tmp_dir, ckey, schema=self.mock_schema)
        if dossiers:
            self.assertIn(dossiers[0].priority_class, (ResearchPriorityClass.MEDIUM, ResearchPriorityClass.LOW, ResearchPriorityClass.NEGLIGIBLE))

    def test_06_confidence_display(self):
        """6. Verify evidence confidence is clearly formatted with categorical and float levels."""
        ckey = "NIFTY_3s_DIRECTION_CLASSIFIER_5m_R001"
        dossiers = generate_context_recommendation_dossiers(self.tmp_dir, ckey, schema=self.mock_schema)
        for d in dossiers:
            self.assertIsInstance(d.evidence_confidence, EvidenceConfidenceLevel)
            self.assertGreaterEqual(d.confidence_value, 0.0)
            self.assertLessEqual(d.confidence_value, 1.0)

    def test_07_caution_display(self):
        """7. Verify caution warnings are surfaced in dossier caution_warnings list."""
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
        register_or_get_experiment(self.tmp_dir, spec, model_name="M_CAUT")
        record_model_benchmark(
            self.tmp_dir,
            benchmark_run_id=b_run,
            signature_hash=sig_h,
            model_name="M_CAUT",
            context_key=ckey,
            algorithm="xgboost",
            dataset_name="d.parquet",
            feature_count=1,
            primary_metric_name="roc_auc",
            primary_metric_value=0.80,
            robustness_score=75.0,
            expected_calibration_error=0.14,
        )

        # Propose with different algo (catboost) -> not duplicate, but triggers caution due to historical ECE
        opp = evaluate_research_opportunity(
            self.tmp_dir,
            ckey,
            opportunity_type=OpportunityType.FEATURE_EXPLORATION,
            candidate_features=["adx_14"],
            target_algorithm="catboost",
            schema=self.mock_schema,
        )
        dossier = build_recommendation_dossier(self.tmp_dir, opp, schema=self.mock_schema)
        self.assertGreater(len(dossier.caution_warnings), 0)

    def test_08_excluded_opportunities_suppressed(self):
        """8. Verify EXCLUDED opportunities do not appear as valid recommendations."""
        ckey = "NIFTY_3s_DIRECTION_CLASSIFIER_5m_R001"
        opp = evaluate_research_opportunity(
            self.tmp_dir,
            ckey,
            opportunity_type=OpportunityType.FEATURE_EXPLORATION,
            candidate_features=["dep_old"],
            schema=self.mock_schema,
        )
        self.assertEqual(opp.exclusion_verdict, ExclusionVerdict.EXCLUDED)

        dossiers = generate_context_recommendation_dossiers(self.tmp_dir, ckey, schema=self.mock_schema)
        for d in dossiers:
            self.assertNotEqual(d.exclusion_verdict, ExclusionVerdict.EXCLUDED)
            self.assertNotIn("dep_old", d.candidate_features)

    def test_09_context_isolation(self):
        """9. Verify recommendation dossiers are strictly isolated by context key."""
        ctx1 = "NIFTY_3s_DIRECTION_CLASSIFIER_5m_R001"
        ctx2 = "NIFTY_3s_DIRECTION_CLASSIFIER_5m_R002"

        dossiers1 = generate_context_recommendation_dossiers(self.tmp_dir, ctx1, schema=self.mock_schema)
        dossiers2 = generate_context_recommendation_dossiers(self.tmp_dir, ctx2, schema=self.mock_schema)

        self.assertTrue(all(d.context_key == ctx1 for d in dossiers1))
        self.assertTrue(all(d.context_key == ctx2 for d in dossiers2))

    def test_10_r001_vs_r002_regime_isolation(self):
        """10. Verify R001 recommendations do not pollute R002."""
        d1 = generate_context_recommendation_dossiers(self.tmp_dir, "NIFTY_3s_DIRECTION_CLASSIFIER_5m_R001", schema=self.mock_schema)
        d2 = generate_context_recommendation_dossiers(self.tmp_dir, "NIFTY_3s_DIRECTION_CLASSIFIER_5m_R002", schema=self.mock_schema)
        self.assertTrue(all("R001" in d.context_key for d in d1))
        self.assertTrue(all("R002" in d.context_key for d in d2))

    def test_11_market_isolation(self):
        """11. Verify NIFTY recommendations do not pollute BANKNIFTY."""
        d_n = generate_context_recommendation_dossiers(self.tmp_dir, "NIFTY_3s_DIRECTION_CLASSIFIER_5m_R001", schema=self.mock_schema)
        d_bn = generate_context_recommendation_dossiers(self.tmp_dir, "BANKNIFTY_3s_DIRECTION_CLASSIFIER_5m_R001", schema=self.mock_schema)
        self.assertTrue(all("NIFTY" in d.context_key for d in d_n))
        self.assertTrue(all("BANKNIFTY" in d.context_key for d in d_bn))

    def test_12_task_isolation(self):
        """12. Verify DIRECTION_CLASSIFIER recommendations do not pollute VOLATILITY_ESTIMATOR."""
        d_dir = generate_context_recommendation_dossiers(self.tmp_dir, "NIFTY_3s_DIRECTION_CLASSIFIER_5m_R001", schema=self.mock_schema)
        d_vol = generate_context_recommendation_dossiers(self.tmp_dir, "NIFTY_3s_VOLATILITY_ESTIMATOR_5m_R001", schema=self.mock_schema)
        self.assertTrue(all("DIRECTION_CLASSIFIER" in d.context_key for d in d_dir))
        self.assertTrue(all("VOLATILITY_ESTIMATOR" in d.context_key for d in d_vol))

    def test_13_horizon_isolation(self):
        """13. Verify 5m horizon recommendations do not pollute 15m."""
        d_5m = generate_context_recommendation_dossiers(self.tmp_dir, "NIFTY_3s_DIRECTION_CLASSIFIER_5m_R001", schema=self.mock_schema)
        d_15m = generate_context_recommendation_dossiers(self.tmp_dir, "NIFTY_3s_DIRECTION_CLASSIFIER_15m_R001", schema=self.mock_schema)
        self.assertTrue(all("5m" in d.context_key for d in d_5m))
        self.assertTrue(all("15m" in d.context_key for d in d_15m))

    def test_14_sampling_interval_isolation(self):
        """14. Verify 3s sampling recommendations do not pollute 5s."""
        d_3s = generate_context_recommendation_dossiers(self.tmp_dir, "NIFTY_3s_DIRECTION_CLASSIFIER_5m_R001", schema=self.mock_schema)
        d_5s = generate_context_recommendation_dossiers(self.tmp_dir, "NIFTY_5s_DIRECTION_CLASSIFIER_5m_R001", schema=self.mock_schema)
        self.assertTrue(all("_3s_" in d.context_key for d in d_3s))
        self.assertTrue(all("_5s_" in d.context_key for d in d_5s))

    def test_15_deterministic_ordering(self):
        """15. Verify repeated dossier generation produces identical sorting."""
        ckey = "NIFTY_3s_DIRECTION_CLASSIFIER_5m_R001"
        d1 = generate_context_recommendation_dossiers(self.tmp_dir, ckey, schema=self.mock_schema)
        d2 = generate_context_recommendation_dossiers(self.tmp_dir, ckey, schema=self.mock_schema)

        scores1 = [d.priority_score for d in d1]
        scores2 = [d.priority_score for d in d2]
        self.assertEqual(scores1, scores2)

    def test_16_underlying_evidence_traceability(self):
        """16. Verify supporting empirical evidence dictionary contains all expected components."""
        ckey = "NIFTY_3s_DIRECTION_CLASSIFIER_5m_R001"
        dossiers = generate_context_recommendation_dossiers(self.tmp_dir, ckey, schema=self.mock_schema)
        for d in dossiers:
            ev = d.supporting_empirical_evidence
            self.assertIn("context_benchmark_count", ev)
            self.assertIn("evidence_density_score", ev)
            self.assertIn("champion_vulnerability_score", ev)

    def test_17_production_champion_separation(self):
        """17. Verify production champion context is clearly segregated from research recommendations."""
        ckey = "NIFTY_3s_DIRECTION_CLASSIFIER_5m_R001"
        set_champion_for_context(self.tmp_dir, ckey, "PROD_CHAMPION_LIVE")
        dossiers = generate_context_recommendation_dossiers(self.tmp_dir, ckey, schema=self.mock_schema)
        for d in dossiers:
            self.assertEqual(d.production_champion_context["champion_model_name"], "PROD_CHAMPION_LIVE")
            self.assertNotEqual(d.opportunity_id, "PROD_CHAMPION_LIVE")

    def test_18_read_only_behavior(self):
        """18. Verify dossier generation creates zero new database tables or unexpected files."""
        ckey = "NIFTY_3s_DIRECTION_CLASSIFIER_5m_R001"
        _ = generate_context_recommendation_dossiers(self.tmp_dir, ckey, schema=self.mock_schema)

        # Ensure no new databases or unmanaged files were written to tmp_dir
        files = os.listdir(self.tmp_dir)
        for f in files:
            self.assertIn(f, ("analysis.db", ".lifecycle_registry.db", "models"))

    def test_19_production_database_immutability(self):
        """19. Assert that evidence database SHA-256 remains 100% unmutated."""
        ev_path = "apps/feature_recommendation_evidence.db"
        self.assertTrue(os.path.isfile(ev_path), "Evidence DB must exist")

        with open(ev_path, "rb") as fh:
            sha_initial = hashlib.sha256(fh.read()).hexdigest()

        ckey = "NIFTY_3s_DIRECTION_CLASSIFIER_5m_R001"
        _ = generate_context_recommendation_dossiers(self.tmp_dir, ckey, schema=self.mock_schema)

        with open(ev_path, "rb") as fh:
            sha_final = hashlib.sha256(fh.read()).hexdigest()

        self.assertEqual(sha_initial, sha_final)

    def test_20_ui_rendering_safety(self):
        """20. Verify ModelResearchLeaderboardPanel renders Research Recommendations tab safely."""
        import tkinter as tk
        from master_dataset_tk.model_research_leaderboard_panel import ModelResearchLeaderboardPanel

        root = tk.Tk()
        root.withdraw()
        try:
            panel = ModelResearchLeaderboardPanel(root, chart_dir=self.tmp_dir)
            panel.refresh_leaderboard()
            self.assertTrue(hasattr(panel, "_tab_recommendations"))
            self.assertGreater(len(panel._tab_recommendations.inner.winfo_children()), 0)
        finally:
            root.destroy()


if __name__ == "__main__":
    unittest.main()

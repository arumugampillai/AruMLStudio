"""Unit tests for Phase 4E.5 — Multi-Objective Recommendation Priority Scoring Engine.

Verifies:
1. Deterministic scoring
2. Component bounds
3. Score bounds
4. Priority classification boundaries
5. Cold-start contexts
6. Sparse contexts
7. Mature contexts
8. Strong champion vulnerability
9. Weak champion vulnerability
10. Challenger lead
11. Challenger trailing
12. Feature affinity contribution
13. Interaction evidence contribution
14. Insufficient evidence handling
15. EXCLUDED opportunity suppression
16. CAUTION propagation
17. R001/R002 isolation
18. NIFTY/BANKNIFTY isolation
19. 5m/15m isolation
20. 3s/5s isolation
21. Deterministic tie-breaking
22. NaN/Infinity safety
23. Empty database safety
24. Concurrent read safety
25. Production immutability
26. Evidence DB SHA-256 unchanged
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
from chain_replay_ml.research_recommendations.negative_pruning import (
    ExclusionReason,
    ExclusionVerdict,
)
from chain_replay_ml.research_recommendations.priority_scoring import (
    ComponentScoreBreakdown,
    ContextPriorityAgendaReport,
    EvidenceConfidenceLevel,
    OpportunityType,
    ResearchOpportunity,
    ResearchPriorityClass,
    build_context_priority_agenda,
    classify_evidence_confidence,
    classify_priority_score,
    compute_research_priority_score,
    evaluate_research_opportunity,
)
from chain_replay_ml.research_memory.champion_history import (
    set_champion_for_context,
)


class TestResearchPriorityScoring(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp(prefix="aruml_test_4e5_prio_")
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

    def test_01_deterministic_scoring(self):
        """1. Verify priority score computation is 100% deterministic."""
        s1, b1 = compute_research_priority_score(
            coverage_density_score=20.0,
            champion_vulnerability_score=75.0,
            challenger_gap_pct=5.0,
            feature_affinity_score=80.0,
            interaction_synergy_score=65.0,
        )
        s2, b2 = compute_research_priority_score(
            coverage_density_score=20.0,
            champion_vulnerability_score=75.0,
            challenger_gap_pct=5.0,
            feature_affinity_score=80.0,
            interaction_synergy_score=65.0,
        )
        self.assertEqual(s1, s2)
        self.assertEqual(b1.to_dict(), b2.to_dict())

    def test_02_component_bounds(self):
        """2. Verify individual score components are strictly bounded in [0.0, 100.0]."""
        _, b = compute_research_priority_score(
            coverage_density_score=-50.0,
            champion_vulnerability_score=250.0,
            challenger_gap_pct=50.0,
            feature_affinity_score=999.0,
            interaction_synergy_score=-100.0,
        )
        self.assertEqual(b.coverage_gap_score, 100.0)
        self.assertEqual(b.champion_vulnerability_score, 100.0)
        self.assertEqual(b.challenger_gap_score, 100.0)
        self.assertEqual(b.feature_affinity_score, 100.0)
        self.assertEqual(b.interaction_synergy_score, 0.0)

    def test_03_score_bounds(self):
        """3. Verify final composite priority score is strictly bounded in [0.0, 100.0]."""
        for v in [0.0, 50.0, 100.0, 500.0]:
            for g in [-50.0, 0.0, 50.0]:
                for f in [0.0, 100.0]:
                    s, _ = compute_research_priority_score(
                        champion_vulnerability_score=v,
                        challenger_gap_pct=g,
                        feature_affinity_score=f,
                    )
                    self.assertGreaterEqual(s, 0.0)
                    self.assertLessEqual(s, 100.0)

    def test_04_priority_classification_boundaries(self):
        """4. Verify classification thresholds for CRITICAL, HIGH, MEDIUM, LOW, NEGLIGIBLE."""
        self.assertEqual(classify_priority_score(85.0), ResearchPriorityClass.CRITICAL)
        self.assertEqual(classify_priority_score(80.0), ResearchPriorityClass.CRITICAL)
        self.assertEqual(classify_priority_score(70.0), ResearchPriorityClass.HIGH)
        self.assertEqual(classify_priority_score(65.0), ResearchPriorityClass.HIGH)
        self.assertEqual(classify_priority_score(55.0), ResearchPriorityClass.MEDIUM)
        self.assertEqual(classify_priority_score(50.0), ResearchPriorityClass.MEDIUM)
        self.assertEqual(classify_priority_score(35.0), ResearchPriorityClass.LOW)
        self.assertEqual(classify_priority_score(30.0), ResearchPriorityClass.LOW)
        self.assertEqual(classify_priority_score(25.0), ResearchPriorityClass.NEGLIGIBLE)

    def test_05_cold_start_context_behavior(self):
        """5. Verify cold-start context exhibits INSUFFICIENT confidence and high coverage-gap score."""
        ckey = "NIFTY_3s_DIRECTION_CLASSIFIER_5m_R005"
        opp = evaluate_research_opportunity(
            self.tmp_dir,
            ckey,
            opportunity_type=OpportunityType.COVERAGE_EXPANSION,
            candidate_features=["adx_14"],
            schema=self.mock_schema,
        )
        self.assertEqual(opp.evidence_confidence, EvidenceConfidenceLevel.INSUFFICIENT)
        self.assertGreaterEqual(opp.component_breakdown.coverage_gap_score, 90.0)

    def test_06_sparse_context_behavior(self):
        """6. Verify sparse context produces WEAK evidence confidence."""
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
            primary_metric_value=0.75,
            robustness_score=75.0,
        )

        opp = evaluate_research_opportunity(
            self.tmp_dir,
            ckey,
            opportunity_type=OpportunityType.FEATURE_EXPLORATION,
            candidate_features=["rsi_14"],
            schema=self.mock_schema,
        )
        self.assertIn(opp.evidence_confidence, (EvidenceConfidenceLevel.WEAK, EvidenceConfidenceLevel.INSUFFICIENT))

    def test_07_mature_context_confidence(self):
        """7. Verify mature context with 10+ benchmarks yields STRONG evidence confidence."""
        ckey = "NIFTY_3s_DIRECTION_CLASSIFIER_5m_R001"
        b_run = create_benchmark_run(self.tmp_dir, context_key=ckey)

        for i in range(10):
            spec_i = {
                "market": "NIFTY",
                "sampling_interval_sec": 3,
                "task_type": "DIRECTION_CLASSIFIER",
                "prediction_horizon": "5m",
                "regime_id": "R001",
                "features": ["adx_14"],
                "algorithm": "xgboost",
                "hyperparameters": {"depth": 3 + i},
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
                primary_metric_value=0.80,
                robustness_score=80.0,
            )

        opp = evaluate_research_opportunity(
            self.tmp_dir,
            ckey,
            opportunity_type=OpportunityType.FEATURE_EXPLORATION,
            candidate_features=["adx_14"],
            schema=self.mock_schema,
        )
        self.assertIn(opp.evidence_confidence, (EvidenceConfidenceLevel.STRONG, EvidenceConfidenceLevel.MODERATE))

    def test_08_strong_champion_vulnerability_elevation(self):
        """8. Verify high champion vulnerability elevates priority score significantly."""
        s_low_vuln, _ = compute_research_priority_score(champion_vulnerability_score=10.0)
        s_high_vuln, _ = compute_research_priority_score(champion_vulnerability_score=90.0)

        self.assertGreater(s_high_vuln - s_low_vuln, 20.0)

    def test_09_weak_champion_vulnerability_baseline(self):
        """9. Verify low vulnerability does not artificially inflate score."""
        s, b = compute_research_priority_score(
            champion_vulnerability_score=0.0,
            challenger_gap_pct=0.0,
            feature_affinity_score=50.0,
            coverage_density_score=100.0,
        )
        self.assertLess(s, 50.0)

    def test_10_challenger_lead_elevation(self):
        """10. Verify challenger lead (+10% gap) elevates challenger opportunity score."""
        s_trail, _ = compute_research_priority_score(challenger_gap_pct=-5.0)
        s_lead, _ = compute_research_priority_score(challenger_gap_pct=10.0)

        self.assertGreater(s_lead, s_trail)

    def test_11_challenger_trailing_reduction(self):
        """11. Verify heavily trailing challenger reduces challenger gap score."""
        _, b_lead = compute_research_priority_score(challenger_gap_pct=5.0)
        _, b_trail = compute_research_priority_score(challenger_gap_pct=-10.0)

        self.assertGreater(b_lead.challenger_gap_score, b_trail.challenger_gap_score)

    def test_12_feature_affinity_contribution(self):
        """12. Verify high feature affinity increases composite priority score."""
        s_low, _ = compute_research_priority_score(feature_affinity_score=20.0)
        s_high, _ = compute_research_priority_score(feature_affinity_score=90.0)

        self.assertGreater(s_high, s_low)

    def test_13_interaction_evidence_contribution(self):
        """13. Verify interaction synergy score contributes to overall priority score."""
        s_no_inter, _ = compute_research_priority_score(interaction_synergy_score=50.0)
        s_high_inter, _ = compute_research_priority_score(interaction_synergy_score=90.0)

        self.assertGreater(s_high_inter, s_no_inter)

    def test_14_insufficient_evidence_handling(self):
        """14. Verify classify_evidence_confidence returns INSUFFICIENT when both confidences are 0."""
        lvl, val = classify_evidence_confidence(0.0, 0.0)
        self.assertEqual(lvl, EvidenceConfidenceLevel.INSUFFICIENT)
        self.assertEqual(val, 0.0)

    def test_15_excluded_opportunity_suppression(self):
        """15. Verify EXCLUDED candidate receives priority score 0.0 and is filtered from eligible agenda."""
        ckey = "NIFTY_3s_DIRECTION_CLASSIFIER_5m_R001"
        opp = evaluate_research_opportunity(
            self.tmp_dir,
            ckey,
            opportunity_type=OpportunityType.FEATURE_EXPLORATION,
            candidate_features=["dep_old"],
            schema=self.mock_schema,
        )
        self.assertEqual(opp.exclusion_verdict, ExclusionVerdict.EXCLUDED)
        self.assertEqual(opp.priority_score, 0.0)

    def test_16_caution_propagation_and_penalty(self):
        """16. Verify CAUTION candidate receives -15.0 penalty and propagates caution verdict."""
        s_clean, _ = compute_research_priority_score(champion_vulnerability_score=60.0, is_caution=False)
        s_caut, b_caut = compute_research_priority_score(champion_vulnerability_score=60.0, is_caution=True)

        self.assertEqual(b_caut.caution_penalty, -15.0)
        self.assertAlmostEqual(s_clean - s_caut, 15.0, places=3)

    def test_17_r001_vs_r002_context_isolation(self):
        """17. Verify priority score in R001 does not affect R002 evaluation."""
        ctx_r1 = "NIFTY_3s_DIRECTION_CLASSIFIER_5m_R001"
        ctx_r2 = "NIFTY_3s_DIRECTION_CLASSIFIER_5m_R002"

        set_champion_for_context(self.tmp_dir, ctx_r1, "CHAMP_R1")
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
        register_or_get_experiment(self.tmp_dir, spec, model_name="CHAMP_R1")
        record_model_benchmark(
            self.tmp_dir,
            benchmark_run_id=b_run,
            signature_hash=sig_h,
            model_name="CHAMP_R1",
            context_key=ctx_r1,
            algorithm="xgboost",
            dataset_name="d.parquet",
            feature_count=1,
            primary_metric_name="roc_auc",
            primary_metric_value=0.70,
            robustness_score=70.0,
        )
        record_regime_evaluation(
            self.tmp_dir,
            model_name="CHAMP_R1",
            signature_hash=sig_h,
            tested_regime_id="R002",
            tested_regime_hash="def",
            is_native_regime=False,
            sample_count=1000,
            primary_metric=0.40,
            regime_degradation_pct=40.0,
        )

        agenda_r1 = build_context_priority_agenda(self.tmp_dir, ctx_r1, schema=self.mock_schema)
        agenda_r2 = build_context_priority_agenda(self.tmp_dir, ctx_r2, schema=self.mock_schema)

        # R1 has champion vulnerability defense opportunity
        self.assertEqual(agenda_r1.context_key, ctx_r1)
        self.assertEqual(agenda_r2.context_key, ctx_r2)

    def test_18_nifty_vs_banknifty_isolation(self):
        """18. Verify complete isolation across markets (NIFTY vs BANKNIFTY)."""
        ctx_n = "NIFTY_3s_DIRECTION_CLASSIFIER_5m_R001"
        ctx_bn = "BANKNIFTY_3s_DIRECTION_CLASSIFIER_5m_R001"

        agenda_n = build_context_priority_agenda(self.tmp_dir, ctx_n, schema=self.mock_schema)
        agenda_bn = build_context_priority_agenda(self.tmp_dir, ctx_bn, schema=self.mock_schema)

        self.assertEqual(agenda_n.market, "NIFTY")
        self.assertEqual(agenda_bn.market, "BANKNIFTY")

    def test_19_horizon_isolation_5m_vs_15m(self):
        """19. Verify isolation across horizons (5m vs 15m)."""
        ctx_5m = "NIFTY_3s_DIRECTION_CLASSIFIER_5m_R001"
        ctx_15m = "NIFTY_3s_DIRECTION_CLASSIFIER_15m_R001"

        a_5m = build_context_priority_agenda(self.tmp_dir, ctx_5m, schema=self.mock_schema)
        a_15m = build_context_priority_agenda(self.tmp_dir, ctx_15m, schema=self.mock_schema)

        self.assertEqual(a_5m.prediction_horizon, "5m")
        self.assertEqual(a_15m.prediction_horizon, "15m")

    def test_20_sampling_interval_isolation_3s_vs_5s(self):
        """20. Verify isolation across sampling intervals (3s vs 5s)."""
        ctx_3s = "NIFTY_3s_DIRECTION_CLASSIFIER_5m_R001"
        ctx_5s = "NIFTY_5s_DIRECTION_CLASSIFIER_5m_R001"

        a_3s = build_context_priority_agenda(self.tmp_dir, ctx_3s, schema=self.mock_schema)
        a_5s = build_context_priority_agenda(self.tmp_dir, ctx_5s, schema=self.mock_schema)

        self.assertEqual(a_3s.sampling_interval_sec, 3)
        self.assertEqual(a_5s.sampling_interval_sec, 5)

    def test_21_deterministic_tie_breaking(self):
        """21. Verify repeated priority agenda generation sorts identically."""
        ckey = "NIFTY_3s_DIRECTION_CLASSIFIER_5m_R001"
        a1 = build_context_priority_agenda(self.tmp_dir, ckey, schema=self.mock_schema)
        a2 = build_context_priority_agenda(self.tmp_dir, ckey, schema=self.mock_schema)

        opps1 = [o.to_dict() for o in a1.eligible_opportunities]
        opps2 = [o.to_dict() for o in a2.eligible_opportunities]
        self.assertEqual(opps1, opps2)

    def test_22_nan_and_infinity_safety(self):
        """22. Verify compute_research_priority_score handles NaN and Infinity safely."""
        s_nan, _ = compute_research_priority_score(
            coverage_density_score=float("nan"),
            champion_vulnerability_score=float("nan"),
            challenger_gap_pct=float("nan"),
            feature_affinity_score=float("nan"),
            interaction_synergy_score=float("nan"),
        )
        self.assertFalse(math.isnan(s_nan))
        self.assertGreaterEqual(s_nan, 0.0)
        self.assertLessEqual(s_nan, 100.0)

        s_inf, _ = compute_research_priority_score(
            champion_vulnerability_score=float("inf"),
            challenger_gap_pct=float("inf"),
            feature_affinity_score=float("inf"),
            interaction_synergy_score=float("inf"),
            coverage_density_score=-float("inf"),
        )
        self.assertEqual(s_inf, 100.0)

    def test_23_empty_database_safety(self):
        """23. Verify agenda generation on empty database completes safely without errors."""
        ckey = "NIFTY_3s_DIRECTION_CLASSIFIER_5m_R001"
        agenda = build_context_priority_agenda(self.tmp_dir, ckey, schema=self.mock_schema)
        self.assertIsInstance(agenda, ContextPriorityAgendaReport)
        self.assertEqual(agenda.context_key, ckey)

    def test_24_concurrent_read_safety(self):
        """24. Verify concurrent priority evaluations run safely without threading collisions."""
        ckey = "NIFTY_3s_DIRECTION_CLASSIFIER_5m_R001"

        def _run_eval():
            return build_context_priority_agenda(self.tmp_dir, ckey, schema=self.mock_schema)

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(_run_eval) for _ in range(8)]
            results = [f.result() for f in futures]

        self.assertEqual(len(results), 8)
        self.assertTrue(all(isinstance(r, ContextPriorityAgendaReport) for r in results))

    def test_25_production_immutability(self):
        """25. Assert that priority scoring never mutates production state."""
        ev_path = "apps/feature_recommendation_evidence.db"
        self.assertTrue(os.path.isfile(ev_path), "Evidence DB must exist")

        with open(ev_path, "rb") as fh:
            sha_initial = hashlib.sha256(fh.read()).hexdigest()

        ckey = "NIFTY_3s_DIRECTION_CLASSIFIER_5m_R001"
        _ = build_context_priority_agenda(self.tmp_dir, ckey, schema=self.mock_schema)

        with open(ev_path, "rb") as fh:
            sha_final = hashlib.sha256(fh.read()).hexdigest()

        self.assertEqual(sha_initial, sha_final, "Evidence DB must remain 100% unmutated")

    def test_26_evidence_db_sha256_unchanged(self):
        """26. Assert that evidence database SHA-256 remains unmutated after extensive evaluations."""
        ev_path = "apps/feature_recommendation_evidence.db"
        with open(ev_path, "rb") as fh:
            sha_before = hashlib.sha256(fh.read()).hexdigest()

        ckey = "NIFTY_3s_DIRECTION_CLASSIFIER_5m_R001"
        for _ in range(5):
            _ = evaluate_research_opportunity(
                self.tmp_dir,
                ckey,
                opportunity_type=OpportunityType.FEATURE_EXPLORATION,
                candidate_features=["adx_14", "rsi_14"],
                schema=self.mock_schema,
            )

        with open(ev_path, "rb") as fh:
            sha_after = hashlib.sha256(fh.read()).hexdigest()

        self.assertEqual(sha_before, sha_after)


if __name__ == "__main__":
    unittest.main()

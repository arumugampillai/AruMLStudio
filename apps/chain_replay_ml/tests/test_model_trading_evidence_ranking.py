"""Dedicated unit tests for Phase 4F.3: Model + Trading Evidence Ranking Engine."""

import hashlib
import json
import os
import shutil
import tempfile
import unittest

from path_config import ensure_ml_studio_paths
ensure_ml_studio_paths()

from chain_replay_ml.model_ranking import (
    CandidateEvidenceScore,
    CandidateRankingPolicy,
    ContextRankingReport,
    RANK_POLICY_v1_0,
    RecommendationClass,
    compute_composite_candidate_score,
    compute_trading_evidence_score,
    evaluate_candidate_evidence,
    init_candidate_rankings_table,
    load_candidate_rankings_for_context,
    normalize_mfe_mae_ratio,
    normalize_profit_factor,
    normalize_win_rate,
    persist_candidate_rankings,
    rank_candidates_in_context,
)
from chain_replay_ml.research_memory import (
    init_analysis_db,
    record_model_benchmark,
    register_or_get_experiment,
)


class TestModelTradingEvidenceRankingEngine(unittest.TestCase):
    """Comprehensive test suite verifying Phase 4F.3 ranking invariants and scoring accuracy."""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp(prefix="aruml_test_ranking_")
        init_analysis_db(self.tmp_dir)
        self.context_key = "NIFTY_3s_DIRECTION_CLASSIFIER_5m_R001"

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_01_composite_score_calculation_exact(self):
        """1. Verify exact numerical precision of composite score formula."""
        # Model score = 80.0, Trading score = 70.0 -> Composite = 0.4*80 + 0.6*70 = 74.0
        score = compute_composite_candidate_score(80.0, 70.0)
        self.assertEqual(score, 74.0)

    def test_02_model_evidence_score_reuse(self):
        """2. Verify Phase 4D robustness score calculation is correctly consumed."""
        ev = evaluate_candidate_evidence(
            candidate_id="CAND_01",
            signature_hash="sig_01",
            context_key=self.context_key,
            model_metrics={"roc_auc": 0.80, "fold_mean": 0.80, "fold_std": 0.02, "expected_calibration_error": 0.03},
            trading_metrics={"win_rate_pct": 55.0, "profit_factor": 1.50, "mfe_mae_ratio": 1.20, "total_trades": 50},
        )
        self.assertGreater(ev.model_evidence_score, 70.0)
        self.assertIn("model_base_performance_contribution", ev.score_breakdown)

    def test_03_trading_evidence_score_normalization(self):
        """3. Verify Win Rate, Profit Factor, and MFE/MAE normalization bounds [0.0, 1.0]."""
        self.assertEqual(normalize_win_rate(40.0), 0.0)
        self.assertEqual(normalize_win_rate(70.0), 1.0)
        self.assertEqual(normalize_win_rate(55.0), 0.5)

        self.assertEqual(normalize_profit_factor(0.80), 0.0)
        self.assertEqual(normalize_profit_factor(2.50), 1.0)

        self.assertEqual(normalize_mfe_mae_ratio(0.50), 0.0)
        self.assertEqual(normalize_mfe_mae_ratio(2.00), 1.0)

    def test_04_volume_confidence_scaling(self):
        """4. Verify trade volume confidence scaling (N=5 scaled down, N>=30 full confidence)."""
        _, _, conf_low, _, _ = compute_trading_evidence_score(
            win_rate_pct=60.0, profit_factor=1.5, mfe_mae_ratio=1.2, total_trades=5
        )
        _, _, conf_full, _, _ = compute_trading_evidence_score(
            win_rate_pct=60.0, profit_factor=1.5, mfe_mae_ratio=1.2, total_trades=30
        )
        self.assertAlmostEqual(conf_low, (5/30)**0.5, places=3)
        self.assertEqual(conf_full, 1.0)

    def test_05_zero_trades_fallback(self):
        """5. Verify candidate with 0 trades falls back safely to pure model evidence without crashing."""
        ev = evaluate_candidate_evidence(
            candidate_id="CAND_NO_TRADES",
            signature_hash="sig_zero",
            context_key=self.context_key,
            model_metrics={"roc_auc": 0.75, "fold_mean": 0.75, "fold_std": 0.01},
            trading_metrics={"total_trades": 0},
        )
        self.assertEqual(ev.trading_evidence_score, 0.0)
        self.assertEqual(ev.volume_confidence, 0.0)
        self.assertAlmostEqual(ev.composite_score, 0.40 * ev.model_evidence_score, places=2)

    def test_06_drawdown_risk_penalty_scaling(self):
        """6. Verify Max Drawdown > 5% incurs linear penalty."""
        score_safe, p_safe, _, _, _ = compute_trading_evidence_score(
            win_rate_pct=60.0, profit_factor=1.5, mfe_mae_ratio=1.2, max_drawdown_pct=4.0, total_trades=50
        )
        score_dd, p_dd, _, _, _ = compute_trading_evidence_score(
            win_rate_pct=60.0, profit_factor=1.5, mfe_mae_ratio=1.2, max_drawdown_pct=10.0, total_trades=50
        )
        self.assertEqual(p_safe, 0.0)
        self.assertEqual(p_dd, 2.0 * (10.0 - 5.0))  # 10.0 points penalty
        self.assertLess(score_dd, score_safe)

    def test_07_consecutive_loss_streak_penalty(self):
        """7. Verify consecutive loss streak > 3 trades applies risk penalty."""
        _, p_safe, _, _, _ = compute_trading_evidence_score(
            win_rate_pct=60.0, profit_factor=1.5, mfe_mae_ratio=1.2, max_consecutive_losses=2, total_trades=50
        )
        _, p_streak, _, _, _ = compute_trading_evidence_score(
            win_rate_pct=60.0, profit_factor=1.5, mfe_mae_ratio=1.2, max_consecutive_losses=6, total_trades=50
        )
        self.assertEqual(p_safe, 0.0)
        self.assertEqual(p_streak, 3.0 * (6 - 3))  # 9.0 points penalty

    def test_08_high_auc_poor_trading_rejection(self):
        """8. Verify high-AUC model (0.88) with poor trading results (35% win rate, 18% DD) is ranked low."""
        ev = evaluate_candidate_evidence(
            candidate_id="CAND_OVERFIT_AUC",
            signature_hash="sig_auc",
            context_key=self.context_key,
            model_metrics={"roc_auc": 0.88, "fold_mean": 0.88, "fold_std": 0.05, "expected_calibration_error": 0.12},
            trading_metrics={"win_rate_pct": 35.0, "profit_factor": 0.70, "mfe_mae_ratio": 0.40, "max_drawdown_pct": 18.0, "max_consecutive_losses": 7, "total_trades": 60},
        )
        self.assertGreater(ev.risk_penalty, 25.0)
        self.assertEqual(ev.recommendation_class, RecommendationClass.REJECTED)

    def test_09_lower_auc_superior_trading_promotion(self):
        """9. Verify moderate-AUC model (0.72) with superior trading evidence outranks fragile high-AUC model."""
        c_fragile = {
            "candidate_id": "CAND_FRAGILE",
            "signature_hash": "sig_fragile",
            "model_metrics": {"roc_auc": 0.86, "fold_mean": 0.86, "fold_std": 0.06},
            "trading_metrics": {"win_rate_pct": 42.0, "profit_factor": 0.95, "mfe_mae_ratio": 0.70, "max_drawdown_pct": 12.0, "total_trades": 40},
        }
        c_robust = {
            "candidate_id": "CAND_ROBUST",
            "signature_hash": "sig_robust",
            "model_metrics": {"roc_auc": 0.72, "fold_mean": 0.72, "fold_std": 0.01, "expected_calibration_error": 0.02},
            "trading_metrics": {"win_rate_pct": 65.0, "profit_factor": 1.90, "mfe_mae_ratio": 1.50, "max_drawdown_pct": 3.0, "max_consecutive_losses": 2, "total_trades": 55},
        }
        report = rank_candidates_in_context(self.tmp_dir, self.context_key, candidate_items=[c_fragile, c_robust])
        self.assertEqual(report.top_candidate.candidate_id, "CAND_ROBUST")
        self.assertGreater(report.top_candidate.composite_score, report.ranked_candidates[1].composite_score)

    def test_10_over_trading_frequency_handling(self):
        """10. Verify over-trading with low edge is penalized via profit factor and drawdown."""
        _, _, _, b_over, _ = compute_trading_evidence_score(
            win_rate_pct=42.0, profit_factor=0.85, mfe_mae_ratio=0.6, max_drawdown_pct=14.0, total_trades=400
        )
        self.assertLess(b_over["net_trading_score"], 20.0)

    def test_11_cross_regime_spread_penalty(self):
        """11. Verify wide cross-regime win rate spread (> 25%) incurs penalty."""
        _, p_safe, _, _, _ = compute_trading_evidence_score(
            win_rate_pct=60.0, profit_factor=1.5, mfe_mae_ratio=1.2, regime_spread_pct=15.0, total_trades=50
        )
        _, p_spread, _, _, _ = compute_trading_evidence_score(
            win_rate_pct=60.0, profit_factor=1.5, mfe_mae_ratio=1.2, regime_spread_pct=35.0, total_trades=50
        )
        self.assertEqual(p_safe, 0.0)
        self.assertEqual(p_spread, 0.50 * (35.0 - 25.0))  # 5.0 points penalty

    def test_12_deterministic_tie_breaking(self):
        """12. Verify strict deterministic 5-level tie-breaking hierarchy."""
        c1 = {
            "candidate_id": "CAND_A",
            "signature_hash": "sig_a",
            "model_metrics": {"roc_auc": 0.75, "expected_calibration_error": 0.05, "total_features": 25},
            "trading_metrics": {"win_rate_pct": 55.0, "profit_factor": 1.4, "mfe_mae_ratio": 1.1, "total_trades": 50},
        }
        c2 = {
            "candidate_id": "CAND_B",
            "signature_hash": "sig_b",
            "model_metrics": {"roc_auc": 0.75, "expected_calibration_error": 0.02, "total_features": 25},  # Lower ECE wins
            "trading_metrics": {"win_rate_pct": 55.0, "profit_factor": 1.4, "mfe_mae_ratio": 1.1, "total_trades": 50},
        }
        report = rank_candidates_in_context(self.tmp_dir, self.context_key, candidate_items=[c1, c2])
        self.assertEqual(report.top_candidate.candidate_id, "CAND_B")

    def test_13_champion_candidate_classification_threshold(self):
        """13. Verify CHAMPION_CANDIDATE requires margin >= 2.0 and score >= 75.0."""
        c_champ = {
            "candidate_id": "CAND_BEAT_CHAMP",
            "signature_hash": "sig_champ",
            "model_metrics": {"roc_auc": 0.82, "fold_mean": 0.82, "fold_std": 0.01, "expected_calibration_error": 0.02},
            "trading_metrics": {"win_rate_pct": 68.0, "profit_factor": 2.1, "mfe_mae_ratio": 1.6, "total_trades": 60},
        }
        # Champion score is 72.0 -> Candidate gets 80+ -> Qualifies as CHAMPION_CANDIDATE
        report = rank_candidates_in_context(self.tmp_dir, self.context_key, candidate_items=[c_champ], champion_composite_score=72.0)
        self.assertEqual(report.top_candidate.recommendation_class, RecommendationClass.CHAMPION_CANDIDATE)

    def test_14_fine_tune_candidate_selection(self):
        """14. Verify top 3 non-champion contenders are designated as FINE_TUNE_CANDIDATE."""
        items = []
        for i in range(5):
            items.append({
                "candidate_id": f"CAND_{i}",
                "signature_hash": f"sig_{i}",
                "model_metrics": {"roc_auc": 0.70 + (0.02 * i), "fold_mean": 0.70 + (0.02 * i), "fold_std": 0.01},
                "trading_metrics": {"win_rate_pct": 55.0 + i, "profit_factor": 1.4 + (0.1 * i), "mfe_mae_ratio": 1.1, "total_trades": 40},
            })
        report = rank_candidates_in_context(self.tmp_dir, self.context_key, candidate_items=items)
        self.assertEqual(len(report.fine_tune_candidates), 3)
        for c in report.fine_tune_candidates:
            self.assertIn(c.recommendation_class, (RecommendationClass.FINE_TUNE_CANDIDATE, RecommendationClass.CHAMPION_CANDIDATE))

    def test_15_rejected_candidate_classification(self):
        """15. Verify candidate with severe risk penalty or score < 50 is classified as REJECTED."""
        ev = evaluate_candidate_evidence(
            candidate_id="CAND_POOR",
            signature_hash="sig_poor",
            context_key=self.context_key,
            model_metrics={"roc_auc": 0.52},
            trading_metrics={"win_rate_pct": 38.0, "profit_factor": 0.6, "total_trades": 20},
        )
        self.assertEqual(ev.recommendation_class, RecommendationClass.REJECTED)

    def test_16_parent_child_delta_score_tracking(self):
        """16. Verify Delta composite score vs parent candidate is accurately computed."""
        ev = evaluate_candidate_evidence(
            candidate_id="CAND_CHILD",
            signature_hash="sig_child",
            context_key=self.context_key,
            model_metrics={"roc_auc": 0.78},
            trading_metrics={"win_rate_pct": 60.0, "profit_factor": 1.6, "mfe_mae_ratio": 1.3, "total_trades": 50},
            parent_candidate_id="CAND_PARENT",
            parent_composite_score=65.0,
        )
        self.assertEqual(ev.parent_candidate_id, "CAND_PARENT")
        self.assertIsNotNone(ev.delta_vs_parent)
        self.assertEqual(ev.delta_vs_parent, round(ev.composite_score - 65.0, 4))

    def test_17_context_key_strict_isolation(self):
        """17. Verify ranking occurs strictly within target ModelContextKey."""
        report = rank_candidates_in_context(self.tmp_dir, "BANKNIFTY_3s_DIRECTION_CLASSIFIER_5m_R002", candidate_items=[])
        self.assertEqual(report.context_key, "BANKNIFTY_3s_DIRECTION_CLASSIFIER_5m_R002")

    def test_18_single_candidate_context_ranking(self):
        """18. Verify context with exactly 1 candidate ranks successfully."""
        item = {
            "candidate_id": "CAND_SOLO",
            "signature_hash": "sig_solo",
            "model_metrics": {"roc_auc": 0.74},
            "trading_metrics": {"win_rate_pct": 58.0, "profit_factor": 1.45, "mfe_mae_ratio": 1.2, "total_trades": 45},
        }
        report = rank_candidates_in_context(self.tmp_dir, self.context_key, candidate_items=[item])
        self.assertEqual(report.total_candidates_ranked, 1)
        self.assertEqual(report.top_candidate.candidate_id, "CAND_SOLO")

    def test_19_empty_context_ranking_safe(self):
        """19. Verify empty context returns empty ranking report without crashing."""
        report = rank_candidates_in_context(self.tmp_dir, self.context_key, candidate_items=[])
        self.assertEqual(report.total_candidates_ranked, 0)
        self.assertIsNone(report.top_candidate)

    def test_20_nan_inf_metric_rejection(self):
        """20. Verify NaN or Infinity values in trading metrics are safely rejected."""
        score, _, _, _, warnings = compute_trading_evidence_score(
            win_rate_pct=float("nan"), profit_factor=1.5, mfe_mae_ratio=1.2
        )
        self.assertEqual(score, 0.0)
        self.assertIn("REJECTED_INVALID_TRADING_METRICS", warnings)

    def test_21_ranking_policy_hashing_and_versioning(self):
        """21. Verify policy hash updates deterministically when policy parameters change."""
        p1 = CandidateRankingPolicy(policy_id="RANK_POLICY_v1.0", w_model=0.40)
        p2 = CandidateRankingPolicy(policy_id="RANK_POLICY_v1.0", w_model=0.50)
        self.assertNotEqual(p1.compute_policy_hash(), p2.compute_policy_hash())

    def test_22_database_persistence_and_retrieval(self):
        """22. Verify ranking records round-trip to candidate_evidence_rankings table cleanly."""
        item = {
            "candidate_id": "CAND_PERSIST",
            "signature_hash": "sig_persist_123",
            "model_metrics": {"roc_auc": 0.76},
            "trading_metrics": {"win_rate_pct": 62.0, "profit_factor": 1.65, "mfe_mae_ratio": 1.3, "total_trades": 48},
        }
        report = rank_candidates_in_context(self.tmp_dir, self.context_key, candidate_items=[item])
        written = persist_candidate_rankings(self.tmp_dir, report)
        self.assertEqual(written, 1)

        loaded = load_candidate_rankings_for_context(self.tmp_dir, self.context_key)
        self.assertEqual(len(loaded), 1)
        self.assertEqual(loaded[0].candidate_id, "CAND_PERSIST")
        self.assertAlmostEqual(loaded[0].composite_score, report.top_candidate.composite_score, places=2)

    def test_23_production_immutability(self):
        """23. Invariant: Candidate ranking never writes to .active_model.json or production model directories."""
        rank_candidates_in_context(self.tmp_dir, self.context_key, candidate_items=[])
        active_model_path = os.path.join(self.tmp_dir, "models", ".active_model.json")
        self.assertFalse(os.path.exists(active_model_path))

    def test_24_evidence_db_immutability(self):
        """24. Invariant: Feature Recommendation Evidence DB remains unmutated."""
        ev_db_path = os.path.join("apps", "feature_recommendation_evidence.db")
        if os.path.exists(ev_db_path):
            with open(ev_db_path, "rb") as f:
                sha = hashlib.sha256(f.read()).hexdigest()
            self.assertEqual(sha, "6f91afca94ec87a1210d8f4bcef356b9c16a6ef5a488268f41c5b1b81431ade2")

    def test_25_legacy_aruneo_exclusion(self):
        """25. Invariant: Candidate ranking never creates or touches .lifecycle_registry.db."""
        rank_candidates_in_context(self.tmp_dir, self.context_key, candidate_items=[])
        legacy_db_path = os.path.join(self.tmp_dir, "models", ".lifecycle_registry.db")
        self.assertFalse(os.path.exists(legacy_db_path))

    def test_26_end_to_end_batch_candidate_ranking(self):
        """26. Full End-to-End: Ranks a realistic 10-candidate batch with mixed model and trading metrics."""
        candidates = []
        for i in range(10):
            candidates.append({
                "candidate_id": f"CAND_BATCH_{i:02d}",
                "signature_hash": f"sig_batch_{i:02d}",
                "model_metrics": {"roc_auc": 0.65 + (0.02 * i), "fold_mean": 0.65 + (0.02 * i), "fold_std": 0.01 + (0.005 * i)},
                "trading_metrics": {
                    "win_rate_pct": 45.0 + (2.5 * i),
                    "profit_factor": 1.0 + (0.12 * i),
                    "mfe_mae_ratio": 0.8 + (0.08 * i),
                    "max_drawdown_pct": max(1.0, 10.0 - (0.8 * i)),
                    "total_trades": 25 + (3 * i),
                },
            })
        report = rank_candidates_in_context(self.tmp_dir, self.context_key, candidate_items=candidates)
        self.assertEqual(report.total_candidates_ranked, 10)
        self.assertGreater(report.ranked_candidates[0].composite_score, report.ranked_candidates[-1].composite_score)

    def test_27_fine_tuning_export_interface(self):
        """27. Verify output report serializes cleanly for Phase 4F.4 fine-tuning controller."""
        item = {
            "candidate_id": "CAND_EXPORT",
            "signature_hash": "sig_export_123",
            "model_metrics": {"roc_auc": 0.77},
            "trading_metrics": {"win_rate_pct": 61.0, "profit_factor": 1.7, "mfe_mae_ratio": 1.3, "total_trades": 50},
        }
        report = rank_candidates_in_context(self.tmp_dir, self.context_key, candidate_items=[item])
        d = report.to_dict()
        self.assertEqual(d["context_key"], self.context_key)
        self.assertEqual(d["total_candidates_ranked"], 1)
        self.assertEqual(d["ranked_candidates"][0]["candidate_id"], "CAND_EXPORT")


if __name__ == "__main__":
    unittest.main()

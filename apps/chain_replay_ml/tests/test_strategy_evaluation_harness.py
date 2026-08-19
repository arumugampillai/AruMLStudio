"""Unit Tests for Phase 4F.1: Deterministic Strategy Evaluation Harness & Telemetry Engine."""

import os
import shutil
import tempfile
import unittest
import numpy as np
import pandas as pd

from path_config import ensure_ml_studio_paths
ensure_ml_studio_paths()

from chain_replay_ml.research_memory import (
    create_benchmark_run,
    init_analysis_db,
    record_model_benchmark,
    register_or_get_experiment,
)
from chain_replay_ml.strategy_evaluation import (
    EvaluationTrade,
    ExitReason,
    StrategyEvaluationPolicy,
    TradeDirection,
    TradingEvidenceDossier,
    compute_trading_evidence,
    evaluate_model_predictions,
    get_trading_evidence_for_benchmark,
    persist_trading_evidence,
    run_deterministic_replay,
)


class TestStrategyEvaluationHarness(unittest.TestCase):
    """Comprehensive test suite for Phase 4F.1 Strategy Evaluation Harness."""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp(prefix="aruml_test_strategy_eval_")
        init_analysis_db(self.tmp_dir)

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def _sample_synthetic_predictions(self) -> pd.DataFrame:
        """Create synthetic out-of-fold predictions with varying price movements."""
        n = 100
        # Price path with distinct moves: Up (+3%), Down (-3%), Flat
        prices = [100.0]
        for i in range(1, n):
            if i < 20:
                prices.append(prices[-1] * 1.002)  # Steady rise (+4%)
            elif i < 40:
                prices.append(prices[-1] * 0.998)  # Steady drop (-4%)
            elif i < 60:
                prices.append(prices[-1] * 1.001)  # Moderate rise
            else:
                prices.append(prices[-1] * 0.9995) # Slight chop

        probs = [0.40] * n
        preds = ["FLAT"] * n
        # Inject high-confidence long signals
        probs[2] = 0.85
        preds[2] = "UP"

        probs[25] = 0.90
        preds[25] = "DOWN"

        probs[45] = 0.80
        preds[45] = "UP"

        regimes = ["R001"] * 50 + ["R002"] * 50

        return pd.DataFrame({
            "ts": np.arange(1000, 1000 + n * 5, 5),
            "ltp": prices,
            "predicted_prob": probs,
            "predicted_class": preds,
            "regime_id": regimes,
            "fold_index": [1] * n,
        })

    def test_policy_defaults_and_hypotheses(self):
        """1. Verify policy defaults: +2%/-2%/55%/60 bars as research-configurable candidate hypotheses."""
        pol = StrategyEvaluationPolicy()
        self.assertEqual(pol.policy_id, "EVAL_POLICY_BASELINE_v1.0")
        self.assertEqual(pol.min_confidence_threshold, 0.55)
        self.assertEqual(pol.target_return_pct, 2.0)
        self.assertEqual(pol.stop_loss_pct, 2.0)
        self.assertEqual(pol.max_holding_bars, 60)
        self.assertEqual(pol.cooldown_bars, 5)
        self.assertFalse(pol.allow_multiple_open)

    def test_deterministic_replay_target_hit(self):
        """2. Verify deterministic target hit (+2.0%) execution."""
        # Simple dataframe: starts at 100.0, signal at idx 0, price rises to 102.5 (+2.5%) at idx 5
        prices = [100.0, 100.5, 101.0, 101.5, 102.0, 102.5, 103.0]
        df = pd.DataFrame({
            "ts": np.arange(len(prices)),
            "ltp": prices,
            "predicted_prob": [0.80] + [0.30] * (len(prices) - 1),
            "predicted_class": ["UP"] + ["FLAT"] * (len(prices) - 1),
            "regime_id": ["R001"] * len(prices),
        })
        policy = StrategyEvaluationPolicy(target_return_pct=2.0, stop_loss_pct=2.0)
        trades = run_deterministic_replay(df, policy)

        self.assertEqual(len(trades), 1)
        t = trades[0]
        self.assertEqual(t.direction, TradeDirection.LONG)
        self.assertEqual(t.exit_reason, ExitReason.TARGET_HIT)
        self.assertTrue(t.is_win)
        self.assertGreaterEqual(t.realized_return_pct, 2.0)
        self.assertGreater(t.mfe_pct, 0.0)

    def test_deterministic_replay_stop_hit(self):
        """3. Verify deterministic stop loss (-2.0%) execution."""
        # Starts at 100.0, signal at idx 0, price drops to 97.5 (-2.5%) at idx 5
        prices = [100.0, 99.5, 99.0, 98.5, 98.0, 97.5, 97.0]
        df = pd.DataFrame({
            "ts": np.arange(len(prices)),
            "ltp": prices,
            "predicted_prob": [0.80] + [0.30] * (len(prices) - 1),
            "predicted_class": ["UP"] + ["FLAT"] * (len(prices) - 1),
            "regime_id": ["R001"] * len(prices),
        })
        policy = StrategyEvaluationPolicy(target_return_pct=2.0, stop_loss_pct=2.0)
        trades = run_deterministic_replay(df, policy)

        self.assertEqual(len(trades), 1)
        t = trades[0]
        self.assertEqual(t.direction, TradeDirection.LONG)
        self.assertEqual(t.exit_reason, ExitReason.STOP_HIT)
        self.assertTrue(t.is_loss)
        self.assertLessEqual(t.realized_return_pct, -2.0)
        self.assertLess(t.mae_pct, 0.0)

    def test_deterministic_replay_time_expired(self):
        """4. Verify max holding bars expiration exit."""
        # Flat price 100.0 across 20 bars, max_holding_bars=10
        prices = [100.0] * 20
        df = pd.DataFrame({
            "ts": np.arange(len(prices)),
            "ltp": prices,
            "predicted_prob": [0.80] + [0.30] * 19,
            "predicted_class": ["UP"] + ["FLAT"] * 19,
            "regime_id": ["R002"] * len(prices),
        })
        policy = StrategyEvaluationPolicy(target_return_pct=5.0, stop_loss_pct=5.0, max_holding_bars=10)
        trades = run_deterministic_replay(df, policy)

        self.assertEqual(len(trades), 1)
        t = trades[0]
        self.assertEqual(t.exit_reason, ExitReason.TIME_EXPIRED)
        self.assertEqual(t.holding_bars, 10)

    def test_single_position_and_cooldown_invariants(self):
        """5. Verify single-position constraint: no compounding or multiple open trades."""
        # 10 consecutive high-confidence signals
        prices = [100.0 + i * 0.1 for i in range(30)]
        df = pd.DataFrame({
            "ts": np.arange(len(prices)),
            "ltp": prices,
            "predicted_prob": [0.90] * len(prices),
            "predicted_class": ["UP"] * len(prices),
            "regime_id": ["R001"] * len(prices),
        })
        policy = StrategyEvaluationPolicy(target_return_pct=1.0, stop_loss_pct=1.0, cooldown_bars=5)
        trades = run_deterministic_replay(df, policy)

        # Should execute sequential trades with cooldown gaps, NOT 30 simultaneous trades
        self.assertGreater(len(trades), 0)
        self.assertLess(len(trades), 10)
        for i in range(len(trades) - 1):
            self.assertGreaterEqual(trades[i+1].entry_index, trades[i].exit_index + policy.cooldown_bars)

    def test_metrics_dossier_aggregation(self):
        """6. Verify comprehensive trading telemetry dossier calculation."""
        df = self._sample_synthetic_predictions()
        dossier = evaluate_model_predictions(df, context_key="NIFTY_3s_DIRECTION_CLASSIFIER_5m_R001", model_name="CAT_TEST_v1")

        self.assertIsInstance(dossier, TradingEvidenceDossier)
        self.assertEqual(dossier.context_key, "NIFTY_3s_DIRECTION_CLASSIFIER_5m_R001")
        self.assertEqual(dossier.model_name, "CAT_TEST_v1")
        self.assertGreater(dossier.total_trades_executed, 0)
        self.assertGreaterEqual(dossier.win_rate_pct, 0.0)
        self.assertLessEqual(dossier.win_rate_pct, 100.0)
        self.assertIn("R001", dossier.regime_breakdown)
        self.assertGreaterEqual(dossier.max_drawdown_pct, 0.0)
        self.assertGreaterEqual(dossier.max_consecutive_losses, 0)

    def test_persistence_to_analysis_db(self):
        """7. Verify persistence of TRADING_EVALUATION metrics into analysis.db benchmark_metrics."""
        spec = {
            "market": "NIFTY",
            "sampling_interval_sec": 3,
            "task_type": "DIRECTION_CLASSIFIER",
            "prediction_horizon": "5m",
            "regime_id": "R001",
            "regime_definition_hash": "def_hash_1",
            "dataset_snapshot_hash": "ds_hash_1",
            "features": ["adx_14", "rsi_14"],
            "algorithm": "xgboost",
            "hyperparameters": {"max_depth": 4},
            "walk_forward_config": {"folds": 5, "window_mode": "expanding"},
            "random_seed": 42,
        }
        _, rec = register_or_get_experiment(self.tmp_dir, spec, model_name="XGB_TEST_v1")
        run_id = create_benchmark_run(self.tmp_dir, context_key="NIFTY_3s_DIRECTION_CLASSIFIER_5m_R001")
        bench_id = record_model_benchmark(
            self.tmp_dir,
            benchmark_run_id=run_id,
            signature_hash=rec["signature_hash"],
            model_name="XGB_TEST_v1",
            context_key="NIFTY_3s_DIRECTION_CLASSIFIER_5m_R001",
            algorithm="xgboost",
            dataset_name="ds.parquet",
            feature_count=2,
            primary_metric_name="roc_auc",
            primary_metric_value=0.78,
            fold_metric_mean=0.78,
            fold_metric_std=0.01,
        )

        df = self._sample_synthetic_predictions()
        dossier = evaluate_model_predictions(
            df,
            context_key="NIFTY_3s_DIRECTION_CLASSIFIER_5m_R001",
            model_name="XGB_TEST_v1",
            data_dir=self.tmp_dir,
            benchmark_id=bench_id,
        )

        # Query persisted metrics
        persisted = get_trading_evidence_for_benchmark(self.tmp_dir, bench_id)
        self.assertIn("WIN_RATE_PCT", persisted)
        self.assertIn("PROFIT_FACTOR", persisted)
        self.assertIn("MAX_DRAWDOWN_PCT", persisted)
        self.assertIn("MEAN_MFE_PCT", persisted)
        self.assertIn("MEAN_MAE_PCT", persisted)
        self.assertIn("MAX_CONSECUTIVE_LOSSES", persisted)
        self.assertEqual(persisted["WIN_RATE_PCT"], dossier.win_rate_pct)
        self.assertEqual(persisted["MAX_DRAWDOWN_PCT"], dossier.max_drawdown_pct)


if __name__ == "__main__":
    unittest.main()

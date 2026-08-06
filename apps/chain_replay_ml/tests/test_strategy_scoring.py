"""Unit tests for Strategy Simulator quality / evidence scoring (freeze v1)."""

from __future__ import annotations

import unittest

from chain_replay_ml.strategy_simulator.scoring import (
    SCORING_VERSION,
    attach_strategy_score,
    count_active_trading_days,
    evaluate_strategy,
    evaluate_strategy_from_run,
    get_strategy_grade,
    sample_reliability_label,
)


class StrategyGradeTests(unittest.TestCase):
    def test_grade_boundaries(self) -> None:
        self.assertEqual(get_strategy_grade(90.0), "A+")
        self.assertEqual(get_strategy_grade(89.9), "A")
        self.assertEqual(get_strategy_grade(80.0), "A")
        self.assertEqual(get_strategy_grade(79.9), "B")
        self.assertEqual(get_strategy_grade(70.0), "B")
        self.assertEqual(get_strategy_grade(69.9), "C")
        self.assertEqual(get_strategy_grade(60.0), "C")
        self.assertEqual(get_strategy_grade(59.9), "F")
        self.assertEqual(get_strategy_grade(0.0), "F")


class SampleReliabilityTests(unittest.TestCase):
    def test_thresholds(self) -> None:
        self.assertEqual(sample_reliability_label(92.0), "High")
        self.assertEqual(sample_reliability_label(70.0), "High")
        self.assertEqual(sample_reliability_label(69.9), "Medium")
        self.assertEqual(sample_reliability_label(40.0), "Medium")
        self.assertEqual(sample_reliability_label(39.9), "Low")
        self.assertEqual(sample_reliability_label(0.0), "Low")


class EvaluateStrategyTests(unittest.TestCase):
    def test_mock_like_scores(self) -> None:
        """Rough check against freeze mock layout numbers (targets = defaults)."""
        out = evaluate_strategy(
            executed_trades=500,
            active_trading_days=50,  # 50/60 → day conf 0.833 → evidence ~91.7
            net_profit=45200.0,
            net_pf=1.455,
            max_dd=45200.0 / 1.660,  # → romad ≈ 1.660
            expectancy=0.079 * 1000.0,  # stop=1000 → exp/risk 0.079
            stop_loss=1000.0,
            win_rate=50.0,  # percent form; sanitizer → 0.50
        )
        self.assertEqual(out["scoring_version"], SCORING_VERSION)
        self.assertAlmostEqual(out["raw_metrics"]["romad"], 1.66, places=2)
        self.assertAlmostEqual(out["raw_metrics"]["expectancy_ratio"], 0.079, places=3)
        self.assertAlmostEqual(out["raw_metrics"]["win_rate"], 50.0, places=1)
        self.assertAlmostEqual(out["component_scores"]["profit_factor"], 91.0, places=0)
        self.assertAlmostEqual(out["component_scores"]["romad"], 83.0, places=0)
        self.assertAlmostEqual(out["component_scores"]["expectancy"], 79.0, places=0)
        # win: (0.50 - 0.30) / 0.30 * 100 ≈ 66.7
        self.assertAlmostEqual(out["component_scores"]["win_rate"], 66.7, places=0)
        self.assertGreaterEqual(out["strategy_score"], 80.0)
        self.assertEqual(out["grade"], "A")
        self.assertEqual(out["sample_reliability"], "High")

    def test_weights_at_targets(self) -> None:
        out = evaluate_strategy(
            executed_trades=500,
            active_trading_days=60,
            net_profit=10000.0,
            net_pf=1.50,
            max_dd=5000.0,  # romad = 2.0 = target
            expectancy=100.0,
            stop_loss=1000.0,  # exp_ratio = 0.10 = target
            win_rate=0.60,
        )
        self.assertEqual(out["component_scores"]["profit_factor"], 100.0)
        self.assertEqual(out["component_scores"]["romad"], 100.0)
        self.assertEqual(out["component_scores"]["expectancy"], 100.0)
        self.assertEqual(out["component_scores"]["win_rate"], 100.0)
        self.assertEqual(out["strategy_score"], 100.0)
        self.assertEqual(out["grade"], "A+")
        self.assertEqual(out["evidence_score"], 100.0)

    def test_net_pf_none_with_profit(self) -> None:
        """No losers + profit → PF component 100 (freeze §13 Q3)."""
        out = evaluate_strategy(
            executed_trades=10,
            active_trading_days=5,
            net_profit=5000.0,
            net_pf=None,
            max_dd=1000.0,
            expectancy=500.0,
            stop_loss=1000.0,
            win_rate=100.0,
        )
        self.assertEqual(out["component_scores"]["profit_factor"], 100.0)
        self.assertIsNone(out["raw_metrics"]["net_profit_factor"])

    def test_net_pf_none_without_profit(self) -> None:
        out = evaluate_strategy(
            executed_trades=10,
            active_trading_days=5,
            net_profit=0.0,
            net_pf=None,
            max_dd=0.0,
            expectancy=0.0,
            stop_loss=1000.0,
            win_rate=0.0,
        )
        self.assertEqual(out["component_scores"]["profit_factor"], 0.0)

    def test_max_dd_zero_romad_zero(self) -> None:
        """Max DD = 0 → RoMaD component 0 even with positive profit (freeze §13 Q5)."""
        out = evaluate_strategy(
            executed_trades=20,
            active_trading_days=10,
            net_profit=5000.0,
            net_pf=1.5,
            max_dd=0.0,
            expectancy=250.0,
            stop_loss=1000.0,
            win_rate=0.55,
        )
        self.assertEqual(out["raw_metrics"]["romad"], 0.0)
        self.assertEqual(out["component_scores"]["romad"], 0.0)

    def test_win_rate_fraction_vs_percent(self) -> None:
        a = evaluate_strategy(
            executed_trades=1,
            active_trading_days=1,
            net_profit=0.0,
            net_pf=1.0,
            max_dd=1.0,
            expectancy=0.0,
            stop_loss=1.0,
            win_rate=50.0,
        )
        b = evaluate_strategy(
            executed_trades=1,
            active_trading_days=1,
            net_profit=0.0,
            net_pf=1.0,
            max_dd=1.0,
            expectancy=0.0,
            stop_loss=1.0,
            win_rate=0.50,
        )
        self.assertEqual(a["component_scores"]["win_rate"], b["component_scores"]["win_rate"])


class MapperTests(unittest.TestCase):
    def test_count_active_trading_days(self) -> None:
        trades = [
            {"trading_day": "2026-07-01", "net_pnl": 1},
            {"trading_day": "2026-07-01", "net_pnl": -1},
            {"trading_day": "2026-07-02", "net_pnl": 2},
            {"trading_day": None, "net_pnl": 0},
            {"trading_day": "—", "net_pnl": 0},
        ]
        self.assertEqual(count_active_trading_days(trades), 2)
        self.assertEqual(count_active_trading_days([]), 0)

    def test_from_run_uses_net_pf_and_stop_rupees(self) -> None:
        metrics = {
            "trade_count": 4,
            "net_profit": 1000.0,
            "profit_factor": 1.5,
            "account_equity_max_drawdown": 500.0,
            "expectancy": 250.0,
            "stop_loss_per_trade_rupees": 1000.0,
            "win_rate_pct": 50.0,
        }
        trades = [
            {"trading_day": "2026-07-01"},
            {"trading_day": "2026-07-02"},
        ]
        out = evaluate_strategy_from_run(metrics, trades)
        self.assertEqual(out["sample_telemetry"]["executed_trades"], 4)
        self.assertEqual(out["sample_telemetry"]["active_trading_days"], 2)
        self.assertEqual(out["raw_metrics"]["net_profit_factor"], 1.5)
        self.assertAlmostEqual(out["raw_metrics"]["expectancy_ratio"], 0.25, places=4)

    def test_from_run_falls_back_to_after_fees_pf(self) -> None:
        metrics = {
            "trade_count": 2,
            "net_profit": 100.0,
            "profit_factor": None,
            "outcome_audit": {"profit_factor_after_fees": 2.0},
            "max_drawdown": 50.0,
            "expectancy": 50.0,
            "stop_loss_per_trade_rupees": 100.0,
            "win_rate_pct": 100.0,
        }
        out = evaluate_strategy_from_run(metrics, [{"trading_day": "2026-07-01"}])
        self.assertEqual(out["raw_metrics"]["net_profit_factor"], 2.0)

    def test_attach_strategy_score(self) -> None:
        metrics = {
            "trade_count": 2,
            "net_profit": 100.0,
            "profit_factor": 1.2,
            "account_equity_max_drawdown": 40.0,
            "expectancy": 50.0,
            "stop_loss_per_trade_rupees": 200.0,
            "win_rate_pct": 50.0,
        }
        out = attach_strategy_score(
            metrics,
            [{"trading_day": "2026-07-01"}, {"trading_day": "2026-07-02"}],
        )
        self.assertEqual(out["active_trading_days"], 2)
        self.assertIn("strategy_score_v1", out)
        self.assertEqual(out["strategy_grade"], out["strategy_score_v1"]["grade"])
        self.assertEqual(out["evidence_score"], out["strategy_score_v1"]["evidence_score"])


if __name__ == "__main__":
    unittest.main()

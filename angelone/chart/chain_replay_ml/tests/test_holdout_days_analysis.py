"""Tests for holdout trading-day deep-dive analysis."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from chain_replay_ml.training.holdout_days_analysis import (
    build_holdout_days_analysis,
    build_holdout_days_analysis_csv,
    premium_hit_rate_pct,
)


class PremiumHitRateTests(unittest.TestCase):
    def test_all_within_tolerance(self) -> None:
        y = np.array([100.0, 50.0, 20.0])
        pred = np.array([102.0, 51.0, 20.5])  # <= 5%
        self.assertAlmostEqual(premium_hit_rate_pct(y, pred), 100.0, places=2)

    def test_partial_hits(self) -> None:
        y = np.array([100.0, 100.0])
        pred = np.array([104.0, 120.0])  # 4% hit, 20% miss
        self.assertAlmostEqual(premium_hit_rate_pct(y, pred), 50.0, places=2)


class HoldoutDaysAnalysisTests(unittest.TestCase):
    def _make_frames(self) -> tuple[pd.DataFrame, pd.Series, np.ndarray, pd.DataFrame, pd.Series, np.ndarray]:
        rng = np.random.default_rng(7)
        # Quiet training days
        train_rows = []
        for day, spot0 in (("2026-06-01", 24000.0), ("2026-06-02", 24100.0), ("2026-06-03", 24050.0)):
            for i in range(40):
                spot = spot0 + i * 0.5
                ltp = 100.0 + rng.normal(0, 1)
                train_rows.append({
                    "trading_day": day,
                    "spot": spot,
                    "ltp": ltp,
                    "iv_zscore_1m": rng.normal(0, 0.5),
                    "is_expiry_day": 0,
                })
        train_df = pd.DataFrame(train_rows)
        y_train = train_df["ltp"] + rng.normal(0, 0.5, len(train_df))
        pred_train = y_train.to_numpy() + rng.normal(0, 0.4, len(train_df))

        # Volatile holdout day + quieter holdout day
        ho_rows = []
        for i in range(50):
            ho_rows.append({
                "trading_day": "2026-06-25",
                "spot": 24500.0 + i * 8.0,  # strong trend / range
                "ltp": 110.0 + rng.normal(0, 8),
                "iv_zscore_1m": rng.normal(2.0, 0.8),
                "is_expiry_day": 0,
            })
        for i in range(50):
            ho_rows.append({
                "trading_day": "2026-07-01",
                "spot": 24600.0 + i * 0.4,
                "ltp": 105.0 + rng.normal(0, 1.5),
                "iv_zscore_1m": rng.normal(0.2, 0.4),
                "is_expiry_day": 1,
            })
        ho_df = pd.DataFrame(ho_rows)
        y_ho = ho_df["ltp"].copy()
        # Day 1 predictions much worse
        pred_ho = y_ho.to_numpy().copy()
        pred_ho[:50] = pred_ho[:50] + 25.0
        pred_ho[50:] = pred_ho[50:] + rng.normal(0, 1.0, 50)
        return ho_df, y_ho, pred_ho, train_df, y_train, pred_train

    def test_builds_two_days_with_regime_flags(self) -> None:
        ho_df, y_ho, pred_ho, train_df, y_train, pred_train = self._make_frames()
        out = build_holdout_days_analysis(
            holdout_df=ho_df,
            y_holdout=y_ho,
            pred_holdout=pred_ho,
            train_df=train_df,
            y_train=y_train,
            pred_train=pred_train,
            model_name="test_model",
        )
        self.assertTrue(out.get("ok"))
        self.assertEqual(out["executive_summary"]["holdout_day_count"], 2)
        self.assertEqual(out["executive_summary"]["trading_days"], ["2026-06-25", "2026-07-01"])
        days = {d["trading_day"]: d for d in out["days"]}
        d1 = days["2026-06-25"]
        self.assertIn("mae", d1["metrics"])
        self.assertIn("hit_rate_pct", d1["metrics"])
        self.assertIn("directional_accuracy_pct", d1["metrics"])
        self.assertTrue(len(d1["premium_bands"]) >= 1)
        self.assertIsNotNone(d1["volatility"].get("spot_range_pct"))
        self.assertTrue(len((d1.get("vs_training_day_avg") or {}).get("rows") or []) >= 3)
        self.assertIn("regime", d1)
        # Volatile / bad MAE day should flag regime or model stress
        self.assertTrue(
            d1["regime"].get("is_regime_shift")
            or "model_stress" in (d1["regime"].get("flags") or [])
            or "trending" in (d1["regime"].get("flags") or [])
        )
        d2 = days["2026-07-01"]
        self.assertTrue(d2["volatility"].get("is_expiry_day"))

    def test_csv_contains_day_sections(self) -> None:
        ho_df, y_ho, pred_ho, train_df, y_train, pred_train = self._make_frames()
        out = build_holdout_days_analysis(
            holdout_df=ho_df,
            y_holdout=y_ho,
            pred_holdout=pred_ho,
            train_df=train_df,
            y_train=y_train,
            pred_train=pred_train,
        )
        csv_text = build_holdout_days_analysis_csv(out)
        self.assertIn("Trading Day: 2026-06-25", csv_text)
        self.assertIn("Trading Day: 2026-07-01", csv_text)
        self.assertIn("Regime Assessment", csv_text)
        self.assertIn("Endpoint Hit", csv_text)


if __name__ == "__main__":
    unittest.main()

"""Tests for top 1% error driver attribution."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from chain_replay_ml.training.holdout_top1_drivers import (
    analyze_top1_error_drivers,
    score_primary_driver_separation,
)


class _LinearStub:
    def __init__(self, weights: dict[str, float]) -> None:
        self._weights = weights

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        out = np.zeros(len(X), dtype=float)
        for feat, w in self._weights.items():
            if feat in X.columns:
                out += pd.to_numeric(X[feat], errors="coerce").fillna(0).to_numpy() * w
        return out


class Top1DriverTests(unittest.TestCase):
    def test_separation_ranks_expiry_gamma(self) -> None:
        top = pd.DataFrame({
            "gamma": [0.005, 0.006, 0.007, 0.0055],
            "minutes_to_expiry": [40, 55, 70, 45],
            "current_iv": [24, 25, 26, 23],
            "spot_change_5m": [-0.002, -0.0015, -0.0025, -0.0018],
        })
        rest = pd.DataFrame({
            "gamma": [0.001, 0.0012, 0.0009, 0.0011],
            "minutes_to_expiry": [7000, 7400, 7200, 7500],
            "current_iv": [11, 12, 10, 11.5],
            "spot_change_5m": [-0.0001, 0.0002, -0.0002, 0.0001],
        })
        rows = score_primary_driver_separation(top_df=top, rest_df=rest)
        by_key = {r["key"]: r for r in rows}
        self.assertGreater(float(by_key["time_to_expiry"]["separation_score"]), 1.0)
        self.assertGreater(float(by_key["gamma"]["separation_score"]), 0.5)
        self.assertEqual(len(rows), 4)

    def test_permutation_importance_on_top1_subset(self) -> None:
        n = 40
        rng = np.random.default_rng(7)
        X = pd.DataFrame({
            "gamma": rng.uniform(0.004, 0.008, n),
            "minutes_to_expiry": rng.uniform(30, 90, n),
            "current_iv": rng.uniform(20, 28, n),
            "spot_change_5m": rng.uniform(-0.02, 0.02, n),
            "noise_feat": rng.uniform(0, 1, n),
        })
        y = (
            20
            + X["gamma"] * 4000
            + (90 - X["minutes_to_expiry"]) * 0.5
            + X["current_iv"] * 0.2
        ).to_numpy()
        model = _LinearStub({
            "gamma": 4000.0,
            "minutes_to_expiry": -0.5,
            "current_iv": 0.2,
            "spot_change_5m": 0.0,
            "noise_feat": 0.0,
        })
        pred = model.predict(X)
        pred_bad = pred.copy()
        pred_bad[:5] = pred_bad[:5] * 8
        top_df = X.iloc[:20]
        rest_df = X.iloc[20:]
        out = analyze_top1_error_drivers(
            top_df=top_df,
            rest_df=rest_df,
            model=model,
            X_top1=top_df,
            y_top1=y[:20],
            features=list(X.columns),
        )
        self.assertTrue(out.get("primary_driver"))
        imp = out.get("feature_error_importance") or []
        self.assertTrue(len(imp) >= 3)
        driver_err = out.get("driver_error_contribution") or []
        self.assertTrue(any(d.get("error_contribution_pct", 0) > 0 for d in driver_err))


if __name__ == "__main__":
    unittest.main()

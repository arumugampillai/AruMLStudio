"""Tests for holdout top 1% error investigation."""

from __future__ import annotations

import os
import tempfile
import unittest

import numpy as np
import pandas as pd

from chain_replay_ml.training.holdout_top1_analysis import (
    build_top1_analysis_csv,
    build_top1_error_analysis,
    save_top1_investigation_knowledge,
)


class Top1ErrorAnalysisTests(unittest.TestCase):
    def test_identifies_expiry_low_premium_pattern(self) -> None:
        n = 1000
        y = np.full(n, 20.0)
        pred = y.copy()
        for i in range(12):
            pred[i] = 200.0 - i * 5
        df = pd.DataFrame({
            "ltp": y,
            "is_expiry_day": [1.0 if i < 12 else 0.0 for i in range(n)],
            "is_expiry_week": 0.0,
            "gamma": 0.08,
            "current_iv": 0.25,
            "iv_zscore_5m": 2.5,
            "delta": 0.4,
            "theta": -0.02,
            "vega": 0.1,
            "minutes_to_expiry": 60.0,
            "strike_distance_from_atm": 3.0,
            "timestamp": np.arange(n) + 1_700_000_000,
        })
        out = build_top1_error_analysis(
            ho_df=df,
            y_ho=pd.Series(y),
            pred_ho=pred,
            feature_drift_ranking=[
                {"feature": "gamma", "drift_pct": 18.0, "importance": 0.17, "drift": 1.2},
            ],
            model_name="test_model",
        )
        self.assertTrue(out.get("ok"))
        self.assertEqual(out["top1_row_count"], 10)
        ex = out["executive_summary"]
        self.assertGreaterEqual(ex["avg_premium_error_pct"], 90.0)
        self.assertTrue(any("Expiry" in p for p in ex.get("patterns") or []))
        conc = out["conclusion"]
        self.assertGreaterEqual(float(conc["confidence_pct"]), 60.0)
        self.assertIn("Executive Summary", build_top1_analysis_csv(out))

    def test_save_knowledge(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            os.makedirs(os.path.join(tmp, "research_lab"), exist_ok=True)
            open(os.path.join(tmp, "research_lab", "sessions.db"), "w").close()
            analysis = build_top1_error_analysis(
                ho_df=pd.DataFrame({"ltp": [20.0], "is_expiry_day": [1.0]}),
                y_ho=pd.Series([20.0]),
                pred_ho=np.array([100.0]),
                model_name="m1",
            )
            out = save_top1_investigation_knowledge(tmp, analysis, model_name="m1")
            self.assertTrue(out.get("ok"))
            self.assertIn("holdout_top1:m1", str(out.get("finding_key")))


if __name__ == "__main__":
    unittest.main()

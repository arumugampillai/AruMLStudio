"""Tests for Feature Profiles (Phase 2.2)."""

from __future__ import annotations

import os
import tempfile
import unittest

import numpy as np
import pandas as pd

from chain_replay_ml.dataset_builder.analysis_correlation import run_correlation_analysis
from chain_replay_ml.dataset_builder.analysis_feature_profiles import (
    build_feature_profiles,
    list_profile_features,
    load_feature_profile,
    load_feature_scorecard,
    profiles_exist,
)
from chain_replay_ml.dataset_builder.analysis_lab_store import (
    ensure_analysis_run,
    register_dataset,
)


class FeatureProfileTests(unittest.TestCase):
    def test_build_profile_from_parquet_and_correlation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            n = 120
            spot = pd.Series(np.linspace(100, 200, n))
            df = pd.DataFrame(
                {
                    "trading_day": ["2026-07-24"] * n,
                    "spot": spot,
                    "spot_ema20": spot * 1.001,
                    "current_iv": np.linspace(0.1, 0.2, n),
                    "noise": np.random.default_rng(0).normal(size=n),
                }
            )
            # Leading nulls → warmup-like for lag column
            lag = spot.shift(3)
            df["spot_lag_9s"] = lag

            path = os.path.join(tmp, "profiles_demo.parquet")
            df.to_parquet(path, index=False)
            # Sidecar with a simple lag transform for lineage
            import json

            with open(os.path.join(tmp, "profiles_demo.json"), "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "transformations": [
                            {
                                "id": "lag",
                                "enabled": True,
                                "params": {
                                    "features": ["spot"],
                                    "horizons": [
                                        {"seconds": 9, "column": "spot_lag_9s"}
                                    ],
                                },
                            }
                        ]
                    },
                    f,
                )

            register_dataset(tmp, path, name="profiles_demo")
            run = ensure_analysis_run(tmp, "profiles_demo")
            run_correlation_analysis(
                tmp, run["run_id"], {"path": path, "dataset_id": "profiles_demo"}
            )
            summary = build_feature_profiles(
                tmp, run["run_id"], {"path": path, "dataset_id": "profiles_demo"}
            )
            self.assertGreaterEqual(int(summary["features"]), 4)
            self.assertTrue(profiles_exist(tmp, run["run_id"]))

            names = list_profile_features(tmp, run["run_id"])
            self.assertIn("spot", names)
            self.assertIn("spot_lag_9s", names)

            prof = load_feature_profile(tmp, run["run_id"], "spot_lag_9s")
            self.assertIsNotNone(prof)
            assert prof is not None
            self.assertEqual(prof["source"], "Pipeline")
            self.assertIn("spot", prof.get("parents") or [])
            self.assertGreater(float(prof.get("null_pct") or 0.0), 0.0)
            self.assertEqual(prof.get("recommendation") in {"Keep", "Review", "Duplicate Candidate"}, True)

            card = load_feature_scorecard(tmp, run["run_id"])
            self.assertTrue(card)
            self.assertTrue(any(r["feature_name"] == "spot" for r in card))
            # Future metrics pending
            spot_row = next(r for r in card if r["feature_name"] == "spot")
            self.assertIsNone(spot_row.get("mi_score"))
            self.assertIsNone(spot_row.get("shap_importance"))


if __name__ == "__main__":
    unittest.main()

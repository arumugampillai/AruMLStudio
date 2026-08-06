"""Wave 6: registry % packaging moved to Interaction / Pipeline Owned."""

from __future__ import annotations

import unittest

import pandas as pd

from chain_replay_ml.dataset_builder.feature_migration import is_pipeline_owned
from chain_replay_ml.dataset_builder.feature_ownership import (
    OWNERSHIP_PIPELINE_OWNED,
    evaluate_registry_admission,
    future_generator_of,
    ownership_of,
)
from chain_replay_ml.dataset_builder.feature_plugins import _REGISTRY_FEATURES
from chain_replay_ml.dataset_builder.transformations import run_transformation_pipeline
from chain_replay_ml.dataset_builder.transformations.base import TransformContext


_MOVED = (
    "spot_vs_ema20_pct",
    "ema_spread_pct",
    "ema_spread_vs_spot_pct",
    "ce_pe_atm6_ltp_diff_pct",
)

_CANONICAL = (
    "spot",
    "spot_ema9",
    "spot_ema20",
    "ce_atm6_ltp_sum",
    "pe_atm6_ltp_sum",
)


class TestWave6RegistryPctPackaging(unittest.TestCase):
    def test_pipeline_owned_and_removed_from_registry(self) -> None:
        all_feats = {f for feats in _REGISTRY_FEATURES.values() for f in feats}
        for name in _MOVED:
            self.assertTrue(is_pipeline_owned(name), msg=name)
            self.assertEqual(ownership_of(name), OWNERSHIP_PIPELINE_OWNED)
            self.assertEqual(future_generator_of(name), "interaction")
            self.assertNotIn(name, all_feats)
            decision = evaluate_registry_admission(
                name,
                generic_registry_math=True,
            )
            self.assertFalse(decision["allowed"], msg=name)

    def test_canonical_inputs_remain(self) -> None:
        all_feats = {f for feats in _REGISTRY_FEATURES.values() for f in feats}
        for name in _CANONICAL:
            self.assertIn(name, all_feats)

    def test_interaction_reconstructs_pct_packaging(self) -> None:
        df = pd.DataFrame(
            {
                "trading_day": ["2026-05-27"] * 3,
                "token": ["T"] * 3,
                "spot": [25000.0, 25100.0, 24900.0],
                "spot_ema9": [24950.0, 25050.0, 24920.0],
                "spot_ema20": [24900.0, 25000.0, 24950.0],
                "ce_atm6_ltp_sum": [120.0, 130.0, 110.0],
                "pe_atm6_ltp_sum": [100.0, 90.0, 115.0],
            }
        )
        cfg = {
            "transformation_pipeline_version": 1,
            "transformations": [
                {
                    "id": "interaction",
                    "enabled": True,
                    "params": {
                        "pairs": [
                            {
                                "left": "spot",
                                "right": "spot_ema20",
                                "op": "subtract",
                                "output": "spot_minus_spot_ema20",
                            },
                            {
                                "left": "spot_minus_spot_ema20",
                                "right": "spot_ema20",
                                "op": "divide",
                                "output": "spot_vs_ema20_pct",
                                "scale": 100.0,
                                "eps": 1.0e-6,
                            },
                            {
                                "left": "spot_ema9",
                                "right": "spot_ema20",
                                "op": "subtract",
                                "output": "spot_ema9_minus_spot_ema20",
                            },
                            {
                                "left": "spot_ema9_minus_spot_ema20",
                                "right": "spot_ema20",
                                "op": "divide",
                                "output": "ema_spread_pct",
                                "scale": 100.0,
                                "eps": 1.0e-6,
                            },
                            {
                                "left": "spot_ema9_minus_spot_ema20",
                                "right": "spot",
                                "op": "divide",
                                "output": "ema_spread_vs_spot_pct",
                                "scale": 100.0,
                                "eps": 1.0e-6,
                            },
                            {
                                "left": "ce_atm6_ltp_sum",
                                "right": "pe_atm6_ltp_sum",
                                "op": "subtract",
                                "output": "ce_minus_pe_tmp",
                            },
                            {
                                "left": "ce_atm6_ltp_sum",
                                "right": "pe_atm6_ltp_sum",
                                "op": "add",
                                "output": "ce_plus_pe_tmp",
                            },
                            {
                                "left": "ce_minus_pe_tmp",
                                "right": "ce_plus_pe_tmp",
                                "op": "divide",
                                "output": "ce_pe_atm6_ltp_diff_pct",
                                "eps": 1.0e-6,
                            },
                        ]
                    },
                }
            ],
        }
        result = run_transformation_pipeline(
            df,
            cfg,
            context=TransformContext(sample_interval_sec=3),
        )
        out = result.frame
        for i in range(3):
            spot = float(df["spot"].iloc[i])
            e9 = float(df["spot_ema9"].iloc[i])
            e20 = float(df["spot_ema20"].iloc[i])
            ce = float(df["ce_atm6_ltp_sum"].iloc[i])
            pe = float(df["pe_atm6_ltp_sum"].iloc[i])
            self.assertAlmostEqual(
                float(out["spot_vs_ema20_pct"].iloc[i]),
                100.0 * (spot - e20) / e20,
                places=9,
            )
            self.assertAlmostEqual(
                float(out["ema_spread_pct"].iloc[i]),
                100.0 * (e9 - e20) / e20,
                places=9,
            )
            self.assertAlmostEqual(
                float(out["ema_spread_vs_spot_pct"].iloc[i]),
                100.0 * (e9 - e20) / spot,
                places=9,
            )
            self.assertAlmostEqual(
                float(out["ce_pe_atm6_ltp_diff_pct"].iloc[i]),
                (ce - pe) / (ce + pe),
                places=9,
            )


if __name__ == "__main__":
    unittest.main()

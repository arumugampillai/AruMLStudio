"""Unit tests for Phase 3 transforms (denom_eps, range_eps, Return+Interaction)."""

from __future__ import annotations

import unittest

import pandas as pd

from chain_replay_ml.dataset_builder.transformations import run_transformation_pipeline
from chain_replay_ml.dataset_builder.transformations.base import TransformContext


def _run_pipeline(df: pd.DataFrame, transformations: list[dict], *, interval: float = 3.0):
    cfg = {
        "transformation_pipeline_version": 1,
        "transformations": transformations,
    }
    ctx = TransformContext(config=cfg, sample_interval_sec=interval)
    return run_transformation_pipeline(df, cfg, context=ctx)


class Phase3TransformTests(unittest.TestCase):
    def test_return_denom_eps_allows_zero_lag(self) -> None:
        df = pd.DataFrame({
            "token": ["A", "A"],
            "volume": [0.0, 10.0],
        })
        result = _run_pipeline(
            df,
            [{
                "id": "return",
                "enabled": True,
                "params": {
                    "features": ["volume"],
                    "horizons": [{"seconds": 3, "column": "volume_frac_3s"}],
                    "scale": 1.0,
                    "denom_eps": 1e-9,
                    "partition_by": ["token"],
                    "sample_interval_sec": 3.0,
                },
            }],
        )
        # (10 - 0) / (0 + 1e-9)
        self.assertAlmostEqual(
            result.frame.loc[1, "volume_frac_3s"],
            10.0 / 1e-9,
            places=3,
        )

    def test_rolling_ohlc_range_pos_eps_when_flat(self) -> None:
        # Constant series → high == low; with eps range_pos stays finite (0).
        vals = [100.0, 100.0, 100.0]
        df = pd.DataFrame({"token": ["A"] * 3, "spot": vals})
        result = _run_pipeline(
            df,
            [{
                "id": "rolling_ohlc",
                "enabled": True,
                "params": {
                    "features": ["spot"],
                    "windows": [{"seconds": 9, "suffix": "9s"}],
                    "outputs": ["range_pos"],
                    "range_eps": 1e-9,
                    "partition_by": ["token"],
                    "sample_interval_sec": 3.0,
                },
            }],
        )
        col = "spot_range_pos_9s"
        self.assertIn(col, result.frame.columns)
        self.assertTrue(pd.notna(result.frame.loc[2, col]))
        self.assertAlmostEqual(result.frame.loc[2, col], 0.0)

    def test_return_then_interaction_ltp_x_volume(self) -> None:
        df = pd.DataFrame({
            "token": ["A"] * 3,
            "ltp": [2.0, 2.0, 2.0],
            "volume": [100.0, 110.0, 121.0],
        })
        result = _run_pipeline(
            df,
            [
                {
                    "id": "return",
                    "enabled": True,
                    "params": {
                        "features": ["volume"],
                        "horizons": [{"seconds": 3, "column": "volume_return_frac_3s"}],
                        "scale": 1.0,
                        "denom_eps": 1e-9,
                        "partition_by": ["token"],
                        "sample_interval_sec": 3.0,
                    },
                },
                {
                    "id": "interaction",
                    "enabled": True,
                    "params": {
                        "pairs": [{
                            "left": "ltp",
                            "right": "volume_return_frac_3s",
                            "op": "mul",
                            "output": "ltp_x_volume_change_pct_3s",
                        }],
                        "sample_interval_sec": 3.0,
                    },
                },
            ],
        )
        # row1: vol_return = (110-100)/(100+1e-9) ≈ 0.1; product = 2 * 0.1 = 0.2
        expected = 2.0 * ((110.0 - 100.0) / (100.0 + 1e-9))
        self.assertAlmostEqual(
            result.frame.loc[1, "ltp_x_volume_change_pct_3s"],
            expected,
        )


if __name__ == "__main__":
    unittest.main()

"""Unit tests for Phase 2 feature transforms."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from chain_replay_ml.dataset_builder.transformations import run_transformation_pipeline
from chain_replay_ml.dataset_builder.transformations.base import TransformContext


def _run(df: pd.DataFrame, transform_id: str, params: dict, *, interval: float = 3.0):
    cfg = {
        "transformation_pipeline_version": 1,
        "transformations": [{
            "id": transform_id,
            "enabled": True,
            "params": {**params, "sample_interval_sec": interval},
        }],
    }
    ctx = TransformContext(config=cfg, sample_interval_sec=interval)
    return run_transformation_pipeline(df, cfg, context=ctx)


class Phase2TransformTests(unittest.TestCase):
    def test_rolling_zscore(self) -> None:
        # window=3 rows @ interval 3s → 9s; values constant then spike
        vals = [10.0, 10.0, 10.0, 13.0, 10.0, 10.0]
        df = pd.DataFrame({
            "token": ["A"] * len(vals),
            "x": vals,
        })
        result = _run(
            df,
            "rolling_statistics",
            {
                "features": ["x"],
                "windows": [{"seconds": 9, "suffix": "9s"}],
                "stat": "zscore",
                "ddof": 0,
                "partition_by": ["token"],
            },
        )
        col = "x_zscore_9s"
        self.assertIn(col, result.frame.columns)
        # First two rows incomplete window
        self.assertTrue(pd.isna(result.frame.loc[0, col]))
        self.assertTrue(pd.isna(result.frame.loc[1, col]))
        # Row 2: window all 10 → std 0 → NaN
        self.assertTrue(pd.isna(result.frame.loc[2, col]))
        # Row 3: [10,10,13] mean=11, std=np.std([10,10,13], ddof=0)=√2
        expected = (13.0 - 11.0) / float(np.std([10.0, 10.0, 13.0], ddof=0))
        self.assertAlmostEqual(result.frame.loc[3, col], expected)

    def test_rolling_ohlc_body_range_pos(self) -> None:
        vals = [100.0, 102.0, 101.0, 104.0]
        df = pd.DataFrame({"token": ["A"] * 4, "spot": vals})
        result = _run(
            df,
            "rolling_ohlc",
            {
                "features": ["spot"],
                "windows": [{"seconds": 9, "suffix": "9s"}],
                "outputs": ["body_pct", "range_pct", "range_pos"],
                "partition_by": ["token"],
            },
            interval=3.0,
        )
        frame = result.frame
        # window rows=3; row 2: open=vals[0]=100, close=101, high=102, low=100
        self.assertAlmostEqual(frame.loc[2, "spot_body_pct_9s"], (101 - 100) / 100 * 100)
        self.assertAlmostEqual(frame.loc[2, "spot_range_pct_9s"], (102 - 100) / 100 * 100)
        self.assertAlmostEqual(frame.loc[2, "spot_range_pos_9s"], (101 - 100) / (102 - 100))

    def test_interaction_mul_div_scale(self) -> None:
        df = pd.DataFrame({"a": [2.0, 4.0], "b": [3.0, 0.0]})
        result = _run(
            df,
            "interaction",
            {
                "pairs": [
                    {"left": "a", "right": "b", "op": "mul", "output": "a_x_b", "scale": 2.0},
                    {"left": "a", "right": "b", "op": "div", "output": "a_div_b"},
                ],
            },
        )
        self.assertEqual(result.frame.loc[0, "a_x_b"], 12.0)
        self.assertEqual(result.frame.loc[1, "a_x_b"], 0.0)
        self.assertEqual(result.frame.loc[0, "a_div_b"], 2.0 / 3.0)
        self.assertTrue(pd.isna(result.frame.loc[1, "a_div_b"]))

    def test_derived_slope_and_accel(self) -> None:
        # interval 60s so 300s = 5 rows, 60s = 1 row
        n = 8
        s = [float(i) for i in range(n)]
        df = pd.DataFrame({"token": ["A"] * n, "atm_straddle": s})
        result = _run(
            df,
            "derived",
            {
                "outputs": [
                    {
                        "feature": "atm_straddle",
                        "column": "atm_straddle_slope_5m",
                        "terms": [
                            {"seconds": 0, "coeff": 1.0 / 5.0},
                            {"seconds": 300, "coeff": -1.0 / 5.0},
                        ],
                    },
                    {
                        "feature": "atm_straddle",
                        "column": "atm_straddle_change_accel",
                        "terms": [
                            {"seconds": 0, "coeff": 0.8},
                            {"seconds": 60, "coeff": -1.0},
                            {"seconds": 300, "coeff": 0.2},
                        ],
                    },
                ],
                "partition_by": ["token"],
            },
            interval=60.0,
        )
        frame = result.frame
        # row 5: (s5 - s0) / 5 = (5-0)/5 = 1
        self.assertAlmostEqual(frame.loc[5, "atm_straddle_slope_5m"], 1.0)
        # row 5: 0.8*5 - 1.0*4 + 0.2*0 = 4 - 4 + 0 = 0
        self.assertAlmostEqual(frame.loc[5, "atm_straddle_change_accel"], 0.0)

    def test_difference_clip_non_negative(self) -> None:
        df = pd.DataFrame({
            "token": ["A"] * 4,
            "volume": [10.0, 8.0, 12.0, 11.0],
        })
        result = _run(
            df,
            "difference_clip",
            {
                "features": ["volume"],
                "horizons": [{"seconds": 3, "suffix": "3s"}],
                "clip_min": 0.0,
                "partition_by": ["token"],
            },
        )
        col = "volume_flow_3s"
        self.assertIn(col, result.frame.columns)
        self.assertTrue(pd.isna(result.frame.loc[0, col]))
        # 8-10 = -2 → clipped to 0
        self.assertEqual(result.frame.loc[1, col], 0.0)
        # 12-8 = 4
        self.assertEqual(result.frame.loc[2, col], 4.0)

    def test_anchor_return_from_first(self) -> None:
        df = pd.DataFrame({
            "trading_day": ["d1"] * 3 + ["d2"] * 2,
            "atm_straddle": [100.0, 110.0, 90.0, 50.0, 55.0],
        })
        result = _run(
            df,
            "anchor_return",
            {
                "outputs": [{
                    "feature": "atm_straddle",
                    "column": "atm_straddle_pct_change_from_open",
                }],
                "scale": 100.0,
                "partition_by": ["trading_day"],
            },
        )
        col = "atm_straddle_pct_change_from_open"
        self.assertEqual(result.frame.loc[0, col], 0.0)
        self.assertAlmostEqual(result.frame.loc[1, col], 10.0)
        self.assertAlmostEqual(result.frame.loc[2, col], -10.0)
        self.assertEqual(result.frame.loc[3, col], 0.0)
        self.assertAlmostEqual(result.frame.loc[4, col], 10.0)


if __name__ == "__main__":
    unittest.main()

"""Unit tests for the Rolling feature transformation."""

from __future__ import annotations

import unittest

import pandas as pd

from chain_replay_ml.dataset_builder.transformations import (
    default_transformation_config,
    run_transformation_pipeline,
)
from chain_replay_ml.dataset_builder.transformations.base import TransformContext
from chain_replay_ml.dataset_builder.transformations.rolling import (
    rolling_column_name,
)
from chain_replay_ml.dataset_builder.transformations.rolling_ui import (
    build_rolling_transformation_config,
    merge_rolling_into_config,
)


class RollingTransformTests(unittest.TestCase):
    def test_rolling_column_naming(self) -> None:
        self.assertEqual(rolling_column_name("ltp", "mean", 5), "ltp_roll_mean_5")
        self.assertEqual(rolling_column_name("spot", "std", 20), "spot_roll_std_20")
        self.assertEqual(rolling_column_name("iv", "median", 10), "iv_roll_median_10")

    def test_multi_ops_share_feature_window(self) -> None:
        """Multiple ops on same (feature, window) produce distinct columns."""
        vals = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        df = pd.DataFrame({
            "trading_day": ["2024-01-01"] * len(vals),
            "token": ["A"] * len(vals),
            "x": vals,
        })
        cfg = {
            "transformation_pipeline_version": 1,
            "transformations": [{
                "id": "rolling",
                "enabled": True,
                "params": {
                    "features": ["x"],
                    "windows": [3],
                    "operations": ["mean", "min", "max"],
                    "partition_by": ["trading_day", "token"],
                    "sample_interval_sec": 3.0,
                },
            }],
        }
        ctx = TransformContext(config=cfg, sample_interval_sec=3.0)
        result = run_transformation_pipeline(df, cfg, context=ctx)
        self.assertEqual(result.executed, 1)
        self.assertEqual(
            set(result.created_columns),
            {"x_roll_mean_3", "x_roll_min_3", "x_roll_max_3"},
        )
        # Row 2: window [1,2,3]
        self.assertAlmostEqual(result.frame.loc[2, "x_roll_mean_3"], 2.0)
        self.assertAlmostEqual(result.frame.loc[2, "x_roll_min_3"], 1.0)
        self.assertAlmostEqual(result.frame.loc[2, "x_roll_max_3"], 3.0)

    def test_disabled_passthrough(self) -> None:
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
        cfg = build_rolling_transformation_config(
            enabled=False,
            features=["x"],
            windows=[5],
            operations=["mean"],
        )
        self.assertEqual(cfg["transformations"], [])
        result = run_transformation_pipeline(df, cfg)
        self.assertIs(result.frame, df)
        self.assertEqual(result.executed, 0)
        self.assertTrue(all(not c.startswith("x_roll_") for c in result.frame.columns))

        # Explicit enabled=False entry in config is also skipped by pipeline.
        cfg2 = {
            "transformation_pipeline_version": 1,
            "transformations": [{
                "id": "rolling",
                "enabled": False,
                "params": {
                    "features": ["x"],
                    "windows": [2],
                    "operations": ["mean"],
                },
            }],
        }
        result2 = run_transformation_pipeline(df, cfg2)
        self.assertEqual(result2.executed, 0)
        self.assertNotIn("x_roll_mean_2", result2.frame.columns)

    def test_mean_window_correctness(self) -> None:
        vals = [10.0, 20.0, 30.0, 40.0, 50.0]
        df = pd.DataFrame({
            "trading_day": ["2024-01-01"] * len(vals),
            "token": ["T"] * len(vals),
            "ltp": vals,
        })
        cfg = {
            "transformation_pipeline_version": 1,
            "transformations": [{
                "id": "rolling",
                "enabled": True,
                "params": {
                    "features": ["ltp"],
                    "windows": [3],
                    "operations": ["mean"],
                    "partition_by": ["trading_day", "token"],
                    "sample_interval_sec": 10.0,
                },
            }],
        }
        ctx = TransformContext(config=cfg, sample_interval_sec=10.0)
        result = run_transformation_pipeline(df, cfg, context=ctx)
        col = "ltp_roll_mean_3"
        self.assertIn(col, result.frame.columns)
        self.assertTrue(pd.isna(result.frame.loc[0, col]))
        self.assertTrue(pd.isna(result.frame.loc[1, col]))
        self.assertAlmostEqual(result.frame.loc[2, col], 20.0)  # (10+20+30)/3
        self.assertAlmostEqual(result.frame.loc[3, col], 30.0)  # (20+30+40)/3
        self.assertAlmostEqual(result.frame.loc[4, col], 40.0)  # (30+40+50)/3

    def test_merge_rolling_into_config_when_disabled(self) -> None:
        base = default_transformation_config()
        merged = merge_rolling_into_config(
            base,
            enabled=False,
            features=["ltp"],
            windows=[5, 10],
            operations=["mean", "std"],
        )
        self.assertEqual(merged["transformations"], [])
        ids = [t.get("id") for t in merged["transformations"]]
        self.assertNotIn("rolling", ids)


if __name__ == "__main__":
    unittest.main()

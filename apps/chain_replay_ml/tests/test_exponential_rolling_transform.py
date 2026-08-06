"""Unit tests for the Exponential Rolling feature transformation."""

from __future__ import annotations

import unittest

import pandas as pd

from chain_replay_ml.dataset_builder.transformations import (
    default_transformation_config,
    run_transformation_pipeline,
)
from chain_replay_ml.dataset_builder.transformations.base import TransformContext
from chain_replay_ml.dataset_builder.transformations.exponential_rolling import (
    EXPONENTIAL_ROLLING_OPS,
    exponential_rolling_column_name,
    normalize_exponential_rolling_op,
)
from chain_replay_ml.dataset_builder.transformations.exponential_rolling_ui import (
    build_exponential_rolling_config,
    merge_exponential_rolling_into_config,
)
from chain_replay_ml.dataset_builder.transformations.rolling_ui import (
    merge_rolling_into_config,
)
from chain_replay_ml.dataset_builder.transformations.time_shift import LagConfigError


class ExponentialRollingTransformTests(unittest.TestCase):
    def test_exponential_rolling_column_naming(self) -> None:
        self.assertEqual(
            exponential_rolling_column_name("delta", "ema", 20),
            "delta_ema_20",
        )
        self.assertEqual(
            exponential_rolling_column_name("iv", "ewm_mean", 50),
            "iv_ewm_mean_50",
        )
        self.assertEqual(
            exponential_rolling_column_name("iv", "ewm_std", 20),
            "iv_ewm_std_20",
        )
        self.assertEqual(
            exponential_rolling_column_name("gamma", "ema", 10),
            "gamma_ema_10",
        )

    def test_supported_ops(self) -> None:
        self.assertEqual(EXPONENTIAL_ROLLING_OPS, ("ema", "ewm_mean", "ewm_std"))
        self.assertEqual(normalize_exponential_rolling_op("EWM_MEAN"), "ewm_mean")
        with self.assertRaises(LagConfigError):
            normalize_exponential_rolling_op("ewm_var")

    def test_disabled_passthrough(self) -> None:
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
        cfg = build_exponential_rolling_config(
            enabled=False,
            features=["x"],
            periods=[5],
            operations=["ema"],
        )
        self.assertEqual(cfg["transformations"], [])
        result = run_transformation_pipeline(df, cfg)
        self.assertIs(result.frame, df)
        self.assertEqual(result.executed, 0)
        self.assertTrue(all(not c.endswith("_ema_5") for c in result.frame.columns))

        cfg2 = {
            "transformation_pipeline_version": 1,
            "transformations": [{
                "id": "exponential_rolling",
                "enabled": False,
                "params": {
                    "features": ["x"],
                    "periods": [2],
                    "operations": ["ema"],
                },
            }],
        }
        result2 = run_transformation_pipeline(df, cfg2)
        self.assertEqual(result2.executed, 0)
        self.assertNotIn("x_ema_2", result2.frame.columns)

    def test_ema_matches_pandas_ewm_adjust_false(self) -> None:
        vals = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]
        df = pd.DataFrame({
            "trading_day": ["2024-01-01"] * len(vals),
            "token": ["T"] * len(vals),
            "delta": vals,
        })
        cfg = {
            "transformation_pipeline_version": 1,
            "transformations": [{
                "id": "exponential_rolling",
                "enabled": True,
                "params": {
                    "features": ["delta"],
                    "periods": [20],
                    "operations": ["ema"],
                    "partition_by": ["trading_day", "token"],
                    "sample_interval_sec": 10.0,
                },
            }],
        }
        ctx = TransformContext(config=cfg, sample_interval_sec=10.0)
        result = run_transformation_pipeline(df, cfg, context=ctx)
        col = "delta_ema_20"
        self.assertIn(col, result.frame.columns)
        expected = df["delta"].astype(float).ewm(span=20, adjust=False).mean()
        pd.testing.assert_series_equal(
            result.frame[col].reset_index(drop=True),
            expected.reset_index(drop=True),
            check_names=False,
        )

    def test_ewm_mean_and_std_match_pandas_default_adjust(self) -> None:
        vals = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]
        df = pd.DataFrame({
            "trading_day": ["2024-01-01"] * len(vals),
            "token": ["T"] * len(vals),
            "iv": vals,
        })
        cfg = {
            "transformation_pipeline_version": 1,
            "transformations": [{
                "id": "exponential_rolling",
                "enabled": True,
                "params": {
                    "features": ["iv"],
                    "periods": [20],
                    "operations": ["ewm_mean", "ewm_std"],
                    "partition_by": [],
                    "sample_interval_sec": 10.0,
                },
            }],
        }
        result = run_transformation_pipeline(df, cfg)
        ewm = df["iv"].astype(float).ewm(span=20, adjust=True)
        pd.testing.assert_series_equal(
            result.frame["iv_ewm_mean_20"].reset_index(drop=True),
            ewm.mean().reset_index(drop=True),
            check_names=False,
        )
        pd.testing.assert_series_equal(
            result.frame["iv_ewm_std_20"].reset_index(drop=True),
            ewm.std().reset_index(drop=True),
            check_names=False,
        )
        # EMA (adjust=False) differs from EWM mean (adjust=True) on this series.
        ema = df["iv"].astype(float).ewm(span=20, adjust=False).mean()
        self.assertFalse(
            result.frame["iv_ewm_mean_20"].equals(ema),
            "ewm_mean should not equal ema (adjust differs)",
        )

    def test_multi_op_multi_period(self) -> None:
        df = pd.DataFrame({
            "trading_day": ["2024-01-01"] * 5,
            "token": ["A"] * 5,
            "delta": [1.0, 2.0, 3.0, 4.0, 5.0],
        })
        cfg = build_exponential_rolling_config(
            enabled=True,
            features=["delta"],
            periods=[5, 10],
            operations=["ema", "ewm_std"],
            partition_by=["trading_day", "token"],
            sample_interval_sec=3.0,
        )
        result = run_transformation_pipeline(df, cfg)
        for name in (
            "delta_ema_5",
            "delta_ema_10",
            "delta_ewm_std_5",
            "delta_ewm_std_10",
        ):
            self.assertIn(name, result.frame.columns)

    def test_order_after_rolling_before_interaction(self) -> None:
        vals = [1.0, 2.0, 3.0, 4.0, 5.0]
        df = pd.DataFrame({
            "trading_day": ["2024-01-01"] * len(vals),
            "token": ["A"] * len(vals),
            "x": vals,
            "y": [v * 2 for v in vals],
        })
        cfg = default_transformation_config()
        cfg = merge_rolling_into_config(
            cfg,
            enabled=True,
            features=["x"],
            windows=[3],
            operations=["mean"],
            partition_by=["trading_day", "token"],
            sample_interval_sec=3.0,
        )
        cfg = merge_exponential_rolling_into_config(
            cfg,
            enabled=True,
            features=["x"],
            periods=[3],
            operations=["ema"],
            partition_by=["trading_day", "token"],
            sample_interval_sec=3.0,
        )
        cfg["transformations"].append({
            "id": "interaction",
            "enabled": True,
            "order": 50,
            "params": {
                "pairs": [{
                    "left": "x",
                    "right": "y",
                    "op": "multiply",
                    "output": "x_x_y",
                }],
            },
        })
        ctx = TransformContext(config=cfg, sample_interval_sec=3.0)
        result = run_transformation_pipeline(df, cfg, context=ctx)
        self.assertEqual(
            result.executed_ids,
            ["rolling", "exponential_rolling", "interaction"],
        )
        self.assertIn("x_roll_mean_3", result.created_columns)
        self.assertIn("x_ema_3", result.created_columns)
        self.assertIn("x_x_y", result.created_columns)

    def test_reject_duplicate_existing_column(self) -> None:
        df = pd.DataFrame({
            "trading_day": ["2024-01-01"] * 3,
            "token": ["A"] * 3,
            "delta": [1.0, 2.0, 3.0],
            "delta_ema_20": [0.0, 0.0, 0.0],
        })
        cfg = {
            "transformation_pipeline_version": 1,
            "transformations": [{
                "id": "exponential_rolling",
                "enabled": True,
                "params": {
                    "features": ["delta"],
                    "periods": [20],
                    "operations": ["ema"],
                    "partition_by": ["trading_day", "token"],
                },
            }],
        }
        with self.assertRaises(LagConfigError) as err:
            run_transformation_pipeline(df, cfg)
        self.assertIn("already exists", str(err.exception))

    def test_reject_period_non_positive(self) -> None:
        with self.assertRaises(LagConfigError):
            exponential_rolling_column_name("delta", "ema", 0)
        with self.assertRaises(LagConfigError):
            exponential_rolling_column_name("delta", "ema", -5)

        df = pd.DataFrame({
            "trading_day": ["2024-01-01"] * 3,
            "token": ["A"] * 3,
            "delta": [1.0, 2.0, 3.0],
        })
        cfg = {
            "transformation_pipeline_version": 1,
            "transformations": [{
                "id": "exponential_rolling",
                "enabled": True,
                "params": {
                    "features": ["delta"],
                    "periods": [0],
                    "operations": ["ema"],
                    "partition_by": ["trading_day", "token"],
                },
            }],
        }
        with self.assertRaises(LagConfigError) as err:
            run_transformation_pipeline(df, cfg)
        self.assertIn("positive", str(err.exception).lower())

    def test_default_operations_remain_ema_only(self) -> None:
        """Omitting operations keeps prior EMA-only behavior."""
        cfg = build_exponential_rolling_config(
            enabled=True,
            features=["x"],
            periods=[5],
        )
        entry = cfg["transformations"][0]
        self.assertEqual(entry["params"]["operations"], ["ema"])


if __name__ == "__main__":
    unittest.main()

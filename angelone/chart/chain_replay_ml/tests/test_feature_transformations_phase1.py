"""Feature transformation framework tests (Phase 1 architecture + Phase 2 Lag)."""

from __future__ import annotations

import unittest

import pandas as pd

from chain_replay_ml.dataset_builder.transformations import (
    TRANSFORMATION_PIPELINE_VERSION,
    default_transformation_config,
    describe_pipeline,
    format_pipeline_log_lines,
    list_registered_transformations,
    normalize_transformation_config,
    registered_transformation_count,
    run_transformation_pipeline,
)
from chain_replay_ml.dataset_builder.transformations.base import TransformContext
from chain_replay_ml.dataset_builder.transformations.lag import (
    LagConfigError,
    lag_column_name,
    resolve_lag_row_offsets,
)


class TransformationFrameworkTests(unittest.TestCase):
    def test_default_config_versioned(self) -> None:
        cfg = default_transformation_config()
        self.assertEqual(
            cfg,
            {
                "transformation_pipeline_version": TRANSFORMATION_PIPELINE_VERSION,
                "transformations": [],
            },
        )

    def test_registry_has_phase2_transforms(self) -> None:
        self.assertEqual(registered_transformation_count(), 12)
        ids = [t.id for t in list_registered_transformations()]
        self.assertEqual(
            ids,
            [
                "lag",
                "difference",
                "difference_clip",
                "return",
                "anchor_return",
                "rolling",
                "rolling_statistics",
                "exponential_rolling",
                "ohlc_aggregation",
                "rolling_ohlc",
                "interaction",
                "derived",
            ],
        )

    def test_depends_on_declared(self) -> None:
        by_id = {t.id: t for t in list_registered_transformations()}
        self.assertEqual(by_id["difference"].depends_on, [])
        self.assertEqual(by_id["return"].depends_on, [])
        self.assertEqual(by_id["lag"].depends_on, [])

    def test_context_has_extensible_fields(self) -> None:
        ctx = TransformContext(
            sample_interval_sec=10.0,
            warmup_seconds=60.0,
            prediction_minutes=5.0,
            dataset_info={"market": "NIFTY"},
            metadata={"row_count": 1},
        )
        self.assertEqual(ctx.sample_interval_sec, 10.0)
        self.assertEqual(ctx.warmup_seconds, 60.0)
        self.assertEqual(ctx.prediction_minutes, 5.0)
        self.assertEqual(ctx.dataset_info["market"], "NIFTY")

    def test_pipeline_passthrough_same_object(self) -> None:
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})
        result = run_transformation_pipeline(df, default_transformation_config())
        self.assertIs(result.frame, df)
        self.assertEqual(result.registered, 12)
        self.assertEqual(result.enabled, 0)
        self.assertEqual(result.executed, 0)
        self.assertEqual(result.metadata_block["transformations"], [])
        self.assertEqual(
            result.metadata_block["transformation_pipeline_version"],
            TRANSFORMATION_PIPELINE_VERSION,
        )

    def test_describe_pipeline_log_format(self) -> None:
        lines = format_pipeline_log_lines(describe_pipeline())
        self.assertEqual(lines[0], "Transformation Pipeline")
        self.assertIn("Registered : 12", lines[1])
        self.assertIn("Enabled    : 0", lines[2])

    def test_dependency_enforced_when_enabled(self) -> None:
        df = pd.DataFrame({"x": [1, 2]})
        cfg = {
            "transformation_pipeline_version": 1,
            "transformations": [
                {"id": "difference", "enabled": True, "depends_on": ["lag"]},
            ],
        }
        with self.assertRaises(ValueError) as ctx:
            run_transformation_pipeline(df, cfg)
        self.assertIn("depends on", str(ctx.exception).lower())

    def test_lag_seconds_to_rows_exact(self) -> None:
        self.assertEqual(
            resolve_lag_row_offsets([30, 60, 90], 3.0),
            [(30.0, 10), (60.0, 20), (90.0, 30)],
        )

    def test_lag_seconds_rejects_non_multiple(self) -> None:
        with self.assertRaises(LagConfigError) as ctx:
            resolve_lag_row_offsets([30], 7.0)
        self.assertIn("not divisible", str(ctx.exception).lower())

    def test_lag_column_name_uses_seconds(self) -> None:
        self.assertEqual(lag_column_name("ltp", 30), "ltp_lag_30s")
        self.assertEqual(lag_column_name("spot", 120), "spot_lag_120s")

    def test_lag_creates_columns(self) -> None:
        df = pd.DataFrame({
            "trading_day": ["2024-01-01"] * 6,
            "token": ["A"] * 3 + ["B"] * 3,
            "ltp": [10.0, 11.0, 12.0, 20.0, 21.0, 22.0],
            "spot": [100.0, 101.0, 102.0, 200.0, 201.0, 202.0],
        })
        cfg = {
            "transformation_pipeline_version": 1,
            "transformations": [{
                "id": "lag",
                "enabled": True,
                "params": {
                    "features": ["ltp", "spot"],
                    "lag_seconds": [3, 6],
                    "partition_by": ["trading_day", "token"],
                },
            }],
        }
        ctx = TransformContext(config=cfg, sample_interval_sec=3.0)
        result = run_transformation_pipeline(df, cfg, context=ctx)
        self.assertEqual(result.executed, 1)
        self.assertEqual(len(result.created_columns), 4)
        self.assertIn("ltp_lag_3s", result.frame.columns)
        self.assertIn("spot_lag_6s", result.frame.columns)
        # 3s @ 3s interval = 1 row
        self.assertTrue(pd.isna(result.frame.loc[0, "ltp_lag_3s"]))
        self.assertEqual(result.frame.loc[1, "ltp_lag_3s"], 10.0)
        self.assertTrue(pd.isna(result.frame.loc[3, "ltp_lag_3s"]))
        self.assertEqual(result.frame.loc[4, "ltp_lag_3s"], 20.0)
        # Full params preserved for reproducibility
        entry = result.metadata_block["transformations"][0]
        self.assertTrue(entry["enabled"])
        self.assertEqual(entry["params"]["lag_seconds"], [3, 6])
        self.assertEqual(entry["params"]["features"], ["ltp", "spot"])
        self.assertEqual(entry["params"]["partition_by"], ["trading_day", "token"])
        self.assertEqual(entry["params"]["sample_interval_sec"], 3)
        self.assertEqual(result.metadata_block["sample_interval_sec"], 3)
        self.assertEqual(result.sample_interval_sec, 3)

    def test_lag_fails_fast_on_missing_feature(self) -> None:
        df = pd.DataFrame({"token": ["A", "A"], "ltp": [1.0, 2.0]})
        cfg = {
            "transformations": [{
                "id": "lag",
                "enabled": True,
                "params": {
                    "features": ["weighted_gamma"],
                    "lag_seconds": [3],
                    "partition_by": ["token"],
                },
            }]
        }
        ctx = TransformContext(config=cfg, sample_interval_sec=3.0)
        with self.assertRaises(LagConfigError) as err:
            run_transformation_pipeline(df, cfg, context=ctx)
        msg = str(err.exception)
        self.assertIn("Feature not found", msg)
        self.assertIn("weighted_gamma", msg)

    def test_lag_rejects_legacy_row_offsets(self) -> None:
        df = pd.DataFrame({"token": ["A", "A"], "ltp": [1.0, 2.0]})
        cfg = {
            "transformations": [{
                "id": "lag",
                "enabled": True,
                "params": {"features": ["ltp"], "lags": [1], "partition_by": ["token"]},
            }]
        }
        ctx = TransformContext(config=cfg, sample_interval_sec=3.0)
        with self.assertRaises(LagConfigError) as err:
            run_transformation_pipeline(df, cfg, context=ctx)
        self.assertIn("lag_seconds", str(err.exception))

    def test_lag_log_includes_created_columns(self) -> None:
        df = pd.DataFrame({
            "token": ["A", "A", "A"],
            "ltp": [1.0, 2.0, 3.0],
        })
        cfg = {
            "transformations": [{
                "id": "lag",
                "enabled": True,
                "params": {
                    "features": ["ltp"],
                    "lag_seconds": [3],
                    "partition_by": ["token"],
                },
            }]
        }
        lines: list[str] = []
        ctx = TransformContext(config=cfg, sample_interval_sec=3.0, logger=lines.append)
        run_transformation_pipeline(df, cfg, context=ctx, log_fn=lines.append)
        joined = "\n".join(lines)
        self.assertIn("Created Columns : 1", joined)
        self.assertIn("Lag", joined)

    def test_normalize_fills_version(self) -> None:
        cfg = normalize_transformation_config({"transformations": []})
        self.assertEqual(cfg["transformation_pipeline_version"], TRANSFORMATION_PIPELINE_VERSION)


if __name__ == "__main__":
    unittest.main()

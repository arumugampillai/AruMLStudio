"""Schema validation — missing columns fail clearly before consumer math."""

from __future__ import annotations

import os
import tempfile
import unittest

import pandas as pd

from chain_replay_ml.dataset_engine import query_dataset
from chain_replay_ml.dataset_engine.planner import plan_query
from chain_replay_ml.tests.dataset_engine._fixtures import (
    DemoParquetCase,
    HAS_PYARROW,
    require_duckdb,
)


def require_pyarrow() -> None:
    if not HAS_PYARROW:
        raise unittest.SkipTest("pyarrow not installed")


class TestSchemaValidation(DemoParquetCase):
    def setUp(self) -> None:
        require_pyarrow()

    def test_missing_requested_column_raises_keyerror(self) -> None:
        with self.assertRaises(KeyError) as ctx:
            query_dataset(
                self.path,
                columns=["trading_day", "not_a_real_column"],
            )
        msg = str(ctx.exception)
        self.assertIn("missing columns", msg)
        self.assertIn("not_a_real_column", msg)

    def test_filter_on_missing_column_raises_keyerror(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "no_ltp.parquet")
            pd.DataFrame(
                {"trading_day": ["2026-07-23"], "token": ["A"]}
            ).to_parquet(path, index=False)
            with self.assertRaises(KeyError) as ctx:
                query_dataset(path, filters={"ltp_min": 15})
            self.assertIn("ltp", str(ctx.exception))

    def test_schema_version_noted_on_plan(self) -> None:
        plan = plan_query(
            self.path,
            columns=["trading_day"],
            schema_version="analysis-v1",
        )
        self.assertEqual(plan.schema_version, "analysis-v1")

    def test_schema_version_recorded_in_stats_when_query_runs(self) -> None:
        require_duckdb()
        result = query_dataset(
            self.path,
            columns=["trading_day", "token"],
            schema_version="analysis-v1",
        )
        notes = result.stats.extra.get("plan_notes") or []
        self.assertTrue(any("schema_version requested=analysis-v1" in n for n in notes))


if __name__ == "__main__":
    unittest.main()

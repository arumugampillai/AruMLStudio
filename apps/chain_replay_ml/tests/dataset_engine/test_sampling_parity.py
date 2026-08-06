"""Sampling behaviour for Phase-1 Dataset Engine (deterministic LIMIT)."""

from __future__ import annotations

import unittest

import pandas as pd

from chain_replay_ml.dataset_engine import SampleSpec, query_dataset
from chain_replay_ml.tests.dataset_engine._fixtures import (
    DemoParquetCase,
    pandas_apply_filters,
    require_duckdb,
)


class TestSamplingParity(DemoParquetCase):
    """Phase-1 sampling is LIMIT after filters — not random ``df.sample``.

    Parity means: Engine returns at most ``max_rows`` of the filtered set,
    matching ``pandas_apply_filters(...).head(max_rows)`` for this stand-in.
    """

    def setUp(self) -> None:
        require_duckdb()

    def test_limit_after_filter_matches_pandas_head(self) -> None:
        filters = {"ltp_min": 15, "ltp_max": 100}
        max_rows = 2
        src = pd.read_parquet(self.path)
        expected = pandas_apply_filters(src, filters).head(max_rows)

        result = query_dataset(
            self.path,
            filters=filters,
            sample=SampleSpec(max_rows=max_rows, seed=42),
        )
        got = result.table.to_pandas()
        self.assertEqual(result.stats.rows_returned, max_rows)
        self.assertEqual(result.stats.extra.get("sample_mode"), "limit")
        # DuckDB LIMIT without ORDER BY is implementation-defined order;
        # assert size + membership in the filtered set, not exact head order.
        filtered = pandas_apply_filters(src, filters)
        self.assertEqual(len(got), max_rows)
        merged = got.merge(filtered, on=list(got.columns), how="left", indicator=True)
        self.assertTrue((merged["_merge"] == "both").all())

    def test_limit_larger_than_filtered_returns_all(self) -> None:
        filters = {"trading_day": "2026-07-23", "ltp_min": 100}
        src = pd.read_parquet(self.path)
        expected_n = len(pandas_apply_filters(src, filters))
        result = query_dataset(
            self.path,
            filters=filters,
            sample=SampleSpec(max_rows=10_000),
        )
        self.assertEqual(result.stats.rows_returned, expected_n)


if __name__ == "__main__":
    unittest.main()

"""Filter parity: pandas reference path vs Dataset Engine must return identical rows."""

from __future__ import annotations

import unittest

import pandas as pd

from chain_replay_ml.dataset_engine import query_dataset
from chain_replay_ml.tests.dataset_engine._fixtures import (
    DemoParquetCase,
    frames_equal_sorted,
    pandas_apply_filters,
    require_duckdb,
)

_KEY = ["trading_day", "token"]


class TestFilterParity(DemoParquetCase):
    def setUp(self) -> None:
        require_duckdb()

    def _assert_parity(self, filters: dict) -> None:
        src = pd.read_parquet(self.path)
        expected = pandas_apply_filters(src, filters)
        result = query_dataset(self.path, filters=filters)
        got = result.table.to_pandas()
        self.assertEqual(result.stats.rows_returned, len(expected))
        frames_equal_sorted(expected, got, key=_KEY)

    def test_premium_range(self) -> None:
        self._assert_parity({"ltp_min": 15, "ltp_max": 100})

    def test_premium_aliases(self) -> None:
        self._assert_parity({"premium_min": 15, "premium_max": 100})

    def test_trading_day(self) -> None:
        self._assert_parity({"trading_day": "2026-07-23"})

    def test_trading_days(self) -> None:
        self._assert_parity({"trading_days": ["2026-07-23", "2026-07-24"]})

    def test_dte_and_atm(self) -> None:
        self._assert_parity(
            {
                "ltp_min": 15,
                "ltp_max": 100,
                "dte_max": 2,
                "atm_distance_max": 5,
            }
        )

    def test_column_prune_with_filters(self) -> None:
        filters = {"ltp_min": 15, "ltp_max": 100}
        cols = ["trading_day", "token", "ltp"]
        src = pd.read_parquet(self.path, columns=cols)
        expected = pandas_apply_filters(src, filters)[cols]
        result = query_dataset(self.path, columns=cols, filters=filters)
        got = result.table.to_pandas()
        self.assertEqual(list(got.columns), cols)
        frames_equal_sorted(expected, got, key=_KEY)


if __name__ == "__main__":
    unittest.main()

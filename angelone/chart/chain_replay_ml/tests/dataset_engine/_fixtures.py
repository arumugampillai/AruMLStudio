"""Shared fixtures for Dataset Engine integration tests."""

from __future__ import annotations

import os
import tempfile
import unittest
from typing import Any

import pandas as pd

try:
    import duckdb  # noqa: F401

    HAS_DUCKDB = True
except ImportError:
    HAS_DUCKDB = False

try:
    import pyarrow  # noqa: F401

    HAS_PYARROW = True
except ImportError:
    HAS_PYARROW = False


def require_duckdb() -> None:
    if not HAS_DUCKDB:
        raise unittest.SkipTest("duckdb not installed")
    if not HAS_PYARROW:
        raise unittest.SkipTest("pyarrow not installed")


def write_demo_parquet(path: str) -> str:
    """Small multi-day Analysis-like frame for parity tests."""
    rows = [
        # day 23
        {"trading_day": "2026-07-23", "token": "A", "ltp": 10.0, "days_to_expiry": 5, "atm_distance": 0},
        {"trading_day": "2026-07-23", "token": "B", "ltp": 25.0, "days_to_expiry": 2, "atm_distance": 1},
        {"trading_day": "2026-07-23", "token": "C", "ltp": 80.0, "days_to_expiry": 1, "atm_distance": 3},
        {"trading_day": "2026-07-23", "token": "D", "ltp": 150.0, "days_to_expiry": 0, "atm_distance": 8},
        # day 24
        {"trading_day": "2026-07-24", "token": "A", "ltp": 20.0, "days_to_expiry": 4, "atm_distance": 0},
        {"trading_day": "2026-07-24", "token": "B", "ltp": 55.0, "days_to_expiry": 2, "atm_distance": 2},
        {"trading_day": "2026-07-24", "token": "C", "ltp": 99.0, "days_to_expiry": 1, "atm_distance": 5},
        {"trading_day": "2026-07-24", "token": "D", "ltp": 12.0, "days_to_expiry": 3, "atm_distance": 10},
    ]
    df = pd.DataFrame(rows)
    df.to_parquet(path, index=False)
    return path


class DemoParquetCase(unittest.TestCase):
    """Base case with a temp parquet file."""

    path: str
    _tmpdir: tempfile.TemporaryDirectory[str]

    @classmethod
    def setUpClass(cls) -> None:
        cls._tmpdir = tempfile.TemporaryDirectory()
        cls.path = os.path.join(cls._tmpdir.name, "demo.parquet")
        write_demo_parquet(cls.path)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmpdir.cleanup()


def pandas_apply_filters(df: pd.DataFrame, filters: dict[str, Any]) -> pd.DataFrame:
    """Reference filter semantics — must match DuckDB `_filters_to_sql`."""
    out = df
    if "trading_days" in filters and filters["trading_days"] is not None:
        days = [str(d) for d in filters["trading_days"]]
        out = out[out["trading_day"].astype(str).isin(days)]
    elif filters.get("trading_day") is not None:
        out = out[out["trading_day"].astype(str) == str(filters["trading_day"])]

    lo = filters.get("ltp_min", filters.get("premium_min"))
    hi = filters.get("ltp_max", filters.get("premium_max"))
    if lo is not None:
        out = out[out["ltp"] >= float(lo)]
    if hi is not None:
        out = out[out["ltp"] <= float(hi)]

    dte_max = filters.get("dte_max", filters.get("days_to_expiry_max"))
    if dte_max is not None:
        out = out[out["days_to_expiry"] <= float(dte_max)]

    atm_max = filters.get("atm_distance_max")
    if atm_max is not None:
        out = out[out["atm_distance"] <= float(atm_max)]

    return out.reset_index(drop=True)


def frames_equal_sorted(a: pd.DataFrame, b: pd.DataFrame, *, key: list[str]) -> None:
    """Assert two frames contain the same rows (order-independent)."""
    left = a.sort_values(key).reset_index(drop=True)
    right = b.sort_values(key).reset_index(drop=True)
    pd.testing.assert_frame_equal(left, right, check_dtype=False)

"""Sanity checks for ORMP builds."""

from __future__ import annotations

import math
import sqlite3
from typing import Any

from .feature_export import FEATURE_COLUMNS, PRICE_COLUMNS
from .profile_engine import OrmpProfile


def assert_band_mapping_examples() -> None:
    """Spec examples: spot_open=24000, band_size=0.05% → 12 pts."""
    p = OrmpProfile.create(24000.0, 0.05)
    cases = [
        (23976.0, -2),
        (23988.0, -1),
        (24000.0, 0),
        (24006.0, 0),
        (24011.99, 0),
        (24012.0, 1),
        (24018.0, 1),
        (24023.99, 1),
        (24024.0, 2),
        (24036.0, 3),
        (24000.0 - 6.0, -1),  # floor(-0.5) → -1, not 0
    ]
    for price, expected in cases:
        got = p.band_index_for_price(price)
        if got != expected:
            raise AssertionError(f"price={price}: expected band {expected}, got {got}")


def summarize_dataset(db_path: str) -> dict[str, Any]:
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT COUNT(*) AS n FROM ormp_samples").fetchone()["n"]
        days = conn.execute("SELECT COUNT(*) AS n FROM ormp_day_summary").fetchone()["n"]
        fail = conn.execute(
            "SELECT COUNT(*) AS n FROM ormp_day_summary WHERE validation_ok = 0"
        ).fetchone()["n"]
        day_range = conn.execute(
            "SELECT MIN(trading_day) AS a, MAX(trading_day) AS b FROM ormp_day_summary"
        ).fetchone()
        ratio_bad = conn.execute(
            """
            SELECT COUNT(*) AS n FROM ormp_samples
            WHERE ormp_time_above_ratio < 0 OR ormp_time_above_ratio > 1
               OR ormp_time_below_ratio < 0 OR ormp_time_below_ratio > 1
            """
        ).fetchone()["n"]
        # Spot-check feature columns exist
        info = {r[1] for r in conn.execute("PRAGMA table_info(ormp_samples)").fetchall()}
        missing = [c for c in FEATURE_COLUMNS if c not in info]
        missing_prices = [c for c in PRICE_COLUMNS if c not in info]
        price_coverage = {}
        for col in PRICE_COLUMNS:
            if col in info and col != "spot_ltp":
                n_nonnull = conn.execute(
                    f'SELECT COUNT(*) FROM ormp_samples WHERE "{col}" IS NOT NULL'
                ).fetchone()[0]
                price_coverage[col] = {
                    "non_null": int(n_nonnull),
                    "null": int(rows) - int(n_nonnull),
                }
    return {
        "db_path": db_path,
        "sample_rows": int(rows),
        "days": int(days),
        "validation_failures": int(fail),
        "from_date": day_range["a"],
        "to_date": day_range["b"],
        "ratio_out_of_range": int(ratio_bad),
        "missing_feature_columns": missing,
        "missing_price_columns": missing_prices,
        "future_ltp_coverage": price_coverage,
        "ok": fail == 0 and ratio_bad == 0 and not missing and not missing_prices,
    }


def run_unit_sanity() -> dict[str, Any]:
    assert_band_mapping_examples()

    # Synthetic day: stay in 0, jump to +1, back to 0
    p = OrmpProfile.create(24000.0, 0.05)
    prices = [24000.0, 24005.0, 24012.0, 24012.0, 24000.0]
    for i, px in enumerate(prices):
        p.update(px, 1_000_000.0 + i * 60.0, duration_sec=60)
    val = p.validate_time_accounting()
    if not val["ok"]:
        raise AssertionError(f"time accounting failed: {val}")
    if p.current_band != 0:
        raise AssertionError("expected back at band 0")
    if p.return_to_open_count != 1:
        raise AssertionError(f"expected 1 return-to-open, got {p.return_to_open_count}")
    if p.total_band_transitions != 2:
        raise AssertionError(f"expected 2 transitions, got {p.total_band_transitions}")
    if abs(p.band_size_points - 12.0) > 1e-9:
        raise AssertionError(f"band_size_points={p.band_size_points}")
    # Negatives: truncation trap
    if int(-0.5) == 0 and math.floor(-0.5) == -1:
        pass
    return {"ok": True, "synthetic_validation": val}

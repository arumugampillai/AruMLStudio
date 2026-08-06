"""Precomputed filter distributions for master dataset (premium / ATM / delta)."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

DIST_PREMIUM = "PREMIUM"
DIST_ATM = "ATM"
DIST_DELTA = "DELTA"

# (label, min inclusive, max exclusive) — last bucket max=None means open-ended
PREMIUM_BUCKETS: tuple[tuple[str, float, float | None], ...] = (
    ("0-10", 0.0, 10.0),
    ("10-20", 10.0, 20.0),
    ("20-50", 20.0, 50.0),
    ("50-100", 50.0, 100.0),
    ("100+", 100.0, None),
)

DELTA_BUCKETS: tuple[tuple[str, float, float | None], ...] = (
    ("0.00-0.10", 0.0, 0.10),
    ("0.10-0.20", 0.10, 0.20),
    ("0.20-0.30", 0.20, 0.30),
    ("0.30-0.40", 0.30, 0.40),
    ("0.40-0.50", 0.40, 0.50),
    ("0.50+", 0.50, None),
)

MAX_ATM_BUCKET = 10


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sample_columns(conn: Any) -> set[str]:
    return {str(row[1]) for row in conn.execute("PRAGMA table_info(samples)").fetchall()}


def _premium_case_sql() -> str:
    parts: list[str] = []
    for label, lo, hi in PREMIUM_BUCKETS:
        if hi is None:
            parts.append(f"WHEN ltp >= {lo} THEN '{label}'")
        else:
            parts.append(f"WHEN ltp >= {lo} AND ltp < {hi} THEN '{label}'")
    return f"CASE {' '.join(parts)} ELSE NULL END"


def _delta_expr(sample_cols: set[str]) -> str | None:
    if "abs_delta" in sample_cols:
        return "abs_delta"
    if "delta" in sample_cols:
        return 'ABS("delta")'
    return None


def _delta_case_sql(delta_expr: str) -> str:
    parts: list[str] = []
    for label, lo, hi in DELTA_BUCKETS:
        if hi is None:
            parts.append(f"WHEN {delta_expr} >= {lo} THEN '{label}'")
        else:
            parts.append(f"WHEN {delta_expr} >= {lo} AND {delta_expr} < {hi} THEN '{label}'")
    return f"CASE {' '.join(parts)} ELSE NULL END"


def aggregate_day_distributions(
    conn: Any,
    trading_day: str,
    *,
    sample_cols: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Return per-bucket row/token counts for one trading day."""
    cols = sample_cols or _sample_columns(conn)
    td = str(trading_day)
    out: list[dict[str, Any]] = []

    if "ltp" in cols:
        prem_case = _premium_case_sql()
        rows = conn.execute(
            f"""
            SELECT bucket, COUNT(*) AS rows, COUNT(DISTINCT token) AS tokens
            FROM (
                SELECT token, {prem_case} AS bucket
                FROM samples
                WHERE trading_day = ? AND ltp IS NOT NULL
            )
            WHERE bucket IS NOT NULL
            GROUP BY bucket
            """,
            (td,),
        ).fetchall()
        for bucket, row_count, token_count in rows:
            out.append({
                "distribution_type": DIST_PREMIUM,
                "bucket": str(bucket),
                "rows": int(row_count or 0),
                "tokens": int(token_count or 0),
            })

    if "strike_distance_from_atm" in cols:
        rows = conn.execute(
            """
            SELECT CAST(ABS(strike_distance_from_atm) AS INTEGER) AS dist,
                   COUNT(*) AS rows,
                   COUNT(DISTINCT token) AS tokens
            FROM samples
            WHERE trading_day = ?
              AND strike_distance_from_atm IS NOT NULL
              AND ABS(strike_distance_from_atm) <= ?
            GROUP BY dist
            """,
            (td, MAX_ATM_BUCKET),
        ).fetchall()
        for dist, row_count, token_count in rows:
            out.append({
                "distribution_type": DIST_ATM,
                "bucket": str(int(dist)),
                "rows": int(row_count or 0),
                "tokens": int(token_count or 0),
            })

    delta_expr = _delta_expr(cols)
    if delta_expr:
        delta_case = _delta_case_sql(delta_expr)
        rows = conn.execute(
            f"""
            SELECT bucket, COUNT(*) AS rows, COUNT(DISTINCT token) AS tokens
            FROM (
                SELECT token, {delta_case} AS bucket
                FROM samples
                WHERE trading_day = ?
            )
            WHERE bucket IS NOT NULL
            GROUP BY bucket
            """,
            (td,),
        ).fetchall()
        for bucket, row_count, token_count in rows:
            out.append({
                "distribution_type": DIST_DELTA,
                "bucket": str(bucket),
                "rows": int(row_count or 0),
                "tokens": int(token_count or 0),
            })

    return out


def rebuild_all_distributions(conn: Any) -> None:
    """Full recompute from samples — used by refresh-metadata only."""
    conn.execute("DELETE FROM master_dataset_distribution")
    sample_cols = _sample_columns(conn)
    now = _utc_now()
    if "ltp" in sample_cols:
        prem_case = _premium_case_sql()
        rows = conn.execute(
            f"""
            SELECT bucket, COUNT(*) AS rows, COUNT(DISTINCT token) AS tokens
            FROM (
                SELECT token, {prem_case} AS bucket
                FROM samples
                WHERE ltp IS NOT NULL
            )
            WHERE bucket IS NOT NULL
            GROUP BY bucket
            """
        ).fetchall()
        for bucket, row_count, token_count in rows:
            conn.execute(
                """
                INSERT INTO master_dataset_distribution (
                    distribution_type, bucket, rows, tokens, updated_at
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (DIST_PREMIUM, str(bucket), int(row_count or 0), int(token_count or 0), now),
            )

    if "strike_distance_from_atm" in sample_cols:
        rows = conn.execute(
            """
            SELECT CAST(ABS(strike_distance_from_atm) AS INTEGER) AS dist,
                   COUNT(*) AS rows,
                   COUNT(DISTINCT token) AS tokens
            FROM samples
            WHERE strike_distance_from_atm IS NOT NULL
              AND ABS(strike_distance_from_atm) <= ?
            GROUP BY dist
            """,
            (MAX_ATM_BUCKET,),
        ).fetchall()
        for dist, row_count, token_count in rows:
            conn.execute(
                """
                INSERT INTO master_dataset_distribution (
                    distribution_type, bucket, rows, tokens, updated_at
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (DIST_ATM, str(int(dist)), int(row_count or 0), int(token_count or 0), now),
            )

    delta_expr = _delta_expr(sample_cols)
    if delta_expr:
        delta_case = _delta_case_sql(delta_expr)
        rows = conn.execute(
            f"""
            SELECT bucket, COUNT(*) AS rows, COUNT(DISTINCT token) AS tokens
            FROM (
                SELECT token, {delta_case} AS bucket
                FROM samples
            )
            WHERE bucket IS NOT NULL
            GROUP BY bucket
            """
        ).fetchall()
        for bucket, row_count, token_count in rows:
            conn.execute(
                """
                INSERT INTO master_dataset_distribution (
                    distribution_type, bucket, rows, tokens, updated_at
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (DIST_DELTA, str(bucket), int(row_count or 0), int(token_count or 0), now),
            )


def apply_day_distribution_delta(
    conn: Any,
    trading_day: str,
    *,
    sign: int,
    sample_cols: set[str] | None = None,
) -> None:
    """Add (sign=1) or subtract (sign=-1) one day's bucket counts."""
    if sign not in (1, -1):
        return
    buckets = aggregate_day_distributions(conn, trading_day, sample_cols=sample_cols)
    now = _utc_now()
    for item in buckets:
        dist_type = item["distribution_type"]
        bucket = item["bucket"]
        rows = int(item["rows"])
        tokens = int(item.get("tokens") or 0)
        if sign == 1:
            conn.execute(
                """
                INSERT INTO master_dataset_distribution (
                    distribution_type, bucket, rows, tokens, updated_at
                ) VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(distribution_type, bucket) DO UPDATE SET
                    rows = rows + excluded.rows,
                    tokens = COALESCE(tokens, 0) + excluded.tokens,
                    updated_at = excluded.updated_at
                """,
                (dist_type, bucket, rows, tokens, now),
            )
        else:
            conn.execute(
                """
                UPDATE master_dataset_distribution SET
                    rows = MAX(0, rows - ?),
                    tokens = CASE
                        WHEN tokens IS NULL THEN NULL
                        ELSE MAX(0, tokens - ?)
                    END,
                    updated_at = ?
                WHERE distribution_type = ? AND bucket = ?
                """,
                (rows, tokens, now, dist_type, bucket),
            )

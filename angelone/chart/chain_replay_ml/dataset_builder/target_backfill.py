"""Backfill prediction target columns when horizons expand on an existing master DB."""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any, Callable

from storage.chain_replay_export import ChainReplayError

from .day_context import DayContext, load_day_context
from .feature_migration_engine import (
    _day_source_from_samples,
    _read_day_samples,
    make_sample_id,
)
from .master_store import MasterStore, _SAFE_COL, _sql_type

_CHART_DIR = __import__("os").path.dirname(
    __import__("os").path.dirname(__import__("os").path.dirname(__import__("os").path.abspath(__file__)))
)

BACKFILL_TEMP_TABLE = "master_target_backfill_temp"
CancelFn = Callable[[], bool]
# (message, current, total, unit) — unit is "days" for the outer day loop and
# "rows" for the inner per-day row loop so callers can label progress correctly.
ProgressFn = Callable[[str, int, int, str], None]


class TargetBackfillError(Exception):
    pass


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def horizon_sec_from_column(col: str) -> int:
    """Inverse of horizon_column_name for future_ltp_* columns."""
    if col == "future_ltp_1m":
        return 60
    if col == "future_ltp_3m":
        return 180
    if col == "future_ltp_5m":
        return 300
    m = re.match(r"future_ltp_(\d+)m$", col)
    if m:
        return int(m.group(1)) * 60
    m = re.match(r"future_ltp_(\d+)s$", col)
    if m:
        return int(m.group(1))
    return 0


def horizons_sec_for_columns(target_columns: list[str]) -> list[int]:
    out: list[int] = []
    for col in target_columns:
        sec = horizon_sec_from_column(col)
        if sec > 0 and sec not in out:
            out.append(sec)
    return sorted(out)


def analyze_target_backfill(store: MasterStore, target_columns: list[str]) -> dict[str, Any]:
    """Detect target columns/days that need backfill after horizon expansion.

    Backfill is scoped **per trading day**: a day is only queued for
    reprocessing when a requested target column has zero populated rows for
    that specific day (i.e. it was never computed for that day). Days that
    already have the column filled (aside from a handful of naturally
    trimmed NULLs near session close / illiquid strikes) are left untouched.

    Previously this scoped backfill to *every* trading day in the master DB
    whenever a single column was missing anywhere, which meant adding one
    new day (or expanding horizons) could silently trigger a full re-scan —
    including a full tick-database reload — of every previously built day,
    even ones that already had the column fully populated.
    """
    build_schema = store.get_meta("build_schema") or {}
    stored_targets = list(build_schema.get("target_columns") or [])

    needs_backfill: list[str] = []
    days_needing_backfill: set[str] = set()

    if store.total_row_count() > 0:
        conn = store.conn
        existing_cols = {row[1] for row in conn.execute("PRAGMA table_info(samples)").fetchall()}
        all_days = store.distinct_trading_days()
        for col in target_columns:
            if not _SAFE_COL.match(col):
                continue
            if col not in existing_cols:
                # Column has never been created — every existing day is missing it.
                if all_days:
                    needs_backfill.append(col)
                    days_needing_backfill.update(all_days)
                continue
            rows = conn.execute(
                f'SELECT trading_day, COUNT(*), COUNT("{col}") FROM samples GROUP BY trading_day'
            ).fetchall()
            col_needs_day = False
            for day, total, filled in rows:
                if int(total) > 0 and int(filled) == 0:
                    days_needing_backfill.add(str(day))
                    col_needs_day = True
            if col_needs_day:
                needs_backfill.append(col)

    needs_backfill = list(dict.fromkeys(needs_backfill))
    trading_days = sorted(days_needing_backfill)

    return {
        "needed": bool(needs_backfill) and bool(trading_days),
        "columns": needs_backfill,
        "stored_target_columns": stored_targets,
        "all_target_columns": list(target_columns),
        "trading_days": trading_days,
    }


def compute_targets_for_row(
    *,
    ts: float,
    opt_tl: Any,
    horizons_sec: list[int],
    max_stale_sec: float,
    columns: list[str] | None = None,
) -> dict[str, Any]:
    """Match stages.py stage 5 via OLE Fixed Horizon (identical freshness + LTP lookup)."""
    from chain_replay_ml.outcome_label_engine.fixed_horizon import (
        compute_fixed_horizon_targets,
    )

    return compute_fixed_horizon_targets(
        ts=ts,
        opt_tl=opt_tl,
        horizons_sec=horizons_sec,
        max_stale_sec=max_stale_sec,
        columns=columns,
    )

def _resolve_opt_tl(ctx: DayContext, row: dict[str, Any]) -> Any:
    token = str(row.get("token") or "")
    for (_strike, _opt_type), (tok, _sym, opt_tl) in ctx.strike_mapping.items():
        if str(tok) == token:
            return opt_tl
    strike_r = float(row.get("strike") or 0)
    opt_type = str(row.get("option_type") or "")
    entry = ctx.strike_mapping.get((strike_r, opt_type))
    return entry[2] if entry else None


def _drop_temp_table(conn) -> None:
    conn.execute(f'DROP TABLE IF EXISTS "{BACKFILL_TEMP_TABLE}"')


def _create_temp_table(conn, columns: list[str]) -> None:
    _drop_temp_table(conn)
    col_defs = ", ".join(f'"{c}" REAL' for c in columns)
    conn.execute(
        f"""
        CREATE TABLE "{BACKFILL_TEMP_TABLE}" (
            sample_id TEXT PRIMARY KEY,
            trading_day TEXT NOT NULL,
            timestamp REAL NOT NULL,
            token TEXT NOT NULL,
            {col_defs}
        )
        """
    )


def _insert_temp_batch(conn, columns: list[str], batch: list[dict[str, Any]]) -> None:
    if not batch:
        return
    cols = ["sample_id", "trading_day", "timestamp", "token", *columns]
    placeholders = ", ".join("?" for _ in cols)
    col_sql = ", ".join(f'"{c}"' for c in cols)
    sql = f'INSERT OR REPLACE INTO "{BACKFILL_TEMP_TABLE}" ({col_sql}) VALUES ({placeholders})'
    params = [
        [
            row["sample_id"],
            row["trading_day"],
            float(row["timestamp"]),
            row["token"],
            *[row.get(c) for c in columns],
        ]
        for row in batch
    ]
    conn.executemany(sql, params)


def _merge_temp_into_samples(store: MasterStore, columns: list[str]) -> None:
    conn = store.conn
    existing = {row[1] for row in conn.execute("PRAGMA table_info(samples)").fetchall()}
    for col in columns:
        if col not in existing:
            if not _SAFE_COL.match(col):
                raise TargetBackfillError(f"Unsafe column: {col}")
            conn.execute(f'ALTER TABLE samples ADD COLUMN "{col}" {_sql_type(col)}')
    for col in columns:
        conn.execute(
            f"""
            UPDATE samples SET "{col}" = (
                SELECT t."{col}" FROM "{BACKFILL_TEMP_TABLE}" t
                WHERE t.trading_day = samples.trading_day
                  AND t.timestamp = samples.timestamp
                  AND t.token = samples.token
            )
            WHERE trading_day IN (
                SELECT DISTINCT trading_day FROM "{BACKFILL_TEMP_TABLE}"
            )
            """
        )
    _drop_temp_table(conn)


def backfill_day_targets(
    store: MasterStore,
    *,
    trading_day: str,
    target_columns: list[str],
    horizons_sec: list[int],
    ctx: DayContext | None = None,
    chart_dir: str | None = None,
    step_sec: int = 10,
    default_market: str = "NIFTY",
    max_stale_sec: float = 10.0,
    on_progress: ProgressFn | None = None,
) -> dict[str, Any]:
    """Recompute new target columns for one trading day and merge into samples."""
    day = str(trading_day).strip()
    conn = store.conn
    sample_rows = _read_day_samples(conn, day)
    if not sample_rows:
        return {"trading_day": day, "rows": 0, "skipped": True}

    owned_ctx = False
    if ctx is None:
        source = _day_source_from_samples(conn, day, default_market)
        if not source:
            raise TargetBackfillError(f"Cannot resolve market/expiry for {day}")
        chart = chart_dir or _CHART_DIR
        # Loading the tick DB for a day can take minutes on its own (option
        # chain + spot ticks for the full session) with no internal progress
        # hooks. Emit a heartbeat immediately before it so the UI timestamp
        # refreshes and doesn't look frozen while this runs.
        if on_progress:
            on_progress(f"{day}: loading tick database…", 0, len(sample_rows), "rows")
        ctx = load_day_context(chart, source, feature_grid_step_sec=step_sec)
        owned_ctx = True

    batch: list[dict[str, Any]] = []
    rows_done = 0
    _create_temp_table(conn, target_columns)

    try:
        for row in sample_rows:
            token = str(row["token"])
            ts = float(row["timestamp"])
            opt_tl = _resolve_opt_tl(ctx, row)
            targets = compute_targets_for_row(
                ts=ts,
                opt_tl=opt_tl,
                horizons_sec=horizons_sec,
                max_stale_sec=max_stale_sec,
                columns=target_columns,
            )
            out_row: dict[str, Any] = {
                "sample_id": make_sample_id(day, ts, token),
                "trading_day": day,
                "timestamp": ts,
                "token": token,
            }
            for col in target_columns:
                out_row[col] = targets.get(col)
            batch.append(out_row)
            rows_done += 1
            if len(batch) >= 500:
                _insert_temp_batch(conn, target_columns, batch)
                batch.clear()
                conn.commit()
                if on_progress:
                    on_progress(day, rows_done, len(sample_rows), "rows")

        if batch:
            _insert_temp_batch(conn, target_columns, batch)
        conn.commit()
        _merge_temp_into_samples(store, target_columns)
        conn.commit()
    except Exception:
        conn.rollback()
        _drop_temp_table(conn)
        conn.commit()
        raise
    finally:
        if owned_ctx:
            del ctx

    return {"trading_day": day, "rows": rows_done, "skipped": False}


def run_target_horizon_backfill(
    store: MasterStore,
    *,
    target_columns: list[str],
    columns_to_backfill: list[str],
    horizons_sec: list[int],
    trading_days: list[str],
    chart_dir: str | None = None,
    step_sec: int = 10,
    default_market: str = "NIFTY",
    max_stale_sec: float = 10.0,
    on_progress: ProgressFn | None = None,
    cancel_check: CancelFn | None = None,
) -> dict[str, Any]:
    """Backfill new target columns for all affected trading days."""
    from .orchestrator import _Cancelled

    store.ensure_columns(target_columns)
    backfill_horizons = horizons_sec_for_columns(columns_to_backfill) or horizons_sec
    days_total = len(trading_days)
    day_results: list[dict[str, Any]] = []
    warnings: list[str] = []
    rows_updated = 0

    for di, td in enumerate(trading_days):
        if cancel_check and cancel_check():
            raise _Cancelled()
        if on_progress:
            on_progress(f"Backfilling targets: {td}", di, days_total, "days")
        try:
            result = backfill_day_targets(
                store,
                trading_day=td,
                target_columns=columns_to_backfill,
                horizons_sec=backfill_horizons,
                chart_dir=chart_dir,
                step_sec=step_sec,
                default_market=default_market,
                max_stale_sec=max_stale_sec,
                on_progress=on_progress,
            )
            day_results.append(result)
            rows_updated += int(result.get("rows") or 0)
        except (ChainReplayError, OSError) as exc:
            msg = f"{td}: skipped — {exc}"
            warnings.append(msg)
            day_results.append({"trading_day": td, "status": "skipped", "error": str(exc)})
        except TargetBackfillError as exc:
            msg = f"{td}: failed — {exc}"
            warnings.append(msg)
            day_results.append({"trading_day": td, "status": "failed", "error": str(exc)})

    return {
        "backfilled": rows_updated > 0,
        "columns": columns_to_backfill,
        "days_total": days_total,
        "days_backfilled": sum(1 for r in day_results if not r.get("skipped")),
        "rows_updated": rows_updated,
        "day_results": day_results,
        "warnings": warnings,
        "completed_at": _utc_now(),
    }


def maybe_backfill_expanded_targets(
    store: MasterStore,
    *,
    target_columns: list[str],
    horizons_sec: list[int],
    chart_dir: str | None = None,
    step_sec: int = 10,
    default_market: str = "NIFTY",
    max_stale_sec: float = 10.0,
    on_progress: ProgressFn | None = None,
    cancel_check: CancelFn | None = None,
) -> dict[str, Any]:
    """Run target backfill for any (column, day) pairs missing computed values."""
    analysis = analyze_target_backfill(store, target_columns)
    if not analysis["needed"]:
        return {"backfilled": False, "needed": False, "columns": []}

    columns_to_backfill = list(analysis["columns"])
    trading_days = list(analysis["trading_days"])
    if not trading_days:
        return {
            "backfilled": False,
            "needed": True,
            "columns": columns_to_backfill,
            "warnings": ["No trading days in master DB"],
        }

    result = run_target_horizon_backfill(
        store,
        target_columns=target_columns,
        columns_to_backfill=columns_to_backfill,
        horizons_sec=horizons_sec,
        trading_days=trading_days,
        chart_dir=chart_dir,
        step_sec=step_sec,
        default_market=default_market,
        max_stale_sec=max_stale_sec,
        on_progress=on_progress,
        cancel_check=cancel_check,
    )

    build_schema = store.get_meta("build_schema") or {}
    build_schema.update({
        "target_columns": list(target_columns),
        "target_count": len(target_columns),
    })
    store.set_meta("build_schema", build_schema)

    result["needed"] = True
    return result

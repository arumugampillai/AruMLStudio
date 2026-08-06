"""Non-Null filter pipeline for Master Dataset export / preview.

Analysis Dataset (Registry → Pipeline) uses a **staged** No-Null:

  1. Registry No-Null on Master columns (same policy as a Master export).
  2. Add Pipeline Features (transforms).
  3. Pipeline No-Null Step 2 only on transform-created columns.

Registry features (``current_iv``, ``vega``, ``option_low``, …) therefore cannot
drive a second row reduction after step 1 — they were already enforced.

Manual / Master exports with transforms still defer a single full-frame No-Null
until after transformations (Lag must see the selected SQL partitions).

Non-Null policy (exact order) — on its input row set:

  Step 1 — Remove empty columns
      Drop every column that is 100% NULL on the filtered set.
      When multiple trading_day values are present, a column is empty if it is
      100% NULL on *any* selected day (day-independent column discovery).
      This keeps multi-day selection additive: A+B+C ≈ sum of per-day results.
      The Nullable Feature List is not consulted here.

  Step 2 — Remove incomplete rows
      After Step 1, keep only rows where every *mandatory* remaining column is
      non-NULL. Mandatory = kept columns minus the central Nullable Feature
      List (:mod:`nullable_features`). When ``step2_columns`` is set (Analysis
      Pipeline stage), only that subset is mandatory. NULLs in listed nullable
      features are allowed and do not discard the row.

Final matrix: zero NULL cells among mandatory kept columns. Nullable listed
features may still contain NULLs.
"""

from __future__ import annotations

import logging
import sqlite3
from typing import Any, Callable, Sequence

from .nullable_features import (
    mandatory_columns_for_step2,
    nullable_columns_present,
    resolve_nullable_features,
)

logger = logging.getLogger(__name__)

StageFn = Callable[[str], None]


def _columns_non_null_sql(column_names: Sequence[str]) -> str:
    names = [str(c).strip() for c in column_names if str(c).strip()]
    if not names:
        return "1=1"
    return " AND ".join(f'"{c}" IS NOT NULL' for c in names)


def _count_rows(
    conn: sqlite3.Connection,
    where_sql: str,
    where_params: list[Any],
) -> int:
    row = conn.execute(
        f"SELECT COUNT(*) FROM samples WHERE {where_sql}",
        list(where_params or []),
    ).fetchone()
    return int(row[0] or 0) if row else 0


def _count_rows_by_day(
    conn: sqlite3.Connection,
    where_sql: str,
    where_params: list[Any],
) -> dict[str, int]:
    """Per trading_day row counts under where_sql (empty if no trading_day col used)."""
    try:
        rows = conn.execute(
            f'SELECT "trading_day", COUNT(*) AS n FROM samples '
            f"WHERE {where_sql} GROUP BY \"trading_day\" ORDER BY 1",
            list(where_params or []),
        ).fetchall()
    except sqlite3.OperationalError:
        return {}
    out: dict[str, int] = {}
    for row in rows:
        day = str(row[0] or "").strip()
        if day:
            out[day] = int(row[1] or 0)
    return out


def _log_day_counts(title: str, by_day: dict[str, int]) -> None:
    logger.info("%s", title)
    if not by_day:
        logger.info("  (no trading_day breakdown)")
        return
    for day, n in by_day.items():
        logger.info("  %s : %s", day, f"{n:,}")
    logger.info("  TOTAL : %s", f"{sum(by_day.values()):,}")


def discover_kept_columns_step1(
    conn: sqlite3.Connection,
    column_names: Sequence[str],
    where_sql: str,
    where_params: list[Any],
) -> tuple[list[str], list[str]]:
    """
    Step 1: columns to keep vs remove.

    Single-day / no trading_day: drop columns that are 100% NULL on the filter.
    Multi-day: drop columns that are 100% NULL on any trading_day in the filter
    (so one day cannot re-activate a column that other selected days lack).
    """
    names = [str(c).strip() for c in column_names if str(c).strip()]
    if not names:
        return [], []

    has_day = "trading_day" in names
    agg_parts = [
        f'MAX(CASE WHEN "{c}" IS NOT NULL THEN 1 ELSE 0 END) AS "a{i}"'
        for i, c in enumerate(names)
    ]
    params = list(where_params or [])

    if has_day:
        sql = (
            f'SELECT "trading_day", {", ".join(agg_parts)} '
            f'FROM samples WHERE {where_sql} GROUP BY "trading_day"'
        )
        day_rows = conn.execute(sql, params).fetchall()
        if not day_rows:
            return [], list(names)
        keep_flags = [True] * len(names)
        for row in day_rows:
            for i in range(len(names)):
                if int(row[i + 1] or 0) == 0:
                    keep_flags[i] = False
        kept = [names[i] for i, ok in enumerate(keep_flags) if ok]
    else:
        sql = f"SELECT {', '.join(agg_parts)} FROM samples WHERE {where_sql}"
        row = conn.execute(sql, params).fetchone()
        if not row:
            return [], list(names)
        kept = [names[i] for i, flag in enumerate(row) if int(flag or 0) > 0]

    dropped = [c for c in names if c not in set(kept)]
    return kept, dropped


def apply_non_null_filter(
    conn: sqlite3.Connection,
    column_names: Sequence[str],
    where_sql: str,
    where_params: list[Any],
    *,
    log: bool = True,
    debug: bool = False,
    on_stage: StageFn | None = None,
) -> dict[str, Any]:
    """
    Run Non-Null Step 1 then Step 2 on the currently filtered dataset.

    Step 1 (column discovery) always runs — required for correct export columns.
    Diagnostic COUNT / GROUP BY / NULL-audit SQL run only when ``debug=True``
    (preview / unit tests). Create Dataset export uses ``debug=False``.

    Returns:
      where_sql, where_params, kept_columns, dropped_columns, report
    """
    import threading
    import time

    names = [str(c).strip() for c in column_names if str(c).strip()]
    params = list(where_params or [])
    cols_before = len(names)

    def _stage(msg: str) -> None:
        if on_stage is not None:
            try:
                on_stage(msg)
            except Exception:
                pass

    rows_before: int | None = None
    by_day_before: dict[str, int] = {}
    if debug:
        _stage("No-Null: counting filtered rows (debug)…")
        rows_before = _count_rows(conn, where_sql, params)
        by_day_before = _count_rows_by_day(conn, where_sql, params) if log else {}

    # —— Step 1: remove empty columns (day-independent when multi-day) ——
    _stage(
        f"No-Null Step 1: scanning {cols_before} columns for empty fields "
        "(large masters can take several minutes; still running)…"
    )
    stop_hb = threading.Event()
    t0 = time.monotonic()

    def _heartbeat() -> None:
        while not stop_hb.wait(15.0):
            elapsed = int(time.monotonic() - t0)
            _stage(
                f"No-Null Step 1 still running… {elapsed}s elapsed "
                f"(scanning {cols_before} columns — please wait)"
            )

    hb: threading.Thread | None = None
    if on_stage is not None:
        hb = threading.Thread(
            target=_heartbeat,
            name="non-null-step1-heartbeat",
            daemon=True,
        )
        hb.start()
    try:
        kept, dropped = discover_kept_columns_step1(conn, names, where_sql, params)
    finally:
        stop_hb.set()
        if hb is not None:
            hb.join(timeout=0.2)

    rows_after_cols = rows_before
    _stage(
        f"No-Null Step 1 done: kept {len(kept)} columns, "
        f"dropped {len(dropped)} empty columns"
    )

    if log and debug:
        logger.info(
            "Remove all-null columns\n"
            "Columns before: %s\n"
            "Columns removed: %s\n"
            "%s",
            cols_before,
            len(dropped),
            (", ".join(dropped) if dropped else "(none)"),
        )
        if by_day_before:
            _log_day_counts("Rows before Non-Null (per day)", by_day_before)

    if not kept:
        empty_where = f"({where_sql}) AND 0"
        report = {
            "rows_before": int(rows_before or 0),
            "columns_before": cols_before,
            "empty_columns_removed": len(dropped),
            "columns_removed": list(dropped),
            "columns_removed_count": len(dropped),
            "rows_after_column_cleanup": int(rows_after_cols or 0),
            "incomplete_rows_removed": int(rows_before or 0),
            "rows_removed": int(rows_before or 0),
            "rows_after": 0,
            "columns_after": 0,
            "remaining_null_cells": 0,
            "null_columns_remaining": [],
            "rows_by_day_before": by_day_before,
            "rows_by_day_after": {d: 0 for d in by_day_before},
            "rows_removed_by_day": dict(by_day_before),
            "ok": False,
            "error": "No columns remain after removing 100% NULL columns.",
            "debug": bool(debug),
        }
        if log and debug:
            logger.info(
                "Non-null row filter\n  all rows removed (no active columns)\n"
                "Final dataset\n  TOTAL : 0\nRemaining NULL cells: 0"
            )
        return {
            "where_sql": empty_where,
            "where_params": params,
            "kept_columns": [],
            "dropped_columns": list(dropped),
            "report": report,
        }

    # —— Step 2: incomplete-row filter via WHERE (no diagnostic scan unless debug) ——
    # Mandatory columns only — Nullable Feature List is ignored for row drops.
    mandatory = mandatory_columns_for_step2(kept)
    nullable_present = nullable_columns_present(kept)
    _stage(
        "No-Null Step 2: applying NOT NULL row filter"
        + (
            f" ({len(nullable_present)} nullable feature(s) ignored)…"
            if nullable_present
            else "…"
        )
    )
    null_sql = _columns_non_null_sql(mandatory)
    step2_where = f"({where_sql}) AND ({null_sql})"

    rows_after: int | None = None
    rows_removed = 0
    by_day_after: dict[str, int] = {}
    removed_by_day: dict[str, int] = {}
    remaining_null = 0
    null_cols: list[str] = []
    ok = True
    error = None

    if debug:
        rows_after = _count_rows(conn, step2_where, params)
        rows_removed = max(int(rows_before or 0) - rows_after, 0)
        by_day_after = _count_rows_by_day(conn, step2_where, params) if log else {}
        removed_by_day = {
            d: max(int(by_day_before.get(d, 0)) - int(by_day_after.get(d, 0)), 0)
            for d in sorted(set(by_day_before) | set(by_day_after))
        }
        # Audit: mandatory columns must be NULL-free; nullable list may still have NULLs.
        if rows_after > 0 and mandatory:
            null_checks = ", ".join(
                f'SUM(CASE WHEN "{c}" IS NULL THEN 1 ELSE 0 END) AS "n{i}"'
                for i, c in enumerate(mandatory)
            )
            null_row = conn.execute(
                f"SELECT {null_checks} FROM samples WHERE {step2_where}",
                params,
            ).fetchone()
            if null_row:
                for i, c in enumerate(mandatory):
                    n = int(null_row[i] or 0)
                    if n > 0:
                        remaining_null += n
                        null_cols.append(c)
        ok = remaining_null == 0
        if not ok:
            preview = ", ".join(null_cols[:20])
            error = (
                "ERROR:\n"
                "Non-Null filter failed.\n"
                f"Remaining NULL cells: {remaining_null}.\n"
                f"Columns still containing NULL: {preview}"
            )
            logger.error(error)

    if log and debug:
        logger.info("Non-null row filter")
        if nullable_present:
            logger.info(
                "  Nullable features ignored in Step 2: %s",
                ", ".join(nullable_present),
            )
        for day, n in removed_by_day.items():
            logger.info("  %s removed: %s", day, f"{n:,}")
        _log_day_counts("Final dataset", by_day_after)
        logger.info(
            "Final rows: %s\nFinal columns: %s\nRemaining NULL cells (mandatory): %s",
            f"{int(rows_after or 0):,}",
            len(kept),
            remaining_null,
        )

    report = {
        "rows_before": int(rows_before or 0),
        "columns_before": cols_before,
        "empty_columns_removed": len(dropped),
        "columns_removed": list(dropped),
        "columns_removed_count": len(dropped),
        "rows_after_column_cleanup": int(rows_after_cols or 0),
        "incomplete_rows_removed": rows_removed,
        "rows_removed": rows_removed,
        "rows_after": int(rows_after or 0),
        "columns_after": len(kept),
        "remaining_null_cells": remaining_null,
        "null_columns_remaining": null_cols[:20],
        "nullable_features_ignored": list(nullable_present),
        "mandatory_columns_step2": list(mandatory),
        "rows_by_day_before": by_day_before,
        "rows_by_day_after": by_day_after,
        "rows_removed_by_day": removed_by_day,
        "ok": ok,
        "error": error,
        "debug": bool(debug),
    }
    _stage(
        f"No-Null ready: {len(kept)} columns kept"
        + (f", {int(rows_after):,} rows after filter" if rows_after is not None else "")
    )
    return {
        "where_sql": step2_where,
        "where_params": params,
        "kept_columns": list(kept),
        "dropped_columns": list(dropped),
        "report": report,
    }


def apply_non_null_filter_frame(
    frame: Any,
    *,
    day_column: str = "trading_day",
    step2_columns: Sequence[str] | None = None,
    transformation_config: dict[str, Any] | None = None,
    on_stage: StageFn | None = None,
) -> dict[str, Any]:
    """Apply the canonical two-step Non-Null policy to a transformed DataFrame.

    This is the post-transformation equivalent of :func:`apply_non_null_filter`.
    Multi-day column discovery deliberately preserves the SQL implementation's
    rule: a column is dropped when it is 100% NULL on any selected day.

    ``step2_columns``: when set, Step 2 only requires those columns (intersected
    with Step 1 kept). Used by Analysis Dataset so Registry features already
    validated pre-transform are not re-checked after Pipeline Features are added.
    Step 1 still considers every column.

    ``transformation_config``: when provided, Explicit Nullable parents propagate
    through the pipeline dependency graph so Inherited Nullable outputs are
    also ignored in Step 2.

    Internals use Polars (P2); return type stays a Pandas ``frame`` for callers.
    """
    import pandas as pd

    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame")

    def _stage(msg: str) -> None:
        if on_stage is not None:
            try:
                on_stage(msg)
            except Exception:
                pass

    try:
        return _apply_non_null_filter_frame_polars(
            frame,
            day_column=day_column,
            step2_columns=step2_columns,
            transformation_config=transformation_config,
            stage=_stage,
        )
    except Exception:
        return _apply_non_null_filter_frame_pandas(
            frame,
            day_column=day_column,
            step2_columns=step2_columns,
            transformation_config=transformation_config,
            stage=_stage,
        )


def _nullable_step2_meta(
    kept: list[str],
    *,
    step2_columns: Sequence[str] | None,
    transformation_config: dict[str, Any] | None,
) -> tuple[list[str], list[str], list[str], list[str]]:
    """Return ``(step2_candidates, mandatory, nullable_explicit, nullable_inherited)``."""
    if step2_columns is not None:
        scope = {str(c).strip() for c in step2_columns if str(c).strip()}
        step2_candidates = [c for c in kept if c in scope]
    else:
        step2_candidates = list(kept)
    resolution = resolve_nullable_features(
        list(kept),
        transformation_config=transformation_config,
    )
    mandatory = mandatory_columns_for_step2(
        step2_candidates,
        nullable=resolution.effective,
    )
    nullable_explicit = resolution.present_explicit(step2_candidates or kept)
    nullable_inherited = resolution.present_inherited(step2_candidates or kept)
    if step2_columns is not None:
        for name in resolution.present_explicit(kept):
            if name not in nullable_explicit:
                nullable_explicit.append(name)
    return step2_candidates, mandatory, nullable_explicit, nullable_inherited


def _non_null_frame_report(
    *,
    rows_before: int,
    names: list[str],
    kept: list[str],
    dropped: list[str],
    filtered: Any,
    mandatory: list[str],
    nullable_explicit: list[str],
    nullable_inherited: list[str],
    step2_scope: str,
    day_column: str,
    frame_before: Any,
    remaining_null: int,
    frame_backend: str,
) -> dict[str, Any]:
    rows_after = int(len(filtered))
    rows_before_by_day: dict[str, int] = {}
    rows_after_by_day: dict[str, int] = {}
    if day_column in getattr(frame_before, "columns", []):
        rows_before_by_day = {
            str(k): int(v)
            for k, v in frame_before.groupby(day_column, dropna=False).size().items()
        }
    if day_column in getattr(filtered, "columns", []):
        rows_after_by_day = {
            str(k): int(v)
            for k, v in filtered.groupby(day_column, dropna=False).size().items()
        }
    all_days = sorted(set(rows_before_by_day) | set(rows_after_by_day))
    removed_by_day = {
        day: max(rows_before_by_day.get(day, 0) - rows_after_by_day.get(day, 0), 0)
        for day in all_days
    }
    nullable_present = list(nullable_explicit) + [
        c for c in nullable_inherited if c not in nullable_explicit
    ]
    return {
        "rows_before": rows_before,
        "columns_before": len(names),
        "empty_columns_removed": len(dropped),
        "columns_removed": list(dropped),
        "columns_removed_count": len(dropped),
        "rows_after_column_cleanup": rows_before,
        "incomplete_rows_removed": max(rows_before - rows_after, 0),
        "rows_removed": max(rows_before - rows_after, 0),
        "rows_after": rows_after,
        "columns_after": len(kept),
        "remaining_null_cells": remaining_null,
        "null_columns_remaining": [],
        "nullable_features_ignored": list(nullable_present),
        "nullable_explicit": list(nullable_explicit),
        "nullable_inherited": list(nullable_inherited),
        "mandatory_columns_step2": list(mandatory),
        "step2_scope": step2_scope,
        "rows_by_day_before": rows_before_by_day,
        "rows_by_day_after": rows_after_by_day,
        "rows_removed_by_day": removed_by_day,
        "ok": bool(kept) and remaining_null == 0,
        "error": None if kept else "No columns remain after removing 100% NULL columns.",
        "debug": True,
        "stage": (
            "pipeline_post_transformation"
            if step2_scope == "pipeline_only"
            else "post_transformation"
        ),
        "frame_backend": frame_backend,
    }


def _apply_non_null_filter_frame_polars(
    frame: Any,
    *,
    day_column: str,
    step2_columns: Sequence[str] | None,
    transformation_config: dict[str, Any] | None,
    stage: StageFn,
) -> dict[str, Any]:
    from chain_replay_ml.frame_backend import (
        arrow_table_to_polars,
        polars_to_pandas,
        require_polars,
    )

    pl = require_polars()
    try:
        import pyarrow as pa

        pl_df = arrow_table_to_polars(pa.Table.from_pandas(frame, preserve_index=False))
    except Exception:
        pl_df = pl.from_pandas(frame)

    rows_before = int(pl_df.height)
    names = [str(c) for c in pl_df.columns]
    step2_scope = "pipeline_only" if step2_columns is not None else "all"
    stage(
        f"No-Null Step 1: scanning {len(names)} transformed columns for empty fields…"
    )

    multi_day = False
    if day_column in names:
        n_days = pl_df.select(pl.col(day_column).drop_nulls().n_unique()).item()
        multi_day = int(n_days or 0) > 1

    if multi_day:
        # Drop a column if it is 100% NULL on *any* trading_day (matches SQL / pandas).
        day_flags = pl_df.group_by(day_column, maintain_order=True).agg(
            [pl.col(c).is_not_null().any().alias(c) for c in names]
        )
        kept = [
            c
            for c in names
            if bool(day_flags.select(pl.col(c).all()).item())
        ]
    else:
        kept = [
            c
            for c in names
            if bool(pl_df.select(pl.col(c).is_not_null().any()).item())
        ]
    kept_set = set(kept)
    dropped = [c for c in names if c not in kept_set]

    stage(
        f"No-Null Step 1 done: kept {len(kept)} columns, "
        f"dropped {len(dropped)} empty columns"
    )

    nullable_explicit: list[str] = []
    nullable_inherited: list[str] = []
    mandatory: list[str] = []
    if not kept:
        filtered_pl = pl.DataFrame()
        remaining_null = 0
    else:
        step2_candidates, mandatory, nullable_explicit, nullable_inherited = (
            _nullable_step2_meta(
                kept,
                step2_columns=step2_columns,
                transformation_config=transformation_config,
            )
        )
        scope_note = (
            f" (Pipeline Features only: {len(step2_candidates)} cols)"
            if step2_columns is not None
            else ""
        )
        ign_bits = []
        if nullable_explicit:
            ign_bits.append(f"{len(nullable_explicit)} explicit")
        if nullable_inherited:
            ign_bits.append(f"{len(nullable_inherited)} inherited")
        ign_note = (
            f" ({', '.join(ign_bits)} nullable ignored)…"
            if ign_bits
            else "…"
        )
        stage(
            "No-Null Step 2: applying NOT NULL row filter after transformations"
            + scope_note
            + ign_note
        )
        subset = pl_df.select(kept)
        if mandatory:
            mask = pl.all_horizontal([pl.col(c).is_not_null() for c in mandatory])
            filtered_pl = subset.filter(mask)
        else:
            filtered_pl = subset
        if filtered_pl.height == 0 or not mandatory:
            remaining_null = 0
        else:
            remaining_null = int(
                filtered_pl.select(
                    pl.sum_horizontal([pl.col(c).null_count() for c in mandatory])
                ).item()
                or 0
            )

    filtered = polars_to_pandas(filtered_pl)
    report = _non_null_frame_report(
        rows_before=rows_before,
        names=names,
        kept=kept,
        dropped=dropped,
        filtered=filtered,
        mandatory=mandatory,
        nullable_explicit=nullable_explicit,
        nullable_inherited=nullable_inherited,
        step2_scope=step2_scope,
        day_column=day_column,
        frame_before=frame,
        remaining_null=remaining_null,
        frame_backend="polars",
    )
    stage(f"No-Null ready: {len(kept)} columns kept, {int(len(filtered)):,} rows after filter")
    return {
        "frame": filtered,
        "kept_columns": kept,
        "dropped_columns": dropped,
        "report": report,
    }


def _apply_non_null_filter_frame_pandas(
    frame: Any,
    *,
    day_column: str,
    step2_columns: Sequence[str] | None,
    transformation_config: dict[str, Any] | None,
    stage: StageFn,
) -> dict[str, Any]:
    """Legacy Pandas implementation (fallback)."""
    import pandas as pd

    rows_before = len(frame)
    names = [str(c) for c in frame.columns]
    step2_scope = "pipeline_only" if step2_columns is not None else "all"
    stage(
        f"No-Null Step 1: scanning {len(names)} transformed columns for empty fields…"
    )

    if day_column in frame.columns and frame[day_column].nunique(dropna=True) > 1:
        day_has_value = frame.groupby(day_column, dropna=False)[names].count().gt(0)
        kept = [c for c in names if bool(day_has_value[c].all())]
    else:
        kept = [c for c in names if bool(frame[c].notna().any())]
    kept_set = set(kept)
    dropped = [c for c in names if c not in kept_set]

    stage(
        f"No-Null Step 1 done: kept {len(kept)} columns, "
        f"dropped {len(dropped)} empty columns"
    )
    nullable_explicit: list[str] = []
    nullable_inherited: list[str] = []
    mandatory: list[str] = []
    if not kept:
        filtered = frame.iloc[0:0, 0:0].copy()
        remaining_null = 0
    else:
        step2_candidates, mandatory, nullable_explicit, nullable_inherited = (
            _nullable_step2_meta(
                kept,
                step2_columns=step2_columns,
                transformation_config=transformation_config,
            )
        )
        scope_note = (
            f" (Pipeline Features only: {len(step2_candidates)} cols)"
            if step2_columns is not None
            else ""
        )
        ign_bits = []
        if nullable_explicit:
            ign_bits.append(f"{len(nullable_explicit)} explicit")
        if nullable_inherited:
            ign_bits.append(f"{len(nullable_inherited)} inherited")
        ign_note = (
            f" ({', '.join(ign_bits)} nullable ignored)…"
            if ign_bits
            else "…"
        )
        stage(
            "No-Null Step 2: applying NOT NULL row filter after transformations"
            + scope_note
            + ign_note
        )
        subset = frame.loc[:, kept]
        if mandatory:
            mask = subset[mandatory].notna().all(axis=1)
            filtered = subset.loc[mask].reset_index(drop=True)
        else:
            filtered = subset.reset_index(drop=True)
        if filtered.empty or not mandatory:
            remaining_null = 0
        else:
            remaining_null = int(filtered[mandatory].isna().sum().sum())

    report = _non_null_frame_report(
        rows_before=rows_before,
        names=names,
        kept=kept,
        dropped=dropped,
        filtered=filtered,
        mandatory=mandatory,
        nullable_explicit=nullable_explicit,
        nullable_inherited=nullable_inherited,
        step2_scope=step2_scope,
        day_column=day_column,
        frame_before=frame,
        remaining_null=remaining_null,
        frame_backend="pandas",
    )
    stage(f"No-Null ready: {len(kept)} columns kept, {int(len(filtered)):,} rows after filter")
    return {
        "frame": filtered,
        "kept_columns": kept,
        "dropped_columns": dropped,
        "report": report,
    }


def trace_filter_pipeline(
    conn: sqlite3.Connection,
    column_names: Sequence[str],
    *,
    selected_days: Sequence[str] | None = None,
    trading_day: str | None = None,
    all_days: bool = False,
    atm_band_filter: int | None = None,
    premium_min: float | None = None,
    premium_max: float | None = None,
    delta_min: float | None = None,
    delta_max: float | None = None,
    no_null_data: bool = True,
    log: bool = True,
) -> dict[str, Any]:
    """
    Trace Master → ATM → LTP → Non-Null with per-day counts.

    Identifies which stage collapses multi-day selection to 0 rows.
    """
    from .master_status import _sample_filter_where

    cols = [str(c).strip() for c in column_names if str(c).strip()]
    days = [str(d).strip() for d in (selected_days or []) if str(d).strip()]
    td = str(trading_day or "").strip() or None

    stages: dict[str, Any] = {}

    def _stage(name: str, where: str, params: list[Any]) -> dict[str, int]:
        by_day = _count_rows_by_day(conn, where, params)
        total = _count_rows(conn, where, params)
        stages[name] = {"by_day": by_day, "total": total}
        if log:
            _log_day_counts(name, by_day if by_day else {"(all)": total})
        return by_day

    where_base, p_base = _sample_filter_where(
        trading_day=td,
        selected_days=days or None,
        all_days=all_days,
        column_names=cols,
    )
    _stage("Master DB load", where_base, p_base)

    where_atm, p_atm = _sample_filter_where(
        trading_day=td,
        selected_days=days or None,
        all_days=all_days,
        atm_band_filter=atm_band_filter,
        column_names=cols,
    )
    _stage("ATM filter", where_atm, p_atm)

    where_ltp, p_ltp = _sample_filter_where(
        trading_day=td,
        selected_days=days or None,
        all_days=all_days,
        atm_band_filter=atm_band_filter,
        premium_min=premium_min,
        premium_max=premium_max,
        delta_min=delta_min,
        delta_max=delta_max,
        column_names=cols,
    )
    _stage("LTP / Delta filter", where_ltp, p_ltp)

    nn: dict[str, Any] | None = None
    if no_null_data:
        nn = apply_non_null_filter(conn, cols, where_ltp, p_ltp, log=log, debug=True)
        stages["Non-Null"] = {
            "by_day": (nn.get("report") or {}).get("rows_by_day_after") or {},
            "total": int((nn.get("report") or {}).get("rows_after") or 0),
            "dropped_columns": list(nn.get("dropped_columns") or []),
            "report": nn.get("report"),
        }

    return {"stages": stages, "non_null": nn}


def format_non_null_report(report: dict[str, Any]) -> str:
    """Human-readable Non-Null Filter summary for UI / logs."""
    if not report:
        return ""
    if str(report.get("stage") or "") == "analysis_staged":
        lines = [
            "Non-Null Filter (Analysis — staged)",
            "Registry Features checked before Pipeline Features.",
            "Second row reduction (if any) is from Pipeline Features only "
            "(e.g. Lag/Diff warmup NULLs) — not Registry.",
        ]
        reg = report.get("registry") or {}
        pipe = report.get("pipeline") or {}
        if reg:
            lines.append("")
            lines.append("--- Registry ---")
            lines.append(format_non_null_report(dict(reg)))
        if pipe:
            lines.append("")
            lines.append("--- Pipeline ---")
            lines.append(format_non_null_report(dict(pipe)))
        return "\n".join(lines)
    debug = bool(report.get("debug"))
    lines = ["Non-Null Filter"]
    if report.get("step2_scope") == "pipeline_only":
        lines.append("Step 2 scope: Pipeline Features only")
    if not debug:
        # Create Dataset uses debug=False — row COUNTs are skipped; do not show fake zeros.
        lines.extend(
            [
                "(row counts omitted — fast export mode)",
                f"Columns before: {int(report.get('columns_before') or 0)}",
                f"Removed empty columns: "
                f"{int(report.get('empty_columns_removed') or report.get('columns_removed_count') or 0)}",
            ]
        )
        dropped = report.get("columns_removed") or []
        if dropped:
            lines.append("Columns removed: " + ", ".join(str(c) for c in dropped[:40]))
            if len(dropped) > 40:
                lines.append(f"  … and {len(dropped) - 40} more")
        lines.extend(
            [
                f"Kept columns: {int(report.get('columns_after') or 0)}",
                "Incomplete rows are excluded during Parquet write.",
            ]
        )
        ignored = report.get("nullable_features_ignored") or []
        if ignored:
            lines.append(
                "Nullable features ignored in Step 2: " + ", ".join(str(c) for c in ignored)
            )
        explicit = report.get("nullable_explicit") or []
        inherited = report.get("nullable_inherited") or []
        if explicit or inherited:
            if explicit:
                lines.append(
                    "  Explicit (Registry): " + ", ".join(str(c) for c in explicit)
                )
            if inherited:
                lines.append(
                    "  Inherited (Pipeline): "
                    + ", ".join(str(c) for c in inherited[:40])
                    + (
                        f" … +{len(inherited) - 40} more"
                        if len(inherited) > 40
                        else ""
                    )
                )
        if report.get("error"):
            lines.extend(["", str(report["error"])])
        return "\n".join(lines)

    lines.extend(
        [
            f"Rows before: {int(report.get('rows_before') or 0):,}",
            f"Columns before: {int(report.get('columns_before') or 0)}",
            "",
            f"Removed empty columns: "
            f"{int(report.get('empty_columns_removed') or report.get('columns_removed_count') or 0)}",
        ]
    )
    dropped = report.get("columns_removed") or []
    if dropped:
        lines.append("Columns removed: " + ", ".join(str(c) for c in dropped[:40]))
        if len(dropped) > 40:
            lines.append(f"  … and {len(dropped) - 40} more")
    lines.extend(
        [
            "",
            f"Rows after column cleanup: {int(report.get('rows_after_column_cleanup') or 0):,}",
            "",
            f"Removed rows with NULL: "
            f"{int(report.get('incomplete_rows_removed') or report.get('rows_removed') or 0):,}",
        ]
    )
    removed_by_day = report.get("rows_removed_by_day") or {}
    for day, n in removed_by_day.items():
        lines.append(f"  {day} removed: {int(n):,}")
    lines.extend(
        [
            "",
            f"Final rows: {int(report.get('rows_after') or 0):,}",
            f"Final columns: {int(report.get('columns_after') or 0)}",
        ]
    )
    ignored = report.get("nullable_features_ignored") or []
    if ignored:
        lines.append(
            "Nullable features ignored in Step 2: " + ", ".join(str(c) for c in ignored)
        )
    explicit = report.get("nullable_explicit") or []
    inherited = report.get("nullable_inherited") or []
    if explicit or inherited:
        if explicit:
            lines.append(
                "  Explicit (Registry): " + ", ".join(str(c) for c in explicit)
            )
        if inherited:
            shown = ", ".join(str(c) for c in inherited[:40])
            extra = (
                f" … +{len(inherited) - 40} more" if len(inherited) > 40 else ""
            )
            lines.append(f"  Inherited (Pipeline): {shown}{extra}")
    after_by_day = report.get("rows_by_day_after") or {}
    for day, n in after_by_day.items():
        lines.append(f"  {day} : {int(n):,}")
    lines.extend(
        [
            "",
            f"Remaining NULL cells (mandatory): {int(report.get('remaining_null_cells') or 0)}",
        ]
    )
    if report.get("error"):
        lines.extend(["", str(report["error"])])
    return "\n".join(lines)


def apply_non_null_filter_on_parquet(
    parquet_path: str,
    *,
    step2_columns: Sequence[str] | None = None,
    transformation_config: dict[str, Any] | None = None,
    day_column: str = "trading_day",
    premium_min: float | None = None,
    premium_max: float | None = None,
    on_stage: StageFn | None = None,
) -> dict[str, Any]:
    """Apply No-Null (and optional LTP premium) day-at-a-time on a parquet file.

    Avoids loading the full multi-day frame into RAM — the path that OOM/crashed
    Analysis builds after Feature Transformations completed.
    Rewrites ``parquet_path`` in place when filtering changes rows/columns.
    """
    import gc
    import os

    import pyarrow as pa
    import pyarrow.parquet as pq

    def _stage(msg: str) -> None:
        if on_stage is not None:
            try:
                on_stage(msg)
            except Exception:
                pass

    if not os.path.isfile(parquet_path):
        raise FileNotFoundError(parquet_path)

    pf = pq.ParquetFile(parquet_path)
    schema_names = [str(n) for n in pf.schema_arrow.names]
    del pf

    if day_column not in schema_names:
        # Single-shot fallback for files without trading_day.
        import pandas as pd

        frame = pd.read_parquet(parquet_path)
        nn = apply_non_null_filter_frame(
            frame,
            step2_columns=step2_columns,
            transformation_config=transformation_config,
            day_column=day_column,
            on_stage=on_stage,
        )
        out = nn["frame"]
        premium_report = None
        if premium_min is not None and premium_max is not None:
            from .premium_ltp_filter import apply_premium_ltp_filter_frame

            prem = apply_premium_ltp_filter_frame(
                out, premium_min=float(premium_min), premium_max=float(premium_max)
            )
            out = prem["frame"]
            premium_report = dict(prem["report"])
        from .writer import _write_parquet

        _write_parquet(out, parquet_path)
        return {
            "kept_columns": list(nn["kept_columns"]),
            "dropped_columns": list(nn["dropped_columns"]),
            "report": dict(nn["report"]),
            "premium_report": premium_report,
            "row_count": int(len(out)),
            "column_count": int(len(out.columns)),
        }

    days_table = pq.read_table(parquet_path, columns=[day_column])
    days = sorted(
        {str(x) for x in days_table.column(0).to_pylist() if x is not None and str(x).strip()}
    )
    del days_table
    gc.collect()

    _stage(
        f"No-Null Step 1: scanning {len(schema_names)} columns across {len(days)} day(s)…"
    )
    empty_on_any_day: set[str] = set()
    for i, day in enumerate(days, 1):
        t = pq.read_table(parquet_path, filters=[(day_column, "=", day)])
        n = int(t.num_rows)
        for name in schema_names:
            if int(t.column(name).null_count) >= n:
                empty_on_any_day.add(name)
        _stage(
            f"No-Null Step 1: day {i}/{len(days)} ({day}) · "
            f"empty columns so far={len(empty_on_any_day)}"
        )
        del t
        gc.collect()

    kept = [c for c in schema_names if c not in empty_on_any_day]
    dropped = [c for c in schema_names if c in empty_on_any_day]
    _stage(
        f"No-Null Step 1 done: kept {len(kept)} columns, "
        f"dropped {len(dropped)} empty columns"
    )
    if not kept:
        raise ValueError("No columns remain after removing 100% NULL columns.")

    out_tmp = f"{parquet_path}.nonull.tmp"
    if os.path.isfile(out_tmp):
        try:
            os.remove(out_tmp)
        except OSError:
            pass

    writer = None
    rows_before = 0
    rows_after = 0
    rows_before_by_day: dict[str, int] = {}
    rows_after_by_day: dict[str, int] = {}
    premium_before = 0
    premium_after = 0
    final_cols: list[str] | None = None
    last_report: dict[str, Any] = {}

    for i, day in enumerate(days, 1):
        df = pq.read_table(parquet_path, filters=[(day_column, "=", day)]).to_pandas()
        before = int(len(df))
        rows_before += before
        rows_before_by_day[day] = before
        nn = apply_non_null_filter_frame(
            df,
            step2_columns=step2_columns,
            transformation_config=transformation_config,
            day_column=day_column,
            on_stage=None,
        )
        out = nn["frame"]
        keep_here = [c for c in kept if c in out.columns]
        out = out.loc[:, keep_here].copy()
        last_report = dict(nn["report"])
        del df, nn
        gc.collect()

        if premium_min is not None and premium_max is not None and not out.empty:
            from .premium_ltp_filter import apply_premium_ltp_filter_frame

            premium_before += int(len(out))
            prem = apply_premium_ltp_filter_frame(
                out, premium_min=float(premium_min), premium_max=float(premium_max)
            )
            out = prem["frame"]
            premium_after += int(len(out))

        after = int(len(out))
        rows_after += after
        rows_after_by_day[day] = after
        _stage(
            f"No-Null Step 2: day {i}/{len(days)} ({day}) · "
            f"{before:,} → {after:,} rows"
        )
        if out.empty:
            del out
            gc.collect()
            continue
        try:
            from chain_replay_ml.frame_backend import frame_to_arrow_table_via_polars

            table = frame_to_arrow_table_via_polars(out, coerce=True)
        except Exception:
            table = pa.Table.from_pandas(out, preserve_index=False)
        del out
        gc.collect()
        if final_cols is None:
            final_cols = list(table.schema.names)
            writer = pq.ParquetWriter(out_tmp, table.schema, compression="zstd")
        else:
            for c in final_cols:
                if c not in table.schema.names:
                    table = table.append_column(c, pa.nulls(table.num_rows))
            table = table.select(final_cols)
        writer.write_table(table)
        del table
        gc.collect()

    if writer is not None:
        writer.close()

    if not os.path.isfile(out_tmp):
        raise ValueError("No rows remain after No-Null / premium filtering.")

    os.replace(out_tmp, parquet_path)
    report = {
        **last_report,
        "rows_before": int(rows_before),
        "rows_after": int(rows_after),
        "columns_before": len(schema_names),
        "columns_after": len(kept),
        "empty_columns_removed": len(dropped),
        "columns_removed": list(dropped),
        "columns_removed_count": len(dropped),
        "incomplete_rows_removed": max(int(rows_before - rows_after), 0),
        "rows_removed": max(int(rows_before - rows_after), 0),
        "rows_by_day_before": rows_before_by_day,
        "rows_by_day_after": rows_after_by_day,
        "stage": (
            "pipeline_post_transformation_day_at_a_time"
            if step2_columns is not None
            else "post_transformation_day_at_a_time"
        ),
        "ok": bool(kept) and rows_after > 0,
    }
    premium_report = None
    if premium_min is not None and premium_max is not None:
        premium_report = {
            "stage": "post_no_null_day_at_a_time",
            "premium_min": float(premium_min),
            "premium_max": float(premium_max),
            "rows_before": int(premium_before),
            "rows_after": int(premium_after),
            "rows_dropped": max(int(premium_before - premium_after), 0),
        }
    _stage(
        f"No-Null ready: {len(kept)} columns kept, {rows_after:,} rows after filter"
    )
    return {
        "kept_columns": list(kept),
        "dropped_columns": list(dropped),
        "report": report,
        "premium_report": premium_report,
        "row_count": int(rows_after),
        "column_count": len(final_cols or kept),
    }

"""Sync validated dataset rows into the master SQLite DB before Parquet export."""

from __future__ import annotations

import os
from collections import defaultdict
from typing import Any, Callable

from .master_naming import path_relative_to_data_dir, resolve_master_db_path
from .master_store import MasterStore
from .orchestrator import DatasetBuildConfig

ProgressFn = Callable[[str, int, int], None]
CancelFn = Callable[[], bool]


def sync_build_rows_to_master(
    config: DatasetBuildConfig,
    *,
    all_rows: list[dict[str, Any]],
    valid_ctx: list[Any],
    implemented: list[str],
    target_columns: list[str],
    job_id: str,
    step_sec: int,
    atm_band: int,
    on_progress: ProgressFn | None = None,
    cancel_check: CancelFn | None = None,
) -> dict[str, Any]:
    """Insert validated build rows into master SQLite (one trading day per transaction)."""
    if not all_rows:
        return {"rows_synced": 0, "days_synced": 0}

    data_dir = config.resolved_data_dir()
    market = str(valid_ctx[0].source.market or "NIFTY").upper()
    master_path = config.master_db_path or resolve_master_db_path(
        data_dir,
        market=market,
        sampling_interval_sec=step_sec,
    )
    if master_path and not os.path.isabs(master_path):
        master_path = os.path.join(data_dir, master_path.replace("/", os.sep))

    by_day: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in all_rows:
        td = str(row.get("trading_day") or "").strip()
        if td:
            by_day[td].append(row)

    row_cols = list(dict.fromkeys([*implemented, *target_columns]))
    if all_rows:
        row_cols = list(dict.fromkeys([*row_cols, *all_rows[0].keys()]))

    days_total = len(valid_ctx)
    rows_synced = 0
    days_synced = 0
    days_skipped = 0
    source_results: list[dict[str, Any]] = []

    store = MasterStore(master_path)
    store.open()
    try:
        from .orchestrator import _Cancelled

        store.set_meta("master_config", {
            "market": market,
            "sampling_interval_sec": step_sec,
            "atm_band": atm_band,
            "storage_backend": "master_sqlite",
        })
        store.start_build_job(job_id=job_id, days_total=days_total)
        store.set_meta("build_schema", {
            "feature_count": len(implemented),
            "target_count": len(target_columns),
            "feature_columns": implemented,
            "target_columns": target_columns,
        })

        for di, ctx in enumerate(valid_ctx):
            if cancel_check and cancel_check():
                raise _Cancelled()
            td = ctx.source.trading_day
            src_msg = f"{td} • {ctx.source.market} • {ctx.expiry_norm}"
            if on_progress:
                on_progress(f"Master DB: {src_msg}", di, days_total)

            if store.should_skip_day(td):
                days_skipped += 1
                source_results.append({
                    "trading_day": td,
                    "status": "skipped",
                    "reason": "already in master",
                })
                continue

            day_rows = by_day.get(td) or []
            if not day_rows:
                source_results.append({
                    "trading_day": td,
                    "status": "skipped",
                    "reason": "no rows",
                })
                continue

            day_cols = list(dict.fromkeys([*row_cols, *day_rows[0].keys()]))
            try:
                store.begin_day(td, day_cols)
                inserted = store.insert_rows(day_rows)
                store.commit_day(td)
            except Exception:
                store.rollback_day()
                raise

            rows_synced += inserted
            days_synced += 1
            cov = {"samples_written": inserted, "status": "ok"}
            store.set_day_coverage(td, cov)
            source_results.append({
                "trading_day": td,
                "market": ctx.source.market,
                "expiry": ctx.expiry_norm,
                "status": "loaded",
                "rows": inserted,
                "coverage": cov,
            })
            if on_progress:
                on_progress(
                    f"Master DB: committed {inserted:,} rows for {td}",
                    di + 1,
                    days_total,
                )

        store.mark_build_complete()
        store.set_meta("master_config", {
            "market": market,
            "sampling_interval_sec": step_sec,
            "atm_band": atm_band,
            "storage_backend": "master_sqlite",
            "feature_count": len(implemented),
            "target_count": len(target_columns),
        })

        rel_path = path_relative_to_data_dir(master_path, data_dir)
        return {
            "master_db_path": rel_path,
            "master_db_abs": master_path,
            "rows_synced": rows_synced,
            "days_synced": days_synced,
            "days_skipped": days_skipped,
            "row_counts_by_day": store.row_counts_by_day(),
            "coverage_by_day": store.get_coverage_by_day(),
            "sources": source_results,
        }
    finally:
        store.close()

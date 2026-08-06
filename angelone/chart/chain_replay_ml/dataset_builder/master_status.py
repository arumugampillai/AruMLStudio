"""Read master SQLite dataset status for UI."""



from __future__ import annotations



import gc

import csv
import io
import os
import re
import sqlite3

import time

from dataclasses import dataclass

from typing import Any, Iterator, Sequence



from .master_naming import resolve_master_db_path

from .master_store import MasterStore, close_all_stores_for_path





@dataclass

class MasterDeleteResult:

    removed: list[str]

    closed_connections: int

    still_exists: bool

    errors: list[str]





def read_master_dataset_status(
    data_dir: str,
    *,
    market: str,
    interval_sec: int,
    master_db_path: str | None = None,
) -> dict[str, Any]:
    from .master_dataset_service import MasterDatasetService

    svc = MasterDatasetService.for_market(
        data_dir,
        market=market,
        interval_sec=interval_sec,
        master_db_path=master_db_path,
    )
    return svc.read_status(data_dir=data_dir, market=market, interval_sec=interval_sec)


def _file_size(path: str) -> int:
    try:
        return os.path.getsize(path) if os.path.isfile(path) else 0
    except OSError:
        return 0


def _read_table_info(conn: sqlite3.Connection, table: str) -> list[dict[str, Any]]:
    rows = conn.execute(f'PRAGMA table_info("{table}")').fetchall()
    return [
        {
            "name": str(r[1]),
            "type": str(r[2] or ""),
            "notnull": bool(r[3]),
            "pk": bool(r[5]),
        }
        for r in rows
    ]


def _truncate_preview_value(value: Any, *, max_len: int = 48) -> Any:
    if value is None:
        return None
    if isinstance(value, float):
        if abs(value) >= 1_000_000 or (abs(value) < 0.0001 and value != 0):
            return round(value, 6)
        return round(value, 4) if value != int(value) else int(value)
    text = str(value)
    if len(text) <= max_len:
        return text
    return text[: max_len - 1] + "…"


def _sample_filter_where(
    *,
    trading_day: str | None = None,
    selected_days: Sequence[str] | None = None,
    all_days: bool = False,
    token: str | None = None,
    atm_band_filter: int | None = None,
    premium_min: float | None = None,
    premium_max: float | None = None,
    delta_min: float | None = None,
    delta_max: float | None = None,
    column_names: Sequence[str] | None = None,
) -> tuple[str, list[Any]]:
    """Build WHERE for optional trading_day, token, ATM ±band, LTP premium, and |delta| band."""
    from .dataset_selection_engine import DatasetSelectionSpec, build_selection_sql_where

    days = [str(d).strip() for d in (selected_days or []) if str(d).strip()]
    spec = DatasetSelectionSpec(
        mode="post_filter",
        all_days=bool(all_days),
        single_day=str(trading_day or "").strip() or None,
        selected_days=days,
        token=str(token or "").strip() or None,
        atm_band=int(atm_band_filter) if atm_band_filter is not None else None,
        premium_min=float(premium_min) if premium_min is not None else None,
        premium_max=float(premium_max) if premium_max is not None else None,
        premium_enabled=premium_min is not None and premium_max is not None,
        delta_min=float(delta_min) if delta_min is not None else None,
        delta_max=float(delta_max) if delta_max is not None else None,
        delta_enabled=delta_min is not None and delta_max is not None,
    )
    return build_selection_sql_where(
        spec,
        profile="master_samples",
        column_names=set(column_names or []),
    )


def _columns_non_null_sql(column_names: Sequence[str]) -> str:
    """SQL fragment requiring listed columns to be non-null."""
    from .non_null_filter import _columns_non_null_sql as _impl

    return _impl(column_names)


def _discover_active_columns(
    conn: sqlite3.Connection,
    column_names: Sequence[str],
    where_sql: str,
    where_params: list[Any],
) -> list[str]:
    """Columns kept after Non-Null Step 1 (day-independent when multi-day)."""
    from .non_null_filter import discover_kept_columns_step1

    kept, _dropped = discover_kept_columns_step1(
        conn, column_names, where_sql, where_params
    )
    return kept


def _no_null_complete_where(
    conn: sqlite3.Connection,
    column_names: Sequence[str],
    where_sql: str,
    where_params: list[Any],
) -> tuple[str, list[str], list[str]]:
    """
    Non-Null pipeline: Step 1 drop empty columns, Step 2 require complete rows.

    Returns (where_sql, kept_columns, dropped_columns).
    """
    from .non_null_filter import apply_non_null_filter

    result = apply_non_null_filter(
        conn, column_names, where_sql, where_params, log=False, debug=False
    )
    return (
        str(result["where_sql"]),
        list(result["kept_columns"]),
        list(result["dropped_columns"]),
    )


def apply_no_null_filter_with_report(
    conn: sqlite3.Connection,
    column_names: Sequence[str],
    where_sql: str,
    where_params: list[Any],
) -> dict[str, Any]:
    """Full Non-Null result including debug report (for preview / export)."""
    from .non_null_filter import apply_non_null_filter

    return apply_non_null_filter(
        conn, column_names, where_sql, where_params, log=True, debug=True
    )

_PREVIEW_SAMPLE_COLUMNS = (
    "trading_day",
    "timestamp",
    "token",
    "symbol",
    "ltp",
    "spot",
)


def _preview_select_columns(column_names: Sequence[str]) -> list[str]:
    """Columns fetched for the UI sample table — not the full wide samples row."""
    available = {str(c).strip() for c in column_names if str(c).strip()}
    out = [c for c in _PREVIEW_SAMPLE_COLUMNS if c in available]
    return out or list(column_names)[: min(6, len(column_names))]


def _fetch_sample_preview(
    conn: sqlite3.Connection,
    samples_cols: list[dict[str, Any]],
    *,
    preview_day: str | None,
    selected_days: Sequence[str] | None,
    all_days: bool,
    preview_token: str | None,
    preview_limit: int,
    atm_band_filter: int | None,
    premium_min: float | None,
    premium_max: float | None,
    delta_min: float | None,
    delta_max: float | None,
    no_null_data: bool = False,
) -> dict[str, Any] | None:
    day_filter = [str(d).strip() for d in (selected_days or []) if str(d).strip()]
    preview_td = None if all_days or day_filter else (preview_day or None)
    preview_token_val = str(preview_token or "").strip() or None
    if not (all_days or bool(day_filter) or bool(preview_td)):
        return None
    preview_cap = 5000 if preview_token_val else 25
    preview_limit = max(1, min(int(preview_limit), preview_cap))
    atm_filter = int(atm_band_filter) if atm_band_filter is not None and int(atm_band_filter) >= 0 else None
    prem_lo = float(premium_min) if premium_min is not None else None
    prem_hi = float(premium_max) if premium_max is not None else None
    delta_lo = float(delta_min) if delta_min is not None else None
    delta_hi = float(delta_max) if delta_max is not None else None
    col_names = [c["name"] for c in samples_cols]
    preview_cols = _preview_select_columns(col_names)
    col_sql = ", ".join(f'"{c}"' for c in preview_cols)
    where_sql, where_params = _sample_filter_where(
        trading_day=preview_td,
        selected_days=day_filter or None,
        all_days=all_days,
        token=preview_token_val,
        atm_band_filter=atm_filter,
        premium_min=prem_lo,
        premium_max=prem_hi,
        delta_min=delta_lo,
        delta_max=delta_hi,
        column_names=col_names,
    )
    count_row = conn.execute(
        f"SELECT COUNT(*) FROM samples WHERE {where_sql}",
        where_params,
    ).fetchone()
    match_count = int(count_row[0]) if count_row else 0
    no_null_match_count: int | None = None
    no_null_active_columns: list[str] | None = None
    no_null_dropped_columns: list[str] | None = None
    no_null_report: dict[str, Any] | None = None
    if no_null_data:
        nn = apply_no_null_filter_with_report(
            conn, col_names, where_sql, where_params
        )
        fetch_where = str(nn["where_sql"])
        no_null_active_columns = list(nn["kept_columns"])
        no_null_dropped_columns = list(nn["dropped_columns"])
        no_null_report = dict(nn.get("report") or {})
        no_null_match_count = int(no_null_report.get("rows_after") or 0)
    else:
        fetch_where = where_sql
    multi_day = bool(all_days or len(day_filter) > 1)
    order_sql = "trading_day, timestamp" if multi_day else "timestamp"
    rows = conn.execute(
        f'SELECT {col_sql} FROM samples WHERE {fetch_where} ORDER BY {order_sql} LIMIT ?',
        [*where_params, preview_limit],
    ).fetchall()
    preview_rows = []
    for row in rows:
        item = {}
        for c in preview_cols:
            item[c] = _truncate_preview_value(row[c])
        preview_rows.append(item)
    return {
        "all_days": bool(all_days),
        "selected_days": day_filter,
        "trading_day": preview_td,
        "token": preview_token_val,
        "atm_band_filter": atm_filter,
        "premium_min": prem_lo,
        "premium_max": prem_hi,
        "delta_min": delta_lo,
        "delta_max": delta_hi,
        "limit": preview_limit,
        "match_count": match_count,
        "no_null_match_count": no_null_match_count,
        "no_null_data": bool(no_null_data),
        "no_null_active_columns": no_null_active_columns,
        "no_null_dropped_columns": no_null_dropped_columns,
        "no_null_report": no_null_report,
        "columns": col_names,
        "rows": preview_rows,
    }


def read_master_sample_preview(
    data_dir: str,
    *,
    market: str,
    interval_sec: int,
    master_db_path: str | None = None,
    preview_day: str | None = None,
    preview_token: str | None = None,
    preview_limit: int = 12,
    atm_band_filter: int | None = None,
    premium_min: float | None = None,
    premium_max: float | None = None,
    delta_min: float | None = None,
    delta_max: float | None = None,
    all_days: bool = False,
    selected_days: Sequence[str] | None = None,
    no_null_data: bool = False,
) -> dict[str, Any] | None:
    """Lightweight filtered preview — COUNT + sample rows only (no full DB introspection)."""
    path = master_db_path or resolve_master_db_path(
        data_dir,
        market=market,
        sampling_interval_sec=interval_sec,
    )
    if not os.path.isfile(path):
        return None
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    try:
        samples_cols = _read_table_info(conn, "samples")
        return _fetch_sample_preview(
            conn,
            samples_cols,
            preview_day=preview_day,
            selected_days=selected_days,
            all_days=all_days,
            preview_token=preview_token,
            preview_limit=preview_limit,
            atm_band_filter=atm_band_filter,
            premium_min=premium_min,
            premium_max=premium_max,
            delta_min=delta_min,
            delta_max=delta_max,
            no_null_data=no_null_data,
        )
    finally:
        conn.close()


def read_master_dataset_detail(
    data_dir: str,
    *,
    market: str,
    interval_sec: int,
    master_db_path: str | None = None,
    preview_day: str | None = None,
    preview_token: str | None = None,
    preview_limit: int = 8,
    atm_band_filter: int | None = None,
    premium_min: float | None = None,
    premium_max: float | None = None,
    delta_min: float | None = None,
    delta_max: float | None = None,
    all_days: bool = False,
    selected_days: Sequence[str] | None = None,
    include_preview: bool = True,
) -> dict[str, Any]:
    """Rich SQLite introspection for the master dataset detail page."""
    out = read_master_dataset_status(
        data_dir,
        market=market,
        interval_sec=interval_sec,
        master_db_path=master_db_path,
    )
    path = out.get("master_db_abs") or ""
    meta = out.get("master_meta") or {}
    out["file_sizes"] = {
        "db_bytes": int(meta.get("database_size") or _file_size(path)) if out.get("exists") else 0,
        "wal_bytes": int(meta.get("wal_size") or _file_size(f"{path}-wal")) if out.get("exists") else 0,
        "shm_bytes": _file_size(f"{path}-shm") if out.get("exists") else 0,
    }
    out["file_sizes"]["total_bytes"] = sum(out["file_sizes"].values())

    if not out.get("exists") or not path:
        out["sqlite"] = {"journal_mode": None, "tables": []}
        out["dataset_meta_keys"] = []
        out["day_details"] = []
        out["sample_preview"] = None
        out["column_groups"] = {"pk": [], "metadata": [], "features": [], "targets": [], "other": []}
        return out

    conn: sqlite3.Connection | None = None
    try:
        from .master_dataset_service import MasterDatasetService

        svc = MasterDatasetService(path)
        conn = sqlite3.connect(path)
        conn.row_factory = sqlite3.Row
        journal = conn.execute("PRAGMA journal_mode").fetchone()
        out["sqlite"] = {
            "journal_mode": str(journal[0]) if journal else None,
            "tables": svc.read_sqlite_tables(include_row_counts=include_preview),
        }

        meta_rows = conn.execute(
            "SELECT key, length(value) AS value_len FROM dataset_meta ORDER BY key"
        ).fetchall()
        out["dataset_meta_keys"] = [
            {"key": str(r["key"]), "value_bytes": int(r["value_len"] or 0)}
            for r in meta_rows
        ]

        out["day_details"] = svc.read_day_details(out.get("coverage_by_day"))

        samples_cols = _read_table_info(conn, "samples")
        pk_cols = [c["name"] for c in samples_cols if c["pk"]]
        meta_cols = {"trading_day", "timestamp", "token", "market", "expiry", "option_type", "symbol"}
        build_schema = out.get("build_schema")
        if not isinstance(build_schema, dict):
            store = MasterStore(path)
            store.open()
            try:
                build_schema = store.get_meta("build_schema")
                if isinstance(build_schema, dict):
                    out["build_schema"] = build_schema
            finally:
                store.close()
        feature_set = set(build_schema.get("feature_columns") or []) if isinstance(build_schema, dict) else set()
        target_set = set(build_schema.get("target_columns") or []) if isinstance(build_schema, dict) else set()
        groups: dict[str, list[dict[str, Any]]] = {
            "pk": [],
            "metadata": [],
            "features": [],
            "targets": [],
            "other": [],
        }
        for col in samples_cols:
            name = col["name"]
            if col["pk"] or name in pk_cols:
                groups["pk"].append(col)
            elif name in target_set:
                groups["targets"].append(col)
            elif name in feature_set:
                groups["features"].append(col)
            elif name in meta_cols:
                groups["metadata"].append(col)
            else:
                groups["other"].append(col)
        out["column_groups"] = groups
        out["samples_column_count"] = len(samples_cols)

        day_filter = [str(d).strip() for d in (selected_days or []) if str(d).strip()]
        run_preview = bool(include_preview) and (
            all_days or bool(day_filter) or bool(preview_day)
        )
        if run_preview:
            out["sample_preview"] = _fetch_sample_preview(
                conn,
                samples_cols,
                preview_day=preview_day,
                selected_days=day_filter or None,
                all_days=all_days,
                preview_token=preview_token,
                preview_limit=preview_limit,
                atm_band_filter=atm_band_filter,
                premium_min=premium_min,
                premium_max=premium_max,
                delta_min=delta_min,
                delta_max=delta_max,
            )
        else:
            out["sample_preview"] = None
    finally:
        if conn is not None:
            conn.close()

    return out


MAX_SAMPLE_CSV_ROWS = 500_000
_CSV_FETCH_CHUNK = 5_000


class MasterSampleCsvError(Exception):
    def __init__(self, detail: str) -> None:
        self.detail = detail
        super().__init__(detail)


def master_sample_csv_filename(
    *,
    market: str,
    interval_sec: int,
    trading_day: str | None = None,
    token: str | None = None,
    all_days: bool = False,
) -> str:
    m = str(market or "NIFTY").lower()
    if all_days or not str(trading_day or "").strip():
        base = f"master_{m}_{int(interval_sec)}s_all_days"
    else:
        base = f"master_{m}_{int(interval_sec)}s_{trading_day}"
    if token:
        safe_tok = re.sub(r"[^\w.-]", "_", str(token))
        base += f"_token_{safe_tok}"
    return f"{base}.csv"


def iter_master_sample_csv(
    data_dir: str,
    *,
    market: str,
    interval_sec: int,
    trading_day: str | None = None,
    selected_days: Sequence[str] | None = None,
    token: str | None = None,
    master_db_path: str | None = None,
    atm_band_filter: int | None = None,
    premium_min: float | None = None,
    premium_max: float | None = None,
    delta_min: float | None = None,
    delta_max: float | None = None,
    all_days: bool = False,
    no_null_data: bool = False,
) -> tuple[str, Iterator[str], int]:
    """Stream CSV lines for filtered master samples (all matching rows)."""
    td = str(trading_day or "").strip() or None
    day_filter = [str(d).strip() for d in (selected_days or []) if str(d).strip()]
    if not all_days and not td and not day_filter:
        raise MasterSampleCsvError("trading_day or selected_days is required unless all_days is true")

    path = master_db_path or resolve_master_db_path(
        data_dir,
        market=market,
        sampling_interval_sec=interval_sec,
    )
    if not os.path.isfile(path):
        raise MasterSampleCsvError("Master database file does not exist")

    token_val = str(token or "").strip() or None
    atm_filter = int(atm_band_filter) if atm_band_filter is not None and int(atm_band_filter) >= 0 else None
    prem_lo = float(premium_min) if premium_min is not None else None
    prem_hi = float(premium_max) if premium_max is not None else None
    delta_lo = float(delta_min) if delta_min is not None else None
    delta_hi = float(delta_max) if delta_max is not None else None
    filename = master_sample_csv_filename(
        market=market,
        interval_sec=interval_sec,
        trading_day=td,
        token=token_val,
        all_days=all_days,
    )
    if atm_filter is not None:
        filename = filename.replace(".csv", f"_atm{atm_filter}.csv")
    if prem_lo is not None and prem_hi is not None:
        base = filename.replace(".csv", "")
        filename = f"{base}_prem{int(prem_lo)}_{int(prem_hi)}.csv"
    if delta_lo is not None and delta_hi is not None:
        base = filename.replace(".csv", "")
        filename = f"{base}_delta{int(round(delta_lo * 100))}_{int(round(delta_hi * 100))}.csv"
    if no_null_data:
        base = filename.replace(".csv", "")
        filename = f"{base}_nonull.csv"

    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    col_names = [c["name"] for c in _read_table_info(conn, "samples")]
    if not col_names:
        conn.close()
        raise MasterSampleCsvError("samples table has no columns")

    where_sql, params = _sample_filter_where(
        trading_day=td,
        selected_days=day_filter or None,
        all_days=all_days,
        token=token_val,
        atm_band_filter=atm_filter,
        premium_min=prem_lo,
        premium_max=prem_hi,
        delta_min=delta_lo,
        delta_max=delta_hi,
        column_names=col_names,
    )

    export_cols = list(col_names)
    fetch_where = where_sql
    if no_null_data:
        fetch_where, export_cols, dropped = _no_null_complete_where(
            conn,
            col_names,
            where_sql,
            params,
        )
        if not export_cols:
            conn.close()
            raise MasterSampleCsvError(
                "No active columns remain after dropping all-null features for this filter."
            )

    count_row = conn.execute(
        f"SELECT COUNT(*) FROM samples WHERE {fetch_where}",
        params,
    ).fetchone()
    row_count = int(count_row[0]) if count_row else 0
    if row_count == 0:
        conn.close()
        raise MasterSampleCsvError(
            "No sample rows match the filter"
            + (" (no-null complete rows)" if no_null_data else "")
        )
    if row_count > MAX_SAMPLE_CSV_ROWS:
        conn.close()
        raise MasterSampleCsvError(
            f"Too many rows to export ({row_count:,}). "
            f"Limit is {MAX_SAMPLE_CSV_ROWS:,}. Narrow the filter."
        )

    col_sql = ", ".join(f'"{c}"' for c in export_cols)
    order_sql = "trading_day, timestamp" if all_days or len(day_filter) > 1 else "timestamp"
    cur = conn.execute(
        f'SELECT {col_sql} FROM samples WHERE {fetch_where} ORDER BY {order_sql}',
        params,
    )

    def _lines() -> Iterator[str]:
        buf = io.StringIO()
        writer = csv.writer(buf)
        try:
            writer.writerow(export_cols)
            yield buf.getvalue()
            buf.seek(0)
            buf.truncate(0)
            while True:
                batch = cur.fetchmany(_CSV_FETCH_CHUNK)
                if not batch:
                    break
                for row in batch:
                    writer.writerow([row[c] for c in export_cols])
                chunk = buf.getvalue()
                if chunk:
                    yield chunk
                buf.seek(0)
                buf.truncate(0)
        finally:
            conn.close()

    return filename, _lines(), row_count


def build_master_sample_csv_bytes(
    data_dir: str,
    *,
    market: str,
    interval_sec: int,
    trading_day: str | None = None,
    selected_days: Sequence[str] | None = None,
    token: str | None = None,
    master_db_path: str | None = None,
    atm_band_filter: int | None = None,
    premium_min: float | None = None,
    premium_max: float | None = None,
    delta_min: float | None = None,
    delta_max: float | None = None,
    all_days: bool = False,
    no_null_data: bool = False,
) -> tuple[str, bytes, int]:
    """Build full CSV payload in the current thread (safe for asyncio.to_thread)."""
    filename, line_iter, row_count = iter_master_sample_csv(
        data_dir,
        market=market,
        interval_sec=interval_sec,
        trading_day=trading_day,
        selected_days=selected_days,
        token=token,
        master_db_path=master_db_path,
        atm_band_filter=atm_band_filter,
        premium_min=premium_min,
        premium_max=premium_max,
        delta_min=delta_min,
        delta_max=delta_max,
        all_days=all_days,
        no_null_data=no_null_data,
    )
    return filename, "".join(line_iter).encode("utf-8"), row_count


def _checkpoint_sqlite_file(db_path: str) -> None:
    if not os.path.isfile(db_path):
        return
    conn: sqlite3.Connection | None = None
    try:
        conn = sqlite3.connect(db_path, timeout=5.0)
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        conn.commit()
    except sqlite3.Error:
        # Not a usable SQLite file (corrupt / placeholder) — still delete it.
        pass
    finally:
        if conn is not None:
            try:
                conn.close()
            except sqlite3.Error:
                pass


def related_master_artifact_paths(db_path: str) -> list[str]:
    """Main DB plus WAL/SHM and same-stem backups (``.pre_rebuild_*.bak``, pending deletes).

    Does not touch other interval DBs (e.g. ``master_dataset_nifty_10s.db``).
    """
    abs_path = os.path.abspath(db_path)
    dirname = os.path.dirname(abs_path) or "."
    base = os.path.basename(abs_path)
    if not base or not os.path.isdir(dirname):
        return [abs_path, f"{abs_path}-wal", f"{abs_path}-shm"]
    out: list[str] = []
    try:
        names = os.listdir(dirname)
    except OSError:
        return [abs_path, f"{abs_path}-wal", f"{abs_path}-shm"]
    for name in names:
        if name == base or name.startswith(base):
            out.append(os.path.join(dirname, name))
    # Ensure canonical trio even if listing races / empty dir.
    for path in (abs_path, f"{abs_path}-wal", f"{abs_path}-shm"):
        if path not in out:
            out.append(path)
    out.sort(key=lambda p: (0 if p == abs_path else 1, p))
    return out


def _try_remove_path(path: str, *, removed: list[str], errors: list[str]) -> None:
    if not os.path.isfile(path):
        return
    try:
        os.remove(path)
        if path not in removed:
            removed.append(path)
    except OSError as exc:
        errors.append(f"{os.path.basename(path)}: {exc}")


def delete_master_database_files(db_path: str) -> MasterDeleteResult:
    """Fresh-start wipe: remove master SQLite + WAL/SHM + same-stem backup artifacts.

    Closes open ``MasterStore`` handles first. Meta tables live inside the DB, so
    deleting the file clears days/progress/fingerprint/config stored there.
    """
    abs_path = os.path.abspath(db_path)
    errors: list[str] = []
    removed: list[str] = []

    closed = close_all_stores_for_path(abs_path)
    _checkpoint_sqlite_file(abs_path)
    gc.collect()

    for attempt in range(8):
        attempt_errors: list[str] = []
        for path in related_master_artifact_paths(abs_path):
            _try_remove_path(path, removed=removed, errors=attempt_errors)
        leftovers = [
            p for p in related_master_artifact_paths(abs_path) if os.path.isfile(p)
        ]
        if not leftovers:
            errors = attempt_errors
            break
        errors = attempt_errors
        if attempt == 3 and os.path.isfile(abs_path):
            pending = f"{abs_path}.pending_delete_{int(time.time())}"
            try:
                os.rename(abs_path, pending)
                os.remove(pending)
                if pending not in removed:
                    removed.append(pending)
            except OSError as exc:
                errors.append(f"rename-delete: {exc}")
        time.sleep(0.08 * (attempt + 1))
        gc.collect()

    leftovers = [p for p in related_master_artifact_paths(abs_path) if os.path.isfile(p)]
    still_exists = bool(leftovers)
    if leftovers:
        for path in leftovers:
            tip = os.path.basename(path)
            if not any(tip in e for e in errors):
                errors.append(f"still present: {tip}")

    return MasterDeleteResult(
        removed=removed,
        closed_connections=closed,
        still_exists=still_exists,
        errors=errors,
    )



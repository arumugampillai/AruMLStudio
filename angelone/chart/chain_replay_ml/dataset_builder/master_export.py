"""Export master SQLite dataset to Parquet (chunked stream)."""

from __future__ import annotations

import os
from typing import Callable, Sequence

import pandas as pd

from .master_store import MasterStore
from .writer import _coerce_parquet_frame, ensure_parquet_engine

_CHUNK = 50_000
_SINGLE_PASS_MAX = 100_000


def export_master_to_parquet(
    store: MasterStore,
    parquet_path: str,
    columns: Sequence[str],
    *,
    chunk_size: int = _CHUNK,
    on_progress: Callable[[str, int, int], None] | None = None,
) -> int:
    """Stream samples table to Parquet; returns row count written."""
    ensure_parquet_engine()
    os.makedirs(os.path.dirname(parquet_path) or ".", exist_ok=True)
    total = store.total_row_count()
    col_list = list(columns)
    if total == 0:
        from chain_replay_ml.frame_backend import write_parquet_via_polars

        write_parquet_via_polars(pd.DataFrame(columns=col_list), parquet_path)
        return 0

    if total <= _SINGLE_PASS_MAX:
        return _export_single_pass(store, parquet_path, col_list, total, on_progress)

    import pyarrow.parquet as pq
    from chain_replay_ml.frame_backend import frame_to_arrow_table_via_polars

    col_sql = ", ".join(f'"{c}"' for c in col_list)
    cur = store.conn.execute(
        f"SELECT {col_sql} FROM samples ORDER BY trading_day, timestamp, token"
    )
    written = 0
    writer = None
    tmp_path = parquet_path + ".tmp"
    try:
        while True:
            raw = cur.fetchmany(chunk_size)
            if not raw:
                break
            chunk = pd.DataFrame(raw, columns=col_list)
            try:
                table = frame_to_arrow_table_via_polars(chunk, coerce=True)
            except Exception:
                table = __import__("pyarrow").Table.from_pandas(
                    _coerce_parquet_frame(chunk), preserve_index=False
                )
            if writer is None:
                writer = pq.ParquetWriter(tmp_path, table.schema)
            writer.write_table(table)
            written += len(raw)
            if on_progress:
                on_progress(
                    f"Exporting master → Parquet… {written:,}/{total:,} rows",
                    min(written, total),
                    max(total, 1),
                )
        if writer is not None:
            writer.close()
        if os.path.isfile(parquet_path):
            os.remove(parquet_path)
        os.replace(tmp_path, parquet_path)
    except Exception:
        if writer is not None:
            writer.close()
        if os.path.isfile(tmp_path):
            os.remove(tmp_path)
        raise
    return written


def _export_single_pass(
    store: MasterStore,
    parquet_path: str,
    col_list: list[str],
    total: int,
    on_progress: Callable[[str, int, int], None] | None,
) -> int:
    from .writer import _write_parquet

    if on_progress:
        on_progress(f"Reading {total:,} rows from master…", 0, 3)
    col_sql = ", ".join(f'"{c}"' for c in col_list)
    rows = store.conn.execute(
        f"SELECT {col_sql} FROM samples ORDER BY trading_day, timestamp, token"
    ).fetchall()
    df = pd.DataFrame(rows, columns=col_list)
    if on_progress:
        on_progress("Writing Parquet file…", 1, 3)
    _write_parquet(df, parquet_path)
    if on_progress:
        on_progress("Completed", 3, 3)
    return len(df)

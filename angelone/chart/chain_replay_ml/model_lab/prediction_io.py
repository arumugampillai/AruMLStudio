"""Streaming parquet IO for Model Lab prediction dataset (day batches)."""

from __future__ import annotations

import json
import os
from typing import Any, Callable

import pandas as pd

from chain_replay_ml.dataset_builder.writer import _safe_filename, datasets_dir
from chain_replay_ml.training.dataset_loader import DatasetLoaderError, _parquet_columns

from .prediction_progress import PARQUET_BATCH_ROWS

BatchProgressFn = Callable[[dict[str, Any]], None]


def resolve_dataset_parquet(data_dir: str, dataset_name: str) -> tuple[str, str]:
    safe = _safe_filename(dataset_name)
    out_dir = datasets_dir(data_dir)
    parquet_path = os.path.join(out_dir, f"{safe}.parquet")
    meta_path = os.path.join(out_dir, f"{safe}.json")
    if not os.path.isfile(parquet_path):
        raise DatasetLoaderError(f"Parquet not found: {parquet_path}")
    return parquet_path, meta_path


def load_dataset_meta(meta_path: str) -> dict[str, Any]:
    if not meta_path or not os.path.isfile(meta_path):
        return {}
    try:
        with open(meta_path, encoding="utf-8") as fh:
            doc = json.load(fh)
        return doc if isinstance(doc, dict) else {}
    except (OSError, json.JSONDecodeError, TypeError):
        return {}


def dataset_meta_path(data_dir: str, dataset_name: str) -> str:
    safe = _safe_filename(dataset_name)
    return os.path.join(datasets_dir(data_dir), f"{safe}.json")


def dataset_row_counts_from_meta(meta: dict[str, Any] | None) -> dict[str, int]:
    """
    Per-day parent dataset row counts from registry metadata.

    Prefers ``sources[].rows`` (written by master→registry export). Does not
    scan parquet.
    """
    meta = meta or {}
    out: dict[str, int] = {}
    for block_key in ("sources", "days"):
        raw = meta.get(block_key)
        if not isinstance(raw, list):
            continue
        for item in raw:
            if not isinstance(item, dict):
                continue
            day = str(item.get("trading_day") or item.get("day") or "").strip()
            if not day:
                continue
            raw_n = item.get("rows")
            if raw_n is None:
                raw_n = item.get("row_count")
            if raw_n is None:
                continue
            try:
                n = int(raw_n)
            except (TypeError, ValueError):
                continue
            # sources wins over days when both present
            if block_key == "sources" or day not in out:
                out[day] = n
    return out


def load_parent_dataset_row_counts(data_dir: str, dataset_name: str) -> dict[str, int]:
    """Load ``sources[].rows`` from the dataset registry JSON (no parquet)."""
    return dataset_row_counts_from_meta(
        load_dataset_meta(dataset_meta_path(data_dir, dataset_name)),
    )


def parquet_num_rows(parquet_path: str) -> int | None:
    try:
        import pyarrow.parquet as pq

        meta = pq.ParquetFile(parquet_path).metadata
        return int(meta.num_rows) if meta is not None else None
    except Exception:
        return None


def list_trading_days_from_meta(meta: dict[str, Any]) -> list[str]:
    days: list[str] = []
    for block_key in ("days", "sources"):
        raw = meta.get(block_key)
        if not isinstance(raw, list):
            continue
        for item in raw:
            if isinstance(item, dict):
                d = str(item.get("trading_day") or item.get("day") or "").strip()
            else:
                d = str(item or "").strip()
            if d:
                days.append(d)
        if days:
            break
    if days:
        return sorted(dict.fromkeys(days))
    return []


def catalog_trading_days(
    parquet_path: str,
    *,
    meta: dict[str, Any] | None = None,
    on_progress: BatchProgressFn | None = None,
) -> list[str]:
    """Prefer metadata day list; otherwise scan only the trading_day column."""
    meta = meta or {}
    days = list_trading_days_from_meta(meta)
    if days:
        if on_progress:
            on_progress({
                "stage_detail": f"{len(days)} trading days found (metadata)",
                "days_found": len(days),
            })
        return days

    try:
        import pyarrow.dataset as ds

        dataset = ds.dataset(parquet_path, format="parquet")
        scanner = dataset.scanner(columns=["trading_day"], batch_size=PARQUET_BATCH_ROWS * 4)
        found: set[str] = set()
        scanned = 0
        for batch in scanner.to_batches():
            scanned += batch.num_rows
            for v in batch.column(0).to_pylist():
                if v is None:
                    continue
                s = str(v).strip()
                if s:
                    found.add(s)
            if on_progress:
                on_progress({
                    "stage_detail": f"Scanning trading_day… {scanned:,} rows, {len(found)} days",
                    "rows_loaded": scanned,
                    "days_found": len(found),
                })
        out = sorted(found)
        if on_progress:
            on_progress({"stage_detail": f"{len(out)} trading days found", "days_found": len(out)})
        return out
    except Exception:
        try:
            import pyarrow.parquet as pq

            table = pq.read_table(parquet_path, columns=["trading_day"])
            values = table.column(0).to_pylist()
            out = sorted({str(v).strip() for v in values if v is not None and str(v).strip()})
            if on_progress:
                on_progress({"stage_detail": f"{len(out)} trading days found", "days_found": len(out)})
            return out
        except Exception as exc:
            raise DatasetLoaderError(f"Could not list trading days: {exc}") from exc


def estimate_sample_total(
    parquet_path: str,
    *,
    meta: dict[str, Any] | None = None,
) -> int:
    meta = meta or {}
    for key in ("row_count", "rows", "sample_count"):
        if meta.get(key) is not None:
            try:
                return int(meta[key])
            except (TypeError, ValueError):
                pass
    n = parquet_num_rows(parquet_path)
    return int(n or 0)


def count_trading_day_rows(
    parquet_path: str,
    trading_day: str,
    *,
    on_progress: BatchProgressFn | None = None,
) -> int:
    """Count rows for one day (single-column scan) with incremental progress."""
    try:
        import pyarrow.dataset as ds

        dataset = ds.dataset(parquet_path, format="parquet")
        filt = ds.field("trading_day") == trading_day
        scanner = dataset.scanner(
            columns=["trading_day"],
            filter=filt,
            batch_size=PARQUET_BATCH_ROWS * 4,
        )
        total = 0
        for batch in scanner.to_batches():
            total += int(batch.num_rows)
            if on_progress:
                on_progress({
                    "stage_detail": f"Counting rows… {total:,}",
                    "rows_loaded": total,
                    "rows_day_total": total,
                    "current_day": trading_day,
                })
        return total
    except Exception:
        df = load_trading_day_frame(parquet_path, trading_day, columns=["trading_day"])
        n = int(len(df))
        if on_progress:
            on_progress({"rows_loaded": n, "rows_day_total": n, "current_day": trading_day})
        return n


def load_trading_day_frame(
    parquet_path: str,
    trading_day: str,
    *,
    columns: list[str],
    on_progress: BatchProgressFn | None = None,
    batch_rows: int = PARQUET_BATCH_ROWS,
    day_row_total: int | None = None,
) -> pd.DataFrame:
    """
    Load one trading day only, in Arrow record batches with optional progress.

    Never loads the full parent parquet — only the filtered day.
    """
    read_cols = _parquet_columns(parquet_path, columns)
    if not read_cols:
        raise DatasetLoaderError("No requested columns found in parquet")
    if "trading_day" not in read_cols:
        read_cols = ["trading_day", *read_cols]

    try:
        import pyarrow as pa
        import pyarrow.dataset as ds

        dataset = ds.dataset(parquet_path, format="parquet")
        filt = ds.field("trading_day") == trading_day
        scanner = dataset.scanner(
            columns=read_cols,
            filter=filt,
            batch_size=max(1_000, int(batch_rows)),
        )
        batches: list[Any] = []
        loaded = 0
        for batch in scanner.to_batches():
            batches.append(batch)
            loaded += int(batch.num_rows)
            if on_progress:
                on_progress({
                    "current_day": trading_day,
                    "rows_loaded": loaded,
                    "rows_day_total": int(day_row_total or loaded),
                    "stage_detail": (
                        f"Rows loaded {loaded:,}"
                        + (f" / {int(day_row_total):,}" if day_row_total else "")
                    ),
                })
        if not batches:
            if on_progress:
                on_progress({
                    "current_day": trading_day,
                    "rows_loaded": 0,
                    "rows_day_total": int(day_row_total or 0),
                    "stage_detail": "Rows loaded 0",
                })
            return pd.DataFrame(columns=read_cols)
        table = pa.Table.from_batches(batches)
        df = table.to_pandas()
        if on_progress:
            on_progress({
                "current_day": trading_day,
                "rows_loaded": int(len(df)),
                "rows_day_total": int(day_row_total or len(df)),
                "stage_detail": f"Rows loaded {len(df):,} / {int(day_row_total or len(df)):,}",
            })
        return df
    except DatasetLoaderError:
        raise
    except Exception as exc:
        try:
            import pyarrow.parquet as pq

            table = pq.read_table(
                parquet_path,
                columns=read_cols,
                filters=[("trading_day", "=", trading_day)],
            )
            df = table.to_pandas()
            if on_progress:
                on_progress({
                    "current_day": trading_day,
                    "rows_loaded": int(len(df)),
                    "rows_day_total": int(len(df)),
                    "stage_detail": f"Rows loaded {len(df):,}",
                })
            return df
        except TypeError:
            df = pd.read_parquet(parquet_path, columns=read_cols)
            out = df[df["trading_day"].astype(str) == str(trading_day)].copy()
            if on_progress:
                on_progress({
                    "current_day": trading_day,
                    "rows_loaded": int(len(out)),
                    "rows_day_total": int(len(out)),
                    "stage_detail": f"Rows loaded {len(out):,}",
                })
            return out
        except Exception as inner:
            raise DatasetLoaderError(f"Failed to load day {trading_day}: {exc}; {inner}") from inner

"""Write Parquet dataset and companion metadata JSON."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from typing import Any, Callable

import pandas as pd

from .pipeline_identity import BUILDER_VERSION, METADATA_VERSION
from .master_naming import path_relative_to_data_dir


def _parquet_engine_available() -> bool:
    try:
        import pyarrow  # noqa: F401
        return True
    except ImportError:
        pass
    try:
        import fastparquet  # noqa: F401
        return True
    except ImportError:
        return False


def ensure_parquet_engine(*, auto_install: bool = True) -> None:
    """Raise a clear error if neither pyarrow nor fastparquet is available."""
    if _parquet_engine_available():
        return

    if auto_install:
        try:
            subprocess.check_call(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    "pyarrow",
                    "--trusted-host",
                    "pypi.org",
                    "--trusted-host",
                    "files.pythonhosted.org",
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Parquet export requires pyarrow. Auto-install failed for {sys.executable}: {exc}. "
                f"Try manually: {sys.executable} -m pip install pyarrow"
            ) from exc
        if _parquet_engine_available():
            return

    raise RuntimeError(
        f"Parquet export requires pyarrow or fastparquet in {sys.executable}. "
        f"Install with: {sys.executable} -m pip install pyarrow"
    )


def parquet_engine_status() -> dict[str, Any]:
    ok = _parquet_engine_available()
    return {
        "python": sys.executable,
        "pyarrow": ok,
        "install_hint": f"{sys.executable} -m pip install pyarrow",
    }


_META_TEXT_COLS = frozenset({
    "trading_day",
    "market",
    "expiry",
    "option_type",
    "symbol",
    "token",
})


def _coerce_parquet_frame_pandas(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize mixed/extension dtypes to plain numpy columns for parquet export."""
    out: dict[str, pd.Series] = {}
    for col in df.columns:
        s = df[col]
        if isinstance(s.dtype, pd.ArrowDtype):
            s = pd.Series(s.to_numpy(dtype=object), index=s.index)
        if col in _META_TEXT_COLS:
            out[col] = s.map(
                lambda v: None
                if v is None or (isinstance(v, float) and pd.isna(v))
                else str(v)
            )
            continue
        # Always float64 for feature/target columns so chunked ParquetWriter schemas match
        # (all-null early chunks must not become string/null while later chunks are double).
        out[col] = pd.to_numeric(s, errors="coerce").astype("float64")
    return pd.DataFrame(out)


def _coerce_parquet_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize dtypes for parquet (Polars internals, Pandas out for callers/tests)."""
    try:
        from chain_replay_ml.frame_backend import (
            coerce_frame_for_parquet,
            polars_to_pandas,
        )

        return polars_to_pandas(coerce_frame_for_parquet(df, meta_text_cols=_META_TEXT_COLS))
    except Exception:
        return _coerce_parquet_frame_pandas(df)


def _write_parquet(
    df: pd.DataFrame,
    parquet_path: str,
    *,
    on_progress: Callable[[str, int, int], None] | None = None,
) -> None:
    n_rows = len(df)
    n_cols = len(df.columns)
    if on_progress:
        on_progress(f"Coercing {n_cols} columns ({n_rows:,} rows)…", 1, 3)
    if on_progress:
        on_progress(f"Encoding Parquet ({n_rows:,} rows × {n_cols} cols)…", 1, 3)
    try:
        from chain_replay_ml.frame_backend import write_parquet_via_polars

        if on_progress:
            on_progress("Writing Parquet file to disk…", 1, 3)
        write_parquet_via_polars(df, parquet_path, meta_text_cols=_META_TEXT_COLS)
        return
    except Exception:
        pass
    # Fallback: Pandas coerce → Arrow → Parquet
    import pyarrow as pa
    import pyarrow.parquet as pq

    safe = _coerce_parquet_frame_pandas(df)
    if on_progress:
        on_progress("Writing Parquet file to disk…", 1, 3)
    table = pa.Table.from_pandas(safe, preserve_index=False)
    pq.write_table(table, parquet_path)


def datasets_dir(data_dir: str | None = None) -> str:
    r"""Canonical analysis datasets directory: D:\data\datasets\analysis."""
    if data_dir is None:
        from chain_replay_ml.core.data_root import get_data_root_service
        return get_data_root_service().get_datasets_dir("analysis")
    d_str = str(data_dir).strip()
    sub_analysis = os.path.join(d_str, "datasets", "analysis")
    if os.path.isdir(sub_analysis):
        return sub_analysis
    if os.path.basename(os.path.normpath(d_str)).lower() in ("datasets", "analysis"):
        return os.path.abspath(d_str)
    # Check legacy data_dir/datasets if exists
    legacy = os.path.join(d_str, "datasets")
    if os.path.isdir(legacy):
        return legacy
    from chain_replay_ml.core.data_root import get_data_root_service
    return get_data_root_service().get_datasets_dir("analysis")


def read_dataset_parquet(parquet_path: str) -> pd.DataFrame:
    ensure_parquet_engine()
    return pd.read_parquet(parquet_path)


_DF_CHUNK_ROWS = 50_000
_LARGE_DATASET_ROWS = 100_000


def _dataframe_from_rows(
    rows: list[dict[str, Any]],
    on_progress: Callable[[str, int, int], None] | None = None,
) -> pd.DataFrame:
    n = len(rows)
    if n == 0:
        return pd.DataFrame()
    n_cols = len(rows[0]) if rows else 0
    if on_progress:
        on_progress(f"Building DataFrame ({n:,} rows × {n_cols} cols)…", 0, 3)
    if n <= _LARGE_DATASET_ROWS:
        cols = list(rows[0].keys())
        return pd.DataFrame.from_records(rows, columns=cols)
    frames: list[pd.DataFrame] = []
    cols = list(rows[0].keys())
    for i in range(0, n, _DF_CHUNK_ROWS):
        end = min(i + _DF_CHUNK_ROWS, n)
        frames.append(pd.DataFrame.from_records(rows[i:end], columns=cols))
        if on_progress:
            on_progress(f"Building DataFrame… {end:,}/{n:,} rows", 0, 3)
    return pd.concat(frames, ignore_index=True)


def write_dataset(
    *,
    data_dir: str,
    dataset_name: str,
    rows: list[dict[str, Any]],
    metadata: dict[str, Any],
    on_progress: Callable[[str, int, int], None] | None = None,
    existing_df: pd.DataFrame | None = None,
    preserve_created_at: str | None = None,
) -> tuple[str, str, int, int]:
    out_dir = datasets_dir(data_dir)
    safe_name = _safe_filename(dataset_name)
    parquet_path = os.path.join(out_dir, f"{safe_name}.parquet")
    json_path = os.path.join(out_dir, f"{safe_name}.json")

    ensure_parquet_engine()
    df_new = _dataframe_from_rows(rows, on_progress=on_progress)
    if existing_df is not None and len(existing_df):
        df = pd.concat([existing_df, df_new], ignore_index=True)
    else:
        df = df_new

    from .transformations import (
        normalize_transformation_config,
        run_transformation_pipeline,
    )
    from .transformations.base import TransformContext

    meta_in = dict(metadata or {})
    if isinstance(meta_in.get("transformations"), dict):
        xform_config = normalize_transformation_config(meta_in.get("transformations"))
    else:
        xform_config = normalize_transformation_config({
            "transformation_pipeline_version": meta_in.get(
                "transformation_pipeline_version"
            ),
            "transformations": meta_in.get("transformations"),
        })

    def _log(msg: str) -> None:
        if on_progress:
            on_progress(msg, 0, 3)

    interval = None
    sampling = meta_in.get("sampling") if isinstance(meta_in.get("sampling"), dict) else {}
    ds_cfg = meta_in.get("dataset_configuration") if isinstance(meta_in.get("dataset_configuration"), dict) else {}
    ds_sampling = ds_cfg.get("sampling") if isinstance(ds_cfg.get("sampling"), dict) else {}
    for candidate in (
        sampling.get("interval_sec"),
        ds_sampling.get("interval_sec"),
        meta_in.get("sample_interval_sec"),
    ):
        if candidate is not None:
            try:
                interval = float(candidate)
                break
            except (TypeError, ValueError):
                pass

    ctx = TransformContext(
        config=xform_config,
        data_dir=data_dir,
        dataset_name=safe_name,
        sample_interval_sec=interval,
        metadata=meta_in,
        dataset_info={"sample_interval_sec": interval} if interval is not None else {},
        logger=_log,
        progress_callback=on_progress,
    )

    if on_progress:
        on_progress("Loading Master Dataset...", 0, 3)
    pipe = run_transformation_pipeline(df, xform_config, context=ctx, log_fn=_log)
    df = pipe.frame
    if on_progress:
        on_progress("Export Dataset...", 1, 3)
        on_progress("Writing Parquet file…", 1, 3)
    _write_parquet(df, parquet_path, on_progress=on_progress)
    if on_progress:
        on_progress("Writing metadata JSON…", 2, 3)

    meta = meta_in
    meta.setdefault("dataset_name", safe_name)
    created_at = preserve_created_at or meta.get("created_at")
    if created_at:
        meta["created_at"] = created_at
    else:
        meta.setdefault("created_at", datetime.now(timezone.utc).isoformat())
    meta.setdefault("version", METADATA_VERSION)
    meta.setdefault("builder_version", BUILDER_VERSION)
    meta["row_count"] = int(len(df))
    meta["column_count"] = int(len(df.columns))
    meta.update(pipe.metadata_block)
    # Experiment identity — prefer explicit top-level sample_interval_sec.
    if "sample_interval_sec" not in meta or meta.get("sample_interval_sec") in (None, "", 0):
        from .transformations.time_shift import normalize_sample_interval_value

        stamped = normalize_sample_interval_value(
            getattr(pipe, "sample_interval_sec", None)
            or (meta.get("sampling") or {}).get("interval_sec")
            or interval
        )
        if stamped is not None:
            meta["sample_interval_sec"] = stamped
    meta["output_parquet"] = path_relative_to_data_dir(parquet_path, data_dir)
    meta["output_json"] = path_relative_to_data_dir(json_path, data_dir)

    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, ensure_ascii=False)
        fh.write("\n")

    parquet_bytes = os.path.getsize(parquet_path)
    json_bytes = os.path.getsize(json_path)

    if on_progress:
        on_progress("Completed", 3, 3)

    return parquet_path, json_path, parquet_bytes, json_bytes


def patch_dataset_metadata(json_path: str, updates: dict[str, Any]) -> None:
    with open(json_path, encoding="utf-8") as fh:
        meta = json.load(fh)
    meta.update(updates)
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, ensure_ascii=False)
        fh.write("\n")


def _safe_filename(name: str) -> str:
    cleaned = "".join(c if c.isalnum() or c in ("_", "-", ".") else "_" for c in str(name).strip())
    return cleaned or "dataset"


def normalize_days_meta(meta_doc: dict[str, Any]) -> list[dict[str, Any]]:
    """Return days as list of dicts; tolerate legacy list of trading-day strings."""
    market = str(meta_doc.get("market") or "NIFTY")
    out: list[dict[str, Any]] = []
    for d in meta_doc.get("days") or []:
        if isinstance(d, dict):
            td = str(d.get("trading_day") or "").strip()
            if not td:
                continue
            out.append({
                "trading_day": td,
                "market": str(d.get("market") or market),
                "expiry": str(d.get("expiry") or ""),
                "source_id": str(d.get("source_id") or td),
            })
        elif isinstance(d, str) and d.strip():
            td = d.strip()
            expiry = ""
            sid = td
            for s in meta_doc.get("sources") or []:
                if not isinstance(s, dict):
                    continue
                if str(s.get("trading_day") or "").strip() == td:
                    expiry = str(s.get("expiry") or "")
                    sid = str(s.get("source_id") or sid)
                    break
            out.append({
                "trading_day": td,
                "market": market,
                "expiry": expiry,
                "source_id": sid,
            })
    if out:
        return out
    for s in meta_doc.get("sources") or []:
        if not isinstance(s, dict):
            continue
        td = str(s.get("trading_day") or "").strip()
        if not td:
            continue
        out.append({
            "trading_day": td,
            "market": str(s.get("market") or market),
            "expiry": str(s.get("expiry") or ""),
            "source_id": str(s.get("source_id") or td),
        })
    return out

"""Immutable Label Run writer — parquet + meta only (no feature datasets)."""

from __future__ import annotations

import json
import os
from typing import Any

import pandas as pd

from .paths import label_run_meta_path, label_run_parquet_path, label_runs_dir, mint_label_run_id
from .types import assert_label_only_columns, label_run_meta_template


def write_label_run(
    data_dir: str,
    *,
    frame: pd.DataFrame,
    strategy: str,
    strategy_version: str,
    engine_version: str,
    dataset_id: str,
    dataset_hash: str | None,
    parameters: dict[str, Any],
    primary_target: str,
    display_target: str | None = None,
    label_encoding: dict[str, int] | None = None,
    join_keys: list[str] | None = None,
    run_id: str | None = None,
    valid_mask: pd.Series | None = None,
) -> dict[str, Any]:
    """Write ``{run_id}.parquet`` + ``{run_id}_meta.json`` under ``data/label_runs/``.

    Never writes under ``data/datasets/``.
    """
    if frame is None or frame.empty:
        raise ValueError("Label Run frame is empty")
    rid = str(run_id or mint_label_run_id(strategy)).strip()
    if not rid:
        raise ValueError("run_id is required")

    out_dir = label_runs_dir(data_dir)
    os.makedirs(out_dir, exist_ok=True)
    pq_path = label_run_parquet_path(data_dir, rid)
    meta_path = label_run_meta_path(data_dir, rid)
    if os.path.exists(pq_path) or os.path.exists(meta_path):
        raise FileExistsError(f"Label Run already exists (immutable): {rid}")

    cols = list(frame.columns)
    assert_label_only_columns(cols, primary_target=primary_target)
    if primary_target not in frame.columns:
        raise ValueError(f"primary_target {primary_target!r} missing from Label Run frame")

    keys = list(join_keys or [])
    if not keys:
        if "master_row_id" in frame.columns:
            keys = ["master_row_id"]
        elif "sample_id" in frame.columns:
            keys = ["sample_id"]
        else:
            keys = [c for c in ("trading_day", "timestamp", "token") if c in frame.columns]
    missing_keys = [k for k in keys if k not in frame.columns]
    if missing_keys:
        raise ValueError(f"Label Run missing join keys: {missing_keys}")

    if valid_mask is None and "is_valid" in frame.columns:
        valid_mask = frame["is_valid"].fillna(True).astype(bool)
    if valid_mask is None:
        valid_n = int(len(frame))
        invalid_n = 0
    else:
        valid_n = int(valid_mask.sum())
        invalid_n = int((~valid_mask).sum())

    # Atomic-ish write: parquet then meta.
    tmp_pq = pq_path + ".tmp"
    frame.to_parquet(tmp_pq, index=False)
    os.replace(tmp_pq, pq_path)

    meta = label_run_meta_template(
        run_id=rid,
        strategy=str(strategy),
        strategy_version=str(strategy_version),
        engine_version=str(engine_version),
        dataset_id=str(dataset_id),
        dataset_hash=dataset_hash,
        parameters=dict(parameters or {}),
        rows=int(len(frame)),
        valid_rows=valid_n,
        invalid_rows=invalid_n,
        primary_target=str(primary_target),
        display_target=display_target,
        label_encoding=label_encoding,
        join_keys=keys,
    )
    tmp_meta = meta_path + ".tmp"
    with open(tmp_meta, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, default=str)
    os.replace(tmp_meta, meta_path)

    return {
        "ok": True,
        "run_id": rid,
        "parquet_path": pq_path,
        "meta_path": meta_path,
        "meta": meta,
    }

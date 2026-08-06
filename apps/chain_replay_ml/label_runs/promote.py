"""Promote an existing feature-parquet target column into a Label Run (no feature copy)."""

from __future__ import annotations

import hashlib
import os
from typing import Any

import pandas as pd

from chain_replay_ml.dataset_builder.writer import _safe_filename, datasets_dir
from chain_replay_ml.outcome_label_engine.types import ENGINE_VERSION

from .writer import write_label_run


def _dataset_hash(parquet_path: str) -> str | None:
    try:
        h = hashlib.sha256()
        with open(parquet_path, "rb") as fh:
            # Sample head+tail for large files (reproducibility fingerprint, not full hash).
            head = fh.read(1024 * 1024)
            h.update(head)
            fh.seek(0, os.SEEK_END)
            size = fh.tell()
            h.update(str(size).encode())
            if size > 1024 * 1024:
                fh.seek(max(0, size - 1024 * 1024))
                h.update(fh.read(1024 * 1024))
        return h.hexdigest()[:24]
    except Exception:
        return None


def promote_feature_column_to_label_run(
    data_dir: str,
    dataset_name: str,
    target_column: str,
    *,
    strategy: str = "fixed_horizon",
    strategy_version: str = "1.0",
    parameters: dict[str, Any] | None = None,
    run_id: str | None = None,
    primary_target: str | None = None,
) -> dict[str, Any]:
    """Extract identity + one target column from a feature dataset into a Label Run.

    Does not modify the feature parquet.
    """
    safe = _safe_filename(dataset_name)
    parquet_path = os.path.join(datasets_dir(data_dir), f"{safe}.parquet")
    if not os.path.isfile(parquet_path):
        raise FileNotFoundError(f"Feature dataset missing: {parquet_path}")

    target = str(target_column or "").strip()
    if not target:
        raise ValueError("target_column is required")

    # Column prune read.
    import pyarrow.parquet as pq

    schema_names = set(pq.read_schema(parquet_path).names)
    if target not in schema_names:
        raise KeyError(f"Column {target!r} not in feature dataset")

    id_candidates = ["master_row_id", "sample_id", "trading_day", "timestamp", "token"]
    id_cols = [c for c in id_candidates if c in schema_names]
    if "master_row_id" not in id_cols and not all(
        c in id_cols for c in ("trading_day", "timestamp", "token")
    ):
        raise ValueError(
            "Feature dataset needs master_row_id or (trading_day, timestamp, token) "
            "to promote a Label Run."
        )

    cols = list(dict.fromkeys([*id_cols, target]))
    df = pd.read_parquet(parquet_path, columns=cols)
    out_target = str(primary_target or target)
    if out_target != target:
        df = df.rename(columns={target: out_target})

    df["is_valid"] = df[out_target].notna()
    # Drop rows without a label value from the run (still immutable feature file).
    df = df[df["is_valid"]].copy()

    join_keys = (
        ["master_row_id"]
        if "master_row_id" in df.columns
        else [c for c in ("trading_day", "timestamp", "token") if c in df.columns]
    )

    params = dict(parameters or {})
    params.setdefault("source_column", target)
    params.setdefault("promote", True)

    return write_label_run(
        data_dir,
        frame=df,
        strategy=strategy,
        strategy_version=strategy_version,
        engine_version=ENGINE_VERSION,
        dataset_id=str(dataset_name),
        dataset_hash=_dataset_hash(parquet_path),
        parameters=params,
        primary_target=out_target,
        display_target=out_target,
        join_keys=join_keys,
        run_id=run_id,
    )

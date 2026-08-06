"""Create Triple Barrier Label Runs from a Feature Dataset sample grid.

Uses identity + ``ltp`` from the immutable feature parquet (column-pruned,
day-chunked). Forward mark paths are built from the same-day sample grid
(``TripleBarrierStrategy``), matching Prediction-path TB semantics without
mutating the feature file.
"""

from __future__ import annotations

import hashlib
import os
from typing import Any

import pandas as pd

from chain_replay_ml.dataset_builder.writer import _safe_filename, datasets_dir
from chain_replay_ml.outcome_label_engine import (
    ENGINE_VERSION,
    TRIPLE_BARRIER_LABEL_ENCODING,
    TRIPLE_BARRIER_STRATEGY_ID,
    LabelSourceContext,
    LabelStrategyConfig,
    ensure_builtin_strategies,
    get_triple_barrier_strategy,
)
from chain_replay_ml.outcome_label_engine.triple_barrier import STRATEGY_VERSION

from .writer import write_label_run

_LABEL_OUT_COLS = (
    "master_row_id",
    "trading_day",
    "timestamp",
    "token",
    "label_id",
    "label_name",
    "entry_price",
    "exit_price",
    "exit_reason",
    "holding_seconds",
    "is_valid",
    "invalid_reason",
    "realized_return",
)


def _dataset_hash(parquet_path: str) -> str | None:
    try:
        h = hashlib.sha256()
        with open(parquet_path, "rb") as fh:
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


def _feature_parquet_path(data_dir: str, dataset_name: str) -> str:
    safe = _safe_filename(dataset_name)
    return os.path.join(datasets_dir(data_dir), f"{safe}.parquet")


def _read_identity_frame(parquet_path: str) -> pd.DataFrame:
    import pyarrow.parquet as pq

    schema_names = set(pq.read_schema(parquet_path).names)
    id_candidates = ["master_row_id", "sample_id", "trading_day", "timestamp", "token", "ltp"]
    cols = [c for c in id_candidates if c in schema_names]
    if "ltp" not in cols:
        raise ValueError(
            "Feature dataset needs an ``ltp`` column for Triple Barrier sample-grid paths. "
            "Re-export analysis with LTP, or provide a denser Prediction path source."
        )
    if "master_row_id" not in cols and not all(
        c in cols for c in ("trading_day", "timestamp", "token")
    ):
        raise ValueError(
            "Feature dataset needs master_row_id or (trading_day, timestamp, token) "
            "for Triple Barrier Label Runs."
        )
    if "trading_day" not in cols or "timestamp" not in cols or "token" not in cols:
        raise ValueError(
            "Feature dataset needs trading_day, timestamp, and token for day-chunked "
            "Triple Barrier labeling."
        )
    return pd.read_parquet(parquet_path, columns=cols)


def _day_samples(day_df: pd.DataFrame) -> list[dict[str, Any]]:
    """Build TB sample dicts; preserve master_row_id for join after labeling."""
    samples: list[dict[str, Any]] = []
    records = day_df.to_dict(orient="records")
    for row in records:
        sample: dict[str, Any] = {
            "trading_day": str(row.get("trading_day") or ""),
            "token": str(row.get("token") or ""),
            "timestamp": row.get("timestamp"),
            "ltp": row.get("ltp"),
            "current_ltp": row.get("ltp"),
        }
        if "master_row_id" in row and row.get("master_row_id") is not None:
            sample["master_row_id"] = row["master_row_id"]
        if "sample_id" in row and row.get("sample_id") is not None:
            sample["sample_id"] = row["sample_id"]
        samples.append(sample)
    return samples


def _labeled_to_frame(
    samples: list[dict[str, Any]],
    labeled_rows: list[dict[str, Any]],
) -> pd.DataFrame:
    out: list[dict[str, Any]] = []
    for sample, lab in zip(samples, labeled_rows):
        row = {
            "trading_day": lab.get("trading_day", sample.get("trading_day")),
            "timestamp": lab.get("timestamp", sample.get("timestamp")),
            "token": lab.get("token", sample.get("token")),
            "label_id": lab.get("label_id"),
            "label_name": lab.get("label_name"),
            "entry_price": lab.get("entry_price"),
            "exit_price": lab.get("exit_price"),
            "exit_reason": lab.get("exit_reason"),
            "holding_seconds": lab.get("holding_seconds"),
            "is_valid": bool(lab.get("is_valid")),
            "invalid_reason": lab.get("invalid_reason"),
            "realized_return": lab.get("realized_return"),
        }
        if "master_row_id" in sample:
            row["master_row_id"] = sample["master_row_id"]
        if "sample_id" in sample:
            row["sample_id"] = sample["sample_id"]
        out.append(row)
    frame = pd.DataFrame(out)
    keep = [c for c in _LABEL_OUT_COLS if c in frame.columns]
    # Keep sample_id if present and no master_row_id.
    if "sample_id" in frame.columns and "master_row_id" not in frame.columns:
        keep = ["sample_id", *[c for c in keep if c != "sample_id"]]
    return frame[keep]


def create_triple_barrier_label_run(
    data_dir: str,
    dataset_name: str,
    *,
    barrier_type: str = "percentage",
    holding_seconds: float | int = 300,
    tp_value: float = 20.0,
    sl_value: float = 10.0,
    truncate_at_close: bool = True,
    max_path_gap_sec: float | None = None,
    run_id: str | None = None,
    parameters: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Label a Feature Dataset with Triple Barrier → write Label Run (no feature mutate)."""
    ensure_builtin_strategies()
    name = str(dataset_name or "").strip()
    if not name:
        raise ValueError("dataset_name is required")

    parquet_path = _feature_parquet_path(data_dir, name)
    if not os.path.isfile(parquet_path):
        raise FileNotFoundError(f"Feature dataset missing: {parquet_path}")

    identity = _read_identity_frame(parquet_path)
    if identity.empty:
        raise ValueError("Feature dataset has no rows to label")

    strategy = get_triple_barrier_strategy()
    params: dict[str, Any] = {
        "barrier_type": str(barrier_type or "percentage"),
        "holding_seconds": int(holding_seconds),
        "tp_value": float(tp_value),
        "sl_value": float(sl_value),
        "truncate_at_close": bool(truncate_at_close),
        "max_path_gap_sec": max_path_gap_sec,
    }
    if parameters:
        params.update({k: v for k, v in parameters.items() if k not in params or v is not None})
    # Drop None max_path_gap for schema defaults.
    if params.get("max_path_gap_sec") is None:
        params.pop("max_path_gap_sec", None)

    config = LabelStrategyConfig(
        strategy_id=TRIPLE_BARRIER_STRATEGY_ID,
        version=STRATEGY_VERSION,
        params=params,
    )

    frames: list[pd.DataFrame] = []
    days = sorted({str(d) for d in identity["trading_day"].tolist() if d is not None and str(d)})
    if not days:
        raise ValueError("Feature dataset has no trading_day values")

    for day in days:
        day_df = identity[identity["trading_day"].astype(str) == day]
        if day_df.empty:
            continue
        samples = _day_samples(day_df)
        source = LabelSourceContext(
            source_kind="prediction",
            day=day,
            handles={"path_source": "feature_sample_grid", "dataset_id": name},
        )
        batch = strategy.build_labels(source, samples, config)
        frames.append(_labeled_to_frame(samples, list(batch.rows)))

    if not frames:
        raise ValueError("Triple Barrier produced no label rows")

    frame = pd.concat(frames, ignore_index=True)
    join_keys = (
        ["master_row_id"]
        if "master_row_id" in frame.columns
        else (
            ["sample_id"]
            if "sample_id" in frame.columns
            else [c for c in ("trading_day", "timestamp", "token") if c in frame.columns]
        )
    )

    meta_params = dict(params)
    meta_params["path_source"] = "feature_sample_grid"
    meta_params["days"] = days

    return write_label_run(
        data_dir,
        frame=frame,
        strategy=TRIPLE_BARRIER_STRATEGY_ID,
        strategy_version=STRATEGY_VERSION,
        engine_version=ENGINE_VERSION,
        dataset_id=name,
        dataset_hash=_dataset_hash(parquet_path),
        parameters=meta_params,
        primary_target="label_id",
        display_target="label_name",
        label_encoding=dict(TRIPLE_BARRIER_LABEL_ENCODING),
        join_keys=join_keys,
        run_id=run_id,
    )

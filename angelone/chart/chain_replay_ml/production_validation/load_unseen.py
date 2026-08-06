"""Load Unseen dataframe for Production Validation Phase B."""

from __future__ import annotations

import json
import os
from typing import Any

import pandas as pd

from chain_replay_ml.dataset_builder.append_ops import load_dataset_metadata
from chain_replay_ml.dataset_builder.writer import read_dataset_parquet
from chain_replay_ml.training.label_prep import adapt_target_for_prediction_type


def _read_json(path: str) -> dict[str, Any]:
    if not path or not os.path.isfile(path):
        return {}
    try:
        with open(path, encoding="utf-8") as fh:
            doc = json.load(fh)
        return doc if isinstance(doc, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def resolve_unseen_parquet_paths(
    data_dir: str,
    dataset_name: str,
    *,
    parquet_path: str | None = None,
    json_path: str | None = None,
) -> tuple[str, str, dict[str, Any]]:
    """Return ``(parquet_path, json_path, meta)`` for a registry unseen dataset."""
    name = str(dataset_name or "").strip()
    if parquet_path and os.path.isfile(parquet_path):
        meta = _read_json(json_path or "") if json_path else {}
        if not meta and name:
            try:
                meta, _ = load_dataset_metadata(data_dir, name)
            except Exception:
                meta = {}
        return os.path.abspath(parquet_path), (
            os.path.abspath(json_path) if json_path and os.path.isfile(json_path) else ""
        ), meta

    if not name:
        raise FileNotFoundError("Unseen dataset_name is required")
    meta, paths = load_dataset_metadata(data_dir, name)
    pq = str(paths.get("parquet") or "")
    js = str(paths.get("json") or "")
    if not pq or not os.path.isfile(pq):
        raise FileNotFoundError(f"Unseen parquet not found for dataset {name!r}")
    return os.path.abspath(pq), (os.path.abspath(js) if js and os.path.isfile(js) else ""), meta


def load_unseen_xy(
    *,
    data_dir: str,
    dataset_name: str,
    features: list[str],
    target: str,
    unseen_days: list[str] | None = None,
    prediction_type: str = "regression",
    parquet_path: str | None = None,
    json_path: str | None = None,
    max_rows: int | None = None,
) -> tuple[pd.DataFrame, pd.Series, dict[str, Any]]:
    """Load all rows for unseen trading day(s); optional ``max_rows`` cap (labeled).

    Uses model selected ``features`` ∩ available columns after verifying critical
    features are present. Target must exist on the unseen parquet.
    """
    pq, js, meta = resolve_unseen_parquet_paths(
        data_dir,
        dataset_name,
        parquet_path=parquet_path,
        json_path=json_path,
    )
    if not target:
        raise ValueError("Model target is required to score Unseen importance")

    wanted = list(dict.fromkeys([*features, target, "trading_day"]))
    # Column projection when engine supports it.
    try:
        import pyarrow.parquet as pq_mod

        schema_names = set(pq_mod.ParquetFile(pq).schema_arrow.names)
        cols = [c for c in wanted if c in schema_names]
        df = pd.read_parquet(pq, columns=cols)
    except Exception:
        df = read_dataset_parquet(pq)
        cols = [c for c in wanted if c in df.columns]
        df = df[cols]

    if target not in df.columns:
        raise ValueError(
            f"Target {target!r} missing from unseen dataset {dataset_name!r}. "
            "Rebuild unseen with prediction targets (Registry + Pipeline path)."
        )

    missing = [f for f in features if f not in df.columns]
    if missing:
        preview = ", ".join(missing[:12])
        more = f" (+{len(missing) - 12} more)" if len(missing) > 12 else ""
        raise ValueError(
            f"{len(missing)} model selected feature(s) missing from unseen dataset "
            f"{dataset_name!r}: {preview}{more}"
        )

    days_filter = [str(d).strip() for d in (unseen_days or []) if str(d).strip()]
    if days_filter and "trading_day" in df.columns:
        day_series = df["trading_day"].astype(str)
        df = df.loc[day_series.isin(set(days_filter))].copy()
    elif days_filter and "trading_day" not in df.columns:
        raise ValueError(
            "Unseen parquet has no trading_day column; cannot restrict to whole unseen day(s)."
        )

    if df.empty:
        raise ValueError(
            f"Unseen dataset {dataset_name!r} has no rows"
            + (f" for days {days_filter}" if days_filter else "")
        )

    capped = False
    full_rows = int(len(df))
    if max_rows is not None and len(df) > int(max_rows):
        df = df.iloc[: int(max_rows)].copy()
        capped = True

    use_features = list(features)
    X = df[use_features].copy()
    y = df[target].copy()
    kind = str(prediction_type or "regression").strip().lower()
    if kind in ("binary", "classification", "classifier"):
        y, _adapt = adapt_target_for_prediction_type(
            y, prediction_type="binary", target=target
        )
    else:
        _adapt = {}

    present_days: list[str] = []
    if "trading_day" in df.columns:
        present_days = sorted({str(d) for d in df["trading_day"].astype(str).tolist() if str(d).strip()})

    load_meta: dict[str, Any] = {
        "dataset_name": dataset_name,
        "parquet_path": pq,
        "json_path": js or None,
        "target": target,
        "feature_count": len(use_features),
        "unseen_rows": int(len(X)),
        "unseen_rows_full": full_rows,
        "unseen_row_cap": int(max_rows) if max_rows is not None else None,
        "unseen_rows_capped": capped,
        "unseen_days_requested": days_filter,
        "unseen_days_present": present_days,
        "unseen_day_count": len(present_days) if present_days else len(days_filter),
        "coverage": "whole_unseen_days" if not capped else "capped_unseen_rows",
        "label_adapt": _adapt,
    }
    return X, y, load_meta

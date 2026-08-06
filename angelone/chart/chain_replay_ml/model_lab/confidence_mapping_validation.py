"""Confidence Dataset ↔ Prediction Dataset mapping validation (read-only).

Picks one random positive sample per RR classifier target and verifies the
join key maps to exactly one Prediction Dataset row with matching RR labels.
Outcome metadata only — never feature values.
"""

from __future__ import annotations

import json
import os
import random
from typing import Any

import pandas as pd

from .confidence_dataset import LABEL_COLUMNS, confidence_dataset_paths
from .rr_dataset_enrich import RR_LABEL_COLUMNS
from .store import ModelLabStore

RR_MAPPING_TARGETS: tuple[dict[str, str], ...] = (
    {"key": "rr_1_1", "column": "rr_1_1_hit", "label": "RR 1:1"},
    {"key": "rr_2_3", "column": "rr_2_3_hit", "label": "RR 2:3"},
    {"key": "rr_1_2", "column": "rr_1_2_hit", "label": "RR 1:2"},
    {"key": "rr_1_3", "column": "rr_1_3_hit", "label": "RR 1:3"},
    {"key": "rr_1_4", "column": "rr_1_4_hit", "label": "RR 1:4"},
)

_OUTCOME_PRED_COLS = (
    "prediction_id",
    "trading_day",
    "timestamp",
    "token",
    "symbol",
    "master_row_id",
    "target_reached",
    "maximum_profit",
    "maximum_drawdown",
    *RR_LABEL_COLUMNS,
)


def _fmt_id(val: Any) -> str | None:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    if isinstance(val, float) and val == int(val):
        return str(int(val))
    return str(val)


def _primary_row_id(row: dict[str, Any] | pd.Series, join_keys: list[str]) -> str | None:
    """Prefer master_row_id, else compact join-key string."""
    mid = None
    if isinstance(row, pd.Series):
        mid = row.get("master_row_id")
        getter = row.get
    else:
        mid = row.get("master_row_id")
        getter = row.get
    mid_s = _fmt_id(mid)
    if mid_s is not None:
        return mid_s
    parts = []
    for k in join_keys:
        v = _fmt_id(getter(k))
        if v is not None:
            parts.append(f"{k}={v}")
    return " · ".join(parts) if parts else None


def _as_int_label(val: Any) -> int | None:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    try:
        return int(val)
    except (TypeError, ValueError):
        return None


def _ratio(profit: Any, drawdown: Any) -> float | None:
    try:
        p = float(profit)
        d = float(drawdown)
    except (TypeError, ValueError):
        return None
    if d == 0 or pd.isna(p) or pd.isna(d):
        return None
    return p / d


def _join_keys_from_meta(meta: dict[str, Any], conf_cols: list[str]) -> list[str]:
    keys = meta.get("join_keys")
    if isinstance(keys, list) and keys:
        return [str(k) for k in keys if str(k) in conf_cols]
    if "master_row_id" in conf_cols:
        return ["master_row_id"]
    natural = ["trading_day", "timestamp", "token"]
    if all(c in conf_cols for c in natural):
        if "symbol" in conf_cols:
            return [*natural, "symbol"]
        return list(natural)
    return []


def _lookup_prediction_rows(
    lab_db_path: str,
    join_keys: list[str],
    key_values: dict[str, Any],
) -> list[dict[str, Any]]:
    with ModelLabStore(lab_db_path) as store:
        store.ensure_prediction_schema()
        avail = set(store._prediction_table_columns())
        select = [c for c in _OUTCOME_PRED_COLS if c in avail]
        for k in join_keys:
            if k not in select and k in avail:
                select.append(k)
        if not select:
            return []
        where_parts = []
        params: list[Any] = []
        for k in join_keys:
            if k not in avail:
                return []
            where_parts.append(f'"{k}" = ?')
            params.append(key_values.get(k))
        if not where_parts:
            return []
        col_sql = ", ".join(f'"{c}"' for c in select)
        where_sql = " AND ".join(where_parts)
        raw = store.conn.execute(
            f"SELECT {col_sql} FROM prediction_dataset WHERE {where_sql}",
            params,
        ).fetchall()
    return [dict(zip(select, row)) for row in raw]


def _sample_positive(
    df: pd.DataFrame,
    column: str,
    rng: random.Random,
) -> pd.Series | None:
    if column not in df.columns:
        return None
    mask = pd.to_numeric(df[column], errors="coerce") == 1
    positives = df.loc[mask]
    if positives.empty:
        return None
    idx = rng.randrange(len(positives))
    return positives.iloc[idx]


def _build_sample_payload(
    *,
    target: dict[str, str],
    conf_row: pd.Series | None,
    join_keys: list[str],
    lab_db_path: str,
) -> dict[str, Any]:
    label = target["label"]
    column = target["column"]
    if conf_row is None:
        return {
            "key": target["key"],
            "column": column,
            "label": label,
            "available": False,
            "message": f"No positive sample available for {label}.",
            "mapping_ok": False,
            "fields": {},
        }

    key_values = {k: conf_row.get(k) for k in join_keys}
    pred_rows = _lookup_prediction_rows(lab_db_path, join_keys, key_values)

    dataset_label = _as_int_label(conf_row.get(column))
    conf_hit = _as_int_label(conf_row.get("target_reached"))

    trading_day = conf_row.get("trading_day")
    timestamp = conf_row.get("timestamp")
    token = conf_row.get("token")

    mapping_ok = False
    pred_label = None
    pred_hit = None
    max_profit = None
    max_dd = None
    issues: list[str] = []

    dataset_row_id = _primary_row_id(conf_row, join_keys)
    prediction_row_id = None
    prediction_id = None

    if len(pred_rows) == 0:
        issues.append("No matching Prediction Dataset row for join key.")
    elif len(pred_rows) > 1:
        issues.append(
            f"Join key matched {len(pred_rows)} Prediction Dataset rows (expected 1)."
        )
    else:
        pred = pred_rows[0]
        mapping_ok = True
        prediction_row_id = _primary_row_id(pred, join_keys)
        prediction_id = _fmt_id(pred.get("prediction_id"))
        pred_label = _as_int_label(pred.get(column))
        pred_hit = _as_int_label(pred.get("target_reached"))
        max_profit = pred.get("maximum_profit")
        max_dd = pred.get("maximum_drawdown")
        # Prefer prediction identity when present
        if pred.get("trading_day") is not None:
            trading_day = pred.get("trading_day")
        if pred.get("timestamp") is not None:
            timestamp = pred.get("timestamp")
        if pred.get("token") is not None:
            token = pred.get("token")
        if pred_hit is None and conf_hit is not None:
            pred_hit = conf_hit
        if dataset_label is not None and pred_label is not None and dataset_label != pred_label:
            mapping_ok = False
            issues.append(
                f"Label mismatch: Confidence Dataset {column}={dataset_label}, "
                f"Prediction Dataset {column}={pred_label}."
            )
        elif pred_label is None:
            mapping_ok = False
            issues.append(f"Prediction Dataset missing {column}.")

    row_ids_match = (
        dataset_row_id is not None
        and prediction_row_id is not None
        and str(dataset_row_id) == str(prediction_row_id)
    )

    ratio = _ratio(max_profit, max_dd)
    fields = {
        "trading_day": trading_day,
        "timestamp": timestamp,
        "token": token,
        "target_hit": pred_hit if pred_hit is not None else conf_hit,
        "maximum_profit": max_profit,
        "maximum_drawdown": max_dd,
        "profit_dd_ratio": ratio,
        "dataset_label": dataset_label,
        "prediction_label": pred_label,
        "dataset_row_id": dataset_row_id,
        "prediction_row_id": prediction_row_id,
        "prediction_id": prediction_id,
        "row_ids_match": row_ids_match,
        "join_key_values": key_values,
    }
    return {
        "key": target["key"],
        "column": column,
        "label": label,
        "available": True,
        "mapping_ok": mapping_ok,
        "message": None if mapping_ok else ("; ".join(issues) or "Mapping failed."),
        "fields": fields,
    }


def validate_confidence_dataset_mapping(
    lab_db_path: str,
    *,
    seed: int | None = None,
) -> dict[str, Any]:
    """
    Sample one random positive row per RR target and verify Prediction Dataset join.

    Returns a validation-view payload for the Confidence Dataset Mapping panel.
    """
    paths = confidence_dataset_paths(lab_db_path)
    parquet = paths["parquet"]
    meta_path = paths["json"]
    if not os.path.isfile(parquet) or not os.path.isfile(meta_path):
        return {
            "ok": False,
            "available": False,
            "error": "Confidence Dataset not found. Build it first.",
            "samples": [],
            "join_keys": [],
        }

    with open(meta_path, encoding="utf-8") as fh:
        meta = json.load(fh)

    # Outcome / identity columns only — never load or return feature values
    from chain_replay_ml.training.dataset_loader import parquet_column_names

    try:
        schema_cols = set(parquet_column_names(parquet))
    except Exception as exc:
        return {
            "ok": False,
            "available": False,
            "error": f"Failed to read Confidence Dataset: {exc}",
            "samples": [],
            "join_keys": [],
        }
    identity = [
        c
        for c in (
            "trading_day",
            "timestamp",
            "token",
            "symbol",
            "master_row_id",
            *LABEL_COLUMNS,
        )
        if c in schema_cols
    ]
    if not identity:
        return {
            "ok": False,
            "available": False,
            "error": "Confidence Dataset has no identity / label columns.",
            "samples": [],
            "join_keys": [],
        }

    df = pd.read_parquet(parquet, columns=identity)
    join_keys = _join_keys_from_meta(meta if isinstance(meta, dict) else {}, list(df.columns))
    if not join_keys:
        return {
            "ok": False,
            "available": False,
            "error": "Cannot resolve join keys for Confidence Dataset mapping.",
            "samples": [],
            "join_keys": [],
        }

    rng = random.Random(seed)
    samples = []
    for target in RR_MAPPING_TARGETS:
        conf_row = _sample_positive(df, target["column"], rng)
        samples.append(
            _build_sample_payload(
                target=target,
                conf_row=conf_row,
                join_keys=join_keys,
                lab_db_path=lab_db_path,
            )
        )

    any_available = any(s.get("available") for s in samples)
    all_mapped = all(
        (not s.get("available")) or s.get("mapping_ok") for s in samples
    )
    return {
        "ok": True,
        "available": any_available,
        "all_mapping_ok": all_mapped and any_available,
        "join_keys": join_keys,
        "samples": samples,
        "seed": seed,
        "title": "Confidence Dataset Mapping Validation",
    }

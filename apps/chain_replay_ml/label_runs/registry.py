"""Label Registry — discover Label Runs from disk (no DB required)."""

from __future__ import annotations

import json
import os
from typing import Any

import pandas as pd

from .paths import label_run_meta_path, label_run_parquet_path, label_runs_dir
from .types import LabelRunRecord


def _read_json(path: str) -> dict[str, Any]:
    if not os.path.isfile(path):
        return {}
    with open(path, encoding="utf-8") as fh:
        doc = json.load(fh)
    return doc if isinstance(doc, dict) else {}


def label_run_exists(data_dir: str, run_id: str) -> bool:
    rid = str(run_id or "").strip()
    if not rid:
        return False
    return os.path.isfile(label_run_parquet_path(data_dir, rid)) and os.path.isfile(
        label_run_meta_path(data_dir, rid)
    )


def load_label_run_meta(data_dir: str, run_id: str) -> dict[str, Any]:
    path = label_run_meta_path(data_dir, str(run_id).strip())
    doc = _read_json(path)
    if not doc:
        raise FileNotFoundError(f"Label Run meta missing: {run_id}")
    return doc


def load_label_run_frame(
    data_dir: str,
    run_id: str,
    *,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    path = label_run_parquet_path(data_dir, str(run_id).strip())
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Label Run parquet missing: {run_id}")
    return pd.read_parquet(path, columns=columns)


def get_label_run(data_dir: str, run_id: str) -> LabelRunRecord:
    rid = str(run_id or "").strip()
    pq = label_run_parquet_path(data_dir, rid)
    meta_path = label_run_meta_path(data_dir, rid)
    exists = os.path.isfile(pq) and os.path.isfile(meta_path)
    meta = _read_json(meta_path) if os.path.isfile(meta_path) else {}
    enc = meta.get("label_encoding")
    return LabelRunRecord(
        run_id=rid or str(meta.get("run_id") or ""),
        strategy=str(meta.get("strategy") or ""),
        strategy_version=str(meta.get("strategy_version") or ""),
        engine_version=str(meta.get("engine_version") or ""),
        dataset_id=str(meta.get("dataset_id") or ""),
        dataset_hash=meta.get("dataset_hash"),
        created_at=str(meta.get("created_at") or ""),
        rows=int(meta.get("rows") or 0),
        valid_rows=int(meta.get("valid_rows") or 0),
        invalid_rows=int(meta.get("invalid_rows") or 0),
        parameters=dict(meta.get("parameters") or {}),
        primary_target=str(meta.get("primary_target") or "label_id"),
        display_target=meta.get("display_target"),
        label_encoding=dict(enc) if isinstance(enc, dict) else None,
        join_keys=list(meta.get("join_keys") or []),
        status="ready" if exists else "deleted",
        parquet_path=pq,
        meta_path=meta_path,
        exists=exists,
    )


def list_label_runs(
    data_dir: str,
    *,
    dataset_id: str | None = None,
    strategy: str | None = None,
) -> list[LabelRunRecord]:
    """Discover ``*_meta.json`` under ``data/label_runs/``."""
    root = label_runs_dir(data_dir)
    if not os.path.isdir(root):
        return []
    rows: list[LabelRunRecord] = []
    for name in sorted(os.listdir(root)):
        if not name.endswith("_meta.json"):
            continue
        run_id = name[: -len("_meta.json")]
        rec = get_label_run(data_dir, run_id)
        if dataset_id and str(rec.dataset_id) != str(dataset_id):
            continue
        if strategy and str(rec.strategy).strip().lower() != str(strategy).strip().lower():
            continue
        rows.append(rec)
    rows.sort(key=lambda r: str(r.created_at or ""), reverse=True)
    return rows

"""Resolve training-dataset selection method for prediction analytics scope."""

from __future__ import annotations

import json
import os
from typing import Any

from chain_replay_ml.dataset_builder.dataset_selection_engine import (
    DatasetSelectionSpec,
    build_selection_sql_where,
)
from chain_replay_ml.dataset_builder.master_registry_export import selection_method_for_registry
from chain_replay_ml.replay_config import load_dataset_metadata_json
from chain_replay_ml.training.paths import model_artifact_paths


def _load_json(path: str) -> dict[str, Any]:
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, encoding="utf-8") as fh:
            doc = json.load(fh)
        return doc if isinstance(doc, dict) else {}
    except (OSError, json.JSONDecodeError, TypeError):
        return {}


def _dataset_name_from_model(data_dir: str, model_name: str) -> str | None:
    paths = model_artifact_paths(data_dir, model_name)
    cfg = _load_json(paths.get("config_json", ""))
    ds = str(cfg.get("dataset") or "").strip()
    if ds:
        return ds
    ds_meta = cfg.get("dataset_metadata") or {}
    if isinstance(ds_meta, dict):
        name = str(ds_meta.get("dataset_name") or "").strip()
        if name:
            return name
    return None


def extract_selection_criteria(meta: dict[str, Any]) -> dict[str, Any]:
    """Normalize premium / ATM / delta filters from dataset metadata."""
    sm = meta.get("selection_method")
    if isinstance(sm, dict) and isinstance(sm.get("criteria"), dict):
        return dict(sm["criteria"])

    mf = meta.get("master_filter")
    if isinstance(mf, dict):
        return dict(mf)

    crit: dict[str, Any] = {}
    ss = meta.get("strike_selection") or {}
    if isinstance(ss, dict) and ss.get("band") is not None:
        try:
            crit["atm_band_filter"] = int(ss["band"])
        except (TypeError, ValueError):
            pass
    return crit


def resolve_training_context(
    data_dir: str,
    model_names: list[str] | None,
) -> dict[str, Any]:
    """Load shared training dataset + selection method for a model set."""
    names = [str(n).strip() for n in (model_names or []) if str(n).strip()]
    if not names or not data_dir:
        return {"resolved": False}

    datasets: dict[str, list[str]] = {}
    for name in names:
        ds = _dataset_name_from_model(data_dir, name)
        if ds:
            datasets.setdefault(ds, []).append(name)

    if not datasets:
        return {"resolved": False, "model_names": names}

    if len(datasets) > 1:
        primary_ds = max(datasets.items(), key=lambda kv: len(kv[1]))[0]
        mixed = True
    else:
        primary_ds = next(iter(datasets))
        mixed = False

    meta = load_dataset_metadata_json(data_dir, primary_ds)
    criteria = extract_selection_criteria(meta)
    summary = selection_method_for_registry(meta)
    if not summary:
        parts: list[str] = []
        band = criteria.get("atm_band_filter")
        if band is not None:
            parts.append(f"ATM ±{band}")
        if criteria.get("premium_enabled") or (
            criteria.get("premium_min") is not None and criteria.get("premium_max") is not None
        ):
            lo = criteria.get("premium_min")
            hi = criteria.get("premium_max")
            parts.append(f"LTP {lo}–{hi}")
        summary = " · ".join(parts) if parts else None

    return {
        "resolved": True,
        "dataset_name": primary_ds,
        "selection_method": summary,
        "selection_source": (
            (meta.get("selection_method") or {}).get("source")
            if isinstance(meta.get("selection_method"), dict)
            else None
        ),
        "criteria": criteria,
        "model_names": names,
        "models_per_dataset": {k: v for k, v in datasets.items()},
        "mixed_datasets": mixed,
        "strike_selection": meta.get("strike_selection"),
        "sampling_interval_sec": (meta.get("sampling") or {}).get("interval_sec"),
    }


def build_training_scope_sql(
    conn: Any,
    criteria: dict[str, Any] | None,
) -> tuple[str, dict[str, Any]]:
    """SQL fragment (no leading AND) restricting samples to training-dataset scope."""
    crit = dict(criteria or {})
    if not crit:
        return "", {"active": False}

    cols = {row[1] for row in conn.execute("PRAGMA table_info(samples)").fetchall()}
    spec = DatasetSelectionSpec.from_registry_criteria(crit)
    sql, _params = build_selection_sql_where(
        spec,
        profile="prediction_meta",
        column_names=cols,
        param_style="inline",
    )
    if sql == "1=1":
        return "", {"active": False, "filters": []}

    info: dict[str, Any] = {"active": True, "filters": []}
    if spec.premium_min is not None and spec.premium_max is not None:
        info["filters"].append(f"premium {spec.premium_min:g}–{spec.premium_max:g}")
        info["premium_min"] = spec.premium_min
        info["premium_max"] = spec.premium_max
    if spec.atm_band is not None:
        info["filters"].append(f"ATM ±{spec.atm_band}")
        info["atm_band_filter"] = spec.atm_band
    if spec.delta_enabled and spec.delta_min is not None and spec.delta_max is not None:
        info["filters"].append(f"|δ| {spec.delta_min:g}–{spec.delta_max:g}")

    return sql, info

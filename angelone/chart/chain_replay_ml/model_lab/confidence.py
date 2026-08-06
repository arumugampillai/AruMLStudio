"""Confidence Model subsystem — lab-owned classifiers for one Research Lab.

Artifacts live under ``{lab_stem}_confidence/`` (dataset + models), not in
Dataset Registry / Model Registry.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any

from .confidence_dataset import (
    CONFIDENCE_DATASET_NAME,
    LABEL_COLUMNS,
    confidence_dataset_paths,
    confidence_legacy_sidecar_path,
    confidence_manifest_path,
    confidence_package_dir,
    create_confidence_dataset,
    resolve_regression_selected_features,
    resolve_training_dataset,
)
from .confidence_mapping_validation import validate_confidence_dataset_mapping
from .confidence_inference import (
    audit_probability_distributions,
    clear_confidence_inference,
    confidence_filter_available,
    dashboard_confidence_filter_options,
    format_probability_distribution_report,
    inference_status,
    run_confidence_inference,
    score_probability_distribution,
    target_hit_filter_available,
)
from .confidence_manifest import (
    CONFIDENCE_TARGETS,
    COLUMN_BY_KEY,
    TARGET_BY_KEY,
    confidence_context,
    delete_confidence_model,
    mark_inference_out_of_date,
    read_manifest,
    set_active_model,
    set_operating_threshold,
    write_manifest,
)
from .confidence_train import evaluate_confidence_model, train_confidence_model
from .confidence_label_builder import (
    assess_label_run_staleness,
    confidence_labels_status,
    list_confidence_label_runs,
    load_replay_outcome_frames,
    read_latest_label_run,
    run_confidence_label_builder,
)
from .store import ModelLabStore
from .target_spec import (
    ALL_TARGET_SPECS,
    MARKET_TARGET_SPECS,
    REPLAY_TARGET_SPECS,
    TARGET_SPEC_BY_KEY,
    derive_binary_labels,
)

# Backward-compatible aliases
CONFIDENCE_TARGET = "target_reached"
CONFIDENCE_TARGET_ALIAS = "hit"


def confidence_sidecar_path(lab_db_path: str) -> str:
    """Legacy path; prefer confidence_manifest_path."""
    return confidence_legacy_sidecar_path(lab_db_path)


def read_confidence_link(lab_db_path: str) -> dict[str, Any] | None:
    """Legacy reader — returns active model summary from the v2 manifest."""
    if not os.path.isfile(lab_db_path) and not os.path.isfile(confidence_manifest_path(lab_db_path)):
        legacy = confidence_legacy_sidecar_path(lab_db_path)
        if not os.path.isfile(legacy):
            return None
    doc = read_manifest(lab_db_path)
    active_key = doc.get("active_model_key") or "target_hit"
    models = doc.get("models") or {}
    active = models.get(active_key) or {}
    ds = doc.get("dataset") or {}
    return {
        "status": active.get("status") or ds.get("status") or "empty",
        "model_name": active.get("legacy_model_name")
        or (f"confidence_{active_key}" if active.get("status") == "ready" else None),
        "dataset_name": ds.get("legacy_dataset_name")
        or (CONFIDENCE_DATASET_NAME if ds.get("status") == "ready" else None),
        "metrics": active.get("metrics") or {},
        "calibration": active.get("calibration") or [],
        "feature_importance": active.get("feature_importance") or [],
        "selected_feature_count": ds.get("feature_count"),
        "training_rows": ds.get("rows"),
        "version": "v2",
        "target": active.get("column") or CONFIDENCE_TARGET,
        "active_model_key": active_key,
        "models": models,
        "dataset": ds,
    }


def write_confidence_link(lab_db_path: str, doc: dict[str, Any]) -> str:
    """Legacy writer — merges into v2 manifest."""
    manifest = read_manifest(lab_db_path)
    if doc.get("dataset_name") or doc.get("training_rows") or doc.get("status"):
        ds = dict(manifest.get("dataset") or {})
        if doc.get("training_rows"):
            ds["rows"] = int(doc["training_rows"])
        if doc.get("status") in ("pending_train", "ready"):
            ds["status"] = "ready" if doc.get("dataset_name") or ds.get("rows") else ds.get("status")
            if doc.get("status") == "pending_train":
                ds["status"] = "ready"
        if doc.get("dataset_name"):
            ds["legacy_dataset_name"] = doc["dataset_name"]
        if doc.get("selected_feature_count"):
            ds["feature_count"] = int(doc["selected_feature_count"])
        manifest["dataset"] = ds
    if doc.get("model_name"):
        entry = dict((manifest.get("models") or {}).get("target_hit") or {})
        entry.update(
            {
                "status": str(doc.get("status") or "ready"),
                "metrics": doc.get("metrics") or entry.get("metrics") or {},
                "feature_importance": doc.get("feature_importance")
                or entry.get("feature_importance")
                or [],
                "legacy_model_name": doc.get("model_name"),
                "active": True,
            }
        )
        manifest.setdefault("models", {})["target_hit"] = entry
        manifest["active_model_key"] = "target_hit"
    write_manifest(lab_db_path, manifest)
    return confidence_legacy_sidecar_path(lab_db_path)


def confidence_status(lab_db_path: str, data_dir: str | None = None) -> dict[str, Any]:
    """UI payload for the Confidence Model tab (context + models)."""
    ctx = confidence_context(lab_db_path, data_dir=data_dir)
    active = next((m for m in ctx["models"] if m.get("active")), None)
    try:
        # Default panel: active model, else Target Hit
        panel_key = (active or {}).get("key") or ctx.get("active_model_key") or "target_hit"
        inf = inference_status(lab_db_path, panel_key)
    except Exception as exc:
        inf = {"status": "not_run", "error": str(exc), "can_run": False}
    try:
        filter_gate = target_hit_filter_available(lab_db_path)
    except Exception:
        filter_gate = {"available": False, "reason": "Inference status unavailable."}
    try:
        filter_options = dashboard_confidence_filter_options(lab_db_path)
    except Exception:
        filter_options = {"classifiers": []}
    return {
        **ctx,
        "status": ctx["confidence_dataset_status"],
        "has_prediction_dataset": ctx["has_prediction_dataset"],
        "model_name": (active or {}).get("label") if active and active.get("status") == "ready" else None,
        "dataset_name": CONFIDENCE_DATASET_NAME
        if ctx["confidence_dataset_status"] in ("ready", "stale")
        else None,
        "selected_feature_count": ctx.get("feature_count"),
        "training_rows": ctx.get("confidence_dataset_rows"),
        "version": "v2",
        "metrics": (active or {}).get("metrics") or {},
        "feature_importance": (active or {}).get("feature_importance") or [],
        "calibration": (active or {}).get("calibration") or [],
        "prediction_type": "binary",
        "target": (active or {}).get("column") or CONFIDENCE_TARGET,
        "operating_threshold": (active or {}).get("operating_threshold"),
        "inference": inf,
        "dashboard_confidence_filter": filter_gate,
        "dashboard_confidence_filters": filter_options,
        "parent_lab_db": lab_db_path,
        "message": None
        if ctx["has_prediction_dataset"]
        else "Build a prediction dataset first.",
    }


def export_hit_confidence_dataset(
    lab_db_path: str,
    data_dir: str,
    *,
    parent_model_name: str,
    lab_version: int | None = None,
) -> dict[str, Any]:
    """
    Backward-compatible entry: create lab-local Confidence Dataset (Seen only).

    ``data_dir`` is used to resolve the parent training export; the Confidence
    Dataset itself is stored under the Research Lab package (not Dataset Registry).
    """
    _ = parent_model_name, lab_version  # resolved from lab
    return create_confidence_dataset(lab_db_path, data_dir=data_dir)


def link_trained_confidence_model(
    lab_db_path: str,
    *,
    model_name: str,
    metrics: dict[str, Any] | None = None,
    feature_importance: list[dict[str, Any]] | None = None,
    selected_feature_count: int | None = None,
    training_rows: int | None = None,
    model_key: str = "target_hit",
) -> dict[str, Any]:
    """Mark a confidence model Ready (legacy / manual link)."""
    if model_key not in TARGET_BY_KEY:
        model_key = "target_hit"
    doc = read_manifest(lab_db_path)
    entry = dict((doc.get("models") or {}).get(model_key) or {})
    entry.update(
        {
            "key": model_key,
            "label": TARGET_BY_KEY[model_key]["label"],
            "column": COLUMN_BY_KEY[model_key],
            "status": "ready",
            "metrics": metrics or entry.get("metrics") or {},
            "feature_importance": feature_importance or entry.get("feature_importance") or [],
            "legacy_model_name": model_name,
            "created_at": entry.get("created_at") or datetime.now(timezone.utc).isoformat(),
            "active": True,
        }
    )
    if selected_feature_count is not None:
        ds = dict(doc.get("dataset") or {})
        ds["feature_count"] = int(selected_feature_count)
        doc["dataset"] = ds
    if training_rows is not None:
        ds = dict(doc.get("dataset") or {})
        ds["rows"] = int(training_rows)
        doc["dataset"] = ds
    for k, m in (doc.get("models") or {}).items():
        m["active"] = k == model_key
    doc.setdefault("models", {})[model_key] = entry
    doc["active_model_key"] = model_key
    write_manifest(lab_db_path, doc)
    return entry


def confidence_band(probability: float) -> str:
    p = float(probability)
    if p >= 0.90:
        return "Very High"
    if p >= 0.75:
        return "High"
    if p >= 0.60:
        return "Medium"
    if p >= 0.50:
        return "Low"
    return "Very Low"


def compute_calibration_bins(
    y_true: list[int] | Any,
    y_prob: list[float] | Any,
) -> list[dict[str, Any]]:
    """Predicted-probability buckets → actual hit rate."""
    import numpy as np

    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_prob, dtype=float)
    mask = np.isfinite(yt) & np.isfinite(yp)
    yt, yp = yt[mask], yp[mask]
    bins = [
        ("<50%", None, 0.5),
        ("50–60%", 0.5, 0.6),
        ("60–75%", 0.6, 0.75),
        ("75–90%", 0.75, 0.9),
        (">90%", 0.9, 1.0001),
    ]
    cleaned: list[dict[str, Any]] = []
    for label, a, b in bins:
        if a is None:
            band_mask = yp < b
        else:
            band_mask = (yp >= a) & (yp < b)
        n = int(band_mask.sum())
        hit = float(yt[band_mask].mean()) if n else None
        cleaned.append(
            {
                "band": label,
                "rows": n,
                "actual_hit_rate": hit,
                "actual_hit_pct": (100.0 * hit) if hit is not None else None,
            }
        )
    return cleaned


__all__ = [
    "CONFIDENCE_DATASET_NAME",
    "CONFIDENCE_TARGET",
    "CONFIDENCE_TARGET_ALIAS",
    "CONFIDENCE_TARGETS",
    "LABEL_COLUMNS",
    "COLUMN_BY_KEY",
    "TARGET_BY_KEY",
    "clear_confidence_inference",
    "confidence_band",
    "confidence_context",
    "confidence_dataset_paths",
    "confidence_filter_available",
    "confidence_legacy_sidecar_path",
    "confidence_manifest_path",
    "confidence_package_dir",
    "confidence_sidecar_path",
    "confidence_status",
    "compute_calibration_bins",
    "create_confidence_dataset",
    "dashboard_confidence_filter_options",
    "audit_probability_distributions",
    "format_probability_distribution_report",
    "score_probability_distribution",
    "delete_confidence_model",
    "evaluate_confidence_model",
    "export_hit_confidence_dataset",
    "inference_status",
    "link_trained_confidence_model",
    "mark_inference_out_of_date",
    "read_confidence_link",
    "read_manifest",
    "resolve_regression_selected_features",
    "resolve_training_dataset",
    "run_confidence_inference",
    "run_confidence_label_builder",
    "load_replay_outcome_frames",
    "read_latest_label_run",
    "list_confidence_label_runs",
    "assess_label_run_staleness",
    "confidence_labels_status",
    "derive_binary_labels",
    "ALL_TARGET_SPECS",
    "MARKET_TARGET_SPECS",
    "REPLAY_TARGET_SPECS",
    "TARGET_SPEC_BY_KEY",
    "set_active_model",
    "set_operating_threshold",
    "target_hit_filter_available",
    "train_confidence_model",
    "validate_confidence_dataset_mapping",
    "write_confidence_link",
    "write_manifest",
]

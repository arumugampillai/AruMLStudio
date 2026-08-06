"""Confidence Model manifest — lab-local multi-classifier registry."""

from __future__ import annotations

import json
import os
import shutil
from datetime import datetime, timezone
from typing import Any

from .confidence_dataset import (
    confidence_dataset_paths,
    confidence_legacy_sidecar_path,
    confidence_manifest_path,
    confidence_package_dir,
    resolve_training_dataset,
)
from .service import load_lab
from .store import ModelLabStore

# Canonical confidence classifiers — Market Outcomes + Replay-Based Outcomes.
# Source of truth: model_lab.target_spec (TargetSpec registry).
from .target_spec import confidence_targets_for_manifest

CONFIDENCE_TARGETS: tuple[dict[str, str], ...] = confidence_targets_for_manifest()

TARGET_BY_KEY = {t["key"]: t for t in CONFIDENCE_TARGETS}
COLUMN_BY_KEY = {t["key"]: t["column"] for t in CONFIDENCE_TARGETS}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _empty_model_entry(key: str) -> dict[str, Any]:
    spec = TARGET_BY_KEY[key]
    return {
        "key": key,
        "label": spec["label"],
        "column": spec["column"],
        "status": "not_created",
        "metrics": {},
        "created_at": None,
        "package_dir": None,
        "active": False,
        # Operating threshold is required for inference — set after Threshold Analysis
        "operating_threshold": None,
    }


def default_manifest() -> dict[str, Any]:
    return {
        "schema_version": 4,
        "regression_model": None,
        "training_dataset": None,
        "prediction_lab_version": None,
        "prediction_build_timestamp": None,
        "prediction_rows": 0,
        "dataset": {
            "status": "not_created",
            "rows": 0,
            "rr_labels": {},
            "feature_count": 0,
            "created_at": None,
            "parquet": None,
            "json": None,
        },
        "models": {t["key"]: _empty_model_entry(t["key"]) for t in CONFIDENCE_TARGETS},
        "active_model_key": None,
        # Per-model inference over Prediction Dataset
        "inference": {
            t["key"]: {
                "status": "not_run",  # not_run | completed | out_of_date | running | failed
                "rows": 0,
                "positive": 0,
                "negative": 0,
                "nulls": 0,
                "model_key": None,
                "model_id": None,
                "threshold": None,
                "completed_at": None,
                "validation": None,
                "error": None,
            }
            for t in CONFIDENCE_TARGETS
        },
        "updated_at": None,
    }


def read_manifest(lab_db_path: str) -> dict[str, Any]:
    path = confidence_manifest_path(lab_db_path)
    doc: dict[str, Any] | None = None
    if os.path.isfile(path):
        try:
            with open(path, encoding="utf-8") as fh:
                loaded = json.load(fh)
            if isinstance(loaded, dict):
                doc = loaded
        except (OSError, json.JSONDecodeError):
            doc = None

    if doc is None:
        # Migrate legacy single-model sidecar
        legacy = confidence_legacy_sidecar_path(lab_db_path)
        if os.path.isfile(legacy):
            try:
                with open(legacy, encoding="utf-8") as fh:
                    old = json.load(fh)
                if isinstance(old, dict):
                    doc = _migrate_legacy(old)
            except (OSError, json.JSONDecodeError):
                doc = None

    base = default_manifest()
    if not doc:
        return base
    # Merge defaults so new target keys appear
    out = {**base, **doc}
    out["dataset"] = {**base["dataset"], **(doc.get("dataset") or {})}
    base_inf = base.get("inference") or {}
    doc_inf = doc.get("inference") if isinstance(doc.get("inference"), dict) else {}
    merged_inf = dict(base_inf)
    for ik, iv in doc_inf.items():
        if ik in merged_inf and isinstance(iv, dict) and isinstance(merged_inf[ik], dict):
            merged_inf[ik] = {**merged_inf[ik], **iv}
        else:
            merged_inf[ik] = iv
    out["inference"] = merged_inf
    models = dict(base["models"])
    for key, entry in (doc.get("models") or {}).items():
        if key in models and isinstance(entry, dict):
            models[key] = {**models[key], **entry}
    # Legacy: single model_name → target_hit
    if not doc.get("models") and doc.get("model_name"):
        models["target_hit"] = {
            **models["target_hit"],
            "status": str(doc.get("status") or "ready"),
            "metrics": doc.get("metrics") or {},
            "created_at": doc.get("updated_at"),
            "package_dir": doc.get("model_package_dir"),
            "active": True,
            "legacy_model_name": doc.get("model_name"),
        }
        out["active_model_key"] = "target_hit"
    out["models"] = models
    return out


def write_manifest(lab_db_path: str, doc: dict[str, Any]) -> str:
    paths = confidence_dataset_paths(lab_db_path)
    os.makedirs(paths["package_dir"], exist_ok=True)
    payload = dict(doc)
    payload["updated_at"] = _utc_now()
    path = paths["manifest"]
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
        fh.write("\n")
    # Keep legacy sidecar in sync for older readers
    legacy = {
        "schema_version": 2,
        "status": (payload.get("dataset") or {}).get("status"),
        "dataset_name": "confidence_dataset",
        "active_model_key": payload.get("active_model_key"),
        "models": payload.get("models"),
        "updated_at": payload["updated_at"],
    }
    try:
        with open(confidence_legacy_sidecar_path(lab_db_path), "w", encoding="utf-8") as fh:
            json.dump(legacy, fh, indent=2, ensure_ascii=False)
            fh.write("\n")
    except OSError:
        pass
    return path


def _migrate_legacy(old: dict[str, Any]) -> dict[str, Any]:
    doc = default_manifest()
    doc["dataset"]["status"] = (
        "ready"
        if old.get("dataset_name") or old.get("export_parquet")
        else "not_created"
    )
    if old.get("training_rows"):
        doc["dataset"]["rows"] = int(old["training_rows"])
    if old.get("dataset_name"):
        doc["dataset"]["legacy_dataset_name"] = old["dataset_name"]
    if old.get("model_name"):
        doc["models"]["target_hit"] = {
            **doc["models"]["target_hit"],
            "status": str(old.get("status") or "ready"),
            "metrics": old.get("metrics") or {},
            "legacy_model_name": old.get("model_name"),
            "active": True,
        }
        doc["active_model_key"] = "target_hit"
    return doc


def update_manifest_after_dataset(
    lab_db_path: str,
    *,
    dataset_meta: dict[str, Any],
    paths: dict[str, str],
    report: dict[str, Any],
    type_stats: dict[str, Any],
) -> dict[str, Any]:
    lab = load_lab(lab_db_path)
    with ModelLabStore(lab_db_path) as store:
        summary = store.read_prediction_summary() or {}
        pred_n = int(store.prediction_row_count() or 0)

    doc = read_manifest(lab_db_path)
    labels = dataset_meta.get("classifier_labels") or {}
    doc["regression_model"] = (lab.parent_model_name if lab else None) or summary.get(
        "parent_model_name"
    )
    doc["regression_model_checksum"] = (
        (lab.model_checksum if lab else None) or None
    )
    doc["training_dataset"] = dataset_meta.get("training_dataset")
    doc["prediction_lab_version"] = int(lab.version) if lab else None
    doc["prediction_build_timestamp"] = summary.get("created_at")
    doc["prediction_rows"] = pred_n
    # Seen trading days from day catalog
    seen_days = 0
    if lab is not None:
        try:
            with ModelLabStore(lab_db_path) as store:
                store.ensure_prediction_schema()
                lab_uuid = str(summary.get("lab_uuid") or "").strip()
                if lab_uuid:
                    from .prediction_schema import DATASET_TYPE_SEEN

                    days = store.list_build_days(lab_uuid)
                    seen_days = sum(
                        1
                        for d in days
                        if str(d.get("dataset_type") or "") == DATASET_TYPE_SEEN
                    )
        except Exception:
            seen_days = 0
    doc["dataset"] = {
        "status": "ready",
        "rows": int(dataset_meta.get("row_count") or 0),
        "rr_labels": {
            "rr_1_1_hit": bool(labels.get("rr_1_1_hit")),
            "rr_2_3_hit": bool(labels.get("rr_2_3_hit")),
            "rr_1_2_hit": bool(labels.get("rr_1_2_hit")),
            "rr_1_3_hit": bool(labels.get("rr_1_3_hit")),
            "rr_1_4_hit": bool(labels.get("rr_1_4_hit")),
            "target_reached": bool(labels.get("target_reached")),
        },
        "feature_count": int(dataset_meta.get("feature_count") or 0),
        "feature_source": dataset_meta.get("feature_source") or "regression_model",
        "feature_source_detail": dataset_meta.get("feature_source_detail"),
        "created_at": dataset_meta.get("created_at"),
        "parquet": paths.get("parquet"),
        "json": paths.get("json"),
        "seen_prediction_rows": type_stats.get("seen_prediction_rows"),
        "unseen_ignored": type_stats.get("unseen_ignored"),
        "matched": report.get("matched"),
        "join_keys": report.get("join_keys"),
        "seen_trading_days": seen_days,
        "dropped_unmatched": report.get("dropped_unmatched") or 0,
    }
    # Invalidate models when dataset is rebuilt
    for key in doc["models"]:
        entry = dict(doc["models"][key])
        if entry.get("status") == "ready":
            entry["status"] = "stale"
            entry["stale_reason"] = "Confidence Dataset rebuilt"
        doc["models"][key] = entry
    write_manifest(lab_db_path, doc)
    return doc


def _display_model_status(raw: str | None, *, is_legacy: bool = False) -> str:
    if is_legacy and str(raw or "").strip().lower() in ("ready", "stale"):
        return "Legacy"
    s = str(raw or "not_created").strip().lower()
    if s in ("ready",):
        return "Ready"
    if s in ("training", "pending_train"):
        return "Training"
    if s in ("failed", "error"):
        return "Failed"
    if s in ("stale",):
        return "Stale"
    if s in ("not_created", "empty", "missing", ""):
        return "Missing"
    return s.replace("_", " ").title()


def confidence_context(lab_db_path: str, data_dir: str | None = None) -> dict[str, Any]:
    """Section 1 payload for the Confidence Model tab."""
    lab = load_lab(lab_db_path)
    manifest = read_manifest(lab_db_path)
    paths = confidence_dataset_paths(lab_db_path)

    with ModelLabStore(lab_db_path) as store:
        store.ensure_prediction_schema()
        summary = store.read_prediction_summary() or {}
        pred_n = int(store.prediction_row_count() or 0)
        cols = store._prediction_table_columns()
        lab_uuid = str(summary.get("lab_uuid") or "").strip()
        seen_days_live = 0
        if lab_uuid:
            from .prediction_schema import DATASET_TYPE_SEEN

            days = store.list_build_days(lab_uuid)
            seen_days_live = sum(
                1 for d in days if str(d.get("dataset_type") or "") == DATASET_TYPE_SEEN
            )

    training_name = manifest.get("training_dataset")
    if not training_name and lab is not None:
        try:
            resolved = resolve_training_dataset(lab_db_path, data_dir=data_dir)
            training_name = resolved["dataset_name"]
        except Exception:
            training_name = summary.get("parent_dataset")

    rr_available = all(
        c in cols
        for c in ("rr_1_1_hit", "rr_2_3_hit", "rr_1_2_hit", "rr_1_3_hit", "rr_1_4_hit")
    )
    hit_available = "target_reached" in cols
    ds = manifest.get("dataset") or {}
    ds_ready = (
        str(ds.get("status") or "") == "ready"
        and os.path.isfile(str(ds.get("parquet") or paths["parquet"]))
    )

    # Regression model changed → confidence artifacts stale
    current_checksum = (lab.model_checksum if lab else None) or None
    stored_checksum = manifest.get("regression_model_checksum")
    regression_stale = bool(
        ds_ready
        and stored_checksum
        and current_checksum
        and str(stored_checksum) != str(current_checksum)
    )
    if regression_stale:
        ds_status = "stale"
    elif ds_ready:
        ds_status = "ready"
    else:
        ds_status = "not_created"

    labels = ds.get("rr_labels") or {}
    label_count = 0
    if ds_ready:
        for k in (
            "target_reached",
            "rr_1_1_hit",
            "rr_2_3_hit",
            "rr_1_2_hit",
            "rr_1_3_hit",
            "rr_1_4_hit",
        ):
            if labels.get(k):
                label_count += 1
    elif rr_available:
        label_count = 5 + (1 if hit_available else 0)

    models_out = []
    from .confidence_train import model_has_threshold_analysis

    for spec in CONFIDENCE_TARGETS:
        entry = (manifest.get("models") or {}).get(spec["key"]) or _empty_model_entry(spec["key"])
        metrics = entry.get("metrics") if isinstance(entry.get("metrics"), dict) else {}
        raw_status = entry.get("status") or "not_created"
        if regression_stale and raw_status == "ready":
            raw_status = "stale"
        is_legacy = False
        if raw_status in ("ready", "stale"):
            is_legacy = not model_has_threshold_analysis(
                lab_db_path, spec["key"], entry=entry
            )
        models_out.append(
            {
                "key": spec["key"],
                "label": spec["label"],
                "type": spec["label"],
                "column": spec["column"],
                "status": raw_status,
                "status_display": _display_model_status(raw_status, is_legacy=is_legacy),
                "is_legacy": is_legacy,
                "has_threshold_analysis": not is_legacy
                if raw_status in ("ready", "stale")
                else False,
                "accuracy_pct": metrics.get("accuracy_pct"),
                "precision_pct": metrics.get("precision_pct"),
                "recall_pct": metrics.get("recall_pct"),
                "f1_pct": metrics.get("f1_pct"),
                "roc_auc": metrics.get("roc_auc"),
                "pr_auc": metrics.get("pr_auc"),
                "created_at": entry.get("created_at"),
                "operating_threshold": entry.get("operating_threshold"),
                "active": bool(entry.get("active"))
                or manifest.get("active_model_key") == spec["key"],
                "metrics": metrics,
                "calibration": entry.get("calibration") or [],
                "confusion": (metrics.get("confusion") if isinstance(metrics, dict) else None),
            }
        )

    created_at = ds.get("created_at") if ds_ready else None
    created_display = None
    if created_at:
        created_display = str(created_at)[:19].replace("T", " ")

    seen_days = int(ds.get("seen_trading_days") or 0) if ds_ready else seen_days_live
    if not seen_days:
        seen_days = seen_days_live

    # Inherited regression feature space (informational; not user-editable)
    from .confidence_dataset import resolve_regression_selected_features

    feat_resolve = resolve_regression_selected_features(lab_db_path, lab=lab)
    if feat_resolve.get("ok"):
        feature_source = "Regression Model"
        feature_count = int(feat_resolve.get("feature_count") or len(feat_resolve["features"]))
    elif ds_ready and ds.get("feature_count"):
        feature_source = (
            "Regression Model"
            if str(ds.get("feature_source") or "") == "regression_model"
            else str(ds.get("feature_source") or "—")
        )
        feature_count = int(ds.get("feature_count") or 0)
    else:
        feature_source = "—"
        feature_count = 0

    pipeline = {
        "regression_model": {
            "ok": bool(lab and lab.parent_model_name),
            "label": (lab.parent_model_name if lab else None)
            or manifest.get("regression_model")
            or "—",
        },
        "prediction_dataset": {
            "ok": pred_n > 0,
            "rows": pred_n,
        },
        "confidence_dataset": {
            "ok": ds_status == "ready",
            "stale": ds_status == "stale",
            "rows": int(ds.get("rows") or 0) if ds_ready else 0,
            "status": ds_status,
        },
        "confidence_models": [
            {
                "key": m["key"],
                "label": m["label"],
                "ok": m["status"] == "ready",
                "status": m["status_display"],
            }
            for m in models_out
        ],
    }

    return {
        "regression_model": pipeline["regression_model"]["label"],
        "training_dataset": training_name,
        "prediction_rows": pred_n,
        "prediction_lab_version": int(lab.version) if lab else manifest.get("prediction_lab_version"),
        "prediction_build_timestamp": summary.get("created_at")
        or manifest.get("prediction_build_timestamp"),
        "confidence_dataset_status": ds_status,
        "confidence_dataset_rows": int(ds.get("rows") or 0) if ds_ready else 0,
        "confidence_dataset_created_at": created_display,
        "seen_trading_days": seen_days,
        "rr_labels_available": rr_available,
        "rr_labels": labels,
        "available_classifier_labels": label_count,
        "feature_source": feature_source,
        "feature_count": feature_count,
        "selected_features": feature_count,
        "has_prediction_dataset": pred_n > 0,
        "package_dir": paths["package_dir"],
        "models": models_out,
        "active_model_key": manifest.get("active_model_key"),
        "pipeline": pipeline,
        "regression_stale": regression_stale,
        "manifest": manifest,
    }


def set_active_model(lab_db_path: str, model_key: str) -> dict[str, Any]:
    if model_key not in TARGET_BY_KEY:
        raise ValueError(f"Unknown confidence model key: {model_key}")
    doc = read_manifest(lab_db_path)
    entry = (doc.get("models") or {}).get(model_key) or {}
    if str(entry.get("status") or "") not in ("ready", "stale"):
        raise ValueError(f"Model {model_key} is not trained.")
    prev = doc.get("active_model_key")
    for key, model in (doc.get("models") or {}).items():
        model["active"] = key == model_key
    doc["active_model_key"] = model_key
    # Active link does not invalidate per-model inference artifacts
    _ = prev
    write_manifest(lab_db_path, doc)
    return doc


def set_operating_threshold(
    lab_db_path: str,
    model_key: str,
    threshold: float,
) -> dict[str, Any]:
    """
    Persist Operating Threshold on a Confidence Model (deployable artifact).

    Changing the threshold marks that model's inference Out of Date.
    """
    if model_key not in TARGET_BY_KEY:
        raise ValueError(f"Unknown confidence model key: {model_key}")
    thr = float(threshold)
    if thr != thr or thr < 0.0 or thr > 1.0:
        raise ValueError(f"Operating threshold must be in [0, 1], got {threshold}")

    doc = read_manifest(lab_db_path)
    entry = (doc.get("models") or {}).get(model_key) or {}
    if str(entry.get("status") or "") not in ("ready", "stale"):
        raise ValueError(f"Model {model_key} is not trained.")

    from .confidence_train import model_has_threshold_analysis

    if not model_has_threshold_analysis(lab_db_path, model_key, entry=entry):
        raise ValueError(
            f"{TARGET_BY_KEY[model_key]['label']} is Legacy (no Threshold Analysis). "
            "Retrain the model, then select a row and Save Operating Threshold."
        )

    prev = entry.get("operating_threshold")
    entry["operating_threshold"] = round(thr, 4)
    doc.setdefault("models", {})[model_key] = entry

    # Mirror into metrics.json when package exists
    pkg = entry.get("package_dir") or model_package_dir_for(lab_db_path, model_key)
    metrics_path = os.path.join(pkg, "metrics.json") if pkg else None
    if metrics_path and os.path.isfile(metrics_path):
        try:
            with open(metrics_path, encoding="utf-8") as fh:
                meta = json.load(fh)
            if isinstance(meta, dict):
                meta["operating_threshold"] = entry["operating_threshold"]
                with open(metrics_path, "w", encoding="utf-8") as fh:
                    json.dump(meta, fh, indent=2)
                    fh.write("\n")
        except (OSError, json.JSONDecodeError):
            pass

    changed = prev is None or abs(float(prev) - thr) > 1e-9
    if changed:
        _mark_model_inference_stale(
            doc, model_key, reason=f"Operating threshold changed → {thr:.2f}"
        )
    write_manifest(lab_db_path, doc)
    return {
        "ok": True,
        "model_key": model_key,
        "operating_threshold": entry["operating_threshold"],
        "inference_stale": changed,
        "manifest": doc,
    }


def mark_inference_out_of_date(
    lab_db_path: str,
    *,
    reason: str | None = None,
    model_key: str = "target_hit",
) -> dict[str, Any]:
    """Mark stored confidence inference Out of Date (Prediction rebuild / retrain)."""
    doc = read_manifest(lab_db_path)
    if model_key == "*":
        for key in TARGET_BY_KEY:
            _mark_model_inference_stale(doc, key, reason=reason)
    elif model_key in TARGET_BY_KEY:
        _mark_model_inference_stale(doc, model_key, reason=reason)
    write_manifest(lab_db_path, doc)
    return doc


def _mark_model_inference_stale(
    doc: dict[str, Any],
    model_key: str,
    *,
    reason: str | None = None,
) -> None:
    inf = doc.setdefault("inference", {}).setdefault(model_key, {})
    prev = str(inf.get("status") or "not_run")
    if prev in ("completed", "out_of_date", "failed", "running"):
        inf["status"] = "out_of_date"
        if reason:
            inf["stale_reason"] = reason


def _mark_target_hit_inference_stale(
    doc: dict[str, Any],
    *,
    reason: str | None = None,
) -> None:
    """Backward-compatible alias."""
    _mark_model_inference_stale(doc, "target_hit", reason=reason)


def delete_confidence_model(lab_db_path: str, model_key: str) -> dict[str, Any]:
    if model_key not in TARGET_BY_KEY:
        raise ValueError(f"Unknown confidence model key: {model_key}")
    doc = read_manifest(lab_db_path)
    entry = (doc.get("models") or {}).get(model_key) or {}
    pkg = entry.get("package_dir")
    if pkg and os.path.isdir(pkg):
        shutil.rmtree(pkg, ignore_errors=True)
    doc["models"][model_key] = _empty_model_entry(model_key)
    if doc.get("active_model_key") == model_key:
        doc["active_model_key"] = None
    _mark_model_inference_stale(doc, model_key, reason="Confidence Model deleted")
    # Reset inference block for deleted model
    doc.setdefault("inference", {})[model_key] = {
        "status": "not_run",
        "rows": 0,
        "positive": 0,
        "negative": 0,
        "nulls": 0,
        "model_key": None,
        "model_id": None,
        "threshold": None,
        "completed_at": None,
        "validation": None,
        "error": None,
    }
    write_manifest(lab_db_path, doc)
    return doc


def model_package_dir_for(lab_db_path: str, model_key: str) -> str:
    return os.path.join(confidence_dataset_paths(lab_db_path)["models_dir"], model_key)

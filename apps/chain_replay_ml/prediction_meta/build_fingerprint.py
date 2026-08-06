"""Build fingerprint — immutable identity card for each prediction dataset project."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from typing import Any, Mapping


def _project_val(project: Mapping[str, Any] | Any, key: str, default: Any = None) -> Any:
    if isinstance(project, Mapping):
        return project.get(key, default)
    return getattr(project, key, default)


def _fmt_date_short(iso: str | None) -> str:
    if not iso:
        return "—"
    try:
        dt = datetime.fromisoformat(str(iso).replace("Z", "+00:00"))
        return dt.strftime("%Y-%m-%d")
    except (TypeError, ValueError):
        text = str(iso)
        return text[:10] if len(text) >= 10 else text


def _inference_registry_display(signature: str | None) -> str:
    """Short display label for the inference model registry generation."""
    sig = str(signature or "").strip()
    if not sig:
        return "—"
    parts = [p for p in sig.split("|") if p]
    if len(parts) >= 2:
        try:
            return str(len(parts))
        except (TypeError, ValueError):
            pass
    digest = hashlib.sha256(sig.encode("utf-8")).hexdigest()
    return str(int(digest[:4], 16) % 1000)


def _fingerprint_id(payload: dict[str, Any]) -> str:
    """Stable id for comparing fingerprints (excludes volatile build progress fields)."""
    stable_keys = (
        "prediction_dataset",
        "master_dataset",
        "market",
        "sampling_interval_sec",
        "trading_days_filter",
        "selected_models",
        "feature_version",
        "prediction_version",
        "model_registry_version",
        "inference_registry_signature",
    )
    stable = {k: payload[k] for k in stable_keys if k in payload}
    blob = json.dumps(stable, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]


def compute_build_fingerprint(
    project: Mapping[str, Any] | Any,
    *,
    master_row_count: int | None = None,
    rows_planned: int | None = None,
    prediction_row_count: int | None = None,
    prediction_version: int | None = None,
    model_registry_version: str | None = None,
    inference_registry_signature: str | None = None,
    model_registry_slot_count: int | None = None,
    build_status: str = "pending",
    completed_at: str | None = None,
) -> dict[str, Any]:
    """Assemble the human-readable build fingerprint for a prediction project."""
    days_filter = list(_project_val(project, "trading_days_filter") or [])
    trading_days_count = len(days_filter) if days_filter else None
    models = list(_project_val(project, "selected_models") or [])
    feat_ver = _project_val(project, "feature_version")
    rows = prediction_row_count
    if rows is None and rows_planned is not None:
        rows = rows_planned
    if rows is None:
        rows = master_row_count

    registry_display = (
        str(model_registry_slot_count)
        if model_registry_slot_count is not None
        else (_inference_registry_display(inference_registry_signature) if inference_registry_signature else "—")
    )

    fp: dict[str, Any] = {
        "prediction_dataset": _project_val(project, "display_name"),
        "master_dataset": _project_val(project, "source_master_db"),
        "market": _project_val(project, "market"),
        "sampling_interval_sec": int(_project_val(project, "sampling_interval_sec") or 3),
        "trading_days_count": trading_days_count,
        "trading_days_filter": days_filter or None,
        "rows": rows,
        "master_rows_at_build": master_row_count,
        "rows_planned": rows_planned,
        "rows_written": prediction_row_count,
        "models_count": len(models),
        "selected_models": models,
        "feature_version": feat_ver,
        "prediction_version": prediction_version,
        "model_registry_version": model_registry_version,
        "model_registry_version_display": registry_display,
        "inference_registry_signature": inference_registry_signature,
        "created_at": _project_val(project, "created_at"),
        "created_label": _fmt_date_short(_project_val(project, "created_at")),
        "completed_at": completed_at or _project_val(project, "last_build_finished_at"),
        "completed_label": _fmt_date_short(completed_at or _project_val(project, "last_build_finished_at")),
        "build_status": str(build_status or "pending"),
        "cloned_from": _project_val(project, "cloned_from"),
        "project_id": _project_val(project, "project_id"),
        "db_filename": _project_val(project, "db_filename"),
    }
    fp["fingerprint_id"] = _fingerprint_id(fp)
    return fp


def merge_fingerprint(
    existing: dict[str, Any] | None,
    updates: dict[str, Any],
) -> dict[str, Any]:
    merged = dict(existing or {})
    merged.update(updates)
    merged["fingerprint_id"] = _fingerprint_id(merged)
    if merged.get("created_at"):
        merged["created_label"] = _fmt_date_short(merged.get("created_at"))
    if merged.get("completed_at"):
        merged["completed_label"] = _fmt_date_short(merged.get("completed_at"))
    return merged

"""Create / open Model Lab workspaces (Phase 1)."""

from __future__ import annotations

import os
import uuid
from typing import Any

from .paths import (
    iter_all_lab_db_paths,
    lab_db_filename,
    lab_db_path,
    lab_db_stem,
    latest_lab_path,
    list_lab_db_paths,
    next_lab_version,
    resolve_model_research_dir,
)
from chain_replay_ml.training.paths import safe_model_name
from .snapshots import build_lab_snapshots
from .store import (
    LAB_PHASE,
    LAB_SCHEMA_VERSION,
    STATUS_READY,
    ModelLabInfo,
    ModelLabStore,
)


def refresh_artifact_availability(pointers: dict[str, Any] | None) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, item in (pointers or {}).items():
        if not isinstance(item, dict):
            out[key] = item
            continue
        path = str(item.get("path") or "")
        kind = str(item.get("kind") or "file")
        if kind == "directory":
            available = bool(path and os.path.isdir(path))
        else:
            available = bool(path and os.path.isfile(path))
        out[key] = {
            **item,
            "available": available,
            "status": "available" if available else "unavailable",
        }
    return out


def load_lab(db_path: str) -> ModelLabInfo | None:
    if not os.path.isfile(db_path):
        return None
    with ModelLabStore(db_path) as store:
        info = store.read_info()
        pred_summary = store.read_prediction_summary() if info else None
        pred_n = store.prediction_row_count() if info else 0
    if info is None:
        return None
    if info.artifact_pointers:
        info.artifact_pointers = refresh_artifact_availability(info.artifact_pointers)
    # Attach live prediction status for Overview / Prediction tab
    if pred_summary:
        status = str(pred_summary.get("status") or "")
        info._prediction_overview = {  # type: ignore[attr-defined]
            "prediction_dataset_status": status or ("ready" if pred_n else "not_generated"),
            "prediction_row_count": int(pred_summary.get("row_count") or pred_n),
            "prediction_trading_days": pred_summary.get("trading_days"),
            "prediction_created_at": pred_summary.get("created_at"),
            "prediction_dataset_hash": pred_summary.get("dataset_hash"),
            "prediction_direction_accuracy": pred_summary.get("direction_accuracy"),
            "prediction_average_error": pred_summary.get("average_error"),
            "prediction_premium_error": pred_summary.get("premium_error"),
        }
    else:
        info._prediction_overview = {  # type: ignore[attr-defined]
            "prediction_dataset_status": "ready" if pred_n else "not_generated",
            "prediction_row_count": pred_n,
        }
    return info


def _parent_names_match(stored: str | None, model_name: str) -> bool:
    a = safe_model_name(str(stored or "").strip())
    b = safe_model_name(str(model_name or "").strip())
    return bool(a and b and a == b)


def find_latest_lab(model_name: str, research_dir: str | None = None) -> ModelLabInfo | None:
    """Resolve the newest lab for *model_name*.

    Prefer filename stem match (`model_lab_<safe>_vN.db`). If that misses
    (renamed package / stem drift), fall back to scanning lab DBs by
    ``parent_model_name``.
    """
    latest = latest_lab_path(model_name, research_dir=research_dir)
    if latest:
        info = load_lab(latest[1])
        if info is not None:
            return info

    # Fallback: match by parent_model_name inside each lab DB
    best: ModelLabInfo | None = None
    for path in iter_all_lab_db_paths(research_dir=research_dir):
        info = load_lab(path)
        if info is None:
            continue
        if not _parent_names_match(info.parent_model_name, model_name):
            continue
        if best is None or int(info.version or 0) >= int(best.version or 0):
            best = info
    return best


def list_research_lab_summaries(research_dir: str | None = None) -> list[dict[str, Any]]:
    """Lightweight inventory of labs on disk (for empty-state hints)."""
    out: list[dict[str, Any]] = []
    for path in iter_all_lab_db_paths(research_dir=research_dir):
        info = load_lab(path)
        if info is None:
            continue
        ov = getattr(info, "_prediction_overview", None) or {}
        out.append(
            {
                "parent_model_name": info.parent_model_name,
                "lab_name": info.lab_name,
                "version": info.version,
                "status": info.status,
                "db_path": info.db_path,
                "prediction_row_count": int(ov.get("prediction_row_count") or 0),
                "prediction_dataset_status": ov.get("prediction_dataset_status"),
            }
        )
    out.sort(key=lambda r: (str(r.get("parent_model_name") or ""), int(r.get("version") or 0)))
    return out


def list_labs_for_model(model_name: str, research_dir: str | None = None) -> list[ModelLabInfo]:
    by_path: dict[str, ModelLabInfo] = {}
    for _ver, path in list_lab_db_paths(model_name, research_dir=research_dir):
        info = load_lab(path)
        if info:
            by_path[os.path.abspath(info.db_path)] = info
    for path in iter_all_lab_db_paths(research_dir=research_dir):
        abs_path = os.path.abspath(path)
        if abs_path in by_path:
            continue
        info = load_lab(path)
        if info is None:
            continue
        if _parent_names_match(info.parent_model_name, model_name):
            by_path[abs_path] = info
    return sorted(by_path.values(), key=lambda i: int(i.version or 0))


def default_lab_display_name(model_name: str, version: int) -> str:
    return lab_db_filename(model_name, version).removesuffix(".db")


def create_model_lab(
    data_dir: str,
    detail_doc: dict[str, Any],
    *,
    research_dir: str | None = None,
    created_by: str = "ml_research_studio",
    lab_name: str | None = None,
    description: str | None = None,
    purpose: str | None = None,
) -> ModelLabInfo:
    """Create a new versioned Model Lab SQLite with immutable Phase-1 snapshots."""
    model_name = str(detail_doc.get("model_name") or "").strip()
    if not model_name:
        raise ValueError("model_name is required to create a Model Lab")

    root = research_dir or resolve_model_research_dir()
    version = next_lab_version(model_name, research_dir=root)
    db_path = lab_db_path(model_name, version, research_dir=root)
    if os.path.isfile(db_path):
        raise FileExistsError(f"Lab database already exists: {db_path}")

    snaps = build_lab_snapshots(data_dir, detail_doc)
    lab_uuid = str(uuid.uuid4())
    lab_id = f"{lab_db_stem(model_name)}_v{version}"
    resolved_name = str(lab_name or "").strip() or default_lab_display_name(model_name, version)

    with ModelLabStore(db_path) as store:
        info = store.write_info(
            lab_uuid=lab_uuid,
            lab_id=lab_id,
            lab_name=resolved_name,
            parent_model_id=str(snaps["parent_model_id"]),
            parent_model_name=model_name,
            model_checksum=snaps.get("model_checksum"),
            description=(str(description).strip() if description else None) or None,
            purpose=(str(purpose).strip() if purpose else None) or "General Research",
            version=version,
            original_feature_count=int(snaps["original_feature_count"])
            if snaps.get("original_feature_count") is not None
            else None,
            selected_feature_count=int(snaps["selected_feature_count"] or 0)
            if snaps.get("selected_feature_count") is not None
            else None,
            training_rows=int(snaps["training_rows"]) if snaps.get("training_rows") is not None else None,
            target=snaps.get("target"),
            algorithm=snaps.get("algorithm"),
            dataset_snapshot=snaps.get("dataset_snapshot"),
            model_snapshot=snaps.get("model_snapshot"),
            training_config_snapshot=snaps.get("training_config_snapshot"),
            wf_snapshot=snaps.get("wf_snapshot"),
            metrics_snapshot=snaps.get("metrics_snapshot"),
            selected_features_snapshot=snaps.get("selected_features_snapshot"),
            feature_ranking_snapshot=snaps.get("feature_ranking_snapshot"),
            artifact_pointers=snaps.get("artifact_pointers"),
            created_by=created_by,
            status=STATUS_READY,
            lab_schema_version=LAB_SCHEMA_VERSION,
            phase=LAB_PHASE,
        )
    return info

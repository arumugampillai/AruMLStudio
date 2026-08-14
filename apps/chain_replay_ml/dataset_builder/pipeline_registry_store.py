"""Persistent pipeline feature registry — stable PL IDs, registry membership, candidates."""

from __future__ import annotations

import json
import os
import re
from copy import deepcopy
from datetime import datetime, timezone
from typing import Any

_PIPELINE_ID_RE = re.compile(r"^PL_\d{4}$", re.IGNORECASE)
_VALID_TYPES = frozenset({"existing", "manual", "auto"})
_VALID_STATUSES = frozenset({"draft", "ready"})


def store_path(data_dir: str) -> str:
    return os.path.join(data_dir, "pipeline_registry_store.json")


def _load_json(path: str) -> dict[str, Any]:
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _empty_store() -> dict[str, Any]:
    now = _utc_now()
    return {
        "registry_version": "1.0",
        "created_on": now,
        "next_pipeline_id_seq": 1,
        "next_display_seq": 1,
        "pipelines": {},
        "history": [],
    }


def format_pipeline_id(seq: int) -> str:
    return f"PL_{seq:04d}"


def format_display_name(seq: int) -> str:
    return f"Pipeline_{seq:03d}"


def load_store(data_dir: str) -> dict[str, Any]:
    doc = _load_json(store_path(data_dir))
    if not doc:
        return _empty_store()
    doc.setdefault("registry_version", "1.0")
    doc.setdefault("next_pipeline_id_seq", 1)
    doc.setdefault("next_display_seq", 1)
    doc.setdefault("pipelines", {})
    doc.setdefault("history", [])
    return doc


def save_store(data_dir: str, doc: dict[str, Any]) -> None:
    path = store_path(data_dir)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=2)


def list_pipelines(doc: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for pid, rec in sorted((doc.get("pipelines") or {}).items()):
        if not isinstance(rec, dict):
            continue
        rows.append(pipeline_summary(rec, pipeline_id=str(pid)))
    return rows


def pipeline_summary(rec: dict[str, Any], *, pipeline_id: str) -> dict[str, Any]:
    reg_ids = list(rec.get("registry_feature_ids") or [])
    candidates = list(rec.get("candidate_features") or [])
    ptype = str(rec.get("type") or "manual").lower()
    status = str(rec.get("status") or "draft").lower()
    return {
        "pipeline_id": pipeline_id,
        "name": str(rec.get("name") or pipeline_id),
        "type": ptype,
        "type_label": _type_label(ptype),
        "status": status,
        "status_label": status.upper(),
        "registry_feature_ids": reg_ids,
        "registry_feature_count": len(reg_ids),
        "candidate_features": candidates,
        "candidate_count": len(candidates),
        "feature_count": len(reg_ids) + len(candidates),
        "created_at": rec.get("created_at"),
        "updated_at": rec.get("updated_at"),
    }


def _type_label(ptype: str) -> str:
    if ptype == "existing":
        return "Existing"
    if ptype == "auto":
        return "Auto"
    return "Manual"


def get_pipeline(doc: dict[str, Any], pipeline_id: str) -> dict[str, Any] | None:
    pid = str(pipeline_id or "").strip().upper()
    rec = (doc.get("pipelines") or {}).get(pid)
    if not isinstance(rec, dict):
        return None
    return deepcopy(rec)


def get_pipeline_summary(doc: dict[str, Any], pipeline_id: str) -> dict[str, Any] | None:
    pid = str(pipeline_id or "").strip().upper()
    rec = (doc.get("pipelines") or {}).get(pid)
    if not isinstance(rec, dict):
        return None
    return pipeline_summary(rec, pipeline_id=pid)


def _next_ids(doc: dict[str, Any]) -> tuple[str, str, int]:
    seq = int(doc.get("next_pipeline_id_seq") or 1)
    display_seq = int(doc.get("next_display_seq") or 1)
    pipeline_id = format_pipeline_id(seq)
    name = format_display_name(display_seq)
    doc["next_pipeline_id_seq"] = seq + 1
    doc["next_display_seq"] = display_seq + 1
    return pipeline_id, name, display_seq


def create_pipeline(
    doc: dict[str, Any],
    *,
    name: str | None = None,
    pipeline_type: str = "manual",
    status: str = "draft",
) -> dict[str, Any]:
    ptype = str(pipeline_type or "manual").strip().lower()
    if ptype not in _VALID_TYPES:
        ptype = "manual"
    st = str(status or "draft").strip().lower()
    if st not in _VALID_STATUSES:
        st = "draft"
    pipeline_id, default_name, _ = _next_ids(doc)
    display_name = str(name or default_name).strip() or default_name
    now = _utc_now()
    rec = {
        "pipeline_id": pipeline_id,
        "name": display_name,
        "type": ptype,
        "status": st,
        "registry_feature_ids": [],
        "candidate_features": [],
        "transformation_config": None,
        "created_at": now,
        "updated_at": now,
    }
    pipelines = doc.setdefault("pipelines", {})
    pipelines[pipeline_id] = rec
    doc.setdefault("history", []).append(
        {"ts": now, "action": "create", "pipeline_id": pipeline_id, "name": display_name}
    )
    return deepcopy(rec)


def update_pipeline(
    doc: dict[str, Any],
    pipeline_id: str,
    *,
    name: str | None = None,
    status: str | None = None,
    transformation_config: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    pid = str(pipeline_id or "").strip().upper()
    pipelines = doc.get("pipelines") or {}
    rec = pipelines.get(pid)
    if not isinstance(rec, dict):
        return None
    if name is not None:
        rec["name"] = str(name).strip() or rec.get("name") or pid
    if status is not None:
        st = str(status).strip().lower()
        if st in _VALID_STATUSES:
            rec["status"] = st
    if transformation_config is not None:
        rec["transformation_config"] = deepcopy(transformation_config)
    rec["updated_at"] = _utc_now()
    doc.setdefault("history", []).append({"ts": rec["updated_at"], "action": "update", "pipeline_id": pid})
    return deepcopy(rec)


def delete_pipeline(doc: dict[str, Any], pipeline_id: str) -> bool:
    """Remove a pipeline from the store. The existing default pipeline cannot be deleted."""
    pid = str(pipeline_id or "").strip().upper()
    pipelines = doc.get("pipelines") or {}
    rec = pipelines.get(pid)
    if not isinstance(rec, dict):
        return False
    if str(rec.get("type") or "") == "existing":
        raise ValueError("The existing default pipeline cannot be deleted.")
    del pipelines[pid]
    doc["pipelines"] = pipelines
    now = _utc_now()
    doc.setdefault("history", []).append(
        {
            "ts": now,
            "action": "delete",
            "pipeline_id": pid,
            "name": rec.get("name"),
        }
    )
    return True


def set_registry_members(
    doc: dict[str, Any],
    pipeline_id: str,
    feature_ids: list[str],
) -> dict[str, Any] | None:
    pid = str(pipeline_id or "").strip().upper()
    pipelines = doc.get("pipelines") or {}
    rec = pipelines.get(pid)
    if not isinstance(rec, dict):
        return None
    clean: list[str] = []
    seen: set[str] = set()
    for raw in feature_ids:
        fid = str(raw or "").strip().upper()
        if not fid or fid in seen:
            continue
        if fid.startswith("FR"):
            seen.add(fid)
            clean.append(fid)
    rec["registry_feature_ids"] = clean
    rec["updated_at"] = _utc_now()
    return deepcopy(rec)


def add_candidate_features(
    doc: dict[str, Any],
    pipeline_id: str,
    names: list[str],
    *,
    replace: bool = False,
) -> dict[str, Any] | None:
    pid = str(pipeline_id or "").strip().upper()
    pipelines = doc.get("pipelines") or {}
    rec = pipelines.get(pid)
    if not isinstance(rec, dict):
        return None
    incoming = [str(n).strip() for n in names if str(n).strip()]
    if replace:
        rec["candidate_features"] = list(dict.fromkeys(incoming))
    else:
        merged = list(rec.get("candidate_features") or [])
        seen = set(merged)
        for n in incoming:
            if n not in seen:
                merged.append(n)
                seen.add(n)
        rec["candidate_features"] = merged
    rec["updated_at"] = _utc_now()
    doc.setdefault("history", []).append(
        {
            "ts": rec["updated_at"],
            "action": "add_candidates",
            "pipeline_id": pid,
            "count": len(incoming),
        }
    )
    return deepcopy(rec)


def ensure_default_existing_pipeline(data_dir: str) -> dict[str, Any]:
    """Seed PL_0001 with legacy pipeline-owned features if store is empty."""
    doc = load_store(data_dir)
    pipelines = doc.get("pipelines") or {}
    if pipelines:
        return doc
    from .feature_migration import PIPELINE_OWNED_FEATURES
    from .pipeline_features_prefs import active_pipeline_feature_names

    candidates = active_pipeline_feature_names(sorted(PIPELINE_OWNED_FEATURES), data_dir=data_dir)
    now = _utc_now()
    pipeline_id = format_pipeline_id(1)
    doc["next_pipeline_id_seq"] = max(int(doc.get("next_pipeline_id_seq") or 1), 2)
    doc["next_display_seq"] = max(int(doc.get("next_display_seq") or 1), 2)
    pipelines[pipeline_id] = {
        "pipeline_id": pipeline_id,
        "name": format_display_name(1),
        "type": "existing",
        "status": "ready",
        "registry_feature_ids": [],
        "candidate_features": list(candidates),
        "transformation_config": None,
        "created_at": now,
        "updated_at": now,
    }
    doc["pipelines"] = pipelines
    save_store(data_dir, doc)
    return doc


def resolve_registry_names(data_dir: str, feature_ids: list[str]) -> list[str]:
    from .feature_registry_store import load_store as load_fr_store

    store = load_fr_store(data_dir)
    identities = store.get("feature_identities") or {}
    names: list[str] = []
    for fid in feature_ids:
        ident = identities.get(str(fid).strip().upper())
        if isinstance(ident, dict) and ident.get("name"):
            names.append(str(ident["name"]))
    return names


def pipeline_feature_names(doc: dict[str, Any], pipeline_id: str) -> list[str]:
    """All feature names associated with a pipeline (candidates only for Phase 1 builds)."""
    summary = get_pipeline_summary(doc, pipeline_id)
    if not summary:
        return []
    return list(summary.get("candidate_features") or [])


__all__ = [
    "add_candidate_features",
    "create_pipeline",
    "delete_pipeline",
    "ensure_default_existing_pipeline",
    "format_display_name",
    "format_pipeline_id",
    "get_pipeline",
    "get_pipeline_summary",
    "list_pipelines",
    "load_store",
    "pipeline_feature_names",
    "pipeline_summary",
    "resolve_registry_names",
    "save_store",
    "set_registry_members",
    "store_path",
    "update_pipeline",
]

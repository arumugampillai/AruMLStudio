"""Feature delete preview and safe removal for backlog/imported features."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any

from .auditor import _load_json, list_datasets
from .feature_plugins import GROUP_FEATURE_SOURCES
from .feature_registry_catalog import (
    build_feature_registry_catalog,
    load_feature_backlog,
    feature_backlog_path,
)
from .feature_registry_store import (
    _FEATURE_ID_RE,
    load_store,
    save_store,
)
from .schema_registry import load_schema_registry
from .writer import datasets_dir


def _resolve_feature(catalog: dict[str, Any], ref: str) -> dict[str, Any] | None:
    text = str(ref or "").strip()
    if not text:
        return None
    features = catalog.get("features") or []
    if _FEATURE_ID_RE.match(text):
        fid = text.upper()
        return next((f for f in features if f.get("feature_id") == fid), None)
    return next((f for f in features if f.get("name") == text), None)


def _scan_registry_dependents(name: str, catalog: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for feat in catalog.get("features") or []:
        deps = list(feat.get("dependencies") or [])
        if name not in deps:
            continue
        out.append({
            "feature_id": feat.get("feature_id"),
            "name": feat.get("name"),
            "display_name": feat.get("display_name"),
        })
    return out


def _feature_in_schema(name: str) -> bool:
    schema = load_schema_registry()
    col = (schema.get("columns") or {}).get(name)
    if col and str(col.get("type") or "").lower() == "feature":
        return True
    for gid, mapping in GROUP_FEATURE_SOURCES.items():
        if name in mapping:
            return True
    return False


def _schema_groups_for_feature(name: str) -> list[str]:
    groups: list[str] = []
    schema = load_schema_registry()
    col = (schema.get("columns") or {}).get(name)
    if col and col.get("group"):
        groups.append(str(col["group"]))
    for gid, mapping in GROUP_FEATURE_SOURCES.items():
        if name in mapping and gid not in groups:
            groups.append(gid)
    reg_groups = (schema.get("groups") or {})
    for gid, block in reg_groups.items():
        feats = list((block or {}).get("features") or [])
        if name in feats and gid not in groups:
            groups.append(gid)
    return groups


def _scan_dataset_schemas(data_dir: str, name: str) -> list[dict[str, Any]]:
    hits: list[dict[str, Any]] = []
    if _feature_in_schema(name):
        hits.append({"kind": "schema_registry", "label": "ML schema registry"})
    out_dir = datasets_dir(data_dir)
    if not os.path.isdir(out_dir):
        return hits
    for fname in os.listdir(out_dir):
        if not fname.endswith(".expected.json"):
            continue
        path = os.path.join(out_dir, fname)
        try:
            doc = _load_json(path)
        except (OSError, json.JSONDecodeError):
            continue
        cols = list(doc.get("feature_column_names") or doc.get("enabled_features") or [])
        if name in cols:
            hits.append({
                "kind": "expected_spec",
                "label": fname.replace(".expected.json", ""),
            })
    return hits


def _scan_datasets(data_dir: str, name: str) -> list[dict[str, Any]]:
    hits: list[dict[str, Any]] = []
    for row in list_datasets(data_dir):
        meta_path = row.get("metadata_path")
        if not meta_path or not os.path.isfile(meta_path):
            continue
        try:
            meta = _load_json(meta_path)
        except (OSError, json.JSONDecodeError):
            continue
        cols = set(meta.get("feature_columns") or [])
        cols.update(meta.get("feature_columns_pending") or [])
        cols.update(meta.get("enabled_features") or [])
        if name in cols:
            hits.append({
                "dataset_name": row.get("dataset_name"),
                "row_count": row.get("row_count"),
            })
    return hits


def _scan_models(data_dir: str, name: str, feature_id: str | None) -> list[dict[str, Any]]:
    hits: list[dict[str, Any]] = []
    models_dir = os.path.join(data_dir, "models")
    if not os.path.isdir(models_dir):
        return hits
    for entry in os.listdir(models_dir):
        if entry.startswith("."):
            continue
        pkg = os.path.join(models_dir, entry)
        config_path = os.path.join(pkg, "config.json")
        if not os.path.isfile(config_path):
            continue
        try:
            with open(config_path, encoding="utf-8") as fh:
                config = json.load(fh)
        except (OSError, json.JSONDecodeError):
            continue
        features = list(config.get("features") or [])
        feature_ids = list(config.get("feature_ids") or [])
        if name in features or (feature_id and feature_id in feature_ids):
            hits.append({
                "model_name": str(config.get("model_name") or entry),
                "package": entry,
            })
    return hits


def preview_feature_delete(data_dir: str, ref: str) -> dict[str, Any]:
    catalog = build_feature_registry_catalog(data_dir)
    feat = _resolve_feature(catalog, ref)
    if not feat:
        raise ValueError(f"Feature not found: {ref}")

    name = str(feat.get("name") or "")
    feature_id = feat.get("feature_id")
    source = str(feat.get("source") or "registry")

    registry_deps = _scan_registry_dependents(name, catalog)
    schema_hits = _scan_dataset_schemas(data_dir, name)
    dataset_hits = _scan_datasets(data_dir, name)
    model_hits = _scan_models(data_dir, name, feature_id)
    groups = _schema_groups_for_feature(name)
    if feat.get("group_id") and feat.get("group_id") not in groups:
        groups.append(str(feat["group_id"]))

    blockers: list[str] = []
    if source == "registry":
        blockers.append("Built-in schema feature (implemented in code)")
    if _feature_in_schema(name) and source != "registry":
        blockers.append("Referenced in ML schema registry")
    if registry_deps:
        blockers.append(f"{len(registry_deps)} registry feature(s) depend on it")
    if schema_hits:
        blockers.append(f"Present in {len(schema_hits)} dataset schema(s)")
    if dataset_hits:
        blockers.append(f"Used by {len(dataset_hits)} dataset(s)")
    if model_hits:
        blockers.append(f"Used by {len(model_hits)} trained model(s)")

    can_delete = len(blockers) == 0 and source in ("planned", "imported")
    if source not in ("planned", "imported") and not blockers:
        blockers.append("Only backlog or imported features can be removed from the registry overlay")

    status = "safe_to_delete" if can_delete else "cannot_safely_delete"
    status_label = "Safe to delete." if can_delete else "Cannot safely delete."

    return {
        "ok": True,
        "feature_id": feature_id,
        "name": name,
        "display_name": feat.get("display_name"),
        "source": source,
        "owner": feat.get("owner"),
        "currently_used_by": {
            "registry_dependencies": {
                "count": len(registry_deps),
                "items": registry_deps[:20],
            },
            "dataset_schemas": {
                "count": len(schema_hits),
                "items": schema_hits[:20],
            },
            "datasets": {
                "count": len(dataset_hits),
                "items": dataset_hits[:20],
            },
            "models": {
                "count": len(model_hits),
                "items": model_hits[:20],
            },
            "feature_groups": {
                "count": len(groups),
                "items": groups,
            },
        },
        "can_delete": can_delete,
        "status": status,
        "status_label": status_label,
        "blockers": blockers,
    }


def apply_feature_delete(data_dir: str, ref: str) -> dict[str, Any]:
    preview = preview_feature_delete(data_dir, ref)
    if not preview.get("can_delete"):
        raise ValueError("; ".join(preview.get("blockers") or ["Cannot safely delete this feature"]))

    name = str(preview["name"])
    feature_id = preview.get("feature_id")
    now = datetime.now(timezone.utc).isoformat()

    # Remove from backlog
    backlog_doc = load_feature_backlog(data_dir)
    backlog_features = [
        r for r in (backlog_doc.get("features") or [])
        if str(r.get("name") or "") != name
    ]
    backlog_doc["features"] = backlog_features
    path = feature_backlog_path(data_dir)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(backlog_doc, fh, indent=2)

    # Remove from store overlay
    store = load_store(data_dir)
    imported = dict(store.get("imported_features") or {})
    overrides = dict(store.get("overrides") or {})
    ids = dict(store.get("feature_ids") or {})
    identities = dict(store.get("feature_identities") or {})
    deleted = dict(store.get("deleted_feature_ids") or {})

    imported.pop(name, None)
    overrides.pop(name, None)
    if name in ids:
        fid = ids.pop(name)
        if fid in identities:
            tomb = dict(identities.pop(fid))
            tomb["deleted_at"] = now
            tomb["deleted_by"] = "User"
            deleted[fid] = tomb

    if feature_id and feature_id in identities:
        tomb = dict(identities.pop(feature_id))
        tomb["deleted_at"] = now
        tomb["deleted_by"] = "User"
        deleted[feature_id] = tomb
        for alias, fid in list(ids.items()):
            if fid == feature_id:
                del ids[alias]

    store["imported_features"] = imported
    store["overrides"] = overrides
    store["feature_ids"] = ids
    store["feature_identities"] = identities
    store["deleted_feature_ids"] = deleted
    save_store(data_dir, store)

    return {
        "ok": True,
        "deleted": True,
        "feature_id": feature_id,
        "name": name,
        "message": f"Removed {name} ({feature_id or '—'}) from registry overlay. ID preserved in tombstone.",
    }

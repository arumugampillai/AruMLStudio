"""Feature Project organization — canonical Registry domains + project-only overrides.

Canonical groups match Feature Transformations → Feature Sources (``feature_domains``).
Projects may add custom organizational groups and reassign features within the project only.
"""

from __future__ import annotations

from typing import Any

from .feature_domains import DOMAIN_LABELS, DOMAIN_ORDER, primary_domain_of

CANONICAL_DOMAIN_IDS: frozenset[str] = frozenset(DOMAIN_ORDER)
RESERVED_ALL_PROJECT_ID = "all"


def is_reserved_all_project_id(project_id: str) -> bool:
    return str(project_id or "").strip().lower() == RESERVED_ALL_PROJECT_ID


def active_registry_feature_names(*, data_dir: str | None = None) -> list[str]:
    from .feature_sources_catalog import registry_feature_names

    return list(registry_feature_names(data_dir=data_dir))


def build_default_all_project_doc(*, data_dir: str | None = None) -> dict[str, Any]:
    from datetime import datetime, timezone

    names = active_registry_feature_names(data_dir=data_dir)
    project_groups: list[dict[str, str]] = []
    map_out = backfill_feature_group_map(
        names,
        project_groups=project_groups,
        feature_group_map={},
    )
    group_ids = sync_project_group_ids(names, map_out)
    now = datetime.now(timezone.utc).isoformat()
    return {
        "label": "all",
        "description": "All active Feature Registry features.",
        "group_ids": group_ids,
        "feature_names": sorted(names),
        "project_groups": [],
        "feature_group_map": map_out,
        "warmup_minutes": None,
        "default_sampling": "",
        "notes": "",
        "version": "1",
        "reserved": True,
        "created_at": now,
        "updated_at": now,
    }


def sync_all_project_membership(
    doc: dict[str, Any],
    *,
    data_dir: str | None = None,
) -> dict[str, Any]:
    """Drop retired registry features from the reserved all project membership."""
    active = frozenset(active_registry_feature_names(data_dir=data_dir))
    names = sorted(
        {
            str(n).strip()
            for n in (doc.get("feature_names") or doc.get("enabled_features") or [])
            if str(n).strip() and str(n).strip() in active
        }
    )
    out = dict(doc)
    out["feature_names"] = names
    out["feature_group_map"] = {
        str(k): str(v)
        for k, v in dict(doc.get("feature_group_map") or {}).items()
        if str(k).strip() in active
    }
    return migrate_project_organization(out, data_dir=data_dir)


def is_canonical_domain_id(group_id: str) -> bool:
    return str(group_id or "").strip() in CANONICAL_DOMAIN_IDS


def canonical_group_for_feature(feature_name: str) -> str:
    return str(primary_domain_of(str(feature_name or "").strip()))


def canonical_group_label(group_id: str) -> str:
    gid = str(group_id or "").strip()
    if is_canonical_domain_id(gid):
        return str(DOMAIN_LABELS.get(gid, gid))  # type: ignore[arg-type]
    return gid


def canonical_registry_groups(*, data_dir: str | None = None) -> list[dict[str, Any]]:
    """Same group list as Feature Transformations → Registry Features tree."""
    from .feature_sources_catalog import registry_feature_source

    source = registry_feature_source(data_dir=data_dir)
    return list(source.get("groups") or [])


def normalize_custom_project_groups(groups: list[Any] | None) -> list[dict[str, str]]:
    """Project-specific groups only — never duplicate canonical domain ids."""
    out: list[dict[str, str]] = []
    seen: set[str] = set()
    for raw in list(groups or []):
        if not isinstance(raw, dict):
            continue
        gid = str(raw.get("id") or "").strip()
        label = str(raw.get("label") or gid).strip()
        if not gid or gid in seen or is_canonical_domain_id(gid):
            continue
        seen.add(gid)
        out.append({"id": gid, "label": label or gid})
    return out


def project_group_label(
    project_groups: list[dict[str, Any]],
    group_id: str,
) -> str:
    gid = str(group_id or "").strip()
    for g in project_groups:
        if str(g.get("id") or "") == gid:
            return str(g.get("label") or gid)
    if is_canonical_domain_id(gid):
        return canonical_group_label(gid)
    return gid or "—"


def backfill_feature_group_map(
    feature_names: set[str] | list[str],
    *,
    project_groups: list[dict[str, Any]],
    feature_group_map: dict[str, str] | None = None,
) -> dict[str, str]:
    out = dict(feature_group_map or {})
    custom_ids = {str(g.get("id") or "").strip() for g in project_groups if str(g.get("id") or "").strip()}
    for raw in feature_names:
        name = str(raw or "").strip()
        if not name:
            continue
        current = str(out.get(name) or "").strip()
        if current in custom_ids or is_canonical_domain_id(current):
            continue
        out[name] = canonical_group_for_feature(name)
    return out


def migrate_project_organization(
    doc: dict[str, Any],
    *,
    data_dir: str | None = None,
) -> dict[str, Any]:
    """Remap legacy schema-registry group ids to canonical domain ids (Registry unchanged)."""
    features = {
        str(n).strip()
        for n in (doc.get("feature_names") or doc.get("enabled_features") or [])
        if str(n).strip()
    }
    project_groups = normalize_custom_project_groups(doc.get("project_groups"))
    custom_ids = {g["id"] for g in project_groups}
    raw_map = dict(doc.get("feature_group_map") or {})
    canonical_by_feature: dict[str, str] = {}
    for group in canonical_registry_groups(data_dir=data_dir):
        gid = str(group.get("id") or "").strip()
        for feat in group.get("features") or []:
            name = str(feat or "").strip()
            if name:
                canonical_by_feature[name] = gid

    new_map: dict[str, str] = {}
    for name in sorted(features | set(raw_map.keys())):
        current = str(raw_map.get(name) or "").strip()
        canonical = canonical_by_feature.get(name) or canonical_group_for_feature(name)
        if current in custom_ids and not is_canonical_domain_id(current):
            new_map[name] = current
        elif is_canonical_domain_id(current):
            new_map[name] = current
        else:
            new_map[name] = canonical

    group_ids = sorted({new_map[n] for n in features if n in new_map})
    migrated = dict(doc)
    migrated["project_groups"] = project_groups
    migrated["feature_group_map"] = new_map
    migrated["group_ids"] = group_ids
    return migrated


def project_group_tree(
    *,
    project_groups: list[dict[str, Any]],
    feature_group_map: dict[str, str],
    data_dir: str | None = None,
) -> list[dict[str, Any]]:
    """Canonical groups (Feature Sources order) + custom project groups at the end."""
    canonical = canonical_registry_groups(data_dir=data_dir)
    canonical_ids = {str(g.get("id") or "") for g in canonical}
    out: list[dict[str, Any]] = [
        {
            "id": str(g.get("id") or ""),
            "label": str(g.get("label") or g.get("id") or ""),
            "registry_features": list(g.get("features") or []),
        }
        for g in canonical
        if str(g.get("id") or "").strip()
    ]
    custom = []
    for g in normalize_custom_project_groups(project_groups):
        gid = g["id"]
        if gid in canonical_ids:
            continue
        custom.append({
            "id": gid,
            "label": g["label"],
            "registry_features": [],
        })
    custom.sort(key=lambda row: str(row.get("label") or "").lower())
    out.extend(custom)
    seen = {row["id"] for row in out}
    for gid in sorted({str(v).strip() for v in feature_group_map.values() if str(v).strip()}):
        if gid in seen:
            continue
        out.append({
            "id": gid,
            "label": project_group_label(project_groups, gid),
            "registry_features": [],
        })
        seen.add(gid)
    return out


def sync_project_group_ids(
    feature_names: set[str] | list[str],
    feature_group_map: dict[str, str],
) -> list[str]:
    names = {str(n).strip() for n in feature_names if str(n).strip()}
    ids: set[str] = set()
    for name in names:
        gid = str(feature_group_map.get(name) or canonical_group_for_feature(name) or "").strip()
        if gid:
            ids.add(gid)
    return sorted(ids)


def load_project_doc(
    data_dir: str,
    project_id: str,
) -> dict[str, Any]:
    """Load one project document with organization fields normalized."""
    from .feature_registry_store import ensure_all_project, load_store

    ensure_all_project(data_dir)
    store = load_store(data_dir)
    projects = dict(store.get("projects") or {})
    pid = str(project_id or "").strip().lower()
    if pid not in projects:
        raise ValueError(f"Project not found: {pid}")
    return migrate_project_organization(dict(projects[pid]), data_dir=data_dir)


def project_registry_groups(
    project_doc: dict[str, Any],
    *,
    data_dir: str | None = None,
) -> list[dict[str, Any]]:
    """UI/registry-source groups for a project: id, label, count, features."""
    feature_names = {
        str(n).strip()
        for n in (project_doc.get("feature_names") or project_doc.get("enabled_features") or [])
        if str(n).strip()
    }
    project_groups = list(project_doc.get("project_groups") or [])
    feature_group_map = dict(project_doc.get("feature_group_map") or {})
    rows = project_group_tree(
        project_groups=project_groups,
        feature_group_map=feature_group_map,
        data_dir=data_dir,
    )
    out: list[dict[str, Any]] = []
    for row in rows:
        gid = str(row.get("id") or "").strip()
        if not gid:
            continue
        feats = sorted(
            name
            for name in feature_names
            if str(feature_group_map.get(name) or "") == gid
        )
        out.append({
            "id": gid,
            "label": str(row.get("label") or gid),
            "count": len(feats),
            "features": feats,
        })
    return out


def project_registry_feature_source(
    *,
    data_dir: str,
    project_id: str,
) -> dict[str, Any]:
    """Registry Features source tree using project organization (not raw domains)."""
    from .feature_domains import validate_domain_coverage
    from .feature_ownership import canonical_registry_features
    from .feature_sources_catalog import FEATURE_SOURCE_REGISTRY, registry_retired_feature_names

    doc = load_project_doc(data_dir, project_id)
    feature_names = sorted(
        {
            str(n).strip()
            for n in (doc.get("feature_names") or doc.get("enabled_features") or [])
            if str(n).strip()
        }
    )
    groups = project_registry_groups(doc, data_dir=data_dir)
    retired = registry_retired_feature_names(data_dir)
    canonical_total = len(canonical_registry_features())
    coverage = validate_domain_coverage(expected_total=206)
    label = str(doc.get("label") or project_id)
    return {
        "id": FEATURE_SOURCE_REGISTRY,
        "label": "Registry Features",
        "description": f"Project organization: {label} ({project_id})",
        "total": len(feature_names),
        "expected_total": 206,
        "retired_count": len(retired),
        "ready": bool(coverage.get("ok")) and canonical_total == 206,
        "groups": groups,
        "features": feature_names,
        "project_id": str(project_id).strip().lower(),
    }


__all__ = [
    "CANONICAL_DOMAIN_IDS",
    "RESERVED_ALL_PROJECT_ID",
    "active_registry_feature_names",
    "backfill_feature_group_map",
    "build_default_all_project_doc",
    "canonical_group_for_feature",
    "canonical_group_label",
    "canonical_registry_groups",
    "is_canonical_domain_id",
    "is_reserved_all_project_id",
    "load_project_doc",
    "migrate_project_organization",
    "normalize_custom_project_groups",
    "project_group_label",
    "project_group_tree",
    "project_registry_feature_source",
    "project_registry_groups",
    "sync_all_project_membership",
    "sync_project_group_ids",
]

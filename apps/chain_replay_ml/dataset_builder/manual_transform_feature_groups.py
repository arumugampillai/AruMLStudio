"""Feature Project group tree for Manual Feature Transformation pickers."""

from __future__ import annotations

from typing import Any, Sequence

from .feature_project_organization import (
    load_project_doc,
    project_registry_feature_source,
    project_registry_groups,
)


def grouped_features_for_manual_transform(
    data_dir: str,
    feature_project_id: str,
    available_features: Sequence[str],
) -> list[dict[str, Any]]:
    """
    Project organization groups (FPM / Auto Feature Sources parity) limited to
    features present in the master dataset and eligible for transformation.
    """
    pid = str(feature_project_id or "").strip().lower()
    avail = {str(f).strip() for f in available_features if str(f).strip()}
    if not avail or not pid:
        return []

    src = project_registry_feature_source(data_dir=data_dir, project_id=pid)
    out: list[dict[str, Any]] = []
    covered: set[str] = set()
    for group in src.get("groups") or []:
        gid = str(group.get("id") or "").strip()
        if not gid:
            continue
        feats = sorted(f for f in (group.get("features") or []) if f in avail)
        if not feats:
            continue
        covered.update(feats)
        out.append({
            "id": gid,
            "label": str(group.get("label") or gid),
            "features": feats,
        })
    orphans = sorted(avail - covered)
    if orphans:
        out.append({
            "id": "__other__",
            "label": "Other",
            "features": orphans,
        })
    return out


def project_groups_for_features(
    data_dir: str,
    feature_project_id: str,
    feature_names: Sequence[str],
) -> list[dict[str, Any]]:
    """Group an explicit feature list using project organization (no master filter)."""
    pid = str(feature_project_id or "").strip().lower()
    names = {str(f).strip() for f in feature_names if str(f).strip()}
    if not names or not pid:
        return []
    doc = load_project_doc(data_dir, pid)
    groups = project_registry_groups(doc, data_dir=data_dir)
    out: list[dict[str, Any]] = []
    covered: set[str] = set()
    for group in groups:
        gid = str(group.get("id") or "").strip()
        if not gid:
            continue
        feats = sorted(f for f in (group.get("features") or []) if f in names)
        if not feats:
            continue
        covered.update(feats)
        out.append({
            "id": gid,
            "label": str(group.get("label") or gid),
            "features": feats,
        })
    orphans = sorted(names - covered)
    if orphans:
        out.append({
            "id": "__other__",
            "label": "Other",
            "features": orphans,
        })
    return out

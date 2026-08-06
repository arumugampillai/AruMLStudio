"""Feature selection logic — parity with web Create Dataset UI."""

from __future__ import annotations

from typing import Any


def all_group_ids(registry: dict[str, Any]) -> list[str]:
    return [str(g) for g in (registry.get("groupOrder") or [])]


def group_meta(registry: dict[str, Any], group_id: str) -> dict[str, Any]:
    return dict((registry.get("groups") or {}).get(group_id) or {})


def group_feature_count(registry: dict[str, Any], group_id: str) -> int:
    return len(group_meta(registry, group_id).get("features") or [])


def total_registry_features(registry: dict[str, Any]) -> int:
    return sum(group_feature_count(registry, gid) for gid in all_group_ids(registry))


def active_registry_feature_names(
    registry: dict[str, Any],
    *,
    exclude: set[str] | frozenset[str] | None = None,
) -> list[str]:
    names = all_registry_feature_names(registry)
    if not exclude:
        return names
    blocked = {str(n) for n in exclude}
    return [n for n in names if n not in blocked]


def total_active_registry_features(
    registry: dict[str, Any],
    *,
    exclude: set[str] | frozenset[str] | None = None,
) -> int:
    return len(active_registry_feature_names(registry, exclude=exclude))


def transitive_requires(registry: dict[str, Any], group_id: str) -> set[str]:
    dep_map = registry.get("dependencies") or {}
    deps: set[str] = set()
    stack = [str(d) for d in (dep_map.get(group_id) or [])]
    while stack:
        dep = stack.pop()
        if dep in deps:
            continue
        deps.add(dep)
        stack.extend(str(d) for d in (dep_map.get(dep) or []))
    return deps


def normalize_enabled_groups(
    registry: dict[str, Any],
    enabled: set[str] | list[str],
) -> set[str]:
    out = {str(g) for g in enabled}
    for gid in registry.get("hardMandatory") or []:
        out.add(str(gid))
    changed = True
    while changed:
        changed = False
        for gid in list(out):
            for req in transitive_requires(registry, gid):
                if req not in out:
                    out.add(req)
                    changed = True
    return out


def all_registry_feature_names(registry: dict[str, Any]) -> list[str]:
    names: list[str] = []
    for gid in all_group_ids(registry):
        for feat in group_meta(registry, gid).get("features") or []:
            names.append(str(feat))
    return list(dict.fromkeys(names))


def export_feature_columns(
    registry: dict[str, Any],
    enabled_groups: set[str],
    *,
    feature_set: set[str] | None = None,
) -> list[str]:
    if feature_set:
        order = all_registry_feature_names(registry)
        return [f for f in order if f in feature_set]
    cols: list[str] = []
    for gid in all_group_ids(registry):
        if gid not in enabled_groups:
            continue
        for feat in group_meta(registry, gid).get("features") or []:
            cols.append(str(feat))
    return cols


def is_hard_mandatory(registry: dict[str, Any], group_id: str) -> bool:
    return str(group_id) in {str(g) for g in (registry.get("hardMandatory") or [])}


def locked_by_group_id(
    registry: dict[str, Any],
    enabled_groups: set[str],
) -> dict[str, set[str]]:
    locks: dict[str, set[str]] = {}
    for dep_id in enabled_groups:
        label = str(group_meta(registry, dep_id).get("label") or dep_id)
        for req_id in transitive_requires(registry, dep_id):
            locks.setdefault(req_id, set()).add(label)
    return locks


def group_display_state(
    registry: dict[str, Any],
    group_id: str,
    enabled_groups: set[str],
) -> dict[str, Any]:
    if is_hard_mandatory(registry, group_id):
        return {"kind": "mandatory"}
    if group_id not in enabled_groups:
        return {"kind": "off"}
    locks = locked_by_group_id(registry, enabled_groups)
    if group_id in locks:
        names = ", ".join(sorted(locks[group_id]))
        return {"kind": "locked", "required_by": names}
    return {"kind": "on"}


def is_default_feature_selection(
    registry: dict[str, Any],
    enabled_groups: set[str],
    enabled_features: list[str] | None = None,
    *,
    exclude_features: set[str] | frozenset[str] | None = None,
) -> bool:
    all_names = active_registry_feature_names(registry, exclude=exclude_features)
    feats = enabled_features or export_feature_columns(registry, enabled_groups)
    if exclude_features:
        blocked = {str(n) for n in exclude_features}
        feats = [f for f in feats if f not in blocked]
    if len(feats) != len(all_names):
        return False
    feat_set = set(feats)
    return all(f in feat_set for f in all_names)


def detect_feature_profile(
    registry: dict[str, Any],
    profile: str,
    enabled_groups: set[str],
    enabled_features: list[str],
) -> str:
    if str(profile).lower() == "custom":
        return "custom"
    if is_default_feature_selection(registry, enabled_groups, enabled_features):
        return "default"
    return "custom"


def default_feature_config(registry: dict[str, Any]) -> dict[str, Any]:
    enabled = normalize_enabled_groups(registry, set(all_group_ids(registry)))
    return {
        "configVersion": int(registry.get("version") or 1),
        "profile": "default",
        "enabledGroups": sorted(enabled),
        "enabledFeatures": all_registry_feature_names(registry),
        "applied": True,
    }


def read_feature_config(
    registry: dict[str, Any],
    *,
    profile: str,
    enabled_groups: set[str],
    enabled_features: set[str],
) -> dict[str, Any]:
    groups = normalize_enabled_groups(registry, enabled_groups)
    features = export_feature_columns(registry, groups, feature_set=enabled_features)
    return {
        "configVersion": int(registry.get("version") or 1),
        "profile": detect_feature_profile(registry, profile, groups, features),
        "enabledGroups": sorted(groups),
        "enabledFeatures": features,
        "applied": True,
    }


def sync_enabled_groups_from_features(
    registry: dict[str, Any],
    enabled_features: set[str],
) -> set[str]:
    next_groups: set[str] = set()
    for gid in all_group_ids(registry):
        feats = [str(f) for f in (group_meta(registry, gid).get("features") or [])]
        if any(f in enabled_features for f in feats):
            next_groups.add(gid)
    return normalize_enabled_groups(registry, next_groups)


def enforce_mandatory_features(
    registry: dict[str, Any],
    enabled_groups: set[str],
    enabled_features: set[str],
    *,
    except_groups: set[str] | None = None,
) -> None:
    skip = {str(g) for g in (except_groups or set())}
    for gid in registry.get("hardMandatory") or []:
        if str(gid) in skip:
            continue
        for feat in group_meta(registry, str(gid)).get("features") or []:
            enabled_features.add(str(feat))
    locks = locked_by_group_id(registry, enabled_groups)
    for gid in locks:
        if str(gid) in skip:
            continue
        for feat in group_meta(registry, gid).get("features") or []:
            enabled_features.add(str(feat))


def enable_feature_group(
    registry: dict[str, Any],
    enabled_groups: set[str],
    group_id: str,
) -> set[str]:
    out = set(enabled_groups)
    out.add(str(group_id))
    return normalize_enabled_groups(registry, out)


def disable_feature_group(
    registry: dict[str, Any],
    enabled_groups: set[str],
    group_id: str,
    *,
    except_groups: set[str] | None = None,
) -> set[str]:
    gid = str(group_id)
    skip = {str(g) for g in (except_groups or set())}
    if is_hard_mandatory(registry, gid) and gid not in skip:
        return set(enabled_groups)
    to_remove = {gid}
    for other in list(enabled_groups):
        if other == gid:
            continue
        if gid in transitive_requires(registry, other):
            to_remove.add(other)
    return {g for g in enabled_groups if g not in to_remove}


def set_group_features_enabled(
    registry: dict[str, Any],
    enabled_groups: set[str],
    enabled_features: set[str],
    group_id: str,
    *,
    enabled: bool,
    except_groups: set[str] | None = None,
) -> tuple[set[str], set[str]]:
    feats = [str(f) for f in (group_meta(registry, group_id).get("features") or [])]
    if enabled:
        enabled_features.update(feats)
    else:
        for feat in feats:
            enabled_features.discard(feat)
    enforce_mandatory_features(
        registry, enabled_groups, enabled_features, except_groups=except_groups,
    )
    groups = sync_enabled_groups_from_features(registry, enabled_features)
    return groups, enabled_features


def profile_label(registry: dict[str, Any], profile_id: str) -> str:
    if profile_id == "custom":
        return "Custom"
    profiles = registry.get("profiles") or {}
    return str((profiles.get("default") or {}).get("label") or "Default")


def feature_config_from_project(
    registry: dict[str, Any],
    project: dict[str, Any] | None,
) -> dict[str, Any]:
    """Map a feature-registry project to build-config feature selection."""
    if not project:
        return default_feature_config(registry)
    if "feature_names" in project or "enabled_features" in project:
        feats = {
            str(n)
            for n in (project.get("feature_names") or project.get("enabled_features") or [])
        }
        groups = sync_enabled_groups_from_features(registry, feats)
        profile = "custom"
        if is_default_feature_selection(registry, groups, sorted(feats)):
            profile = "default"
        return read_feature_config(
            registry,
            profile=profile,
            enabled_groups=groups,
            enabled_features=feats,
        )
    group_ids = project.get("group_ids") or []
    if group_ids:
        groups = normalize_enabled_groups(registry, {str(g) for g in group_ids})
        feats = set(export_feature_columns(registry, groups))
        profile = "custom"
        if is_default_feature_selection(registry, groups, sorted(feats)):
            profile = "default"
        return read_feature_config(
            registry,
            profile=profile,
            enabled_groups=groups,
            enabled_features=feats,
        )
    return default_feature_config(registry)

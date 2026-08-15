"""Persistent feature registry store — stable IDs, imports, merges, versioning."""

from __future__ import annotations

import json
import os
import re
from copy import deepcopy
from datetime import datetime, timezone
from typing import Any

_VALID_STATUSES = frozenset({
    "implemented", "planned", "in_progress", "not_implemented",
    "deprecated", "experimental",
})
_VALID_DTYPES = frozenset({"float", "int", "bool", "string"})
_VALID_IMPORT_TYPES = frozenset({"new_group", "existing_group", "merge_registry", "preview_only"})
_VALID_CONFLICT_POLICIES = frozenset({"skip", "replace", "ask_each", "rename"})
_FEATURE_NAME_RE = re.compile(r"^[a-z][a-z0-9_]*$")

_COMPARE_FIELDS = ("description", "formula", "implementation_status", "group", "dependencies", "expected_data_type", "expected_range")
_FEATURE_ID_RE = re.compile(r"^FR\d+$", re.IGNORECASE)


def store_path(data_dir: str) -> str:
    return os.path.join(data_dir, "feature_registry_store.json")


def _load_json(path: str) -> dict[str, Any]:
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _empty_store() -> dict[str, Any]:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    return {
        "registry_version": "1.0",
        "created_by": "System",
        "created_on": now,
        "description": "Feature registry overlay",
        "next_feature_id_seq": 1,
        "feature_ids": {},
        "feature_identities": {},
        "custom_groups": {},
        "projects": {},
        "overrides": {},
        "imported_features": {},
        "disabled_features": {},
        "deleted_feature_ids": {},
        "history": [],
    }


DISABLED_GROUP_ID = "disabled"


def load_store(data_dir: str) -> dict[str, Any]:
    doc = _load_json(store_path(data_dir))
    if not doc:
        return _empty_store()
    doc.setdefault("registry_version", "1.0")
    doc.setdefault("feature_ids", {})
    doc.setdefault("feature_identities", {})
    doc.setdefault("custom_groups", {})
    doc.setdefault("projects", {})
    doc.setdefault("overrides", {})
    doc.setdefault("imported_features", {})
    doc.setdefault("disabled_features", {})
    doc.setdefault("deleted_feature_ids", {})
    doc.setdefault("history", [])
    doc.setdefault("next_feature_id_seq", len(doc.get("feature_ids") or {}) + 1)
    return doc


def save_store(data_dir: str, doc: dict[str, Any]) -> None:
    path = store_path(data_dir)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=2)


def format_feature_id(seq: int) -> str:
    return f"FR{seq:04d}"


_DEFAULT_OWNER_BY_GROUP: dict[str, str] = {
    "price": "Price Module",
    "dgt_reiv": "DGT REIV Module",
    "ratio": "Fair-Value Ratio Module",
    "greeks": "Greeks Module",
    "iv": "IV Module",
    "iv_zscore": "IV Z-Score Module",
    "iv_ema_ratio": "IV EMA Ratio Module",
    "oi": "Open Interest Module",
    "volume": "Volume Module",
    "momentum": "Momentum Module",
    "time": "Session Module",
    "moneyness": "Moneyness Module",
    "ltp_to_spot": "LTP-to-Spot Module",
    "ltp_to_others": "LTP-to-Others Ratio Module",
    "spot_and_other_ratio": "Spot-and-Other Ratio Module",
    "atm_straddle": "ATM Straddle Module",
    "atm6_ltp": "ATM+6 LTP Module",
    "chain": "Chain Module",
    "historical": "Historical Module",
    "spot_hl": "Spot HL Module",
    "advanced": "Advanced Module",
}


def default_owner_for_group(group_id: str | None) -> str:
    gid = str(group_id or "advanced").strip()
    return _DEFAULT_OWNER_BY_GROUP.get(gid, f"{gid.replace('_', ' ').title()} Module")


def _max_id_seq(store: dict[str, Any]) -> int:
    max_n = 0
    for fid in set(list((store.get("feature_identities") or {}).keys()) + list((store.get("feature_ids") or {}).values())):
        m = _FEATURE_ID_RE.match(str(fid))
        if m:
            max_n = max(max_n, int(str(fid)[2:]))
    return max_n


def _migrate_identities(store: dict[str, Any]) -> None:
    identities: dict[str, dict[str, Any]] = dict(store.get("feature_identities") or {})
    if identities:
        return
    now = datetime.now(timezone.utc).isoformat()
    for name, fid in (store.get("feature_ids") or {}).items():
        if fid in identities:
            continue
        identities[fid] = {
            "feature_id": fid,
            "name": name,
            "display_name": None,
            "previous_names": [],
            "created_at": now,
            "created_by": store.get("created_by") or "System",
            "updated_at": now,
            "version": "1.0",
            "owner": None,
            "group_id": None,
        }
    store["feature_identities"] = identities


def _next_free_feature_id(store: dict[str, Any]) -> str:
    used = set((store.get("feature_identities") or {}).keys())
    used.update((store.get("feature_ids") or {}).values())
    seq = max(int(store.get("next_feature_id_seq") or 1), _max_id_seq(store) + 1)
    while True:
        fid = format_feature_id(seq)
        if fid not in used:
            store["next_feature_id_seq"] = seq + 1
            return fid
        seq += 1


def _touch_identity(
    store: dict[str, Any],
    feature_id: str,
    *,
    name: str | None = None,
    display_name: str | None = None,
    group_id: str | None = None,
    owner: str | None = None,
    created_by: str | None = None,
    bump_version: bool = False,
) -> dict[str, Any]:
    identities: dict[str, dict[str, Any]] = dict(store.get("feature_identities") or {})
    ids: dict[str, str] = dict(store.get("feature_ids") or {})
    now = datetime.now(timezone.utc).isoformat()
    ident = dict(identities.get(feature_id) or {})
    is_new = not ident

    if is_new:
        ident = {
            "feature_id": feature_id,
            "name": name or "",
            "display_name": display_name,
            "previous_names": [],
            "created_at": now,
            "created_by": created_by or store.get("created_by") or "System",
            "updated_at": now,
            "version": "1.0",
            "owner": owner or default_owner_for_group(group_id),
            "group_id": group_id,
        }
    else:
        old_name = str(ident.get("name") or "")
        if name and name != old_name and old_name:
            prev = list(ident.get("previous_names") or [])
            if old_name not in prev:
                prev.append(old_name)
            ident["previous_names"] = prev
            if old_name in ids and ids[old_name] == feature_id:
                del ids[old_name]
        if name:
            ident["name"] = name
        if display_name:
            ident["display_name"] = display_name
        if group_id:
            ident["group_id"] = group_id
        if owner:
            ident["owner"] = owner
        elif not ident.get("owner") and group_id:
            ident["owner"] = default_owner_for_group(group_id)
        ident["updated_at"] = now
        if bump_version:
            try:
                ver = float(str(ident.get("version") or "1.0"))
                ident["version"] = f"{ver + 0.1:.1f}".rstrip("0").rstrip(".")
            except ValueError:
                ident["version"] = "1.1"

    if name:
        ids[name] = feature_id
    identities[feature_id] = ident
    store["feature_ids"] = ids
    store["feature_identities"] = identities
    return ident


def ensure_feature_identities(
    store: dict[str, Any],
    features: list[dict[str, Any]],
    *,
    created_by: str | None = None,
) -> dict[str, str]:
    """Assign stable FR#### IDs and identity records; returns name → feature_id."""
    _migrate_identities(store)
    name_to_id: dict[str, str] = dict(store.get("feature_ids") or {})

    for feat in features:
        name = str(feat.get("name") or "").strip()
        if not name:
            continue
        preset = str(feat.get("feature_id") or "").strip().upper() or None
        group_id = str(feat.get("group_id") or feat.get("group") or "").strip() or None
        display_name = feat.get("display_name")

        if name in name_to_id:
            fid = name_to_id[name]
            _touch_identity(
                store, fid, name=name, display_name=display_name, group_id=group_id,
                owner=feat.get("owner"), created_by=created_by,
            )
            continue
        if preset and _FEATURE_ID_RE.match(preset):
            fid = preset
        else:
            fid = _next_free_feature_id(store)

        name_to_id[name] = fid
        _touch_identity(
            store,
            fid,
            name=name,
            display_name=display_name,
            group_id=group_id,
            owner=feat.get("owner"),
            created_by=created_by,
        )

    store["feature_ids"] = name_to_id
    return name_to_id


def ensure_feature_ids(store: dict[str, Any], names: list[str]) -> dict[str, str]:
    """Backward-compatible wrapper — assigns IDs from feature names only."""
    features = [{"name": n} for n in names if n]
    return ensure_feature_identities(store, features)


def allocate_feature_identity(
    data_dir: str,
    name: str,
    *,
    group: str | None = None,
    display_name: str | None = None,
    owner: str | None = None,
    created_by: str | None = None,
) -> str:
    """System-assigned stable ID for a newly added feature (never duplicates)."""
    store = load_store(data_dir)
    _migrate_identities(store)
    ids = store.get("feature_ids") or {}
    if name in ids:
        save_store(data_dir, store)
        return ids[name]
    fid = _next_free_feature_id(store)
    _touch_identity(
        store,
        fid,
        name=name,
        display_name=display_name,
        group_id=group,
        owner=owner,
        created_by=created_by or "User",
    )
    save_store(data_dir, store)
    return fid


def resolve_feature_ref(
    ref: str,
    *,
    name_to_id: dict[str, str],
    id_to_name: dict[str, str],
    features_by_name: dict[str, dict[str, Any]],
) -> dict[str, Any] | None:
    """Resolve a dependency name or FR#### to a catalog cross-reference."""
    text = str(ref or "").strip()
    if not text:
        return None
    if _FEATURE_ID_RE.match(text):
        fid = text.upper()
        name = id_to_name.get(fid)
        if not name:
            return {"feature_id": fid, "name": None, "display_name": fid, "ref": text}
        feat = features_by_name.get(name) or {}
        return {
            "feature_id": fid,
            "name": name,
            "display_name": feat.get("display_name") or name,
            "ref": text,
        }
    if text in name_to_id:
        feat = features_by_name.get(text) or {}
        return {
            "feature_id": name_to_id[text],
            "name": text,
            "display_name": feat.get("display_name") or text,
            "ref": text,
        }
    return {"feature_id": None, "name": text, "display_name": text, "ref": text}


def resolve_dependency_refs(
    dependencies: list[str],
    *,
    name_to_id: dict[str, str],
    id_to_name: dict[str, str],
    features_by_name: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for dep in dependencies or []:
        if dep in ("timestamp", "token", "symbol"):
            continue
        row = resolve_feature_ref(
            dep,
            name_to_id=name_to_id,
            id_to_name=id_to_name,
            features_by_name=features_by_name,
        )
        if row:
            out.append(row)
    return out


def _normalize_expected_range(raw: Any) -> str | list[float] | None:
    if raw is None or raw == "":
        return None
    if isinstance(raw, (list, tuple)) and len(raw) == 2:
        try:
            return [float(raw[0]), float(raw[1])]
        except (TypeError, ValueError):
            return [raw[0], raw[1]]
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return None
        if "," in text:
            parts = [p.strip() for p in text.split(",", 1)]
            if len(parts) == 2:
                try:
                    return [float(parts[0]), float(parts[1])]
                except ValueError:
                    pass
        return text
    return str(raw)


def _parse_name_and_feature_id(raw: dict[str, Any]) -> tuple[str, str | None]:
    name = str(raw.get("name") or "").strip()
    feature_id = str(raw.get("feature_id") or "").strip() or None
    raw_id = str(raw.get("id") or "").strip()
    if raw_id and _FEATURE_ID_RE.match(raw_id):
        feature_id = (feature_id or raw_id).upper()
    elif not name and raw_id:
        name = raw_id
    return name, feature_id


def _normalize_feature_row(raw: dict[str, Any]) -> dict[str, Any]:
    name, feature_id = _parse_name_and_feature_id(raw)
    group = str(raw.get("group") or raw.get("group_id") or "advanced").strip()
    deps = list(raw.get("dependencies") or raw.get("depends_on") or [])
    inputs = list(raw.get("inputs_required") or deps)
    status = str(
        raw.get("implementation_status") or raw.get("status") or "planned"
    ).strip().lower()
    return {
        "name": name,
        "feature_id": feature_id,
        "group": group,
        "display_name": str(raw.get("display_name") or "").strip() or None,
        "description": str(raw.get("description") or "").strip(),
        "why_needed": str(raw.get("why_needed") or raw.get("why") or "").strip(),
        "formula": str(raw.get("formula") or raw.get("formula_doc") or raw.get("formula_ref") or "").strip(),
        "dependencies": [str(d) for d in deps if d],
        "inputs_required": [str(i) for i in inputs if i],
        "expected_data_type": str(raw.get("expected_data_type") or raw.get("unit") or "float").strip(),
        "expected_range": _normalize_expected_range(raw.get("expected_range")),
        "implementation_status": status,
        "priority": str(raw.get("priority") or "medium").strip(),
        "notes": str(raw.get("notes") or raw.get("developer_notes") or "").strip(),
        "tags": list(raw.get("tags") or []),
        "source_model": str(raw.get("source_model") or "").strip() or None,
        "implementation_module": raw.get("implementation_module"),
        "implementation_function": raw.get("implementation_function"),
    }


def parse_import_payload(raw: Any) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any] | None]:
    """Accept export bundles, group packs, arrays, or single-feature objects."""
    meta: dict[str, Any] = {}
    group: dict[str, Any] | None = None
    rows: list[dict[str, Any]] = []

    if isinstance(raw, list):
        rows = [r for r in raw if isinstance(r, dict)]
    elif isinstance(raw, dict):
        meta = {
            k: raw[k]
            for k in ("registry_version", "created_by", "created_on", "description", "exported_at")
            if k in raw
        }
        if isinstance(raw.get("group"), dict):
            group = dict(raw["group"])
        elif raw.get("group_id") or raw.get("group_label"):
            group = {
                "id": str(raw.get("group_id") or "").strip(),
                "label": str(raw.get("group_label") or raw.get("group_name") or "").strip(),
            }
        feat_src = raw.get("features") or raw.get("filtered_features") or raw.get("imported_features")
        if isinstance(feat_src, list):
            rows = [r for r in feat_src if isinstance(r, dict)]
        elif raw.get("name"):
            rows = [raw]
        elif isinstance(raw.get("imported_features"), dict):
            rows = list(raw["imported_features"].values())
    else:
        raise ValueError("Import JSON must be an object or array of features")

    normalized = [_normalize_feature_row(r) for r in rows if r.get("name") or r.get("id")]
    if not normalized:
        raise ValueError("No features found in import JSON")
    return meta, normalized, group


def _known_groups(store: dict[str, Any], catalog_groups: list[dict[str, Any]]) -> set[str]:
    known = {str(g.get("id")) for g in catalog_groups if g.get("id")}
    known.update(store.get("custom_groups") or {})
    return known


def _all_feature_names(
    catalog_features: list[dict[str, Any]],
    store: dict[str, Any],
) -> set[str]:
    names = {str(f.get("name")) for f in catalog_features if f.get("name")}
    names.update(store.get("imported_features") or {})
    names.update(store.get("overrides") or {})
    return names


def _detect_cycles(deps_map: dict[str, list[str]]) -> list[str]:
    cycles: list[str] = []

    def visit(node: str, stack: list[str], seen: set[str]) -> None:
        if node in stack:
            cycles.append(" → ".join(stack + [node]))
            return
        if node in seen:
            return
        seen.add(node)
        stack.append(node)
        for dep in deps_map.get(node) or []:
            if dep in deps_map:
                visit(dep, stack, seen)
        stack.pop()

    for name in deps_map:
        visit(name, [], set())
    return cycles


def validate_import(
    *,
    store: dict[str, Any],
    catalog_features: list[dict[str, Any]],
    catalog_groups: list[dict[str, Any]],
    incoming: list[dict[str, Any]],
    import_type: str,
    target_group: str | None = None,
    new_group: dict[str, Any] | None = None,
) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    checks: list[dict[str, str]] = []

    if import_type not in _VALID_IMPORT_TYPES:
        errors.append(f"Invalid import type: {import_type}")

    names_in_batch = [f["name"] for f in incoming]
    dupes = {n for n in names_in_batch if names_in_batch.count(n) > 1}
    if dupes:
        errors.append(f"Duplicate feature names in import: {', '.join(sorted(dupes))}")
    checks.append({"id": "duplicate_names", "status": "fail" if dupes else "pass"})

    known_groups = _known_groups(store, catalog_groups)
    if import_type == "new_group":
        gid = str((new_group or {}).get("id") or "").strip()
        if not gid:
            errors.append("New group import requires group id")
        elif gid in known_groups and not (store.get("custom_groups") or {}).get(gid):
            warnings.append(f"Group `{gid}` already exists — features will be added to it")
    elif import_type == "existing_group":
        if not target_group:
            errors.append("Existing group import requires target_group")
        elif target_group not in known_groups:
            errors.append(f"Invalid target group: {target_group}")
    checks.append({"id": "invalid_group", "status": "fail" if any("group" in e.lower() for e in errors) else "pass"})

    registry_names = _all_feature_names(catalog_features, store)
    batch_set = set(names_in_batch)

    missing_desc = [f["name"] for f in incoming if not f.get("description")]
    if missing_desc:
        warnings.append(f"Missing description: {', '.join(missing_desc[:8])}" + ("…" if len(missing_desc) > 8 else ""))
    checks.append({"id": "missing_description", "status": "warn" if missing_desc else "pass"})

    missing_deps: list[str] = []
    deps_map = {f["name"]: list(f.get("dependencies") or []) for f in incoming}
    for f in incoming:
        for dep in f.get("dependencies") or []:
            if dep not in registry_names and dep not in batch_set and dep not in ("timestamp", "token", "symbol"):
                missing_deps.append(f"{f['name']} → {dep}")
    if missing_deps:
        warnings.append("Missing dependencies: " + "; ".join(missing_deps[:6]) + ("…" if len(missing_deps) > 6 else ""))
    checks.append({"id": "missing_dependency", "status": "warn" if missing_deps else "pass"})

    cycles = _detect_cycles(deps_map)
    if cycles:
        errors.append("Circular dependency: " + cycles[0])
    checks.append({"id": "circular_dependency", "status": "fail" if cycles else "pass"})

    bad_status = [f["name"] for f in incoming if f.get("implementation_status") not in _VALID_STATUSES]
    if bad_status:
        errors.append(f"Invalid implementation status: {', '.join(bad_status[:6])}")
    checks.append({"id": "invalid_implementation_status", "status": "fail" if bad_status else "pass"})

    bad_dtype = [f["name"] for f in incoming if f.get("expected_data_type") not in _VALID_DTYPES]
    if bad_dtype:
        errors.append(f"Invalid feature type: {', '.join(bad_dtype[:6])}")
    checks.append({"id": "invalid_feature_type", "status": "fail" if bad_dtype else "pass"})

    bad_names = [f["name"] for f in incoming if not _FEATURE_NAME_RE.match(f["name"] or "")]
    if bad_names:
        errors.append(f"Invalid feature names (use snake_case): {', '.join(bad_names[:6])}")

    # Ownership gate: new historical/derived names must not enter the Registry.
    from .feature_ownership import evaluate_registry_admission

    admission_blocked: list[str] = []
    for f in incoming:
        name = str(f.get("name") or "").strip()
        if not name or name in registry_names:
            continue  # updates to existing entries are not re-admissions
        result = evaluate_registry_admission(
            name,
            ownership=f.get("ownership") or f.get("owner") or f.get("category"),
            requires_prior_rows=f.get("requires_prior_rows"),
            allow_historical=bool(f.get("allow_historical")),
            historical_exception_reason=f.get("historical_exception_reason"),
            produced_by=f.get("produced_by") or f.get("generator"),
            dataset_builder_configurable=(
                f.get("dataset_builder_configurable")
                if f.get("dataset_builder_configurable") is not None
                else f.get("configurable_in_dataset_builder")
            ),
            generic_registry_math=(
                f.get("generic_registry_math")
                if f.get("generic_registry_math") is not None
                else f.get("derived_solely_from_registry_math")
            ),
            foundational_market_observation=(
                f.get("foundational_market_observation")
                if f.get("foundational_market_observation") is not None
                else f.get("raw_market_observation")
            ),
            raw_market_observation=f.get("raw_market_observation"),
            canonical_controller_or_market_model=f.get(
                "canonical_controller_or_market_model"
            ),
            recreatable_from_registry_or_helpers=f.get(
                "recreatable_from_registry_or_helpers"
            ),
        )
        if not result.get("allowed"):
            admission_blocked.append(f"{name}: {result.get('reason')}")
    if admission_blocked:
        errors.append(
            "Registry admission denied (semantic ownership gate → Transformation Pipeline): "
            + "; ".join(admission_blocked[:4])
            + ("…" if len(admission_blocked) > 4 else "")
        )
    checks.append({
        "id": "ownership_admission",
        "status": "fail" if admission_blocked else "pass",
    })

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "checks": checks,
    }


def _feature_snapshot(f: dict[str, Any]) -> dict[str, Any]:
    return {
        "description": f.get("description") or "",
        "formula": f.get("formula") or "",
        "implementation_status": f.get("implementation_status") or "",
        "group": f.get("group_id") or f.get("group") or "",
        "dependencies": sorted(f.get("dependencies") or []),
        "expected_data_type": f.get("expected_data_type") or "",
        "expected_range": f.get("expected_range"),
    }


def _diff_feature(current: dict[str, Any], incoming: dict[str, Any]) -> dict[str, Any]:
    cur = {
        "description": current.get("description") or "",
        "formula": current.get("formula") or "",
        "implementation_status": current.get("implementation_status") or "",
        "group": current.get("group_id") or current.get("group") or "",
        "dependencies": sorted(current.get("dependencies") or []),
        "expected_data_type": current.get("expected_data_type") or "",
        "expected_range": current.get("expected_range"),
    }
    inc = {
        "description": incoming.get("description") or "",
        "formula": incoming.get("formula") or "",
        "implementation_status": incoming.get("implementation_status") or "",
        "group": incoming.get("group") or "",
        "dependencies": sorted(incoming.get("dependencies") or []),
        "expected_data_type": incoming.get("expected_data_type") or "",
        "expected_range": incoming.get("expected_range"),
    }
    changed: dict[str, dict[str, Any]] = {}
    for field in _COMPARE_FIELDS:
        if cur.get(field) != inc.get(field):
            changed[field] = {"current": cur.get(field), "incoming": inc.get(field)}
    return changed


def _catalog_by_name(catalog_features: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(f["name"]): f for f in catalog_features if f.get("name")}


def preview_import(
    *,
    store: dict[str, Any],
    catalog_features: list[dict[str, Any]],
    catalog_groups: list[dict[str, Any]],
    incoming: list[dict[str, Any]],
    import_type: str,
    target_group: str | None = None,
    new_group: dict[str, Any] | None = None,
    conflict_policy: str = "skip",
    resolutions: dict[str, str] | None = None,
) -> dict[str, Any]:
    validation = validate_import(
        store=store,
        catalog_features=catalog_features,
        catalog_groups=catalog_groups,
        incoming=incoming,
        import_type=import_type,
        target_group=target_group,
        new_group=new_group,
    )
    if not validation["valid"]:
        return {"ok": False, "validation": validation, "preview": None}

    by_name = _catalog_by_name(catalog_features)
    registry_names = _all_feature_names(catalog_features, store)
    resolutions = resolutions or {}

    new_features: list[dict[str, Any]] = []
    conflicts: list[dict[str, Any]] = []
    modified: list[dict[str, Any]] = []
    unchanged: list[str] = []
    skipped: list[str] = []

    effective_group = target_group
    group_label = None
    if import_type == "new_group" and new_group:
        effective_group = str(new_group.get("id") or "").strip()
        group_label = str(new_group.get("label") or effective_group).strip()

    for row in incoming:
        name = row["name"]
        exists = name in registry_names
        if import_type == "merge_registry":
            if not exists:
                new_features.append(row)
            else:
                diff = _diff_feature(by_name.get(name, row), row)
                if diff:
                    modified.append({"name": name, "changes": diff, "incoming": row, "current": by_name.get(name)})
                else:
                    unchanged.append(name)
            continue

        if not exists:
            feat = dict(row)
            if effective_group:
                feat["group"] = effective_group
            new_features.append(feat)
            continue

        # conflict
        policy = resolutions.get(name) or conflict_policy
        if policy == "ask_each" and name not in resolutions:
            conflicts.append({
                "name": name,
                "current": _feature_snapshot(by_name.get(name, {})),
                "incoming": _feature_snapshot(row),
                "diff": _diff_feature(by_name.get(name, {}), row),
                "needs_resolution": True,
            })
            continue
        if policy in ("skip", "ask_each") and name not in resolutions:
            skipped.append(name)
            conflicts.append({
                "name": name,
                "action": "skip",
                "current": _feature_snapshot(by_name.get(name, {})),
                "incoming": _feature_snapshot(row),
                "diff": _diff_feature(by_name.get(name, {}), row),
            })
        elif policy == "replace" or resolutions.get(name) == "replace":
            modified.append({
                "name": name,
                "changes": _diff_feature(by_name.get(name, {}), row),
                "incoming": row,
                "current": by_name.get(name),
                "action": "replace",
            })
        elif policy == "rename" or resolutions.get(name) == "rename":
            suffix = 2
            new_name = f"{name}_imported"
            while new_name in registry_names or any(r["name"] == new_name for r in new_features):
                new_name = f"{name}_v{suffix}"
                suffix += 1
            renamed = dict(row)
            renamed["name"] = new_name
            renamed["display_name"] = renamed.get("display_name") or new_name.replace("_", " ").title()
            if effective_group:
                renamed["group"] = effective_group
            new_features.append(renamed)
            conflicts.append({"name": name, "action": "rename", "renamed_to": new_name})
        else:
            skipped.append(name)

    preview: dict[str, Any] = {
        "import_type": import_type,
        "validation": validation,
        "new_count": len(new_features),
        "conflict_count": len(conflicts),
        "modified_count": len(modified),
        "unchanged_count": len(unchanged),
        "skipped_count": len(skipped),
        "new_features": [{"name": f["name"], "group": f.get("group")} for f in new_features],
        "conflicts": conflicts,
        "modified": modified,
        "unchanged": unchanged,
        "skipped": skipped,
        "can_apply": import_type != "preview_only" and not any(c.get("needs_resolution") for c in conflicts),
    }

    if import_type == "new_group":
        preview["group"] = {"id": effective_group, "label": group_label or effective_group}
        preview["headline"] = "New Group"
    elif import_type == "existing_group":
        grp = next((g for g in catalog_groups if g.get("id") == target_group), {})
        preview["group"] = {"id": target_group, "label": grp.get("label") or target_group}
        preview["headline"] = "Existing Group"
    elif import_type == "merge_registry":
        preview["headline"] = "Registry Comparison"
        preview["current_count"] = len(catalog_features)
        preview["incoming_count"] = len(incoming)
        preview["summary"] = {
            "new": len(new_features),
            "modified": len(modified),
            "unchanged": len(unchanged),
        }

    if import_type == "preview_only":
        preview["can_apply"] = False

    return {"ok": True, "validation": validation, "preview": preview}


def apply_import(
    *,
    data_dir: str,
    store: dict[str, Any],
    catalog_features: list[dict[str, Any]],
    catalog_groups: list[dict[str, Any]],
    incoming: list[dict[str, Any]],
    import_type: str,
    target_group: str | None = None,
    new_group: dict[str, Any] | None = None,
    conflict_policy: str = "skip",
    resolutions: dict[str, str] | None = None,
    import_meta: dict[str, Any] | None = None,
    bulk_defaults: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if import_type == "preview_only":
        raise ValueError("Cannot apply preview_only import")

    preview_result = preview_import(
        store=store,
        catalog_features=catalog_features,
        catalog_groups=catalog_groups,
        incoming=incoming,
        import_type=import_type,
        target_group=target_group,
        new_group=new_group,
        conflict_policy=conflict_policy,
        resolutions=resolutions,
    )
    if not preview_result.get("ok"):
        raise ValueError("; ".join(preview_result.get("validation", {}).get("errors") or ["Invalid import"]))
    preview = preview_result["preview"]
    if not preview.get("can_apply"):
        raise ValueError("Unresolved conflicts — resolve before applying")

    incoming_by_name = {r["name"]: r for r in incoming}
    store = deepcopy(store)
    now = datetime.now(timezone.utc).isoformat()
    imported_names: list[str] = []

    if import_type == "new_group" and new_group:
        gid = str(new_group.get("id") or "").strip()
        label = str(new_group.get("label") or gid).strip()
        custom = dict(store.get("custom_groups") or {})
        custom[gid] = {"label": label, "filter": label}
        store["custom_groups"] = custom

    overrides = dict(store.get("overrides") or {})
    imported = dict(store.get("imported_features") or {})
    by_name = _catalog_by_name(catalog_features)
    registry_names = _all_feature_names(catalog_features, store)

    effective_group = target_group
    if import_type == "new_group" and new_group:
        effective_group = str(new_group.get("id") or "").strip()

    applied: set[str] = set()

    def _apply_row(name: str, row: dict[str, Any]) -> None:
        if name in applied:
            return
        applied.add(name)
        feat = dict(row)
        if bulk_defaults:
            feat.update({k: v for k, v in bulk_defaults.items() if v is not None})
        if effective_group and import_type in ("new_group", "existing_group"):
            feat["group"] = effective_group
        exists_in_schema = name in by_name
        if exists_in_schema or name in overrides:
            _apply_overlay(overrides, imported, name, feat, by_name, exists_in_schema=exists_in_schema, now=now)
        else:
            imported[name] = _store_feature_record(feat, now=now)
        imported_names.append(name)

    for item in preview.get("new_features") or []:
        fname = str(item.get("name") or "")
        row = incoming_by_name.get(fname)
        if not row:
            row = next((r for r in incoming if r["name"] != fname and fname.endswith(r["name"])), None)
        if row:
            apply_row = dict(row)
            apply_row["name"] = fname
            _apply_row(fname, apply_row)

    for mod in preview.get("modified") or []:
        name = mod.get("name")
        row = mod.get("incoming") or incoming_by_name.get(name)
        if name and row:
            _apply_row(name, row)

    for conflict in preview.get("conflicts") or []:
        if conflict.get("action") == "rename" and conflict.get("renamed_to"):
            src = incoming_by_name.get(conflict["name"])
            if src:
                feat = dict(src)
                feat["name"] = conflict["renamed_to"]
                _apply_row(conflict["renamed_to"], feat)

    store["overrides"] = overrides
    store["imported_features"] = imported

    for row in incoming:
        if row["name"] not in imported_names:
            continue
        preset = row.get("feature_id")
        ids = store.get("feature_ids") or {}
        fid = preset or ids.get(row["name"])
        if fid:
            _touch_identity(
                store,
                str(fid).upper(),
                name=row["name"],
                display_name=row.get("display_name"),
                group_id=row.get("group"),
                owner=row.get("owner"),
                created_by=(import_meta or {}).get("created_by") or "Import",
                bump_version=True,
            )

    old_ver = str(store.get("registry_version") or "1.0")
    try:
        parts = old_ver.split(".")
        major, minor = int(parts[0]), int(parts[1]) if len(parts) > 1 else 0
        new_ver = f"{major}.{minor + 1}"
    except (ValueError, IndexError):
        new_ver = "1.1"
    store["registry_version"] = new_ver

    summary_parts = []
    if import_type == "new_group" and new_group:
        summary_parts.append(f"+ {new_group.get('label') or new_group.get('id')}")
    elif import_meta and import_meta.get("description"):
        summary_parts.append(import_meta["description"])
    else:
        summary_parts.append(f"Import {len(imported_names)} features")

    new_total = len(catalog_features) + len([n for n in imported_names if n not in registry_names])
    history = list(store.get("history") or [])
    history.append({
        "version": new_ver,
        "timestamp": now,
        "import_type": import_type,
        "summary": " ".join(summary_parts),
        "feature_count": new_total,
        "imported": len(imported_names),
        "created_by": (import_meta or {}).get("created_by"),
    })
    store["history"] = history[-50:]
    save_store(data_dir, store)

    return {
        "ok": True,
        "registry_version": new_ver,
        "imported_count": len(imported_names),
        "imported_names": imported_names,
        "history_entry": history[-1],
    }


def _store_feature_record(row: dict[str, Any], *, now: str) -> dict[str, Any]:
    return {
        "name": row["name"],
        "group": row.get("group") or "advanced",
        "display_name": row.get("display_name"),
        "description": row.get("description") or "",
        "why_needed": row.get("why_needed") or "",
        "formula": row.get("formula") or "",
        "dependencies": list(row.get("dependencies") or []),
        "inputs_required": list(row.get("inputs_required") or row.get("dependencies") or []),
        "expected_data_type": row.get("expected_data_type") or "float",
        "expected_range": row.get("expected_range"),
        "implementation_status": row.get("implementation_status") or "planned",
        "priority": row.get("priority") or "medium",
        "notes": row.get("notes") or "",
        "tags": list(row.get("tags") or []),
        "source_model": row.get("source_model"),
        "implementation_module": row.get("implementation_module"),
        "implementation_function": row.get("implementation_function"),
        "created_at": now,
        "updated_at": now,
    }


def _apply_overlay(
    overrides: dict[str, Any],
    imported: dict[str, Any],
    name: str,
    row: dict[str, Any],
    by_name: dict[str, dict[str, Any]],
    *,
    exists_in_schema: bool,
    now: str,
) -> None:
    record = _store_feature_record(row, now=now)
    if exists_in_schema:
        overrides[name] = {k: v for k, v in record.items() if k != "name" and v not in (None, "", [])}
        overrides[name]["updated_at"] = now
    else:
        imported[name] = record


def disabled_registry_feature_names(store: dict[str, Any]) -> set[str]:
    """Feature names marked inactive in the registry overlay."""
    disabled = store.get("disabled_features") or {}
    return {str(k) for k in disabled.keys()}


def set_feature_registry_active(
    data_dir: str,
    store: dict[str, Any],
    name: str,
    *,
    active: bool,
    home_group_id: str | None = None,
) -> dict[str, Any]:
    """Enable or disable a catalog feature for dataset builds."""
    feature_name = str(name or "").strip()
    if not feature_name:
        raise ValueError("feature name is required")
    store = deepcopy(store)
    disabled = dict(store.get("disabled_features") or {})
    now = datetime.now(timezone.utc).isoformat()

    if active:
        if feature_name not in disabled:
            return {"ok": True, "name": feature_name, "active": True, "changed": False}
        disabled.pop(feature_name)
    else:
        if feature_name in disabled:
            return {"ok": True, "name": feature_name, "active": False, "changed": False}
        gid = str(home_group_id or "").strip()
        if not gid or gid == DISABLED_GROUP_ID:
            raise ValueError("home_group_id is required when disabling a feature")
        disabled[feature_name] = {
            "home_group_id": gid,
            "disabled_at": now,
        }

    store["disabled_features"] = disabled
    history = list(store.get("history") or [])
    history.append({
        "action": "enable_feature" if active else "disable_feature",
        "name": feature_name,
        "at": now,
    })
    store["history"] = history[-200:]
    save_store(data_dir, store)
    return {"ok": True, "name": feature_name, "active": active, "changed": True}


def bulk_update_features(
    data_dir: str,
    store: dict[str, Any],
    names: list[str],
    updates: dict[str, Any],
) -> dict[str, Any]:
    if not names:
        raise ValueError("No features selected for bulk update")
    store = deepcopy(store)
    overrides = dict(store.get("overrides") or {})
    imported = dict(store.get("imported_features") or {})
    now = datetime.now(timezone.utc).isoformat()
    updated: list[str] = []

    allowed = {"group", "implementation_status", "implementation_module", "implementation_function",
               "tags", "source_model", "priority", "notes"}
    patch = {k: v for k, v in updates.items() if k in allowed and v is not None}

    for name in names:
        if name in imported:
            imported[name].update(patch)
            imported[name]["updated_at"] = now
            updated.append(name)
        elif name in overrides:
            overrides[name].update(patch)
            overrides[name]["updated_at"] = now
            updated.append(name)
        else:
            overrides[name] = {**patch, "updated_at": now}
            updated.append(name)

    store["overrides"] = overrides
    store["imported_features"] = imported
    save_store(data_dir, store)
    return {"ok": True, "updated": updated, "count": len(updated)}


_PROJECT_ID_RE = re.compile(r"^[a-z][a-z0-9_]*$")


def _slug_project_id(label: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", str(label or "").strip().lower()).strip("_")
    return slug or "project"


def _slug_project_group_id(label: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", str(label or "").strip().lower()).strip("_")
    return slug or "group"


def _normalize_project_group_list(groups: list[Any] | None) -> list[dict[str, str]]:
    from .feature_project_organization import normalize_custom_project_groups

    return normalize_custom_project_groups(groups)


def _normalize_feature_group_map(raw: dict[str, Any] | None) -> dict[str, str]:
    out: dict[str, str] = {}
    for key, val in dict(raw or {}).items():
        name = str(key or "").strip()
        gid = str(val or "").strip()
        if name and gid:
            out[name] = gid
    return out


def suggest_project_id(store: dict[str, Any] | None, label: str = "") -> str:
    """Suggest a unique snake_case project id (does not reserve it)."""
    projects = dict((store or {}).get("projects") or {})
    base = _slug_project_id(label)
    if not _PROJECT_ID_RE.match(base) or base == "all":
        base = "project"
    if base not in projects:
        return base
    n = 2
    while f"{base}_{n}" in projects:
        n += 1
    candidate = f"{base}_{n}"
    return candidate if _PROJECT_ID_RE.match(candidate) else f"project_{n}"


def ensure_all_project(data_dir: str) -> dict[str, Any]:
    """Ensure reserved ``all`` project exists with active registry membership."""
    from .feature_project_organization import (
        RESERVED_ALL_PROJECT_ID,
        build_default_all_project_doc,
        sync_all_project_membership,
    )

    store = load_store(data_dir)
    projects = dict(store.get("projects") or {})
    pid = RESERVED_ALL_PROJECT_ID
    if pid not in projects:
        doc = build_default_all_project_doc(data_dir=data_dir)
        projects[pid] = doc
        store["projects"] = projects
        save_store(data_dir, store)
        return {"id": pid, **doc}

    raw = dict(projects[pid])
    synced = sync_all_project_membership(raw, data_dir=data_dir)
    if synced != raw:
        synced["updated_at"] = datetime.now(timezone.utc).isoformat()
        projects[pid] = synced
        store["projects"] = projects
        save_store(data_dir, store)
    return {"id": pid, **synced}


def list_projects(store: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    projects = dict((store or {}).get("projects") or {})
    rows = []
    for pid, meta in sorted(projects.items(), key=lambda x: str((x[1] or {}).get("label") or x[0]).lower()):
        if not isinstance(meta, dict):
            continue
        warmup = meta.get("warmup_minutes", meta.get("warmup"))
        try:
            warmup_minutes = int(warmup) if warmup is not None and str(warmup).strip() != "" else None
        except (TypeError, ValueError):
            warmup_minutes = None
        rows.append({
            "id": pid,
            "label": str(meta.get("label") or pid),
            "description": str(meta.get("description") or ""),
            "group_ids": list(meta.get("group_ids") or []),
            "feature_names": list(meta.get("feature_names") or meta.get("enabled_features") or []),
            "project_groups": _normalize_project_group_list(meta.get("project_groups")),
            "feature_group_map": _normalize_feature_group_map(meta.get("feature_group_map")),
            "warmup_minutes": warmup_minutes,
            "default_sampling": str(meta.get("default_sampling") or ""),
            "notes": str(meta.get("notes") or ""),
            "version": str(meta.get("version") or "1"),
            "created_at": meta.get("created_at"),
            "updated_at": meta.get("updated_at"),
        })
    return rows


def create_project(
    data_dir: str,
    *,
    label: str,
    project_id: str | None = None,
    description: str = "",
    group_ids: list[str] | None = None,
    feature_names: list[str] | None = None,
    project_groups: list[dict[str, str]] | list[Any] | None = None,
    feature_group_map: dict[str, str] | None = None,
    warmup_minutes: int | None = None,
    default_sampling: str = "",
    notes: str = "",
    version: str = "1",
) -> dict[str, Any]:
    store = load_store(data_dir)
    projects = dict(store.get("projects") or {})
    text = str(label or "").strip()
    if not text:
        raise ValueError("Project label is required")
    pid = str(project_id or _slug_project_id(text)).strip().lower()
    if not _PROJECT_ID_RE.match(pid):
        raise ValueError("Project id must be snake_case (e.g. options_ml)")
    from .feature_project_organization import is_reserved_all_project_id

    if is_reserved_all_project_id(pid):
        raise ValueError("Project id 'all' is reserved for the default registry project")
    if pid in projects:
        raise ValueError(f"Project already exists: {pid}")
    now = datetime.now(timezone.utc).isoformat()
    doc: dict[str, Any] = {
        "label": text,
        "description": str(description or "").strip(),
        "group_ids": list(dict.fromkeys(str(g).strip() for g in (group_ids or []) if str(g).strip())),
        "feature_names": list(dict.fromkeys(str(n).strip() for n in (feature_names or []) if str(n).strip())),
        "project_groups": _normalize_project_group_list(project_groups),
        "feature_group_map": _normalize_feature_group_map(feature_group_map),
        "warmup_minutes": int(warmup_minutes) if warmup_minutes is not None else None,
        "default_sampling": str(default_sampling or "").strip(),
        "notes": str(notes or "").strip(),
        "version": str(version or "1").strip() or "1",
        "created_at": now,
        "updated_at": now,
    }
    from .feature_project_organization import migrate_project_organization

    doc = migrate_project_organization(doc, data_dir=data_dir)
    projects[pid] = doc
    store["projects"] = projects
    save_store(data_dir, store)
    return {"id": pid, **doc}


def update_project(
    data_dir: str,
    project_id: str,
    *,
    label: str | None = None,
    description: str | None = None,
    group_ids: list[str] | None = None,
    feature_names: list[str] | None = None,
    project_groups: list[dict[str, str]] | list[Any] | None = None,
    feature_group_map: dict[str, str] | None = None,
    warmup_minutes: int | None | object = ...,
    default_sampling: str | None = None,
    notes: str | None = None,
    version: str | None = None,
) -> dict[str, Any]:
    store = load_store(data_dir)
    projects = dict(store.get("projects") or {})
    pid = str(project_id or "").strip().lower()
    if pid not in projects:
        raise ValueError(f"Project not found: {pid}")
    doc = dict(projects[pid])
    if label is not None:
        text = str(label).strip()
        if not text:
            raise ValueError("Project label cannot be empty")
        doc["label"] = text
    if description is not None:
        doc["description"] = str(description).strip()
    if group_ids is not None:
        doc["group_ids"] = list(dict.fromkeys(str(g).strip() for g in group_ids if str(g).strip()))
    if feature_names is not None:
        doc["feature_names"] = list(dict.fromkeys(str(n).strip() for n in feature_names if str(n).strip()))
    if project_groups is not None:
        doc["project_groups"] = _normalize_project_group_list(project_groups)
    if feature_group_map is not None:
        doc["feature_group_map"] = _normalize_feature_group_map(feature_group_map)
    if warmup_minutes is not ...:
        if warmup_minutes is None or str(warmup_minutes).strip() == "":
            doc["warmup_minutes"] = None
        else:
            doc["warmup_minutes"] = int(warmup_minutes)
    if default_sampling is not None:
        doc["default_sampling"] = str(default_sampling).strip()
    if notes is not None:
        doc["notes"] = str(notes).strip()
    if version is not None:
        doc["version"] = str(version).strip() or "1"
    from .feature_project_organization import migrate_project_organization

    doc = migrate_project_organization(doc, data_dir=data_dir)
    doc["updated_at"] = datetime.now(timezone.utc).isoformat()
    projects[pid] = doc
    store["projects"] = projects
    save_store(data_dir, store)
    return {"id": pid, **doc}


def clone_project(
    data_dir: str,
    source_project_id: str,
    *,
    label: str,
    project_id: str | None = None,
) -> dict[str, Any]:
    store = load_store(data_dir)
    projects = dict(store.get("projects") or {})
    src_id = str(source_project_id or "").strip().lower()
    src = projects.get(src_id)
    if not isinstance(src, dict):
        raise ValueError(f"Project not found: {src_id}")
    warm_raw = src.get("warmup_minutes", src.get("warmup"))
    try:
        warm = int(warm_raw) if warm_raw is not None and str(warm_raw).strip() != "" else None
    except (TypeError, ValueError):
        warm = None
    return create_project(
        data_dir,
        label=label,
        project_id=project_id,
        description=str(src.get("description") or ""),
        group_ids=list(src.get("group_ids") or []),
        feature_names=list(src.get("feature_names") or src.get("enabled_features") or []),
        project_groups=list(src.get("project_groups") or []),
        feature_group_map=dict(src.get("feature_group_map") or {}),
        warmup_minutes=warm,
        default_sampling=str(src.get("default_sampling") or ""),
        notes=str(src.get("notes") or ""),
        version="1",
    )


def delete_project(data_dir: str, project_id: str) -> dict[str, Any]:
    from .feature_project_organization import is_reserved_all_project_id

    store = load_store(data_dir)
    projects = dict(store.get("projects") or {})
    pid = str(project_id or "").strip().lower()
    if is_reserved_all_project_id(pid):
        raise ValueError("The reserved project 'all' cannot be deleted")
    if pid not in projects:
        raise ValueError(f"Project not found: {pid}")
    removed = projects.pop(pid)
    store["projects"] = projects
    save_store(data_dir, store)
    return {"deleted": True, "id": pid, "label": removed.get("label")}

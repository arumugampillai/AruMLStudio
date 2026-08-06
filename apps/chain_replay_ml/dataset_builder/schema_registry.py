"""Central dataset schema registry — single source of truth for column metadata."""

from __future__ import annotations

import hashlib
import json
import os
from typing import Any

from path_config import STATIC_DIR

_SCHEMA_PATH = os.path.join(STATIC_DIR, "ml_schema_registry.json")
_LEGACY_FEATURE_REGISTRY_PATH = os.path.join(STATIC_DIR, "ml_feature_registry.json")
_INTRODUCED_VERSION = "1.4.2"

_SCHEMA_CACHE: dict[str, Any] | None = None
_SCHEMA_CACHE_MTIME: float | None = None


def _compute_content_hash(material: Any) -> str:
    blob = json.dumps(material, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:8].upper()


def schema_registry_path() -> str:
    return _SCHEMA_PATH


def _next_target_column_id(columns: dict[str, Any]) -> int:
    n = 1
    for col in columns.values():
        cid = str(col.get("id") or "")
        if cid.startswith("target_"):
            try:
                n = max(n, int(cid.split("_", 1)[1]) + 1)
            except (IndexError, ValueError):
                pass
    return n


def _sync_builtin_target_definitions(data: dict[str, Any]) -> dict[str, Any]:
    """Merge TARGET_DEFINITIONS into an on-disk schema when new horizons are added."""
    if not data.get("columns"):
        return data
    from .schema_feature_meta import TARGET_DEFINITIONS, build_column_meta_extras, build_targets_registry
    from .schema_implementation import implementation_for_column

    columns = dict(data.get("columns") or {})
    targets = dict(data.get("targets") or {})
    registry = build_targets_registry(TARGET_DEFINITIONS)
    next_idx = _next_target_column_id(columns)

    for target in TARGET_DEFINITIONS:
        name = str(target["name"])
        if name in columns:
            if name not in targets and name in registry:
                targets[name] = registry[name]
            continue
        doc = dict(target)
        extras = build_column_meta_extras(name, col_type="target", doc=doc)
        impl = implementation_for_column(name, formula_ref=doc.get("formula_ref"), doc=doc) or {}
        columns[name] = {
            "id": f"target_{next_idx:03d}",
            "name": name,
            "type": "target",
            "display_name": doc.get("display_name") or name,
            "description": doc.get("description", ""),
            "prediction_horizon_sec": int(target["prediction_horizon_sec"]),
            "nullable": False,
            "formula_ref": doc.get("formula_ref"),
            **extras,
            **impl,
        }
        if name in registry:
            targets[name] = registry[name]
        next_idx += 1

    data["columns"] = columns
    data["targets"] = targets
    return data


_GROUP_LABELS: dict[str, str] = {
    "price": "Price & Returns",
    "market_microstructure": "Market Microstructure",
    "dgt_reiv": "DGT REIV Dynamics",
    "ratio": "Fair-Value Ratios",
    "greeks": "Greeks",
    "iv": "Implied Volatility",
    "iv_zscore": "IV Z-Score",
    "iv_ema_ratio": "IV EMA Ratio Features",
    "oi": "Open Interest",
    "volume": "Volume & Flow",
    "momentum": "Momentum & Realized Vol",
    "sharp_momentum": "Sharp Momentum",
    "spot_hl": "Spot HL",
    "time": "Session & Time",
    "moneyness": "Moneyness",
    "ltp_to_spot": "LTP/Spot",
    "ltp_to_others": "LTP to Others Ratio",
    "spot_and_other_ratio": "Spot and Other Ratio",
    "historic_spot_ema": "Historic Spot EMA (NIFTY bars)",
    "atm_straddle": "ATM Straddle",
    "atm6_ltp": "ATM+6 LTP",
    "chain": "Chain-Wide",
    "chain_flow": "Chain Flow",
    "historical": "Historical OHLC",
    "advanced": "Advanced",
}

_METADATA_DEFINITIONS: dict[str, dict[str, Any]] = {
    "trading_day": {
        "display_name": "Trading Day",
        "description": "Calendar trading date for the replay session.",
        "interpretation": "Partitions rows by session; required for multi-day datasets.",
        "formula_ref": "session_trading_day",
        "used_by": ["audit"],
        "learning_level": "Beginner",
    },
    "market": {
        "display_name": "Market",
        "description": "Underlying market identifier (e.g. NIFTY, BANKNIFTY).",
        "formula_ref": "market_id",
        "used_by": ["audit"],
        "learning_level": "Beginner",
    },
    "expiry": {
        "display_name": "Expiry",
        "description": "Option expiry date for the contract row.",
        "formula_ref": "contract_expiry",
        "used_by": ["audit"],
        "learning_level": "Beginner",
    },
    "timestamp": {
        "display_name": "Timestamp",
        "description": "Replay sample timestamp (IST).",
        "formula_ref": "sample_timestamp",
        "used_by": ["audit", "prediction"],
        "learning_level": "Beginner",
    },
    "option_type": {
        "display_name": "Option Type",
        "description": "Call (CE) or Put (PE) identifier.",
        "example": "CE",
        "formula_ref": "option_type",
        "used_by": ["audit", "training"],
        "learning_level": "Beginner",
    },
    "token": {
        "display_name": "Token",
        "description": "Exchange instrument token for the option contract.",
        "formula_ref": "instrument_token",
        "used_by": ["audit"],
        "learning_level": "Beginner",
    },
    "symbol": {
        "display_name": "Symbol",
        "description": "Trading symbol for the option contract.",
        "formula_ref": "trading_symbol",
        "used_by": ["audit"],
        "learning_level": "Beginner",
    },
}


def _snake_to_display(name: str) -> str:
    return " ".join(part.capitalize() for part in str(name).replace("-", "_").split("_") if part)


def _build_feature_column(name: str, group_id: str) -> dict[str, Any]:
    """Build a feature column entry from plugins + rich docs (no catalog import)."""
    from .schema_column_docs import RICH_COLUMN_DOCS
    from .schema_feature_meta import build_column_meta_extras
    from .schema_implementation import implementation_for_column

    doc = dict(RICH_COLUMN_DOCS.get(name) or {})
    entry: dict[str, Any] = {
        "id": name,
        "name": name,
        "display_name": doc.get("display_name") or _snake_to_display(name),
        "description": doc.get("description")
        or f"Registry feature `{name}` ({_GROUP_LABELS.get(group_id, group_id)}).",
        "type": "feature",
        "group": group_id,
        "formula_ref": doc.get("formula_ref") or name,
        "introduced_version": _INTRODUCED_VERSION,
    }
    if doc.get("formula_doc"):
        entry["formula_doc"] = doc["formula_doc"]
    if doc.get("interpretation"):
        entry["interpretation"] = doc["interpretation"]
    if doc.get("expected_range") is not None:
        entry["expected_range"] = doc["expected_range"]
    if doc.get("expected_data_type"):
        entry["expected_data_type"] = doc["expected_data_type"]
    if doc.get("unit"):
        entry["unit"] = doc["unit"]
    entry.update(build_column_meta_extras(name, group_id=group_id, col_type="feature", doc=doc))
    entry["implementation"] = implementation_for_column(
        name,
        formula_ref=entry.get("formula_ref"),
        group_id=group_id,
        doc=doc,
    )
    if doc.get("used_by"):
        entry["used_by"] = list(doc["used_by"])
    return entry


def canonical_plugin_feature_names() -> set[str]:
    """Single source of truth: feature names from ``_REGISTRY_FEATURES``."""
    from .feature_plugins import _REGISTRY_FEATURES

    return {str(f) for feats in _REGISTRY_FEATURES.values() for f in feats}


def schema_feature_column_names(schema: dict[str, Any] | None = None) -> set[str]:
    """Feature-type column names in a schema document (excludes metadata/targets)."""
    data = schema if schema is not None else load_schema_registry(use_cache=False)
    out: set[str] = set()
    for name, col in (data.get("columns") or {}).items():
        if str((col or {}).get("type") or "").lower() == "feature":
            out.add(str(name))
    return out


class SchemaRegistrySyncError(ValueError):
    """Canonical plugin registry and generated schema are out of sync."""


def validate_schema_plugin_parity(
    schema: dict[str, Any] | None = None,
    *,
    raise_on_error: bool = True,
) -> list[str]:
    """Enforce 1:1 parity between ``_REGISTRY_FEATURES`` and schema feature columns.

    Every schema feature must exist in plugins; every plugin feature must exist
    in the schema. No extras, no missing entries.

    Also rejects Interaction products (``_x_`` / InteractionTransformation naming)
    from both plugins and schema — those belong only in the Dataset Builder pipeline.
    """
    from .feature_ownership import is_interaction_feature

    plugin = canonical_plugin_feature_names()
    schema_feats = schema_feature_column_names(schema)
    only_schema = sorted(schema_feats - plugin)
    only_plugin = sorted(plugin - schema_feats)
    errors: list[str] = []
    plugin_ix = sorted(n for n in plugin if is_interaction_feature(n))
    schema_ix = sorted(n for n in schema_feats if is_interaction_feature(n))
    if plugin_ix:
        errors.append(
            "Interaction features must not appear in _REGISTRY_FEATURES "
            f"({len(plugin_ix)}): {', '.join(plugin_ix[:12])}"
            + ("…" if len(plugin_ix) > 12 else "")
        )
    if schema_ix:
        errors.append(
            "Interaction features must not appear in schema "
            f"({len(schema_ix)}): {', '.join(schema_ix[:12])}"
            + ("…" if len(schema_ix) > 12 else "")
        )
    if only_schema:
        errors.append(
            "Schema has features absent from _REGISTRY_FEATURES "
            f"({len(only_schema)}): {', '.join(only_schema[:12])}"
            + ("…" if len(only_schema) > 12 else "")
        )
    if only_plugin:
        errors.append(
            "Plugins have features absent from schema "
            f"({len(only_plugin)}): {', '.join(only_plugin[:12])}"
            + ("…" if len(only_plugin) > 12 else "")
        )
    if len(plugin) != len(schema_feats):
        errors.append(
            f"Count mismatch: plugins={len(plugin)} schema_features={len(schema_feats)}"
        )
    if errors and raise_on_error:
        raise SchemaRegistrySyncError(
            "Feature Registry / schema sync failed.\n"
            "ml_schema_registry.json is a generated artifact — rebuild from "
            "_REGISTRY_FEATURES (scripts/generate_schema_registry.py).\n"
            + "\n".join(errors)
        )
    return errors


def _sync_builtin_plugin_features(data: dict[str, Any]) -> dict[str, Any]:
    """Reconcile on-disk schema with canonical plugins (full feature replace).

    - Feature columns are rebuilt from ``_REGISTRY_FEATURES`` (stale removed).
    - Metadata / target columns are preserved.
    - Group feature lists are replaced from plugins (not additive).
    """
    if not data.get("columns"):
        return data
    from .feature_plugins import _REGISTRY_FEATURES

    columns = dict(data.get("columns") or {})

    # Preserve non-feature columns; keep only plugin features for type=feature.
    preserved: dict[str, Any] = {
        name: dict(col)
        for name, col in columns.items()
        if str((col or {}).get("type") or "").lower() != "feature"
    }
    for gid, feats in _REGISTRY_FEATURES.items():
        for feat in feats:
            existing = columns.get(feat)
            if (
                isinstance(existing, dict)
                and str(existing.get("type") or "").lower() == "feature"
            ):
                col = dict(existing)
                col["group"] = gid
                col["name"] = feat
                preserved[feat] = col
            else:
                preserved[feat] = _build_feature_column(feat, gid)

    groups: dict[str, Any] = {}
    group_order: list[str] = []
    for gid, plugin_feats in _REGISTRY_FEATURES.items():
        prior = (data.get("groups") or {}).get(gid) or {}
        groups[gid] = {
            "label": prior.get("label") or _GROUP_LABELS.get(gid, gid),
            "features": list(plugin_feats),
        }
        group_order.append(gid)

    deps_in = dict(data.get("dependencies") or {})
    dependencies = {
        gid: [d for d in (deps_in.get(gid) or []) if d in groups]
        for gid in groups
    }

    data["columns"] = preserved
    data["groups"] = groups
    data["groupOrder"] = group_order
    if dependencies:
        data["dependencies"] = dependencies
    return data


def rebuild_schema_registry_from_plugins(
    *,
    legacy_meta: dict[str, Any] | None = None,
    schema_version: int = 6,
) -> dict[str, Any]:
    """Full rebuild of ``ml_schema_registry.json`` from ``_REGISTRY_FEATURES``.

    This is a replace, not an additive merge: obsolete feature columns are not
    preserved. Metadata and targets are regenerated from code definitions.
    """
    from .feature_plugins import _REGISTRY_FEATURES
    from .schema_feature_meta import (
        TARGET_DEFINITIONS,
        build_column_meta_extras,
        build_targets_registry,
    )
    from .schema_implementation import implementation_for_column

    legacy = dict(legacy_meta or {})
    columns: dict[str, Any] = {}

    for i, (name, doc) in enumerate(_METADATA_DEFINITIONS.items(), start=1):
        entry: dict[str, Any] = {
            "id": f"meta_{i:03d}",
            "name": name,
            "display_name": doc.get("display_name") or name,
            "description": doc.get("description") or name,
            "type": "metadata",
            "formula_ref": doc.get("formula_ref") or name,
            "introduced_version": _INTRODUCED_VERSION,
            "nullable": False,
            "used_by": list(doc.get("used_by") or ["audit"]),
            "learning_level": doc.get("learning_level") or "Beginner",
        }
        if doc.get("interpretation"):
            entry["interpretation"] = doc["interpretation"]
        if doc.get("example"):
            entry["example"] = doc["example"]
        entry.update(build_column_meta_extras(name, col_type="metadata", doc=doc))
        entry["implementation"] = implementation_for_column(
            name, formula_ref=entry.get("formula_ref"), doc=doc
        )
        columns[name] = entry

    feat_idx = 1
    groups: dict[str, Any] = {}
    group_order: list[str] = []
    for gid, feats in _REGISTRY_FEATURES.items():
        group_order.append(gid)
        groups[gid] = {
            "label": _GROUP_LABELS.get(gid, gid),
            "features": list(feats),
        }
        for feat in feats:
            col = _build_feature_column(feat, gid)
            col["id"] = f"feature_{feat_idx:03d}"
            columns[feat] = col
            feat_idx += 1

    targets_registry = build_targets_registry(TARGET_DEFINITIONS)
    for i, target in enumerate(TARGET_DEFINITIONS, start=1):
        name = str(target["name"])
        doc = dict(target)
        extras = build_column_meta_extras(name, col_type="target", doc=doc)
        columns[name] = {
            "id": f"target_{i:03d}",
            "name": name,
            "display_name": doc.get("display_name") or name,
            "description": doc.get("description") or name,
            "type": "target",
            "prediction_horizon_sec": int(target["prediction_horizon_sec"]),
            "nullable": False,
            "formula_ref": doc.get("formula_ref"),
            "introduced_version": _INTRODUCED_VERSION,
            **extras,
            "implementation": implementation_for_column(
                name, formula_ref=doc.get("formula_ref"), doc=doc
            ),
            "used_by": list(doc.get("used_by") or ["training", "prediction", "audit"]),
        }

    legacy_deps = dict(legacy.get("dependencies") or {})
    dependencies: dict[str, list[str]] = {}
    for gid in group_order:
        raw = list(legacy_deps.get(gid) or [])
        dependencies[gid] = [d for d in raw if d in groups]

    schema: dict[str, Any] = {
        "version": int(schema_version),
        "hardMandatory": list(legacy.get("hardMandatory") or ["price"]),
        "dependencies": dependencies,
        "groupOrder": group_order,
        "groups": groups,
        "profiles": dict(legacy.get("profiles") or {}),
        "metadata_identities": [
            "trading_day", "market", "expiry", "timestamp", "strike",
            "option_type", "token", "symbol",
        ],
        "targets": targets_registry,
        "tag_catalog": [],
        "columns": columns,
        "generated_from": "_REGISTRY_FEATURES",
        "generated_note": (
            "GENERATED ARTIFACT — do not edit by hand. "
            "Rebuild with scripts/generate_schema_registry.py"
        ),
    }
    counts: dict[str, int] = {}
    for col in list(columns.values()) + list(targets_registry.values()):
        for tag in col.get("tags") or []:
            counts[str(tag)] = counts.get(str(tag), 0) + 1
    schema["tag_catalog"] = [{"id": tid, "count": counts[tid]} for tid in sorted(counts)]

    validate_schema_plugin_parity(schema, raise_on_error=True)
    return schema


def write_schema_registry(schema: dict[str, Any], path: str | None = None) -> str:
    """Persist schema JSON and invalidate the load cache."""
    global _SCHEMA_CACHE, _SCHEMA_CACHE_MTIME
    reg_path = path or _SCHEMA_PATH
    parent = os.path.dirname(reg_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(reg_path, "w", encoding="utf-8") as fh:
        json.dump(schema, fh, indent=2, ensure_ascii=False)
        fh.write("\n")
    _SCHEMA_CACHE = None
    _SCHEMA_CACHE_MTIME = None
    return reg_path


def load_schema_registry(path: str | None = None, *, use_cache: bool = True) -> dict[str, Any]:
    global _SCHEMA_CACHE, _SCHEMA_CACHE_MTIME
    reg_path = path or _SCHEMA_PATH
    try:
        file_mtime = os.path.getmtime(reg_path)
    except OSError:
        file_mtime = None
    if (
        use_cache
        and path is None
        and _SCHEMA_CACHE is not None
        and file_mtime is not None
        and _SCHEMA_CACHE_MTIME == file_mtime
    ):
        return _SCHEMA_CACHE
    try:
        with open(reg_path, encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError):
        data = {}
    data = _sync_builtin_target_definitions(data)
    data = _sync_builtin_plugin_features(data)
    if data.get("columns"):
        validate_schema_plugin_parity(data, raise_on_error=True)
    elif path is None:
        # Generated artifact missing or empty — rebuild from canonical plugins so
        # Feature Registry and dataset tooling work out of the box.
        data = rebuild_schema_registry_from_plugins()
        validate_schema_plugin_parity(data, raise_on_error=True)
        write_schema_registry(data)
        try:
            file_mtime = os.path.getmtime(reg_path)
        except OSError:
            file_mtime = None
    if path is None and use_cache:
        _SCHEMA_CACHE = data
        _SCHEMA_CACHE_MTIME = file_mtime
    return data


def _load_legacy_feature_registry_file(path: str | None = None) -> dict[str, Any]:
    reg_path = path or _LEGACY_FEATURE_REGISTRY_PATH
    try:
        with open(reg_path, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError):
        return {}


def schema_to_legacy_feature_registry(schema: dict[str, Any]) -> dict[str, Any]:
    """Expose legacy group-based API from schema registry."""
    if not schema:
        return {}
    return {
        "version": schema.get("version", 1),
        "hardMandatory": list(schema.get("hardMandatory") or ["price"]),
        "dependencies": dict(schema.get("dependencies") or {}),
        "groupOrder": list(schema.get("groupOrder") or []),
        "groups": dict(schema.get("groups") or {}),
        "profiles": dict(schema.get("profiles") or {}),
    }


def load_feature_registry(path: str | None = None) -> dict[str, Any]:
    """Backward-compatible loader — reads schema registry, falls back to legacy JSON."""
    schema = load_schema_registry()
    if schema.get("columns"):
        return schema_to_legacy_feature_registry(schema)
    return _load_legacy_feature_registry_file(path)


def columns_map(schema: dict[str, Any] | None = None) -> dict[str, dict[str, Any]]:
    reg = schema or load_schema_registry()
    raw = reg.get("columns") or {}
    return {str(k): dict(v) for k, v in raw.items()}


def column_meta(name: str, schema: dict[str, Any] | None = None) -> dict[str, Any] | None:
    col = columns_map(schema).get(str(name or "").strip())
    return dict(col) if col else None


def column_display_name(name: str, schema: dict[str, Any] | None = None) -> str:
    col = column_meta(name, schema)
    if col and col.get("display_name"):
        return str(col["display_name"])
    return str(name or "—")


def column_description(name: str, schema: dict[str, Any] | None = None) -> str:
    col = column_meta(name, schema)
    if col and col.get("description"):
        return str(col["description"])
    return "—"


def column_category(name: str, schema: dict[str, Any] | None = None) -> str:
    col = column_meta(name, schema)
    if not col:
        return "—"
    group_id = col.get("group")
    if group_id:
        groups = (schema or load_schema_registry()).get("groups") or {}
        label = (groups.get(group_id) or {}).get("label")
        if label:
            return str(label)
    return str(col.get("category") or "—")


def column_unit(name: str, schema: dict[str, Any] | None = None) -> str | None:
    col = column_meta(name, schema)
    unit = (col or {}).get("unit")
    return str(unit) if unit else None


def target_horizon_sec(name: str, schema: dict[str, Any] | None = None) -> int | None:
    col = column_meta(name, schema)
    if not col:
        return None
    horizon = col.get("prediction_horizon_sec")
    return int(horizon) if horizon is not None else None


def target_predicts_label(name: str, schema: dict[str, Any] | None = None) -> str:
    """Human-readable prediction label for a target column."""
    col = column_meta(name, schema)
    if col and col.get("description"):
        return str(col["description"])
    horizon = target_horizon_sec(name, schema)
    if horizon is not None:
        if horizon < 60:
            unit = "second" if horizon == 1 else "seconds"
            return f"Option LTP after {horizon} {unit}"
        if horizon % 60 == 0:
            mins = horizon // 60
            unit = "minute" if mins == 1 else "minutes"
            return f"Option LTP after {mins} {unit}"
    return "Option LTP at future horizon"


def metadata_column_names(schema: dict[str, Any] | None = None) -> list[str]:
    """Identity columns stored in each dataset row (strike may also be a feature)."""
    reg = schema or load_schema_registry()
    cols = columns_map(reg)
    identity = list(reg.get("metadata_identities") or [
        "trading_day", "market", "expiry", "timestamp", "strike",
        "option_type", "token", "symbol",
    ])
    return [name for name in identity if name in cols or name in identity]


def feature_column_names(schema: dict[str, Any] | None = None) -> list[str]:
    cols = columns_map(schema)
    return [name for name, col in cols.items() if str(col.get("type") or "").lower() == "feature"]


def target_column_names(schema: dict[str, Any] | None = None) -> list[str]:
    reg = schema or load_schema_registry()
    targets = reg.get("targets")
    if isinstance(targets, dict) and targets:
        return list(targets.keys())
    cols = columns_map(reg)
    return [
        name for name, col in cols.items()
        if str(col.get("type") or "").lower() == "target"
    ]


def targets_map(schema: dict[str, Any] | None = None) -> dict[str, dict[str, Any]]:
    """Versioned prediction-target registry (parallel to feature columns)."""
    reg = schema or load_schema_registry()
    raw = reg.get("targets") or {}
    if raw:
        return {str(k): dict(v) for k, v in raw.items()}
    cols = columns_map(reg)
    return {
        name: dict(col)
        for name, col in cols.items()
        if str(col.get("type") or "").lower() == "target"
    }


def target_meta(name: str, schema: dict[str, Any] | None = None) -> dict[str, Any] | None:
    entry = targets_map(schema).get(str(name or "").strip())
    return dict(entry) if entry else None


def registry_features_for_groups(enabled_groups: list[str], registry: dict[str, Any] | None = None) -> list[str]:
    """Feature names for enabled groups (legacy group API)."""
    reg = registry or load_feature_registry()
    groups_meta = reg.get("groups") or {}
    out: list[str] = []
    seen: set[str] = set()
    for gid in enabled_groups:
        feats = list((groups_meta.get(gid) or {}).get("features") or [])
        if not feats:
            from .feature_plugins import _REGISTRY_FEATURES
            feats = list(_REGISTRY_FEATURES.get(gid) or [])
        for feat in feats:
            if feat not in seen:
                seen.add(feat)
                out.append(feat)
    return out


def all_registry_feature_names(registry: dict[str, Any] | None = None) -> list[str]:
    """All feature column names in registry group order."""
    reg = registry or load_feature_registry()
    return registry_features_for_groups(list(reg.get("groupOrder") or []), reg)


def resolve_feature_selection(
    feature_selection: dict[str, Any],
    registry: dict[str, Any] | None = None,
) -> tuple[list[str], list[str]]:
    """Return ``(enabled_groups, feature_names)`` in registry order."""
    reg = registry or load_feature_registry()
    groups_meta = reg.get("groups") or {}
    group_order = list(reg.get("groupOrder") or [])
    all_ordered = all_registry_feature_names(reg)
    all_set = set(all_ordered)

    profile = str(feature_selection.get("profile") or "default").lower()
    enabled_groups_input = list(feature_selection.get("enabledGroups") or group_order)
    enabled_features_input = list(feature_selection.get("enabledFeatures") or [])

    if profile == "custom" and enabled_features_input:
        selected_set = {f for f in enabled_features_input if f in all_set}
        for gid in reg.get("hardMandatory") or []:
            for feat in (groups_meta.get(gid) or {}).get("features") or []:
                selected_set.add(feat)
        ordered = [f for f in all_ordered if f in selected_set]
        enabled_groups = [
            gid
            for gid in group_order
            if any(f in selected_set for f in ((groups_meta.get(gid) or {}).get("features") or []))
        ]
        return enabled_groups, ordered

    groups = enabled_groups_input if enabled_groups_input else group_order
    return list(groups), registry_features_for_groups(groups, reg)


def schema_registry_identity_material(schema: dict[str, Any] | None = None) -> dict[str, Any]:
    reg = schema or load_schema_registry()
    return {
        "version": reg.get("version"),
        "groupOrder": reg.get("groupOrder"),
        "groups": reg.get("groups"),
        "columns": reg.get("columns"),
        "targets": reg.get("targets"),
        "hardMandatory": reg.get("hardMandatory"),
        "dependencies": reg.get("dependencies"),
    }


def schema_registry_hash(schema: dict[str, Any] | None = None) -> str:
    return _compute_content_hash(schema_registry_identity_material(schema))


def implementation_identity_material() -> dict[str, Any]:
    from .feature_plugins import GROUP_FEATURE_SOURCES, _EXTRACTOR_ALIASES

    plugin_groups = {
        gid: dict(sorted(mapping.items()))
        for gid, mapping in sorted(GROUP_FEATURE_SOURCES.items())
    }
    return {
        "plugin_aliases": dict(sorted(_EXTRACTOR_ALIASES.items())),
        "plugin_groups": plugin_groups,
    }


def implementation_hash() -> str:
    return _compute_content_hash(implementation_identity_material())


def validate_build_schema(
    *,
    enabled_groups: list[str],
    target_columns: list[str],
    schema: dict[str, Any] | None = None,
) -> list[str]:
    """Return schema integrity errors (empty list = pass)."""
    reg = schema or load_schema_registry()
    columns = columns_map(reg)
    groups = reg.get("groups") or {}
    errors: list[str] = []

    if not columns:
        return ["Schema registry is empty or unreadable"]

    seen_ids: dict[str, str] = {}
    for name, col in columns.items():
        cid = str(col.get("id") or "").strip()
        if not cid:
            errors.append(f"Column {name} missing id")
            continue
        if cid in seen_ids:
            errors.append(f"Duplicate column id {cid}: {seen_ids[cid]} and {name}")
        seen_ids[cid] = name

    feature_to_group: dict[str, str] = {}
    for gid, block in groups.items():
        for feat in (block or {}).get("features") or []:
            if feat in feature_to_group:
                errors.append(f"Feature {feat} listed in multiple groups ({feature_to_group[feat]}, {gid})")
            feature_to_group[feat] = gid
            if feat not in columns:
                errors.append(f"Group {gid} references unknown column {feat}")
            elif str(columns[feat].get("type") or "").lower() != "feature":
                errors.append(f"{feat} in group {gid} is not type feature")
            elif columns[feat].get("group") != gid:
                errors.append(
                    f"{feat} group mismatch: column.group={columns[feat].get('group')}, groups.{gid}"
                )

    for name, col in columns.items():
        if str(col.get("type") or "").lower() == "feature" and name not in feature_to_group:
            errors.append(f"Orphan feature {name} not listed in any group")

    for gid in enabled_groups:
        for feat in registry_features_for_groups([gid], schema_to_legacy_feature_registry(reg)):
            if feat not in columns:
                errors.append(f"Selected feature {feat} missing from schema registry")
            elif str(columns[feat].get("type") or "").lower() != "feature":
                errors.append(f"Selected column {feat} is not type feature")

    for tc in target_columns:
        if tc not in columns:
            errors.append(f"Target {tc} missing from schema registry")
        elif str(columns[tc].get("type") or "").lower() != "target":
            errors.append(f"Column {tc} is not type target")

    for mc in metadata_column_names(reg):
        if mc not in columns:
            errors.append(f"Metadata column {mc} missing from schema registry")
        else:
            col_type = str(columns[mc].get("type") or "").lower()
            if col_type not in ("metadata", "feature"):
                errors.append(f"Column {mc} is not type metadata or feature")

    return errors


def column_interpretation(name: str, schema: dict[str, Any] | None = None) -> str | None:
    col = column_meta(name, schema)
    if col and col.get("interpretation"):
        return str(col["interpretation"])
    return None


def column_used_by(name: str, schema: dict[str, Any] | None = None) -> list[str]:
    col = column_meta(name, schema) or {}
    used = col.get("used_by")
    if isinstance(used, list) and used:
        return [str(x) for x in used]
    t = str(col.get("type") or "").lower()
    if t == "target":
        return ["training", "prediction", "audit"]
    if t == "metadata":
        return ["audit"]
    return ["training", "audit"]


def _used_by_features(name: str, schema: dict[str, Any] | None = None) -> list[str]:
    """Feature columns that list `name` in depends_on (reverse dependency index)."""
    reg = schema or load_schema_registry()
    cols = columns_map(reg)
    key = str(name or "").strip()
    if not key:
        return []
    out: list[str] = []
    for col_name, col in cols.items():
        deps = col.get("depends_on") or []
        if key in deps:
            out.append(col_name)
    out.sort(key=lambda n: column_display_name(n, reg).lower())
    return out


def enrich_column_view(name: str, schema: dict[str, Any] | None = None) -> dict[str, Any]:
    """Column metadata bundle for API/UI."""
    col = column_meta(name, schema) or {}
    return {
        "name": name,
        "display_name": column_display_name(name, schema),
        "description": column_description(name, schema),
        "interpretation": col.get("interpretation"),
        "category": column_category(name, schema),
        "unit": column_unit(name, schema),
        "type": col.get("type"),
        "group": col.get("group"),
        "formula_ref": col.get("formula_ref"),
        "formula_doc": col.get("formula_doc"),
        "example": col.get("example"),
        "expected_range": col.get("expected_range"),
        "nullable": col.get("nullable"),
        "expected_null_reason": col.get("expected_null_reason"),
        "learning_level": col.get("learning_level"),
        "used_by": column_used_by(name, schema),
        "prediction_horizon_sec": col.get("prediction_horizon_sec"),
        "introduced_version": col.get("introduced_version"),
        "tags": list(col.get("tags") or []),
        "depends_on": list(col.get("depends_on") or []),
        "used_by_features": _used_by_features(name, schema),
        "compute_cost": col.get("compute_cost"),
        "status": col.get("status"),
        "importance": col.get("importance"),
        "implementation": col.get("implementation"),
        "related_features": list(col.get("related_features") or []),
    }

"""Feature Registry catalog — single API payload for the create-dataset Feature Registry tab."""

from __future__ import annotations

import csv
import json
import os
import re
from datetime import datetime, timezone
from typing import Any

from .feature_plugins import GROUP_FEATURE_SOURCES
from .schema_column_docs import RICH_COLUMN_DOCS
from .schema_feature_meta import build_column_meta_extras
from .schema_implementation import implementation_for_column
from .schema_registry import (
    load_schema_registry,
    schema_registry_hash,
    validate_schema_plugin_parity,
)
from .feature_registry_store import (
    DISABLED_GROUP_ID,
    allocate_feature_identity,
    ensure_feature_identities,
    list_projects,
    load_store,
    resolve_dependency_refs,
    save_store,
)

_GROUP_LABELS: dict[str, str] = {
    "price": "Price & Returns",
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
    "market_microstructure": "Market Microstructure",
    "atm_straddle": "ATM Straddle",
    "atm6_ltp": "ATM+6 LTP",
    "chain": "Chain-Wide",
    "chain_flow": "Chain Flow",
    "historical": "Historical OHLC",
    "advanced": "Advanced",
}

_GROUP_FILTER: dict[str, str] = {
    "price": "Price",
    "dgt_reiv": "DGT REIV",
    "ratio": "Ratio",
    "greeks": "Greeks",
    "iv": "IV",
    "iv_zscore": "IV Z",
    "iv_ema_ratio": "IV EMA Ratio",
    "oi": "OI",
    "volume": "Volume",
    "momentum": "Momentum",
    "sharp_momentum": "Sharp Mom",
    "spot_hl": "Spot HL",
    "time": "Time",
    "moneyness": "Moneyness",
    "ltp_to_spot": "LTP/Spot",
    "ltp_to_others": "LTP/Others",
    "spot_and_other_ratio": "Spot/Ratio",
    "historic_spot_ema": "Hist EMA",
    "market_microstructure": "Book",
    "atm_straddle": "ATM",
    "atm6_ltp": "ATM6",
    "chain": "Chain",
    "chain_flow": "Flow",
    "historical": "Historical",
    "advanced": "Advanced",
}

_DISABLED_GROUP_LABEL = "Disabled"
_DISABLED_GROUP_FILTER = "Disabled"

_STATUS_LABELS: dict[str, str] = {
    "implemented": "🟢 Implemented",
    "planned": "🟡 Planned",
    "in_progress": "🟠 In Progress",
    "not_implemented": "🔴 Not Implemented",
    "deprecated": "⚫ Deprecated",
    "experimental": "🟡 Experimental",
}

# Stages where the feature itself is computed or selected — not downstream model consumers.
_PIPELINE_SURFACES: list[tuple[str, str]] = [
    ("dataset_builder", "Dataset Builder"),
    ("live_prediction", "Live Feature Builder"),
    ("training", "Training"),
    ("inference", "Inference"),
]

_INTRODUCED_VERSION = "1.4"
_FEATURE_ID_RE = re.compile(r"^FR\d+$", re.IGNORECASE)


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


def _expected_range_label(raw: Any) -> str:
    if raw is None or raw == "":
        return ""
    if isinstance(raw, list) and len(raw) == 2:
        return f"{raw[0]} to {raw[1]}"
    return str(raw)


def _infer_expected_data_type(col: dict[str, Any]) -> str:
    explicit = str(col.get("expected_data_type") or "").strip().lower()
    if explicit in ("float", "int", "bool", "string"):
        return explicit
    unit = str(col.get("unit") or "").strip()
    if unit in ("0 or 1", "bool"):
        return "bool"
    if unit in ("any integer", "integer", "int"):
        return "int"
    return "float"


def feature_backlog_path(data_dir: str) -> str:
    return os.path.join(data_dir, "feature_registry_backlog.json")


def _load_json(path: str) -> dict[str, Any]:
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def load_feature_backlog(data_dir: str) -> dict[str, Any]:
    doc = _load_json(feature_backlog_path(data_dir))
    if not doc:
        return {"version": 1, "features": []}
    doc.setdefault("version", 1)
    doc.setdefault("features", [])
    return doc


def save_planned_feature(data_dir: str, entry: dict[str, Any]) -> dict[str, Any]:
    doc = load_feature_backlog(data_dir)
    features: list[dict[str, Any]] = list(doc.get("features") or [])
    name = str(entry.get("name") or "").strip()
    if not name:
        raise ValueError("Feature name is required")
    now = datetime.now(timezone.utc).isoformat()
    normalized = {
        "name": name,
        "group": str(entry.get("group") or "advanced").strip(),
        "description": str(entry.get("description") or "").strip(),
        "why_needed": str(entry.get("why_needed") or entry.get("why") or "").strip(),
        "formula": str(entry.get("formula") or "").strip(),
        "inputs_required": list(entry.get("inputs_required") or []),
        "dependencies": list(entry.get("dependencies") or []),
        "expected_data_type": str(entry.get("expected_data_type") or "float").strip(),
        "expected_range": _normalize_expected_range(entry.get("expected_range")),
        "implementation_status": str(entry.get("implementation_status") or "planned").strip(),
        "priority": str(entry.get("priority") or "medium").strip(),
        "notes": str(entry.get("notes") or entry.get("developer_notes") or "").strip(),
        "created_at": now,
        "updated_at": now,
    }
    replaced = False
    for i, row in enumerate(features):
        if str(row.get("name") or "") == name:
            normalized["created_at"] = row.get("created_at") or now
            features[i] = normalized
            replaced = True
            break
    if not replaced:
        features.append(normalized)
    doc["features"] = features
    path = feature_backlog_path(data_dir)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=2)
    group = str(normalized.get("group") or "advanced")
    feature_id = allocate_feature_identity(
        data_dir,
        name,
        group=group,
        display_name=str(entry.get("display_name") or "").strip() or None,
        owner=str(entry.get("owner") or "").strip() or None,
        created_by=str(entry.get("created_by") or "User").strip() or "User",
    )
    normalized["feature_id"] = feature_id
    return normalized


def _snake_to_display(name: str) -> str:
    return " ".join(part.capitalize() for part in str(name).replace("-", "_").split("_") if part)


def _build_column_from_plugins(name: str, group_id: str) -> dict[str, Any]:
    doc = dict(RICH_COLUMN_DOCS.get(name) or {})
    entry: dict[str, Any] = {
        "id": name,
        "name": name,
        "display_name": doc.get("display_name") or _snake_to_display(name),
        "description": doc.get("description") or f"Registry feature `{name}` ({_GROUP_LABELS.get(group_id, group_id)}).",
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


def _runtime_schema_columns() -> dict[str, dict[str, Any]]:
    columns: dict[str, dict[str, Any]] = {}
    for gid, mapping in GROUP_FEATURE_SOURCES.items():
        for feat in mapping:
            columns[feat] = _build_column_from_plugins(feat, gid)
    return columns


def _schema_columns() -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Feature columns come from the canonical plugin registry (1:1 with schema)."""
    schema = load_schema_registry()
    validate_schema_plugin_parity(schema, raise_on_error=True)
    return schema, _runtime_schema_columns()


def _implemented_feature_names() -> set[str]:
    out: set[str] = set()
    for mapping in GROUP_FEATURE_SOURCES.values():
        for feat, src in mapping.items():
            if src is not None:
                out.add(feat)
    return out


def _scan_models(data_dir: str) -> dict[str, list[dict[str, Any]]]:
    """Map feature name → list of {model_name, importance_pct}."""
    models_dir = os.path.join(data_dir, "models")
    by_feat: dict[str, list[dict[str, Any]]] = {}
    if not os.path.isdir(models_dir):
        return by_feat
    for entry in os.listdir(models_dir):
        if entry.startswith("."):
            continue
        pkg = os.path.join(models_dir, entry)
        if not os.path.isdir(pkg):
            continue
        config_path = os.path.join(pkg, "config.json")
        fi_path = os.path.join(pkg, "feature_importance.csv")
        if not os.path.isfile(config_path):
            continue
        try:
            with open(config_path, encoding="utf-8") as fh:
                config = json.load(fh)
        except (OSError, json.JSONDecodeError):
            continue
        model_name = str(config.get("model_name") or entry)
        features = list(config.get("features") or [])
        importance: dict[str, float] = {}
        if os.path.isfile(fi_path):
            try:
                with open(fi_path, encoding="utf-8") as fh:
                    for row in csv.DictReader(fh):
                        feat = row.get("Feature") or row.get("feature")
                        imp = row.get("Importance") or row.get("importance_pct")
                        if feat and imp not in (None, ""):
                            importance[str(feat)] = float(imp)
            except (OSError, ValueError, csv.Error):
                pass
        for feat in features:
            by_feat.setdefault(str(feat), []).append({
                "model_name": model_name,
                "importance_pct": importance.get(str(feat)),
            })
    return by_feat


def _importance_summary(models: list[dict[str, Any]]) -> dict[str, Any]:
    vals = [float(m["importance_pct"]) for m in models if m.get("importance_pct") is not None]
    if not vals:
        return {"average_pct": None, "best_pct": None, "best_model": None}
    best = max(models, key=lambda m: float(m.get("importance_pct") or 0))
    return {
        "average_pct": round(sum(vals) / len(vals), 1),
        "best_pct": round(float(best.get("importance_pct") or 0), 1),
        "best_model": best.get("model_name"),
    }


def _resolve_status(name: str, col: dict[str, Any], backlog: dict[str, Any] | None) -> str:
    if backlog:
        st = str(backlog.get("implementation_status") or "planned").lower()
        if st in _STATUS_LABELS:
            return st
        return "planned"
    raw = str(col.get("status") or "").lower()
    if raw == "deprecated":
        return "deprecated"
    if raw == "experimental":
        return "experimental"
    if name in _implemented_feature_names():
        return "implemented"
    return "not_implemented"


def _infer_used_in(
    name: str,
    *,
    implemented: bool,
    models_using: list[str],
    col: dict[str, Any],
) -> list[dict[str, str]]:
    active: set[str] = set()
    if implemented:
        active.update({"dataset_builder", "live_prediction"})
    for raw in col.get("used_by") or []:
        key = str(raw).lower().replace(" ", "_")
        if key in ("training", "prediction"):
            active.add("training" if key == "training" else "inference")
    if models_using:
        active.add("training")
        active.add("inference")
    order = [k for k, _ in _PIPELINE_SURFACES]
    return [
        {"id": sid, "label": label, "active": sid in active}
        for sid, label in _PIPELINE_SURFACES
        if sid in active
    ]


def _build_dependency_tree(name: str, columns: dict[str, dict[str, Any]], depth: int = 0, seen: set[str] | None = None) -> list[dict[str, Any]]:
    if seen is None:
        seen = set()
    if name in seen or depth > 8:
        return []
    seen.add(name)
    col = columns.get(name) or {}
    deps = list(col.get("depends_on") or [])
    nodes: list[dict[str, Any]] = []
    for dep in deps:
        if dep in ("timestamp", "token", "symbol"):
            continue
        nodes.append({
            "name": dep,
            "label": columns.get(dep, {}).get("display_name") or _snake_to_display(dep),
            "children": _build_dependency_tree(dep, columns, depth + 1, set(seen)),
        })
    return nodes


def _attach_domain_meta(entry: dict[str, Any]) -> dict[str, Any]:
    """Attach primary domain + Auto Feature Generation flags (no ID/formula changes)."""
    name = str(entry.get("name") or "")
    if not name:
        return entry
    try:
        from .feature_domains import build_feature_domain_meta, DOMAIN_LABELS

        meta = build_feature_domain_meta(name)
        entry["primary_domain"] = meta["primary_domain"]
        entry["primary_domain_label"] = DOMAIN_LABELS[meta["primary_domain"]]
        entry["ownership"] = meta["ownership"]
        entry["domain_data_type"] = meta["data_type"]
        entry["can_apply_lag"] = meta["can_apply_lag"]
        entry["can_apply_difference"] = meta["can_apply_difference"]
        entry["can_apply_return"] = meta["can_apply_return"]
        entry["can_apply_rolling"] = meta["can_apply_rolling"]
        entry["can_apply_zscore"] = meta["can_apply_zscore"]
        entry["can_participate_in_interaction"] = meta["can_participate_in_interaction"]
        # Prefer domain label in the table "group" chip column for Registry UI.
        entry["domain_filter"] = entry["primary_domain"]
        entry["domain"] = entry["primary_domain_label"]
    except Exception:
        pass
    return entry


def _catalog_entry_from_registry(
    name: str,
    col: dict[str, Any],
    *,
    models_by_feat: dict[str, list[dict[str, Any]]],
    columns: dict[str, dict[str, Any]],
    feature_id: str | None = None,
    policy_by_name: dict[str, Any] | None = None,
) -> dict[str, Any]:
    group_id = str(col.get("group") or "")
    models = models_by_feat.get(name, [])
    model_names = sorted({str(m["model_name"]) for m in models})
    status = _resolve_status(name, col, None)
    implemented = status == "implemented"
    impl = col.get("implementation") or {}
    entry = {
        "feature_id": feature_id,
        "name": name,
        "display_name": col.get("display_name") or _snake_to_display(name),
        "group_id": group_id,
        "group": _GROUP_LABELS.get(group_id, _snake_to_display(group_id) if group_id else "—"),
        "group_filter": _GROUP_FILTER.get(group_id, group_id or "Other"),
        "category": col.get("learning_level") or _GROUP_LABELS.get(group_id, "Feature"),
        "description": col.get("description") or "",
        "why_needed": col.get("interpretation") or "",
        "formula": col.get("formula_doc") or col.get("formula_ref") or "",
        "dependencies": list(col.get("depends_on") or []),
        "inputs_required": list(col.get("depends_on") or []),
        "expected_data_type": _infer_expected_data_type(col),
        "expected_range": col.get("expected_range"),
        "expected_range_label": _expected_range_label(col.get("expected_range")),
        "implementation_status": status,
        "implementation_label": _STATUS_LABELS.get(status, status),
        "implementation_module": impl.get("module"),
        "implementation_function": impl.get("function"),
        "first_version": f"v{str(col.get('introduced_version') or _INTRODUCED_VERSION).replace('v', '')}",
        "last_updated": None,
        "developer_notes": "",
        "priority": None,
        "importance": _importance_summary(models),
        "models_using": model_names,
        "used_in": _infer_used_in(name, implemented=implemented, models_using=model_names, col=col),
        "dependency_tree": _build_dependency_tree(name, columns),
        "source": "registry",
        "tags": list(col.get("tags") or []),
    }
    try:
        if policy_by_name and name in policy_by_name:
            policy = policy_by_name[name]
        else:
            from chain_replay_ml.feature_policy import build_feature_policy_metadata

            policy = build_feature_policy_metadata(name, col)
        entry["learning_level"] = col.get("learning_level") or entry.get("category")
        entry["feature_category"] = policy.feature_category.value
        entry["lifecycle"] = policy.lifecycle.value
        entry["policy"] = policy.as_dict()
    except Exception:
        entry["feature_category"] = "raw" if col.get("type") != "target" else "target"
        entry["lifecycle"] = "tick"
        entry["policy"] = {}
    return _attach_domain_meta(entry)


def _catalog_entry_from_store_row(
    name: str,
    row: dict[str, Any],
    *,
    feature_id: str | None,
    columns: dict[str, dict[str, Any]],
    source: str = "imported",
) -> dict[str, Any]:
    group_id = str(row.get("group") or "advanced")
    status = _resolve_status(name, {}, row)
    deps = list(row.get("dependencies") or [])
    impl_mod = row.get("implementation_module")
    impl_fn = row.get("implementation_function")
    entry = {
        "feature_id": feature_id,
        "name": name,
        "display_name": row.get("display_name") or _snake_to_display(name),
        "group_id": group_id,
        "group": _GROUP_LABELS.get(group_id, _snake_to_display(group_id)),
        "group_filter": _GROUP_FILTER.get(group_id, group_id),
        "category": _GROUP_LABELS.get(group_id, "Feature"),
        "description": row.get("description") or "",
        "why_needed": row.get("why_needed") or "",
        "formula": row.get("formula") or "",
        "dependencies": deps,
        "inputs_required": list(row.get("inputs_required") or deps),
        "expected_data_type": row.get("expected_data_type") or "float",
        "expected_range": row.get("expected_range"),
        "expected_range_label": _expected_range_label(row.get("expected_range")),
        "implementation_status": status,
        "implementation_label": _STATUS_LABELS.get(status, status),
        "implementation_module": impl_mod,
        "implementation_function": impl_fn,
        "first_version": None,
        "last_updated": row.get("updated_at") or row.get("created_at"),
        "developer_notes": row.get("notes") or "",
        "priority": row.get("priority"),
        "importance": {"average_pct": None, "best_pct": None, "best_model": None},
        "models_using": [],
        "used_in": [],
        "dependency_tree": _build_dependency_tree(name, columns) if name in columns else [
            {"name": d, "label": _snake_to_display(d), "children": []} for d in deps
        ],
        "source": source,
        "tags": list(row.get("tags") or []),
        "source_model": row.get("source_model"),
    }
    return _attach_domain_meta(entry)


def _apply_store_overlay(entry: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    out = dict(entry)
    if overlay.get("group"):
        gid = str(overlay["group"])
        out["group_id"] = gid
        out["group"] = _GROUP_LABELS.get(gid, _snake_to_display(gid))
        out["group_filter"] = _GROUP_FILTER.get(gid, gid)
    if overlay.get("description"):
        out["description"] = overlay["description"]
    if overlay.get("why_needed"):
        out["why_needed"] = overlay["why_needed"]
    if overlay.get("formula"):
        out["formula"] = overlay["formula"]
    if overlay.get("dependencies"):
        out["dependencies"] = list(overlay["dependencies"])
    if overlay.get("expected_data_type"):
        out["expected_data_type"] = overlay["expected_data_type"]
    if overlay.get("expected_range") is not None:
        out["expected_range"] = overlay["expected_range"]
        out["expected_range_label"] = _expected_range_label(overlay["expected_range"])
    if overlay.get("implementation_status"):
        st = str(overlay["implementation_status"])
        out["implementation_status"] = st
        out["implementation_label"] = _STATUS_LABELS.get(st, st)
    if overlay.get("implementation_module") is not None:
        out["implementation_module"] = overlay["implementation_module"]
    if overlay.get("implementation_function") is not None:
        out["implementation_function"] = overlay["implementation_function"]
    if overlay.get("priority"):
        out["priority"] = overlay["priority"]
    if overlay.get("notes"):
        out["developer_notes"] = overlay["notes"]
    if overlay.get("tags"):
        out["tags"] = list(overlay["tags"])
    if overlay.get("source_model"):
        out["source_model"] = overlay["source_model"]
    if overlay.get("updated_at"):
        out["last_updated"] = overlay["updated_at"]
    return out


def _apply_registry_active_state(
    features: list[dict[str, Any]],
    disabled_map: dict[str, Any],
) -> None:
    for feat in features:
        name = str(feat.get("name") or "")
        rec = disabled_map.get(name)
        if not rec:
            feat["registry_active"] = True
            continue
        home_gid = str(rec.get("home_group_id") or feat.get("group_id") or "")
        feat["registry_active"] = False
        feat["home_group_id"] = home_gid
        feat["home_group"] = _GROUP_LABELS.get(home_gid, _snake_to_display(home_gid) if home_gid else "—")
        feat["home_group_filter"] = _GROUP_FILTER.get(home_gid, home_gid or "—")
        feat["group_id"] = DISABLED_GROUP_ID
        feat["group"] = _DISABLED_GROUP_LABEL
        feat["group_filter"] = _DISABLED_GROUP_FILTER


def _catalog_entry_from_backlog(
    row: dict[str, Any],
    *,
    columns: dict[str, dict[str, Any]],
    feature_id: str | None = None,
) -> dict[str, Any]:
    name = str(row.get("name") or "")
    group_id = str(row.get("group") or "advanced")
    status = _resolve_status(name, {}, row)
    deps = list(row.get("dependencies") or [])
    entry = {
        "feature_id": feature_id,
        "name": name,
        "display_name": _snake_to_display(name),
        "group_id": group_id,
        "group": _GROUP_LABELS.get(group_id, _snake_to_display(group_id)),
        "group_filter": _GROUP_FILTER.get(group_id, group_id),
        "category": _GROUP_LABELS.get(group_id, "Planned"),
        "description": row.get("description") or "",
        "why_needed": row.get("why_needed") or "",
        "formula": row.get("formula") or "",
        "dependencies": deps,
        "inputs_required": list(row.get("inputs_required") or deps),
        "expected_data_type": row.get("expected_data_type") or "float",
        "expected_range": row.get("expected_range"),
        "expected_range_label": _expected_range_label(row.get("expected_range")),
        "implementation_status": status,
        "implementation_label": _STATUS_LABELS.get(status, status),
        "implementation_module": None,
        "implementation_function": None,
        "first_version": None,
        "last_updated": row.get("updated_at") or row.get("created_at"),
        "developer_notes": row.get("notes") or "",
        "priority": row.get("priority"),
        "importance": {"average_pct": None, "best_pct": None, "best_model": None},
        "models_using": [],
        "used_in": [],
        "dependency_tree": _build_dependency_tree(name, columns) if name in columns else [
            {"name": d, "label": _snake_to_display(d), "children": []} for d in deps
        ],
        "source": "planned",
        "tags": ["planned"],
    }
    return _attach_domain_meta(entry)


def _apply_identity_to_entry(entry: dict[str, Any], identity: dict[str, Any] | None) -> dict[str, Any]:
    if not identity:
        return entry
    out = dict(entry)
    out["feature_id"] = identity.get("feature_id") or out.get("feature_id")
    if identity.get("display_name"):
        out["display_name"] = identity["display_name"]
    out["created_at"] = identity.get("created_at")
    out["created_by"] = identity.get("created_by")
    out["updated_at"] = identity.get("updated_at")
    out["feature_version"] = identity.get("version") or "1.0"
    out["owner"] = identity.get("owner") or _default_owner_for_entry(out)
    out["previous_names"] = list(identity.get("previous_names") or [])
    return out


def _default_owner_for_entry(entry: dict[str, Any]) -> str:
    from .feature_registry_store import default_owner_for_group
    return default_owner_for_group(entry.get("group_id"))


def _enrich_features_with_identities(
    features: list[dict[str, Any]],
    store: dict[str, Any],
) -> None:
    identities = dict(store.get("feature_identities") or {})
    name_to_id = dict(store.get("feature_ids") or {})
    id_to_name = {fid: n for n, fid in name_to_id.items()}
    by_name = {str(f["name"]): f for f in features if f.get("name")}

    for feat in features:
        name = str(feat.get("name") or "")
        fid = name_to_id.get(name) or feat.get("feature_id")
        if fid:
            feat["feature_id"] = fid
            feat.update(_apply_identity_to_entry(feat, identities.get(fid)))
        elif not feat.get("owner"):
            feat["owner"] = _default_owner_for_entry(feat)

        deps = list(feat.get("dependencies") or [])
        feat["dependencies_resolved"] = resolve_dependency_refs(
            deps,
            name_to_id=name_to_id,
            id_to_name=id_to_name,
            features_by_name=by_name,
        )

        model_names = list(feat.get("models_using") or [])
        feat["models_using_detail"] = [
            {
                "model_name": m,
                "feature_id": fid,
            }
            for m in model_names
        ]


def build_feature_registry_catalog(data_dir: str) -> dict[str, Any]:
    schema, columns = _schema_columns()
    models_by_feat = _scan_models(data_dir)
    backlog_doc = load_feature_backlog(data_dir)
    backlog_by_name = {str(r.get("name")): r for r in (backlog_doc.get("features") or []) if r.get("name")}
    store = load_store(data_dir)
    overrides = dict(store.get("overrides") or {})
    imported = dict(store.get("imported_features") or {})
    custom_groups = dict(store.get("custom_groups") or {})

    policy_by_name: dict[str, Any] = {}
    try:
        from chain_replay_ml.feature_policy import load_feature_policy_registry

        policy_by_name = load_feature_policy_registry().features
    except Exception:
        policy_by_name = {}

    features: list[dict[str, Any]] = []
    seen: set[str] = set()
    for name in sorted(columns.keys()):
        entry = _catalog_entry_from_registry(
            name, columns[name], models_by_feat=models_by_feat, columns=columns,
            policy_by_name=policy_by_name,
        )
        if name in overrides:
            entry = _apply_store_overlay(entry, overrides[name])
        features.append(entry)
        seen.add(name)

    for name, row in sorted(backlog_by_name.items()):
        if name in seen:
            continue
        features.append(_catalog_entry_from_backlog(row, columns=columns))
        seen.add(name)

    for name, row in sorted(imported.items()):
        if name in seen:
            continue
        features.append(_catalog_entry_from_store_row(name, row, feature_id=None, columns=columns))
        seen.add(name)

    all_names = [f["name"] for f in features]
    ensure_feature_identities(store, features, created_by=store.get("created_by") or "System")
    _enrich_features_with_identities(features, store)
    disabled_map = dict(store.get("disabled_features") or {})
    _apply_registry_active_state(features, disabled_map)
    save_store(data_dir, store)

    id_index = {
        str(f.get("feature_id")): f
        for f in features
        if f.get("feature_id")
    }

    implemented_n = sum(1 for f in features if f["implementation_status"] == "implemented")
    planned_n = sum(1 for f in features if f["implementation_status"] in ("planned", "in_progress"))
    not_impl_n = sum(1 for f in features if f["implementation_status"] == "not_implemented")

    groups = [
        {"id": gid, "label": _GROUP_LABELS.get(gid, gid), "filter": _GROUP_FILTER.get(gid, gid)}
        for gid in GROUP_FEATURE_SOURCES
    ]
    for gid, meta in custom_groups.items():
        groups.append({
            "id": gid,
            "label": meta.get("label") or _snake_to_display(gid),
            "filter": meta.get("filter") or meta.get("label") or gid,
        })
    if disabled_map:
        groups.append({
            "id": DISABLED_GROUP_ID,
            "label": _DISABLED_GROUP_LABEL,
            "filter": _DISABLED_GROUP_FILTER,
        })

    projects = list_projects(store)
    group_to_projects: dict[str, list[str]] = {}
    for proj in projects:
        for gid in proj.get("group_ids") or []:
            group_to_projects.setdefault(str(gid), []).append(str(proj["id"]))
    for g in groups:
        g["project_ids"] = sorted(group_to_projects.get(g["id"], []))

    from .feature_domains import DOMAIN_LABELS, DOMAIN_ORDER, domain_counts

    counts = domain_counts()
    domains = [
        {
            "id": d,
            "label": DOMAIN_LABELS[d],
            "filter": d,
            "count": int(counts.get(d, 0)),
            "chip_label": f"{DOMAIN_LABELS[d]} ({counts.get(d, 0)})",
        }
        for d in DOMAIN_ORDER
    ]

    return {
        "registry_version": store.get("registry_version"),
        "registry_created_by": store.get("created_by"),
        "registry_created_on": store.get("created_on"),
        "registry_description": store.get("description"),
        "registry_history": list(store.get("history") or []),
        "schema_version": schema.get("version"),
        "schema_registry_hash": schema_registry_hash(schema) if schema.get("columns") else None,
        "feature_count": len(features),
        "stats": {
            "implemented": implemented_n,
            "planned": planned_n,
            "not_implemented": not_impl_n,
            "deprecated": sum(1 for f in features if f["implementation_status"] == "deprecated"),
            "disabled": len(disabled_map),
        },
        "groups": groups,
        "domains": domains,
        "projects": projects,
        "status_options": [
            {"id": k, "label": v} for k, v in _STATUS_LABELS.items()
        ],
        "pipeline_surfaces": [{"id": k, "label": v} for k, v in _PIPELINE_SURFACES],
        "features": features,
        "feature_id_index": {fid: row.get("name") for fid, row in id_index.items()},
    }

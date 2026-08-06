"""Feature Test UI — dataset row preview + formula inspection."""

from __future__ import annotations

import ast
import math
import operator
import os
import random
import re
from typing import Any

import pandas as pd

from .auditor import list_datasets
from .schema_column_docs import RICH_COLUMN_DOCS
from .schema_registry import column_display_name, columns_map, enrich_column_view, load_schema_registry
from .writer import datasets_dir

_IDENTITY_COLS = frozenset({
    "trading_day", "market", "expiry", "timestamp", "token", "symbol",
    "strike", "option_type",
})

_SKIP_INSPECTION_DEPS = frozenset({
    "timestamp", "token", "symbol", "trading_day", "market", "expiry",
    "strike", "option_type", "minutes_to_expiry", "minutes_since_open",
    "minutes_to_close", "is_expiry_day", "is_call", "is_first_hour", "is_last_hour",
})

_RAW_STORED_FEATURES = frozenset({
    "spot", "ltp", "oi", "volume", "bid_ask_spread", "current_iv", "roll_iv",
    "delta", "gamma", "theta", "vega", "vanna", "volga", "charm", "speed", "abs_delta",
})

# Formula doc names → dataset parquet column names
_FORMULA_TOKEN_TO_COLUMN = {
    "current_ltp": "ltp",
    "current_spot": "spot",
    "current_iv": "current_iv",
    "iv": "current_iv",
    "current_volume": "volume",
}

# Greek / display names in formula docs → registry column names
_FORMULA_TOKEN_ALIASES = {
    "Delta": "delta",
    "Gamma": "gamma",
    "Theta": "theta",
    "Vega": "vega",
    "IV": "current_iv",
}

_FORMULA_ID_RE = re.compile(r"\b([a-zA-Z_][a-zA-Z0-9_]*)\b")
_FORMULA_KEYWORDS = frozenset({
    "and", "or", "not", "min", "max", "abs", "if", "else", "from", "over", "per", "to",
    "the", "with", "for", "by", "day", "min", "sec", "minutes", "change", "option_type",
    "price", "BS",
})
_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
}
_PREVIEW_ROW_LIMITS = frozenset({10, 20, 50, 100})


def list_registry_datasets(data_dir: str) -> list[dict[str, Any]]:
    """Built datasets with parquet (ready for sampling)."""
    rows = list_datasets(data_dir)
    return [
        {
            "dataset_name": r["dataset_name"],
            "market": r.get("market"),
            "row_count": r.get("row_count"),
            "feature_count": r.get("feature_count"),
            "target_count": r.get("target_count"),
            "has_parquet": bool(r.get("has_parquet")),
            "is_draft": bool(r.get("is_draft")),
        }
        for r in rows
        if r.get("has_parquet") and not r.get("is_draft")
    ]


def load_feature_group_catalog() -> list[dict[str, Any]]:
    """Feature groups with feature names for the test UI tree."""
    from .feature_plugins import GROUP_FEATURE_SOURCES, _REGISTRY_FEATURES
    from .feature_registry_catalog import _GROUP_LABELS

    schema = load_schema_registry()
    groups_meta = dict(schema.get("groups") or {})
    group_order = list(schema.get("groupOrder") or list(GROUP_FEATURE_SOURCES.keys()))

    out: list[dict[str, Any]] = []
    seen_gids: set[str] = set()
    for gid in group_order:
        if gid in seen_gids:
            continue
        seen_gids.add(gid)
        block = groups_meta.get(gid) or {}
        feats = list(block.get("features") or _REGISTRY_FEATURES.get(gid) or [])
        if not feats:
            continue
        out.append({
            "id": gid,
            "label": block.get("label") or _GROUP_LABELS.get(gid, gid),
            "features": feats,
            "feature_count": len(feats),
        })
    for gid, feats in _REGISTRY_FEATURES.items():
        if gid in seen_gids:
            continue
        if not feats:
            continue
        out.append({
            "id": gid,
            "label": _GROUP_LABELS.get(gid, gid),
            "features": list(feats),
            "feature_count": len(feats),
        })
    return out


def load_feature_metadata_index(
    *,
    parquet_cols: set[str] | None = None,
) -> dict[str, dict[str, Any]]:
    """Lightweight feature metadata map for Feature Test UI."""
    schema = load_schema_registry()
    cols = columns_map(schema)
    names: set[str] = set(cols.keys())
    for group in load_feature_group_catalog():
        names.update(group.get("features") or [])
    return {
        name: _feature_meta_core(
            name, schema=schema, columns=cols, parquet_cols=parquet_cols,
        )
        for name in sorted(names)
    }


def load_feature_catalog(
    data_dir: str | None = None,
    *,
    dataset_name: str | None = None,
) -> list[dict[str, Any]]:
    """All registry/plugin features with formula, derivation, and validation summary."""
    parquet_cols: set[str] | None = None
    if data_dir and dataset_name:
        try:
            parquet_cols = _parquet_column_names(_parquet_path(data_dir, dataset_name))
        except (FileNotFoundError, ValueError, OSError):
            parquet_cols = None

    index = load_feature_metadata_index(parquet_cols=parquet_cols)
    group_map: dict[str, dict[str, str]] = {}
    for group in load_feature_group_catalog():
        gid = str(group.get("id") or "")
        label = str(group.get("label") or gid)
        for feat in group.get("features") or []:
            group_map.setdefault(str(feat), {"group_id": gid, "group_label": label})

    rows: list[dict[str, Any]] = []
    for name, meta in index.items():
        grp = group_map.get(name, {})
        validation = dict(meta.get("dependency_validation") or {})
        ratio = meta.get("ratio_inspection")
        rows.append({
            "name": name,
            "group_id": grp.get("group_id", ""),
            "group_label": grp.get("group_label", ""),
            "display_name": meta.get("display_name") or name,
            "description": meta.get("description") or "",
            "formula": meta.get("formula") or "",
            "formula_display": meta.get("formula_display") or "",
            "is_derived": bool(meta.get("is_derived")),
            "ratio_inspection": ratio,
            "dependency_validation": validation,
            "missing": list(validation.get("missing") or []),
            "validation_passed": bool(validation.get("passed")),
            "direct_dependencies": meta.get("direct_dependencies") or [],
            "in_dataset": name in parquet_cols if parquet_cols is not None else None,
        })
    return rows


def _parquet_path(data_dir: str, dataset_name: str) -> str:
    safe = str(dataset_name).strip()
    if not safe or "/" in safe or "\\" in safe or ".." in safe:
        raise ValueError("Invalid dataset_name")
    path = os.path.join(datasets_dir(data_dir), f"{safe}.parquet")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Parquet not found for dataset {safe}")
    return path


def _parquet_column_names(path: str) -> set[str]:
    try:
        import pyarrow.parquet as pq

        return set(pq.ParquetFile(path).schema_arrow.names)
    except ImportError:
        return set(pd.read_parquet(path, columns=[]).columns)


def _json_safe(val: Any) -> Any:
    if val is None:
        return None
    if isinstance(val, (str, bool, int)):
        return val
    if isinstance(val, float):
        if math.isnan(val) or math.isinf(val):
            return None
        return float(val)
    try:
        if pd.isna(val):
            return None
    except (TypeError, ValueError):
        pass
    return str(val)


def _normalize_formula(expr: str) -> str:
    text = str(expr or "").strip()
    text = text.replace("×", "*").replace("÷", "/")
    text = re.sub(r"\|([a-zA-Z_][a-zA-Z0-9_]*)\|", r"abs(\1)", text)
    text = re.sub(r"\s+", " ", text)
    return text


def _parse_formula_identifiers(formula: str) -> list[str]:
    norm = _normalize_formula(formula)
    out: list[str] = []
    seen: set[str] = set()
    for tok in _FORMULA_ID_RE.findall(norm):
        if tok.lower() in _FORMULA_KEYWORDS:
            continue
        if tok not in seen:
            seen.add(tok)
            out.append(tok)
    return out


def _extract_formula_ast_references(formula: str) -> list[str]:
    """Walk formula AST and collect identifier names in source order."""
    norm = _normalize_formula(formula)
    if not norm or not re.search(r"[a-zA-Z_]", norm):
        return []
    try:
        tree = ast.parse(norm, mode="eval")
    except SyntaxError:
        return []

    ordered: list[tuple[int, int, str]] = []

    class _RefCollector(ast.NodeVisitor):
        def visit_Name(self, node: ast.Name) -> None:
            ordered.append((node.lineno, node.col_offset, str(node.id)))
            self.generic_visit(node)

    _RefCollector().visit(tree.body)
    ordered.sort(key=lambda item: (item[0], item[1]))

    refs: list[str] = []
    seen: set[str] = set()
    for _, _, tok in ordered:
        if not tok or tok.lower() in _FORMULA_KEYWORDS:
            continue
        if tok not in seen:
            seen.add(tok)
            refs.append(tok)
    return refs


def _extract_formula_references(formula: str, *, feature_name: str = "") -> list[str]:
    """All formula identifiers: AST walk + regex fallback, deduped, order preserved."""
    norm = _normalize_formula(formula)
    if not norm:
        return []

    refs: list[str] = []
    seen: set[str] = set()

    def _add(tok: str) -> None:
        if not tok or tok == feature_name or tok.lower() in _FORMULA_KEYWORDS:
            return
        if tok not in seen:
            seen.add(tok)
            refs.append(tok)

    for tok in _extract_formula_ast_references(norm):
        _add(tok)
    for tok in _parse_formula_identifiers(norm):
        _add(tok)
    return refs


def _registry_resolve_token(token: str, registry_cols: dict[str, Any]) -> str | None:
    """Return registry column name if token exists in Feature Registry."""
    keys = set(registry_cols.keys())
    if token in keys:
        return token
    mapped = _FORMULA_TOKEN_TO_COLUMN.get(token)
    if mapped and mapped in keys:
        return mapped
    alias = _FORMULA_TOKEN_ALIASES.get(token)
    if alias and alias in keys:
        return alias
    lower = str(token or "").lower()
    if lower in keys:
        return lower
    if lower:
        alias_lower = _FORMULA_TOKEN_ALIASES.get(token) or _FORMULA_TOKEN_ALIASES.get(
            token[:1].upper() + token[1:] if token else "",
        )
        if alias_lower and alias_lower in keys:
            return alias_lower
    return None


def validate_formula_dependencies(
    feature_name: str,
    formula: str,
    registry_cols: dict[str, Any],
    *,
    parquet_cols: set[str] | None = None,
) -> dict[str, Any]:
    """Compare formula references against Feature Registry (and optional dataset parquet)."""
    tokens = _extract_formula_references(formula, feature_name=feature_name)
    references: list[dict[str, Any]] = []
    missing: list[str] = []

    for token in tokens:
        registry_column = _registry_resolve_token(token, registry_cols)
        in_registry = registry_column is not None
        in_dataset = None
        if parquet_cols is not None:
            in_dataset = bool(
                registry_column and registry_column in parquet_cols
            ) or token in parquet_cols
        entry = {
            "token": token,
            "registry_column": registry_column,
            "in_registry": in_registry,
            "in_dataset": in_dataset,
            "status": "ok" if in_registry else "missing",
        }
        references.append(entry)
        if not in_registry:
            missing.append(token)

    detected = len(tokens) - len(missing)
    return {
        "expected_count": len(tokens),
        "detected_count": detected,
        "passed": len(missing) == 0 and len(tokens) > 0,
        "references": references,
        "missing": missing,
        "detected_tokens": tokens,
    }


def _preview_columns_from_validation(
    validation: dict[str, Any],
    parquet_cols: set[str],
) -> list[str]:
    """Columns for preview: registry-resolved names first, else raw token if in parquet."""
    out: list[str] = []
    for ref in validation.get("references") or []:
        col = ref.get("registry_column")
        token = str(ref.get("token") or "").strip()
        if col and col in parquet_cols:
            out.append(str(col))
        elif token and token in parquet_cols:
            out.append(token)
    return list(dict.fromkeys(out))


def _infer_ratio_parts_from_name(name: str) -> tuple[str, str] | None:
    """Infer numerator/denominator from feature names like spot_ema9_to_ltp_ratio."""
    for suffix, denom in (
        ("_to_ltp_ratio", "ltp"),
        ("_to_spot_ratio", "spot"),
    ):
        if name.endswith(suffix):
            return name[: -len(suffix)], denom
    return None


def _ratio_denominator_column(denominator: str, available: set[str]) -> str:
    mapped = _FORMULA_TOKEN_TO_COLUMN.get(denominator) or denominator
    if mapped in available:
        return mapped
    if denominator in available:
        return denominator
    return mapped


def _parse_simple_ratio_parts(formula: str) -> tuple[str, str] | None:
    """Return (numerator, denominator) for simple A / B formulas."""
    norm = _normalize_formula(formula)
    if not _is_simple_ratio_formula(norm):
        return None
    parts = [p.strip() for p in norm.split("/") if p.strip()]
    if len(parts) != 2:
        return None
    return parts[0], parts[1]


def _build_ratio_inspection(
    name: str,
    formula: str,
    cols: dict[str, Any],
    parquet_cols: set[str] | None,
) -> dict[str, Any] | None:
    """Ratio feature: rearranged formula + derived numerator metadata."""
    parts = _parse_simple_ratio_parts(formula) or _infer_ratio_parts_from_name(name)
    if not parts:
        return None
    numerator, denominator = parts
    available = set(parquet_cols or ())
    denom_col = _ratio_denominator_column(denominator, available or set(cols.keys()))
    num_registry = _registry_resolve_token(numerator, cols)
    den_registry = _registry_resolve_token(denom_col, cols) or _registry_resolve_token(denominator, cols)
    return {
        "numerator": numerator,
        "denominator": denom_col,
        "numerator_registry_column": num_registry,
        "denominator_registry_column": den_registry,
        "numerator_in_registry": num_registry is not None,
        "denominator_in_registry": den_registry is not None,
        "numerator_in_dataset": numerator in available if parquet_cols is not None else None,
        "denominator_in_dataset": denom_col in available if parquet_cols is not None else None,
        "rearranged_formula": f"{numerator} = {name} × {denom_col}",
        "rearranged_formula_display": f"{numerator} =\n{name} × {denom_col}",
        "inverse_verify_formula": f"{name} * {denom_col}",
        "inverse_verify_label": f"{numerator} (derived)",
    }


def _ratio_preview_dependencies(
    ratio: dict[str, Any],
    available: set[str],
    columns: dict[str, dict[str, Any]],
) -> list[str]:
    """Preview input columns for ratio inspection (denominator + optional stored numerator)."""
    denom = str(ratio.get("denominator") or "")
    denom_col = _ratio_denominator_column(denom, available)
    deps: list[str] = []
    if denom_col in available:
        deps.append(denom_col)
    num = str(ratio.get("numerator") or "")
    if num in available and num not in deps:
        deps.append(num)
    return deps


def _compute_ratio_derived_numerator(
    ratio_val: Any,
    denom_val: Any,
) -> float | None:
    if ratio_val is None or denom_val is None:
        return None
    try:
        return float(ratio_val) * float(denom_val)
    except (TypeError, ValueError):
        return None


def _apply_ratio_row_fields(
    *,
    ratio: dict[str, Any],
    feature_name: str,
    row: dict[str, Any],
    parquet_cols: set[str],
    values: dict[str, Any],
    entry: dict[str, Any] | None = None,
) -> None:
    denom_col = _ratio_denominator_column(str(ratio.get("denominator") or ""), parquet_cols)
    numerator = str(ratio.get("numerator") or "")
    denom_val = row.get(denom_col)
    ratio_val = row.get(feature_name)
    derived_num = _compute_ratio_derived_numerator(ratio_val, denom_val)
    values[denom_col] = denom_val
    values[numerator] = _json_safe(derived_num)
    if numerator in parquet_cols:
        values[f"{numerator}__stored"] = row.get(numerator)
    if entry is not None:
        entry["derived_numerator"] = _json_safe(derived_num)
        entry["denominator_value"] = _json_safe(denom_val)
        if derived_num is not None:
            entry["inverse_calculated"] = _json_safe(derived_num)
        stored_num = row.get(numerator) if numerator in parquet_cols else None
        if stored_num is not None:
            entry["inverse_stored"] = _json_safe(stored_num)
            entry["inverse_match"] = _values_close(derived_num, stored_num)


def _ratio_table_columns(
    ratio: dict[str, Any],
    feature_name: str,
    *,
    schema: dict[str, Any] | None,
    parquet_cols: set[str],
    result_display: str = "Result",
) -> list[dict[str, Any]]:
    denom_col = _ratio_denominator_column(str(ratio.get("denominator") or ""), parquet_cols)
    numerator = str(ratio.get("numerator") or "")
    cols = [_column_entry(denom_col, schema=schema)]
    if numerator in parquet_cols:
        cols.append({
            "name": f"{numerator}__stored",
            "display_name": f"{numerator} (stored)",
            "role": "stored_numerator",
        })
    cols.append({
        "name": numerator,
        "display_name": f"{numerator} (derived)",
        "role": "derived",
    })
    cols.append({
        "name": feature_name,
        "display_name": result_display,
        "role": "result" if result_display == "Result" else None,
    })
    return cols


def _plan_raw_preview(
    selected: list[str],
    *,
    schema: dict[str, Any],
    columns: dict[str, dict[str, Any]],
    parquet_cols: set[str],
) -> tuple[list[str], list[dict[str, Any]], list[tuple[str, dict[str, Any]]]]:
    """Build raw-mode read columns, table headers, and ratio specs for derived numerators."""
    read_cols: set[str] = set(selected)
    table_columns: list[dict[str, Any]] = []
    ratio_specs: list[tuple[str, dict[str, Any]]] = []
    added_cols: set[str] = set()

    def _append_col(col: dict[str, Any]) -> None:
        key = col.get("name")
        if not key or key in added_cols:
            return
        added_cols.add(str(key))
        table_columns.append(col)

    for name in selected:
        meta = _feature_meta_core(
            name, schema=schema, columns=columns, parquet_cols=parquet_cols,
        )
        ratio = meta.get("ratio_inspection")
        if ratio:
            ratio_specs.append((name, ratio))
            dep_names = _ratio_preview_dependencies(ratio, parquet_cols, columns)
            read_cols.update(dep_names)
            read_cols.add(name)
            denom_col = _ratio_denominator_column(str(ratio.get("denominator") or ""), parquet_cols)
            numerator = str(ratio.get("numerator") or "")
            _append_col(_column_entry(denom_col, schema=schema))
            if numerator in parquet_cols:
                _append_col({
                    "name": f"{numerator}__stored",
                    "display_name": f"{numerator} (stored)",
                    "role": "stored_numerator",
                })
            _append_col({
                "name": numerator,
                "display_name": f"{numerator} (derived)",
                "role": "derived",
                "source_feature": name,
            })
        _append_col(_column_entry(name, schema=schema))

    return sorted(read_cols), table_columns, ratio_specs


def _is_simple_ratio_formula(formula: str) -> bool:
    norm = _normalize_formula(formula)
    if not norm or "(" in norm:
        return False
    if "/" not in norm:
        return False
    parts = [p.strip() for p in norm.split("/") if p.strip()]
    return len(parts) == 2 and all(_FORMULA_ID_RE.fullmatch(p.replace(" ", "")) or "_" in p for p in parts)


def _resolve_spot_change_column(available: set[str]) -> str | None:
    if "spot_change" in available:
        return "spot_change"
    matches = sorted(c for c in available if c.startswith("spot_change_"))
    return matches[0] if matches else None


def _resolve_formula_token_to_column(token: str, available: set[str]) -> str | None:
    """Map a formula identifier to a stored dataset column."""
    if token in available:
        return token
    mapped = _FORMULA_TOKEN_TO_COLUMN.get(token)
    if mapped and mapped in available:
        return mapped
    return None


def _formula_tokens_to_columns(tokens: list[str], available: set[str]) -> list[str]:
    out: list[str] = []
    for tok in tokens:
        col = _resolve_formula_token_to_column(tok, available)
        if col:
            out.append(col)
    return list(dict.fromkeys(out))


def _resolve_inspection_dependency_names(
    name: str,
    view: dict[str, Any],
    columns: dict[str, dict[str, Any]],
    *,
    parquet_cols: set[str] | None = None,
    dependency_validation: dict[str, Any] | None = None,
) -> list[str]:
    """Resolve dependency columns for formula inspection preview."""
    doc = dict(RICH_COLUMN_DOCS.get(name) or {})
    available = set(parquet_cols or columns.keys())
    explicit = list(doc.get("inspection_dependencies") or doc.get("formula_dependencies") or [])
    formula_for_ratio = str(view.get("formula_doc") or "")
    ratio_inspection = _build_ratio_inspection(name, formula_for_ratio, columns, parquet_cols)
    use_ratio_deps = bool(
        ratio_inspection
        and not explicit
        and not ratio_inspection.get("numerator_in_registry")
    )

    if explicit:
        resolved: list[str] = []
        for dep in explicit:
            if dep == "spot_change":
                sc = _resolve_spot_change_column(available)
                if sc:
                    resolved.append(sc)
            elif dep in available:
                resolved.append(dep)
            else:
                mapped = _registry_resolve_token(dep, columns)
                if mapped and mapped in available:
                    resolved.append(mapped)
        deps = resolved
    elif use_ratio_deps and ratio_inspection and parquet_cols is not None:
        deps = _ratio_preview_dependencies(ratio_inspection, available, columns)
    elif dependency_validation and parquet_cols is not None:
        deps = _preview_columns_from_validation(dependency_validation, available)
        if not deps:
            formula = str(view.get("formula_doc") or "")
            tokens = _extract_formula_references(formula, feature_name=name)
            deps = _formula_tokens_to_columns(tokens, available)
    else:
        formula = str(view.get("formula_doc") or "")
        if _is_simple_ratio_formula(formula):
            tokens = [tok for tok in _extract_formula_references(formula, feature_name=name)]
            resolve_available = set(available)
            if parquet_cols is None:
                resolve_available |= set(tokens)
            deps = _formula_tokens_to_columns(tokens, resolve_available)
        else:
            registry_deps = [str(d) for d in (view.get("depends_on") or []) if d]
            deps = [d for d in registry_deps if d not in _SKIP_INSPECTION_DEPS]
            if parquet_cols is not None:
                deps = [d for d in deps if d in available]

    return list(dict.fromkeys(deps))


def _is_derived_feature(name: str, view: dict[str, Any], inspection_deps: list[str]) -> bool:
    if name in _RAW_STORED_FEATURES:
        return False
    if view.get("implementation"):
        return True
    if inspection_deps:
        return True
    formula = str(view.get("formula_doc") or "")
    return _is_simple_ratio_formula(formula)


def _feature_meta_core(
    name: str,
    *,
    schema: dict[str, Any] | None = None,
    columns: dict[str, dict[str, Any]] | None = None,
    parquet_cols: set[str] | None = None,
) -> dict[str, Any]:
    """Metadata without dependency tree (safe for recursive tree walks)."""
    reg = schema or load_schema_registry()
    cols = columns or columns_map(reg)
    view = enrich_column_view(name, reg)
    doc = dict(RICH_COLUMN_DOCS.get(name) or {})
    formula = str(view.get("formula_doc") or view.get("formula_ref") or "").strip()
    ratio_inspection = _build_ratio_inspection(name, formula, cols, parquet_cols)
    explicit_deps = list(doc.get("inspection_dependencies") or doc.get("formula_dependencies") or [])
    use_ratio_ui = bool(
        ratio_inspection
        and not explicit_deps
        and not ratio_inspection.get("numerator_in_registry")
    )
    if not use_ratio_ui:
        ratio_inspection = None
    dependency_validation = validate_formula_dependencies(
        name, formula, cols, parquet_cols=parquet_cols,
    )
    direct_dep_names = _resolve_inspection_dependency_names(
        name, view, cols,
        parquet_cols=parquet_cols,
        dependency_validation=dependency_validation,
    )
    is_derived = _is_derived_feature(name, view, direct_dep_names)
    dep_entries = []
    if use_ratio_ui and ratio_inspection:
        denom = str(ratio_inspection.get("denominator") or "")
        denom_col = _resolve_formula_token_to_column(
            denom, set(parquet_cols or cols.keys()),
        ) or denom
        dep_entries.append({
            "name": denom_col,
            "display_name": column_display_name(denom_col, reg),
            "in_registry": bool(ratio_inspection.get("denominator_in_registry")),
            "role": "denominator",
        })
        num = str(ratio_inspection.get("numerator") or "")
        dep_entries.append({
            "name": num,
            "display_name": num,
            "in_registry": bool(ratio_inspection.get("numerator_in_registry")),
            "role": "derived_numerator",
            "registry_warning": (
                None if ratio_inspection.get("numerator_in_registry")
                else "Derived (not in registry)"
            ),
        })
    else:
        for dep in direct_dep_names:
            entry = {
                "name": dep,
                "display_name": column_display_name(dep, reg),
            }
            ref = next(
                (
                    r for r in (dependency_validation.get("references") or [])
                    if r.get("registry_column") == dep or r.get("token") == dep
                ),
                None,
            )
            if ref:
                entry["in_registry"] = bool(ref.get("in_registry"))
            for tok, col in _FORMULA_TOKEN_TO_COLUMN.items():
                if col == dep:
                    entry["formula_token"] = tok
                    break
            dep_entries.append(entry)
    if is_derived and formula:
        formula_display = f"{name} =\n{formula}"
    elif is_derived:
        formula_display = f"{name} ="
    else:
        formula_display = "Raw dataset column (no calculation)"
    verify_formula = str(
        doc.get("inspection_verify_formula") or doc.get("verify_formula") or ""
    ).strip()
    if not verify_formula and _is_simple_ratio_formula(formula):
        verify_formula = _normalize_formula(formula)
    return {
        "name": name,
        "display_name": view.get("display_name") or column_display_name(name, reg),
        "description": view.get("description") or "",
        "formula": formula,
        "formula_display": formula_display,
        "verify_formula": verify_formula,
        "is_derived": is_derived,
        "is_raw_column": not is_derived,
        "direct_dependencies": dep_entries,
        "dependency_validation": dependency_validation,
        "ratio_inspection": ratio_inspection,
    }


def _build_dependency_tree(
    name: str,
    columns: dict[str, dict[str, Any]],
    *,
    schema: dict[str, Any] | None = None,
    parquet_cols: set[str] | None = None,
    depth: int = 0,
    seen: set[str] | None = None,
    core_cache: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    if seen is None:
        seen = set()
    if core_cache is None:
        core_cache = {}
    if name in seen or depth > 8:
        return []
    seen.add(name)
    if name not in core_cache:
        core_cache[name] = _feature_meta_core(
            name, schema=schema, columns=columns, parquet_cols=parquet_cols,
        )
    meta = core_cache[name]
    nodes: list[dict[str, Any]] = []
    for dep in meta.get("direct_dependencies") or []:
        dep_name = str(dep.get("name") or dep) if isinstance(dep, dict) else str(dep)
        if dep_name in _SKIP_INSPECTION_DEPS:
            continue
        dep_meta = core_cache.get(dep_name) or _feature_meta_core(
            dep_name, schema=schema, columns=columns, parquet_cols=parquet_cols,
        )
        core_cache[dep_name] = dep_meta
        nodes.append({
            "name": dep_name,
            "display_name": dep_meta.get("display_name") or column_display_name(dep_name, schema),
            "is_derived": bool(dep_meta.get("is_derived")),
            "children": _build_dependency_tree(
                dep_name,
                columns,
                schema=schema,
                parquet_cols=parquet_cols,
                depth=depth + 1,
                seen=set(seen),
                core_cache=core_cache,
            ),
        })
    return nodes


def get_feature_inspection_meta(
    name: str,
    *,
    schema: dict[str, Any] | None = None,
    columns: dict[str, dict[str, Any]] | None = None,
    parquet_cols: set[str] | None = None,
) -> dict[str, Any]:
    """Metadata bundle for formula inspection."""
    reg = schema or load_schema_registry()
    cols = columns or columns_map(reg)
    core_cache: dict[str, dict[str, Any]] = {}
    core = _feature_meta_core(name, schema=reg, columns=cols, parquet_cols=parquet_cols)
    core_cache[name] = core
    return {
        **core,
        "dependency_tree": _build_dependency_tree(
            name, cols, schema=reg, parquet_cols=parquet_cols, core_cache=core_cache,
        ),
    }


def _build_verify_env(dep_names: list[str], row: dict[str, Any]) -> dict[str, float]:
    env: dict[str, float] = {}
    row_keys = set(row.keys())
    sc_col = _resolve_spot_change_column(row_keys)
    for dep in dep_names:
        val = row.get(dep)
        if val is not None:
            try:
                env[dep] = float(val)
            except (TypeError, ValueError):
                pass
    if sc_col and sc_col in row and row[sc_col] is not None:
        try:
            env["spot_change"] = float(row[sc_col])
        except (TypeError, ValueError):
            pass
    for extra in ("roll_age_min", "rows_since_roll", "ltp", "spot"):
        if extra in row and row[extra] is not None:
            try:
                env[extra] = float(row[extra])
            except (TypeError, ValueError):
                pass
    # Formula doc aliases (e.g. current_ltp → ltp column value)
    if "ltp" in env:
        env["current_ltp"] = env["ltp"]
    elif row.get("ltp") is not None:
        try:
            env["current_ltp"] = float(row["ltp"])
        except (TypeError, ValueError):
            pass
    if "spot" in env:
        env["current_spot"] = env["spot"]
    elif row.get("spot") is not None:
        try:
            env["current_spot"] = float(row["spot"])
        except (TypeError, ValueError):
            pass
    return env


def _evaluate_formula(formula: str, values: dict[str, Any]) -> float | None:
    expr = _normalize_formula(formula)
    if not expr:
        return None
    env: dict[str, float] = {}
    for key, raw in values.items():
        if raw is None:
            env[key] = float("nan")
        else:
            try:
                env[key] = float(raw)
            except (TypeError, ValueError):
                return None

    def _eval_node(node: ast.AST) -> float | None:
        if isinstance(node, ast.Expression):
            return _eval_node(node.body)
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return float(node.value)
            return None
        if isinstance(node, ast.Num):  # pragma: no cover - py<3.8
            return float(node.n)
        if isinstance(node, ast.Name):
            val = env.get(node.id)
            if val is None or (isinstance(val, float) and math.isnan(val)):
                return None
            return float(val)
        if isinstance(node, ast.BinOp):
            left = _eval_node(node.left)
            right = _eval_node(node.right)
            if left is None or right is None:
                return None
            if isinstance(node.op, ast.Div) and right == 0:
                return None
            op = _BIN_OPS.get(type(node.op))
            if not op:
                return None
            try:
                return float(op(left, right))
            except (ZeroDivisionError, OverflowError, ValueError):
                return None
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            val = _eval_node(node.operand)
            return None if val is None else -val
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            fn = node.func.id
            args = [_eval_node(a) for a in node.args]
            if any(a is None for a in args):
                return None
            if fn == "max" and len(args) == 2:
                return float(max(args[0], args[1]))
            if fn == "min" and len(args) == 2:
                return float(min(args[0], args[1]))
            if fn == "abs" and len(args) == 1:
                return float(abs(args[0]))
        return None

    try:
        tree = ast.parse(expr, mode="eval")
        return _eval_node(tree)
    except (SyntaxError, ValueError, TypeError):
        return None


def _values_close(a: Any, b: Any, *, rtol: float = 1e-4, atol: float = 1e-4) -> bool:
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    try:
        fa = float(a)
        fb = float(b)
    except (TypeError, ValueError):
        return a == b
    if math.isnan(fa) and math.isnan(fb):
        return True
    return math.isclose(fa, fb, rel_tol=rtol, abs_tol=atol)


def _parquet_row_count(path: str) -> int:
    try:
        import pyarrow.parquet as pq

        return int(pq.ParquetFile(path).metadata.num_rows)
    except ImportError:
        return int(len(pd.read_parquet(path)))


def _format_token_label(meta: dict[str, Any]) -> str:
    tok = str(meta.get("token") or "").strip()
    strike = meta.get("strike")
    opt = meta.get("option_type")
    sym = meta.get("symbol")
    rc = meta.get("row_count")
    parts: list[str] = []
    if sym:
        parts.append(str(sym))
    elif strike is not None and opt:
        try:
            parts.append(f"{float(strike):g} {opt}")
        except (TypeError, ValueError):
            parts.append(f"{strike} {opt}")
    if tok:
        parts.append(tok)
    if rc is not None:
        parts.append(f"{int(rc):,} rows")
    return " · ".join(parts) if parts else tok or "—"


def _parquet_token_catalog(path: str) -> dict[str, Any]:
    """Map token → parquet row indices plus display metadata."""
    available = _parquet_column_names(path)
    if "token" not in available:
        return {
            "has_token_column": False,
            "tokens": [],
            "indices_by_token": {},
            "total_rows": _parquet_row_count(path),
        }

    id_cols = [c for c in ("token", "symbol", "strike", "option_type", "expiry") if c in available]
    df = pd.read_parquet(path, columns=id_cols)
    indices_by_token: dict[str, list[int]] = {}
    meta_by_token: dict[str, dict[str, Any]] = {}
    for idx, row in df.iterrows():
        tok = str(row.get("token") or "").strip()
        if not tok:
            continue
        indices_by_token.setdefault(tok, []).append(int(idx))
        if tok not in meta_by_token:
            meta_by_token[tok] = {
                "token": tok,
                "symbol": _json_safe(row.get("symbol")),
                "strike": _json_safe(row.get("strike")),
                "option_type": _json_safe(row.get("option_type")),
                "expiry": _json_safe(row.get("expiry")),
            }

    tokens: list[dict[str, Any]] = []
    for tok, indices in indices_by_token.items():
        entry = dict(meta_by_token.get(tok) or {"token": tok})
        entry["row_count"] = len(indices)
        entry["label"] = _format_token_label(entry)
        tokens.append(entry)
    tokens.sort(key=lambda t: (-int(t.get("row_count") or 0), str(t.get("token") or "")))

    return {
        "has_token_column": True,
        "tokens": tokens,
        "indices_by_token": indices_by_token,
        "total_rows": len(df),
    }


def list_dataset_tokens(data_dir: str, dataset_name: str) -> dict[str, Any]:
    """Distinct tokens in a dataset parquet with row counts."""
    path = _parquet_path(data_dir, dataset_name)
    cat = _parquet_token_catalog(path)
    return {
        "dataset_name": dataset_name,
        "has_token_column": bool(cat.get("has_token_column")),
        "token_count": len(cat.get("tokens") or []),
        "total_rows": int(cat.get("total_rows") or 0),
        "tokens": cat.get("tokens") or [],
    }


def _resolve_token_pool(
    path: str,
    *,
    token: str | None,
) -> tuple[list[int] | None, dict[str, Any] | None]:
    """Return row-index pool for a token filter, or (None, None) for all tokens."""
    tok = str(token or "").strip()
    if not tok:
        return None, None
    cat = _parquet_token_catalog(path)
    if not cat.get("has_token_column"):
        raise ValueError("Dataset parquet has no token column")
    pool = cat.get("indices_by_token", {}).get(tok)
    if not pool:
        raise ValueError(f"Token not found in dataset: {tok}")
    meta = next((t for t in (cat.get("tokens") or []) if t.get("token") == tok), None)
    return list(pool), meta


def _sample_row_indices(
    total_rows: int,
    *,
    row_count: int,
    sampling: str,
    seed: int | None,
    custom_indices: list[int] | None,
    pool: list[int] | None = None,
) -> list[int]:
    if pool is not None:
        pool_set = sorted({int(i) for i in pool})
        if not pool_set:
            raise ValueError("No rows for selected token")
        max_global = max(pool_set)
        pool_lookup = set(pool_set)
        effective_total = len(pool_set)
        n = max(1, min(int(row_count), effective_total))
        mode = str(sampling or "random").strip().lower()
        if mode == "custom":
            raw = [int(i) for i in (custom_indices or [])]
            if not raw:
                raise ValueError("custom sampling requires row_indices")
            out = sorted({
                i for i in (max(0, min(max_global, x)) for x in raw)
                if i in pool_lookup
            })
            if not out:
                raise ValueError("Custom row numbers do not match selected token")
            return out[:100]
        if mode == "first":
            return pool_set[:n]
        rng = random.Random(seed)
        if n >= effective_total:
            return pool_set
        return sorted(rng.sample(pool_set, n))

    if total_rows <= 0:
        raise ValueError("Dataset parquet is empty")
    n = max(1, min(int(row_count), total_rows))
    mode = str(sampling or "random").strip().lower()
    if mode == "custom":
        raw = [int(i) for i in (custom_indices or [])]
        if not raw:
            raise ValueError("custom sampling requires row_indices")
        out = sorted({max(0, min(total_rows - 1, i)) for i in raw})
        return out[:100]
    if mode == "first":
        return list(range(n))
    rng = random.Random(seed)
    if n >= total_rows:
        return list(range(total_rows))
    return sorted(rng.sample(range(total_rows), n))


def _read_rows_parquet(
    path: str,
    indices: list[int],
    columns: list[str] | None,
) -> tuple[list[dict[str, Any]], list[str]]:
    if not indices:
        return [], []
    available = _parquet_column_names(path)
    requested = sorted(set(columns or []))
    use_cols = sorted(set(requested) & available) if requested else None
    missing_requested = [c for c in requested if c not in available]
    df = pd.read_parquet(path, columns=use_cols)
    total = len(df)
    rows: list[dict[str, Any]] = []
    for idx in indices:
        if idx < 0 or idx >= total:
            continue
        row = df.iloc[idx].to_dict()
        rows.append({str(k): _json_safe(v) for k, v in row.items()})
    return rows, missing_requested


def _read_random_row_parquet(path: str, *, seed: int | None = None) -> tuple[int, int, dict[str, Any]]:
    """Return (row_index, total_rows, row_dict)."""
    total = _parquet_row_count(path)
    indices = _sample_row_indices(total, row_count=1, sampling="random", seed=seed, custom_indices=None)
    row_index = indices[0]
    rows, _ = _read_rows_parquet(path, [row_index], columns=None)
    if not rows:
        raise ValueError("Could not resolve random row index")
    return row_index, total, rows[0]


def _resolve_feature_names(
    *,
    group_ids: list[str],
    feature_names: list[str],
) -> tuple[list[str], dict[str, str]]:
    """Return ordered feature names and feature→group map."""
    catalog = {g["id"]: g for g in load_feature_group_catalog()}
    feat_to_group: dict[str, str] = {}
    ordered: list[str] = []
    seen: set[str] = set()

    for gid in group_ids:
        block = catalog.get(gid) or {}
        for feat in block.get("features") or []:
            feat_to_group.setdefault(str(feat), gid)
            if feat not in seen:
                seen.add(feat)
                ordered.append(str(feat))

    for feat in feature_names:
        f = str(feat).strip()
        if not f:
            continue
        if f not in feat_to_group:
            for gid, block in catalog.items():
                if f in (block.get("features") or []):
                    feat_to_group[f] = gid
                    break
        if f not in seen:
            seen.add(f)
            ordered.append(f)

    return ordered, feat_to_group


def _identity_from_row(row: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in _IDENTITY_COLS:
        if key in row:
            out[key] = row[key]
    return out


def _column_entry(name: str, *, schema: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "name": name,
        "display_name": column_display_name(name, schema),
    }


def preview_dataset_features(
    data_dir: str,
    *,
    dataset_name: str,
    group_ids: list[str] | None = None,
    feature_names: list[str] | None = None,
    mode: str = "raw",
    row_count: int = 20,
    sampling: str = "random",
    row_indices: list[int] | None = None,
    seed: int | None = None,
    verify_formula: bool = False,
    expand_dependency_tree: bool = False,
    token: str | None = None,
) -> dict[str, Any]:
    """Preview multiple dataset rows in raw or formula inspection mode."""
    gids = [str(g).strip() for g in (group_ids or []) if str(g).strip()]
    feats = [str(f).strip() for f in (feature_names or []) if str(f).strip()]
    if not gids and not feats:
        raise ValueError("Select at least one feature group or feature")

    view_mode = str(mode or "raw").strip().lower()
    if view_mode not in ("raw", "formula"):
        raise ValueError(f"Unknown mode: {mode}")

    count = int(row_count)
    if count not in _PREVIEW_ROW_LIMITS:
        raise ValueError("row_count must be one of 10, 20, 50, 100")

    schema = load_schema_registry()
    columns = columns_map(schema)
    selected, feat_to_group = _resolve_feature_names(group_ids=gids, feature_names=feats)

    path = _parquet_path(data_dir, dataset_name)
    parquet_cols = _parquet_column_names(path)
    total_rows = _parquet_row_count(path)
    token_pool, token_meta = _resolve_token_pool(path, token=token)
    indices = _sample_row_indices(
        total_rows,
        row_count=count,
        sampling=sampling,
        seed=seed,
        custom_indices=row_indices,
        pool=token_pool,
    )

    if view_mode == "formula":
        if len(selected) != 1:
            raise ValueError("Formula Inspection supports one feature at a time. Please select a single feature.")

        feature_name = selected[0]
        feature_meta = get_feature_inspection_meta(
            feature_name, schema=schema, columns=columns, parquet_cols=parquet_cols,
        )
        ratio = feature_meta.get("ratio_inspection")
        if ratio:
            dep_names = _ratio_preview_dependencies(ratio, parquet_cols, columns)
        else:
            dep_names = [d["name"] for d in feature_meta.get("direct_dependencies") or []]

        read_cols = sorted(set(dep_names + [feature_name]) | (set(_IDENTITY_COLS) & parquet_cols))
        raw_rows, missing_cols = _read_rows_parquet(path, indices, columns=read_cols)

        if ratio:
            table_columns = _ratio_table_columns(
                ratio, feature_name, schema=schema, parquet_cols=parquet_cols,
            )
        else:
            table_columns = [_column_entry(n, schema=schema) for n in dep_names]
            table_columns.append({
                "name": feature_name,
                "display_name": "Result",
                "role": "result",
            })

        verify_expr = str(feature_meta.get("verify_formula") or "").strip()
        can_verify = bool(verify_formula and verify_expr and feature_meta.get("is_derived"))
        can_inverse_verify = bool(verify_formula and ratio)

        out_rows: list[dict[str, Any]] = []
        for row_index, row in zip(indices, raw_rows):
            values = {dep: row.get(dep) for dep in dep_names}
            stored = row.get(feature_name)
            entry: dict[str, Any] = {
                "row_index": row_index,
                "row_number": row_index + 1,
                "identity": _identity_from_row(row),
                "values": values,
                "result": stored,
                "stored": stored,
            }
            if ratio:
                _apply_ratio_row_fields(
                    ratio=ratio,
                    feature_name=feature_name,
                    row=row,
                    parquet_cols=parquet_cols,
                    values=values,
                    entry=entry,
                )
            if can_verify and not ratio:
                calc = _evaluate_formula(verify_expr, _build_verify_env(dep_names, row))
                entry["calculated"] = _json_safe(calc)
                entry["match"] = _values_close(calc, stored) if calc is not None else None
            elif can_verify and ratio:
                num = str(ratio.get("numerator") or "")
                if num in parquet_cols:
                    calc = _evaluate_formula(verify_expr, _build_verify_env(dep_names, row))
                    entry["calculated"] = _json_safe(calc)
                    entry["match"] = _values_close(calc, stored) if calc is not None else None
            out_rows.append(entry)

        return {
            "mode": "formula",
            "dataset_name": dataset_name,
            "total_rows": total_rows,
            "row_count": len(out_rows),
            "sampling": sampling,
            "seed": seed,
            "row_indices": indices,
            "token": str(token or "").strip() or None,
            "token_row_count": len(token_pool) if token_pool is not None else None,
            "token_label": (token_meta or {}).get("label"),
            "verify_formula": can_verify or can_inverse_verify,
            "inverse_verify": can_inverse_verify,
            "expand_dependency_tree": bool(expand_dependency_tree),
            "feature": feature_meta,
            "columns": table_columns,
            "rows": out_rows,
            "missing_columns": missing_cols,
            "groups_requested": gids,
            "features_requested": feats,
            "formula_inspection_available": True,
        }

    read_cols, table_columns, ratio_specs = _plan_raw_preview(
        selected, schema=schema, columns=columns, parquet_cols=parquet_cols,
    )
    read_cols = sorted(set(read_cols) | (set(_IDENTITY_COLS) & parquet_cols))
    raw_rows, missing_cols = _read_rows_parquet(path, indices, columns=read_cols)
    out_rows = []
    missing: set[str] = set(missing_cols)
    ratio_by_feature = dict(ratio_specs)
    for row_index, row in zip(indices, raw_rows):
        values: dict[str, Any] = {}
        entry: dict[str, Any] = {
            "row_index": row_index,
            "row_number": row_index + 1,
            "identity": _identity_from_row(row),
            "values": values,
        }
        for name in selected:
            val = row.get(name)
            if val is None and name not in row:
                missing.add(name)
            values[name] = val
            ratio = ratio_by_feature.get(name)
            if ratio:
                _apply_ratio_row_fields(
                    ratio=ratio,
                    feature_name=name,
                    row=row,
                    parquet_cols=parquet_cols,
                    values=values,
                    entry=entry,
                )
        out_rows.append(entry)

    derived_selected = [
        n for n in selected
        if _feature_meta_core(n, schema=schema, columns=columns, parquet_cols=parquet_cols).get("is_derived")
    ]
    return {
        "mode": "raw",
        "dataset_name": dataset_name,
        "total_rows": total_rows,
        "row_count": len(out_rows),
        "sampling": sampling,
        "seed": seed,
        "row_indices": indices,
        "token": str(token or "").strip() or None,
        "token_row_count": len(token_pool) if token_pool is not None else None,
        "token_label": (token_meta or {}).get("label"),
        "columns": table_columns,
        "rows": out_rows,
        "missing_columns": sorted(missing),
        "groups_requested": gids,
        "features_requested": feats,
        "groups_shown": sorted({feat_to_group.get(n, "") for n in selected if feat_to_group.get(n)}),
        "feature_count": len(selected),
        "formula_inspection_available": len(selected) == 1,
        "derived_feature_count": len(derived_selected),
    }


def sample_dataset_features(
    data_dir: str,
    *,
    dataset_name: str,
    group_ids: list[str] | None = None,
    feature_names: list[str] | None = None,
    seed: int | None = None,
    token: str | None = None,
) -> dict[str, Any]:
    """Pick a random parquet row and return selected feature values (legacy single-row API)."""
    gids = [str(g).strip() for g in (group_ids or []) if str(g).strip()]
    feats = [str(f).strip() for f in (feature_names or []) if str(f).strip()]
    if not gids and not feats:
        raise ValueError("Select at least one feature group or feature")

    path = _parquet_path(data_dir, dataset_name)
    total_rows = _parquet_row_count(path)
    token_pool, token_meta = _resolve_token_pool(path, token=token)
    indices = _sample_row_indices(
        total_rows,
        row_count=1,
        sampling="random",
        seed=seed,
        custom_indices=None,
        pool=token_pool,
    )
    row_index = indices[0]
    rows, _ = _read_rows_parquet(path, [row_index], columns=None)
    if not rows:
        raise ValueError("Could not resolve random row index")
    row = rows[0]
    selected, feat_to_group = _resolve_feature_names(group_ids=gids, feature_names=feats)

    values: list[dict[str, Any]] = []
    missing: list[str] = []
    for name in selected:
        if name not in row:
            missing.append(name)
            continue
        gid = feat_to_group.get(name, "")
        values.append({
            "name": name,
            "display_name": column_display_name(name),
            "group_id": gid,
            "value": row.get(name),
        })

    groups_used = sorted({v["group_id"] for v in values if v.get("group_id")})
    return {
        "dataset_name": dataset_name,
        "row_index": row_index,
        "total_rows": total_rows,
        "seed": seed,
        "token": str(token or "").strip() or None,
        "token_row_count": len(token_pool) if token_pool is not None else None,
        "token_label": (token_meta or {}).get("label"),
        "identity": _identity_from_row(row),
        "groups_requested": gids,
        "features_requested": feats,
        "groups_shown": groups_used,
        "feature_count": len(values),
        "missing_columns": missing,
        "values": values,
    }

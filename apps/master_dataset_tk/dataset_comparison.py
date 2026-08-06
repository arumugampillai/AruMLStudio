"""Dataset comparison logic for Tk Dataset Comparison panel (metadata-only)."""

from __future__ import annotations

import json
import os
from typing import Any

SummaryRow = tuple[str, Any, Any]
SummaryGroup = tuple[str, list[SummaryRow]]


def load_dataset_compare_doc(data_dir: str, dataset_name: str) -> dict[str, Any]:
    """Load metadata + expected spec for comparison (no audit, no parquet reads)."""
    from chain_replay_ml.dataset_builder.expected_spec import expected_spec_path
    from chain_replay_ml.dataset_builder.writer import _safe_filename, datasets_dir

    safe_name = _safe_filename(dataset_name)
    out_dir = datasets_dir(data_dir)
    meta_path = os.path.join(out_dir, f"{safe_name}.json")
    if not os.path.isfile(meta_path):
        raise FileNotFoundError(f"Metadata not found for {safe_name}")
    with open(meta_path, encoding="utf-8") as fh:
        meta = json.load(fh)

    expected_doc: dict[str, Any] | None = None
    exp_path = expected_spec_path(data_dir, safe_name)
    if os.path.isfile(exp_path):
        with open(exp_path, encoding="utf-8") as fh:
            expected_doc = json.load(fh)

    return {
        "dataset_name": safe_name,
        "metadata": meta,
        "expected_spec": expected_doc,
    }


def dataset_display_label(doc: dict[str, Any]) -> str:
    return str(doc.get("dataset_name") or "—")


def _meta(doc: dict[str, Any]) -> dict[str, Any]:
    meta = doc.get("metadata")
    return meta if isinstance(meta, dict) else {}


def _expected(doc: dict[str, Any]) -> dict[str, Any]:
    spec = doc.get("expected_spec")
    return spec if isinstance(spec, dict) else {}


def _expected_block(doc: dict[str, Any]) -> dict[str, Any]:
    exp = _expected(doc).get("expected")
    return exp if isinstance(exp, dict) else {}


def feature_names_from_doc(doc: dict[str, Any]) -> list[str]:
    """Resolved feature column names from metadata or expected spec."""
    meta = _meta(doc)
    cols = list(meta.get("feature_columns") or [])
    if not cols:
        cols = list(_expected_block(doc).get("feature_column_names") or [])
    out: list[str] = []
    seen: set[str] = set()
    for raw in cols:
        name = str(raw).strip()
        if name and name not in seen:
            seen.add(name)
            out.append(name)
    return out


def _target_columns(doc: dict[str, Any]) -> list[str]:
    meta = _meta(doc)
    cols = list(
        meta.get("prediction_target_columns")
        or _expected(doc).get("prediction_target_columns")
        or []
    )
    if cols:
        return cols
    horizons = meta.get("prediction_targets")
    if isinstance(horizons, list) and horizons:
        return [
            f"future_ltp_{h}" if not str(h).startswith("future") else str(h)
            for h in horizons
        ]
    return list(_expected_block(doc).get("target_column_names") or [])


def _day_count(doc: dict[str, Any]) -> int | None:
    meta = _meta(doc)
    if meta.get("day_count") is not None:
        return int(meta["day_count"])
    days = meta.get("days")
    if isinstance(days, list):
        return len(days)
    mf = meta.get("master_filter") or {}
    selected = mf.get("selected_days")
    if isinstance(selected, list):
        return len(selected)
    return None


def _selection_label(doc: dict[str, Any]) -> str:
    meta = _meta(doc)
    sm = meta.get("selection_method")
    if isinstance(sm, dict):
        return str(sm.get("summary") or sm.get("label") or sm.get("method") or sm.get("source") or "—")
    if isinstance(sm, str) and sm.strip():
        return sm.strip()
    src = meta.get("selection_source")
    if src:
        return str(src)
    groups = meta.get("feature_groups") or _expected(doc).get("feature_groups") or []
    if groups:
        return f"{len(groups)} group(s)"
    return "—"


def _sampling_interval(doc: dict[str, Any]) -> Any:
    sampling = _meta(doc).get("sampling") or _expected(doc).get("sampling") or {}
    if not isinstance(sampling, dict):
        return "—"
    sec = sampling.get("interval_sec")
    return f"{sec} sec" if sec is not None else "—"


def _sampling_method(doc: dict[str, Any]) -> str:
    sampling = _meta(doc).get("sampling") or _expected(doc).get("sampling") or {}
    if not isinstance(sampling, dict):
        return "—"
    method = sampling.get("method")
    return str(method).replace("_", " ").title() if method else "—"


def _strike_summary(doc: dict[str, Any]) -> str:
    strike = _meta(doc).get("strike_selection") or _expected(doc).get("strike_selection") or {}
    if not isinstance(strike, dict):
        return "—"
    mode = str(strike.get("mode") or "").upper()
    if mode == "DELTA_RANGE":
        dmin = strike.get("delta_min")
        dmax = strike.get("delta_max")
        dtype = strike.get("delta_type") or "delta"
        return f"Delta {dtype} {dmin}–{dmax}"
    band = strike.get("band")
    if band is not None:
        return f"ATM ±{band}"
    return mode.title() if mode else "—"


def _feature_groups_label(doc: dict[str, Any]) -> str:
    groups = list(_meta(doc).get("feature_groups") or _expected(doc).get("feature_groups") or [])
    if not groups:
        return "—"
    return ", ".join(str(g) for g in groups)


def _summary_value(doc: dict[str, Any], key: str) -> Any:
    meta = _meta(doc)
    exp = _expected(doc)
    fp = meta.get("pipeline_fingerprint") or exp.get("pipeline_fingerprint") or {}
    mapping: dict[str, Any] = {
        "dataset_name": doc.get("dataset_name"),
        "market": meta.get("market") or exp.get("market") or "—",
        "feature_profile": meta.get("feature_profile") or exp.get("feature_profile") or "default",
        "selection": _selection_label(doc),
        "trading_days": _day_count(doc),
        "row_count": meta.get("row_count"),
        "feature_count": len(feature_names_from_doc(doc)) or meta.get("feature_count"),
        "target_count": len(_target_columns(doc)),
        "sampling_interval": _sampling_interval(doc),
        "sampling_method": _sampling_method(doc),
        "strike_selection": _strike_summary(doc),
        "feature_groups": _feature_groups_label(doc),
        "created_at": meta.get("created_at") or meta.get("built_at") or "—",
        "builder_version": meta.get("builder_version") or fp.get("builder_version") or "—",
    }
    return mapping.get(key)


def build_summary_comparison(doc_a: dict[str, Any], doc_b: dict[str, Any]) -> list[SummaryGroup]:
    """Side-by-side summary rows grouped for the Summary tab."""
    overview: list[SummaryRow] = [
        ("Dataset", _summary_value(doc_a, "dataset_name"), _summary_value(doc_b, "dataset_name")),
        ("Market", _summary_value(doc_a, "market"), _summary_value(doc_b, "market")),
        ("Feature profile", _summary_value(doc_a, "feature_profile"), _summary_value(doc_b, "feature_profile")),
        ("Selection", _summary_value(doc_a, "selection"), _summary_value(doc_b, "selection")),
        ("Trading days", _summary_value(doc_a, "trading_days"), _summary_value(doc_b, "trading_days")),
        ("Row count", _summary_value(doc_a, "row_count"), _summary_value(doc_b, "row_count")),
        ("Created", _summary_value(doc_a, "created_at"), _summary_value(doc_b, "created_at")),
    ]
    build_cfg: list[SummaryRow] = [
        ("Sampling interval", _summary_value(doc_a, "sampling_interval"), _summary_value(doc_b, "sampling_interval")),
        ("Sampling method", _summary_value(doc_a, "sampling_method"), _summary_value(doc_b, "sampling_method")),
        ("Strike selection", _summary_value(doc_a, "strike_selection"), _summary_value(doc_b, "strike_selection")),
        ("Feature groups", _summary_value(doc_a, "feature_groups"), _summary_value(doc_b, "feature_groups")),
        ("Builder version", _summary_value(doc_a, "builder_version"), _summary_value(doc_b, "builder_version")),
    ]
    features: list[SummaryRow] = [
        ("Feature count", _summary_value(doc_a, "feature_count"), _summary_value(doc_b, "feature_count")),
        ("Target count", _summary_value(doc_a, "target_count"), _summary_value(doc_b, "target_count")),
    ]
    feat_cmp = build_feature_set_comparison(doc_a, doc_b)
    features.extend([
        ("Common features", feat_cmp["common_count"], feat_cmp["common_count"]),
        ("Only in A", feat_cmp["only_a_count"], "—"),
        ("Only in B", "—", feat_cmp["only_b_count"]),
        ("Set overlap %", f"{feat_cmp['overlap_pct']}%", f"{feat_cmp['overlap_pct']}%"),
    ])
    return [
        ("Dataset Overview", overview),
        ("Build Configuration", build_cfg),
        ("Features & Targets", features),
    ]


def build_feature_set_comparison(
    doc_a: dict[str, Any],
    doc_b: dict[str, Any],
) -> dict[str, Any]:
    """Compare feature sets: common, A-only, B-only."""
    feats_a = feature_names_from_doc(doc_a)
    feats_b = feature_names_from_doc(doc_b)
    set_a = set(feats_a)
    set_b = set(feats_b)
    common = sorted(set_a & set_b)
    only_a = sorted(set_a - set_b)
    only_b = sorted(set_b - set_a)
    union = set_a | set_b
    return {
        "features_a": feats_a,
        "features_b": feats_b,
        "count_a": len(feats_a),
        "count_b": len(feats_b),
        "common": common,
        "only_a": only_a,
        "only_b": only_b,
        "common_count": len(common),
        "only_a_count": len(only_a),
        "only_b_count": len(only_b),
        "overlap_pct": round(100.0 * len(common) / max(len(union), 1), 1) if union else 0.0,
    }

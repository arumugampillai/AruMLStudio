"""Per-sample feature value panels for dataset maturity."""

from __future__ import annotations

import math
from typing import Any

from .engine import FeaturePolicyEngine
from .registry import FeaturePolicyRegistry
from .warmup_calc_debug import _fmt_num, lookup_replay_values


def replay_engine_to_sample(
    trace: list[dict[str, Any]],
    registry: FeaturePolicyRegistry,
    sample_index: int,
    *,
    sampling_interval_sec: float,
    gap_max_sec: float,
) -> FeaturePolicyEngine:
    """Replay policy engine state up to *sample_index* (0-based trace index)."""
    eng = FeaturePolicyEngine(
        registry,
        sampling_interval_sec=float(sampling_interval_sec),
        gap_max_sec=float(gap_max_sec),
    )
    eng.on_session_start()
    end = min(max(sample_index, 0), len(trace) - 1)
    for i in range(end + 1):
        eng.on_sample(float(trace[i]["ts"]))
    return eng


def _category_label(meta: Any) -> str:
    cat = getattr(meta, "feature_category", None)
    if cat is None:
        return "raw"
    return str(cat.value if hasattr(cat, "value") else cat).lower()


def _coerce_value(raw: Any) -> float | None:
    if raw is None:
        return None
    try:
        val = float(raw)
    except (TypeError, ValueError):
        return None
    if math.isnan(val) or math.isinf(val):
        return None
    return val


def build_sample_feature_rows(
    feature_names: list[str],
    *,
    eng: FeaturePolicyEngine,
    registry: FeaturePolicyRegistry,
    replay_vals: dict[str, Any],
) -> list[dict[str, Any]]:
    """One row per feature: readiness + replay value (NULL when policy blocks)."""
    rows: list[dict[str, Any]] = []
    for name in feature_names:
        if not name or str(name).startswith("__roll."):
            continue
        meta = registry.get(name)
        ready = eng.is_ready(name)
        category = _category_label(meta)
        value_f: float | None = None
        if ready:
            value_f = _coerce_value(replay_vals.get(name))
        if not ready:
            status = "NULL"
            display = "NULL"
        elif value_f is not None:
            status = "VALUE"
            display = _fmt_num(value_f) or "—"
        else:
            status = "MISSING"
            display = "—"
        rows.append({
            "name": name,
            "category": category,
            "ready": ready,
            "value": value_f,
            "display": display,
            "status": status,
        })
    return rows


def summarize_feature_rows(rows: list[dict[str, Any]]) -> dict[str, int]:
    total = len(rows)
    ready = sum(1 for r in rows if r.get("ready"))
    with_value = sum(1 for r in rows if r.get("status") == "VALUE")
    null_policy = sum(1 for r in rows if r.get("status") == "NULL")
    missing = sum(1 for r in rows if r.get("status") == "MISSING")
    return {
        "total": total,
        "ready": ready,
        "with_value": with_value,
        "null_policy": null_policy,
        "missing": missing,
    }


def build_sample_feature_panel(
    *,
    trace: list[dict[str, Any]],
    sample_index: int,
    feature_names: list[str],
    registry: FeaturePolicyRegistry,
    replay_lookup: dict[int, dict[str, Any]],
    sampling_interval_sec: float,
    gap_max_sec: float,
    step_sec: int,
) -> dict[str, Any]:
    """Build feature panel payload for one maturity sample."""
    if sample_index < 0 or sample_index >= len(trace):
        return {"ok": False, "error": "Invalid sample index"}
    tr = trace[sample_index]
    eng = replay_engine_to_sample(
        trace,
        registry,
        sample_index,
        sampling_interval_sec=sampling_interval_sec,
        gap_max_sec=gap_max_sec,
    )
    replay_vals = lookup_replay_values(
        replay_lookup,
        float(tr["ts"]),
        step_sec=step_sec,
    )
    rows = build_sample_feature_rows(
        feature_names,
        eng=eng,
        registry=registry,
        replay_vals=replay_vals,
    )
    return {
        "ok": True,
        "sample": int(tr.get("samples") or sample_index + 1),
        "time": str(tr.get("time") or "—"),
        "rows": rows,
        "summary": summarize_feature_rows(rows),
    }

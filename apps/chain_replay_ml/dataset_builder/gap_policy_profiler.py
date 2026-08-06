"""Gap-policy profiler with cProfile and comparison diagnostics (FULL debug only)."""

from __future__ import annotations

import cProfile
import io
import pstats
from contextvars import ContextVar
from typing import Any

from .gap_policy_instrumentation import (
    GapPolicyProfilerStats,
    _begin_profiler_session,
    _end_profiler_session,
    gap_policy_enabled,
    gap_policy_profile_block,
    get_profiler,
    profiler_active,
    record_gap_check,
    record_gap_ema_cache_hit,
    record_gap_ema_rebuild,
    record_gap_reset,
    reset_day_context_feature_grid,
    row_gap_exceeds,
)

__all__ = [
    "GapPolicyProfilerStats",
    "aggregate_diff_by_label",
    "build_gap_pass_comparison_doc",
    "compare_gap_profiles",
    "diff_cprofile_totals",
    "gap_policy_enabled",
    "gap_policy_profile_block",
    "get_profiler",
    "profiler_active",
    "record_gap_check",
    "record_gap_ema_cache_hit",
    "record_gap_ema_rebuild",
    "record_gap_reset",
    "reset_day_context_feature_grid",
    "row_gap_exceeds",
    "short_profile_label",
    "start_gap_policy_profiler",
    "stop_gap_policy_profiler",
]

_cprofile: ContextVar[cProfile.Profile | None] = ContextVar("gap_policy_cprofile", default=None)


def _disable_tracked_cprofile() -> cProfile.Profile | None:
    prof = _cprofile.get()
    if prof is None:
        return None
    try:
        prof.disable()
    except ValueError:
        pass
    _cprofile.set(None)
    return prof


def start_gap_policy_profiler(*, gap_max_sec: float | None = None, use_cprofile: bool = True) -> GapPolicyProfilerStats:
    stop_gap_policy_profiler()
    stats = _begin_profiler_session(gap_max_sec=gap_max_sec)
    if use_cprofile:
        prof = cProfile.Profile()
        prof.enable()
        _cprofile.set(prof)
    return stats


def stop_gap_policy_profiler() -> GapPolicyProfilerStats | None:
    prof = _disable_tracked_cprofile()
    stats = _end_profiler_session()
    if prof is not None and stats is not None:
        stats.cprofile_top = _extract_cprofile_top(prof, limit=20)
        stats.cprofile_totals = _extract_cprofile_totals(prof)
        stats.cprofile_calls = _extract_cprofile_calls(prof)
    return stats


def _extract_cprofile_totals(prof: cProfile.Profile, *, use_cumtime: bool = True) -> dict[str, float]:
    ps = pstats.Stats(prof)
    out: dict[str, float] = {}
    idx = 3 if use_cumtime else 2
    for (filename, line, func), stat in ps.stats.items():
        key = f"{filename}:{line}({func})"
        out[key] = float(stat[idx])
    return out


def _extract_cprofile_calls(prof: cProfile.Profile) -> dict[str, int]:
    ps = pstats.Stats(prof)
    out: dict[str, int] = {}
    for (filename, line, func), stat in ps.stats.items():
        key = f"{filename}:{line}({func})"
        out[key] = int(stat[1])
    return out


def short_profile_label(key: str) -> str:
    """Shorten cProfile key 'path:line(name)' → 'name'."""
    import re

    text = str(key)
    match = re.search(r"\((\w+)\)\s*$", text)
    if match:
        return match.group(1)
    if ":" in text:
        return text.rsplit(":", 1)[-1]
    return text


def diff_cprofile_totals(
    off_totals: dict[str, float],
    on_totals: dict[str, float],
    *,
    limit: int = 30,
) -> list[dict[str, Any]]:
    """Sort functions by how much slower gap-ON is than gap-OFF (tottime delta)."""
    keys = set(off_totals) | set(on_totals)
    rows: list[dict[str, Any]] = []
    for key in keys:
        off_sec = float(off_totals.get(key, 0.0))
        on_sec = float(on_totals.get(key, 0.0))
        delta = on_sec - off_sec
        if abs(delta) < 1e-9 and off_sec < 1e-9 and on_sec < 1e-9:
            continue
        rows.append({
            "function": key,
            "label": short_profile_label(key),
            "gap_off_sec": round(off_sec, 6),
            "gap_on_sec": round(on_sec, 6),
            "delta_sec": round(delta, 6),
        })
    rows.sort(key=lambda r: float(r["delta_sec"]), reverse=True)
    return rows[:limit]


def aggregate_diff_by_label(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Merge cProfile rows that share the same short label (e.g. multiple call sites)."""
    merged: dict[str, dict[str, Any]] = {}
    for row in rows:
        label = str(row.get("label") or row.get("function") or "?")
        bucket = merged.setdefault(label, {
            "function": label,
            "label": label,
            "gap_off_sec": 0.0,
            "gap_on_sec": 0.0,
            "delta_sec": 0.0,
            "calls_off": 0,
            "calls_on": 0,
        })
        bucket["gap_off_sec"] = round(float(bucket["gap_off_sec"]) + float(row.get("gap_off_sec", 0)), 6)
        bucket["gap_on_sec"] = round(float(bucket["gap_on_sec"]) + float(row.get("gap_on_sec", 0)), 6)
        bucket["calls_off"] = int(bucket["calls_off"]) + int(row.get("calls_off", 0))
        bucket["calls_on"] = int(bucket["calls_on"]) + int(row.get("calls_on", 0))
        bucket["delta_sec"] = round(float(bucket["gap_on_sec"]) - float(bucket["gap_off_sec"]), 6)
    out = list(merged.values())
    out.sort(key=lambda r: float(r["delta_sec"]), reverse=True)
    return out


def _row_changed(row: dict[str, Any], *, min_delta_sec: float = 0.001) -> bool:
    if abs(float(row.get("delta_sec", 0))) >= min_delta_sec:
        return True
    if int(row.get("calls_off", 0)) != int(row.get("calls_on", 0)):
        return True
    return False


def build_gap_pass_comparison_doc(
    *,
    off_totals: dict[str, float],
    on_totals: dict[str, float],
    off_calls: dict[str, int],
    on_calls: dict[str, int],
    off_wall_sec: float,
    on_wall_sec: float,
    gap_on_max_sec: float,
    limit: int | None = None,
    only_changed: bool = True,
) -> dict[str, Any]:
    keys = set(off_totals) | set(on_totals) | set(off_calls) | set(on_calls)
    raw_rows: list[dict[str, Any]] = []
    for key in keys:
        off_sec = float(off_totals.get(key, 0.0))
        on_sec = float(on_totals.get(key, 0.0))
        calls_off = int(off_calls.get(key, 0))
        calls_on = int(on_calls.get(key, 0))
        delta = on_sec - off_sec
        if off_sec < 1e-9 and on_sec < 1e-9 and calls_off == 0 and calls_on == 0:
            continue
        raw_rows.append({
            "function": key,
            "label": short_profile_label(key),
            "gap_off_sec": round(off_sec, 6),
            "gap_on_sec": round(on_sec, 6),
            "delta_sec": round(delta, 6),
            "calls_off": calls_off,
            "calls_on": calls_on,
        })
    by_label = aggregate_diff_by_label(raw_rows)
    if only_changed:
        by_label = [r for r in by_label if _row_changed(r)]
    by_label.sort(key=lambda r: float(r["delta_sec"]), reverse=True)
    table = by_label if limit is None else by_label[:limit]
    return {
        "gap_off_wall_sec": round(float(off_wall_sec), 3),
        "gap_on_wall_sec": round(float(on_wall_sec), 3),
        "delta_wall_sec": round(float(on_wall_sec) - float(off_wall_sec), 3),
        "gap_on_max_sec": float(gap_on_max_sec),
        "by_function": table,
        "changed_function_count": len(by_label),
        "dominant_function": table[0]["label"] if table else None,
        "dominant_delta_sec": float(table[0]["delta_sec"]) if table else 0.0,
    }


def _profiled_build_day_rows(
    ctx: Any,
    *,
    gap_max_sec: float | None,
    step_sec: int,
    strike_selection: dict[str, Any],
    horizons_sec: list[int],
    enabled_groups: list[str],
    group_labels: dict[str, str],
    implemented_features: list[str],
    per_group_features: dict[str, list[str]],
    lookback_policy_doc: dict[str, Any] | None,
    trim_target_rows: bool,
    active_features: frozenset[str] | None,
) -> dict[str, Any]:
    """Run build_day_rows once under an isolated cProfile session."""
    from chain_replay_ml.dataset_builder.stages import build_day_rows
    from chain_replay_ml.feature_policy.performance_debug import PerformanceDebugConfig, PerformanceDebugLevel

    reset_day_context_feature_grid(ctx)
    start_gap_policy_profiler(gap_max_sec=gap_max_sec, use_cprofile=True)
    day_stats: dict[str, Any] = {}
    full_perf = PerformanceDebugConfig(level=PerformanceDebugLevel.FULL)
    try:
        _rows, day_stats = build_day_rows(
            ctx,
            step_sec=step_sec,
            strike_selection=strike_selection,
            horizons_sec=horizons_sec,
            enabled_groups=enabled_groups,
            group_labels=group_labels,
            implemented_features=implemented_features,
            per_group_features=per_group_features,
            lookback_policy_doc=lookback_policy_doc,
            trim_target_rows=trim_target_rows,
            active_features=active_features,
            gap_max_sec=gap_max_sec,
            performance_debug=full_perf,
        )
    finally:
        stats = stop_gap_policy_profiler()
    out = stats.to_dict() if stats else {}
    if isinstance(day_stats, dict) and day_stats.get("readiness_profiler"):
        out["readiness_profiler"] = day_stats["readiness_profiler"]
    return out


def compare_gap_profiles(
    ctx: Any,
    *,
    step_sec: int,
    strike_selection: dict[str, Any],
    horizons_sec: list[int],
    enabled_groups: list[str],
    group_labels: dict[str, str],
    implemented_features: list[str],
    per_group_features: dict[str, list[str]],
    lookback_policy_doc: dict[str, Any] | None = None,
    gap_on_max_sec: float = 20.0,
    trim_target_rows: bool = True,
    active_features: frozenset[str] | None = None,
) -> dict[str, Any]:
    """Two isolated build_day_rows passes (gap OFF vs ON) with per-run cProfile diff."""
    shared = dict(
        step_sec=step_sec,
        strike_selection=strike_selection,
        horizons_sec=horizons_sec,
        enabled_groups=enabled_groups,
        group_labels=group_labels,
        implemented_features=implemented_features,
        per_group_features=per_group_features,
        lookback_policy_doc=lookback_policy_doc,
        trim_target_rows=trim_target_rows,
        active_features=active_features,
    )
    off_stats = _profiled_build_day_rows(ctx, gap_max_sec=None, **shared)
    on_stats = _profiled_build_day_rows(ctx, gap_max_sec=float(gap_on_max_sec), **shared)
    off_doc = off_stats if isinstance(off_stats, dict) else (off_stats.to_dict() if off_stats else {})
    on_doc = on_stats if isinstance(on_stats, dict) else (on_stats.to_dict() if on_stats else {})
    doc = build_gap_pass_comparison_doc(
        off_totals=dict(off_doc.get("cprofile_totals") or {}),
        on_totals=dict(on_doc.get("cprofile_totals") or {}),
        off_calls=dict(off_doc.get("cprofile_calls") or {}),
        on_calls=dict(on_doc.get("cprofile_calls") or {}),
        off_wall_sec=float(off_doc.get("wall_sec") or 0.0),
        on_wall_sec=float(on_doc.get("wall_sec") or 0.0),
        gap_on_max_sec=float(gap_on_max_sec),
    )
    doc["gap_off_profiler"] = off_doc
    doc["gap_on_profiler"] = on_doc
    off_readiness = (off_doc.get("readiness_profiler") if isinstance(off_doc, dict) else None)
    on_readiness = (on_doc.get("readiness_profiler") if isinstance(on_doc, dict) else None)
    if off_readiness or on_readiness:
        from chain_replay_ml.feature_policy.readiness_profiler import compare_readiness_profiles

        doc["readiness_comparison"] = compare_readiness_profiles(
            off_readiness or {},
            on_readiness or {},
        )
    return doc


def _extract_cprofile_top(prof: cProfile.Profile, *, limit: int) -> list[dict[str, Any]]:
    stream = io.StringIO()
    ps = pstats.Stats(prof, stream=stream)
    ps.sort_stats(pstats.SortKey.CUMULATIVE)
    ps.print_stats(limit)
    raw = stream.getvalue()
    out: list[dict[str, Any]] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line or line.startswith("ncalls") or line.startswith("Ordered by"):
            continue
        parts = line.split()
        if len(parts) < 6:
            continue
        try:
            ncalls = parts[0]
            tottime = float(parts[1])
            cumtime = float(parts[3])
            func = parts[-1]
        except (ValueError, IndexError):
            continue
        out.append({
            "function": func,
            "ncalls": ncalls,
            "tottime_sec": round(tottime, 6),
            "cumtime_sec": round(cumtime, 6),
        })
        if len(out) >= limit:
            break
    return out

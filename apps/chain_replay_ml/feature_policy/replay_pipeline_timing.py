"""Stage-level timing for the warmup simulator replay pipeline."""

from __future__ import annotations

import time
from typing import Any


def merge_frame_timing(build_stats: dict[str, Any] | None) -> dict[str, float]:
    """Extract build_replay_day_frame substage seconds from build_stats."""
    inner = dict((build_stats or {}).get("timing_sec") or {})
    return {
        "load_day_context_sec": float(inner.get("load_day_context") or 0.0),
        "build_day_rows_sec": float(inner.get("build_day_rows") or 0.0),
        "to_dataframe_sec": float(inner.get("to_dataframe") or 0.0),
    }


def pipeline_total(stages: dict[str, Any]) -> float:
    keys = (
        "load_day_context_sec",
        "build_day_rows_sec",
        "to_dataframe_sec",
        "serialize_replay_rows_sec",
        "replay_statistics_sec",
        "build_replay_lookup_sec",
    )
    return round(sum(float(stages.get(k) or 0.0) for k in keys), 3)


def finalize_pipeline_stages(stages: dict[str, Any]) -> dict[str, Any]:
    out = dict(stages)
    out["total_sec"] = pipeline_total(out)
    return out


def build_day_rows_kwargs_from_replay(
    replay_common: dict[str, Any],
    *,
    lookback_policy_doc: dict[str, Any] | None,
    gap_max_sec: float | None = None,
    performance_debug_level: Any = None,
    performance_debug: Any = None,
) -> dict[str, Any]:
    """Shared kwargs for build_day_rows benchmarks (cold/warm, gap compare)."""
    from chain_replay_ml.dataset_builder.orchestrator import _load_feature_registry
    from chain_replay_ml.replay_feature_scoring import SCORING_INFRA_COLUMNS, merge_replay_feature_build_plan

    registry = _load_feature_registry()
    enabled_groups = list(replay_common.get("enabled_groups") or registry.get("groupOrder") or [])
    feature_names = list(replay_common.get("feature_names") or [])
    enabled_groups, implemented, _, per_group = merge_replay_feature_build_plan(
        enabled_groups,
        registry,
        feature_names or None,
    )
    group_labels = {
        gid: str((registry.get("groups") or {}).get(gid, {}).get("label") or gid)
        for gid in enabled_groups
    }
    active_features = None
    if feature_names:
        active_features = frozenset(feature_names) | frozenset(SCORING_INFRA_COLUMNS)
    from chain_replay_ml.feature_policy.performance_debug import PerformanceDebugConfig

    perf = PerformanceDebugConfig.resolve(
        performance_debug_level or replay_common.get("performance_debug_level"),
        config=performance_debug or replay_common.get("performance_debug"),
    )
    return {
        "step_sec": int(replay_common.get("step_sec") or 10),
        "strike_selection": dict(replay_common.get("strike_selection") or {}),
        "horizons_sec": list(replay_common.get("horizons_sec") or []),
        "enabled_groups": enabled_groups,
        "group_labels": group_labels,
        "implemented_features": implemented,
        "per_group_features": per_group,
        "lookback_policy_doc": lookback_policy_doc,
        "trim_target_rows": bool(replay_common.get("trim_target_rows", False)),
        "active_features": active_features,
        "gap_max_sec": gap_max_sec,
        "performance_debug": perf,
    }


def benchmark_build_day_rows_cold_warm(
    ctx: Any,
    *,
    build_kwargs: dict[str, Any],
    performance_debug: Any = None,
) -> dict[str, Any]:
    """Time build_day_rows on cold ctx (caches cleared) vs warm ctx (immediate rerun)."""
    from chain_replay_ml.dataset_builder.day_context import reset_ctx_build_caches
    from chain_replay_ml.dataset_builder.stages import build_day_rows
    from chain_replay_ml.feature_policy.performance_debug import PerformanceDebugConfig

    perf = PerformanceDebugConfig.resolve(config=performance_debug or build_kwargs.get("performance_debug"))
    if not perf.run_cache_benchmark():
        return {}

    step = int(build_kwargs.get("step_sec") or 10)
    gap = build_kwargs.get("gap_max_sec")
    run_kw = {k: v for k, v in build_kwargs.items() if k != "gap_max_sec"}

    reset_ctx_build_caches(ctx, step_sec=step)
    t0 = time.perf_counter()
    build_day_rows(ctx, gap_max_sec=gap, **run_kw)
    cold_sec = round(time.perf_counter() - t0, 3)

    t0 = time.perf_counter()
    build_day_rows(ctx, gap_max_sec=gap, **run_kw)
    warm_sec = round(time.perf_counter() - t0, 3)

    return {
        "cold_build_day_rows_sec": cold_sec,
        "warm_build_day_rows_sec": warm_sec,
        "cache_savings_sec": round(cold_sec - warm_sec, 3),
        "step_sec": step,
        "gap_max_sec": gap,
    }

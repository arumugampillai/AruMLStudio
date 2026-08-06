"""Shared production build_day_rows path — master build and warmup replay parity."""

from __future__ import annotations

import os
from typing import Any, Callable

from chain_replay_ml.feature_policy.performance_debug import PerformanceDebugConfig, PerformanceDebugLevel

_PRODUCTION_PERF = PerformanceDebugConfig(level=PerformanceDebugLevel.OFF)


def production_performance_debug() -> PerformanceDebugConfig:
    """Profiling OFF — matches warmup simulator production replay attachment."""
    return _PRODUCTION_PERF


def _default_parallel_mode() -> str:
    """Token-parallel Stage-6 on master build unless MASTER_BUILD_PARALLEL=serial."""
    raw = os.getenv("MASTER_BUILD_PARALLEL", "token").strip().lower()
    if raw in ("serial", "off", "0", "false", "no"):
        return "serial"
    return "token"


def build_production_day_rows(
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
    gap_max_sec: float | None = None,
    trim_target_rows: bool = True,
    on_group_start: Callable[[str, str], None] | None = None,
    on_group_progress: Callable[[str, int, int], None] | None = None,
    on_group_done: Callable[[str], None] | None = None,
    on_prep_progress: Callable[[str, str], None] | None = None,
    on_strike_progress: Callable[[int, int], None] | None = None,
    on_targets_progress: Callable[[int, int], None] | None = None,
    cancel_check: Callable[[], bool] | None = None,
    performance_debug: PerformanceDebugConfig | None = None,
    parallel_mode: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run build_day_rows with simulator-equivalent production settings."""
    from .stages import build_day_rows

    perf = performance_debug or _PRODUCTION_PERF
    mode = str(parallel_mode or _default_parallel_mode()).strip().lower()
    if mode not in ("token", "serial"):
        mode = "token"
    return build_day_rows(
        ctx,
        step_sec=step_sec,
        strike_selection=strike_selection,
        horizons_sec=horizons_sec,
        enabled_groups=enabled_groups,
        group_labels=group_labels,
        implemented_features=implemented_features,
        per_group_features=per_group_features,
        lookback_policy_doc=lookback_policy_doc,
        on_group_start=on_group_start,
        on_group_progress=on_group_progress,
        on_group_done=on_group_done,
        on_prep_progress=on_prep_progress,
        on_strike_progress=on_strike_progress,
        on_targets_progress=on_targets_progress,
        cancel_check=cancel_check,
        gap_max_sec=gap_max_sec,
        trim_target_rows=trim_target_rows,
        gap_profile=False,
        readiness_profile=False,
        performance_debug=perf,
        skip_readiness_compliance=True,
        parallel_mode=mode,
    )

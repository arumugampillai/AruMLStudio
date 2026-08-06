"""Build base rows, targets, and features for one trading day."""

from __future__ import annotations

import math
import os
import time
from time import perf_counter
from typing import Any, Callable

from chain_replay_ml.export_atm_pipeline import STRIKE_STEP, normalize_index_name
from chain_replay_ml.features_atm_band import (
    select_option_entries_for_timestamp,
)

from chain_replay_ml.ticks import EMA_BAR_INTERVAL_SEC

from .day_context import DayContext
from .tick_coverage import sync_feature_grid_step
from .chain_maps import precompute_chain_maps
from .extended_features import OptionFeatureState
from .feature_enrichment import build_feature_raw_for_row, attach_scoring_infra_columns
from .rolling_controllers import SpotControllers
from .feature_plugins import horizon_column_name, pick_features_from_row
from .build_profiler import get_profiler, profile_block


LOOKBACK_START_SEC = 60.0
_FEATURE_PROGRESS_INTERVAL_SEC = 1.0


def _feature_progress_due(
    ri: int,
    total: int,
    *,
    progress_every: int,
    last_emit_at: float,
    interval_sec: float = _FEATURE_PROGRESS_INTERVAL_SEC,
) -> bool:
    """Row- and time-based throttle so large builds do not look frozen at row 1."""
    if ri <= 0 or ri >= total - 1:
        return True
    if progress_every > 0 and ri % progress_every == 0:
        return True
    return (time.perf_counter() - last_emit_at) >= interval_sec


def _target_value_ok(val: Any) -> bool:
    if val is None:
        return False
    if isinstance(val, float) and math.isnan(val):
        return False
    return True


def _has_all_targets(row: dict[str, Any], horizons_sec: list[int]) -> bool:
    return all(_target_value_ok(row.get(horizon_column_name(h))) for h in horizons_sec)


def build_sample_timestamps(
    ctx: DayContext,
    *,
    step_sec: int,
    max_horizon_sec: int,
    max_stale_sec: float = 10.0,
    require_fresh_spot: bool = True,
) -> list[float]:
    from .tick_coverage import build_clipped_sample_timestamps

    timestamps, _ = build_clipped_sample_timestamps(
        ctx,
        step_sec=step_sec,
        max_horizon_sec=max_horizon_sec,
        max_stale_sec=max_stale_sec,
    )
    if require_fresh_spot:
        return timestamps

    from .tick_coverage import clipped_grid_bounds

    bounds = clipped_grid_bounds(ctx, max_horizon_sec=max_horizon_sec)
    if not bounds:
        return []
    grid_start, grid_end = bounds
    out: list[float] = []
    t = grid_start
    while t <= grid_end + 0.001:
        out.append(t)
        t += step_sec
    return out


def build_day_rows(
    ctx: DayContext,
    *,
    step_sec: int,
    strike_selection: dict[str, Any],
    horizons_sec: list[int],
    enabled_groups: list[str],
    group_labels: dict[str, str],
    implemented_features: list[str],
    per_group_features: dict[str, list[str]],
    lookback_policy_doc: dict[str, Any] | None = None,
    on_group_progress: Callable[[str, int, int], None] | None = None,
    gap_max_sec: float | None = None,
    on_group_start: Callable[[str, str], None] | None = None,
    on_group_done: Callable[[str], None] | None = None,
    on_strike_progress: Callable[[int, int], None] | None = None,
    on_targets_progress: Callable[[int, int], None] | None = None,
    on_prep_progress: Callable[[str, str], None] | None = None,
    cancel_check: Callable[[], bool] | None = None,
    parallel_mode: str = "serial",
    active_features: frozenset[str] | None = None,
    max_stale_sec: float = 10.0,
    only_timestamp: float | None = None,
    timestamp_eps: float = 0.05,
    include_tokens: frozenset[str] | None = None,
    token_only: bool = False,
    enrich_tokens_only: frozenset[str] | None = None,
    trim_target_rows: bool = True,
    gap_profile: bool = False,
    readiness_profile: bool = False,
    performance_debug_level: Any = None,
    performance_debug: Any = None,
    skip_readiness_compliance: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Stages 3–6 for a single day: sample → strikes → targets → features."""
    from chain_replay_ml.feature_policy.performance_debug import PerformanceDebugConfig

    from .dataset_selection_engine import DatasetSelectionSpec
    from .gap_policy_instrumentation import gap_policy_enabled

    strike_selection = DatasetSelectionSpec.from_strike_selection(strike_selection).to_strike_selection_dict()
    from .tick_coverage import build_clipped_sample_timestamps, clipped_grid_bounds, list_clipped_grid_timestamps

    sync_feature_grid_step(ctx, step_sec, gap_max_sec=gap_max_sec)

    perf = PerformanceDebugConfig.resolve(
        performance_debug_level,
        gap_profile=gap_profile,
        readiness_profile=readiness_profile,
        config=performance_debug,
    )
    gap_profiler_active = perf.collect_gap_profile() and gap_policy_enabled(gap_max_sec)
    if gap_profiler_active:
        from .gap_policy_profiler import start_gap_policy_profiler

        start_gap_policy_profiler(
            gap_max_sec=gap_max_sec,
            use_cprofile=perf.collect_cprofile(),
        )

    max_horizon = max(horizons_sec) if horizons_sec else 0
    with profile_block("stage.sampling_grid"):
        all_clipped_ts = list_clipped_grid_timestamps(
            ctx, step_sec=step_sec, max_horizon_sec=max_horizon,
        )
        timestamps, coverage = build_clipped_sample_timestamps(
            ctx,
            step_sec=step_sec,
            max_horizon_sec=max_horizon,
            max_stale_sec=max_stale_sec,
        )
    rejected_stale_spot = 0
    rejected_stale_option = 0

    if not all_clipped_ts:
        coverage["samples_written"] = 0
        coverage["rejected_samples"] = 0
        coverage["candidate_samples"] = 0
        if gap_profiler_active:
            from .gap_policy_profiler import stop_gap_policy_profiler

            stop_gap_policy_profiler()
        return [], {"target_trimmed_rows": 0, "coverage": coverage}

    grid_target_ts: float | None = None
    if only_timestamp is not None:
        max_hor = max(horizons_sec) if horizons_sec else 0
        bounds = clipped_grid_bounds(ctx, max_horizon_sec=max_hor)
        step_i = max(int(step_sec), 1)
        grid_start = float(bounds[0]) if bounds else None
        from .tick_coverage import resolve_inference_sample_ts

        grid_target_ts = resolve_inference_sample_ts(
            float(only_timestamp),
            horizons_sec=horizons_sec,
            training_step_sec=step_i,
            grid_start=grid_start,
        )
        timestamp_eps = max(
            float(timestamp_eps),
            0.51 if not horizons_sec else step_i * 0.51,
        )

    index_key = normalize_index_name(ctx.source.market)
    step = STRIKE_STEP.get(index_key, 50)
    rows: list[dict[str, Any]] = []
    strike_mode = str(strike_selection.get("mode") or "atm_band").lower()
    delta_stats = None
    if strike_mode == "delta_range":
        from .delta_range_stats import DeltaRangeBuildStats, collect_delta_candidates_for_timestamp

        delta_stats = DeltaRangeBuildStats()

    # Stage 3+4: clipped grid → count rejections + build rows
    if grid_target_ts is not None:
        ts_iter = [grid_target_ts]
    else:
        ts_iter = all_clipped_ts
    n_ts = len(ts_iter)
    if on_prep_progress:
        on_prep_progress("build_rows", f"Building sample rows ({n_ts} grid point{'s' if n_ts != 1 else ''})…")
    _strike_t0 = perf_counter() if get_profiler() else None
    for ts_idx, ts in enumerate(ts_iter):
        if grid_target_ts is None and only_timestamp is not None and abs(float(ts) - float(only_timestamp)) > timestamp_eps:
            continue
        if cancel_check and cancel_check():
            break
        spot = ctx.index_tl.ltp_rupees_at(ts)
        if spot is None or spot <= 0:
            continue
        spot_fresh = ctx.index_tl.is_fresh_at(ts, max_stale_sec)

        if token_only and include_tokens:
            if not spot_fresh:
                rejected_stale_spot += len(include_tokens)
            else:
                from chain_replay_ml.features_atm_band import find_atm_strike

                atm = find_atm_strike(spot, step) if spot else 0
                for (strike_r, opt_type), (tok, symbol, opt_tl) in ctx.strike_mapping.items():
                    tok_s = str(tok)
                    if tok_s not in include_tokens:
                        continue
                    if not opt_tl.is_fresh_at(ts, max_stale_sec):
                        rejected_stale_option += 1
                        continue
                    rows.append({
                        "trading_day": ctx.source.trading_day,
                        "market": ctx.source.market,
                        "expiry": ctx.expiry_norm,
                        "timestamp": ts,
                        "strike": strike_r,
                        "option_type": opt_type,
                        "token": tok,
                        "symbol": symbol,
                        "_opt_tl": opt_tl,
                        "_spot": spot,
                        "_atm": atm,
                    })
            if on_strike_progress and (ts_idx % 5 == 0 or ts_idx == n_ts - 1):
                on_strike_progress(len(rows), 0)
            continue

        if strike_mode == "delta_range" and delta_stats is not None:
            option_entries = collect_delta_candidates_for_timestamp(
                ts=ts,
                ctx=ctx,
                strike_selection=strike_selection,
                stats=delta_stats,
            )
        else:
            option_entries = select_option_entries_for_timestamp(
                ts=ts,
                spot=spot,
                strike_step=step,
                index_timeline=ctx.index_tl,
                strike_mapping=ctx.strike_mapping,
                expiry_ts=ctx.expiry_ts,
                strike_selection=strike_selection,
            )
        n_opts = len(option_entries)
        if not spot_fresh:
            rejected_stale_spot += n_opts
            continue
        for strike_r, opt_type, tok, symbol, opt_tl, atm in option_entries:
            if not opt_tl.is_fresh_at(ts, max_stale_sec):
                rejected_stale_option += 1
                continue
            rows.append({
                "trading_day": ctx.source.trading_day,
                "market": ctx.source.market,
                "expiry": ctx.expiry_norm,
                "timestamp": ts,
                "strike": strike_r,
                "option_type": opt_type,
                "token": tok,
                "symbol": symbol,
                "_opt_tl": opt_tl,
                "_spot": spot,
                "_atm": atm,
            })
        if include_tokens:
            present = {str(r["token"]) for r in rows if r.get("timestamp") == ts}
            for (strike_r, opt_type), (tok, symbol, opt_tl) in ctx.strike_mapping.items():
                tok_s = str(tok)
                if tok_s not in include_tokens or tok_s in present:
                    continue
                if not opt_tl.is_fresh_at(ts, max_stale_sec):
                    rejected_stale_option += 1
                    continue
                rows.append({
                    "trading_day": ctx.source.trading_day,
                    "market": ctx.source.market,
                    "expiry": ctx.expiry_norm,
                    "timestamp": ts,
                    "strike": strike_r,
                    "option_type": opt_type,
                    "token": tok,
                    "symbol": symbol,
                    "_opt_tl": opt_tl,
                    "_spot": spot,
                    "_atm": atm,
                })
                present.add(tok_s)
        if on_strike_progress and (ts_idx % 5 == 0 or ts_idx == n_ts - 1):
            on_strike_progress(len(rows), 0)
    if _strike_t0 is not None:
        prof = get_profiler()
        if prof is not None:
            prof.record("stage.strike_selection", perf_counter() - _strike_t0, rows=len(rows))

    if not rows:
        coverage["samples_written"] = 0
        coverage["rejected_stale_spot"] = rejected_stale_spot
        coverage["rejected_stale_option"] = rejected_stale_option
        coverage["rejected_missing_targets"] = 0
        coverage["rejected_samples"] = rejected_stale_spot + rejected_stale_option
        coverage["candidate_samples"] = coverage["rejected_samples"]
        if gap_profiler_active:
            from .gap_policy_profiler import stop_gap_policy_profiler

            stop_gap_policy_profiler()
        return [], {"target_trimmed_rows": 0, "coverage": coverage}

    unique_ts = sorted({r["timestamp"] for r in rows})
    active_feat_set = active_features or frozenset(implemented_features or [])
    if on_prep_progress:
        on_prep_progress(
            "chain_maps",
            f"Precomputing chain maps for {len(unique_ts)} timestamp{'s' if len(unique_ts) != 1 else ''}…",
        )
    from .sharp_momentum import ensure_spot_momentum_cache, needs_sharp_momentum

    if needs_sharp_momentum(active_feat_set):
        with profile_block("stage.prep.sharp_momentum"):
            ensure_spot_momentum_cache(
                ctx,
                through_ts=float(unique_ts[-1]),
                step_sec=step_sec,
                max_horizon_sec=max_horizon,
            )
    with profile_block("stage.prep.chain_maps"):
        from .chain_delta_volume_flow import needs_delta_w_volume_flow
        from .chain_gex import needs_chain_gex
        from .chain_iv_skew import needs_atm_iv, needs_chain_iv_skew
        from .chain_oi_delta_bands import needs_oi_abs_delta_bands

        chain_maps = precompute_chain_maps(
            index_tl=ctx.index_tl,
            strike_mapping=ctx.strike_mapping,
            timestamps=unique_ts,
            strike_step=step,
            expiry_ts=float(ctx.expiry_ts) if getattr(ctx, "expiry_ts", None) else None,
            include_iv_skew=needs_chain_iv_skew(active_feat_set),
            include_atm_iv=needs_atm_iv(active_feat_set),
            include_delta_flow=needs_delta_w_volume_flow(active_feat_set),
            include_gex=needs_chain_gex(active_feat_set),
            include_oi_delta_bands=needs_oi_abs_delta_bands(active_feat_set),
        )
    if on_prep_progress:
        on_prep_progress("chain_maps", "Chain maps ready — computing features…")
    opt_states: dict[str, OptionFeatureState] = {}
    spot_controllers = SpotControllers()

    from .historic_spot_ema_context import (
        ensure_historic_spot_ema_book,
        needs_historic_spot_ema,
    )

    if needs_historic_spot_ema(active_feat_set):
        with profile_block("stage.prep.historic_spot_ema_book"):
            ensure_historic_spot_ema_book(ctx, active_features=active_feat_set)

    # Stage 5: prediction targets (OLE Fixed Horizon — identical future_ltp_* semantics)
    from chain_replay_ml.outcome_label_engine.fixed_horizon import (
        compute_fixed_horizon_targets,
    )

    n_rows = len(rows)
    with profile_block("stage.prediction_targets", rows=n_rows):
        for ri, row in enumerate(rows):
            if cancel_check and cancel_check():
                break
            opt_tl = row["_opt_tl"]
            targets = compute_fixed_horizon_targets(
                ts=float(row["timestamp"]),
                opt_tl=opt_tl,
                horizons_sec=horizons_sec,
                max_stale_sec=max_stale_sec,
            )
            for col, val in targets.items():
                row[col] = val
            if on_targets_progress and (ri % 500 == 0 or ri == n_rows - 1):
                on_targets_progress(ri + 1, n_rows)

    before_trim = len(rows)
    with profile_block("stage.target_trim", rows=before_trim):
        if horizons_sec and trim_target_rows:
            rows = [r for r in rows if _has_all_targets(r, horizons_sec)]
    target_trimmed_rows = before_trim - len(rows)
    rejected_missing_targets = target_trimmed_rows
    rejected_total = rejected_stale_spot + rejected_stale_option + rejected_missing_targets
    coverage["rejected_stale_spot"] = rejected_stale_spot
    coverage["rejected_stale_option"] = rejected_stale_option
    coverage["rejected_missing_targets"] = rejected_missing_targets
    coverage["rejected_samples"] = rejected_total
    coverage["candidate_samples"] = rejected_total + len(rows)

    if not rows:
        coverage["samples_written"] = 0
        if gap_profiler_active:
            from .gap_policy_profiler import stop_gap_policy_profiler

            stop_gap_policy_profiler()
        return [], {"target_trimmed_rows": target_trimmed_rows, "coverage": coverage}

    # Stage 6: features — one raw compute per row, then pick all groups (or token-parallel).
    group_ids = [g for g in enabled_groups if g in per_group_features]
    # Token parallel is allowed with progress/cancel: overall feature progress, not per-group walks.
    use_parallel = (
        parallel_mode == "token"
        and enrich_tokens_only is None
        and len(group_ids) > 0
    )
    if use_parallel:
        from .stages_parallel import enrich_rows_features_parallel_token

        if on_group_start:
            on_group_start("features", "Features")
        with profile_block("stage.feature_generation", rows=len(rows)):
            enrich_rows_features_parallel_token(
                rows,
                ctx=ctx,
                chain_maps=chain_maps,
                strike_step=step,
                enabled_groups=enabled_groups,
                per_group_features=per_group_features,
                lookback_policy_doc=lookback_policy_doc,
                active_features=active_feat_set,
                gap_max_sec=gap_max_sec,
                cancel_check=cancel_check,
                on_progress=(
                    (lambda cur, tot: on_group_progress("Features", cur, tot))
                    if on_group_progress
                    else None
                ),
            )
        if on_group_progress:
            on_group_progress("Features", len(rows), len(rows))
        if on_group_done:
            on_group_done("features")
    else:
        total = len(rows)
        progress_every = max(1, min(100 if total > 50_000 else 500, total // 20 or 1))
        last_group_progress_at = 0.0
        with profile_block("stage.feature_generation", rows=total):
            if on_group_start:
                on_group_start("features", "Features")
            # One traversal: build raw once per row, pick all groups, write.
            with profile_block("group.all_features", rows=total):
                for ri, row in enumerate(rows):
                    if cancel_check and cancel_check():
                        break
                    if (
                        enrich_tokens_only is not None
                        and str(row.get("token")) not in enrich_tokens_only
                    ):
                        continue
                    if on_group_progress and _feature_progress_due(
                        ri,
                        total,
                        progress_every=progress_every,
                        last_emit_at=last_group_progress_at,
                    ):
                        on_group_progress("Features", ri + 1, total)
                        last_group_progress_at = time.perf_counter()
                    if "_feature_raw" not in row:
                        token = str(row["token"])
                        if token not in opt_states:
                            from .gap_policy_instrumentation import log_controller_reset

                            log_controller_reset(
                                token=token,
                                feature="OptionFeatureState",
                                current_ts=float(row["timestamp"]),
                                reason="new_state",
                            )
                            opt_states[token] = OptionFeatureState()
                        row["_feature_raw"] = build_feature_raw_for_row(
                            row,
                            ctx=ctx,
                            chain_maps=chain_maps,
                            strike_step=step,
                            lookback_policy_doc=lookback_policy_doc,
                            opt_state=opt_states[token],
                            gap_max_sec=gap_max_sec,
                            active_features=active_feat_set,
                            spot_controllers=spot_controllers,
                        )
                    raw = row["_feature_raw"]
                    picked: dict[str, Any] = {}
                    with profile_block("function.pick_features_from_row", rows=1):
                        for gid in group_ids:
                            picked.update(
                                pick_features_from_row(
                                    raw, per_group_features[gid], [gid]
                                )
                            )
                    row.update(picked)
            if on_group_progress:
                on_group_progress("Features", total, total)
            if on_group_done:
                on_group_done("features")

    gap_profiler_stats = None
    if gap_profiler_active:
        from .gap_policy_profiler import stop_gap_policy_profiler

        gap_profiler_stats = stop_gap_policy_profiler()

    # Scoring infra (delta band filter) — always from raw, not only when greeks group selected
    for row in rows:
        if enrich_tokens_only is not None and str(row.get("token")) not in enrich_tokens_only:
            continue
        raw = row.get("_feature_raw")
        if isinstance(raw, dict):
            attach_scoring_infra_columns(row, raw)

    # Stage 6b: enforce feature readiness — NOT READY → NULL (policy engine)
    readiness_stats: dict[str, Any] = {}
    if implemented_features:
        from chain_replay_ml.feature_policy.build_readiness import (
            enforce_readiness_on_rows,
            validate_readiness_compliance,
        )
        from chain_replay_ml.feature_policy.types import DEFAULT_GAP_MAX_SEC

        gap_sec = float(gap_max_sec if gap_max_sec is not None else DEFAULT_GAP_MAX_SEC)
        profile_readiness = perf.collect_readiness_profile()
        if profile_readiness:
            from chain_replay_ml.feature_policy.readiness_profiler import (
                start_readiness_profiler,
                stop_readiness_profiler,
            )

            start_readiness_profiler(
                gap_max_sec=gap_sec,
                feature_count=len(implemented_features),
                row_count=len(rows),
            )
            t_enforce = time.perf_counter()
        from .rolling_controllers import CONTROLLER_OWNED_READINESS_FEATURES

        readiness_feature_names = [
            f for f in implemented_features if f not in CONTROLLER_OWNED_READINESS_FEATURES
        ]
        with profile_block("stage.readiness", rows=len(rows)):
            readiness_stats = enforce_readiness_on_rows(
                rows,
                feature_names=readiness_feature_names,
                sampling_interval_sec=float(step_sec),
                gap_max_sec=gap_sec,
                readiness_profile=profile_readiness,
            )
        from .rolling_controllers import guard_controller_derived_rv_features

        for row in rows:
            guard_controller_derived_rv_features(row)
            raw = row.get("_feature_raw")
            if isinstance(raw, dict):
                guard_controller_derived_rv_features(raw)
        if profile_readiness:
            from chain_replay_ml.feature_policy.readiness_profiler import (
                get_profiler as get_readiness_profiler,
            )

            prof = get_readiness_profiler()
            if prof is not None:
                prof.enforce_wall_sec = round(time.perf_counter() - t_enforce, 6)
            t_validate = time.perf_counter()
            readiness_stats["readiness_compliance"] = validate_readiness_compliance(
                rows,
                feature_names=implemented_features,
                sampling_interval_sec=float(step_sec),
                gap_max_sec=gap_sec,
                readiness_profile=True,
            )
            prof = get_readiness_profiler()
            if prof is not None:
                prof.validate_wall_sec = round(time.perf_counter() - t_validate, 6)
            stopped = stop_readiness_profiler()
            if stopped is not None:
                readiness_stats["readiness_profiler"] = stopped.to_dict()
        elif not skip_readiness_compliance:
            readiness_stats["readiness_compliance"] = validate_readiness_compliance(
                rows,
                feature_names=implemented_features,
                sampling_interval_sec=float(step_sec),
                gap_max_sec=gap_sec,
                readiness_profile=False,
            )

    # cleanup internal keys
    for row in rows:
        row.pop("_opt_tl", None)
        row.pop("_spot", None)
        row.pop("_atm", None)
        row.pop("_feature_raw", None)

    coverage["samples_written"] = len(rows)
    day_stats: dict[str, Any] = {
        "target_trimmed_rows": target_trimmed_rows,
        "parallel_mode": "token" if use_parallel else "serial",
        "coverage": coverage,
    }
    if delta_stats is not None:
        day_stats["delta_range_stats"] = delta_stats.to_dict(strike_selection=strike_selection)
    if readiness_stats:
        day_stats["feature_readiness"] = readiness_stats
        if readiness_stats.get("readiness_profiler"):
            day_stats["readiness_profiler"] = readiness_stats["readiness_profiler"]
    if gap_profiler_stats is not None:
        day_stats["gap_policy_profiler"] = gap_profiler_stats.to_dict()
    return rows, day_stats

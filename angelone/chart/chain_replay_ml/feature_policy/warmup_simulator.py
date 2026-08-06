"""Warm-up simulator — drive FeaturePolicyEngine on real tick-grid timestamps."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable
from zoneinfo import ZoneInfo

from chain_replay_ml.dataset_builder.day_context import SourceSpec, load_day_context
from chain_replay_ml.dataset_builder.tick_coverage import (
    LOOKBACK_START_SEC,
    build_clipped_sample_timestamps,
    clipped_grid_bounds,
    compute_spot_coverage,
    list_clipped_grid_timestamps,
)
from chain_replay_ml.export_atm_pipeline import replay_db_path

from . import DEFAULT_GAP_MAX_SEC
from .engine import FeaturePolicyEngine
from .registry import load_feature_policy_registry

IST = ZoneInfo("Asia/Kolkata")
ProgressCallback = Callable[[str], None]


def fmt_ist_time(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=IST).strftime("%H:%M:%S")


def _list_tick_trading_days(chart_dir: str) -> list[str]:
    """Trading days with a non-empty tick database (all configured tick dirs)."""
    import os

    from tick_data_paths import tick_search_dirs

    days: list[str] = []
    seen: set[str] = set()
    for folder in tick_search_dirs(chart_dir):
        if not os.path.isdir(folder):
            continue
        try:
            names = os.listdir(folder)
        except OSError:
            continue
        for name in names:
            if not name.startswith("angel_market_") or not name.endswith(".db"):
                continue
            day = name.replace("angel_market_", "").replace(".db", "")
            if day in seen:
                continue
            path = os.path.join(folder, name)
            try:
                if os.path.getsize(path) > 0:
                    seen.add(day)
                    days.append(day)
            except OSError:
                continue
    return days


def _list_master_trading_days(
    chart_dir: str,
    *,
    sampling_interval_sec: float | int,
    market: str | None = None,
) -> list[str]:
    """Completed trading days present in the master SQLite for this interval."""
    import os

    from chain_replay_ml.dataset_builder.master_naming import (
        resolve_master_db_path,
        resolve_master_datasets_dir,
    )
    from chain_replay_ml.dataset_builder.master_store import MasterStore

    interval = max(1, int(round(float(sampling_interval_sec))))
    data_dir = os.path.join(str(chart_dir or "").strip() or ".", "data")
    candidates: list[str] = []
    mkt = str(market or "").strip().upper() or None
    if mkt:
        candidates.append(
            resolve_master_db_path(
                data_dir, market=mkt, sampling_interval_sec=interval
            )
        )
    else:
        datasets = resolve_master_datasets_dir(data_dir)
        suffix = f"_{interval}s.db"
        if os.path.isdir(datasets):
            try:
                for name in sorted(os.listdir(datasets)):
                    if name.startswith("master_dataset_") and name.endswith(suffix):
                        candidates.append(os.path.join(datasets, name))
            except OSError:
                pass
        for fallback in ("NIFTY", "SENSEX", "BANKNIFTY"):
            path = resolve_master_db_path(
                data_dir, market=fallback, sampling_interval_sec=interval
            )
            if path not in candidates:
                candidates.append(path)

    days: list[str] = []
    seen: set[str] = set()
    for path in candidates:
        if not path or not os.path.isfile(path):
            continue
        try:
            if os.path.getsize(path) <= 0:
                continue
        except OSError:
            continue
        store = MasterStore(path)
        try:
            store.open()
            store_days: list[str] = []
            for row in store.read_master_days():
                day = str(row.get("trading_day") or "").strip()
                if not day:
                    continue
                if int(row.get("row_count") or 0) <= 0:
                    continue
                store_days.append(day)
            if not store_days:
                for day in store.distinct_trading_days():
                    d = str(day or "").strip()
                    if d:
                        store_days.append(d)
            for day in store_days:
                if day not in seen:
                    seen.add(day)
                    days.append(day)
        except Exception:
            continue
        finally:
            try:
                store.close()
            except Exception:
                pass
    return days


def list_trading_days(
    chart_dir: str,
    *,
    sampling_interval_sec: float | int | None = None,
    market: str | None = None,
) -> list[str]:
    """Trading days for Warm-up Simulator.

    Includes:
    - days with a non-empty tick DB (via ``tick_search_dirs``), and
    - completed days already present in the master dataset for the active
      sampling interval (e.g. 3s), so the combo matches Create Dataset / Master.
    """
    days: set[str] = set(_list_tick_trading_days(chart_dir))
    if sampling_interval_sec is not None:
        days.update(
            _list_master_trading_days(
                chart_dir,
                sampling_interval_sec=sampling_interval_sec,
                market=market,
            )
        )
    return sorted(days, reverse=True)


def resolve_chain_source(chart_dir: str, trading_day: str) -> SourceSpec | None:
    """Pick first NIFTY chain source for a trading day."""
    import os

    from storage.market_db_inventory_cache import load_cache

    data_dir = os.path.join(chart_dir, "data")
    cache = load_cache(data_dir) or {}
    databases = cache.get("databases") or {}
    entry = databases.get(trading_day) or {}
    spot_keys: set[tuple[str, str]] = set()
    for day, ent in databases.items():
        for row in (ent or {}).get("rows") or []:
            if str(row.get("kind") or "").lower() == "spot":
                market = str(row.get("index") or "").strip().upper()
                if market in ("NIFTY", "SENSEX", "BANKNIFTY"):
                    spot_keys.add((day, market))

    candidates: list[dict[str, Any]] = []
    for row in entry.get("rows") or []:
        if str(row.get("kind") or "").lower() != "chain":
            continue
        market = str(row.get("index") or "").strip().upper()
        if market not in ("NIFTY", "SENSEX", "BANKNIFTY"):
            continue
        expiry_raw = str(row.get("expiry") or "").strip()
        if not expiry_raw or expiry_raw.upper() == "SPOT":
            continue
        candidates.append({
            "market": market,
            "expiry": expiry_raw,
            "spot_available": (trading_day, market) in spot_keys,
            "source_id": f"{trading_day}|{market}|{expiry_raw}",
        })
    if not candidates:
        return None
    candidates.sort(key=lambda r: (0 if r["market"] == "NIFTY" else 1, r["expiry"]))
    row = candidates[0]
    return SourceSpec(
        source_id=row["source_id"],
        trading_day=trading_day,
        market=row["market"],
        expiry=row["expiry"],
    )


def _anchor_label(anchor: str | None) -> str | None:
    if not anchor:
        return None
    parts = str(anchor).split(".")
    if len(parts) >= 3 and parts[2].startswith("ema"):
        series = parts[1].upper() if len(parts) > 1 else ""
        ema = f"EMA{parts[2].replace('ema', '')}"
        return f"{series} {ema}" if series else ema
    return anchor.replace("__roll.", "").replace(".", " ").upper()


def _controller_info(meta: Any) -> tuple[str | None, str | None, int]:
    anchor = getattr(meta, "policy_anchor", None)
    eff = int(getattr(meta, "effective_warmup_samples", 0) or getattr(meta, "intrinsic_warmup_samples", 0) or 0)
    return _anchor_label(anchor), (str(anchor) if anchor else None), eff


def _dep_label(dep_id: str) -> str:
    if dep_id.startswith("__roll."):
        return _anchor_label(dep_id) or dep_id
    if dep_id.lower() in ("ltp", "spot"):
        return dep_id.upper()
    return dep_id.replace("_", " ").title()


def expand_timestamps_with_gaps(
    timestamps: list[float],
    gap_injections: list[tuple[int, float]],
) -> list[tuple[float, bool]]:
    """Return (ts, is_gap_event) pairs. Shifts timestamps after each injected gap."""
    if not gap_injections or not timestamps:
        return [(t, False) for t in timestamps]
    out = list(timestamps)
    for after_idx, gap_sec in sorted(gap_injections, key=lambda x: x[0], reverse=True):
        if after_idx < 0 or after_idx >= len(out) - 1:
            continue
        for j in range(after_idx + 1, len(out)):
            out[j] += float(gap_sec)
    pairs: list[tuple[float, bool]] = []
    prev = out[0]
    pairs.append((out[0], False))
    gap_set = {after_idx for after_idx, _ in gap_injections}
    for i in range(1, len(out)):
        is_gap = (i - 1) in gap_set and (out[i] - prev) > DEFAULT_GAP_MAX_SEC
        pairs.append((out[i], is_gap))
        prev = out[i]
    return pairs


@dataclass
class WarmupSimulationResult:
    ok: bool = True
    error: str | None = None
    feature_name: str = ""
    trading_day: str = ""
    duration_minutes: int = 0
    sampling_interval_sec: float = 10.0
    controller_label: str | None = None
    controller_key: str | None = None
    effective_warmup: int = 0
    samples_processed: int = 0
    ready_at_ts: float | None = None
    ready_at_sample: int | None = None
    gap_resets: int = 0
    gap_events: list[dict[str, Any]] = field(default_factory=list)
    timeline: list[dict[str, Any]] = field(default_factory=list)
    chart_points: list[tuple[str, float]] = field(default_factory=list)
    dependency_status: list[dict[str, Any]] = field(default_factory=list)
    events: list[str] = field(default_factory=list)
    policy_pass: bool = False
    policy_reason: str = ""
    progress_steps: list[str] = field(default_factory=list)
    coverage_info: dict[str, Any] = field(default_factory=dict)
    full_trace: list[dict[str, Any]] = field(default_factory=list)
    transition: dict[str, Any] = field(default_factory=dict)
    dependency_timeline: list[dict[str, Any]] = field(default_factory=list)
    output_samples: list[dict[str, Any]] = field(default_factory=list)
    dataset_impact: dict[str, Any] = field(default_factory=dict)
    dependency_labels: list[str] = field(default_factory=list)
    calc_debug: dict[str, Any] = field(default_factory=dict)
    maturity_timeline: list[dict[str, Any]] = field(default_factory=list)
    maturity_summary: dict[str, Any] = field(default_factory=dict)
    dataset_feature_total: int = 0
    maturity_feature_names: list[str] = field(default_factory=list)
    maturity_replay_lookup: dict[int, dict[str, Any]] = field(default_factory=dict)
    maturity_replay_error: str | None = None
    all_features_lookup: dict[int, dict[str, Any]] = field(default_factory=dict)
    all_features_rows: list[dict[str, Any]] = field(default_factory=list)
    replay_token: str | None = None
    strike_selection: dict[str, Any] = field(default_factory=dict)
    horizons_sec: list[int] = field(default_factory=list)
    target_columns: list[str] = field(default_factory=list)
    gap_max_sec: float = DEFAULT_GAP_MAX_SEC
    timing: dict[str, Any] = field(default_factory=dict)


def _progress_log(
    result: WarmupSimulationResult,
    msg: str,
    on_progress: ProgressCallback | None,
) -> None:
    result.progress_steps.append(msg)
    if on_progress:
        on_progress(msg)


def _stamp_run_completed(result: WarmupSimulationResult) -> None:
    result.timing["run_completed_at"] = datetime.now(tz=IST).strftime("%Y-%m-%d %H:%M:%S")


def build_simulator_timestamps(
    ctx: Any,
    *,
    step_sec: int,
    duration_minutes: int,
    session_close_ts: float,
    result: WarmupSimulationResult,
    on_progress: ProgressCallback | None,
    max_horizon_sec: int = 0,
) -> list[float]:
    """Build sample grid for simulator; fall back if freshness filter yields no points.

    Uses the same clipped-grid rules as ``build_day_rows`` (including target horizon trim).
    """
    max_hor = max(int(max_horizon_sec or 0), 0)
    open_ts = float(ctx.open_ts)
    duration_sec = max(duration_minutes, 1) * 60.0
    min_window = LOOKBACK_START_SEC + step_sec * 2

    spot_cov = compute_spot_coverage(ctx)
    result.coverage_info.update(spot_cov)
    first_tick_ts = spot_cov.get("first_tick_ts")
    last_tick_ts = spot_cov.get("last_tick_ts")
    _progress_log(
        result,
        f"Spot coverage: {spot_cov.get('first_tick') or '—'} → {spot_cov.get('last_tick') or '—'} "
        f"({spot_cov.get('coverage_pct', 0)}%)",
        on_progress,
    )

    if not first_tick_ts:
        _progress_log(result, "No spot ticks in database for this day.", on_progress)
        return []

    # Primary window: from first tick when data starts after open; else from session open
    if float(first_tick_ts) > open_ts + 0.001:
        partial_start = float(first_tick_ts)
        partial_close = min(session_close_ts, partial_start + duration_sec)
        window_mode = "from_first_spot_tick"
        gap_min = (partial_start - open_ts) / 60.0
        _progress_log(
            result,
            f"Spot ticks begin at {spot_cov.get('first_tick')} "
            f"({gap_min:.0f} min after open at {fmt_ist_time(open_ts)}) — "
            "warming from first tick, not exchange open.",
            on_progress,
        )
        _progress_log(
            result,
            f"Window (first tick): {fmt_ist_time(partial_start)} → "
            f"{fmt_ist_time(partial_close)} ({duration_minutes} min)",
            on_progress,
        )
    else:
        partial_close = min(session_close_ts, open_ts + duration_sec)
        window_mode = "from_session_open"
        _progress_log(
            result,
            f"Window (session open): {fmt_ist_time(open_ts)} → {fmt_ist_time(partial_close)} "
            f"({duration_minutes} min)",
            on_progress,
        )

    ctx.close_ts = partial_close
    bounds = clipped_grid_bounds(ctx, max_horizon_sec=max_hor)

    if not bounds and float(first_tick_ts) > partial_close:
        # Spot data starts after the open window — shift to first N min of actual ticks
        shifted_close = min(
            float(last_tick_ts or session_close_ts),
            float(first_tick_ts) + duration_sec,
        )
        ctx.close_ts = shifted_close
        window_mode = "from_first_spot_tick"
        gap_min = (float(first_tick_ts) - open_ts) / 60.0
        _progress_log(
            result,
            f"Spot ticks begin at {spot_cov.get('first_tick')} "
            f"({gap_min:.0f} min after open at {fmt_ist_time(open_ts)}) — "
            "not at session open.",
            on_progress,
        )
        _progress_log(
            result,
            f"Shifting window → {fmt_ist_time(float(first_tick_ts))} → "
            f"{fmt_ist_time(shifted_close)} ({duration_minutes} min from first tick)",
            on_progress,
        )
        bounds = clipped_grid_bounds(ctx, max_horizon_sec=max_hor)

    if not bounds:
        _progress_log(
            result,
            "No grid overlap: spot tick span does not intersect the simulation window.",
            on_progress,
        )
        _progress_log(
            result,
            f"Tip: use ≥{max(int((float(first_tick_ts) - open_ts) / 60) + duration_minutes + 1, 10)} min "
            "duration or pick a day where spot starts near open.",
            on_progress,
        )
        return []

    result.coverage_info["window_mode"] = window_mode
    result.coverage_info["grid_start"] = fmt_ist_time(bounds[0])
    result.coverage_info["grid_end"] = fmt_ist_time(bounds[1])
    window_sec = bounds[1] - bounds[0]
    if window_sec < min_window:
        _progress_log(
            result,
            f"Warning: short grid span ({window_sec:.0f}s) — need ≥{min_window:.0f}s for samples",
            on_progress,
        )
    _progress_log(
        result,
        f"Clipped grid: {result.coverage_info['grid_start']} → {result.coverage_info['grid_end']}",
        on_progress,
    )

    timestamps, fresh_cov = build_clipped_sample_timestamps(
        ctx, step_sec=step_sec, max_horizon_sec=max_hor,
    )
    result.coverage_info.update(fresh_cov)
    grid_mode = "fresh grid (10s stale filter)"

    if not timestamps:
        timestamps = list_clipped_grid_timestamps(ctx, step_sec=step_sec, max_horizon_sec=max_hor)
        grid_mode = "clipped grid (freshness filter skipped)"
        _progress_log(
            result,
            f"No fresh grid points — using {len(timestamps)} clipped points instead",
            on_progress,
        )

    result.coverage_info["grid_mode"] = grid_mode
    result.coverage_info["grid_points"] = len(timestamps)
    _progress_log(
        result,
        f"Sample grid: {len(timestamps)} points @ {step_sec}s ({grid_mode})",
        on_progress,
    )
    return timestamps


def simulate_warmup(
    *,
    chart_dir: str,
    trading_day: str,
    feature_name: str,
    duration_minutes: int = 20,
    sampling_interval_sec: float = 10.0,
    gap_max_sec: float = DEFAULT_GAP_MAX_SEC,
    gap_injections: list[tuple[int, float]] | None = None,
    dataset_feature_names: list[str] | None = None,
    source: SourceSpec | None = None,
    max_horizon_sec: int = 0,
    run_dataset_maturity_replay: bool = True,
    run_all_features_calc: bool = False,
    strike_selection: dict[str, Any] | None = None,
    horizons_sec: list[int] | None = None,
    build_replay_settings: dict[str, Any] | None = None,
    run_temp_build_io: bool = False,
    on_progress: ProgressCallback | None = None,
) -> WarmupSimulationResult:
    """Run policy engine on first *duration_minutes* of tick grid for *feature_name*."""
    t_run = time.perf_counter()
    result = WarmupSimulationResult(
        feature_name=feature_name,
        trading_day=trading_day,
        duration_minutes=duration_minutes,
        sampling_interval_sec=sampling_interval_sec,
        gap_max_sec=gap_max_sec,
    )
    _progress_log(result, "Starting warm-up simulation…", on_progress)
    if not chart_dir or not trading_day or not feature_name:
        result.ok = False
        result.error = "Chart folder, trading day, and feature are required."
        _progress_log(result, f"FAILED: {result.error}", on_progress)
        return result

    db = replay_db_path(chart_dir, trading_day)
    if not db:
        result.ok = False
        result.error = f"No tick database for {trading_day}."
        _progress_log(result, f"FAILED: {result.error}", on_progress)
        return result
    _progress_log(result, f"Tick DB: {db}", on_progress)

    src = source or resolve_chain_source(chart_dir, trading_day)
    if not src or not src.expiry:
        result.ok = False
        result.error = f"No chain source found for {trading_day}."
        _progress_log(result, f"FAILED: {result.error}", on_progress)
        return result
    _progress_log(
        result,
        f"Source: {src.market} expiry {src.expiry}",
        on_progress,
    )

    step_sec = int(max(sampling_interval_sec, 1))
    from chain_replay_ml.dataset_builder.day_context import (
        probe_first_spot_tick_ts,
        simulator_tick_load_bounds,
    )
    from chain_replay_ml.dataset_builder.feature_plugins import horizon_column_name
    from chain_replay_ml.dataset_builder.tick_coverage import LOOKBACK_START_SEC

    horizon_list = sorted({int(h) for h in (horizons_sec or []) if int(h) > 0})
    result.horizons_sec = list(horizon_list)
    result.target_columns = [horizon_column_name(h) for h in horizon_list]
    max_horizon = max(horizon_list) if horizon_list else 0
    include_targets = bool(horizon_list) and (
        run_all_features_calc or run_dataset_maturity_replay
    )

    first_spot_ts = probe_first_spot_tick_ts(chart_dir, src)
    tick_start, tick_end = simulator_tick_load_bounds(
        trading_day,
        duration_minutes=duration_minutes,
        first_spot_ts=first_spot_ts,
        pad_before_sec=LOOKBACK_START_SEC,
        max_horizon_sec=float(max_horizon) if include_targets else 0.0,
    )
    horizon_note = f" + {max_horizon}s targets" if include_targets and max_horizon else ""
    _progress_log(
        result,
        f"Tick load window: {fmt_ist_time(tick_start)} → {fmt_ist_time(tick_end)} "
        f"({duration_minutes} min + {int(LOOKBACK_START_SEC)}s lookback{horizon_note})",
        on_progress,
    )
    _progress_log(result, "Loading tick data for simulation window…", on_progress)
    t_load = time.perf_counter()
    try:
        ctx = load_day_context(
            chart_dir,
            src,
            feature_grid_step_sec=step_sec,
            max_tick_ts=tick_end,
            tick_pad_before_sec=LOOKBACK_START_SEC,
        )
    except Exception as exc:
        result.ok = False
        result.error = f"Could not load day context: {exc}"
        _progress_log(result, f"FAILED: {result.error}", on_progress)
        result.timing["load_ticks_sec"] = round(time.perf_counter() - t_load, 3)
        result.timing["total_sec"] = round(time.perf_counter() - t_run, 3)
        _stamp_run_completed(result)
        return result
    result.timing["load_ticks_sec"] = round(time.perf_counter() - t_load, 3)
    result.timing["spot_ticks"] = int(ctx.spot_ticks)
    result.timing["chain_ticks"] = int(ctx.chain_ticks)
    result.timing["source_ticks"] = int(ctx.source_ticks)
    result.timing["tick_load_duration_min"] = int(duration_minutes)
    result.timing["tick_load_until_ts"] = tick_end
    if include_targets:
        result.timing["target_horizons_sec"] = list(horizon_list)
        result.timing["target_columns"] = list(result.target_columns)
    _progress_log(
        result,
        f"Loaded {ctx.spot_ticks:,} spot / {ctx.chain_ticks:,} chain ticks "
        f"({result.timing['load_ticks_sec']:.2f}s)",
        on_progress,
    )

    t_grid = time.perf_counter()
    timestamps = build_simulator_timestamps(
        ctx,
        step_sec=step_sec,
        duration_minutes=duration_minutes,
        session_close_ts=float(ctx.close_ts),
        result=result,
        on_progress=on_progress,
        max_horizon_sec=max_horizon if include_targets else 0,
    )
    result.timing["build_grid_sec"] = round(time.perf_counter() - t_grid, 3)
    if not timestamps:
        result.ok = False
        first = result.coverage_info.get("first_tick") or "—"
        result.error = (
            "No grid samples in the selected window. "
            f"Spot ticks on this day start at {first}. "
            "Try a longer duration or another trading day."
        )
        _progress_log(result, f"FAILED: {result.error}", on_progress)
        return result

    _progress_log(result, f"Loading policy for {feature_name}…", on_progress)
    maturity_names = list(dict.fromkeys(dataset_feature_names or [feature_name]))
    maturity_names = [n for n in maturity_names if n and not str(n).startswith("__roll.")]
    build_settings = dict(build_replay_settings or {})
    from .performance_debug import PerformanceDebugConfig

    perf = PerformanceDebugConfig.resolve(build_settings.get("performance_debug_level"))
    apply_dataset_sel = bool(build_settings.get("match_build_dataset_selection"))
    apply_gap_parity = bool(build_settings.get("match_build_gap_parity"))
    apply_lookback_nearest = bool(build_settings.get("apply_lookback_nearest"))
    run_lookback_dual_pass = perf.run_lookback_dual_pass(
        explicit=bool(build_settings.get("run_lookback_dual_pass_benchmark")),
    )
    run_gap_pass_compare = perf.run_gap_pass_comparison(
        explicit=bool(build_settings.get("run_gap_pass_comparison")),
        gap_parity=apply_gap_parity,
    )
    replay_enabled_groups: list[str] = []
    replay_feature_names = list(maturity_names)
    if apply_dataset_sel and build_settings.get("feature_selection"):
        import os

        from chain_replay_ml.dataset_builder.feature_plugins import (
            resolve_implemented_features_for_selection,
        )
        from chain_replay_ml.dataset_builder.orchestrator import _load_feature_registry

        data_dir = os.path.join(chart_dir, "data")
        registry = _load_feature_registry()
        replay_enabled_groups, implemented, _pending, _per_group = (
            resolve_implemented_features_for_selection(
                build_settings["feature_selection"],
                registry,
                data_dir=data_dir,
            )
        )
        if implemented:
            replay_feature_names = list(dict.fromkeys([*implemented, *maturity_names]))
        result.timing["match_build_dataset_selection"] = True
        result.timing["build_feature_count"] = len(replay_feature_names)
        result.timing["build_enabled_groups"] = len(replay_enabled_groups)
    else:
        result.timing["match_build_dataset_selection"] = False
    if apply_gap_parity:
        result.timing["match_build_gap_parity"] = True
        if build_settings.get("gap_max_sec") is not None:
            result.timing["build_gap_max_sec"] = float(build_settings["gap_max_sec"])
    else:
        result.timing["match_build_gap_parity"] = False
    result.timing["lookback_nearest_snapshot"] = apply_lookback_nearest
    result.timing["lookback_dual_pass_benchmark"] = run_lookback_dual_pass
    result.timing["run_gap_pass_comparison"] = run_gap_pass_compare
    result.timing["performance_debug_level"] = perf.level.value
    lb = build_settings.get("lookback_policy") or {}
    if lb.get("method"):
        result.timing["lookback_policy_method"] = str(lb.get("method"))
    result.dataset_feature_total = len(replay_feature_names)
    result.maturity_feature_names = list(replay_feature_names)
    reg = load_feature_policy_registry(feature_names=maturity_names)
    meta = reg.get(feature_name)
    if not meta:
        result.ok = False
        result.error = f"Feature not in policy registry: {feature_name}"
        _progress_log(result, f"FAILED: {result.error}", on_progress)
        return result

    ctrl_label, ctrl_key, effective = _controller_info(meta)
    result.controller_label = ctrl_label
    result.controller_key = ctrl_key
    result.effective_warmup = effective
    _progress_log(
        result,
        f"Policy: controller={ctrl_label or '—'} effective_warmup={effective} samples · "
        f"dataset features={result.dataset_feature_total}",
        on_progress,
    )

    from .warmup_maturity import explain_feature_readiness, snapshot_maturity

    maturity_timeline: list[dict[str, Any]] = []

    if gap_injections:
        _progress_log(result, f"Gap injections queued: {gap_injections}", on_progress)

    dep_pairs = _dependency_labels_for_meta(meta)
    result.dependency_labels = [label for label, _ in dep_pairs]

    pairs = expand_timestamps_with_gaps(timestamps, gap_injections or [])
    _progress_log(result, "Running FeaturePolicyEngine on sample grid…", on_progress)
    eng = FeaturePolicyEngine(
        reg, sampling_interval_sec=float(step_sec), gap_max_sec=gap_max_sec,
    )
    eng.on_session_start()
    result.events.append(f"{fmt_ist_time(pairs[0][0])} Session started")

    prev_feature_ready = False
    prev_ctrl_ready = False
    trace: list[dict[str, Any]] = []
    report_every = max(1, len(pairs) // 10)

    t_policy = time.perf_counter()
    for sample_n, (ts, gap_flag) in enumerate(pairs, start=1):
        if gap_flag and sample_n > 1:
            gap_sec = pairs[sample_n - 1][0] - pairs[sample_n - 2][0] - step_sec
            gap_evt = {
                "time": fmt_ist_time(ts),
                "gap_sec": round(max(gap_sec, 0), 1),
                "ts": ts,
                "policy": f"Reset {result.controller_label or 'rolling'}",
                "after_sample": sample_n - 1,
            }
            result.gap_events.append(gap_evt)
            result.events.append(f"{fmt_ist_time(ts)} Gap detected — {gap_evt['gap_sec']:.0f} sec")
            result.events.append(f"{fmt_ist_time(ts)} Policy: {gap_evt['policy']}")
            result.events.append(f"{fmt_ist_time(ts)} Samples reset → 0")
            if result.controller_label:
                result.events.append(f"{fmt_ist_time(ts)} {result.controller_label} restarted")
            result.events.append(f"{fmt_ist_time(ts)} Feature → NOT READY")

        eng.on_sample(ts)
        snap = eng.readiness_snapshot()
        ctrl_key = result.controller_key
        ctrl_ready = True
        ctrl_samples = 0
        if ctrl_key and ctrl_key in snap:
            ctrl_ready = bool(snap[ctrl_key].get("ready"))
            ctrl_samples = int(snap[ctrl_key].get("samples_seen") or 0)
        elif result.controller_label:
            ctrl_ready = eng.is_ready(ctrl_key) if ctrl_key else True

        feature_ready = eng.is_ready(feature_name)
        deps_ready = _collect_dep_readiness(eng, dep_pairs)
        output_display = "NULL" if not feature_ready else "VALUE"
        trace.append({
            "time": fmt_ist_time(ts),
            "ts": ts,
            "samples": sample_n,
            "ctrl_ready": ctrl_ready,
            "ctrl_samples": ctrl_samples,
            "feature_ready": feature_ready,
            "deps": deps_ready,
            "output_display": output_display,
        })

        maturity_timeline.append(
            snapshot_maturity(
                eng,
                reg,
                maturity_names,
                sample=sample_n,
                time=fmt_ist_time(ts),
                include_detail=(sample_n <= 1 or sample_n % max(1, len(pairs) // 12) == 0),
            ),
        )
        maturity_timeline[-1]["simulated_feature_explain"] = explain_feature_readiness(
            feature_name,
            eng=eng,
            reg=reg,
            sample=sample_n,
            sampling_interval_sec=float(step_sec),
        )

        if gap_flag and sample_n > 1 and result.gap_events:
            result.gap_events[-1]["samples_after_reset"] = ctrl_samples
            result.gap_events[-1]["feature_ready"] = feature_ready
            result.gap_events[-1]["ctrl_ready"] = ctrl_ready

        if sample_n == 1:
            if result.controller_label:
                result.events.append(f"{fmt_ist_time(ts)} {result.controller_label} initialized")
            result.events.append(f"{fmt_ist_time(ts)} Feature not ready")

        if ctrl_ready and not prev_ctrl_ready and result.controller_label:
            result.events.append(
                f"{fmt_ist_time(ts)} {result.controller_label} reached "
                f"{ctrl_samples} samples",
            )
        if feature_ready and not prev_feature_ready:
            result.ready_at_ts = ts
            result.ready_at_sample = sample_n
            result.events.append(f"{fmt_ist_time(ts)} Feature became READY")
        if not feature_ready and prev_feature_ready:
            result.events.append(f"{fmt_ist_time(ts)} Feature became NOT READY")

        prev_feature_ready = feature_ready
        prev_ctrl_ready = ctrl_ready

        if sample_n == 1 or sample_n == len(pairs) or sample_n % report_every == 0:
            ctrl_mark = "✓" if ctrl_ready else "✗"
            feat_mark = "✓" if feature_ready else "✗"
            _progress_log(
                result,
                f"  [{sample_n:>3}/{len(pairs)}] {fmt_ist_time(ts)}  "
                f"{result.controller_label or 'ctrl'} {ctrl_mark}  feature {feat_mark}",
                on_progress,
            )

    result.timing["policy_engine_sec"] = round(time.perf_counter() - t_policy, 3)
    result.samples_processed = len(trace)
    result.full_trace = trace
    result.maturity_timeline = maturity_timeline
    from .warmup_maturity import build_maturity_summary

    result.maturity_summary = build_maturity_summary(
        maturity_timeline,
        feature_total=result.dataset_feature_total,
    )
    stats = eng.stats_dict()
    result.gap_resets = int(stats.get("gap_resets") or 0)

    result.transition = _compute_transition(trace)
    result.dependency_timeline = _build_dependency_timeline(
        trace, result.dependency_labels, max_rows=8,
    )
    result.output_samples = _build_output_samples(trace)
    result.dataset_impact = _compute_dataset_impact(trace, gap_resets=result.gap_resets)

    need_full_replay = (run_all_features_calc or run_dataset_maturity_replay) and replay_feature_names
    replay_lookup: dict[int, dict[str, Any]] | None = None
    replay_err: str | None = None
    replay_sec = 0.0
    strike_sel = dict(strike_selection or {})
    result.strike_selection = strike_sel
    trim_target_rows = bool(build_settings.get("trim_target_rows", False))
    if need_full_replay:
        if include_targets and max_horizon > 0:
            _extend_ctx_close_for_target_replay(
                ctx,
                trading_day=trading_day,
                max_horizon_sec=float(max_horizon),
            )
            _progress_log(
                result,
                f"Extended replay close for targets (max horizon {max_horizon}s)",
                on_progress,
            )
        t_replay = time.perf_counter()
        replay_common = dict(
            chart_dir=chart_dir,
            trading_day=trading_day,
            src=src,
            ctx=ctx,
            feature_names=replay_feature_names,
            step_sec=step_sec,
            strike_selection=strike_sel,
            horizons_sec=horizon_list if include_targets else [],
            enabled_groups=replay_enabled_groups,
            gap_max_sec=build_settings.get("gap_max_sec"),
            dataset_configuration=build_settings.get("dataset_configuration"),
            trim_target_rows=trim_target_rows,
            trace=trace,
            on_progress=on_progress,
            performance_debug=perf,
        )
        lookback_policy = build_settings.get("lookback_policy")
        gap_on_for_cmp = None
        if run_all_features_calc and (apply_gap_parity or run_gap_pass_compare):
            from chain_replay_ml.dataset_builder.gap_policy import (
                default_gap_policy,
                gap_max_sec_from_policy,
            )

            gap_on_for_cmp = float(
                build_settings.get("gap_max_sec")
                or gap_max_sec_from_policy(
                    build_settings.get("gap_policy") or default_gap_policy(),
                ),
            )
        run_lookback_benchmark = bool(
            run_all_features_calc and apply_lookback_nearest and run_lookback_dual_pass,
        )
        if run_lookback_benchmark:
            _progress_log(
                result,
                "Lookback benchmark: exact_timestamp baseline replay…",
                on_progress,
            )
            t_base = time.perf_counter()
            _attach_maturity_replay(
                **replay_common,
                lookback_policy_doc=_exact_lookback_policy(),
                skip_row_stats=True,
                result=None,
            )
            baseline_sec = round(time.perf_counter() - t_base, 3)
            result.timing["feature_calc_without_lookback_sec"] = baseline_sec
            _progress_log(
                result,
                f"Lookback baseline (exact_timestamp): {baseline_sec:.2f}s",
                on_progress,
            )
            _progress_log(
                result,
                "Main replay with nearest_snapshot lookback…",
                on_progress,
            )
            t_main = time.perf_counter()
            replay_lookup, replay_err = _attach_maturity_replay(
                **replay_common,
                lookback_policy_doc=lookback_policy,
                result=result,
            )
            main_sec = round(time.perf_counter() - t_main, 3)
            result.timing["feature_calc_sec"] = main_sec
            overhead = round(main_sec - baseline_sec, 3)
            result.timing["lookback_nearest_snapshot_sec"] = max(0.0, overhead)
            replay_sec = main_sec
            result.timing["all_features_calc_wall_sec"] = round(time.perf_counter() - t_replay, 3)
        else:
            if run_all_features_calc and apply_lookback_nearest and not run_lookback_benchmark:
                result.timing["lookback_benchmark_skipped"] = (
                    "dual-pass lookback benchmark disabled — single nearest_snapshot pass"
                )
                _progress_log(
                    result,
                    "Lookback: single nearest_snapshot pass (enable dual-pass benchmark for 2× compare)",
                    on_progress,
                )
            replay_lookup, replay_err = _attach_maturity_replay(
                **replay_common,
                lookback_policy_doc=lookback_policy,
                result=result,
            )
            replay_sec = round(time.perf_counter() - t_replay, 3)
            result.timing["feature_calc_sec"] = replay_sec

        if run_all_features_calc and gap_on_for_cmp is not None and run_gap_pass_compare:
            try:
                result.timing["gap_pass_comparison"] = _run_gap_pass_profile_comparison(
                    replay_common,
                    gap_on_sec=gap_on_for_cmp,
                    lookback_policy_doc=lookback_policy,
                    on_progress=on_progress,
                )
            except Exception as exc:
                result.timing["gap_pass_comparison_error"] = str(exc)
                _progress_log(
                    result,
                    f"Gap pass comparison failed: {exc}",
                    on_progress,
                )

        if run_all_features_calc and perf.run_cache_benchmark():
            try:
                from .replay_pipeline_timing import (
                    benchmark_build_day_rows_cold_warm,
                    build_day_rows_kwargs_from_replay,
                )

                build_kw = build_day_rows_kwargs_from_replay(
                    replay_common,
                    lookback_policy_doc=lookback_policy,
                    gap_max_sec=build_settings.get("gap_max_sec"),
                    performance_debug=perf,
                )
                result.timing["replay_context_benchmark"] = benchmark_build_day_rows_cold_warm(
                    ctx,
                    build_kwargs=build_kw,
                    performance_debug=perf,
                )
            except Exception as exc:
                result.timing["replay_context_benchmark_error"] = str(exc)

    if run_all_features_calc:
        result.timing["all_features_calc"] = True
        if "feature_calc_sec" not in result.timing:
            result.timing["feature_calc_sec"] = replay_sec
        result.calc_debug = _attach_calc_debug(
            trace,
            result=result,
            chart_dir=chart_dir,
            trading_day=trading_day,
            src=src,
            ctx=ctx,
            feature_name=feature_name,
            step_sec=step_sec,
            on_progress=on_progress,
            replay_lookup=replay_lookup,
        )
    else:
        t_calc = time.perf_counter()
        result.calc_debug = _attach_calc_debug(
            trace,
            result=result,
            chart_dir=chart_dir,
            trading_day=trading_day,
            src=src,
            ctx=ctx,
            feature_name=feature_name,
            step_sec=step_sec,
            on_progress=on_progress,
            replay_lookup=replay_lookup,
        )
        result.timing["feature_calc_sec"] = round(time.perf_counter() - t_calc, 3)

    if replay_lookup:
        result.all_features_lookup = dict(replay_lookup)
    elif not run_all_features_calc:
        result.all_features_lookup = {}
    elif result.all_features_rows:
        from .warmup_calc_debug import resolve_replay_lookup_from_result

        rebuilt = resolve_replay_lookup_from_result(result)
        if rebuilt:
            replay_lookup = rebuilt
            result.all_features_lookup = dict(rebuilt)

    if run_dataset_maturity_replay:
        if replay_lookup:
            result.maturity_replay_lookup = dict(replay_lookup)
        else:
            from .warmup_calc_debug import resolve_replay_lookup_from_result

            rebuilt = resolve_replay_lookup_from_result(result)
            result.maturity_replay_lookup = dict(rebuilt) if rebuilt else {}
        result.maturity_replay_error = replay_err
        if run_all_features_calc:
            result.timing["maturity_replay_sec"] = 0.0
            result.timing["maturity_replay_shared"] = True
        else:
            result.timing["maturity_replay_sec"] = replay_sec
    else:
        result.maturity_replay_lookup = {}
        result.maturity_replay_error = "Skipped — dataset maturity replay disabled"
        result.timing["maturity_replay_sec"] = 0.0
        result.timing["maturity_replay_skipped"] = True
        _progress_log(result, "Dataset maturity replay skipped (disabled).", on_progress)

    if run_temp_build_io and result.all_features_rows:
        _run_temp_build_io(
            result.all_features_rows,
            trading_day=trading_day,
            result=result,
            on_progress=on_progress,
        )

    result.output_samples = _build_output_samples(trace)

    needed = result.effective_warmup
    if needed <= 0:
        result.policy_pass = True
        result.policy_reason = "No warm-up required"
    elif result.ready_at_sample is not None:
        result.policy_pass = result.ready_at_sample >= needed
        result.policy_reason = (
            f"Ready at sample {result.ready_at_sample} (needs {needed})"
            if result.policy_pass
            else f"Ready too early at sample {result.ready_at_sample}"
        )
    else:
        result.policy_pass = False
        result.policy_reason = f"Needs {needed} samples"

    result.timeline = _milestone_timeline(trace, max_rows=12)
    result.chart_points = _readiness_chart_points(trace)
    result.dependency_status = _dependency_status(eng, meta, feature_name)

    trans = result.transition
    if trans.get("first_ready"):
        fr = trans["first_ready"]
        result.events.append(
            f"Transition: last NOT READY sample {trans['last_not_ready']['samples'] if trans.get('last_not_ready') else '—'} "
            f"→ first READY sample {fr['samples']} @ {fr['time']}",
        )
    if result.gap_events:
        for gap_evt in result.gap_events:
            re_ready = _first_ready_after_sample(trace, int(gap_evt.get("after_sample", 0)))
            if re_ready:
                result.events.append(
                    f"After gap @ {gap_evt['time']}: READY again at sample {re_ready['samples']} ({re_ready['time']})",
                )

    _progress_log(
        result,
        f"Done — {result.samples_processed} samples, "
        f"ready={'yes' if result.ready_at_sample else 'no'}, "
        f"policy={'PASS' if result.policy_pass else 'FAIL'}",
        on_progress,
    )
    result.timing["total_sec"] = round(time.perf_counter() - t_run, 3)
    _stamp_run_completed(result)
    _progress_log(
        result,
        f"Timing — ticks {result.timing.get('load_ticks_sec', 0):.2f}s · "
        f"feature calc {result.timing.get('feature_calc_sec', 0):.2f}s · "
        f"sqlite {result.timing.get('temp_sqlite_insert_sec', 0):.2f}s · "
        f"parquet {result.timing.get('temp_parquet_export_sec', 0):.2f}s · "
        f"total {result.timing.get('total_sec', 0):.2f}s",
        on_progress,
    )
    return result


def _run_temp_build_io(
    rows: list[dict[str, Any]],
    *,
    trading_day: str,
    result: WarmupSimulationResult,
    on_progress: ProgressCallback | None,
) -> None:
    """Benchmark master SQLite insert + Parquet export on simulator rows (temp files)."""
    import os
    import shutil
    import tempfile
    import time

    if not rows:
        return
    prepared = [dict(row) for row in rows]
    for row in prepared:
        row.setdefault("trading_day", trading_day)
    row_cols = list(dict.fromkeys(key for row in prepared for key in row if not str(key).startswith("_")))
    if "trading_day" not in row_cols:
        row_cols.insert(0, "trading_day")

    temp_dir = tempfile.mkdtemp(prefix="warmup_sim_build_")
    db_path = os.path.join(temp_dir, "sim_master.sqlite")
    parquet_path = os.path.join(temp_dir, "sim_export.parquet")
    try:
        from chain_replay_ml.dataset_builder.master_export import export_master_to_parquet
        from chain_replay_ml.dataset_builder.master_store import MasterStore

        _progress_log(result, f"Temp SQLite insert ({len(prepared):,} rows)…", on_progress)
        t_sqlite = time.perf_counter()
        with MasterStore(db_path) as store:
            store.begin_day(trading_day, row_cols)
            inserted = store.insert_rows(prepared)
            store.commit_day(trading_day)
        result.timing["temp_sqlite_insert_sec"] = round(time.perf_counter() - t_sqlite, 3)
        result.timing["temp_sqlite_rows"] = int(inserted)
        if os.path.isfile(db_path):
            result.timing["temp_sqlite_bytes"] = int(os.path.getsize(db_path))
        _progress_log(
            result,
            f"Temp SQLite: {inserted:,} rows in {result.timing['temp_sqlite_insert_sec']:.2f}s",
            on_progress,
        )

        _progress_log(result, "Temp Parquet export…", on_progress)
        t_parquet = time.perf_counter()
        with MasterStore(db_path) as store:
            written = export_master_to_parquet(store, parquet_path, row_cols)
        result.timing["temp_parquet_export_sec"] = round(time.perf_counter() - t_parquet, 3)
        result.timing["temp_parquet_rows"] = int(written)
        if os.path.isfile(parquet_path):
            result.timing["temp_parquet_bytes"] = int(os.path.getsize(parquet_path))
        _progress_log(
            result,
            f"Temp Parquet: {written:,} rows in {result.timing['temp_parquet_export_sec']:.2f}s",
            on_progress,
        )
    except Exception as exc:
        result.timing["temp_build_io_error"] = str(exc)
        _progress_log(result, f"Temp build I/O skipped: {exc}", on_progress)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def _milestone_timeline(trace: list[dict[str, Any]], *, max_rows: int = 8) -> list[dict[str, Any]]:
    if not trace:
        return []
    if len(trace) <= max_rows:
        return trace
    indices = {0, len(trace) - 1}
    step = max(1, (len(trace) - 1) // (max_rows - 1))
    for i in range(0, len(trace), step):
        indices.add(i)
    return [trace[i] for i in sorted(indices)]


def _readiness_chart_points(trace: list[dict[str, Any]]) -> list[tuple[str, float]]:
    if not trace:
        return []
    n = len(trace)
    buckets = 10
    out: list[tuple[str, float]] = []
    for b in range(buckets):
        start = int(b * n / buckets)
        end = max(start + 1, int((b + 1) * n / buckets))
        chunk = trace[start:end]
        pct = sum(1 for row in chunk if row.get("feature_ready")) / len(chunk) * 100.0
        out.append((chunk[-1]["time"], pct))
    return out


def _dependency_status(
    eng: FeaturePolicyEngine,
    meta: Any,
    feature_name: str,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    deps = list(getattr(meta, "dependencies", ()) or ())
    anchor = getattr(meta, "policy_anchor", None)
    if anchor and anchor not in deps:
        deps = list(deps) + [anchor]
    for dep_id in deps:
        if dep_id in ("timestamp", "token", "symbol", "feature_grid") or str(dep_id).startswith("feature_grid"):
            continue
        label = _dep_label(dep_id)
        if label in seen:
            continue
        seen.add(label)
        samples = 0
        if dep_id.startswith("__roll."):
            parts = dep_id.split(".")
            if len(parts) >= 3 and parts[2].startswith("ema"):
                try:
                    samples = int(parts[2].replace("ema", ""))
                except ValueError:
                    pass
        if samples <= 0 and dep_id.lower() in ("ltp", "spot"):
            ready = True
        elif dep_id.startswith("__roll."):
            snap = eng.readiness_snapshot().get(dep_id) or {}
            ready = bool(snap.get("ready", eng.is_ready(dep_id)))
        else:
            ready = eng.is_ready(dep_id) if eng.registry.get(dep_id) else True
        out.append({"label": label, "dep_id": dep_id, "ready": ready, "samples": samples})
    return out


def _dependency_labels_for_meta(meta: Any) -> list[tuple[str, str]]:
    """(display_label, dep_id) in stable order."""
    out: list[tuple[str, str]] = []
    seen: set[str] = set()
    deps = list(getattr(meta, "dependencies", ()) or ())
    anchor = getattr(meta, "policy_anchor", None)
    if anchor and anchor not in deps:
        deps = list(deps) + [anchor]
    for dep_id in deps:
        if dep_id in ("timestamp", "token", "symbol", "feature_grid") or str(dep_id).startswith("feature_grid"):
            continue
        label = _dep_label(dep_id)
        if label in seen:
            continue
        seen.add(label)
        out.append((label, dep_id))
    return out


def _dep_ready_at_sample(eng: FeaturePolicyEngine, dep_id: str) -> bool:
    if dep_id.lower() in ("ltp", "spot"):
        return True
    if dep_id.startswith("__roll."):
        snap = eng.readiness_snapshot().get(dep_id) or {}
        return bool(snap.get("ready", eng.is_ready(dep_id)))
    if not eng.registry.get(dep_id):
        return True
    return eng.is_ready(dep_id)


def _collect_dep_readiness(
    eng: FeaturePolicyEngine,
    dep_pairs: list[tuple[str, str]],
) -> dict[str, bool]:
    return {label: _dep_ready_at_sample(eng, dep_id) for label, dep_id in dep_pairs}


def _compute_transition(trace: list[dict[str, Any]]) -> dict[str, Any]:
    last_nr: dict[str, Any] | None = None
    first_r: dict[str, Any] | None = None
    for row in trace:
        if not row.get("feature_ready"):
            last_nr = row
        elif first_r is None:
            first_r = row
    return {"last_not_ready": last_nr, "first_ready": first_r}


def _first_ready_after_sample(trace: list[dict[str, Any]], after_sample: int) -> dict[str, Any] | None:
    for row in trace:
        if int(row.get("samples", 0)) <= after_sample:
            continue
        if row.get("feature_ready"):
            return row
    return None


def _compute_dataset_impact(trace: list[dict[str, Any]], *, gap_resets: int = 0) -> dict[str, Any]:
    total = len(trace)
    ready_n = sum(1 for r in trace if r.get("feature_ready"))
    null_n = total - ready_n
    pct = round(ready_n / max(total, 1) * 100.0, 1)
    return {
        "total_samples": total,
        "ready_samples": ready_n,
        "null_samples": null_n,
        "ready_pct": pct,
        "gap_resets": gap_resets,
        "effective_training_rows": ready_n,
    }


def _build_dependency_timeline(
    trace: list[dict[str, Any]],
    dep_labels: list[str],
    *,
    max_rows: int = 6,
) -> list[dict[str, Any]]:
    if not trace or not dep_labels:
        return []
    rows = _milestone_timeline(trace, max_rows=max_rows)
    out: list[dict[str, Any]] = []
    for row in rows:
        deps = row.get("deps") or {}
        out.append({
            "time": row.get("time"),
            "samples": row.get("samples"),
            "deps": {label: bool(deps.get(label, True)) for label in dep_labels},
            "feature_ready": bool(row.get("feature_ready")),
        })
    return out


def _build_output_samples(trace: list[dict[str, Any]], *, around: int = 3) -> list[dict[str, Any]]:
    if not trace:
        return []
    trans = _compute_transition(trace)
    first_r = trans.get("first_ready")
    indices: list[int] = []
    if first_r:
        idx = int(first_r.get("samples", 1)) - 1
        start = max(0, idx - around)
        end = min(len(trace), idx + around + 1)
        indices.extend(range(start, end))
    else:
        indices.extend(range(min(5, len(trace))))
    ready_tail = [i for i, r in enumerate(trace) if r.get("feature_ready")]
    if ready_tail:
        tail_start = ready_tail[0]
        indices.extend(range(tail_start, min(len(trace), tail_start + 5)))
    seen: set[int] = set()
    ordered: list[int] = []
    for i in indices:
        if i not in seen:
            seen.add(i)
            ordered.append(i)
    ordered.sort()
    return [trace[i] for i in ordered]


def _extend_ctx_close_for_target_replay(
    ctx: Any,
    *,
    trading_day: str,
    max_horizon_sec: float,
) -> None:
    """Extend session close so clipped grid survives close_ts - max_horizon clipping."""
    if max_horizon_sec <= 0:
        return
    from chain_replay_ml.dataset_builder.day_context import ist_market_session_bounds

    _, session_close = ist_market_session_bounds(trading_day)
    ctx.close_ts = min(session_close, float(ctx.close_ts) + float(max_horizon_sec))


def _strike_band_dims(strike_selection: dict[str, Any]) -> dict[str, int]:
    """Expected strike/row counts for atm_band mode (21 strikes × CE/PE = 42 rows)."""
    mode = str(strike_selection.get("mode") or "atm_band").lower()
    if mode != "atm_band":
        return {}
    band_raw = strike_selection.get("atmBand", 10)
    if str(band_raw).lower() == "all":
        return {}
    band = int(band_raw or 10)
    strikes = 2 * band + 1
    sides = 2
    return {
        "strikes_in_band": strikes,
        "option_sides": sides,
        "rows_per_timestamp": strikes * sides,
    }


def _serialize_replay_rows(df: Any) -> list[dict[str, Any]]:
    """Convert replay dataframe to JSON-safe row dicts (all strikes/tokens)."""
    import pandas as pd

    if df is None or getattr(df, "empty", True):
        return []
    use_cols = [c for c in df.columns if not str(c).startswith("_")]
    if not use_cols:
        return []
    out: list[dict[str, Any]] = []
    for rec in df[use_cols].to_dict(orient="records"):
        row: dict[str, Any] = {}
        for key, val in rec.items():
            if val is None or (isinstance(val, float) and pd.isna(val)):
                row[str(key)] = None
            elif isinstance(val, (int, float)):
                row[str(key)] = float(val) if isinstance(val, float) else int(val)
            else:
                row[str(key)] = val
        out.append(row)
    return out


def _record_chain_row_stats(
    result: WarmupSimulationResult,
    *,
    rows: list[dict[str, Any]],
    trace: list[dict[str, Any]],
    strike_selection: dict[str, Any],
) -> None:
    """Store per-token chain row counts for timing / export."""
    result.all_features_rows = list(rows)
    result.strike_selection = dict(strike_selection or {})
    timing = result.timing
    timing["policy_grid_samples"] = len(trace)
    timing["chain_rows_total"] = len(rows)
    if rows:
        ts_keys = {
            int(round(float(r["timestamp"])))
            for r in rows
            if r.get("timestamp") is not None
        }
        timing["chain_grid_timestamps"] = len(ts_keys)
        timing["avg_rows_per_timestamp"] = round(
            len(rows) / max(len(ts_keys), 1), 1,
        )
        timing["avg_strikes_per_timestamp"] = timing["avg_rows_per_timestamp"]
        tokens = {str(r.get("token")) for r in rows if r.get("token")}
        timing["unique_tokens"] = len(tokens)
    band_dims = _strike_band_dims(strike_selection)
    if band_dims:
        timing.update(band_dims)
        grid_n = int(timing.get("chain_grid_timestamps") or timing.get("policy_grid_samples") or 0)
        if grid_n > 0:
            timing["expected_chain_rows"] = grid_n * band_dims["rows_per_timestamp"]
    try:
        from chain_replay_ml.dataset_builder.expected_spec import format_strike_selection_label

        label = format_strike_selection_label(strike_selection)
        if label:
            timing["strike_selection_label"] = label
    except Exception:
        pass


def _exact_lookback_policy() -> dict[str, Any]:
    from chain_replay_ml.dataset_builder.lookback_policy import (
        POLICY_EXACT_TIMESTAMP,
        normalize_policy_doc,
    )

    return normalize_policy_doc({"method": POLICY_EXACT_TIMESTAMP, "label": "Exact Timestamp"})


def _run_gap_pass_profile_comparison(
    replay_common: dict[str, Any],
    *,
    gap_on_sec: float,
    lookback_policy_doc: dict[str, Any] | None,
    on_progress: ProgressCallback | None,
) -> dict[str, Any]:
    """Gap OFF vs ON via isolated build_day_rows passes (no replay attachment)."""
    from chain_replay_ml.dataset_builder.gap_policy import default_gap_policy, gap_max_sec_from_policy
    from chain_replay_ml.dataset_builder.gap_policy_profiler import compare_gap_profiles

    from .replay_pipeline_timing import build_day_rows_kwargs_from_replay

    ctx = replay_common.get("ctx")
    if ctx is None:
        raise ValueError("gap comparison requires loaded day context")

    gap_on = float(gap_on_sec or gap_max_sec_from_policy(default_gap_policy()))
    build_kw = build_day_rows_kwargs_from_replay(
        replay_common,
        lookback_policy_doc=lookback_policy_doc,
        gap_max_sec=gap_on,
    )

    if on_progress:
        on_progress("Gap diff: build_day_rows OFF then ON (isolated cProfile)…")

    doc = compare_gap_profiles(
        ctx,
        step_sec=build_kw["step_sec"],
        strike_selection=build_kw["strike_selection"],
        horizons_sec=build_kw["horizons_sec"],
        enabled_groups=build_kw["enabled_groups"],
        group_labels=build_kw["group_labels"],
        implemented_features=build_kw["implemented_features"],
        per_group_features=build_kw["per_group_features"],
        lookback_policy_doc=build_kw["lookback_policy_doc"],
        gap_on_max_sec=gap_on,
        trim_target_rows=build_kw["trim_target_rows"],
        active_features=build_kw["active_features"],
    )
    if on_progress:
        on_progress(
            f"Gap diff done: OFF {doc.get('gap_off_wall_sec')}s · ON {doc.get('gap_on_wall_sec')}s "
            f"(Δ {doc.get('delta_wall_sec')}s) · {doc.get('changed_function_count', 0)} changed functions",
        )
    return doc


def _attach_maturity_replay(
    *,
    chart_dir: str,
    trading_day: str,
    src: SourceSpec,
    ctx: Any,
    feature_names: list[str],
    step_sec: int,
    strike_selection: dict[str, Any] | None = None,
    horizons_sec: list[int] | None = None,
    enabled_groups: list[str] | None = None,
    gap_max_sec: float | None = None,
    lookback_policy_doc: dict[str, Any] | None = None,
    dataset_configuration: dict[str, Any] | None = None,
    trim_target_rows: bool = True,
    trace: list[dict[str, Any]] | None = None,
    result: WarmupSimulationResult | None = None,
    skip_row_stats: bool = False,
    performance_debug_level: Any = None,
    performance_debug: Any = None,
    on_progress: ProgressCallback | None,
) -> tuple[dict[int, dict[str, Any]], str | None]:
    """Load replay frame for all dataset features (maturity value panel)."""
    import os
    import time

    from .performance_debug import PerformanceDebugConfig
    from .replay_pipeline_timing import merge_frame_timing
    from .warmup_calc_debug import build_replay_lookup

    perf = PerformanceDebugConfig.resolve(
        performance_debug_level,
        config=performance_debug,
    )

    if not feature_names:
        return {}, "No features"
    data_dir = os.path.join(chart_dir, "data")
    if not os.path.isdir(data_dir):
        return {}, "No data directory"

    try:
        from chain_replay_ml.dataset_builder.dataset_selection_engine import DatasetSelectionSpec
        from chain_replay_ml.dataset_builder.feature_plugins import horizon_column_name
        from chain_replay_ml.replay_feature_scoring import build_replay_day_frame

        columns = list(dict.fromkeys(feature_names + ["ltp", "spot"]))
        horizon_list = sorted({int(h) for h in (horizons_sec or []) if int(h) > 0})
        target_cols = [horizon_column_name(h) for h in horizon_list]
        include_targets = bool(horizon_list)
        strike_sel = DatasetSelectionSpec.from_strike_selection(
            strike_selection or {},
            market=src.market,
            interval_sec=step_sec,
        ).to_strike_selection_dict()
        ds_cfg = dict(dataset_configuration or {})
        ds_cfg.setdefault("sampling_interval_sec", step_sec)
        ds_cfg.setdefault("future_targets_sec", list(horizon_list))
        replay_groups = list(enabled_groups or [])
        if not replay_groups:
            from chain_replay_ml.dataset_builder.orchestrator import _load_feature_registry

            replay_groups = list(_load_feature_registry().get("groupOrder") or [])
        replay_config = {
            "market": src.market,
            "expiry": src.expiry,
            "sampling": {"interval_sec": step_sec},
            "dataset_configuration": ds_cfg,
            "strike_selection": strike_sel,
            "prediction_target_columns": target_cols,
            "feature_groups_implemented": replay_groups,
            "lookback_policy": dict(lookback_policy_doc or ds_cfg.get("lookback_policy") or {}),
        }
        if on_progress:
            parity_bits: list[str] = []
            if trim_target_rows:
                parity_bits.append("trim targets")
            if gap_max_sec is not None:
                parity_bits.append(f"gap {int(gap_max_sec)}s")
            if lookback_policy_doc:
                parity_bits.append("lookback policy")
            if enabled_groups:
                parity_bits.append(f"{len(enabled_groups)} groups")
            parity_note = f" [{', '.join(parity_bits)}]" if parity_bits else ""
            target_note = f", {len(target_cols)} targets" if include_targets else ""
            on_progress(
                f"Loading chain replay ({len(columns)} feature cols, strike band"
                f"{target_note}){parity_note} · production build parity…",
            )
        pipeline: dict[str, float] = {}
        t_frame = time.perf_counter()
        df, err, _, build_stats = build_replay_day_frame(
            data_dir,
            replay_config,
            trading_day,
            expiry_hint=src.expiry,
            underlying=src.market,
            required_features=columns,
            day_context=ctx,
            inference_only=not include_targets,
            trim_target_rows=trim_target_rows,
            gap_max_sec=gap_max_sec,
            performance_debug=perf,
            production_parity=True,
        )
        pipeline.update(merge_frame_timing(build_stats if isinstance(build_stats, dict) else None))
        pipeline["build_replay_day_frame_sec"] = round(time.perf_counter() - t_frame, 3)
        if err or df is None or df.empty:
            return {}, err or "Empty replay frame"
        t_ser = time.perf_counter()
        rows = _serialize_replay_rows(df)
        pipeline["serialize_replay_rows_sec"] = round(time.perf_counter() - t_ser, 3)
        t_stats = time.perf_counter()
        if result is not None and not skip_row_stats:
            result.horizons_sec = list(horizon_list)
            result.target_columns = list(target_cols)
            _record_chain_row_stats(
                result,
                rows=rows,
                trace=trace or [],
                strike_selection=strike_sel,
            )
            if isinstance(build_stats, dict):
                cov = (build_stats.get("coverage") or {})
                if cov.get("samples_written") is not None:
                    result.timing["chain_rows_written"] = int(cov["samples_written"])
                trimmed = int(build_stats.get("target_trimmed_rows") or 0)
                if trimmed:
                    result.timing["target_trimmed_rows"] = trimmed
            profiler = build_stats.get("gap_policy_profiler")
            if isinstance(profiler, dict) and result is not None and perf.collect_gap_profile():
                result.timing["gap_policy_profiler"] = profiler
            readiness_prof = build_stats.get("readiness_profiler")
            if isinstance(readiness_prof, dict) and result is not None and perf.collect_readiness_profile():
                result.timing["readiness_profiler"] = readiness_prof
        pipeline["replay_statistics_sec"] = round(time.perf_counter() - t_stats, 3)
        t_lookup = time.perf_counter()
        from .warmup_calc_debug import build_replay_lookup_from_rows, resolve_primary_replay_token

        anchor_ts = None
        if trace:
            anchor_ts = float(trace[0].get("ts") or 0.0) or None
        replay_token = resolve_primary_replay_token(rows, anchor_ts=anchor_ts, step_sec=step_sec)
        lookup = build_replay_lookup_from_rows(
            rows,
            columns,
            token=replay_token,
            anchor_ts=anchor_ts,
            step_sec=step_sec,
        )
        if not lookup:
            lookup = build_replay_lookup(
                df,
                columns,
                step_sec=step_sec,
                token=replay_token,
                anchor_ts=anchor_ts,
            )
        from .warmup_calc_debug import align_replay_lookup_to_trace, lookup_replay_hit_rate

        lookup = align_replay_lookup_to_trace(
            lookup,
            chain_rows=rows,
            trace=list(trace or []),
            columns=columns,
            step_sec=step_sec,
            token=replay_token,
            anchor_ts=anchor_ts,
        )
        if result is not None and replay_token:
            result.replay_token = replay_token
            result.timing["replay_token"] = replay_token
        pipeline["build_replay_lookup_sec"] = round(time.perf_counter() - t_lookup, 3)
        if result is not None and perf.collect_pipeline_timings():
            from .replay_pipeline_timing import finalize_pipeline_stages

            result.timing["replay_pipeline"] = finalize_pipeline_stages(pipeline)
        if trace:
            hit_rate = lookup_replay_hit_rate(lookup, trace, step_sec=step_sec)
            if result is not None:
                result.timing["replay_lookup_hit_rate"] = round(hit_rate, 4)
        if on_progress:
            hit_lbl = ""
            if trace:
                hit_lbl = f", trace hit {lookup_replay_hit_rate(lookup, trace, step_sec=step_sec):.0%}"
            on_progress(
                f"Chain replay: {len(rows):,} rows "
                f"({len(lookup)} policy timestamps, {len(columns)} features{hit_lbl})",
            )
        return lookup, None
    except Exception as exc:
        return {}, str(exc)


def _attach_calc_debug(
    trace: list[dict[str, Any]],
    *,
    result: WarmupSimulationResult,
    chart_dir: str,
    trading_day: str,
    src: SourceSpec,
    ctx: Any,
    feature_name: str,
    step_sec: int,
    on_progress: ProgressCallback | None,
    replay_lookup: dict[int, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Attach replay values + calculation debugger rows to trace and result."""
    import os

    from .warmup_calc_debug import (
        build_calculation_rows,
        build_replay_lookup,
        build_replay_lookup_from_rows,
        lookup_replay_hit_rate,
        lookup_replay_values,
        replay_columns_for,
    )

    columns = replay_columns_for(feature_name)
    data_dir = os.path.join(chart_dir, "data")
    if not os.path.isdir(data_dir):
        return {"ok": False, "error": "No data directory"}

    try:
        import pandas as pd

        if replay_lookup is not None:
            lookup = replay_lookup
            if on_progress:
                on_progress(
                    f"Using shared replay lookup ({len(lookup)} buckets) for calc debug",
                )
        else:
            from chain_replay_ml.replay_feature_scoring import build_replay_day_frame

            replay_config = {
                "market": src.market,
                "expiry": src.expiry,
                "sampling": {"interval_sec": step_sec},
                "dataset_configuration": {"sampling_interval_sec": step_sec},
            }
            df, err, _, _ = build_replay_day_frame(
                data_dir,
                replay_config,
                trading_day,
                expiry_hint=src.expiry,
                underlying=src.market,
                required_features=columns,
                day_context=ctx,
                inference_only=True,
            )
            if err or df is None or df.empty:
                return {"ok": False, "error": err or "Empty replay frame"}

            lookup = build_replay_lookup(df, columns, step_sec=step_sec)

        chain_rows = list(getattr(result, "all_features_rows", None) or [])
        step_i = int(max(getattr(result, "sampling_interval_sec", 3) or 3, 1))
        trace = list(getattr(result, "full_trace", None) or [])
        anchor_ts = float(trace[0]["ts"]) if trace else None
        token = getattr(result, "replay_token", None)
        if chain_rows:
            from .warmup_calc_debug import align_replay_lookup_to_trace, resolve_primary_replay_token

            token = token or resolve_primary_replay_token(
                chain_rows,
                anchor_ts=anchor_ts,
                step_sec=step_i,
            )
            rows_lookup = build_replay_lookup_from_rows(
                chain_rows,
                columns,
                token=token,
                anchor_ts=anchor_ts,
                step_sec=step_i,
            )
            rows_lookup = align_replay_lookup_to_trace(
                rows_lookup,
                chain_rows=chain_rows,
                trace=trace,
                columns=columns,
                step_sec=step_i,
                token=token,
                anchor_ts=anchor_ts,
            )
            shared_rate = lookup_replay_hit_rate(lookup, trace, step_sec=step_i)
            rows_rate = lookup_replay_hit_rate(rows_lookup, trace, step_sec=step_i)
            if rows_rate > shared_rate:
                lookup = rows_lookup
                if on_progress:
                    on_progress(
                        f"Calc debug: using chain rows lookup "
                        f"({rows_rate:.0%} hit rate vs {shared_rate:.0%})",
                    )
        matched = 0
        for row in trace:
            replay_vals = lookup_replay_values(lookup, row["ts"], step_sec=step_sec)
            feat_val = replay_vals.get(feature_name)
            if feat_val is not None and not (isinstance(feat_val, float) and pd.isna(feat_val)):
                row["computed_value"] = float(feat_val)
                matched += 1
            if not row.get("feature_ready"):
                row["output_display"] = "NULL"
            elif row.get("computed_value") is not None:
                row["output_display"] = _fmt_num_display(row["computed_value"])
            else:
                row["output_display"] = "VALUE"

        calc_rows, spec = build_calculation_rows(
            trace,
            feature_name=feature_name,
            replay_lookup=lookup,
            step_sec=step_sec,
        )

        for calc_row in calc_rows:
            idx = int(calc_row.get("index", -1))
            if 0 <= idx < len(trace):
                tr = trace[idx]
                operands = calc_row.get("operands") or {}
                for key, val in operands.items():
                    if key == "feature_value":
                        continue
                    tr.setdefault("calc_operands", {})[key] = val
                if operands.get("feature_value") is not None:
                    tr["computed_value"] = operands["feature_value"]
                    tr["output_display"] = _fmt_num_display(operands["feature_value"])

        if on_progress:
            on_progress(
                f"Calculation debugger: {len(calc_rows)} rows, "
                f"{matched}/{len(trace)} values attached",
            )
        return {
            "ok": True,
            "formula_spec": spec,
            "rows": calc_rows,
            "replay_columns": columns,
        }
    except Exception as exc:
        if on_progress:
            on_progress(f"Calculation debugger skipped: {exc}")
        return {"ok": False, "error": str(exc)}


def _fmt_num_display(val: Any) -> str:
    if val is None:
        return "NULL"
    try:
        f = float(val)
    except (TypeError, ValueError):
        return str(val)
    if abs(f) >= 1000:
        return f"{f:.2f}"
    if abs(f) >= 1:
        return f"{f:.4f}"
    return f"{f:.5f}"


def _attach_computed_outputs(
    trace: list[dict[str, Any]],
    *,
    chart_dir: str,
    trading_day: str,
    src: SourceSpec,
    ctx: Any,
    feature_name: str,
    step_sec: int,
    on_progress: ProgressCallback | None,
) -> None:
    import os

    import pandas as pd

    data_dir = os.path.join(chart_dir, "data")
    if not os.path.isdir(data_dir):
        return
    try:
        from chain_replay_ml.replay_feature_scoring import build_replay_day_frame

        replay_config = {
            "market": src.market,
            "expiry": src.expiry,
            "sampling": {"interval_sec": step_sec},
            "dataset_configuration": {"sampling_interval_sec": step_sec},
        }
        from .warmup_calc_debug import replay_columns_for

        columns = replay_columns_for(feature_name)
        df, err, _, _ = build_replay_day_frame(
            data_dir,
            replay_config,
            trading_day,
            expiry_hint=src.expiry,
            underlying=src.market,
            required_features=columns,
            day_context=ctx,
            inference_only=True,
        )
        if err or df is None or df.empty or feature_name not in df.columns:
            return
        if "timestamp" not in df.columns:
            return
        ts_vals = pd.to_numeric(df["timestamp"], errors="coerce")
        lookup: dict[int, Any] = {}
        for ts_raw, val in zip(ts_vals, df[feature_name]):
            if pd.isna(ts_raw):
                continue
            lookup[int(round(float(ts_raw)))] = val
        matched = 0
        for row in trace:
            ts_key = int(round(float(row["ts"])))
            val = lookup.get(ts_key)
            if val is None:
                for delta in (-step_sec, 0, step_sec):
                    val = lookup.get(ts_key + delta)
                    if val is not None:
                        break
            if val is not None and not (isinstance(val, float) and pd.isna(val)):
                row["computed_value"] = float(val) if isinstance(val, (int, float)) else val
                matched += 1
            if not row.get("feature_ready"):
                row["output_display"] = "NULL"
            elif row.get("computed_value") is not None:
                cv = row["computed_value"]
                row["output_display"] = f"{cv:.3f}" if isinstance(cv, float) else str(cv)
            else:
                row["output_display"] = "VALUE"
        if on_progress:
            on_progress(f"Attached computed values for {matched}/{len(trace)} samples")
    except Exception as exc:
        if on_progress:
            on_progress(f"Computed values skipped: {exc}")


def _find_feature_for_ema_period(
    period: int,
    feature_names: list[str] | None = None,
) -> str | None:
    reg = load_feature_policy_registry(feature_names=feature_names)
    needle = f"ema{period}"
    candidates: list[str] = []
    for name, meta in reg.features.items():
        if name.startswith("__roll."):
            continue
        anchor = getattr(meta, "policy_anchor", "") or ""
        if needle in anchor.lower() or f"ema{period}" in name.lower():
            eff = int(getattr(meta, "effective_warmup_samples", 0) or 0)
            if eff == period or f"ema{period}" in name.lower():
                candidates.append(name)
    if not candidates:
        for name in reg.features:
            if f"ltp_ema{period}" in name or f"spot_ema{period}" in name:
                return name
        return None
    candidates.sort(key=lambda n: (0 if "_to_" in n else 1, len(n)))
    return candidates[0]


def _format_ready_duration(sec: float | None) -> str:
    if sec is None:
        return "—"
    if sec < 60:
        return f"{sec:.0f} sec"
    mins = sec / 60.0
    return f"{mins:.1f} min" if abs(mins - round(mins)) > 0.05 else f"{mins:.0f} min"


def compare_ema_readiness(
    *,
    chart_dir: str,
    trading_day: str,
    duration_minutes: int = 50,
    sampling_interval_sec: float = 10.0,
    gap_max_sec: float = DEFAULT_GAP_MAX_SEC,
    ema_periods: tuple[int, ...] = (9, 20, 50, 100, 200, 300),
    feature_names: list[str] | None = None,
    on_progress: ProgressCallback | None = None,
) -> list[dict[str, Any]]:
    """Run warm-up sim for one feature per EMA period and return ready times."""
    rows: list[dict[str, Any]] = []
    for period in ema_periods:
        ema_label = f"EMA{period}"
        feat = _find_feature_for_ema_period(period, feature_names)
        if not feat:
            ready_sec = period * sampling_interval_sec
            rows.append({
                "ema": ema_label,
                "feature": "—",
                "ready_sample": period,
                "ready_time": _format_ready_duration(ready_sec),
                "ready_sec": ready_sec,
                "policy_pass": None,
            })
            continue
        if on_progress:
            on_progress(f"Compare: {ema_label} via {feat}…")
        sim = simulate_warmup(
            chart_dir=chart_dir,
            trading_day=trading_day,
            feature_name=feat,
            duration_minutes=duration_minutes,
            sampling_interval_sec=sampling_interval_sec,
            gap_max_sec=gap_max_sec,
            on_progress=on_progress,
        )
        ready_row = None
        if sim.ready_at_sample and sim.full_trace:
            ready_row = sim.full_trace[sim.ready_at_sample - 1]
        if sim.ready_at_sample:
            ready_sec = sim.ready_at_sample * sampling_interval_sec
        else:
            ready_sec = None
        rows.append({
            "ema": ema_label,
            "feature": feat,
            "ready_sample": sim.ready_at_sample,
            "ready_time": _format_ready_duration(ready_sec) if ready_sec else "Never",
            "ready_sec": ready_sec,
            "ready_clock": ready_row.get("time") if ready_row else "—",
            "policy_pass": sim.policy_pass,
        })
    return rows

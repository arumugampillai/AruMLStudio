"""Generate replay-day features from tick DB for registry model inference.

Training datasets are used only at train time. Replay scoring always rebuilds
feature rows from the replay tick database using the saved replay configuration.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Callable

import pandas as pd

from chain_replay_ml.dataset_builder.day_context import SourceSpec, load_day_context, token_timelines_from_day_context
from chain_replay_ml.dataset_builder.feature_enrichment import SCORING_INFRA_COLUMNS
from chain_replay_ml.dataset_builder.feature_plugins import (
    implemented_features_for_groups,
    implemented_features_from_names,
)
from chain_replay_ml.dataset_builder.schema_registry import load_feature_registry
from chain_replay_ml.dataset_builder.stages import build_day_rows
from chain_replay_ml.export_atm_pipeline import replay_db_path
from chain_replay_ml.replay_config import build_replay_config_from_metadata, load_dataset_metadata_json
from chain_replay_ml.replay_scoring_cache import (
    get_cached_day_frame,
    replay_cache_key,
    replay_timeline_cache_key,
    set_cached_day_frame,
    set_cached_token_timelines,
)


def chart_dir_from_data_dir(data_dir: str) -> str:
    return os.path.dirname(os.path.abspath(data_dir))


def resolve_replay_feature_config(
    data_dir: str,
    model_config: dict[str, Any],
) -> tuple[dict[str, Any], str]:
    """Return (replay_config, source_label). Dataset parquet is never loaded."""
    embedded = model_config.get("replay_config")
    if isinstance(embedded, dict) and embedded:
        return embedded, "model_package"

    dataset = str(model_config.get("dataset") or "")
    if dataset:
        meta = load_dataset_metadata_json(data_dir, dataset)
        if meta:
            return build_replay_config_from_metadata(meta), "dataset_metadata_config"
    return {}, "none"


def target_horizon_sec(target: str) -> int | None:
    name = str(target or "").strip()
    if name == "future_ltp_1m":
        return 60
    if name == "future_ltp_5m":
        return 300
    if m := re.fullmatch(r"future_ltp_(\d+)s", name):
        return int(m.group(1))
    if m := re.fullmatch(r"future_ltp_(\d+)m", name):
        return int(m.group(1)) * 60
    return None


def horizons_sec_from_replay_config(replay_config: dict[str, Any], target: str = "") -> list[int]:
    cfg = replay_config.get("dataset_configuration") or {}
    horizons = [int(h) for h in (cfg.get("future_targets_sec") or []) if h is not None]
    if not horizons:
        for col in replay_config.get("prediction_target_columns") or []:
            sec = target_horizon_sec(str(col))
            if sec is not None:
                horizons.append(sec)
    th = target_horizon_sec(target)
    if th is not None:
        horizons.append(th)
    return sorted(set(horizons)) or [300]


def resolve_scoring_expiry(
    chart_dir: str,
    date_str: str,
    expiry_hint: str | None,
    *,
    underlying: str = "NIFTY",
) -> dict[str, Any]:
    """Pick an expiry with option ticks for ML scoring; auto-fallback when URL expiry has none."""
    import sqlite3

    from chain_replay_ml.export_atm_pipeline import normalize_index_name, replay_db_path
    from storage.chain_replay_export import chain_expiries_with_option_ticks, require_v1_ticks_schema

    requested = str(expiry_hint or "").strip()
    db_path = replay_db_path(chart_dir, date_str)
    base: dict[str, Any] = {
        "requested_expiry": requested or None,
        "resolved_expiry": requested or None,
        "auto_resolved": False,
        "expiries_with_ticks": [],
        "reason": "no_tick_db",
    }
    if not db_path or not os.path.isfile(db_path):
        return base

    index_key = normalize_index_name(underlying)
    try:
        conn = sqlite3.connect(db_path)
        try:
            require_v1_ticks_schema(conn)
            with_ticks = chain_expiries_with_option_ticks(
                conn,
                underlying=index_key,
                as_of_date=date_str,
                normalize_index_name=normalize_index_name,
            )
        finally:
            conn.close()
    except Exception as exc:
        return {**base, "reason": str(exc)}

    base["expiries_with_ticks"] = with_ticks
    if requested and requested in with_ticks:
        return {
            **base,
            "resolved_expiry": requested,
            "auto_resolved": False,
            "reason": "ok",
        }

    if not with_ticks:
        return {
            **base,
            "resolved_expiry": None,
            "auto_resolved": False,
            "reason": "no_option_ticks_in_session",
        }

    resolved = with_ticks[0]
    auto = bool(requested and resolved != requested)
    return {
        **base,
        "resolved_expiry": resolved,
        "auto_resolved": auto or not requested,
        "reason": "auto_resolved" if auto or not requested else "ok",
    }


def resolve_replay_source_spec(
    replay_config: dict[str, Any],
    date_str: str,
    expiry_hint: str | None,
) -> dict[str, str] | None:
    market = str(replay_config.get("market") or "NIFTY")
    expiry = str(expiry_hint or "").strip()
    if not expiry:
        return None
    return {
        "source_id": f"{date_str}|{market}|{expiry}",
        "trading_day": str(date_str),
        "market": market,
        "expiry": expiry,
    }


def merge_replay_feature_build_plan(
    enabled_groups: list[str],
    registry: dict[str, Any],
    required_features: list[str] | None = None,
) -> tuple[list[str], list[str], list[str], dict[str, list[str]]]:
    """Merge replay-config groups with explicit model feature requirements."""
    groups = list(enabled_groups or [])
    implemented, pending, per_group = implemented_features_for_groups(groups, registry)
    if required_features:
        infra = [c for c in SCORING_INFRA_COLUMNS if c not in required_features]
        req_impl, req_pending, req_per_group = implemented_features_from_names(
            list(required_features) + infra,
            registry,
        )
        for gid, feats in req_per_group.items():
            if gid not in groups:
                groups.append(gid)
            cur = per_group.setdefault(gid, [])
            for feat in feats:
                if feat not in cur:
                    cur.append(feat)
                if feat not in implemented:
                    implemented.append(feat)
        pending = sorted({*pending, *req_pending})
    return groups, implemented, pending, per_group


def build_replay_day_frame(
    data_dir: str,
    replay_config: dict[str, Any],
    date_str: str,
    *,
    expiry_hint: str | None,
    target: str = "",
    underlying: str = "NIFTY",
    expiry_resolution: dict[str, Any] | None = None,
    parallel_features: bool = False,
    required_features: list[str] | None = None,
    on_step_progress: Callable[[str, str], None] | None = None,
    on_feature_group_start: Callable[[str, str], None] | None = None,
    on_feature_group_progress: Callable[[str, int, int], None] | None = None,
    on_feature_group_done: Callable[[str], None] | None = None,
    inference_only: bool = False,
    only_timestamp: float | None = None,
    include_tokens: frozenset[str] | None = None,
    token_only: bool = False,
    enrich_tokens_only: frozenset[str] | None = None,
    day_context: Any | None = None,
    trim_target_rows: bool = True,
    gap_max_sec: float | None = None,
    gap_profile: bool = False,
    readiness_profile: bool = False,
    performance_debug_level: Any = None,
    performance_debug: Any = None,
    production_parity: bool = False,
) -> tuple[pd.DataFrame, str | None, dict[str, Any], dict[str, Any]]:
    """Build one replay day from tick DB. Returns (frame, error, expiry_resolution, build_stats)."""
    import time

    def step(name: str, detail: str = "") -> None:
        if on_step_progress:
            on_step_progress(name, detail)

    timing: dict[str, float] = {}
    chart_dir = chart_dir_from_data_dir(data_dir)
    market = str(replay_config.get("market") or underlying or "NIFTY")
    if expiry_resolution is None:
        expiry_resolution = resolve_scoring_expiry(
            chart_dir, date_str, expiry_hint, underlying=market,
        )
    resolved_expiry = str(expiry_resolution.get("resolved_expiry") or "").strip()
    if not resolved_expiry:
        return pd.DataFrame(), "No expiry with option ticks found for this replay day.", expiry_resolution, {}

    source_info = resolve_replay_source_spec(replay_config, date_str, resolved_expiry)
    if not source_info:
        return pd.DataFrame(), "Replay expiry is required to generate features from ticks.", expiry_resolution, {}

    tick_db = replay_db_path(chart_dir, date_str)
    if not tick_db or not os.path.isfile(tick_db):
        return pd.DataFrame(), f"No tick database for {date_str} (expected angel_market_{date_str}.db).", expiry_resolution, {}

    source = SourceSpec(
        source_id=source_info["source_id"],
        trading_day=source_info["trading_day"],
        market=source_info["market"],
        expiry=source_info["expiry"],
    )
    cfg = replay_config.get("dataset_configuration") or {}
    sampling = replay_config.get("sampling") or {}
    step_sec = int(
        sampling.get("interval_sec")
        or sampling.get("trainingIntervalSec")
        or cfg.get("sampling_interval_sec")
        or 10
    )
    t0 = time.perf_counter()
    if day_context is not None:
        ctx = day_context
        step("load_day_context", f"Reusing market state ({ctx.spot_ticks:,} spot / {ctx.chain_ticks:,} chain ticks)")
        timing["load_day_context"] = 0.0
    else:
        step("load_day_context", f"Loading option chain for {date_str}")
        try:
            ctx = load_day_context(chart_dir, source, feature_grid_step_sec=step_sec)
        except Exception as exc:
            return pd.DataFrame(), f"Could not load replay chain for {date_str} / expiry {source_info['expiry']}: {exc}", expiry_resolution, {}
        timing["load_day_context"] = round(time.perf_counter() - t0, 3)
        step("load_day_context", f"Chain loaded ({timing['load_day_context']}s)")

    timeline_key = replay_timeline_cache_key(data_dir, date_str, resolved_expiry)
    set_cached_token_timelines(timeline_key, token_timelines_from_day_context(ctx))

    from chain_replay_ml.dataset_builder.tick_coverage import sync_feature_grid_step
    sync_feature_grid_step(ctx, step_sec)

    horizons_sec = [] if inference_only else horizons_sec_from_replay_config(replay_config, target)

    registry = load_feature_registry()
    enabled_groups = list(
        replay_config.get("feature_groups_implemented") or replay_config.get("feature_groups") or []
    )
    enabled_groups, implemented, pending, per_group = merge_replay_feature_build_plan(
        enabled_groups,
        registry,
        required_features,
    )
    active_compute: frozenset[str] | None = None
    if required_features:
        req_set = frozenset(required_features) | frozenset(SCORING_INFRA_COLUMNS)
        active_compute = req_set
        filtered_groups: list[str] = []
        filtered_per_group: dict[str, list[str]] = {}
        for gid in enabled_groups:
            feats = [f for f in per_group.get(gid, []) if f in req_set]
            if feats:
                filtered_groups.append(gid)
                filtered_per_group[gid] = feats
        if filtered_groups:
            enabled_groups = filtered_groups
            per_group = filtered_per_group
            implemented = [f for f in implemented if f in req_set]
    group_labels = {
        gid: str((registry.get("groups") or {}).get(gid, {}).get("label") or gid)
        for gid in enabled_groups
    }
    lb_policy_doc = replay_config.get("lookback_policy") or cfg.get("lookback_policy") or {}
    strike_selection = dict(replay_config.get("strike_selection") or {})

    parallel_mode = "token" if parallel_features else "serial"
    group_ids_list = [g for g in enabled_groups if g in per_group]
    total_groups = len(group_ids_list)
    group_counter = {"i": 0}

    def on_group_start(gid: str, label: str) -> None:
        group_counter["i"] += 1
        step(
            "build_day_rows",
            f"Feature group {group_counter['i']}/{total_groups}: {label}",
        )
        if on_feature_group_start:
            on_feature_group_start(gid, label)

    def on_group_progress(label: str, current: int, total: int) -> None:
        if total <= 0:
            return
        if current == total or current % max(1, min(500, total // 20 or 1)) == 0:
            step("build_day_rows", f"{label} · {current:,}/{total:,} rows")
        if on_feature_group_progress:
            on_feature_group_progress(label, current, total)

    def on_prep_progress(step_name: str, detail: str) -> None:
        step(step_name, detail)

    def _on_group_done(gid: str) -> None:
        if on_feature_group_done:
            on_feature_group_done(gid)

    from chain_replay_ml.feature_policy.performance_debug import PerformanceDebugConfig

    perf = PerformanceDebugConfig.resolve(
        performance_debug_level,
        gap_profile=gap_profile,
        readiness_profile=readiness_profile,
        config=performance_debug,
    )

    t0 = time.perf_counter()
    use_group_hooks = any((
        on_step_progress,
        on_feature_group_start,
        on_feature_group_progress,
        on_feature_group_done,
    ))
    if production_parity:
        from chain_replay_ml.dataset_builder.production_day_build import build_production_day_rows

        rows, stats = build_production_day_rows(
            ctx,
            step_sec=step_sec,
            strike_selection=strike_selection,
            horizons_sec=horizons_sec,
            enabled_groups=enabled_groups,
            group_labels=group_labels,
            implemented_features=implemented,
            per_group_features=per_group,
            lookback_policy_doc=lb_policy_doc,
            on_group_start=on_group_start if use_group_hooks else None,
            on_group_progress=on_group_progress if use_group_hooks else None,
            on_group_done=_on_group_done if on_feature_group_done else None,
            on_prep_progress=on_prep_progress if on_step_progress else None,
            gap_max_sec=gap_max_sec,
            trim_target_rows=trim_target_rows,
            performance_debug=perf,
        )
    else:
        rows, stats = build_day_rows(
            ctx,
            step_sec=step_sec,
            strike_selection=strike_selection,
            horizons_sec=horizons_sec,
            enabled_groups=enabled_groups,
            group_labels=group_labels,
            implemented_features=implemented,
            per_group_features=per_group,
            lookback_policy_doc=lb_policy_doc,
            parallel_mode=parallel_mode,
            active_features=active_compute,
            on_group_start=on_group_start if use_group_hooks else None,
            on_group_progress=on_group_progress if use_group_hooks else None,
            on_group_done=_on_group_done if on_feature_group_done else None,
            on_prep_progress=on_prep_progress if on_step_progress else None,
            only_timestamp=only_timestamp,
            include_tokens=include_tokens,
            token_only=token_only,
            enrich_tokens_only=enrich_tokens_only,
            trim_target_rows=trim_target_rows,
            gap_max_sec=gap_max_sec,
            gap_profile=gap_profile,
            readiness_profile=readiness_profile,
            performance_debug_level=performance_debug_level,
            performance_debug=performance_debug,
        )
    timing["build_day_rows"] = round(time.perf_counter() - t0, 3)
    step("build_day_rows", f"Built {len(rows):,} rows ({timing['build_day_rows']}s)")
    if not rows:
        trimmed = int(stats.get("target_trimmed_rows") or 0)
        hint = f" ({trimmed} rows dropped for missing future targets)" if trimmed else ""
        return pd.DataFrame(), f"No feature rows generated for {date_str}{hint}.", expiry_resolution, stats

    t0 = time.perf_counter()
    step("to_dataframe", f"Converting {len(rows):,} rows to dataframe")
    df = pd.DataFrame(rows)
    timing["to_dataframe"] = round(time.perf_counter() - t0, 3)
    if "trading_day" not in df.columns:
        df["trading_day"] = str(date_str)
    if pending:
        stats = dict(stats or {})
        stats["pending_registry_features"] = pending
    stats = dict(stats or {})
    stats["implemented_feature_count"] = len(implemented)
    stats["timing_sec"] = timing
    stats["feature_count"] = len(implemented)
    stats["row_count"] = len(df)
    return df, None, expiry_resolution, stats


def load_model_inference_config(data_dir: str, model_name: str) -> dict[str, Any] | None:
    from chain_replay_ml.training.model_runtime import normalize_algorithm, resolve_production_model_path
    from chain_replay_ml.training.paths import model_artifact_paths
    from chain_replay_ml.training.registry import get_trained_model

    row = get_trained_model(data_dir, model_name)
    if not row:
        return None
    paths = model_artifact_paths(data_dir, model_name)
    if not os.path.isfile(paths["config_json"]):
        return None
    production_name = ""
    if os.path.isfile(paths.get("training_metadata_json", "")):
        try:
            with open(paths["training_metadata_json"], encoding="utf-8") as fh:
                tmeta = json.load(fh)
            production_name = str(tmeta.get("production_model") or "").strip()
        except Exception:
            pass
    with open(paths["config_json"], encoding="utf-8") as fh:
        config = json.load(fh)
    algorithm = normalize_algorithm(config.get("algorithm"))
    model_path = resolve_production_model_path(
        paths["package_dir"],
        algorithm=algorithm,
        production_name=production_name or None,
    )
    if not model_path:
        return None
    features = list(config.get("features") or [])
    target = str(config.get("target") or "")
    if not features or not target:
        return None
    replay_config, replay_config_source = resolve_replay_feature_config(data_dir, config)
    return {
        "config": config,
        "dataset": str(config.get("dataset") or ""),
        "target": target,
        "features": features,
        "algorithm": algorithm,
        "model_path": model_path,
        "replay_config": replay_config,
        "replay_config_source": replay_config_source,
    }


def replay_scoring_coverage(
    data_dir: str,
    model_name: str,
    date_str: str,
    *,
    expiry_hint: str | None = None,
    underlying: str = "NIFTY",
    expiry_resolution: dict[str, Any] | None = None,
) -> dict[str, Any]:
    loaded = load_model_inference_config(data_dir, model_name)
    if not loaded:
        return {"ok": False, "reason": "model_not_found", "replay_day": date_str}

    chart_dir = chart_dir_from_data_dir(data_dir)
    tick_db = replay_db_path(chart_dir, date_str)
    replay_config = loaded.get("replay_config") or {}
    market = str(replay_config.get("market") or underlying or "NIFTY")
    if expiry_resolution is None:
        expiry_resolution = resolve_scoring_expiry(
            chart_dir, date_str, expiry_hint, underlying=market,
        )
    resolved = str(expiry_resolution.get("resolved_expiry") or "").strip()
    can_score = bool(tick_db and os.path.isfile(tick_db) and resolved and replay_config)
    error = None
    if not replay_config:
        error = "No replay feature config on model package or linked dataset metadata."
    elif not tick_db or not os.path.isfile(tick_db):
        error = f"No tick database for {date_str}."
    elif not resolved:
        ticks = expiry_resolution.get("expiries_with_ticks") or []
        if ticks:
            error = f"No option ticks for requested expiry; available: {', '.join(ticks[:4])}."
        else:
            error = "No expiry with option ticks found for this replay day."

    return {
        "ok": True,
        "replay_day": str(date_str),
        "model_name": model_name,
        "training_dataset": loaded.get("dataset") or "",
        "replay_config_source": loaded.get("replay_config_source"),
        "tick_db_available": bool(tick_db and os.path.isfile(tick_db)),
        "tick_db_path": tick_db or "",
        "resolved_expiry": resolved or None,
        "expiry_resolution": expiry_resolution,
        "can_score": can_score,
        "error": error,
        "source": "live_ticks",
        "feature_count": len(loaded.get("features") or []),
    }


def load_scoring_day_frame(
    data_dir: str,
    model_name: str,
    date_str: str,
    *,
    expiry_hint: str | None = None,
    allow_dataset_fallback: bool = False,
    underlying: str = "NIFTY",
    expiry_resolution: dict[str, Any] | None = None,
    parallel_features: bool = False,
    on_step_progress: Callable[[str, str], None] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    """Primary path: tick DB feature generation. Dataset parquet is optional debug fallback."""
    loaded = load_model_inference_config(data_dir, model_name)
    empty_cov: dict[str, Any] = {"source": "none", "engine": "replay_ticks"}
    if not loaded:
        return pd.DataFrame(), {}, empty_cov

    config = loaded["config"]
    replay_config = loaded.get("replay_config") or {}
    market = str(replay_config.get("market") or underlying or "NIFTY")
    if expiry_resolution is None:
        expiry_resolution = resolve_scoring_expiry(
            chart_dir_from_data_dir(data_dir), date_str, expiry_hint, underlying=market,
        )
    resolved_expiry = expiry_resolution.get("resolved_expiry")
    cache_key = replay_cache_key(data_dir, model_name, date_str, str(resolved_expiry or ""))
    coverage = replay_scoring_coverage(
        data_dir,
        model_name,
        date_str,
        expiry_hint=expiry_hint,
        underlying=market,
        expiry_resolution=expiry_resolution,
    )
    coverage["engine"] = "replay_ticks"

    cached_day = get_cached_day_frame(cache_key)
    if cached_day is not None and not cached_day.empty:
        coverage["source"] = "live_ticks"
        coverage["cache_hit"] = coverage.get("cache_hit") or "day_frame"
        if on_step_progress:
            on_step_progress("build_day_rows", f"Using cached feature rows ({len(cached_day):,} rows)")
        return cached_day, config, coverage

    day_df, err, expiry_resolution, build_stats = build_replay_day_frame(
        data_dir,
        replay_config,
        date_str,
        expiry_hint=expiry_hint,
        target=loaded["target"],
        underlying=market,
        expiry_resolution=expiry_resolution,
        parallel_features=parallel_features,
        required_features=list(loaded.get("features") or []),
        on_step_progress=on_step_progress,
    )
    coverage["expiry_resolution"] = expiry_resolution
    coverage["resolved_expiry"] = expiry_resolution.get("resolved_expiry")
    if build_stats:
        coverage["feature_build"] = build_stats
    source = "live_ticks" if not day_df.empty else "none"
    if err:
        coverage["error"] = err

    if not day_df.empty:
        set_cached_day_frame(cache_key, day_df)

    if day_df.empty and allow_dataset_fallback:
        from chain_replay_ml.training.dataset_loader import load_dataset_frame

        dataset = loaded.get("dataset") or ""
        try:
            df, _, _ = load_dataset_frame(data_dir, dataset)
            if "trading_day" in df.columns:
                fallback = df[df["trading_day"].astype(str) == str(date_str)].copy()
                if not fallback.empty:
                    day_df = fallback
                    source = "dataset_fallback"
                    coverage["fallback_warning"] = (
                        "Using training-dataset parquet fallback — for debugging/validation only."
                    )
        except Exception as exc:
            coverage["dataset_fallback_error"] = str(exc)

    coverage["source"] = source
    return day_df, config, coverage


def scoring_required_columns(features: list[str], target: str, *, require_target: bool) -> list[str]:
    base = list(features) + ["ltp", "token", "timestamp", "delta", "symbol", "option_type"]
    if require_target:
        base.append(target)
    return base

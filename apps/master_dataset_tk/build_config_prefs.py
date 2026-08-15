"""Persist Create Dataset build config and Feature Policy UI selections."""

from __future__ import annotations

import json
import os
import time
from typing import Any

STORAGE = "ml_master_build_config_tk.json"
STANDARD_SIMULATOR_DURATION_PRESETS = frozenset({5, 10, 15})


def resolve_simulator_duration_minutes(*, preset_minutes: int, custom_minutes: str) -> int:
    """Runtime duration; preset radio value 0 means read *custom_minutes* text."""
    if int(preset_minutes) == 0:
        try:
            return max(1, int(str(custom_minutes).strip()))
        except ValueError:
            return 20
    return max(1, int(preset_minutes))


def simulator_duration_preset_for_save(preset_minutes: int) -> int:
    """Persist preset radio as-is; 0 = custom mode (must not use ``or 15``)."""
    return int(preset_minutes)


def infer_simulator_duration_preset(saved_preset: Any, custom_minutes: str) -> int:
    """Restore radio selection; recover custom mode when prefs stored preset 15 by mistake."""
    try:
        preset = int(saved_preset) if saved_preset is not None else 15
    except (TypeError, ValueError):
        preset = 15
    if preset == 0:
        return 0
    if preset in STANDARD_SIMULATOR_DURATION_PRESETS:
        return preset
    custom = str(custom_minutes or "").strip()
    if not custom:
        return preset
    try:
        custom_val = int(custom)
    except ValueError:
        return preset
    if custom_val not in STANDARD_SIMULATOR_DURATION_PRESETS and custom_val != preset:
        return 0
    return preset


def simulator_duration_prefs_for_save(*, preset_minutes: int, custom_minutes: str) -> dict[str, Any]:
    """Persist duration radio + custom field without stale custom confusing reload."""
    preset = int(preset_minutes)
    custom = str(custom_minutes or "").strip()
    if preset != 0:
        return {
            "duration_minutes": simulator_duration_preset_for_save(preset),
            "custom_duration": "",
        }
    return {
        "duration_minutes": 0,
        "custom_duration": custom,
    }


def storage_path(chart_dir: str) -> str:
    """Build-config prefs live next to master DBs when master_data_dir is set."""
    try:
        from .project_config import load_project_config, resolve_master_data_dir

        if str(load_project_config().get("master_data_dir") or "").strip() or str(
            os.environ.get("ARUNEO_MASTER_DATA_DIR") or ""
        ).strip():
            return os.path.join(resolve_master_data_dir(chart_dir), STORAGE)
    except Exception:
        pass
    return os.path.join(chart_dir, "data", STORAGE)


def load_build_config_prefs(chart_dir: str) -> dict[str, Any] | None:
    path = storage_path(chart_dir)
    if not chart_dir or not os.path.isfile(path):
        return None
    try:
        with open(path, encoding="utf-8") as fh:
            doc = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None
    return doc if isinstance(doc, dict) else None


def save_build_config_prefs(chart_dir: str, patch: dict[str, Any]) -> dict[str, Any]:
    if not chart_dir:
        return {}
    existing = load_build_config_prefs(chart_dir) or {}
    doc = dict(existing)
    for key, val in patch.items():
        if isinstance(val, dict) and isinstance(doc.get(key), dict):
            doc[key] = {**doc[key], **val}
        else:
            doc[key] = val
    doc["version"] = 1
    doc["at"] = int(time.time() * 1000)
    path = storage_path(chart_dir)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=2)
    return doc


def save_chain_source_prefs(
    chart_dir: str,
    *,
    market: str,
    selected_source_ids: list[str],
) -> dict[str, Any]:
    return save_build_config_prefs(chart_dir, {
        "chain_sources": {
            "market": str(market or "NIFTY").strip().upper(),
            "selected_source_ids": sorted({str(s) for s in selected_source_ids if s}),
        },
    })


def load_chain_source_prefs(chart_dir: str) -> dict[str, Any]:
    prefs = load_build_config_prefs(chart_dir) or {}
    doc = prefs.get("chain_sources")
    return doc if isinstance(doc, dict) else {}


def load_master_data_prefs(chart_dir: str) -> dict[str, Any]:
    prefs = load_build_config_prefs(chart_dir) or {}
    doc = prefs.get("master_data")
    return doc if isinstance(doc, dict) else {}


def save_master_data_prefs(chart_dir: str, patch: dict[str, Any]) -> dict[str, Any]:
    return save_build_config_prefs(chart_dir, {"master_data": patch})


def load_auto_feature_transform_prefs(chart_dir: str) -> dict[str, Any]:
    """Feature Transformation → Auto → Build Configuration prefs."""
    prefs = load_build_config_prefs(chart_dir) or {}
    doc = prefs.get("auto_feature_transform")
    return doc if isinstance(doc, dict) else {}


def save_auto_feature_transform_prefs(
    chart_dir: str,
    patch: dict[str, Any],
) -> dict[str, Any]:
    return save_build_config_prefs(chart_dir, {"auto_feature_transform": patch})


def load_feature_project_prefs(chart_dir: str) -> dict[str, Any]:
    """Shared Feature Project selection for Project Manager + Feature Transformations."""
    prefs = load_build_config_prefs(chart_dir) or {}
    doc = prefs.get("feature_project")
    return doc if isinstance(doc, dict) else {}


def save_feature_project_prefs(chart_dir: str, patch: dict[str, Any]) -> dict[str, Any]:
    return save_build_config_prefs(chart_dir, {"feature_project": patch})


def active_feature_project_id(chart_dir: str) -> str:
    from chain_replay_ml.dataset_builder.feature_project_organization import RESERVED_ALL_PROJECT_ID

    doc = load_feature_project_prefs(chart_dir)
    pid = str(doc.get("selected_project_id") or RESERVED_ALL_PROJECT_ID).strip().lower()
    return pid or RESERVED_ALL_PROJECT_ID


def set_active_feature_project_id(chart_dir: str, project_id: str) -> None:
    pid = str(project_id or "").strip().lower()
    if pid:
        save_feature_project_prefs(chart_dir, {"selected_project_id": pid})


def _normalize_day_scope(day_scope: Any, *, all_days: bool) -> str:
    scope = str(day_scope or "").strip().lower()
    if scope not in ("all", "selected"):
        scope = "all" if all_days else "selected"
    return scope


def auto_feature_transform_prefs_snapshot(
    *,
    market: str,
    interval_sec: int,
    include_registry: bool,
    include_pipeline: bool,
    all_days: bool,
    day_scope: str | None = None,
    selected_days: list[str] | set[str] | None = None,
    no_null_data: bool,
    pipeline_no_null_report: bool,
    premium_enabled: bool,
    premium_min: str | float,
    premium_max: str | float,
    target_pipeline_id: str | None = None,
    target_pipeline_mode: str | None = None,
    build_pipeline_id: str | None = None,
) -> dict[str, Any]:
    try:
        interval = max(1, int(interval_sec))
    except (TypeError, ValueError):
        interval = 3
    scope = _normalize_day_scope(day_scope, all_days=bool(all_days))
    return {
        "market": str(market or "NIFTY").strip().upper() or "NIFTY",
        "interval_sec": interval,
        "include_registry": bool(include_registry),
        "include_pipeline": bool(include_pipeline),
        # "all_days" kept for backward compatibility with older prefs files;
        # "day_scope" is the source of truth for All days vs Selected days.
        "all_days": scope == "all",
        "day_scope": scope,
        "selected_days": sorted({str(d).strip() for d in (selected_days or []) if str(d).strip()}),
        "no_null_data": bool(no_null_data),
        "pipeline_no_null_report": bool(pipeline_no_null_report),
        "premium_enabled": bool(premium_enabled),
        "premium_min": str(premium_min if premium_min is not None else "15"),
        "premium_max": str(premium_max if premium_max is not None else "40"),
        "target_pipeline_id": str(target_pipeline_id or "").strip().upper(),
        "target_pipeline_mode": (
            "create_new"
            if str(target_pipeline_mode or "").strip().lower() == "create_new"
            else "existing"
        ),
        "build_pipeline_id": str(build_pipeline_id or "").strip().upper(),
    }


def apply_auto_feature_transform_prefs(
    prefs: dict[str, Any] | None,
) -> dict[str, Any]:
    """Normalize stored prefs into values suitable for the Auto Build UI."""
    src = prefs if isinstance(prefs, dict) else {}
    market = str(src.get("market") or "NIFTY").strip().upper() or "NIFTY"
    if market not in {"NIFTY", "BANKNIFTY", "SENSEX"}:
        market = "NIFTY"
    try:
        interval = int(src.get("interval_sec") or 3)
    except (TypeError, ValueError):
        interval = 3
    if interval < 1:
        interval = 3
    all_days = bool(src.get("all_days", True))
    scope = _normalize_day_scope(src.get("day_scope"), all_days=all_days)
    selected_days = sorted({str(d).strip() for d in (src.get("selected_days") or []) if str(d).strip()})
    from .auto_candidate_generation import normalize_candidate_generation_prefs

    return {
        "market": market,
        "interval_sec": interval,
        "include_registry": bool(src.get("include_registry", True)),
        "include_pipeline": bool(src.get("include_pipeline", True)),
        "all_days": scope == "all",
        "day_scope": scope,
        "selected_days": selected_days,
        "no_null_data": bool(src.get("no_null_data", False)),
        "pipeline_no_null_report": bool(src.get("pipeline_no_null_report", False)),
        "premium_enabled": bool(src.get("premium_enabled", True)),
        "premium_min": str(src.get("premium_min") if src.get("premium_min") is not None else "15"),
        "premium_max": str(src.get("premium_max") if src.get("premium_max") is not None else "40"),
        "target_pipeline_id": str(src.get("target_pipeline_id") or "").strip().upper(),
        "target_pipeline_mode": (
            "create_new"
            if str(src.get("target_pipeline_mode") or "").strip().lower() == "create_new"
            else "existing"
        ),
        "build_pipeline_id": str(src.get("build_pipeline_id") or "").strip().upper(),
        "candidate_generation": normalize_candidate_generation_prefs(
            src.get("candidate_generation") if isinstance(src.get("candidate_generation"), dict) else None
        ),
    }


def resolve_master_data_interval_sec(
    prefs: dict[str, Any] | None,
    *,
    build_prefs: dict[str, Any] | None = None,
    default: int = 10,
    allowed: tuple[int, ...] = (3, 6, 9, 10, 15, 30, 60),
) -> int:
    """Interval for Trading Days panel — master_data prefs, then build config, then default."""
    for source in (prefs, build_prefs):
        if not isinstance(source, dict):
            continue
        try:
            val = int(source.get("interval_sec") or source.get("sampling_interval_sec") or 0)
        except (TypeError, ValueError):
            continue
        if val in allowed:
            return val
    return default if default in allowed else allowed[0]


def master_data_prefs_snapshot(
    *,
    market: str,
    interval_sec: int,
    selected_days: set[str] | list[str],
    day_selection_explicit: bool,
    atm_band: str,
    premium_enabled: bool,
    premium_min: str,
    premium_max: str,
    audit_required: bool,
    no_null_data: bool = False,
    no_null_filter_report: bool = False,
    trading_day_filter: str = "all",
    lag_enabled: bool = False,
    lag_features: list[str] | None = None,
    lag_seconds: list[int] | None = None,
    difference_enabled: bool = False,
    difference_features: list[str] | None = None,
    difference_seconds: list[int] | None = None,
    return_enabled: bool = False,
    return_features: list[str] | None = None,
    return_seconds: list[int] | None = None,
    rolling_enabled: bool = False,
    rolling_features: list[str] | None = None,
    rolling_windows: list[int] | None = None,
    rolling_operations: list[str] | None = None,
    exponential_rolling_enabled: bool = False,
    exponential_rolling_features: list[str] | None = None,
    exponential_rolling_periods: list[int] | None = None,
    exponential_rolling_operations: list[str] | None = None,
    ohlc_aggregation_enabled: bool = False,
    ohlc_aggregation_features: list[str] | None = None,
    ohlc_aggregation_timeframes: list[str] | None = None,
    ohlc_aggregation_outputs: list[str] | None = None,
    interaction_enabled: bool = False,
    interaction_pairs: list[dict[str, Any]] | None = None,
    interaction_pipeline_pairs: list[dict[str, Any]] | None = None,
    normalization_enabled: bool = False,
    normalization_features: list[str] | None = None,
    normalization_methods: list[str] | None = None,
    normalization_windows: list[int] | None = None,
    regime_enabled: bool = False,
    regime_features: list[str] | None = None,
    regime_methods: list[str] | None = None,
    regime_windows: list[int] | None = None,
    regime_threshold: str = "0",
    regime_low: str = "-1",
    regime_high: str = "1",
    regime_n_bins: str = "5",
    math_enabled: bool = False,
    math_features: list[str] | None = None,
    math_operations: list[str] | None = None,
    math_clip_min: str = "0",
    math_clip_max: str = "",
) -> dict[str, Any]:
    mode = str(trading_day_filter or "all").strip().lower() or "all"
    if mode not in ("all", "exclude_expiry", "expiry_only"):
        mode = "all"
    wins: list[int] = []
    for w in rolling_windows or []:
        try:
            wi = int(w)
        except (TypeError, ValueError):
            continue
        if wi > 0 and wi not in wins:
            wins.append(wi)
    wins.sort()
    ops: list[str] = []
    for op in rolling_operations or []:
        name = str(op or "").strip().lower()
        if name and name not in ops:
            ops.append(name)
    exp_pers: list[int] = []
    for p in exponential_rolling_periods or []:
        try:
            pi = int(p)
        except (TypeError, ValueError):
            continue
        if pi > 0 and pi not in exp_pers:
            exp_pers.append(pi)
    exp_pers.sort()
    exp_ops: list[str] = []
    for op in exponential_rolling_operations or []:
        name = str(op or "").strip().lower()
        if name and name not in exp_ops:
            exp_ops.append(name)
    norm_wins: list[int] = []
    for w in normalization_windows or []:
        try:
            wi = int(w)
        except (TypeError, ValueError):
            continue
        if wi > 0 and wi not in norm_wins:
            norm_wins.append(wi)
    norm_wins.sort()
    norm_meths: list[str] = []
    for m in normalization_methods or []:
        name = str(m or "").strip().lower()
        if name and name not in norm_meths:
            norm_meths.append(name)
    reg_wins: list[int] = []
    for w in regime_windows or []:
        try:
            wi = int(w)
        except (TypeError, ValueError):
            continue
        if wi > 0 and wi not in reg_wins:
            reg_wins.append(wi)
    reg_wins.sort()
    reg_meths: list[str] = []
    for m in regime_methods or []:
        name = str(m or "").strip().lower()
        if name and name not in reg_meths:
            reg_meths.append(name)
    math_ops: list[str] = []
    for op in math_operations or []:
        name = str(op or "").strip().lower()
        if name and name not in math_ops:
            math_ops.append(name)
    def _clean_feats(raw: list[str] | None) -> list[str]:
        out: list[str] = []
        for feat in raw or []:
            name = str(feat or "").strip()
            if name and name not in out:
                out.append(name)
        return out

    def _clean_secs(raw: list[int] | None) -> list[int]:
        out: list[int] = []
        for s in raw or []:
            try:
                si = int(s)
            except (TypeError, ValueError):
                continue
            if si > 0 and si not in out:
                out.append(si)
        return out

    ohlc_feats: list[str] = []
    for feat in ohlc_aggregation_features or []:
        name = str(feat or "").strip()
        if name and name not in ohlc_feats:
            ohlc_feats.append(name)
    ohlc_tfs: list[str] = []
    for tf in ohlc_aggregation_timeframes or []:
        name = str(tf or "").strip().lower()
        if name and name not in ohlc_tfs:
            ohlc_tfs.append(name)
    ohlc_outs: list[str] = []
    for fld in ohlc_aggregation_outputs or []:
        name = str(fld or "").strip().lower()
        if name and name not in ohlc_outs:
            ohlc_outs.append(name)
    return {
        "market": str(market or "NIFTY").strip().upper(),
        "interval_sec": int(interval_sec),
        "selected_days": sorted({str(d).strip() for d in selected_days if str(d).strip()}),
        "day_selection_explicit": bool(day_selection_explicit),
        "atm_band": str(atm_band or ""),
        "premium_enabled": bool(premium_enabled),
        "premium_min": str(premium_min or ""),
        "premium_max": str(premium_max or ""),
        "audit_required": bool(audit_required),
        "no_null_data": bool(no_null_data),
        "no_null_filter_report": bool(no_null_filter_report),
        "trading_day_filter": mode,
        "lag_enabled": bool(lag_enabled),
        "lag_features": _clean_feats(lag_features),
        "lag_seconds": _clean_secs(lag_seconds),
        "difference_enabled": bool(difference_enabled),
        "difference_features": _clean_feats(difference_features),
        "difference_seconds": _clean_secs(difference_seconds),
        "return_enabled": bool(return_enabled),
        "return_features": _clean_feats(return_features),
        "return_seconds": _clean_secs(return_seconds),
        "rolling_enabled": bool(rolling_enabled),
        "rolling_features": _clean_feats(rolling_features),
        "rolling_windows": wins,
        "rolling_operations": ops,
        "exponential_rolling_enabled": bool(exponential_rolling_enabled),
        "exponential_rolling_features": _clean_feats(exponential_rolling_features),
        "exponential_rolling_periods": exp_pers,
        "exponential_rolling_operations": exp_ops,
        "ohlc_aggregation_enabled": bool(ohlc_aggregation_enabled),
        "ohlc_aggregation_features": ohlc_feats,
        "ohlc_aggregation_timeframes": ohlc_tfs,
        "ohlc_aggregation_outputs": ohlc_outs,
        "interaction_enabled": bool(interaction_enabled),
        "interaction_pairs": [
            dict(p) for p in (interaction_pairs or []) if isinstance(p, dict)
        ],
        "interaction_pipeline_pairs": [
            dict(p) for p in (interaction_pipeline_pairs or []) if isinstance(p, dict)
        ],
        "normalization_enabled": bool(normalization_enabled),
        "normalization_features": _clean_feats(normalization_features),
        "normalization_methods": norm_meths,
        "normalization_windows": norm_wins,
        "regime_enabled": bool(regime_enabled),
        "regime_features": _clean_feats(regime_features),
        "regime_methods": reg_meths,
        "regime_windows": reg_wins,
        "regime_threshold": str(regime_threshold or "0"),
        "regime_low": str(regime_low or "-1"),
        "regime_high": str(regime_high or "1"),
        "regime_n_bins": str(regime_n_bins or "5"),
        "math_enabled": bool(math_enabled),
        "math_features": _clean_feats(math_features),
        "math_operations": math_ops,
        "math_clip_min": str(math_clip_min or "0"),
        "math_clip_max": str(math_clip_max or ""),
    }


def apply_master_data_prefs(
    prefs: dict[str, Any] | None,
    *,
    build_prefs: dict[str, Any] | None = None,
    allowed_markets: tuple[str, ...] = ("NIFTY", "BANKNIFTY", "SENSEX"),
    allowed_intervals: tuple[int, ...] = (3, 6, 9, 10, 15, 30, 60),
) -> dict[str, Any]:
    """Normalize persisted Trading Days panel state for UI restore."""
    src = prefs if isinstance(prefs, dict) else {}
    build = build_prefs if isinstance(build_prefs, dict) else {}
    market = str(src.get("market") or "NIFTY").strip().upper()
    if market not in allowed_markets:
        market = "NIFTY"
    interval_sec = resolve_master_data_interval_sec(
        src,
        build_prefs=build,
        allowed=allowed_intervals,
    )
    selected_days: list[str] = []
    raw_days = src.get("selected_days")
    if isinstance(raw_days, list):
        selected_days = sorted({str(d).strip() for d in raw_days if str(d).strip()})
    mode = str(src.get("trading_day_filter") or "all").strip().lower() or "all"
    if mode not in ("all", "exclude_expiry", "expiry_only"):
        mode = "all"
    lags: list[int] = []
    raw_lags = src.get("lag_seconds")
    if isinstance(raw_lags, list):
        for s in raw_lags:
            try:
                v = int(s)
            except (TypeError, ValueError):
                continue
            if v > 0:
                lags.append(v)
    roll_wins: list[int] = []
    raw_roll_wins = src.get("rolling_windows")
    if isinstance(raw_roll_wins, list):
        for w in raw_roll_wins:
            try:
                wi = int(w)
            except (TypeError, ValueError):
                continue
            if wi > 0 and wi not in roll_wins:
                roll_wins.append(wi)
        roll_wins.sort()
    roll_ops: list[str] = []
    raw_roll_ops = src.get("rolling_operations")
    if isinstance(raw_roll_ops, list):
        for op in raw_roll_ops:
            name = str(op or "").strip().lower()
            if name and name not in roll_ops:
                roll_ops.append(name)
    exp_pers: list[int] = []
    raw_exp_pers = src.get("exponential_rolling_periods")
    if isinstance(raw_exp_pers, list):
        for p in raw_exp_pers:
            try:
                pi = int(p)
            except (TypeError, ValueError):
                continue
            if pi > 0 and pi not in exp_pers:
                exp_pers.append(pi)
        exp_pers.sort()
    exp_ops: list[str] = []
    raw_exp_ops = src.get("exponential_rolling_operations")
    if isinstance(raw_exp_ops, list):
        for op in raw_exp_ops:
            name = str(op or "").strip().lower()
            if name and name not in exp_ops:
                exp_ops.append(name)
    elif raw_exp_ops is None and bool(src.get("exponential_rolling_enabled", False)):
        # Prefs saved before multi-op UI: preserve prior EMA-only behavior.
        exp_ops = ["ema"]
    def _load_feat_list(key: str) -> list[str]:
        out: list[str] = []
        raw = src.get(key)
        if isinstance(raw, list):
            for feat in raw:
                name = str(feat or "").strip()
                if name and name not in out:
                    out.append(name)
        return out

    def _load_sec_list(key: str) -> list[int]:
        out: list[int] = []
        raw = src.get(key)
        if isinstance(raw, list):
            for s in raw:
                try:
                    v = int(s)
                except (TypeError, ValueError):
                    continue
                if v > 0 and v not in out:
                    out.append(v)
        return out

    ohlc_feats: list[str] = []
    raw_ohlc_feats = src.get("ohlc_aggregation_features")
    if isinstance(raw_ohlc_feats, list):
        for feat in raw_ohlc_feats:
            name = str(feat or "").strip()
            if name and name not in ohlc_feats:
                ohlc_feats.append(name)
    ohlc_tfs: list[str] = []
    raw_ohlc_tfs = src.get("ohlc_aggregation_timeframes")
    if isinstance(raw_ohlc_tfs, list):
        for tf in raw_ohlc_tfs:
            name = str(tf or "").strip().lower()
            if name and name not in ohlc_tfs:
                ohlc_tfs.append(name)
    ohlc_outs: list[str] = []
    raw_ohlc_outs = src.get("ohlc_aggregation_outputs")
    if isinstance(raw_ohlc_outs, list):
        for fld in raw_ohlc_outs:
            name = str(fld or "").strip().lower()
            if name and name not in ohlc_outs:
                ohlc_outs.append(name)
    norm_wins_apply: list[int] = []
    raw_norm_wins = src.get("normalization_windows")
    if isinstance(raw_norm_wins, list):
        for w in raw_norm_wins:
            try:
                wi = int(w)
            except (TypeError, ValueError):
                continue
            if wi > 0 and wi not in norm_wins_apply:
                norm_wins_apply.append(wi)
        norm_wins_apply.sort()
    norm_meths_apply: list[str] = []
    raw_norm_meths = src.get("normalization_methods")
    if isinstance(raw_norm_meths, list):
        for m in raw_norm_meths:
            name = str(m or "").strip().lower()
            if name and name not in norm_meths_apply:
                norm_meths_apply.append(name)
    reg_wins_apply: list[int] = []
    raw_reg_wins = src.get("regime_windows")
    if isinstance(raw_reg_wins, list):
        for w in raw_reg_wins:
            try:
                wi = int(w)
            except (TypeError, ValueError):
                continue
            if wi > 0 and wi not in reg_wins_apply:
                reg_wins_apply.append(wi)
        reg_wins_apply.sort()
    reg_meths_apply: list[str] = []
    raw_reg_meths = src.get("regime_methods")
    if isinstance(raw_reg_meths, list):
        for m in raw_reg_meths:
            name = str(m or "").strip().lower()
            if name and name not in reg_meths_apply:
                reg_meths_apply.append(name)
    math_ops_apply: list[str] = []
    raw_math_ops = src.get("math_operations")
    if isinstance(raw_math_ops, list):
        for op in raw_math_ops:
            name = str(op or "").strip().lower()
            if name and name not in math_ops_apply:
                math_ops_apply.append(name)
    return {
        "market": market,
        "interval_sec": interval_sec,
        "selected_days": selected_days,
        "day_selection_explicit": bool(src.get("day_selection_explicit")),
        "atm_band": str(src.get("atm_band") if src.get("atm_band") is not None else "10"),
        "premium_enabled": bool(src.get("premium_enabled", True)),
        "premium_min": str(src.get("premium_min") if src.get("premium_min") is not None else "15"),
        "premium_max": str(src.get("premium_max") if src.get("premium_max") is not None else "40"),
        "audit_required": bool(src.get("audit_required", False)),
        "no_null_data": bool(src.get("no_null_data", False)),
        "no_null_filter_report": bool(src.get("no_null_filter_report", False)),
        "trading_day_filter": mode,
        "lag_enabled": bool(src.get("lag_enabled", False)),
        "lag_features": [
            str(f) for f in (src.get("lag_features") or []) if str(f).strip()
        ],
        "lag_seconds": lags,
        "difference_enabled": bool(src.get("difference_enabled", False)),
        "difference_features": _load_feat_list("difference_features"),
        "difference_seconds": _load_sec_list("difference_seconds"),
        "return_enabled": bool(src.get("return_enabled", False)),
        "return_features": _load_feat_list("return_features"),
        "return_seconds": _load_sec_list("return_seconds"),
        "rolling_enabled": bool(src.get("rolling_enabled", False)),
        "rolling_features": _load_feat_list("rolling_features"),
        "rolling_windows": roll_wins,
        "rolling_operations": roll_ops,
        "exponential_rolling_enabled": bool(src.get("exponential_rolling_enabled", False)),
        "exponential_rolling_features": _load_feat_list("exponential_rolling_features"),
        "exponential_rolling_periods": exp_pers,
        "exponential_rolling_operations": exp_ops,
        "ohlc_aggregation_enabled": bool(src.get("ohlc_aggregation_enabled", False)),
        "ohlc_aggregation_features": ohlc_feats,
        "ohlc_aggregation_timeframes": ohlc_tfs,
        "ohlc_aggregation_outputs": ohlc_outs,
        "interaction_enabled": bool(src.get("interaction_enabled", False)),
        "interaction_pairs": [
            dict(p) for p in (src.get("interaction_pairs") or []) if isinstance(p, dict)
        ],
        "interaction_pipeline_pairs": [
            dict(p) for p in (src.get("interaction_pipeline_pairs") or []) if isinstance(p, dict)
        ],
        "normalization_enabled": bool(src.get("normalization_enabled", False)),
        "normalization_features": _load_feat_list("normalization_features"),
        "normalization_methods": norm_meths_apply,
        "normalization_windows": norm_wins_apply,
        "regime_enabled": bool(src.get("regime_enabled", False)),
        "regime_features": _load_feat_list("regime_features"),
        "regime_methods": reg_meths_apply,
        "regime_windows": reg_wins_apply,
        "regime_threshold": str(src.get("regime_threshold") if src.get("regime_threshold") is not None else "0"),
        "regime_low": str(src.get("regime_low") if src.get("regime_low") is not None else "-1"),
        "regime_high": str(src.get("regime_high") if src.get("regime_high") is not None else "1"),
        "regime_n_bins": str(src.get("regime_n_bins") if src.get("regime_n_bins") is not None else "5"),
        "math_enabled": bool(src.get("math_enabled", False)),
        "math_features": _load_feat_list("math_features"),
        "math_operations": math_ops_apply,
        "math_clip_min": str(src.get("math_clip_min") if src.get("math_clip_min") is not None else "0"),
        "math_clip_max": str(src.get("math_clip_max") or ""),
    }


def resolve_enabled_groups(
    saved_groups: list[str] | None,
    all_group_ids: list[str],
    *,
    default_all: bool = True,
) -> dict[str, bool]:
    """Map group id → checked; unknown new groups default to *default_all*."""
    ids = [str(g) for g in all_group_ids]
    if saved_groups is None:
        return {gid: default_all for gid in ids}
    saved_set = {str(g) for g in saved_groups}
    return {gid: gid in saved_set for gid in ids}

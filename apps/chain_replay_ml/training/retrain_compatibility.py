"""Retrain lifecycle — dataset compatibility against a source model's training profile."""

from __future__ import annotations

import os
from typing import Any

from chain_replay_ml.dataset_builder.expected_spec import (
    format_sampling_interval_label,
    resolve_atm_band,
    strike_selection_display_label,
)
from chain_replay_ml.replay_config import load_dataset_metadata_json

from .registry import load_model_detail


def _norm_market(meta: dict[str, Any]) -> str:
    market = str(meta.get("market") or "").strip().upper()
    if market and market != "—":
        return market
    sources = meta.get("sources") or meta.get("days") or []
    markets = {
        str(s.get("market") or "").strip().upper()
        for s in sources
        if isinstance(s, dict) and s.get("market")
    }
    markets.discard("")
    if len(markets) == 1:
        return next(iter(markets))
    return market or "NIFTY"


def _sampling_sec(meta: dict[str, Any]) -> int | None:
    cfg = meta.get("dataset_configuration") if isinstance(meta.get("dataset_configuration"), dict) else {}
    sampling = meta.get("sampling") if isinstance(meta.get("sampling"), dict) else {}
    sm = meta.get("selection_method")
    for val in (
        cfg.get("sampling_interval_sec"),
        cfg.get("feature_grid_step_sec"),
        sampling.get("interval_sec"),
        sampling.get("trainingIntervalSec"),
        sm.get("interval_sec") if isinstance(sm, dict) else None,
    ):
        if val is not None:
            try:
                return int(val)
            except (TypeError, ValueError):
                continue
    fp = meta.get("pipeline_fingerprint")
    if isinstance(fp, dict) and fp.get("sampling_interval_sec") is not None:
        try:
            return int(fp["sampling_interval_sec"])
        except (TypeError, ValueError):
            pass
    return None


def _strike_band(meta: dict[str, Any]) -> int | str | None:
    ss = meta.get("strike_selection") if isinstance(meta.get("strike_selection"), dict) else {}
    mode = str(ss.get("mode") or "atm_band").lower()
    if mode not in ("", "atm_band"):
        band = ss.get("band") or ss.get("atmBand")
        if band is None:
            return None
        if str(band).lower() == "all":
            return "all"
        try:
            return int(band)
        except (TypeError, ValueError):
            return band
    return resolve_atm_band(meta)


def _premium_filter(meta: dict[str, Any]) -> dict[str, Any] | None:
    mf = meta.get("master_filter") if isinstance(meta.get("master_filter"), dict) else {}
    sm_crit = {}
    sm = meta.get("selection_method")
    if isinstance(sm, dict) and isinstance(sm.get("criteria"), dict):
        sm_crit = sm["criteria"]
    enabled = bool(
        mf.get("premium_enabled")
        or sm_crit.get("premium_enabled")
        or (mf.get("premium_min") is not None and mf.get("premium_max") is not None)
        or (sm_crit.get("premium_min") is not None and sm_crit.get("premium_max") is not None)
    )
    if not enabled:
        return None
    lo = mf.get("premium_min", sm_crit.get("premium_min"))
    hi = mf.get("premium_max", sm_crit.get("premium_max"))
    if lo is None or hi is None:
        return None
    return {"min": float(lo), "max": float(hi)}


def _delta_filter(meta: dict[str, Any]) -> dict[str, Any] | None:
    ss = meta.get("strike_selection") if isinstance(meta.get("strike_selection"), dict) else {}
    if str(ss.get("mode") or "").lower() == "delta_range":
        return {
            "min": float(ss.get("delta_min") or ss.get("deltaMin") or 0),
            "max": float(ss.get("delta_max") or ss.get("deltaMax") or 0),
            "type": str(ss.get("delta_type") or ss.get("deltaType") or "absolute").lower(),
        }
    mf = meta.get("master_filter") if isinstance(meta.get("master_filter"), dict) else {}
    sm_crit = {}
    sm = meta.get("selection_method")
    if isinstance(sm, dict) and isinstance(sm.get("criteria"), dict):
        sm_crit = sm["criteria"]
    enabled = bool(
        mf.get("delta_enabled")
        or sm_crit.get("delta_enabled")
        or (mf.get("delta_min") is not None and mf.get("delta_max") is not None)
        or (sm_crit.get("delta_min") is not None and sm_crit.get("delta_max") is not None)
    )
    if not enabled:
        return None
    lo = mf.get("delta_min", sm_crit.get("delta_min"))
    hi = mf.get("delta_max", sm_crit.get("delta_max"))
    if lo is None or hi is None:
        return None
    return {"min": float(lo), "max": float(hi), "type": "absolute"}


def build_retrain_profile(
    *,
    target: str,
    prediction_type: str,
    dataset_meta: dict[str, Any],
) -> dict[str, Any]:
    """Canonical training profile required for retrain compatibility."""
    interval = _sampling_sec(dataset_meta)
    band = _strike_band(dataset_meta)
    return {
        "target": str(target or "").strip(),
        "prediction_type": str(prediction_type or "regression").strip().lower(),
        "market": _norm_market(dataset_meta),
        "sampling_interval_sec": interval,
        "sampling_interval_label": format_sampling_interval_label(interval) if interval is not None else None,
        "strike_band": band,
        "strike_selection_label": strike_selection_display_label(dataset_meta),
        "premium_filter": _premium_filter(dataset_meta),
        "delta_filter": _delta_filter(dataset_meta),
        "target_horizon": str(target or "").strip(),
    }


def build_retrain_profile_from_model(data_dir: str, source_model: str) -> dict[str, Any]:
    detail = load_model_detail(data_dir, source_model)
    config = dict(detail.get("config") or {})
    dataset_name = str(config.get("dataset") or "").strip()
    if not dataset_name:
        raise ValueError(f"Source model has no dataset: {source_model}")
    meta = _infer_dataset_meta_from_model_detail(data_dir, detail, dataset_name=dataset_name)
    profile = build_retrain_profile(
        target=str(config.get("target") or ""),
        prediction_type=str(config.get("prediction_type") or "regression"),
        dataset_meta=meta,
    )
    profile["source_dataset"] = dataset_name
    profile["source_model"] = source_model
    if not load_dataset_metadata_json(data_dir, dataset_name):
        profile["source_dataset_missing"] = True
        profile["source_dataset_status"] = "deleted_or_unavailable"
    return profile


def _infer_dataset_meta_from_model_detail(
    data_dir: str,
    detail: dict[str, Any],
    *,
    dataset_name: str,
) -> dict[str, Any]:
    """Rebuild dataset-like metadata from model package when registry JSON is gone."""
    meta = load_dataset_metadata_json(data_dir, dataset_name)
    if meta:
        return meta

    from .dataset_build_snapshot import dataset_meta_from_snapshot, resolve_dataset_build_snapshot

    snap = resolve_dataset_build_snapshot(detail)
    if snap:
        rebuilt = dataset_meta_from_snapshot(snap)
        if rebuilt:
            return rebuilt

    config = dict(detail.get("config") or {})
    target = str(config.get("target") or "").strip()
    replay = config.get("replay_config") if isinstance(config.get("replay_config"), dict) else {}
    if replay:
        synthetic: dict[str, Any] = {
            "market": replay.get("market") or "NIFTY",
            "sampling": dict(replay.get("sampling") or {}),
            "strike_selection": dict(replay.get("strike_selection") or {}),
            "dataset_configuration": dict(replay.get("dataset_configuration") or {}),
            "prediction_target_columns": list(replay.get("prediction_target_columns") or []),
            "prediction_type": config.get("prediction_type") or "regression",
        }
        if target and target not in synthetic["prediction_target_columns"]:
            synthetic["prediction_target_columns"].append(target)
        return synthetic

    from .registry import _resolve_sampling_interval_sec

    fp = detail.get("pipeline_fingerprint") if isinstance(detail.get("pipeline_fingerprint"), dict) else {}
    interval = _resolve_sampling_interval_sec(data_dir, config=config, dataset_name=dataset_name)
    synthetic = {
        "market": fp.get("market") or "NIFTY",
        "prediction_type": config.get("prediction_type") or "regression",
        "prediction_target_columns": [target] if target else [],
        "pipeline_fingerprint": fp,
    }
    if interval is not None:
        synthetic["sampling"] = {"interval_sec": interval}
        synthetic["dataset_configuration"] = {"sampling_interval_sec": interval}
    band = fp.get("atm_band")
    if band is not None:
        synthetic["strike_selection"] = {"mode": "atm_band", "band": band}
    return synthetic


def _check_row(
    *,
    check_id: str,
    label: str,
    passed: bool,
    expected: Any,
    actual: Any,
    required: bool = True,
) -> dict[str, Any]:
    return {
        "id": check_id,
        "label": label,
        "required": required,
        "passed": passed,
        "expected": expected,
        "actual": actual,
    }


def _fmt_premium(spec: dict[str, Any] | None) -> str:
    if not spec:
        return "off"
    return f"{spec['min']:g}–{spec['max']:g}"


def _fmt_delta(spec: dict[str, Any] | None) -> str:
    if not spec:
        return "off"
    return f"{spec.get('type', 'absolute')} {spec['min']:g}–{spec['max']:g}"


def compare_dataset_to_retrain_profile(
    profile: dict[str, Any],
    dataset_meta: dict[str, Any],
    *,
    dataset_name: str = "",
) -> dict[str, Any]:
    """Compare candidate dataset metadata to a source retrain profile."""
    cand_target_cols = set(dataset_meta.get("prediction_target_columns") or [])
    target = str(profile.get("target") or "")
    target_ok = bool(cand_target_cols) and target in cand_target_cols

    cand_interval = _sampling_sec(dataset_meta)
    exp_interval = profile.get("sampling_interval_sec")
    interval_ok = exp_interval is None or cand_interval == exp_interval

    cand_band = _strike_band(dataset_meta)
    exp_band = profile.get("strike_band")
    strike_ok = exp_band is None or cand_band == exp_band

    cand_market = _norm_market(dataset_meta)
    exp_market = str(profile.get("market") or "").upper()
    market_ok = not exp_market or cand_market == exp_market

    cand_pred = str(dataset_meta.get("prediction_type") or profile.get("prediction_type") or "regression").lower()
    exp_pred = str(profile.get("prediction_type") or "regression").lower()
    pred_ok = cand_pred == exp_pred

    checks: list[dict[str, Any]] = [
        _check_row(
            check_id="target",
            label="Target",
            passed=target_ok,
            expected=target,
            actual=target if target_ok else (next(iter(cand_target_cols)) if len(cand_target_cols) == 1 else "mismatch"),
        ),
        _check_row(
            check_id="strike_selection",
            label="Strike Selection",
            passed=strike_ok,
            expected=profile.get("strike_selection_label") or (f"ATM ±{exp_band}" if exp_band not in (None, "all") else "ATM"),
            actual=strike_selection_display_label(dataset_meta) or (f"ATM ±{cand_band}" if cand_band not in (None, "all") else "—"),
        ),
        _check_row(
            check_id="sampling_interval",
            label="Sampling Interval",
            passed=interval_ok,
            expected=profile.get("sampling_interval_label") or format_sampling_interval_label(exp_interval),
            actual=format_sampling_interval_label(cand_interval),
        ),
        _check_row(
            check_id="prediction_type",
            label="Prediction Type",
            passed=pred_ok,
            expected=exp_pred,
            actual=cand_pred,
        ),
        _check_row(
            check_id="market",
            label="Market",
            passed=market_ok,
            expected=exp_market,
            actual=cand_market,
        ),
    ]

    exp_premium = profile.get("premium_filter")
    cand_premium = _premium_filter(dataset_meta)
    if exp_premium:
        checks.append(_check_row(
            check_id="premium_filter",
            label="Premium Filter",
            passed=exp_premium == cand_premium,
            expected=_fmt_premium(exp_premium),
            actual=_fmt_premium(cand_premium),
            required=False,
        ))

    exp_delta = profile.get("delta_filter")
    cand_delta = _delta_filter(dataset_meta)
    if exp_delta:
        checks.append(_check_row(
            check_id="delta_filter",
            label="Delta Filter",
            passed=exp_delta == cand_delta,
            expected=_fmt_delta(exp_delta),
            actual=_fmt_delta(cand_delta),
            required=False,
        ))

    exp_horizon = str(profile.get("target_horizon") or profile.get("target") or "")
    cand_horizons = [c for c in cand_target_cols if str(c).startswith("future_ltp")]
    horizon_ok = exp_horizon in cand_horizons if exp_horizon else True
    if exp_horizon:
        checks.append(_check_row(
            check_id="target_horizon",
            label="Target Horizon",
            passed=horizon_ok,
            expected=exp_horizon,
            actual=exp_horizon if horizon_ok else ", ".join(sorted(cand_horizons)[:3]) or "—",
            required=False,
        ))

    required_checks = [c for c in checks if c.get("required", True)]
    passed_required = sum(1 for c in required_checks if c.get("passed"))
    total_required = len(required_checks) or 1
    score_pct = int(round(100.0 * passed_required / total_required))
    compatible = all(c.get("passed") for c in required_checks)

    return {
        "dataset_name": dataset_name,
        "compatible": compatible,
        "score_pct": score_pct,
        "retraining_allowed": compatible,
        "checks": checks,
    }


def list_retrain_compatible_datasets(data_dir: str, source_model: str) -> dict[str, Any]:
    from chain_replay_ml.dataset_builder.auditor import list_datasets

    profile = build_retrain_profile_from_model(data_dir, source_model)
    rows: list[dict[str, Any]] = []
    for ds in list_datasets(data_dir):
        if not ds.get("has_parquet") or ds.get("is_draft"):
            continue
        name = str(ds.get("dataset_name") or "")
        if not name:
            continue
        meta = load_dataset_metadata_json(data_dir, name)
        if not meta:
            continue
        compat = compare_dataset_to_retrain_profile(profile, meta, dataset_name=name)
        if not compat.get("compatible"):
            continue
        rows.append({
            "dataset_name": name,
            "row_count": int(ds.get("row_count") or meta.get("row_count") or 0),
            "day_count": int(ds.get("day_count") or meta.get("trading_days") or len(meta.get("days") or []) or 0),
            "market": ds.get("market") or _norm_market(meta),
            "compatibility": compat,
        })

    rows.sort(key=lambda r: (-int(r.get("day_count") or 0), -int(r.get("row_count") or 0), str(r.get("dataset_name") or "")))

    source_ds = str(profile.get("source_dataset") or "")
    source_missing = bool(profile.get("source_dataset_missing"))
    default_ds = rows[0]["dataset_name"] if rows else ""
    if not source_missing and source_ds and any(r["dataset_name"] == source_ds for r in rows):
        default_ds = source_ds

    return {
        "ok": True,
        "source_model": source_model,
        "source_dataset": source_ds,
        "source_dataset_missing": source_missing,
        "required_profile": profile,
        "compatible_datasets": rows,
        "default_dataset": default_ds,
        "compatible_count": len(rows),
    }


def evaluate_retrain_dataset_choice(
    data_dir: str,
    *,
    source_model: str,
    dataset_name: str,
) -> dict[str, Any]:
    profile = build_retrain_profile_from_model(data_dir, source_model)
    meta = load_dataset_metadata_json(data_dir, dataset_name)
    if not meta:
        compat = {
            "dataset_name": dataset_name,
            "compatible": False,
            "score_pct": 0,
            "retraining_allowed": False,
            "checks": [
                _check_row(
                    check_id="dataset_available",
                    label="Dataset",
                    passed=False,
                    expected="registered dataset with metadata",
                    actual="metadata not found",
                ),
            ],
        }
        return {
            "ok": True,
            "source_model": source_model,
            "required_profile": profile,
            "compatibility": compat,
        }
    compat = compare_dataset_to_retrain_profile(profile, meta, dataset_name=dataset_name)
    return {
        "ok": True,
        "source_model": source_model,
        "required_profile": profile,
        "compatibility": compat,
    }

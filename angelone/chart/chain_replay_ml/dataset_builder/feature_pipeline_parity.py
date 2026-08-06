"""Three-way feature parity audit: dataset builder vs replay vs live FeatureSnapshot."""

from __future__ import annotations

import math
import os
import re
from datetime import datetime, timezone
from typing import Any, Literal

import pandas as pd

from chain_replay_ml.dataset_builder.feature_plugins import implemented_features_from_names
from chain_replay_ml.dataset_builder.schema_registry import all_registry_feature_names, load_feature_registry
from chain_replay_ml.dataset_builder.tick_coverage import clipped_grid_bounds, list_clipped_grid_timestamps
from chain_replay_ml.dataset_builder.writer import _safe_filename, datasets_dir
from chain_replay_ml.export_atm_pipeline import STRIKE_STEP, normalize_index_name, replay_db_path
from chain_replay_ml.features_atm_band import find_atm_strike
from chain_replay_ml.replay_config import build_replay_config_from_metadata, load_dataset_metadata_json
from chain_replay_ml.replay_feature_scoring import (
    build_replay_day_frame,
    chart_dir_from_data_dir,
    resolve_scoring_expiry,
)

PathKind = Literal["dataset", "replay", "live"]

_INTERNAL_ROW_KEYS = frozenset({
    "_opt_tl", "_spot", "_atm", "_feature_raw",
})
_FEATURE_ROW_METADATA = frozenset({
    "trading_day", "market", "expiry", "timestamp", "token", "symbol", "option_type",
})

_COMPARE_ATOL: dict[str, float] = {
    "current_iv": 0.05,
    "delta": 1e-4,
    "gamma": 1e-6,
    "theta": 1e-3,
    "vega": 1e-3,
    "oi": 1.0,
    "moneyness": 1e-4,
}
_DEFAULT_ATOL = 1e-4
_DEFAULT_RTOL = 1e-3
_TICK_DB_DAY_RE = re.compile(r"^angel_market_(\d{4}-\d{2}-\d{2})(?:_[^.]+)?\.db$")
# Known-complete session for automatic parity audit when no trading_day is supplied.
DEFAULT_PARITY_AUDIT_TRADING_DAY = "2026-07-03"


def _tick_db_search_dirs(data_dir: str) -> list[str]:
    """Same scan roots as chart main._replay_db_search_dirs."""
    from tick_data_paths import tick_search_dirs

    return tick_search_dirs(chart_dir_from_data_dir(data_dir))


def _iter_tick_db_days(data_dir: str) -> list[tuple[str, str]]:
    """Return (trading_day, path) for every non-empty angel_market_*.db."""
    found: dict[str, str] = {}
    for search_dir in _tick_db_search_dirs(data_dir):
        try:
            for entry in os.listdir(search_dir):
                match = _TICK_DB_DAY_RE.match(entry)
                if not match:
                    continue
                path = os.path.join(search_dir, entry)
                if not os.path.isfile(path) or os.path.getsize(path) <= 0:
                    continue
                day = match.group(1)
                prev = found.get(day)
                if prev is None or path > prev:
                    found[day] = path
        except OSError:
            continue
    return sorted(found.items(), key=lambda item: item[0])


def _day_has_option_ticks(
    data_dir: str,
    day: str,
    *,
    market: str = "NIFTY",
) -> bool:
    chart_dir = chart_dir_from_data_dir(data_dir)
    resolution = resolve_scoring_expiry(chart_dir, day, None, underlying=market)
    return bool(resolution.get("resolved_expiry"))


def _days_with_option_ticks(
    data_dir: str,
    *,
    market: str = "NIFTY",
) -> list[str]:
    return [
        day
        for day, _path in _iter_tick_db_days(data_dir)
        if _day_has_option_ticks(data_dir, day, market=market)
    ]


def _latest_day_with_option_ticks(
    data_dir: str,
    *,
    market: str = "NIFTY",
) -> str | None:
    for day, _path in reversed(_iter_tick_db_days(data_dir)):
        if _day_has_option_ticks(data_dir, day, market=market):
            return day
    return None


def _latest_tick_day(data_dir: str) -> str | None:
    days = _iter_tick_db_days(data_dir)
    return days[-1][0] if days else None


def _trading_day_from_latest_dataset(data_dir: str) -> str | None:
    """Pick the newest trading day from the latest built dataset that has tick DB coverage."""
    from chain_replay_ml.dataset_builder.auditor import list_datasets

    rows = list_datasets(data_dir)
    built = [r for r in rows if r.get("has_parquet") and int(r.get("row_count") or 0) > 0]
    built.sort(key=lambda r: str(r.get("created_at") or ""), reverse=True)
    chart_dir = chart_dir_from_data_dir(data_dir)
    tick_days = {day for day, _path in _iter_tick_db_days(data_dir)}

    for row in built:
        name = str(row.get("dataset_name") or "")
        meta = load_dataset_metadata_json(data_dir, name)
        if not meta:
            continue
        days: list[str] = [str(d) for d in (meta.get("days") or []) if d]
        if not days:
            for src in meta.get("sources") or []:
                if isinstance(src, dict) and src.get("trading_day"):
                    days.append(str(src["trading_day"]))
        for day in sorted(set(days), reverse=True):
            if (day in tick_days or replay_db_path(chart_dir, day)) and _day_has_option_ticks(
                data_dir, day, market=str(meta.get("market") or "NIFTY"),
            ):
                return day
        for day in sorted(set(days), reverse=True):
            if day in tick_days or replay_db_path(chart_dir, day):
                if _day_has_option_ticks(data_dir, day, market=str(meta.get("market") or "NIFTY")):
                    return day
        if days:
            return max(days)
    return None


def _resolve_trading_day(
    data_dir: str,
    trading_day: str | None,
    *,
    market: str = "NIFTY",
) -> tuple[str, list[str]]:
    """Resolve probe day and return search hints for error messages."""
    if trading_day:
        day = str(trading_day).strip()
        if day:
            return day, []

    preferred = str(os.environ.get("PARITY_AUDIT_TRADING_DAY") or DEFAULT_PARITY_AUDIT_TRADING_DAY).strip()
    if preferred and _day_has_option_ticks(data_dir, preferred, market=market):
        return preferred, []

    day = _trading_day_from_latest_dataset(data_dir)
    if day and _day_has_option_ticks(data_dir, day, market=market):
        return day, []

    day = _latest_day_with_option_ticks(data_dir, market=market)
    if day:
        return day, []

    searched = _tick_db_search_dirs(data_dir)
    available = [d for d, _ in _iter_tick_db_days(data_dir)]
    with_options = _days_with_option_ticks(data_dir, market=market)
    hints: list[str] = [f"Searched: {', '.join(searched)}"]
    if with_options:
        hints.append(f"Days with option ticks: {', '.join(with_options[-5:])}")
    elif available:
        hints.append(
            f"Tick DBs found ({', '.join(available[-5:])}) but none have option-chain ticks yet."
        )
    else:
        hints.append("No angel_market_YYYY-MM-DD.db files found under tick data or legacy data/ paths.")
    hints.append("Export replay ticks or pass trading_day with option coverage.")
    return "", hints


def _is_null(val: Any) -> bool:
    if val is None:
        return True
    try:
        if isinstance(val, float) and math.isnan(val):
            return True
    except (TypeError, ValueError):
        pass
    return False


def values_close(
    feature: str,
    a: Any,
    b: Any,
    *,
    tolerance: float | None = None,
    rtol: float = _DEFAULT_RTOL,
) -> tuple[bool, float | None]:
    if _is_null(a) and _is_null(b):
        return True, 0.0
    if _is_null(a) or _is_null(b):
        return False, None
    try:
        fa = float(a)
        fb = float(b)
    except (TypeError, ValueError):
        return a == b, None
    diff = abs(fa - fb)
    atol = float(tolerance) if tolerance is not None else _COMPARE_ATOL.get(feature, _DEFAULT_ATOL)
    ok = diff <= atol + rtol * max(abs(fa), abs(fb), 1e-9)
    return ok, diff


def _default_replay_config(data_dir: str) -> tuple[dict[str, Any], str | None]:
    from chain_replay_ml.dataset_builder.auditor import list_datasets

    rows = list_datasets(data_dir)
    built = [r for r in rows if r.get("has_parquet") and int(r.get("row_count") or 0) > 0]
    built.sort(key=lambda r: str(r.get("created_at") or ""), reverse=True)
    for row in built:
        name = str(row.get("dataset_name") or "")
        meta = load_dataset_metadata_json(data_dir, name)
        if meta:
            return build_replay_config_from_metadata(meta), name
    return {
        "market": "NIFTY",
        "sampling": {"interval_sec": 10, "trainingIntervalSec": 10},
        "strike_selection": {"mode": "atm_band", "atm_band": 15},
        "feature_groups_implemented": list(load_feature_registry().get("groupOrder") or []),
        "dataset_configuration": {"sampling_interval_sec": 10},
    }, None


def series_to_feature_dict(row: pd.Series) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, val in row.items():
        key_s = str(key)
        if key in _INTERNAL_ROW_KEYS or key_s.startswith("_") or key_s in _FEATURE_ROW_METADATA:
            continue
        if isinstance(val, (int, float, str, bool)) or val is None:
            out[key_s] = val
        elif pd.notna(val):
            try:
                out[key_s] = float(val)
            except (TypeError, ValueError):
                out[key_s] = val
    return out


def feature_row_from_df(
    df: pd.DataFrame,
    token: str,
    grid_ts: float,
    *,
    eps: float = 0.05,
) -> pd.Series | None:
    if df.empty or not token:
        return None
    sub = df[df["token"].astype(str) == str(token)]
    if sub.empty:
        return None
    ts_col = sub["timestamp"].astype(float)
    exact = sub[ts_col.sub(grid_ts).abs() <= eps]
    if not exact.empty:
        return exact.iloc[0]
    prior = sub[ts_col <= grid_ts + eps]
    if not prior.empty:
        return prior.sort_values("timestamp").iloc[-1]
    nearest_idx = (ts_col - grid_ts).abs().idxmin()
    return sub.loc[nearest_idx]


def _resolve_probe_point(
    data_dir: str,
    replay_config: dict[str, Any],
    *,
    trading_day: str | None,
    timestamp: float | None,
    token: str | None,
) -> tuple[str, float, str, dict[str, Any]]:
    from chain_replay_ml.dataset_builder.day_context import SourceSpec, load_day_context
    from chain_replay_ml.dataset_builder.tick_coverage import sync_feature_grid_step

    market = str(replay_config.get("market") or "NIFTY")
    date_str, hints = _resolve_trading_day(data_dir, trading_day, market=market)
    if not date_str:
        raise ValueError("No tick database found — " + " ".join(hints))

    chart_dir = chart_dir_from_data_dir(data_dir)
    expiry_resolution = resolve_scoring_expiry(chart_dir, date_str, None, underlying=market)
    resolved_expiry = str(expiry_resolution.get("resolved_expiry") or "").strip()
    if not resolved_expiry:
        with_options = _days_with_option_ticks(data_dir, market=market)
        extra = f" Days with option ticks: {', '.join(with_options[-5:])}." if with_options else ""
        raise ValueError(f"No expiry with option ticks for {date_str}.{extra}")

    cfg = replay_config.get("dataset_configuration") or {}
    sampling = replay_config.get("sampling") or {}
    step_sec = int(
        sampling.get("interval_sec")
        or sampling.get("trainingIntervalSec")
        or cfg.get("sampling_interval_sec")
        or 10
    )
    source = SourceSpec(
        source_id=f"{date_str}|{market}|{resolved_expiry}",
        trading_day=date_str,
        market=market,
        expiry=resolved_expiry,
    )
    ctx = load_day_context(chart_dir, source, feature_grid_step_sec=step_sec)
    sync_feature_grid_step(ctx, step_sec)

    grid_ts = float(timestamp) if timestamp is not None else None
    if grid_ts is None:
        grid_list = list_clipped_grid_timestamps(ctx, step_sec=step_sec, max_horizon_sec=0)
        if not grid_list:
            raise ValueError(f"No grid timestamps available for {date_str}.")
        warmup_slots = max(200, int(900 / max(step_sec, 1)))
        idx = min(len(grid_list) - 1, warmup_slots + max(1, int(600 / max(step_sec, 1))))
        grid_ts = float(grid_list[idx])

    tok = str(token or "").strip()
    if not tok:
        spot = ctx.index_tl.ltp_rupees_at(grid_ts)
        if spot is None or spot <= 0:
            raise ValueError("Could not resolve spot at probe timestamp.")
        index_key = normalize_index_name(market)
        step = STRIKE_STEP.get(index_key, 50)
        atm = find_atm_strike(spot, step)
        for (strike_r, opt_type), (candidate, _symbol, opt_tl) in ctx.strike_mapping.items():
            if strike_r == atm and str(opt_type).upper() == "CE":
                if opt_tl.is_fresh_at(grid_ts, 10.0):
                    tok = str(candidate)
                    break
        if not tok:
            raise ValueError("Could not resolve ATM CE token at probe timestamp.")

    meta = {
        "trading_day": date_str,
        "timestamp": grid_ts,
        "token": tok,
        "expiry": resolved_expiry,
        "market": market,
        "step_sec": step_sec,
        "tick_db": replay_db_path(chart_dir, date_str),
        "expiry_resolution": expiry_resolution,
    }
    bounds = clipped_grid_bounds(ctx, max_horizon_sec=0)
    if bounds:
        meta["grid_bounds"] = {"start": bounds[0], "end": bounds[1]}
    return date_str, grid_ts, tok, meta


def _build_path_features(
    data_dir: str,
    replay_config: dict[str, Any],
    *,
    trading_day: str,
    grid_ts: float,
    token: str,
    path: PathKind,
    union_features: list[str],
) -> dict[str, Any]:
    include = frozenset({str(token)})
    if path == "live":
        from live_inference.feature_engine import LiveFeatureEngine
        from live_inference.market_state import LiveMarketState

        state = LiveMarketState.from_replay(
            data_dir,
            date_str=trading_day,
            expiry_hint=replay_config.get("expiry"),
            replay_config=replay_config,
            underlying=str(replay_config.get("market") or "NIFTY"),
        )
        engine = LiveFeatureEngine()
        snap, err = engine.build_snapshot(
            data_dir=data_dir,
            state=state,
            token=str(token),
            grid_ts=float(grid_ts),
            union_features=list(union_features),
            replay_config=replay_config,
            expiry_hint=state.expiry,
        )
        if err or snap is None:
            raise RuntimeError(err or "live_snapshot_failed")
        return dict(snap.features)

    inference_only = path != "dataset"
    df, err, _expiry, _stats = build_replay_day_frame(
        data_dir,
        replay_config,
        trading_day,
        expiry_hint=str(replay_config.get("expiry") or ""),
        required_features=list(union_features),
        inference_only=inference_only,
        only_timestamp=float(grid_ts),
        include_tokens=include,
        token_only=True,
        enrich_tokens_only=include,
    )
    if err:
        raise RuntimeError(err)
    row = feature_row_from_df(df, token, grid_ts)
    if row is None:
        raise RuntimeError(f"No {path} feature row for token {token} at {grid_ts}")
    return series_to_feature_dict(row)


def _parquet_features_at_probe(
    data_dir: str,
    dataset_name: str,
    *,
    trading_day: str,
    grid_ts: float,
    token: str,
    eps: float = 0.05,
) -> dict[str, Any] | None:
    safe = _safe_filename(dataset_name)
    path = os.path.join(datasets_dir(data_dir), f"{safe}.parquet")
    if not os.path.isfile(path):
        return None
    df = pd.read_parquet(path, columns=None)
    if df.empty:
        return None
    sub = df[
        (df["trading_day"].astype(str) == str(trading_day))
        & (df["token"].astype(str) == str(token))
    ]
    if sub.empty:
        return None
    ts_col = sub["timestamp"].astype(float)
    exact = sub[ts_col.sub(grid_ts).abs() <= eps]
    row = exact.iloc[0] if not exact.empty else sub.loc[(ts_col - grid_ts).abs().idxmin()]
    return series_to_feature_dict(row)


def compare_three_paths(
    dataset_features: dict[str, Any],
    replay_features: dict[str, Any],
    live_features: dict[str, Any],
    features: list[str],
    *,
    tolerance: float | None = None,
    max_mismatches: int = 100,
) -> dict[str, Any]:
    matches: list[dict[str, Any]] = []
    mismatches: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    match_n = mismatch_n = missing_n = 0
    max_diff: float | None = None

    for name in features:
        vd = dataset_features.get(name)
        vr = replay_features.get(name)
        vl = live_features.get(name)
        if name not in dataset_features and name not in replay_features and name not in live_features:
            missing_n += 1
            missing.append({"feature": name, "status": "missing_all_paths"})
            continue

        ok_dr, diff_dr = values_close(name, vd, vr, tolerance=tolerance)
        ok_dl, diff_dl = values_close(name, vd, vl, tolerance=tolerance)
        ok_rl, diff_rl = values_close(name, vr, vl, tolerance=tolerance)
        ok = ok_dr and ok_dl and ok_rl
        diffs = [d for d in (diff_dr, diff_dl, diff_rl) if d is not None]
        worst = max(diffs) if diffs else None
        if worst is not None:
            max_diff = worst if max_diff is None else max(max_diff, worst)

        row = {
            "feature": name,
            "value_dataset": vd,
            "value_replay": vr,
            "value_live": vl,
            "diff_dataset_replay": diff_dr,
            "diff_dataset_live": diff_dl,
            "diff_replay_live": diff_rl,
        }
        if ok:
            match_n += 1
            if len(matches) < 20:
                matches.append({**row, "status": "match"})
        else:
            mismatch_n += 1
            if len(mismatches) < max_mismatches:
                mismatches.append({**row, "status": "mismatch"})

    total = len(features)
    parity_pct = round(100.0 * match_n / total, 2) if total else 0.0
    status = "pass" if mismatch_n == 0 and missing_n == 0 else ("warn" if mismatch_n == 0 else "fail")
    return {
        "status": status,
        "label": "PASS" if status == "pass" else ("WARN" if status == "warn" else "FAIL"),
        "feature_count": total,
        "match_count": match_n,
        "mismatch_count": mismatch_n,
        "missing_count": missing_n,
        "parity_pct": parity_pct,
        "max_difference": max_diff,
        "matches_sample": matches,
        "mismatches": mismatches,
        "missing": missing[:max_mismatches],
    }


def run_feature_pipeline_parity_audit(
    data_dir: str,
    *,
    trading_day: str | None = None,
    timestamp: float | None = None,
    token: str | None = None,
    dataset_name: str | None = None,
    replay_config: dict[str, Any] | None = None,
    tolerance: float = 1e-6,
    include_parquet: bool = True,
) -> dict[str, Any]:
    """Run dataset builder → replay → live parity at one probe point for all registry features."""
    registry = load_feature_registry()
    all_features = all_registry_feature_names(registry)
    implemented, pending, _per_group = implemented_features_from_names(all_features, registry)
    cfg, cfg_dataset = _default_replay_config(data_dir) if replay_config is None else (replay_config, dataset_name)
    if dataset_name is None and cfg_dataset:
        dataset_name = cfg_dataset

    date_str, grid_ts, tok, probe = _resolve_probe_point(
        data_dir,
        cfg,
        trading_day=trading_day,
        timestamp=timestamp,
        token=token,
    )

    dataset_feats = _build_path_features(
        data_dir, cfg, trading_day=date_str, grid_ts=grid_ts, token=tok,
        path="dataset", union_features=implemented,
    )
    replay_feats = _build_path_features(
        data_dir, cfg, trading_day=date_str, grid_ts=grid_ts, token=tok,
        path="replay", union_features=implemented,
    )
    live_feats = _build_path_features(
        data_dir, cfg, trading_day=date_str, grid_ts=grid_ts, token=tok,
        path="live", union_features=implemented,
    )

    comparison = compare_three_paths(
        dataset_feats, replay_feats, live_feats, implemented, tolerance=tolerance,
    )

    parquet_block: dict[str, Any] | None = None
    if include_parquet and dataset_name:
        parquet_feats = _parquet_features_at_probe(
            data_dir, dataset_name, trading_day=date_str, grid_ts=grid_ts, token=tok,
        )
        if parquet_feats is not None:
            parquet_cmp = compare_three_paths(
                parquet_feats, replay_feats, live_feats, implemented, tolerance=tolerance,
            )
            parquet_block = {
                "dataset_name": _safe_filename(dataset_name),
                "comparison_vs_replay_live": parquet_cmp,
            }

    return {
        "audit_type": "feature_pipeline_parity",
        "audited_at": datetime.now(timezone.utc).isoformat(),
        "golden_rule": (
            "value_dataset (build_day_rows) == value_replay (FeatureSnapshot) "
            "== value_live (LiveFeatureEngine) at the same grid timestamp"
        ),
        "probe": probe,
        "replay_config_source": dataset_name or "default",
        "registry_feature_count": len(all_features),
        "implemented_feature_count": len(implemented),
        "pending_features": pending,
        "tolerance": tolerance,
        "paths": {
            "dataset": {"feature_keys": len(dataset_feats), "inference_only": False},
            "replay": {"feature_keys": len(replay_feats), "inference_only": True},
            "live": {"feature_keys": len(live_feats), "source": "LiveFeatureEngine"},
        },
        "comparison": comparison,
        "stored_parquet": parquet_block,
    }

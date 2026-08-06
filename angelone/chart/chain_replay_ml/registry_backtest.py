"""Registry-model trade backtest for chain replay analytics."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pandas as pd

from chain_replay_ml.backtest_ranking import filter_by_delta_band_name
from chain_replay_ml.recompute_2_1_ratio import (
    SCORE_THRESHOLD,
    _avg_phit,
    _empty_entry_price_audit,
    _empty_label_leakage,
    _empty_lookahead_leakage,
    _empty_score_distribution,
    _empty_tick_path_audit,
    _empty_winner_path_audit,
    _execution_integrity,
    _outcome_summary,
    _random_rows_per_timestamp,
    _rows_to_trades_for_tokens,
    _score_bucket,
    _target_validation,
    _top_rows_for_score_threshold,
    simulate_positions,
)
from chain_replay_ml.replay_feature_scoring import (
    load_model_inference_config,
    load_scoring_day_frame,
    replay_scoring_coverage,
    scoring_required_columns,
)
from chain_replay_ml.replay_scoring_cache import (
    get_cached_scored_frame,
    replay_cache_key,
    set_cached_scored_frame,
)

# Backwards-compatible aliases
model_scoring_coverage = replay_scoring_coverage


def load_model_training_config(data_dir: str, model_name: str) -> dict[str, Any] | None:
    """Legacy name — returns inference config (dataset link optional)."""
    return load_model_inference_config(data_dir, model_name)


def _registry_score_series(y_pred: pd.Series, ltp: pd.Series, target: str) -> pd.Series:
    pred = y_pred.astype(float)
    spot = ltp.astype(float)
    if str(target).startswith("future_ltp"):
        with np.errstate(divide="ignore", invalid="ignore"):
            return (pred - spot) / spot * 100.0
    return pred


def _scoring_null_diagnostics(day_df: pd.DataFrame, features: list[str]) -> list[tuple[str, float]]:
    if day_df.empty:
        return []
    out: list[tuple[str, float]] = []
    for col in features:
        if col not in day_df.columns:
            continue
        null_pct = float(day_df[col].isna().mean())
        if null_pct > 0:
            out.append((col, round(null_pct, 4)))
    out.sort(key=lambda item: item[1], reverse=True)
    return out[:10]


def build_registry_scored_frame(
    data_dir: str,
    model_name: str,
    date_str: str,
    *,
    expiry_hint: str | None = None,
    allow_dataset_fallback: bool = False,
    scored_df: pd.DataFrame | None = None,
    parallel_features: bool = False,
    step_times: dict[str, float] | None = None,
    scoring_diagnostics: dict[str, Any] | None = None,
    on_step_progress: Callable[[str, str], None] | None = None,
) -> pd.DataFrame:
    cache_key = replay_cache_key(data_dir, model_name, date_str, expiry_hint)
    if scored_df is not None and not scored_df.empty:
        return scored_df.copy()

    cached = get_cached_scored_frame(cache_key)
    if cached is not None and not cached.empty:
        return cached

    loaded = load_model_inference_config(data_dir, model_name)
    if not loaded:
        return pd.DataFrame()

    target = loaded["target"]
    features = loaded["features"]
    model_path = loaded["model_path"]
    algorithm = loaded.get("algorithm") or loaded["config"].get("algorithm")
    from chain_replay_ml.training.model_runtime import load_prediction_model

    import time

    t0 = time.perf_counter()
    day_df, _config, coverage = load_scoring_day_frame(
        data_dir,
        model_name,
        date_str,
        expiry_hint=expiry_hint,
        allow_dataset_fallback=allow_dataset_fallback,
        parallel_features=parallel_features,
        on_step_progress=on_step_progress,
    )
    if step_times is not None:
        step_times["build_feature_rows"] = round(time.perf_counter() - t0, 3)
        sub = dict((coverage.get("feature_build") or {}).get("timing_sec") or {})
        for key, sec in sub.items():
            step_times[key] = sec
    diag: dict[str, Any] = {
        "feature_rows_built": int(len(day_df)),
        "feature_build": dict(coverage.get("feature_build") or {}),
    }
    if coverage.get("error"):
        diag["feature_build_error"] = coverage.get("error")
    pending = list((coverage.get("feature_build") or {}).get("pending_registry_features") or [])
    if pending:
        diag["pending_registry_features"] = pending[:20]

    if day_df.empty:
        err = coverage.get("error")
        if err:
            print(f"[registry_backtest] no replay features for {date_str}: {err}")
        if scoring_diagnostics is not None:
            scoring_diagnostics.update(diag)
        return pd.DataFrame()

    required = scoring_required_columns(features, target, require_target=False)
    missing = [c for c in required if c not in day_df.columns]
    if missing:
        diag["missing_model_columns"] = missing
        print(f"[registry_backtest] missing columns for scoring on {date_str}: {missing[:8]}")
        if scoring_diagnostics is not None:
            scoring_diagnostics.update(diag)
        return pd.DataFrame()

    if "trading_day" not in day_df.columns:
        day_df = day_df.copy()
        day_df["trading_day"] = str(date_str)

    drop_cols = features + ["ltp", "token", "timestamp", "delta"]
    before_dropna = len(day_df)
    work = day_df.dropna(subset=drop_cols).copy()
    diag["rows_before_dropna"] = int(before_dropna)
    diag["rows_after_dropna"] = int(len(work))
    if work.empty and before_dropna > 0:
        diag["top_null_features"] = _scoring_null_diagnostics(day_df, features)
    if work.empty:
        if scoring_diagnostics is not None:
            scoring_diagnostics.update(diag)
        return pd.DataFrame()

    model = load_prediction_model(model_path, algorithm)
    t0 = time.perf_counter()
    if on_step_progress:
        on_step_progress("model_predict", f"Predicting {len(work):,} rows")
    y_pred = pd.Series(model.predict(work[features]), index=work.index)
    if step_times is not None:
        step_times["model_predict"] = round(time.perf_counter() - t0, 3)
    work["pred_ltp"] = y_pred.astype(float)
    work["score"] = _registry_score_series(y_pred, work["ltp"], target)
    work["pred_max_return"] = work["score"]
    work["pred_min_return"] = 0.0
    work["P_hit"] = 0.5
    work["delta_band"] = filter_by_delta_band_name(work)
    work = work.dropna(subset=["delta_band", "score"]).copy()
    diag["rows_after_delta_band"] = int(len(work))
    if not work.empty:
        set_cached_scored_frame(cache_key, work)
    if scoring_diagnostics is not None:
        scoring_diagnostics.update(diag)
    return work


def run_registry_backtest_for_date(
    date_str: str,
    data_dir: str,
    model_name: str,
    score_threshold: float = SCORE_THRESHOLD,
    *,
    expiry_hint: str | None = None,
) -> list[dict[str, Any]]:
    df = build_registry_scored_frame(
        data_dir, model_name, date_str, expiry_hint=expiry_hint,
    )
    if df.empty:
        return []
    trade_rows = _top_rows_for_score_threshold(df, score_threshold)
    return _rows_to_trades_for_tokens(date_str, trade_rows)


def _registry_score_distribution(date_str: str, df: pd.DataFrame) -> list[dict[str, Any]]:
    if df.empty:
        return _empty_score_distribution()

    tops_by_bucket: dict[int, list] = {n: [] for n in range(6)}
    for _, group in df.groupby("timestamp"):
        top = group.sort_values(by="score", ascending=False).iloc[0]
        tops_by_bucket[_score_bucket(float(top["score"]))].append(top)

    tokens: set[str] = set()
    for rows in tops_by_bucket.values():
        for row in rows:
            tokens.add(str(row["token"]))
    timelines = None
    if tokens:
        from chain_replay_ml.recompute_2_1_ratio import _load_timelines_for_tokens

        timelines = _load_timelines_for_tokens(date_str, sorted(tokens))

    out = []
    for score in range(6):
        trades = _rows_to_trades_for_tokens(date_str, tops_by_bucket[score], timelines)
        summary = _outcome_summary(trades)
        out.append({
            "score": score,
            "trades": int(summary["count"]),
            "target_pct": summary["target_pct"],
            "avg_phit": _avg_phit(trades),
            "net_pnl": summary["net_pnl"],
            "pf": summary["pf"],
        })
    return out


def _registry_baseline_comparison(
    date_str: str,
    df: pd.DataFrame,
    premium_trades: list[dict[str, Any]],
    entered_trades: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if df.empty:
        rows = [
            ("Score ≥3", premium_trades),
            ("Final", entered_trades),
        ]
        out = []
        for label, trades in rows:
            summary = _outcome_summary(trades)
            out.append({
                "strategy": label,
                "trades": int(summary["count"]),
                "target_pct": summary["target_pct"],
                "net_pnl": summary["net_pnl"],
                "pf": summary["pf"],
            })
        return out

    tokens = set(df["token"].astype(str))
    for t in premium_trades:
        tokens.add(str(t["token"]))
    from chain_replay_ml.recompute_2_1_ratio import _load_timelines_for_tokens

    timelines = _load_timelines_for_tokens(date_str, sorted(tokens))
    random_trades = _rows_to_trades_for_tokens(date_str, _random_rows_per_timestamp(df), timelines)
    score_1_trades = _rows_to_trades_for_tokens(
        date_str, _top_rows_for_score_threshold(df, 1.0), timelines,
    )
    score_2_trades = _rows_to_trades_for_tokens(
        date_str, _top_rows_for_score_threshold(df, 2.0), timelines,
    )
    rows = [
        ("Random", random_trades),
        ("Score ≥1", score_1_trades),
        ("Score ≥2", score_2_trades),
        ("Score ≥3", premium_trades),
        ("Final", entered_trades),
    ]
    out = []
    for label, trades in rows:
        summary = _outcome_summary(trades)
        out.append({
            "strategy": label,
            "trades": int(summary["count"]),
            "target_pct": summary["target_pct"],
            "net_pnl": summary["net_pnl"],
            "pf": summary["pf"],
        })
    return out


def compute_registry_model_audit(
    date_str: str,
    data_dir: str,
    model_name: str,
    max_concurrent: int = 1,
    score_threshold: float = SCORE_THRESHOLD,
    *,
    expiry_hint: str | None = None,
) -> dict[str, Any]:
    df = build_registry_scored_frame(
        data_dir, model_name, date_str, expiry_hint=expiry_hint,
    )
    premium_rows = _top_rows_for_score_threshold(df, score_threshold) if not df.empty else []
    premium_trades = _rows_to_trades_for_tokens(date_str, premium_rows)
    concurrent = max_concurrent if max_concurrent > 0 else 9999
    entered_trades = simulate_positions(premium_trades, concurrent)
    entered_ts = {t["entry_ts"] for t in entered_trades}
    rejected_trades = [t for t in premium_trades if t["entry_ts"] not in entered_ts]

    timestamps = int(df["timestamp"].nunique()) if not df.empty else 0
    delta_rows = int(df["delta_band"].notna().sum()) if not df.empty else 0
    empty_audit = {
        "engine": "registry",
        "selection_bias": {
            "all_candidate_signals": timestamps,
            "passed_phit_filter": timestamps,
            "passed_delta_filter": delta_rows,
            "passed_premium_filter": len(premium_rows),
        },
        "headline_comparison": {
            "all_candidates": _outcome_summary(premium_trades),
            "entered": _outcome_summary(entered_trades),
        },
        "rejected_outcomes": {
            "entered": _outcome_summary(entered_trades),
            "position_blocked": _outcome_summary(rejected_trades),
            "ml_signals": _outcome_summary(premium_trades),
            "filtered_out": _outcome_summary([]),
            "suppressed_count": 0,
        },
        "phit_calibration": [],
        "filter_contribution": [
            {
                "stage": "Registry signal (score≥3)",
                "trades": len(premium_trades),
                "target_pct": _outcome_summary(premium_trades).get("target_pct"),
            },
            {
                "stage": "Final entry",
                "trades": len(entered_trades),
                "target_pct": _outcome_summary(entered_trades).get("target_pct"),
            },
        ],
        "top_pick_funnel": [],
        "baseline_comparison": _registry_baseline_comparison(
            date_str, df, premium_trades, entered_trades,
        ),
        "score_distribution": _registry_score_distribution(date_str, df),
        "target_validation": _target_validation(entered_trades, date_str),
        "execution_integrity": _execution_integrity(entered_trades, date_str),
        "lookahead_leakage": _empty_lookahead_leakage(),
        "label_leakage": _empty_label_leakage(),
        "entry_price": _empty_entry_price_audit(),
        "winner_path": _empty_winner_path_audit(),
        "tick_path": _empty_tick_path_audit(),
        "position_limits": [],
        "high_confidence_misses": [],
    }

    limit_label = f"{max_concurrent} Pos" if max_concurrent > 0 else "Unconstrained"
    empty_audit["position_limits"] = [{
        "limit": limit_label,
        "trades": len(entered_trades),
        "target_pct": _outcome_summary(entered_trades).get("target_pct"),
    }]
    return empty_audit

"""Replay-day model quality analysis — prediction accuracy, calibration, errors."""

from __future__ import annotations

import json
import os
from typing import Any

import numpy as np
import pandas as pd

from datetime import datetime, timedelta, timezone

from chain_replay_ml.recompute_2_1_ratio import (
    _build_scored_ml_frame,
    _load_timelines_for_tokens,
    _outcome_summary,
    _trade_net_pnl_rs,
    simulate_positions,
)
from chain_replay_ml.training.evaluator import evaluate_regression
from chain_replay_ml.training.paths import model_artifact_paths
from chain_replay_ml.training.registry import get_trained_model

from path_config import CHART_DATA_ROOT as _CHART_DIR
CONFIDENCE_BUCKETS = (
    ("50-60%", 0.50, 0.60),
    ("60-70%", 0.60, 0.70),
    ("70-80%", 0.70, 0.80),
    ("80-90%", 0.80, 0.90),
    ("90-100%", 0.90, 1.01),
)

SCORE_BUCKETS = (
    ("Score 2–3", 2.0, 3.0),
    ("Score 3–4", 3.0, 4.0),
    ("Score 4–5", 4.0, 5.0),
    ("Score 5+", 5.0, 9999.0),
)

ERROR_RS_BANDS = (
    ("within_2", 0.0, 2.0),
    ("within_5", 2.0, 5.0),
    ("within_10", 5.0, 10.0),
    ("above_10", 10.0, None),
)

SHAP_SAMPLE_SIZE = 400
SPARK_CHARS = "▁▂▃▄▅▆█"
_IST = timezone(timedelta(hours=5, minutes=30))


def _round(v: float | None, n: int = 2) -> float | None:
    if v is None or not np.isfinite(v):
        return None
    return round(float(v), n)


def _pct(n: int, total: int) -> float | None:
    if total <= 0:
        return None
    return round(n / total * 100.0, 1)


def _calibration_table(trades: list[dict]) -> list[dict]:
    rows = []
    for label, lo, hi in CONFIDENCE_BUCKETS:
        bucket = [t for t in trades if lo <= float(t.get("p_hit") or 0) < hi]
        n = len(bucket)
        if not n:
            rows.append({"confidence": label, "signals": 0, "actual_win_rate": None})
            continue
        if bucket[0].get("ltp") is not None:
            summary = _outcome_summary(bucket)
            win_rate = summary["target_pct"]
        else:
            wins = sum(1 for t in bucket if t.get("outcome_type") == "target")
            win_rate = round(wins / n * 100.0, 2)
        rows.append({
            "confidence": label,
            "signals": n,
            "actual_win_rate": win_rate,
        })
    return rows


def _calibration_label(buckets: list[dict]) -> tuple[str, float | None]:
    gaps: list[float] = []
    for b in buckets:
        if (b.get("signals") or 0) < 5:
            continue
        label = str(b.get("confidence") or "")
        if "-" in label:
            parts = label.replace("%", "").split("-")
            try:
                mid = (float(parts[0]) + float(parts[1])) / 200.0
            except ValueError:
                continue
        else:
            continue
        actual = float(b.get("actual_win_rate") or 0) / 100.0
        gaps.append(abs(mid - actual))
    if not gaps:
        return "Unknown", None
    mae = float(np.mean(gaps))
    if mae < 0.05:
        return "Excellent", _round(mae, 3)
    if mae < 0.10:
        return "Good", _round(mae, 3)
    return "Needs Improvement", _round(mae, 3)


def _error_distribution_rs(abs_errors: np.ndarray) -> list[dict]:
    total = len(abs_errors)
    if total == 0:
        return []
    rows = []
    for key, lo, hi in ERROR_RS_BANDS:
        if hi is None:
            mask = abs_errors >= lo
            label = "Above ₹10"
        elif key == "within_2":
            mask = abs_errors <= hi
            label = "Within ₹2"
        elif key == "within_5":
            mask = (abs_errors > 2.0) & (abs_errors <= hi)
            label = "Within ₹5"
        else:
            mask = (abs_errors > 5.0) & (abs_errors <= hi)
            label = "Within ₹10"
        count = int(mask.sum())
        rows.append({"band": label, "key": key, "count": count, "pct": _pct(count, total)})
    return rows


def _regime_label(spot_change_5m: float | None) -> str:
    if spot_change_5m is None or not np.isfinite(spot_change_5m):
        return "sideways"
    if spot_change_5m > 0.15:
        return "bullish"
    if spot_change_5m < -0.15:
        return "bearish"
    return "sideways"


def _timeout_path_stats(trade: dict, timeline) -> tuple[float, float]:
    if timeline is None:
        return 0.0, 0.0
    entry_ts = float(trade["entry_ts"])
    exit_ts = float(trade.get("exit_ts") or entry_ts)
    entry_p = float(trade["ltp"])
    if entry_p <= 0:
        return 0.0, 0.0
    lowest = entry_p
    highest = entry_p
    stamps = getattr(timeline, "timestamps", None) or []
    paise = getattr(timeline, "ltps_paise", None) or []
    for ts, lp in zip(stamps, paise):
        if ts < entry_ts:
            continue
        if ts > exit_ts:
            break
        if lp is None or lp <= 0:
            continue
        px = lp / 100.0
        lowest = min(lowest, px)
        highest = max(highest, px)
    mae_pct = (lowest - entry_p) / entry_p * 100.0
    mfe_pct = (highest - entry_p) / entry_p * 100.0
    return mfe_pct, mae_pct


def _timeout_analysis(trades: list[dict], date_str: str) -> dict[str, Any]:
    timeouts = [t for t in trades if t.get("outcome_type") == "timeout"]
    if not timeouts:
        return {
            "timeout_trades": 0,
            "reached_90pct_target": 0,
            "reached_75pct_target": 0,
            "never_moved": 0,
            "avg_mfe_pct": None,
            "avg_mae_pct": None,
            "conclusion": "No timeout trades in this session.",
        }
    tokens = sorted({str(t["token"]) for t in timeouts})
    timelines = _load_timelines_for_tokens(date_str, tokens)
    reached_90 = 0
    reached_75 = 0
    never_moved = 0
    mfes: list[float] = []
    maes: list[float] = []
    for t in timeouts:
        tgt = float(t.get("target_pct") or 0)
        mfe, mae = _timeout_path_stats(t, timelines.get(str(t["token"])))
        mfes.append(mfe)
        maes.append(mae)
        if mfe < 0.05:
            never_moved += 1
        if tgt > 0 and mfe >= tgt * 0.90:
            reached_90 += 1
        if tgt > 0 and mfe >= tgt * 0.75:
            reached_75 += 1
    avg_mfe = float(np.mean(mfes)) if mfes else None
    if reached_90 >= len(timeouts) * 0.4:
        conclusion = "Mostly target too aggressive — price often neared target before timeout."
    elif never_moved >= len(timeouts) * 0.4:
        conclusion = "Mostly poor prediction — many timeouts never moved favorably."
    else:
        conclusion = "Mixed timeout drivers — review worst errors and target sizing."
    return {
        "timeout_trades": len(timeouts),
        "reached_90pct_target": reached_90,
        "reached_75pct_target": reached_75,
        "never_moved": never_moved,
        "avg_mfe_pct": _round(avg_mfe),
        "avg_mae_pct": _round(float(np.mean(maes)) if maes else None),
        "conclusion": conclusion,
    }


def _confidence_vs_profit(trades: list[dict]) -> list[dict]:
    rows = []
    for label, lo, hi in CONFIDENCE_BUCKETS:
        bucket = [t for t in trades if lo <= float(t.get("p_hit") or 0) < hi]
        if not bucket:
            rows.append({"confidence": label, "trades": 0, "avg_profit_rs": None})
            continue
        pnls = [_trade_net_pnl_rs(t) for t in bucket]
        rows.append({
            "confidence": label,
            "trades": len(bucket),
            "avg_profit_rs": _round(float(np.mean(pnls))),
        })
    return rows


def _score_vs_profit(trades: list[dict]) -> list[dict]:
    rows = []
    for label, lo, hi in SCORE_BUCKETS:
        bucket = [t for t in trades if lo <= float(t.get("score") or 0) < hi]
        if not bucket:
            rows.append({"confidence": label, "trades": 0, "avg_profit_rs": None})
            continue
        pnls = [_trade_net_pnl_rs(t) for t in bucket]
        rows.append({
            "confidence": label,
            "trades": len(bucket),
            "avg_profit_rs": _round(float(np.mean(pnls))),
        })
    return rows


def _residual_confidence_calibration(abs_errors: np.ndarray) -> list[dict]:
    """Regression calibration: high-confidence buckets should have lower error."""
    if len(abs_errors) == 0:
        return []
    ranks = pd.Series(abs_errors).rank(pct=True, method="average").values
    confidence = 1.0 - ranks
    rows = []
    for label, lo, hi in CONFIDENCE_BUCKETS:
        mask = (confidence >= lo) & (confidence < hi)
        n = int(mask.sum())
        if not n:
            rows.append({"confidence": label, "signals": 0, "actual_win_rate": None})
            continue
        bucket_err = abs_errors[mask]
        within_2 = float((bucket_err <= 2.0).mean() * 100.0)
        rows.append({
            "confidence": label,
            "signals": n,
            "actual_win_rate": _round(within_2),
        })
    return rows


def _pick_calibration(
    signal_trades: list[dict],
    cal_trades: list[dict],
    abs_errors: np.ndarray,
    *,
    prefer_residual: bool = False,
) -> dict[str, Any]:
    phit_trades = [t for t in (signal_trades or cal_trades) if float(t.get("p_hit") or 0) > 0]
    use_phit = (not prefer_residual) and len(phit_trades) >= 25
    if use_phit:
        buckets = _calibration_table(signal_trades or cal_trades)
        label, score = _calibration_label(buckets)
        return {
            "mode": "phit",
            "metric_label": "Actual win rate",
            "buckets": buckets,
            "score_label": label,
            "score_value": score,
        }
    buckets = _residual_confidence_calibration(abs_errors)
    label, score = _calibration_label(buckets)
    return {
        "mode": "residual",
        "metric_label": "Within ₹2 accuracy",
        "buckets": buckets,
        "score_label": label,
        "score_value": score,
    }


def _pick_confidence_vs_profit(signal_trades: list[dict]) -> tuple[list[dict], str]:
    if not signal_trades:
        return [], "none"
    phit_rows = _confidence_vs_profit(signal_trades)
    phit_with_data = sum(1 for r in phit_rows if (r.get("trades") or 0) > 0)
    if phit_with_data >= 3:
        return phit_rows, "phit"
    score_rows = _score_vs_profit(signal_trades)
    return score_rows, "score"


def _compute_shap_importance(model: Any, X: pd.DataFrame, features: list[str]) -> list[dict]:
    try:
        import shap
    except ImportError:
        return []
    if X.empty or not features:
        return []
    use = X[features].replace([np.inf, -np.inf], np.nan).dropna()
    if use.empty:
        return []
    if len(use) > SHAP_SAMPLE_SIZE:
        use = use.sample(SHAP_SAMPLE_SIZE, random_state=42)
    try:
        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(use)
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[0]
        mean_abs = np.abs(np.asarray(shap_vals)).mean(axis=0)
    except Exception:
        return []
    total = float(mean_abs.sum()) or 1.0
    rows = [
        {
            "feature": feat,
            "importance_pct": _round(float(val) / total * 100.0, 1),
            "source": "shap",
        }
        for feat, val in zip(features, mean_abs)
    ]
    rows.sort(key=lambda r: r["importance_pct"] or 0, reverse=True)
    return rows[:12]


def _shap_for_fold_stamp(stamp: str, work: pd.DataFrame) -> list[dict]:
    from chain_replay_ml.backtest_ranking import load_models_for_stamp
    from chain_replay_ml.constants import FEATURE_COLUMNS

    models_dir = os.path.join(_CHART_DIR, "data", "ml_models")
    try:
        models = load_models_for_stamp(models_dir, stamp)
    except Exception:
        return []
    reg_model = None
    for band in ("A", "B", "C"):
        reg_model = (models.get(band) or {}).get("reg_max")
        if reg_model is not None:
            break
    if reg_model is None:
        return []
    cols = [c for c in FEATURE_COLUMNS if c in work.columns]
    if not cols:
        return []
    sample = work[cols].dropna()
    return _compute_shap_importance(reg_model, sample, cols)


def _sparkline(values: list[float]) -> str:
    if not values:
        return ""
    arr = [float(v) for v in values if v is not None and np.isfinite(v)]
    if not arr:
        return ""
    lo, hi = min(arr), max(arr)
    if hi <= lo:
        return SPARK_CHARS[0] * len(arr)
    out: list[str] = []
    for v in arr:
        idx = int((v - lo) / (hi - lo) * (len(SPARK_CHARS) - 1))
        out.append(SPARK_CHARS[max(0, min(len(SPARK_CHARS) - 1, idx))])
    return "".join(out)


def _error_timeline(timestamps: np.ndarray, abs_errors: np.ndarray, *, bucket_minutes: int = 30) -> list[dict]:
    if len(timestamps) == 0 or len(abs_errors) == 0:
        return []
    buckets: dict[str, list[float]] = {}
    sq_buckets: dict[str, list[float]] = {}
    slot_ts: dict[str, int] = {}
    for ts, err in zip(timestamps, abs_errors):
        if not np.isfinite(ts) or not np.isfinite(err):
            continue
        dt = datetime.fromtimestamp(float(ts), tz=_IST)
        slot_min = (dt.minute // bucket_minutes) * bucket_minutes
        slot = dt.replace(minute=slot_min, second=0, microsecond=0)
        key = slot.strftime("%H:%M")
        buckets.setdefault(key, []).append(float(err))
        sq_buckets.setdefault(key, []).append(float(err) ** 2)
        slot_ts.setdefault(key, int(slot.timestamp()))
    rows: list[dict] = []
    for key in sorted(buckets.keys()):
        errs = buckets[key]
        within_2 = sum(1 for e in errs if e <= 2.0)
        rows.append({
            "time": key,
            "timestamp": slot_ts.get(key),
            "mae_rs": _round(float(np.mean(errs))),
            "rmse_rs": _round(float(np.sqrt(np.mean(sq_buckets[key])))),
            "median_rs": _round(float(np.median(errs))),
            "within_2_pct": _round(within_2 / len(errs) * 100.0, 1),
            "count": len(errs),
        })
    return rows


def _build_sparklines(timeline: list[dict]) -> dict[str, str]:
    if not timeline:
        return {}
    return {
        "mae": _sparkline([r.get("mae_rs") or 0 for r in timeline]),
        "rmse": _sparkline([r.get("rmse_rs") or 0 for r in timeline]),
        "median": _sparkline([r.get("median_rs") or 0 for r in timeline]),
        "within_2": _sparkline([r.get("within_2_pct") or 0 for r in timeline]),
        "p95": _sparkline([r.get("mae_rs") or 0 for r in timeline]),
    }


def _score_band(score: int | float | None) -> str:
    s = int(score or 0)
    if s >= 80:
        return "good"
    if s >= 60:
        return "warn"
    return "bad"


def _mae_accuracy_score(mae: float | None, within_2_pct: float | None) -> int:
    mae_part = 50
    if mae is not None:
        if mae <= 2.0:
            mae_part = 98
        elif mae <= 3.0:
            mae_part = 92
        elif mae <= 5.0:
            mae_part = 78
        elif mae <= 8.0:
            mae_part = 62
        elif mae <= 12.0:
            mae_part = 45
        else:
            mae_part = max(15, int(100 - mae * 4))
    w2_part = int(within_2_pct or 0) if within_2_pct is not None else mae_part
    return max(0, min(100, int(mae_part * 0.45 + w2_part * 0.55)))


def _calibration_health_score(cal: dict[str, Any]) -> int:
    label = str(cal.get("score_label") or "")
    if label == "Excellent":
        return 95
    if label == "Good":
        return 72
    if label == "Needs Improvement":
        return 48
    sv = cal.get("score_value")
    if sv is not None and np.isfinite(sv):
        return max(0, min(100, int((1.0 - float(sv)) * 100)))
    return 55


def _bias_health_score(bias: dict[str, Any]) -> int:
    over = float(bias.get("over_predicted_pct") or 50)
    balance = 100.0 - abs(over - 50.0) * 1.6
    avg_b = abs(float(bias.get("average_bias_rs") or 0))
    penalty = min(28.0, avg_b * 2.5)
    return max(0, min(100, int(balance - penalty)))


def _robustness_health_score(
    regime_rows: list[dict],
    timeout_analysis: dict[str, Any],
    within_2_pct: float | None,
) -> int:
    score = 68.0
    if within_2_pct is not None:
        score += min(18.0, max(-10.0, (within_2_pct - 40.0) * 0.35))
    regimes = [r for r in regime_rows if r.get("mae_rs") is not None and (r.get("samples") or 0) >= 8]
    if len(regimes) >= 2:
        maes = [float(r["mae_rs"]) for r in regimes]
        spread = max(maes) - min(maes)
        score -= min(22.0, spread * 2.2)
    timeouts = int(timeout_analysis.get("timeout_trades") or 0)
    if timeouts >= 12:
        score -= 18.0
    elif timeouts >= 8:
        score -= 10.0
    elif timeouts <= 3:
        score += 6.0
    return max(0, min(100, int(score)))


def _compute_model_health(doc: dict[str, Any]) -> dict[str, Any]:
    acc = doc.get("accuracy") or {}
    bias = doc.get("bias") or {}
    cal = doc.get("calibration") or {}
    timeout = doc.get("timeout_analysis") or {}
    regimes = doc.get("regime_breakdown") or []

    accuracy = _mae_accuracy_score(acc.get("mae_rs"), acc.get("within_2_pct"))
    calibration = _calibration_health_score(cal)
    bias_s = _bias_health_score(bias)
    robustness = _robustness_health_score(regimes, timeout, acc.get("within_2_pct"))

    overall = int(
        accuracy * 0.35 + calibration * 0.25 + bias_s * 0.20 + robustness * 0.20
    )
    overall = max(0, min(100, overall))

    if overall >= 85 and accuracy >= 80 and calibration >= 72:
        deployment = "Ready"
        deploy_band = "good"
    elif overall >= 68 and accuracy >= 62 and calibration >= 50:
        deployment = "Paper Trading"
        deploy_band = "warn"
    else:
        deployment = "Not Ready"
        deploy_band = "bad"

    if overall >= 80:
        emoji = "🟢"
    elif overall >= 65:
        emoji = "🟡"
    else:
        emoji = "🔴"

    return {
        "overall": overall,
        "emoji": emoji,
        "overall_band": _score_band(overall),
        "prediction_accuracy": accuracy,
        "prediction_accuracy_band": _score_band(accuracy),
        "calibration": calibration,
        "calibration_band": _score_band(calibration),
        "bias": bias_s,
        "bias_band": _score_band(bias_s),
        "robustness": robustness,
        "robustness_band": _score_band(robustness),
        "deployment": deployment,
        "deployment_band": deploy_band,
    }


def _attach_timeline_and_health(
    doc: dict[str, Any],
    work: pd.DataFrame,
    abs_errors: np.ndarray,
) -> dict[str, Any]:
    ts_col = work["timestamp"].astype(float).values if "timestamp" in work.columns else np.array([])
    timeline = _error_timeline(ts_col, abs_errors)
    doc["error_timeline"] = timeline
    doc["sparklines"] = _build_sparklines(timeline)
    doc["model_health"] = _compute_model_health(doc)
    return doc


def _premium_band_errors(frame: pd.DataFrame, abs_errors: np.ndarray) -> list[dict]:
    from chain_replay_ml.recompute_2_1_ratio import _premium_bucket

    if frame.empty or "ltp" not in frame.columns:
        return []
    buckets: dict[str, list[float]] = {}
    ltps = frame["ltp"].astype(float).values
    for i, err in enumerate(abs_errors):
        if i >= len(ltps):
            break
        band = _premium_bucket(float(ltps[i]))
        buckets.setdefault(band, []).append(float(err))
    rows = []
    for band, errs in sorted(buckets.items(), key=lambda kv: -len(kv[1])):
        rows.append({
            "regime": band.replace("_", " "),
            "samples": len(errs),
            "mae_rs": _round(float(np.mean(errs)) if errs else None),
        })
    return rows[:6]


def _delta_band_errors(frame: pd.DataFrame, abs_errors: np.ndarray) -> list[dict]:
    if frame.empty or "delta" not in frame.columns:
        return []
    d = frame["delta"].astype(float).values
    buckets: list[tuple[str, float, float]] = [
        ("0.00-0.15", 0.00, 0.15),
        ("0.15-0.25", 0.15, 0.25),
        ("0.25-0.40", 0.25, 0.40),
        ("0.40-0.60", 0.40, 0.60),
        ("0.60-1.00", 0.60, 1.01),
    ]
    rows: list[dict[str, Any]] = []
    abs_d = np.abs(d)
    for label, lo, hi in buckets:
        mask = (abs_d >= lo) & (abs_d < hi)
        idx = np.where(mask)[0]
        errs = [float(abs_errors[i]) for i in idx if i < len(abs_errors)]
        rows.append({
            "bucket": label,
            "samples": len(errs),
            "mae_rs": _round(float(np.mean(errs)) if errs else None),
        })
    return rows


def _hold_time_errors(frame: pd.DataFrame, abs_errors: np.ndarray) -> list[dict]:
    if frame.empty:
        return []
    hold_col = None
    for name in ("hold_sec", "hold_time_sec", "holding_sec"):
        if name in frame.columns:
            hold_col = name
            break
    if hold_col is None:
        return []
    hs = frame[hold_col].astype(float).values
    buckets: list[tuple[str, float, float]] = [
        ("0-60s", 0.0, 60.0),
        ("60-180s", 60.0, 180.0),
        ("180-300s", 180.0, 300.0),
        ("300s+", 300.0, 1e12),
    ]
    rows: list[dict[str, Any]] = []
    for label, lo, hi in buckets:
        mask = (hs >= lo) & (hs < hi)
        idx = np.where(mask)[0]
        errs = [float(abs_errors[i]) for i in idx if i < len(abs_errors)]
        rows.append({
            "bucket": label,
            "samples": len(errs),
            "mae_rs": _round(float(np.mean(errs)) if errs else None),
        })
    return rows


def _time_of_day_errors(frame: pd.DataFrame, abs_errors: np.ndarray) -> list[dict]:
    if frame.empty or "timestamp" not in frame.columns:
        return []
    ts = frame["timestamp"].astype(float).values
    slots: dict[str, list[float]] = {}
    sq_slots: dict[str, list[float]] = {}
    for i, tsv in enumerate(ts):
        if i >= len(abs_errors):
            break
        if not np.isfinite(tsv):
            continue
        dt = datetime.fromtimestamp(float(tsv), tz=_IST)
        key = f"{dt.hour:02d}:{(dt.minute // 30) * 30:02d}"
        err = float(abs_errors[i])
        slots.setdefault(key, []).append(err)
        sq_slots.setdefault(key, []).append(err * err)
    rows = []
    for key in sorted(slots.keys()):
        errs = slots[key]
        sq = sq_slots[key]
        rows.append({
            "slot": key,
            "samples": len(errs),
            "mae_rs": _round(float(np.mean(errs)) if errs else None),
            "rmse_rs": _round(float(np.sqrt(np.mean(sq))) if sq else None),
        })
    return rows


def _prediction_rows(
    frame: pd.DataFrame,
    actual_ltp: pd.Series,
    pred_ltp: pd.Series,
    errors: np.ndarray,
    abs_errors: np.ndarray,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    n = min(len(frame), len(actual_ltp), len(pred_ltp), len(errors), len(abs_errors))
    for i in range(n):
        row = frame.iloc[int(i)]
        ts = row.get("timestamp")
        strike = row.get("strike")
        rows.append({
            "entry_ts": float(ts) if ts is not None and pd.notna(ts) else None,
            "timestamp": float(ts) if ts is not None and pd.notna(ts) else None,
            "strike": float(strike) if strike is not None and pd.notna(strike) else None,
            "symbol": str(row.get("symbol") or row.get("option_type") or ""),
            "actual": _round(float(actual_ltp.iloc[int(i)])),
            "predicted": _round(float(pred_ltp.iloc[int(i)])),
            "error_rs": _round(float(errors[int(i)])),
            "abs_error_rs": _round(float(abs_errors[int(i)])),
        })
    return rows


def _direction_accuracy_pct(actual_ltp: pd.Series, pred_ltp: pd.Series, ltp: pd.Series) -> float | None:
    from chain_replay_ml.training.evaluator import directional_accuracy_pct

    if len(actual_ltp) == 0 or len(pred_ltp) == 0 or len(ltp) == 0:
        return None
    da = directional_accuracy_pct(
        actual_ltp.astype(float).to_numpy(),
        pred_ltp.astype(float).to_numpy(),
        ltp.astype(float).to_numpy(),
    )
    return _round(da, 1) if da is not None else None


def _regime_errors(frame: pd.DataFrame, errors: np.ndarray) -> list[dict]:
    if frame.empty or "spot_change_5m" not in frame.columns:
        return []
    regimes: dict[str, list[float]] = {"bullish": [], "bearish": [], "sideways": []}
    sc = frame["spot_change_5m"].astype(float).values
    for i, err in enumerate(errors):
        if i >= len(sc):
            break
        regimes[_regime_label(float(sc[i]) if np.isfinite(sc[i]) else None)].append(abs(float(err)))
    rows = []
    for name in ("bullish", "bearish", "sideways"):
        arr = regimes[name]
        rows.append({
            "regime": name.capitalize(),
            "samples": len(arr),
            "mae_rs": _round(float(np.mean(arr)) if arr else None),
        })
    return rows


def _load_registry_feature_importance(data_dir: str, model_name: str) -> list[dict]:
    paths = model_artifact_paths(data_dir, model_name)
    csv_path = paths.get("feature_importance_csv")
    if not csv_path or not os.path.isfile(csv_path):
        return []
    try:
        df = pd.read_csv(csv_path)
    except (OSError, pd.errors.EmptyDataError):
        return []
    col_feat = "feature" if "feature" in df.columns else df.columns[0]
    col_imp = "importance_pct" if "importance_pct" in df.columns else (
        "importance" if "importance" in df.columns else df.columns[-1]
    )
    rows = []
    for _, row in df.head(12).iterrows():
        rows.append({
            "feature": str(row[col_feat]),
            "importance_pct": _round(float(row[col_imp]), 1),
        })
    return rows


def _load_stamp_feature_importance(stamp: str) -> list[dict]:
    models_dir = os.path.join(_CHART_DIR, "data", "ml_models")
    merged: dict[str, float] = {}
    for band in ("A", "B", "C"):
        report_path = os.path.join(models_dir, f"training_report_option_delta_{band}_{stamp}.json")
        if not os.path.isfile(report_path):
            continue
        try:
            with open(report_path, encoding="utf-8") as fh:
                report = json.load(fh)
        except (OSError, json.JSONDecodeError):
            continue
        imp = (report.get("feature_importances") or {}).get("classifier") or []
        for item in imp:
            if isinstance(item, dict):
                feat = str(item.get("feature") or "")
                val = float(item.get("importance") or item.get("gain") or 0)
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                feat = str(item[0])
                val = float(item[1])
            else:
                continue
            if not feat:
                continue
            merged[feat] = merged.get(feat, 0.0) + val
    if not merged:
        return []
    total = sum(merged.values()) or 1.0
    rows = sorted(
        [{"feature": k, "importance_pct": _round(v / total * 100, 1)} for k, v in merged.items()],
        key=lambda r: r["importance_pct"] or 0,
        reverse=True,
    )
    return rows[:12]


def _analysis_from_scored_frame(
    frame: pd.DataFrame,
    *,
    date_str: str,
    stamp: str,
    signal_trades: list[dict],
    entered_trades: list[dict],
    engine: str,
    target_label: str,
) -> dict[str, Any]:
    if frame.empty:
        return _empty_analysis(engine=engine, target_label=target_label)

    work = frame.dropna(subset=["ltp", "pred_max_return", "target_max_return_5m_pct"]).copy()
    if work.empty:
        return _empty_analysis(engine=engine, target_label=target_label)

    actual_ltp = work["ltp"].astype(float) * (1.0 + work["target_max_return_5m_pct"].astype(float) / 100.0)
    pred_ltp = work["ltp"].astype(float) * (1.0 + work["pred_max_return"].astype(float) / 100.0)
    errors = (pred_ltp - actual_ltp).astype(float).values
    abs_errors = np.abs(errors)

    reg_metrics = evaluate_regression(actual_ltp.values, pred_ltp.values)
    p95 = float(np.percentile(abs_errors, 95)) if len(abs_errors) else None
    over = int((errors > 0).sum())
    under = int((errors < 0).sum())
    total_n = len(errors)

    cal_trades = []
    for _, row in work.iterrows():
        cal_trades.append({
            "p_hit": float(row["P_hit"]) if pd.notna(row.get("P_hit")) else 0.0,
            "outcome_type": "target" if float(row["target_max_return_5m_pct"]) >= 7.0 else "timeout",
        })

    prediction_rows = _prediction_rows(work, actual_ltp, pred_ltp, errors, abs_errors)
    worst_rows = sorted(
        prediction_rows,
        key=lambda r: abs(float(r.get("error_rs") or 0.0)),
        reverse=True,
    )[:100]
    best_rows = sorted(
        prediction_rows,
        key=lambda r: abs(float(r.get("error_rs") or 0.0)),
    )[:100]

    cal_buckets = _pick_calibration(signal_trades, cal_trades, abs_errors, prefer_residual=False)
    conf_profit, conf_profit_mode = _pick_confidence_vs_profit(signal_trades)
    within_2_pct = _pct(int((abs_errors <= 2.0).sum()), total_n)
    shap_rows = _shap_for_fold_stamp(stamp, work)
    feat_gain = _load_stamp_feature_importance(stamp)

    direction_accuracy = _direction_accuracy_pct(actual_ltp, pred_ltp, work["ltp"].astype(float))
    over_pct = _pct(over, total_n)
    under_pct = _pct(under, total_n)
    mean_bias = _round(float(np.mean(errors)) if len(errors) else None)
    summary = _build_summary(
        mae=reg_metrics.get("mae"),
        cal_label=cal_buckets.get("score_label") or "Unknown",
        over_pct=over_pct,
        timeout_count=sum(1 for t in entered_trades if t.get("outcome_type") == "timeout"),
        regime_rows=_regime_errors(work, errors),
        within_2_pct=within_2_pct,
        calibration_mode=cal_buckets.get("mode"),
        has_shap=bool(shap_rows),
    )

    doc = {
        "engine": engine,
        "target_label": target_label,
        "sample_count": total_n,
        "accuracy": {
            "mae_rs": reg_metrics.get("mae"),
            "rmse_rs": reg_metrics.get("rmse"),
            "mape_pct": reg_metrics.get("mape"),
            "median_error_rs": reg_metrics.get("median_error"),
            "p95_error_rs": _round(p95),
            "max_error_rs": reg_metrics.get("max_error"),
            "within_2_pct": within_2_pct,
        },
        "bias": {
            "over_predicted_pct": over_pct,
            "under_predicted_pct": under_pct,
            "average_bias_rs": mean_bias,
        },
        "prediction_summary": {
            "rmse_rs": reg_metrics.get("rmse"),
            "mae_rs": reg_metrics.get("mae"),
            "direction_accuracy_pct": direction_accuracy,
            "mean_prediction_bias_rs": mean_bias,
            "overprediction_pct": over_pct,
            "underprediction_pct": under_pct,
        },
        "calibration": cal_buckets,
        "error_distribution": _error_distribution_rs(abs_errors),
        "worst_predictions": worst_rows,
        "best_predictions": best_rows,
        "timeout_analysis": _timeout_analysis(entered_trades or signal_trades, date_str),
        "confidence_vs_profit": conf_profit,
        "confidence_vs_profit_mode": conf_profit_mode,
        "regime_breakdown": _regime_errors(work, errors),
        "premium_band_breakdown": _premium_band_errors(work, abs_errors),
        "delta_band_breakdown": _delta_band_errors(work, abs_errors),
        "hold_time_breakdown": _hold_time_errors(work, abs_errors),
        "time_of_day_breakdown": _time_of_day_errors(work, abs_errors),
        "feature_importance": feat_gain,
        "shap_importance": shap_rows,
        "summary": summary,
    }
    return _attach_timeline_and_health(doc, work, abs_errors)


def _analysis_from_registry(
    data_dir: str,
    model_name: str,
    date_str: str,
    signal_trades: list[dict],
    entered_trades: list[dict],
    *,
    expiry_hint: str | None = None,
    scored_df: pd.DataFrame | None = None,
) -> dict[str, Any] | None:
    loaded = None
    try:
        from chain_replay_ml.replay_feature_scoring import (
            load_model_inference_config,
            load_scoring_day_frame,
        )
        from chain_replay_ml.training.model_runtime import load_prediction_model

        loaded = load_model_inference_config(data_dir, model_name)
        if not loaded:
            return None

        target = loaded["target"]
        features = loaded["features"]
        model_path = loaded["model_path"]
        algorithm = loaded.get("algorithm") or loaded["config"].get("algorithm")

        if scored_df is not None and not scored_df.empty:
            day_df = scored_df
        else:
            day_df, _config, _coverage = load_scoring_day_frame(
                data_dir, model_name, date_str, expiry_hint=expiry_hint,
            )
        if day_df.empty:
            return None

        missing = [c for c in features if c not in day_df.columns]
        if missing:
            return None
        if target not in day_df.columns:
            return None

        model = load_prediction_model(model_path, algorithm)

        if "pred_ltp" in day_df.columns:
            X = day_df[features].dropna()
            if X.empty:
                return None
            y_pred = day_df.loc[X.index, "pred_ltp"].astype(float)
        else:
            X = day_df[features].dropna()
            if X.empty:
                return None
            y_pred = pd.Series(model.predict(X), index=X.index)
        y_true = day_df.loc[X.index, target].astype(float)

        valid = y_true.notna() & y_pred.notna() & np.isfinite(y_true) & np.isfinite(y_pred)
        work = day_df.loc[X.index[valid]].copy()
        y_true_v = y_true.loc[valid].astype(float)
        y_pred_v = y_pred.loc[valid].astype(float)
        errors = (y_pred_v - y_true_v).values
        abs_errors = np.abs(errors)

        reg_metrics = evaluate_regression(y_true_v.values, y_pred_v.values)
        p95 = float(np.percentile(abs_errors, 95)) if len(abs_errors) else None
        over = int((errors > 0).sum())
        under = int((errors < 0).sum())
        total_n = len(errors)

        prediction_rows = _prediction_rows(work, y_true_v, y_pred_v, errors, abs_errors)
        worst_rows = sorted(
            prediction_rows,
            key=lambda r: abs(float(r.get("error_rs") or 0.0)),
            reverse=True,
        )[:100]
        best_rows = sorted(
            prediction_rows,
            key=lambda r: abs(float(r.get("error_rs") or 0.0)),
        )[:100]

        cal_buckets = _pick_calibration(signal_trades, [], abs_errors, prefer_residual=True)
        conf_profit, conf_profit_mode = _pick_confidence_vs_profit(signal_trades)
        within_2_pct = _pct(int((abs_errors <= 2.0).sum()), total_n)
        feat_gain = _load_registry_feature_importance(data_dir, model_name)
        shap_rows = _compute_shap_importance(model, work[features], features)

        direction_accuracy = None
        if "ltp" in work.columns:
            direction_accuracy = _direction_accuracy_pct(y_true_v, y_pred_v, work["ltp"].astype(float))
        over_pct = _pct(over, total_n)
        under_pct = _pct(under, total_n)
        mean_bias = _round(float(np.mean(errors)) if len(errors) else None)
        summary = _build_summary(
            mae=reg_metrics.get("mae"),
            cal_label=cal_buckets.get("score_label") or "Unknown",
            over_pct=over_pct,
            timeout_count=sum(1 for t in entered_trades if t.get("outcome_type") == "timeout"),
            regime_rows=_regime_errors(work, errors),
            within_2_pct=within_2_pct,
            calibration_mode=cal_buckets.get("mode"),
            has_shap=bool(shap_rows),
        )

        doc = {
            "engine": "registry",
            "target_label": target,
            "sample_count": total_n,
            "accuracy": {
                "mae_rs": reg_metrics.get("mae"),
                "rmse_rs": reg_metrics.get("rmse"),
                "mape_pct": reg_metrics.get("mape"),
                "median_error_rs": reg_metrics.get("median_error"),
                "p95_error_rs": _round(p95),
                "max_error_rs": reg_metrics.get("max_error"),
                "within_2_pct": within_2_pct,
            },
            "bias": {
                "over_predicted_pct": over_pct,
                "under_predicted_pct": under_pct,
                "average_bias_rs": mean_bias,
            },
            "prediction_summary": {
                "rmse_rs": reg_metrics.get("rmse"),
                "mae_rs": reg_metrics.get("mae"),
                "direction_accuracy_pct": direction_accuracy,
                "mean_prediction_bias_rs": mean_bias,
                "overprediction_pct": over_pct,
                "underprediction_pct": under_pct,
            },
            "calibration": cal_buckets,
            "error_distribution": _error_distribution_rs(abs_errors),
            "worst_predictions": worst_rows,
            "best_predictions": best_rows,
            "timeout_analysis": _timeout_analysis(entered_trades or signal_trades, date_str),
            "confidence_vs_profit": conf_profit,
            "confidence_vs_profit_mode": conf_profit_mode,
            "regime_breakdown": _regime_errors(work, errors),
            "premium_band_breakdown": _premium_band_errors(work, abs_errors),
            "delta_band_breakdown": _delta_band_errors(work, abs_errors),
            "hold_time_breakdown": _hold_time_errors(work, abs_errors),
            "time_of_day_breakdown": _time_of_day_errors(work, abs_errors),
            "feature_importance": feat_gain,
            "shap_importance": shap_rows,
            "summary": summary,
        }
        return _attach_timeline_and_health(doc, work, abs_errors)
    except Exception:
        import traceback
        traceback.print_exc()
        return None


def _build_summary(
    *,
    mae: float | None,
    cal_label: str,
    over_pct: float | None,
    timeout_count: int,
    regime_rows: list[dict],
    within_2_pct: float | None = None,
    calibration_mode: str | None = None,
    has_shap: bool = False,
) -> dict[str, Any]:
    strengths: list[str] = []
    weaknesses: list[str] = []
    if mae is not None and mae <= 3.0:
        strengths.append("Low prediction error on this replay session")
    elif mae is not None and mae <= 6.0:
        strengths.append(f"Moderate MAE (₹{mae:.2f}) — acceptable for short-horizon LTP")
    elif mae is not None:
        weaknesses.append(f"Elevated MAE (₹{mae:.2f}) on replay day")
    if within_2_pct is not None and within_2_pct >= 50:
        strengths.append(f"{within_2_pct:.0f}% of predictions within ₹2")
    elif within_2_pct is not None and within_2_pct < 30:
        weaknesses.append(f"Only {within_2_pct:.0f}% of predictions within ₹2")
    if cal_label in ("Excellent", "Good"):
        mode_note = "P(hit)" if calibration_mode == "phit" else "residual error"
        strengths.append(f"Confidence calibration is reliable ({mode_note})")
    else:
        weaknesses.append("Confidence buckets diverge from expected outcomes")
    if over_pct is not None and over_pct > 58:
        weaknesses.append("Model tends to over-predict (optimistic bias)")
    elif over_pct is not None and over_pct < 42:
        strengths.append("Balanced or conservative prediction bias")
    if timeout_count >= 8:
        weaknesses.append("High timeout count — review target horizon or entries")
    if regime_rows:
        best = min((r for r in regime_rows if r.get("mae_rs") is not None), key=lambda r: r["mae_rs"], default=None)
        worst = max((r for r in regime_rows if r.get("mae_rs") is not None), key=lambda r: r["mae_rs"], default=None)
        if best and worst and best["regime"] != worst["regime"]:
            strengths.append(f"Strongest in {best['regime'].lower()} market conditions")
            weaknesses.append(f"Weaker in {worst['regime'].lower()} market conditions")
    if has_shap:
        strengths.append("SHAP attribution available for this session")
    if not strengths:
        strengths.append("Session scored successfully — review error distribution")
    if not weaknesses:
        weaknesses.append("No major weakness flags on this session")
    if cal_label == "Excellent" and (mae or 99) <= 5 and (within_2_pct or 0) >= 45:
        deploy = "Suitable for controlled paper trading — monitor timeouts live"
    elif cal_label == "Needs Improvement" or (mae or 0) > 8:
        deploy = "Needs tuning before live deployment"
    else:
        deploy = "Suitable for further replay validation before paper trading"
    return {
        "strengths": strengths[:5],
        "weaknesses": weaknesses[:5],
        "recommendation": deploy,
    }


def _empty_analysis(*, engine: str, target_label: str) -> dict[str, Any]:
    return {
        "engine": engine,
        "target_label": target_label,
        "sample_count": 0,
        "accuracy": {},
        "bias": {},
        "calibration": {"buckets": [], "score_label": "Unknown", "score_value": None},
        "prediction_summary": {},
        "error_distribution": [],
        "worst_predictions": [],
        "best_predictions": [],
        "timeout_analysis": _timeout_analysis([], ""),
        "confidence_vs_profit": [],
        "regime_breakdown": [],
        "premium_band_breakdown": [],
        "delta_band_breakdown": [],
        "hold_time_breakdown": [],
        "time_of_day_breakdown": [],
        "feature_importance": [],
        "shap_importance": [],
        "error_timeline": [],
        "sparklines": {},
        "model_health": {
            "overall": 0,
            "emoji": "🔴",
            "overall_band": "bad",
            "prediction_accuracy": 0,
            "prediction_accuracy_band": "bad",
            "calibration": 0,
            "calibration_band": "bad",
            "bias": 0,
            "bias_band": "bad",
            "robustness": 0,
            "robustness_band": "bad",
            "deployment": "Not Ready",
            "deployment_band": "bad",
        },
        "summary": {
            "strengths": ["No scored rows for this session"],
            "weaknesses": ["Run feature export and Load Model first"],
            "recommendation": "Insufficient data",
        },
    }


def compute_model_analysis(
    date_str: str,
    stamp: str,
    *,
    data_dir: str,
    model_name: str | None = None,
    premium_trades: list[dict] | None = None,
    position_limit: int = 1,
    expiry_hint: str | None = None,
    scored_df: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Build replay-day model analysis payload using the registry model package."""
    signal_trades = premium_trades or []
    entered = simulate_positions(signal_trades, position_limit if position_limit > 0 else 9999)

    name = str(model_name or "").strip()
    if not name:
        doc = _empty_analysis(engine="registry", target_label="—")
        doc["summary"] = {
            "strengths": [],
            "weaknesses": ["Select a model and click Load Model"],
            "recommendation": "No model selected",
        }
        return doc

    registry_doc = _analysis_from_registry(
        data_dir, name, date_str, signal_trades, entered,
        expiry_hint=expiry_hint,
        scored_df=scored_df,
    )
    if registry_doc:
        registry_doc["model_name"] = name
        return registry_doc

    doc = _empty_analysis(engine="registry", target_label="—")
    doc["model_name"] = name
    doc["summary"] = {
        "strengths": [],
        "weaknesses": [f"Could not score registry model {name} for {date_str}"],
        "recommendation": "Check model package and dataset day coverage",
    }
    return doc

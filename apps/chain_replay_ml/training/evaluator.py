"""Post-training evaluation metrics."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)

# Premium bands for LTP / actual premium (₹). Half-open intervals except the last.
PREMIUM_METRIC_BANDS: tuple[tuple[str, float, float | None], ...] = (
    ("0-15", 0.0, 15.0),
    ("15-30", 15.0, 30.0),
    ("30-50", 30.0, 50.0),
    ("50-100", 50.0, 100.0),
    ("100-200", 100.0, 200.0),
    ("200+", 200.0, None),
)

_PREMIUM_EPS = 1e-9

# Canonical Model Quality: endpoint hit = |pred−actual|/|actual| ≤ tolerance.
ENDPOINT_HIT_TOLERANCE_PCT = 5.0


def resolve_ltp_baseline(df: pd.DataFrame | None) -> pd.Series | None:
    """Resolve per-row LTP baseline for directional accuracy.

    Prefers an explicit ``ltp`` column; otherwise derives
    ``ltp_to_spot_ratio * spot`` when both are present (datasets without raw LTP).
    """
    if df is None or len(df) == 0:
        return None
    if "ltp" in df.columns:
        series = pd.to_numeric(df["ltp"], errors="coerce")
        if series.notna().any():
            return series
    if "ltp_to_spot_ratio" in df.columns and "spot" in df.columns:
        ratio = pd.to_numeric(df["ltp_to_spot_ratio"], errors="coerce")
        spot = pd.to_numeric(df["spot"], errors="coerce")
        derived = ratio * spot
        if derived.notna().any():
            return derived
    return None


def resolve_ltp_baseline_from_frames(*frames: pd.DataFrame | None) -> pd.Series | None:
    """Resolve LTP baseline using columns spread across one or more position-aligned frames.

    Merges by row position (not index). When frame lengths differ, uses the shortest
    contributing frame so columns align to the same rows.
    """
    parts: list[pd.DataFrame] = []
    for frame in frames:
        if frame is None or len(frame) == 0:
            continue
        cols = [c for c in ("ltp", "ltp_to_spot_ratio", "spot") if c in frame.columns]
        if cols:
            parts.append(frame[cols].reset_index(drop=True))
    if not parts:
        return None
    n = min(len(part) for part in parts)
    if n <= 0:
        return None
    aligned = [part.iloc[:n] for part in parts]
    merged = pd.concat(aligned, axis=1)
    merged = merged.loc[:, ~merged.columns.duplicated()]
    return resolve_ltp_baseline(merged)


def _finite_mask(*arrays: np.ndarray) -> np.ndarray:
    mask = np.ones(len(arrays[0]), dtype=bool) if arrays else np.array([], dtype=bool)
    for arr in arrays:
        mask &= np.isfinite(arr)
    return mask


def _mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = np.abs(y_true) > _PREMIUM_EPS
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def premium_mae_pct(y_true: np.ndarray, y_pred: np.ndarray) -> float | None:
    """mean(|pred - actual| / actual) × 100 over rows with |actual| > eps."""
    mask = _finite_mask(y_true, y_pred) & (np.abs(y_true) > _PREMIUM_EPS)
    if not mask.any():
        return None
    return float(np.mean(np.abs(y_pred[mask] - y_true[mask]) / np.abs(y_true[mask])) * 100)


def premium_rmse_pct(y_true: np.ndarray, y_pred: np.ndarray) -> float | None:
    """sqrt(mean(((pred - actual) / actual)^2)) × 100 over rows with |actual| > eps."""
    mask = _finite_mask(y_true, y_pred) & (np.abs(y_true) > _PREMIUM_EPS)
    if not mask.any():
        return None
    rel = (y_pred[mask] - y_true[mask]) / np.abs(y_true[mask])
    return float(np.sqrt(np.mean(np.square(rel))) * 100)


def premium_band_performance(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    baseline: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    """Group predictions by actual premium and compute MAE/RMSE/premium%/direction per band."""
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    base = np.asarray(baseline, dtype=float) if baseline is not None else None
    rows: list[dict[str, Any]] = []
    for label, lo, hi in PREMIUM_METRIC_BANDS:
        if hi is None:
            band_mask = _finite_mask(yt, yp) & (yt >= lo)
        else:
            band_mask = _finite_mask(yt, yp) & (yt >= lo) & (yt < hi)
        n = int(band_mask.sum())
        if n <= 0:
            rows.append({
                "band": label,
                "band_label": f"₹{label}",
                "samples": 0,
                "mae": None,
                "rmse": None,
                "premium_mae_pct": None,
                "premium_rmse_pct": None,
                "directional_accuracy_pct": None,
            })
            continue
        yt_b = yt[band_mask]
        yp_b = yp[band_mask]
        base_b = base[band_mask] if base is not None and len(base) == len(yt) else None
        mae = float(mean_absolute_error(yt_b, yp_b))
        rmse = float(np.sqrt(mean_squared_error(yt_b, yp_b)))
        p_mae = premium_mae_pct(yt_b, yp_b)
        p_rmse = premium_rmse_pct(yt_b, yp_b)
        da = _directional_accuracy(yt_b, yp_b, base_b)
        rows.append({
            "band": label,
            "band_label": f"₹{label}",
            "samples": n,
            "mae": round(mae, 6),
            "rmse": round(rmse, 6),
            "premium_mae_pct": round(p_mae, 4) if p_mae is not None else None,
            "premium_rmse_pct": round(p_rmse, 4) if p_rmse is not None else None,
            "directional_accuracy_pct": round(da, 2) if da is not None else None,
        })
    return rows


def aggregate_premium_band_performance(band_lists: list[list[dict[str, Any]]]) -> list[dict[str, Any]]:
    """Sample-weighted merge of per-fold (or per-split) premium band rows."""
    by_band: dict[str, dict[str, Any]] = {
        label: {
            "band": label,
            "band_label": f"₹{label}",
            "samples": 0,
            "mae_num": 0.0,
            "rmse_num": 0.0,
            "premium_mae_num": 0.0,
            "premium_rmse_num": 0.0,
            "dir_num": 0.0,
            "mae_w": 0,
            "rmse_w": 0,
            "premium_mae_w": 0,
            "premium_rmse_w": 0,
            "dir_w": 0,
        }
        for label, _, _ in PREMIUM_METRIC_BANDS
    }
    for bands in band_lists:
        for row in bands or []:
            label = str(row.get("band") or "")
            if label not in by_band:
                continue
            n = int(row.get("samples") or 0)
            if n <= 0:
                continue
            agg = by_band[label]
            agg["samples"] += n
            for key, num_key, w_key in (
                ("mae", "mae_num", "mae_w"),
                ("rmse", "rmse_num", "rmse_w"),
                ("premium_mae_pct", "premium_mae_num", "premium_mae_w"),
                ("premium_rmse_pct", "premium_rmse_num", "premium_rmse_w"),
                ("directional_accuracy_pct", "dir_num", "dir_w"),
            ):
                val = row.get(key)
                if val is None:
                    continue
                try:
                    num = float(val)
                except (TypeError, ValueError):
                    continue
                if num != num:
                    continue
                agg[num_key] += num * n
                agg[w_key] += n

    out: list[dict[str, Any]] = []
    for label, _, _ in PREMIUM_METRIC_BANDS:
        agg = by_band[label]
        samples = int(agg["samples"])

        def _wavg(num_key: str, w_key: str, digits: int) -> float | None:
            w = int(agg[w_key])
            if w <= 0:
                return None
            return round(float(agg[num_key]) / w, digits)

        out.append({
            "band": label,
            "band_label": f"₹{label}",
            "samples": samples,
            "mae": _wavg("mae_num", "mae_w", 6),
            "rmse": _wavg("rmse_num", "rmse_w", 6),
            "premium_mae_pct": _wavg("premium_mae_num", "premium_mae_w", 4),
            "premium_rmse_pct": _wavg("premium_rmse_num", "premium_rmse_w", 4),
            "directional_accuracy_pct": _wavg("dir_num", "dir_w", 2),
        })
    return out


def direction_correct_flag(
    predicted: float,
    actual: float,
    baseline: float | None,
) -> int | None:
    """Per-row Model Quality direction flag (canonical).

    Compares sign(predicted − baseline) vs sign(actual − baseline).
    When ``baseline`` is None, treats the target as already relative to zero
    (e.g. ORMP return labels): sign(predicted) vs sign(actual).
    Returns None when the actual move is flat so aggregates match
    ``directional_accuracy_pct`` (flat actuals excluded from the denominator).
    """
    try:
        p = float(predicted)
        a = float(actual)
    except (TypeError, ValueError):
        return None
    if not (np.isfinite(p) and np.isfinite(a)):
        return None
    if baseline is None:
        actual_dir = float(np.sign(a))
        pred_dir = float(np.sign(p))
    else:
        try:
            b = float(baseline)
        except (TypeError, ValueError):
            return None
        if not np.isfinite(b):
            return None
        actual_dir = float(np.sign(a - b))
        pred_dir = float(np.sign(p - b))
    if actual_dir == 0.0:
        return None
    return 1 if pred_dir == actual_dir else 0


def directional_accuracy_pct(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    baseline: np.ndarray | None,
    *,
    relative_to_zero: bool = False,
) -> float | None:
    """Canonical Model Quality Direction Accuracy (0–100).

    With LTP baseline: actual_dir = sign(actual − baseline), pred_dir = sign(pred − baseline).
    With ``relative_to_zero=True`` (ORMP return / residual targets): compare sign(actual)
    vs sign(pred) — baseline is implicitly 0.
    Flat actuals (actual_dir == 0) are excluded.
    """
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    if baseline is None:
        if not relative_to_zero:
            return None
        actual_dir = np.sign(yt)
        pred_dir = np.sign(yp)
    else:
        base = np.asarray(baseline, dtype=float)
        actual_dir = np.sign(yt - base)
        pred_dir = np.sign(yp - base)
    valid = np.isfinite(actual_dir) & np.isfinite(pred_dir) & (actual_dir != 0)
    if not valid.any():
        return None
    return float(np.mean(actual_dir[valid] == pred_dir[valid]) * 100)


# Backward-compatible private alias
_directional_accuracy = directional_accuracy_pct


def endpoint_hit_rate_pct(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    tolerance_pct: float = ENDPOINT_HIT_TOLERANCE_PCT,
) -> float | None:
    """Canonical Model Quality Endpoint Hit Rate (0–100).

    Share of rows with |pred − actual| / |actual| × 100 ≤ tolerance_pct.
    Denominator is |actual| (future/target premium). Default tolerance 5%.
    Rows with non-finite values or |actual| ≤ 1e-9 are excluded.
    """
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(yt) & np.isfinite(yp) & (np.abs(yt) > _PREMIUM_EPS)
    if not mask.any():
        return None
    rel = np.abs(yp[mask] - yt[mask]) / np.abs(yt[mask]) * 100.0
    return float(np.mean(rel <= float(tolerance_pct)) * 100.0)


# Legacy alias used by holdout / older callers
premium_hit_rate_pct = endpoint_hit_rate_pct


def _classification_extras(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Accuracy / F1 for low-cardinality targets (classification-style labels)."""
    uniq = np.unique(y_true)
    if len(uniq) > 5:
        return {}
    labels = sorted(float(v) for v in uniq.tolist())
    label_to_idx = {v: i for i, v in enumerate(labels)}
    yt = np.array([label_to_idx.get(float(v), 0) for v in y_true], dtype=int)
    yp_raw = np.array([min(labels, key=lambda l: abs(l - float(v))) for v in y_pred], dtype=float)
    yp = np.array([label_to_idx[v] for v in yp_raw], dtype=int)
    use_binary_f1 = len(labels) == 2
    return {
        "accuracy_pct": round(float(accuracy_score(yt, yp) * 100), 2),
        "f1_pct": round(float(f1_score(yt, yp, average="binary" if use_binary_f1 else "macro", zero_division=0) * 100), 2),
    }


def prediction_bias(y_true: np.ndarray, y_pred: np.ndarray) -> float | None:
    """Mean signed error: mean(pred - actual). Positive => overprediction."""
    mask = _finite_mask(y_true, y_pred)
    if not mask.any():
        return None
    return float(np.mean(y_pred[mask] - y_true[mask]))


def prediction_bias_pct(y_true: np.ndarray, y_pred: np.ndarray) -> float | None:
    """Mean relative bias: mean((pred - actual) / |actual|) × 100."""
    mask = _finite_mask(y_true, y_pred) & (np.abs(y_true) > _PREMIUM_EPS)
    if not mask.any():
        return None
    return float(np.mean((y_pred[mask] - y_true[mask]) / np.abs(y_true[mask])) * 100)


def p95_abs_error(y_true: np.ndarray, y_pred: np.ndarray) -> float | None:
    """95th percentile of absolute error."""
    mask = _finite_mask(y_true, y_pred)
    if not mask.any():
        return None
    return float(np.percentile(np.abs(y_pred[mask] - y_true[mask]), 95))


def evaluate_classification(
    y_true: pd.Series | np.ndarray,
    y_pred: np.ndarray,
    *,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Binary Hit metrics. ``y_pred`` may be class labels or P(Hit) probabilities."""
    yt = np.asarray(y_true, dtype=float)
    yp_raw = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(yt) & np.isfinite(yp_raw)
    yt, yp_raw = yt[mask], yp_raw[mask]
    if len(yt) == 0:
        return {
            "accuracy_pct": None,
            "precision_pct": None,
            "recall_pct": None,
            "f1_pct": None,
            "roc_auc": None,
            "pr_auc": None,
            "brier_score": None,
            "specificity_pct": None,
            "confusion": {"tn": 0, "fp": 0, "fn": 0, "tp": 0},
            "threshold": float(threshold),
            "n_samples": 0,
            "actual_positives": 0,
            "actual_negatives": 0,
            "predicted_positives": 0,
            "predicted_negatives": 0,
            "positive_rate_pct": None,
            "predicted_positive_rate_pct": None,
            "mean_prob_actual_positive": None,
            "mean_prob_actual_negative": None,
            "calibration": [],
            "threshold_analysis": [],
        }

    yt_bin = (yt >= 0.5).astype(int)
    # Probabilities are in [0, 1]; hard labels are 0/1
    looks_like_proba = float(np.nanmax(yp_raw) - np.nanmin(yp_raw)) > 1e-9 and float(np.nanmax(yp_raw)) <= 1.0 + 1e-6
    if looks_like_proba or float(np.nanmax(yp_raw)) <= 1.0:
        y_prob = np.clip(yp_raw, 0.0, 1.0)
        y_hat = (y_prob >= float(threshold)).astype(int)
    else:
        y_hat = (yp_raw >= 0.5).astype(int)
        y_prob = y_hat.astype(float)

    tn = int(((yt_bin == 0) & (y_hat == 0)).sum())
    fp = int(((yt_bin == 0) & (y_hat == 1)).sum())
    fn = int(((yt_bin == 1) & (y_hat == 0)).sum())
    tp = int(((yt_bin == 1) & (y_hat == 1)).sum())

    metrics: dict[str, Any] = {
        "accuracy_pct": round(float(accuracy_score(yt_bin, y_hat) * 100), 2),
        "precision_pct": round(float(precision_score(yt_bin, y_hat, zero_division=0) * 100), 2),
        "recall_pct": round(float(recall_score(yt_bin, y_hat, zero_division=0) * 100), 2),
        "f1_pct": round(float(f1_score(yt_bin, y_hat, average="binary", zero_division=0) * 100), 2),
        "specificity_pct": (
            round(100.0 * tn / (tn + fp), 2) if (tn + fp) > 0 else None
        ),
        "confusion": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "threshold": float(threshold),
        "n_samples": int(len(yt_bin)),
        "actual_positives": int(yt_bin.sum()),
        "actual_negatives": int((yt_bin == 0).sum()),
        "predicted_positives": int(y_hat.sum()),
        "predicted_negatives": int((y_hat == 0).sum()),
        "positive_rate_pct": round(float(yt_bin.mean() * 100), 2),
        "predicted_positive_rate_pct": round(float(y_hat.mean() * 100), 2),
        "mean_prob_actual_positive": (
            round(float(y_prob[yt_bin == 1].mean()), 6)
            if int((yt_bin == 1).sum()) > 0
            else None
        ),
        "mean_prob_actual_negative": (
            round(float(y_prob[yt_bin == 0].mean()), 6)
            if int((yt_bin == 0).sum()) > 0
            else None
        ),
    }
    try:
        if len(np.unique(yt_bin)) >= 2:
            metrics["roc_auc"] = round(float(roc_auc_score(yt_bin, y_prob)), 6)
            metrics["pr_auc"] = round(float(average_precision_score(yt_bin, y_prob)), 6)
        else:
            metrics["roc_auc"] = None
            metrics["pr_auc"] = None
    except ValueError:
        metrics["roc_auc"] = None
        metrics["pr_auc"] = None

    try:
        metrics["brier_score"] = round(float(brier_score_loss(yt_bin, y_prob)), 6)
    except ValueError:
        metrics["brier_score"] = None

    metrics["calibration"] = _probability_calibration_bins(yt_bin, y_prob)
    metrics["threshold_analysis"] = threshold_analysis(yt_bin, y_prob)
    return metrics


DEFAULT_THRESHOLD_SWEEP: tuple[float, ...] = (
    0.30,
    0.40,
    0.50,
    0.60,
    0.70,
    0.80,
    0.90,
)


def threshold_row_from_confusion(
    *,
    threshold: float,
    tp: int,
    fp: int,
    fn: int,
    tn: int,
    n_days: int | None = None,
) -> dict[str, Any]:
    """Build one Threshold Analysis row from confusion counts."""
    tp_i, fp_i, fn_i, tn_i = int(tp), int(fp), int(fn), int(tn)
    n = tp_i + fp_i + fn_i + tn_i
    prec = (100.0 * tp_i / (tp_i + fp_i)) if (tp_i + fp_i) > 0 else None
    rec = (100.0 * tp_i / (tp_i + fn_i)) if (tp_i + fn_i) > 0 else None
    spec = (100.0 * tn_i / (tn_i + fp_i)) if (tn_i + fp_i) > 0 else None
    acc = (100.0 * (tp_i + tn_i) / n) if n > 0 else None
    if prec is not None and rec is not None and (prec + rec) > 0:
        f1 = 2.0 * prec * rec / (prec + rec)
    else:
        f1 = 0.0 if n else None
    buy_signals = tp_i + fp_i
    # Hit Rate = win rate among BUY signals (same as Precision).
    hit_rate = prec
    good_kept = rec
    good_filtered = (100.0 * fn_i / (tp_i + fn_i)) if (tp_i + fn_i) > 0 else None
    bad_filtered = spec
    bad_passed = (100.0 * fp_i / (tn_i + fp_i)) if (tn_i + fp_i) > 0 else None
    days = int(n_days) if n_days is not None and int(n_days) > 0 else None
    trades_per_day = (
        round(float(buy_signals) / float(days), 2) if days is not None else None
    )
    return {
        "threshold": round(float(threshold), 2),
        "precision_pct": round(prec, 2) if prec is not None else None,
        "recall_pct": round(rec, 2) if rec is not None else None,
        "f1_pct": round(f1, 2) if f1 is not None else None,
        "accuracy_pct": round(acc, 2) if acc is not None else None,
        "hit_rate_pct": round(hit_rate, 2) if hit_rate is not None else None,
        "specificity_pct": round(spec, 2) if spec is not None else None,
        "good_trades_kept_pct": (
            round(good_kept, 2) if good_kept is not None else None
        ),
        "good_trades_filtered_pct": (
            round(good_filtered, 2) if good_filtered is not None else None
        ),
        "bad_trades_filtered_pct": (
            round(bad_filtered, 2) if bad_filtered is not None else None
        ),
        "bad_trades_passed_pct": (
            round(bad_passed, 2) if bad_passed is not None else None
        ),
        "buy_signals": int(buy_signals),
        "predicted_positives": int(buy_signals),
        "predicted_positive_rate_pct": (
            round(100.0 * buy_signals / n, 2) if n else None
        ),
        "n_days": days,
        "trades_per_day": trades_per_day,
        "false_positives": fp_i,
        "false_negatives": fn_i,
        "tp": tp_i,
        "fp": fp_i,
        "fn": fn_i,
        "tn": tn_i,
    }


def attach_trades_per_day(
    rows: list[dict[str, Any]] | None,
    n_days: int | None,
) -> list[dict[str, Any]]:
    """Stamp ``trades_per_day = buy_signals / n_days`` onto Threshold Analysis rows."""
    days = int(n_days) if n_days is not None and int(n_days) > 0 else None
    out: list[dict[str, Any]] = []
    for row in rows or []:
        if not isinstance(row, dict):
            continue
        item = dict(row)
        if days is not None:
            item["n_days"] = days
            buy = item.get("buy_signals")
            if buy is None:
                buy = item.get("predicted_positives")
            try:
                item["trades_per_day"] = round(float(buy) / float(days), 2)
            except (TypeError, ValueError):
                item["trades_per_day"] = None
        out.append(item)
    return out


def threshold_analysis(
    y_true: pd.Series | np.ndarray,
    y_prob: np.ndarray,
    *,
    thresholds: tuple[float, ...] | list[float] | None = None,
    n_days: int | None = None,
) -> list[dict[str, Any]]:
    """
    Precision / Recall / F1 / Accuracy plus trading-filter rates at each threshold.

    BUY Signals           = predicted positives (TP + FP)
    Trades/Day            = BUY Signals / n_days (when n_days provided)
    Hit Rate              = Precision = TP / (TP + FP)
    Good Trades Kept      = Recall      = TP / (TP + FN)
    Good Trades Filtered  = miss rate   = FN / (TP + FN)
    Bad Trades Filtered   = Specificity = TN / (TN + FP)
    Bad Trades Passed     = fallout     = FP / (TN + FP)
    """
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_prob, dtype=float)
    mask = np.isfinite(yt) & np.isfinite(yp)
    yt, yp = yt[mask], yp[mask]
    yt_bin = (yt >= 0.5).astype(int)
    yp = np.clip(yp, 0.0, 1.0)
    sweeps = list(thresholds) if thresholds is not None else list(DEFAULT_THRESHOLD_SWEEP)
    rows: list[dict[str, Any]] = []
    for thr in sweeps:
        y_hat = (yp >= float(thr)).astype(int)
        tp = int(((yt_bin == 1) & (y_hat == 1)).sum())
        fp = int(((yt_bin == 0) & (y_hat == 1)).sum())
        fn = int(((yt_bin == 1) & (y_hat == 0)).sum())
        tn = int(((yt_bin == 0) & (y_hat == 0)).sum())
        rows.append(
            threshold_row_from_confusion(
                threshold=float(thr),
                tp=tp,
                fp=fp,
                fn=fn,
                tn=tn,
                n_days=n_days,
            )
        )
    return rows


def aggregate_threshold_analysis(
    fold_rows: list[list[dict[str, Any]]] | list[dict[str, Any]],
    *,
    n_days: int | None = None,
) -> list[dict[str, Any]]:
    """Sum confusion counts across folds (or row lists) at matching thresholds."""
    by_thr: dict[float, dict[str, int]] = {}
    # Accept either list-of-lists (per fold) or a flat list of rows.
    sequences: list[list[dict[str, Any]]]
    if fold_rows and isinstance(fold_rows[0], dict):
        sequences = [list(fold_rows)]  # type: ignore[arg-type]
    else:
        sequences = [list(block or []) for block in fold_rows]  # type: ignore[arg-type]
    inferred_days: int | None = None
    for block in sequences:
        for row in block:
            if not isinstance(row, dict):
                continue
            try:
                thr = round(float(row.get("threshold")), 2)
            except (TypeError, ValueError):
                continue
            bucket = by_thr.setdefault(thr, {"tp": 0, "fp": 0, "fn": 0, "tn": 0})
            for key in ("tp", "fp", "fn", "tn"):
                try:
                    bucket[key] += int(row.get(key) or 0)
                except (TypeError, ValueError):
                    pass
            if inferred_days is None and row.get("n_days") is not None:
                try:
                    inferred_days = int(row["n_days"])
                except (TypeError, ValueError):
                    pass
    days = n_days if n_days is not None else inferred_days
    return [
        threshold_row_from_confusion(
            threshold=thr,
            tp=counts["tp"],
            fp=counts["fp"],
            fn=counts["fn"],
            tn=counts["tn"],
            n_days=days,
        )
        for thr, counts in sorted(by_thr.items())
    ]


def normalize_threshold_analysis_rows(
    rows: list[dict[str, Any]] | None,
    *,
    n_days: int | None = None,
) -> list[dict[str, Any]]:
    """Recompute derived fields so older saved rows gain Accuracy / BUY Signals / Hit Rate."""
    cleaned: list[dict[str, Any]] = []
    for row in rows or []:
        if not isinstance(row, dict):
            continue
        row_days = n_days
        if row_days is None and row.get("n_days") is not None:
            try:
                row_days = int(row.get("n_days"))
            except (TypeError, ValueError):
                row_days = None
        try:
            thr = float(row.get("threshold"))
            tp = int(row.get("tp") or 0)
            fp = int(row.get("fp") or 0)
            fn = int(row.get("fn") or 0)
            tn = int(row.get("tn") or 0)
        except (TypeError, ValueError):
            continue
        if any(k in row for k in ("tp", "fp", "fn", "tn")):
            cleaned.append(
                threshold_row_from_confusion(
                    threshold=thr, tp=tp, fp=fp, fn=fn, tn=tn, n_days=row_days
                )
            )
            continue
        # Legacy row without confusion cells — keep what we have, fill aliases.
        out = dict(row)
        out["threshold"] = round(thr, 2)
        if out.get("buy_signals") is None and out.get("predicted_positives") is not None:
            out["buy_signals"] = int(out["predicted_positives"])
        if out.get("hit_rate_pct") is None and out.get("precision_pct") is not None:
            out["hit_rate_pct"] = out.get("precision_pct")
        if out.get("false_positives") is None and out.get("fp") is not None:
            out["false_positives"] = out.get("fp")
        if out.get("false_negatives") is None and out.get("fn") is not None:
            out["false_negatives"] = out.get("fn")
        cleaned.append(out)
    if n_days is not None:
        return attach_trades_per_day(cleaned, n_days)
    return cleaned


def threshold_analysis_from_prediction_rows(
    rows: list[dict[str, Any]],
    *,
    thresholds: tuple[float, ...] | list[float] | None = None,
) -> list[dict[str, Any]]:
    """Rebuild Threshold Analysis from Prediction Run OOS rows (prob + label)."""
    if not rows:
        return []
    y_true: list[float] = []
    y_prob: list[float] = []
    days: set[str] = set()
    for row in rows:
        try:
            actual = float(row.get("actual_ltp"))
            pred = float(row.get("predicted_ltp"))
        except (TypeError, ValueError):
            continue
        if not (np.isfinite(actual) and np.isfinite(pred)):
            continue
        y_true.append(actual)
        y_prob.append(pred)
        day = str(row.get("trading_day") or "").strip()
        if day:
            days.add(day)
    if not y_true:
        return []
    return threshold_analysis(
        np.asarray(y_true, dtype=float),
        np.asarray(y_prob, dtype=float),
        thresholds=thresholds,
        n_days=len(days) if days else None,
    )



def trading_filter_summary_from_confusion(
    *,
    tp: int,
    fp: int,
    tn: int,
    fn: int,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Trader-facing filter rates from a confusion matrix at one threshold."""
    good_kept = (100.0 * tp / (tp + fn)) if (tp + fn) > 0 else None
    good_filtered = (100.0 * fn / (tp + fn)) if (tp + fn) > 0 else None
    bad_filtered = (100.0 * tn / (tn + fp)) if (tn + fp) > 0 else None
    bad_passed = (100.0 * fp / (tn + fp)) if (tn + fp) > 0 else None
    return {
        "threshold": round(float(threshold), 2),
        "good_trades_kept_pct": round(good_kept, 2) if good_kept is not None else None,
        "good_trades_filtered_pct": (
            round(good_filtered, 2) if good_filtered is not None else None
        ),
        "bad_trades_filtered_pct": (
            round(bad_filtered, 2) if bad_filtered is not None else None
        ),
        "bad_trades_passed_pct": (
            round(bad_passed, 2) if bad_passed is not None else None
        ),
    }


def _probability_calibration_bins(y_true: np.ndarray, y_prob: np.ndarray) -> list[dict[str, Any]]:
    bins = [
        ("<50%", None, 0.5),
        ("50–60%", 0.5, 0.6),
        ("60–75%", 0.6, 0.75),
        ("75–90%", 0.75, 0.9),
        (">90%", 0.9, 1.0001),
    ]
    cleaned: list[dict[str, Any]] = []
    for label, a, b in bins:
        if a is None:
            band_mask = y_prob < b
        else:
            band_mask = (y_prob >= a) & (y_prob < b)
        n = int(band_mask.sum())
        hit = float(y_true[band_mask].mean()) if n else None
        cleaned.append(
            {
                "band": label,
                "rows": n,
                "actual_hit_rate": hit,
                "actual_hit_pct": (100.0 * hit) if hit is not None else None,
            }
        )
    return cleaned


def evaluate_predictions(
    y_true: pd.Series | np.ndarray,
    y_pred: np.ndarray,
    *,
    prediction_type: str = "regression",
    baseline: pd.Series | np.ndarray | None = None,
    threshold: float = 0.5,
    target: str | None = None,
    direction_relative_to_zero: bool | None = None,
) -> dict[str, Any]:
    pred = str(prediction_type or "regression").strip().lower()
    if pred in ("binary", "classification", "multiclass"):
        return evaluate_classification(y_true, y_pred, threshold=threshold)
    return evaluate_regression(
        y_true,
        y_pred,
        baseline=baseline,
        target=target,
        direction_relative_to_zero=direction_relative_to_zero,
    )


def evaluate_regression(
    y_true: pd.Series | np.ndarray,
    y_pred: np.ndarray,
    *,
    baseline: pd.Series | np.ndarray | None = None,
    target: str | None = None,
    direction_relative_to_zero: bool | None = None,
) -> dict[str, Any]:
    from .target_kinds import is_ormp_return_target

    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    errors = yp - yt
    abs_errors = np.abs(errors)
    rmse = float(np.sqrt(mean_squared_error(yt, yp)))
    mae = float(mean_absolute_error(yt, yp))
    p_mae = premium_mae_pct(yt, yp)
    p_rmse = premium_rmse_pct(yt, yp)
    medae = float(np.median(abs_errors))
    p95 = p95_abs_error(yt, yp)
    bias = prediction_bias(yt, yp)
    bias_pct = prediction_bias_pct(yt, yp)
    metrics: dict[str, Any] = {
        "rmse": round(rmse, 6),
        "mae": round(mae, 6),
        "mape": round(_mape(yt, yp), 4) if np.isfinite(_mape(yt, yp)) else None,
        "r2": round(float(r2_score(yt, yp)), 6),
        "median_error": round(medae, 6),
        "medae": round(medae, 6),
        "max_error": round(float(np.max(abs_errors)), 6),
        "p95_error": round(p95, 6) if p95 is not None else None,
        "prediction_bias": round(bias, 6) if bias is not None else None,
        "prediction_bias_pct": round(bias_pct, 4) if bias_pct is not None else None,
        "premium_mae_pct": round(p_mae, 4) if p_mae is not None else None,
        "premium_rmse_pct": round(p_rmse, 4) if p_rmse is not None else None,
    }
    if direction_relative_to_zero is None:
        direction_relative_to_zero = is_ormp_return_target(target or "")
    base_arr = np.asarray(baseline, dtype=float) if baseline is not None else None
    da = directional_accuracy_pct(
        yt,
        yp,
        base_arr,
        relative_to_zero=bool(direction_relative_to_zero),
    )
    if da is not None:
        metrics["directional_accuracy_pct"] = round(da, 2)
    hit = endpoint_hit_rate_pct(yt, yp)
    if hit is not None:
        # Canonical key + legacy aliases used by holdout / fold UI
        metrics["endpoint_hit_pct"] = round(hit, 2)
        metrics["hit_rate_pct"] = round(hit, 2)
        metrics["hit_rate_tolerance_pct"] = float(ENDPOINT_HIT_TOLERANCE_PCT)
    metrics["premium_band_performance"] = premium_band_performance(yt, yp, baseline=base_arr)
    metrics.update(_classification_extras(yt, yp))
    return metrics

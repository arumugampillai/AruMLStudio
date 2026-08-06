"""Per-holdout-trading-day deep dive — metrics, bands, volatility, regime vs training."""

from __future__ import annotations

import csv
import io
from typing import Any

import numpy as np
import pandas as pd

from .evaluator import (
    ENDPOINT_HIT_TOLERANCE_PCT,
    endpoint_hit_rate_pct,
    evaluate_regression,
    premium_band_performance,
    resolve_ltp_baseline_from_frames,
)

# Relative |error|/|actual| threshold for Model Quality Endpoint Hit %
_HIT_RATE_TOLERANCE_PCT = ENDPOINT_HIT_TOLERANCE_PCT

# Backward-compatible alias
premium_hit_rate_pct = endpoint_hit_rate_pct

_SPOT_CANDIDATES = ("spot", "underlying_spot", "spot_ltp")
_PREMIUM_CANDIDATES = ("ltp",)
_IV_CANDIDATES = ("iv_zscore_1m", "iv_zscore_5m", "iv_zscore_15m", "implied_vol", "current_iv")
_SPOT_MOVE_CANDIDATES = ("spot_change_5m", "spot_change_1m", "spot_change")


def _resolve_col(df: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _numeric(df: pd.DataFrame, col: str | None) -> pd.Series:
    if not col or col not in df.columns:
        return pd.Series(dtype=float)
    return pd.to_numeric(df[col], errors="coerce")


def _day_volatility_stats(day_df: pd.DataFrame) -> dict[str, Any]:
    spot_col = _resolve_col(day_df, _SPOT_CANDIDATES)
    prem_col = _resolve_col(day_df, _PREMIUM_CANDIDATES)
    iv_col = _resolve_col(day_df, _IV_CANDIDATES)
    move_col = _resolve_col(day_df, _SPOT_MOVE_CANDIDATES)

    spot = _numeric(day_df, spot_col)
    prem = _numeric(day_df, prem_col)
    iv = _numeric(day_df, iv_col)
    move = _numeric(day_df, move_col)

    spot_valid = spot.dropna()
    prem_valid = prem.dropna()

    spot_open = float(spot_valid.iloc[0]) if len(spot_valid) else None
    spot_close = float(spot_valid.iloc[-1]) if len(spot_valid) else None
    spot_high = float(spot_valid.max()) if len(spot_valid) else None
    spot_low = float(spot_valid.min()) if len(spot_valid) else None
    spot_range = (
        (spot_high - spot_low) if spot_high is not None and spot_low is not None else None
    )
    spot_range_pct = (
        (spot_range / spot_open * 100.0)
        if spot_range is not None and spot_open and abs(spot_open) > 1e-9
        else None
    )
    spot_return_pct = (
        ((spot_close - spot_open) / abs(spot_open) * 100.0)
        if spot_open is not None and spot_close is not None and abs(spot_open) > 1e-9
        else None
    )
    spot_std = float(spot_valid.std()) if len(spot_valid) > 1 else None

    prem_mean = float(prem_valid.mean()) if len(prem_valid) else None
    prem_std = float(prem_valid.std()) if len(prem_valid) > 1 else None
    prem_cv = (
        (prem_std / abs(prem_mean) * 100.0)
        if prem_std is not None and prem_mean is not None and abs(prem_mean) > 1e-9
        else None
    )
    prem_range = (
        float(prem_valid.max() - prem_valid.min()) if len(prem_valid) > 1 else None
    )

    move_abs_mean = float(move.abs().mean()) if len(move.dropna()) else None
    iv_mean = float(iv.mean()) if len(iv.dropna()) else None
    iv_std = float(iv.std()) if len(iv.dropna()) > 1 else None
    iv_abs_mean = float(iv.abs().mean()) if len(iv.dropna()) else None

    is_expiry = None
    if "is_expiry_day" in day_df.columns:
        try:
            is_expiry = bool(pd.to_numeric(day_df["is_expiry_day"], errors="coerce").fillna(0).max() >= 1)
        except Exception:
            is_expiry = None

    return {
        "spot_column": spot_col,
        "premium_column": prem_col,
        "iv_column": iv_col,
        "spot_open": round(spot_open, 4) if spot_open is not None else None,
        "spot_close": round(spot_close, 4) if spot_close is not None else None,
        "spot_high": round(spot_high, 4) if spot_high is not None else None,
        "spot_low": round(spot_low, 4) if spot_low is not None else None,
        "spot_range": round(spot_range, 4) if spot_range is not None else None,
        "spot_range_pct": round(spot_range_pct, 4) if spot_range_pct is not None else None,
        "spot_return_pct": round(spot_return_pct, 4) if spot_return_pct is not None else None,
        "spot_std": round(spot_std, 6) if spot_std is not None else None,
        "premium_mean": round(prem_mean, 4) if prem_mean is not None else None,
        "premium_std": round(prem_std, 4) if prem_std is not None else None,
        "premium_cv_pct": round(prem_cv, 4) if prem_cv is not None else None,
        "premium_range": round(prem_range, 4) if prem_range is not None else None,
        "spot_move_abs_mean": round(move_abs_mean, 6) if move_abs_mean is not None else None,
        "iv_mean": round(iv_mean, 4) if iv_mean is not None else None,
        "iv_std": round(iv_std, 4) if iv_std is not None else None,
        "iv_abs_mean": round(iv_abs_mean, 4) if iv_abs_mean is not None else None,
        "is_expiry_day": is_expiry,
    }


def _day_metrics(
    y_true: pd.Series | np.ndarray,
    y_pred: np.ndarray,
    *,
    baseline: pd.Series | np.ndarray | None,
) -> dict[str, Any]:
    metrics = evaluate_regression(y_true, y_pred, baseline=baseline)
    hit = metrics.get("endpoint_hit_pct")
    if hit is None:
        hit = endpoint_hit_rate_pct(
            np.asarray(y_true, dtype=float),
            np.asarray(y_pred, dtype=float),
            tolerance_pct=_HIT_RATE_TOLERANCE_PCT,
        )
    return {
        "mae": metrics.get("mae"),
        "rmse": metrics.get("rmse"),
        "premium_mae_pct": metrics.get("premium_mae_pct"),
        "premium_rmse_pct": metrics.get("premium_rmse_pct"),
        "directional_accuracy_pct": metrics.get("directional_accuracy_pct"),
        "endpoint_hit_pct": round(hit, 2) if hit is not None else None,
        "hit_rate_pct": round(hit, 2) if hit is not None else None,
        "hit_rate_tolerance_pct": _HIT_RATE_TOLERANCE_PCT,
        "prediction_bias": metrics.get("prediction_bias"),
        "median_error": metrics.get("medae"),
    }


def _mean_ignore_none(values: list[float | None]) -> float | None:
    cleaned = [float(v) for v in values if v is not None and np.isfinite(float(v))]
    if not cleaned:
        return None
    return float(np.mean(cleaned))


def _ratio(current: float | None, baseline: float | None) -> float | None:
    if current is None or baseline is None:
        return None
    try:
        cur, base = float(current), float(baseline)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(cur) or not np.isfinite(base) or abs(base) < 1e-12:
        return None
    return float(cur / base)


def _delta(current: float | None, baseline: float | None) -> float | None:
    if current is None or baseline is None:
        return None
    try:
        return float(current) - float(baseline)
    except (TypeError, ValueError):
        return None


def _classify_regime(
    vol: dict[str, Any],
    metrics: dict[str, Any],
    train_avg: dict[str, Any],
    *,
    gap_pct: float | None = None,
) -> dict[str, Any]:
    flags: list[str] = []
    reasons: list[str] = []

    spot_range_pct = vol.get("spot_range_pct")
    train_spot_range = train_avg.get("spot_range_pct")
    spot_range_ratio = _ratio(spot_range_pct, train_spot_range)

    prem_std = vol.get("premium_std")
    train_prem_std = train_avg.get("premium_std")
    prem_vol_ratio = _ratio(prem_std, train_prem_std)

    spot_ret = vol.get("spot_return_pct")
    train_abs_ret = train_avg.get("spot_abs_return_pct")
    abs_ret = abs(float(spot_ret)) if spot_ret is not None else None

    iv_abs = vol.get("iv_abs_mean")
    mae_ratio = _ratio(metrics.get("mae"), train_avg.get("mae"))
    dir_delta = _delta(metrics.get("directional_accuracy_pct"), train_avg.get("directional_accuracy_pct"))

    if vol.get("is_expiry_day"):
        flags.append("expiry_day")
        reasons.append("Marked as an expiry trading day (elevated gamma / pinning risk).")

    if spot_range_ratio is not None and spot_range_ratio >= 1.5:
        flags.append("high_spot_volatility")
        reasons.append(
            f"Spot range {float(spot_range_pct):.2f}% is {spot_range_ratio:.1f}× the average training day "
            f"({float(train_spot_range):.2f}%)."
        )
    elif spot_range_pct is not None and float(spot_range_pct) >= 1.0:
        flags.append("elevated_spot_move")
        reasons.append(f"Intraday spot range of {float(spot_range_pct):.2f}% is elevated.")

    if prem_vol_ratio is not None and prem_vol_ratio >= 1.4:
        flags.append("high_premium_volatility")
        reasons.append(
            f"Option premium std {float(prem_std):.2f} is {prem_vol_ratio:.1f}× training-day average "
            f"({float(train_prem_std):.2f})."
        )

    if abs_ret is not None and (
        (train_abs_ret is not None and abs_ret >= float(train_abs_ret) * 1.5)
        or abs_ret >= 0.6
    ):
        direction = "up" if float(spot_ret or 0) > 0 else "down"
        flags.append("trending")
        reasons.append(
            f"Strong {direction} session: spot return {float(spot_ret):+.2f}% "
            f"(training-day avg |return| {float(train_abs_ret):.2f}%)."
            if train_abs_ret is not None
            else f"Strong {direction} session: spot return {float(spot_ret):+.2f}%."
        )

    if gap_pct is not None and abs(float(gap_pct)) >= 0.35:
        flags.append("gap")
        reasons.append(f"Opening gap vs prior session close: {float(gap_pct):+.2f}%.")

    if iv_abs is not None and float(iv_abs) >= 1.5:
        flags.append("iv_shock")
        reasons.append(f"Elevated |IV z-score| mean of {float(iv_abs):.2f} (volatility regime stress).")

    if mae_ratio is not None and mae_ratio >= 1.5:
        flags.append("model_stress")
        reasons.append(
            f"Day MAE {float(metrics.get('mae')):.3f} is {mae_ratio:.1f}× average training-day MAE "
            f"({float(train_avg.get('mae')):.3f})."
        )

    if dir_delta is not None and dir_delta <= -10:
        flags.append("direction_collapse")
        reasons.append(
            f"Direction accuracy {float(metrics.get('directional_accuracy_pct')):.1f}% is "
            f"{dir_delta:.1f} pts below training-day average."
        )

    is_regime_shift = bool(
        {"high_spot_volatility", "high_premium_volatility", "trending", "gap", "iv_shock", "expiry_day"}
        & set(flags)
    ) or (mae_ratio is not None and mae_ratio >= 1.8)

    if not reasons:
        reasons.append(
            "No strong regime-shift flags vs average training day; residual degradation may be "
            "feature drift, premium-band mix, or sample composition."
        )

    label = "Likely regime-shift day" if is_regime_shift else "Similar to training days"
    if is_regime_shift and flags:
        primary = flags[0].replace("_", " ")
        label = f"Likely regime-shift ({primary})"

    return {
        "is_regime_shift": is_regime_shift,
        "label": label,
        "flags": flags,
        "reasons": reasons,
        "spot_range_vs_train": round(spot_range_ratio, 3) if spot_range_ratio is not None else None,
        "premium_vol_vs_train": round(prem_vol_ratio, 3) if prem_vol_ratio is not None else None,
        "mae_vs_train": round(mae_ratio, 3) if mae_ratio is not None else None,
        "direction_pts_vs_train": round(dir_delta, 2) if dir_delta is not None else None,
    }


def _opening_gap_pct(prev_day_df: pd.DataFrame | None, day_df: pd.DataFrame) -> float | None:
    if prev_day_df is None or prev_day_df.empty or day_df.empty:
        return None
    spot_col = _resolve_col(day_df, _SPOT_CANDIDATES)
    if not spot_col or spot_col not in prev_day_df.columns:
        return None
    prev = _numeric(prev_day_df, spot_col).dropna()
    cur = _numeric(day_df, spot_col).dropna()
    if prev.empty or cur.empty:
        return None
    prev_close = float(prev.iloc[-1])
    cur_open = float(cur.iloc[0])
    if abs(prev_close) < 1e-9:
        return None
    return (cur_open - prev_close) / abs(prev_close) * 100.0


def _per_day_frame_stats(
    df: pd.DataFrame,
    y: pd.Series,
    pred: np.ndarray,
    trading_days: pd.Series,
    *,
    baseline: pd.Series | None,
) -> list[dict[str, Any]]:
    days = trading_days.reset_index(drop=True).astype(str)
    y = pd.to_numeric(y, errors="coerce").reset_index(drop=True)
    pred = np.asarray(pred, dtype=float)
    df = df.reset_index(drop=True)
    base = baseline.reset_index(drop=True) if baseline is not None else None
    rows: list[dict[str, Any]] = []
    unique_days = sorted(days.dropna().unique())
    day_to_df = {d: df.loc[days == d] for d in unique_days}

    for i, day in enumerate(unique_days):
        mask = (days == day).to_numpy()
        if not mask.any():
            continue
        day_df = day_to_df[day]
        b_slice = base.iloc[mask] if base is not None else None
        metrics = _day_metrics(y.iloc[mask], pred[mask], baseline=b_slice)
        bands = premium_band_performance(
            y.iloc[mask].to_numpy(),
            pred[mask],
            baseline=b_slice.to_numpy() if b_slice is not None else None,
        )
        vol = _day_volatility_stats(day_df)
        prev_key = unique_days[i - 1] if i > 0 else None
        gap = _opening_gap_pct(day_to_df.get(prev_key) if prev_key else None, day_df)
        rows.append({
            "trading_day": day,
            "rows": int(mask.sum()),
            "metrics": metrics,
            "premium_bands": bands,
            "volatility": vol,
            "gap_pct": round(gap, 4) if gap is not None else None,
        })
    return rows


def _training_day_average(train_days: list[dict[str, Any]]) -> dict[str, Any]:
    if not train_days:
        return {}
    metric_keys = (
        "mae", "rmse", "premium_mae_pct", "premium_rmse_pct",
        "directional_accuracy_pct", "hit_rate_pct",
    )
    vol_keys = (
        "spot_range_pct", "spot_return_pct", "spot_std",
        "premium_std", "premium_cv_pct", "premium_mean",
        "iv_abs_mean", "spot_move_abs_mean",
    )
    out: dict[str, Any] = {
        "trading_days": len(train_days),
        "rows_mean": _mean_ignore_none([d.get("rows") for d in train_days]),
    }
    for key in metric_keys:
        out[key] = _mean_ignore_none([
            (d.get("metrics") or {}).get(key) for d in train_days
        ])
    for key in vol_keys:
        vals = [(d.get("volatility") or {}).get(key) for d in train_days]
        out[key] = _mean_ignore_none(vals)
    abs_rets = []
    for d in train_days:
        r = (d.get("volatility") or {}).get("spot_return_pct")
        if r is not None:
            abs_rets.append(abs(float(r)))
    out["spot_abs_return_pct"] = float(np.mean(abs_rets)) if abs_rets else None
    return out


def _compare_to_train(day: dict[str, Any], train_avg: dict[str, Any]) -> dict[str, Any]:
    m = day.get("metrics") or {}
    v = day.get("volatility") or {}
    comparisons = []
    for key, label, higher_worse in (
        ("mae", "MAE", True),
        ("rmse", "RMSE", True),
        ("directional_accuracy_pct", "Direction Accuracy", False),
        ("hit_rate_pct", "Endpoint Hit %", False),
        ("premium_mae_pct", "Premium MAE %", True),
    ):
        cur = m.get(key)
        base = train_avg.get(key)
        ratio = _ratio(cur, base)
        delta = _delta(cur, base)
        comparisons.append({
            "metric": label,
            "holdout_day": cur,
            "train_day_avg": base,
            "ratio": round(ratio, 3) if ratio is not None else None,
            "delta": round(delta, 4) if delta is not None else None,
            "worse": (
                (ratio is not None and ratio > 1.15) if higher_worse
                else (delta is not None and delta < -5)
            ) if (ratio is not None or delta is not None) else None,
        })
    for key, label in (
        ("spot_range_pct", "Spot Range %"),
        ("premium_std", "Premium Std"),
        ("iv_abs_mean", "|IV z| Mean"),
    ):
        cur = v.get(key)
        base = train_avg.get(key)
        ratio = _ratio(cur, base)
        comparisons.append({
            "metric": label,
            "holdout_day": cur,
            "train_day_avg": base,
            "ratio": round(ratio, 3) if ratio is not None else None,
            "delta": round(_delta(cur, base), 4) if _delta(cur, base) is not None else None,
            "worse": (ratio is not None and ratio > 1.25) if ratio is not None else None,
        })
    return {
        "train_days_used": train_avg.get("trading_days"),
        "rows": comparisons,
    }


def build_holdout_days_analysis(
    *,
    holdout_df: pd.DataFrame,
    y_holdout: pd.Series,
    pred_holdout: np.ndarray,
    train_df: pd.DataFrame,
    y_train: pd.Series,
    pred_train: np.ndarray,
    model_name: str = "",
) -> dict[str, Any]:
    """Build detailed per-holdout-day analysis vs average training (WF) day."""
    if "trading_day" not in holdout_df.columns:
        return {"ok": False, "error": "Holdout frame has no trading_day column"}

    baseline_ho = resolve_ltp_baseline_from_frames(holdout_df)
    baseline_tr = resolve_ltp_baseline_from_frames(train_df)

    ho_days = _per_day_frame_stats(
        holdout_df,
        y_holdout,
        pred_holdout,
        holdout_df["trading_day"],
        baseline=baseline_ho,
    )
    if not ho_days:
        return {"ok": False, "error": "No holdout trading days found"}

    train_days: list[dict[str, Any]] = []
    if "trading_day" in train_df.columns and len(train_df) > 0:
        train_days = _per_day_frame_stats(
            train_df,
            y_train,
            pred_train,
            train_df["trading_day"],
            baseline=baseline_tr,
        )
    train_avg = _training_day_average(train_days)

    analyzed: list[dict[str, Any]] = []
    for day in ho_days:
        compare = _compare_to_train(day, train_avg)
        regime = _classify_regime(
            day.get("volatility") or {},
            day.get("metrics") or {},
            train_avg,
            gap_pct=day.get("gap_pct"),
        )
        analyzed.append({
            **day,
            "vs_training_day_avg": compare,
            "regime": regime,
        })

    names = [d["trading_day"] for d in analyzed]
    n_regime = sum(1 for d in analyzed if (d.get("regime") or {}).get("is_regime_shift"))
    exec_lines = [
        f"Analyzed {len(analyzed)} holdout trading day(s): {', '.join(names)}.",
        f"Training reference: average of {int(train_avg.get('trading_days') or 0)} WF trading day(s).",
        f"Endpoint Hit % uses |pred−actual|/|actual| ≤ {_HIT_RATE_TOLERANCE_PCT:.0f}%.",
    ]
    if n_regime:
        exec_lines.append(
            f"{n_regime} of {len(analyzed)} holdout day(s) look like regime-shift vs training."
        )
    else:
        exec_lines.append("No holdout day flagged as a strong regime shift vs training averages.")

    return {
        "ok": True,
        "model_name": model_name,
        "hit_rate_tolerance_pct": _HIT_RATE_TOLERANCE_PCT,
        "executive_summary": {
            "trading_days": names,
            "holdout_day_count": len(analyzed),
            "train_day_count": int(train_avg.get("trading_days") or 0),
            "regime_shift_days": n_regime,
            "lines": exec_lines,
        },
        "training_day_average": train_avg,
        "days": analyzed,
    }


def build_holdout_days_analysis_csv(analysis: dict[str, Any]) -> str:
    """Serialize Holdout Days Analysis to CSV text."""
    buf = io.StringIO()
    writer = csv.writer(buf)

    def _section(title: str) -> None:
        writer.writerow([])
        writer.writerow([title])

    if not analysis.get("ok"):
        writer.writerow(["Error", analysis.get("error") or "Analysis failed"])
        return buf.getvalue().lstrip("\n")

    _section("Executive Summary")
    writer.writerow(["Field", "Value"])
    ex = analysis.get("executive_summary") or {}
    writer.writerow(["Trading days", ", ".join(ex.get("trading_days") or [])])
    writer.writerow(["Holdout days", ex.get("holdout_day_count")])
    writer.writerow(["Training days (avg basis)", ex.get("train_day_count")])
    writer.writerow(["Regime-shift days", ex.get("regime_shift_days")])
    writer.writerow(["Endpoint Hit tolerance %", analysis.get("hit_rate_tolerance_pct")])
    for line in ex.get("lines") or []:
        writer.writerow(["Note", line])

    train = analysis.get("training_day_average") or {}
    if train:
        _section("Average Training Day")
        writer.writerow(["Metric", "Value"])
        for key, label in (
            ("trading_days", "Days"),
            ("rows_mean", "Rows / day"),
            ("mae", "MAE"),
            ("rmse", "RMSE"),
            ("directional_accuracy_pct", "Direction Accuracy %"),
            ("hit_rate_pct", "Endpoint Hit %"),
            ("premium_mae_pct", "Premium MAE %"),
            ("spot_range_pct", "Spot Range %"),
            ("spot_abs_return_pct", "Spot |Return| %"),
            ("premium_std", "Premium Std"),
            ("iv_abs_mean", "|IV z| Mean"),
        ):
            writer.writerow([label, train.get(key) if train.get(key) is not None else ""])

    for day in analysis.get("days") or []:
        if not isinstance(day, dict):
            continue
        name = day.get("trading_day") or "—"
        _section(f"Trading Day: {name}")
        writer.writerow(["Field", "Value"])
        writer.writerow(["Rows", day.get("rows")])
        writer.writerow(["Opening gap %", day.get("gap_pct") if day.get("gap_pct") is not None else ""])

        m = day.get("metrics") or {}
        writer.writerow(["MAE", m.get("mae") if m.get("mae") is not None else ""])
        writer.writerow(["RMSE", m.get("rmse") if m.get("rmse") is not None else ""])
        writer.writerow([
            "Direction Accuracy %",
            m.get("directional_accuracy_pct") if m.get("directional_accuracy_pct") is not None else "",
        ])
        writer.writerow(["Endpoint Hit %", m.get("hit_rate_pct") if m.get("hit_rate_pct") is not None else ""])
        writer.writerow([
            "Premium MAE %",
            m.get("premium_mae_pct") if m.get("premium_mae_pct") is not None else "",
        ])

        _section(f"Premium Bands — {name}")
        writer.writerow(["Band", "Rows", "MAE", "RMSE", "Premium MAE %", "Direction Acc %"])
        for band in day.get("premium_bands") or []:
            if not isinstance(band, dict):
                continue
            writer.writerow([
                band.get("band_label") or band.get("band") or "",
                band.get("samples") if band.get("samples") is not None else "",
                band.get("mae") if band.get("mae") is not None else "",
                band.get("rmse") if band.get("rmse") is not None else "",
                band.get("premium_mae_pct") if band.get("premium_mae_pct") is not None else "",
                band.get("directional_accuracy_pct") if band.get("directional_accuracy_pct") is not None else "",
            ])

        _section(f"Volatility — {name}")
        writer.writerow(["Metric", "Value"])
        v = day.get("volatility") or {}
        for key, label in (
            ("spot_open", "Spot open"),
            ("spot_close", "Spot close"),
            ("spot_high", "Spot high"),
            ("spot_low", "Spot low"),
            ("spot_range", "Spot range"),
            ("spot_range_pct", "Spot range %"),
            ("spot_return_pct", "Spot return %"),
            ("spot_std", "Spot std"),
            ("premium_mean", "Premium mean"),
            ("premium_std", "Premium std"),
            ("premium_cv_pct", "Premium CV %"),
            ("premium_range", "Premium range"),
            ("iv_mean", "IV mean"),
            ("iv_std", "IV std"),
            ("iv_abs_mean", "|IV| mean"),
            ("is_expiry_day", "Expiry day"),
        ):
            writer.writerow([label, v.get(key) if v.get(key) is not None else ""])

        _section(f"Vs Average Training Day — {name}")
        writer.writerow(["Metric", "Holdout Day", "Train Day Avg", "Ratio", "Delta", "Worse?"])
        vs = (day.get("vs_training_day_avg") or {}).get("rows") or []
        for row in vs:
            if not isinstance(row, dict):
                continue
            worse = row.get("worse")
            writer.writerow([
                row.get("metric") or "",
                row.get("holdout_day") if row.get("holdout_day") is not None else "",
                row.get("train_day_avg") if row.get("train_day_avg") is not None else "",
                row.get("ratio") if row.get("ratio") is not None else "",
                row.get("delta") if row.get("delta") is not None else "",
                "Yes" if worse else ("No" if worse is False else ""),
            ])

        regime = day.get("regime") or {}
        _section(f"Regime Assessment — {name}")
        writer.writerow(["Field", "Value"])
        writer.writerow(["Label", regime.get("label") or ""])
        writer.writerow(["Is regime shift", regime.get("is_regime_shift")])
        writer.writerow(["Flags", ", ".join(regime.get("flags") or [])])
        for reason in regime.get("reasons") or []:
            writer.writerow(["Reason", reason])

    return buf.getvalue().lstrip("\n")

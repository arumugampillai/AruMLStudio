"""Premium RMSE diagnostics for holdout performance analysis."""

from __future__ import annotations

import csv
import io
from typing import Any

import numpy as np
import pandas as pd

from .evaluator import PREMIUM_METRIC_BANDS, premium_rmse_pct

_PREMIUM_EPS = 1e-9


def _valid_premium_mask(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    return np.isfinite(yt) & np.isfinite(yp) & (np.abs(yt) > _PREMIUM_EPS)


def _relative_sq_errors(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mask = _valid_premium_mask(y_true, y_pred)
    yt = np.asarray(y_true, dtype=float)[mask]
    yp = np.asarray(y_pred, dtype=float)[mask]
    rel = (yp - yt) / np.abs(yt)
    return np.square(rel), yt


def premium_rmse_band_breakdown(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    baseline: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    """Per-band premium RMSE and contribution to total relative squared error."""
    mask = _valid_premium_mask(y_true, y_pred)
    if not mask.any():
        return []
    yt = np.asarray(y_true, dtype=float)[mask]
    yp = np.asarray(y_pred, dtype=float)[mask]
    rel_sq = np.square((yp - yt) / np.abs(yt))
    total_sse = float(np.sum(rel_sq))
    if total_sse <= 0:
        return []

    if baseline is not None:
        assign = np.asarray(baseline, dtype=float)[mask]
    else:
        assign = yt
    rows: list[dict[str, Any]] = []
    for label, lo, hi in PREMIUM_METRIC_BANDS:
        if hi is None:
            band_mask = assign >= lo
        else:
            band_mask = (assign >= lo) & (assign < hi)
        n = int(band_mask.sum())
        if n <= 0:
            rows.append({
                "band": label,
                "band_label": f"₹{label}",
                "rows": 0,
                "premium_rmse_pct": None,
                "contribution_pct": 0.0,
            })
            continue
        band_sse = float(np.sum(rel_sq[band_mask]))
        band_rmse = float(np.sqrt(np.mean(rel_sq[band_mask])) * 100.0)
        rows.append({
            "band": label,
            "band_label": f"₹{label}",
            "rows": n,
            "premium_rmse_pct": round(band_rmse, 1),
            "contribution_pct": round(band_sse / total_sse * 100.0, 1),
        })
    rows.sort(key=lambda r: float(r.get("contribution_pct") or 0), reverse=True)
    return rows


def outlier_contribution_to_premium_rmse(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> list[dict[str, Any]]:
    """Share of total relative SSE from top 1%, 5%, 10% largest squared errors.

    Uses sum((pred-actual)/actual)^2 per row — the Premium RMSE numerator before
    sqrt(mean(...)).  Not MAE (|rel|) and not a share of the scalar RMSE value.
    """
    rel_sq, _ = _relative_sq_errors(y_true, y_pred)
    if len(rel_sq) == 0:
        return []
    total_sse = float(np.sum(rel_sq))
    if total_sse <= 0:
        return []
    ordered = np.sort(rel_sq)[::-1]
    n = len(ordered)
    rows: list[dict[str, Any]] = []
    for label, pct in (("Top 1% rows", 1), ("Top 5% rows", 5), ("Top 10% rows", 10)):
        k = max(1, int(np.ceil(n * pct / 100.0)))
        contrib = float(np.sum(ordered[:k]) / total_sse * 100.0)
        rows.append({"label": label, "top_pct": pct, "contribution_pct": round(contrib, 1)})
    return rows


def relative_error_percentiles(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float | None]:
    """Percentiles of |pred−actual|/|actual| × 100 over valid premium rows."""
    mask = _valid_premium_mask(y_true, y_pred)
    if not mask.any():
        return {"median": None, "p90": None, "p95": None, "p99": None}
    yt = np.asarray(y_true, dtype=float)[mask]
    yp = np.asarray(y_pred, dtype=float)[mask]
    rel_pct = np.abs((yp - yt) / np.abs(yt)) * 100.0
    return {
        "median": float(np.median(rel_pct)),
        "p90": float(np.percentile(rel_pct, 90)),
        "p95": float(np.percentile(rel_pct, 95)),
        "p99": float(np.percentile(rel_pct, 99)),
    }


def premium_rmse_excluding_top_pct(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    top_pct: float = 1.0,
) -> float | None:
    """Premium RMSE after dropping the top *top_pct*% rows by squared relative error."""
    rel_sq, _ = _relative_sq_errors(y_true, y_pred)
    n = len(rel_sq)
    if n <= 0:
        return None
    k = max(1, int(np.ceil(n * top_pct / 100.0)))
    if k >= n:
        return None
    order = np.argsort(rel_sq)[::-1]
    keep = np.ones(n, dtype=bool)
    keep[order[:k]] = False
    remaining = rel_sq[keep]
    if len(remaining) <= 0:
        return None
    return float(np.sqrt(np.mean(remaining)) * 100.0)


def _outlier_impact_status(contribution_pct: float) -> str:
    if contribution_pct >= 90:
        return "Extreme"
    if contribution_pct >= 60:
        return "High"
    if contribution_pct >= 30:
        return "Moderate"
    return "Low"


def build_outlier_impact(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    outlier_rows: list[dict[str, Any]] | None = None,
    top_pct: float = 1.0,
) -> dict[str, Any]:
    rel_sq, _ = _relative_sq_errors(y_true, y_pred)
    n = len(rel_sq)
    if n <= 0:
        return {}
    k = max(1, int(np.ceil(n * top_pct / 100.0)))
    contrib = None
    if outlier_rows:
        contrib = next(
            (float(r["contribution_pct"]) for r in outlier_rows if r.get("top_pct") == top_pct),
            None,
        )
    if contrib is None:
        total_sse = float(np.sum(rel_sq))
        if total_sse > 0:
            ordered = np.sort(rel_sq)[::-1]
            contrib = float(np.sum(ordered[:k]) / total_sse * 100.0)
    if contrib is None:
        return {}
    return {
        "top_pct": top_pct,
        "label": f"Top {int(top_pct)}% rows",
        "row_count": k,
        "total_rows": n,
        "contribution_pct": round(contrib, 1),
        "status": _outlier_impact_status(contrib),
    }


def build_prediction_quality_summary(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    premium_rmse: float | None = None,
    exclude_top_pct: float = 1.0,
) -> dict[str, Any]:
    rel_pct = relative_error_percentiles(y_true, y_pred)
    rmse = premium_rmse if premium_rmse is not None else premium_rmse_pct(y_true, y_pred)
    rmse_excl = premium_rmse_excluding_top_pct(y_true, y_pred, top_pct=exclude_top_pct)
    return {
        "relative_error": rel_pct,
        "premium_rmse_pct": rmse,
        "premium_rmse_excl_top_pct": rmse_excl,
        "exclude_top_pct": exclude_top_pct,
    }


def absolute_error_percentiles(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float | None]:
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(yt) & np.isfinite(yp)
    if not mask.any():
        return {"median": None, "p90": None, "p95": None, "p99": None, "max": None}
    err = np.abs(yp[mask] - yt[mask])
    return {
        "median": float(np.median(err)),
        "p90": float(np.percentile(err, 90)),
        "p95": float(np.percentile(err, 95)),
        "p99": float(np.percentile(err, 99)),
        "max": float(np.max(err)),
    }


def _infer_error_reason(row: pd.Series) -> str:
    for col, label in (
        ("is_expiry_day", "Expiry gamma"),
        ("is_expiry_week", "Expiry week"),
        ("is_first_hour", "Opening hour"),
    ):
        if col in row.index:
            try:
                if float(row[col]) >= 0.5:
                    return label
            except (TypeError, ValueError):
                pass
    for col in ("iv_zscore_1m", "iv_zscore_5m", "iv_zscore_15m", "current_iv"):
        if col in row.index:
            try:
                val = float(row[col])
                if np.isfinite(val) and abs(val) >= 2.0:
                    return "High IV spike"
            except (TypeError, ValueError):
                pass
    for col in ("gamma", "gamma_x_spot"):
        if col in row.index:
            try:
                val = float(row[col])
                if np.isfinite(val) and val >= 0.05:
                    return "High gamma"
            except (TypeError, ValueError):
                pass
    return "Large prediction error"


def top_error_samples(
    ho_df: pd.DataFrame,
    y_true: pd.Series,
    y_pred: np.ndarray,
    *,
    limit: int = 15,
) -> list[dict[str, Any]]:
    yt = pd.to_numeric(y_true, errors="coerce").reset_index(drop=True)
    yp = np.asarray(y_pred, dtype=float)
    n = min(len(yt), len(yp), len(ho_df))
    if n <= 0:
        return []
    df = ho_df.iloc[:n].reset_index(drop=True)
    err = np.abs(yp[:n] - yt.to_numpy())
    mask = np.isfinite(err) & np.isfinite(yt.to_numpy()) & np.isfinite(yp[:n])
    if not mask.any():
        return []
    order = np.argsort(err)[::-1]
    rows: list[dict[str, Any]] = []
    for idx in order:
        if not mask[idx]:
            continue
        row = df.iloc[int(idx)]
        ts = row.get("timestamp")
        strike = row.get("strike")
        opt = row.get("option_type")
        strike_label = f"{strike} {opt}".strip() if strike is not None else "—"
        actual = float(yt.iloc[int(idx)])
        predicted = float(yp[int(idx)])
        error = float(predicted - actual)
        rows.append({
            "time": str(ts) if ts is not None else "—",
            "strike": strike_label,
            "actual": round(actual, 2),
            "predicted": round(predicted, 2),
            "error": round(error, 2),
            "reason": _infer_error_reason(row),
        })
        if len(rows) >= limit:
            break
    return rows


def _drift_level_label(score: float) -> str:
    if score >= 25:
        return "High"
    if score >= 15:
        return "Moderate"
    if score > 0:
        return "Low"
    return "None"


def _top_band_contribution(band_breakdown: list[dict[str, Any]]) -> tuple[str | None, float | None]:
    best_label: str | None = None
    best_pct: float | None = None
    for row in band_breakdown:
        if not isinstance(row, dict):
            continue
        if int(row.get("rows") or 0) <= 0:
            continue
        contrib = row.get("contribution_pct")
        if contrib is None:
            continue
        pct = float(contrib)
        if best_pct is None or pct > best_pct:
            best_pct = pct
            best_label = str(row.get("band_label") or row.get("band") or "")
    return best_label, best_pct


_PRIMARY_CAUSE_OUTLIERS = "Extreme prediction errors concentrated in low-premium options"


def _typical_quality_label(
    *,
    median_rel_pct: float | None,
    rmse_excl_top_pct: float | None,
) -> str:
    if median_rel_pct is None and rmse_excl_top_pct is None:
        return "—"
    med = float(median_rel_pct or 999.0)
    excl = float(rmse_excl_top_pct or 999.0)
    if med <= 4.0 and excl <= 15.0:
        return "Excellent"
    if med <= 8.0 and excl <= 35.0:
        return "Good"
    if med <= 15.0 or excl <= 55.0:
        return "Fair"
    return "Poor"


def _outlier_handling_label(contribution_pct: float | None, status: str | None = None) -> str:
    if status in ("Extreme", "High"):
        return "Poor"
    if contribution_pct is None:
        return "—"
    contrib = float(contribution_pct)
    if contrib >= 90:
        return "Poor"
    if contrib >= 60:
        return "Fair"
    if contrib >= 30:
        return "Moderate"
    return "Good"


def _infer_main_weakness(
    *,
    root_cause: dict[str, Any],
    band_breakdown: list[dict[str, Any]] | None,
    top_error_samples: list[dict[str, Any]] | None,
) -> str:
    reasons: list[str] = []
    for row in top_error_samples or []:
        if isinstance(row, dict) and row.get("reason"):
            reasons.append(str(row["reason"]))
    top_reason: str | None = None
    if reasons:
        from collections import Counter
        top_reason = Counter(reasons).most_common(1)[0][0]

    band_label, _ = _top_band_contribution(band_breakdown or [])
    low_premium = bool(
        band_label
        and any(tok in band_label for tok in ("0-15", "15-30", "15–30", "₹15", "₹0"))
    )

    if top_reason == "Expiry gamma" and low_premium:
        return "Low-premium expiry gamma options"
    if top_reason == "Expiry gamma":
        return "Expiry gamma options"
    if top_reason == "Expiry week" and low_premium:
        return "Low-premium expiry-week options"
    if top_reason == "High IV spike" and low_premium:
        return "Low-premium high-IV options"
    if top_reason == "High gamma" and low_premium:
        return "Low-premium high-gamma options"
    if top_reason:
        return top_reason.lower() + " options" if "option" not in top_reason.lower() else top_reason

    primary = str(root_cause.get("primary_cause") or "")
    if primary == _PRIMARY_CAUSE_OUTLIERS and band_label:
        return f"{band_label} premium options"
    if primary:
        return primary
    return "—"


def _overall_quality_stars(typical: str, outlier: str) -> tuple[int, str]:
    """Return filled star count (1–5) and short label."""
    if typical == "—" or outlier == "—":
        return 3, "Fair"
    if typical in ("Excellent", "Good") and outlier in ("Poor", "Fair"):
        if typical == "Excellent":
            return 4, "Good"
        return 4, "Good"
    if typical == "Excellent" and outlier in ("Good", "Moderate"):
        return 5, "Excellent"
    if typical == "Good" and outlier in ("Good", "Moderate"):
        return 4, "Good"
    if typical == "Fair":
        return 3, "Fair"
    return 2, "Poor"


def _star_display(stars: int) -> str:
    filled = max(0, min(5, int(stars)))
    return "★" * filled + "☆" * (5 - filled)


def build_model_summary(
    *,
    root_cause: dict[str, Any],
    quality_summary: dict[str, Any],
    outlier_impact: dict[str, Any],
    band_breakdown: list[dict[str, Any]] | None = None,
    top_error_samples: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    rel = quality_summary.get("relative_error") if isinstance(quality_summary.get("relative_error"), dict) else {}
    typical = _typical_quality_label(
        median_rel_pct=rel.get("median"),
        rmse_excl_top_pct=quality_summary.get("premium_rmse_excl_top_pct"),
    )
    outlier = _outlier_handling_label(
        outlier_impact.get("contribution_pct"),
        outlier_impact.get("status"),
    )
    weakness = _infer_main_weakness(
        root_cause=root_cause,
        band_breakdown=band_breakdown,
        top_error_samples=top_error_samples,
    )
    stars, overall_label = _overall_quality_stars(typical, outlier)
    return {
        "overall_stars": stars,
        "overall_stars_display": _star_display(stars),
        "overall_quality": overall_label,
        "typical_prediction_quality": typical,
        "extreme_outlier_handling": outlier,
        "main_weakness": weakness,
    }


def build_premium_root_cause_summary(
    *,
    drift_scores: dict[str, float],
    similarity_pct: float | None,
    production_premium_rmse: float | None,
    holdout_premium_rmse: float | None,
    outlier_rows: list[dict[str, Any]],
    band_breakdown: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    vol = float(drift_scores.get("volatility") or 0)
    feat = float(drift_scores.get("feature") or 0)
    tgt = float(drift_scores.get("target") or 0)

    bullets: list[str] = []
    if vol >= 10:
        bullets.append(f"Moderate volatility increase (+{vol:.0f}%)")
    elif vol > 0:
        bullets.append(f"Low volatility drift (+{vol:.0f}%)")
    if feat >= 10:
        bullets.append(f"Feature drift (+{feat:.0f}%)")
    elif feat > 0:
        bullets.append(f"Small feature drift (+{feat:.0f}%)")
    if tgt >= 10:
        bullets.append(f"Target drift (+{tgt:.0f}%)")
    elif tgt > 0:
        bullets.append(f"Low target drift (+{tgt:.0f}%)")

    checklist: list[dict[str, str]] = []
    if similarity_pct is not None:
        checklist.append({
            "status": "ok",
            "text": f"Training vs Holdout: {similarity_pct:.0f}% similar",
        })
    checklist.append({
        "status": "ok",
        "text": f"Target drift: {_drift_level_label(tgt)} ({tgt:.0f}%)",
    })
    checklist.append({
        "status": "ok",
        "text": f"Feature drift: {_drift_level_label(feat)} ({feat:.0f}%)",
    })
    checklist.append({
        "status": "ok",
        "text": f"Volatility drift: {_drift_level_label(vol)} ({vol:.0f}%)",
    })

    top1 = next((float(r["contribution_pct"]) for r in outlier_rows if r.get("top_pct") == 1), None)
    top10 = next((float(r["contribution_pct"]) for r in outlier_rows if r.get("top_pct") == 10), None)

    if similarity_pct is not None and similarity_pct >= 80 and top1 is not None and top1 >= 30:
        conclusion = (
            "Data distributions remain similar. Performance degradation is likely caused by "
            "prediction errors on a relatively small number of difficult samples rather than broad data drift."
        )
        primary = _PRIMARY_CAUSE_OUTLIERS
    elif top10 is not None and top10 < 25:
        conclusion = (
            "Premium RMSE degradation appears spread across many predictions, suggesting broader "
            "model weakness on the holdout period rather than isolated outliers."
        )
        primary = "Broad degradation"
    elif top1 is not None and top1 >= 35:
        conclusion = (
            "A few extreme relative errors dominate premium RMSE despite similar overall distributions."
        )
        primary = _PRIMARY_CAUSE_OUTLIERS
    else:
        conclusion = (
            "Mixed signal: moderate distribution drift with meaningful contribution from larger errors."
        )
        primary = "Mixed"

    rmse_note = None
    if production_premium_rmse is not None and holdout_premium_rmse is not None:
        rmse_note = (
            f"Premium RMSE increased from {production_premium_rmse:.2f}% to {holdout_premium_rmse:.2f}%"
            + (f" while Training vs Holdout similarity is {similarity_pct:.0f}%." if similarity_pct is not None else ".")
        )

    warnings: list[str] = []
    detail_bullets: list[str] = []
    if top1 is not None and top1 >= 30:
        warnings.append("Premium RMSE is dominated by a very small number of predictions.")
        detail_bullets.append(
            f"Top 1% of rows contribute {top1:.1f}% of total squared relative error "
            f"(Premium RMSE numerator, before √mean)."
        )
    band_label, band_pct = _top_band_contribution(band_breakdown or [])
    if band_label and band_pct is not None and band_pct >= 5:
        detail_bullets.append(
            f"{band_pct:.0f}% of total squared relative error comes from {band_label} premium options."
        )

    return {
        "primary_cause": primary,
        "bullets": bullets,
        "checklist": checklist,
        "warnings": warnings,
        "detail_bullets": detail_bullets,
        "conclusion": conclusion,
        "rmse_note": rmse_note,
        "top1_contribution_pct": top1,
        "top10_contribution_pct": top10,
        "top_band_label": band_label,
        "top_band_contribution_pct": band_pct,
    }


def build_premium_analysis(
    *,
    y_wf: pd.Series,
    pred_wf: np.ndarray,
    y_ho: pd.Series,
    pred_ho: np.ndarray,
    ho_df: pd.DataFrame,
    baseline_ho: pd.Series | np.ndarray | None,
    drift_scores: dict[str, float],
    similarity_pct: float | None,
    production_wf: dict[str, Any],
    holdout_test: dict[str, Any],
    by_trading_day: list[dict[str, Any]],
    feature_drift_ranking: list[dict[str, Any]] | None = None,
    model_name: str = "",
    model: Any | None = None,
    use_features: list[str] | None = None,
) -> dict[str, Any]:
    ho_base = baseline_ho.to_numpy() if isinstance(baseline_ho, pd.Series) else baseline_ho
    band_breakdown = premium_rmse_band_breakdown(
        y_ho.to_numpy(), pred_ho, baseline=ho_base,
    )
    outliers = outlier_contribution_to_premium_rmse(y_ho.to_numpy(), pred_ho)
    wf_pct = absolute_error_percentiles(y_wf.to_numpy(), pred_wf)
    ho_pct = absolute_error_percentiles(y_ho.to_numpy(), pred_ho)
    worst_days = sorted(
        [d for d in by_trading_day if isinstance(d, dict)],
        key=lambda d: float(d.get("mae") or 0),
        reverse=True,
    )[:10]
    samples = top_error_samples(ho_df, y_ho, pred_ho)
    root_cause = build_premium_root_cause_summary(
        drift_scores=drift_scores,
        similarity_pct=similarity_pct,
        production_premium_rmse=production_wf.get("premium_rmse_pct"),
        holdout_premium_rmse=holdout_test.get("premium_rmse_pct"),
        outlier_rows=outliers,
        band_breakdown=band_breakdown,
    )
    holdout_premium_rmse = premium_rmse_pct(y_ho.to_numpy(), pred_ho)
    quality = build_prediction_quality_summary(
        y_ho.to_numpy(), pred_ho, premium_rmse=holdout_premium_rmse,
    )
    outlier_impact = build_outlier_impact(
        y_ho.to_numpy(), pred_ho, outlier_rows=outliers,
    )
    from .holdout_top1_analysis import build_top1_error_analysis

    top1_analysis = build_top1_error_analysis(
        ho_df=ho_df,
        y_ho=y_ho,
        pred_ho=pred_ho,
        baseline_ho=baseline_ho,
        feature_drift_ranking=feature_drift_ranking,
        model_name=model_name,
        model=model,
        use_features=use_features,
    )
    model_summary = build_model_summary(
        root_cause=root_cause,
        quality_summary=quality,
        outlier_impact=outlier_impact,
        band_breakdown=band_breakdown,
        top_error_samples=samples,
    )

    return {
        "model_summary": model_summary,
        "top1_analysis": top1_analysis,
        "root_cause": root_cause,
        "quality_summary": quality,
        "outlier_impact": outlier_impact,
        "band_breakdown": band_breakdown,
        "outlier_contribution": outliers,
        "error_distribution": {"wf": wf_pct, "holdout": ho_pct},
        "worst_trading_days": worst_days,
        "top_error_samples": samples,
        "holdout_premium_rmse_recomputed": holdout_premium_rmse,
    }


def build_premium_analysis_csv(premium_analysis: dict[str, Any]) -> str:
    """Serialize premium analysis sections to a single CSV file."""
    buf = io.StringIO()
    writer = csv.writer(buf)

    def _section(title: str) -> None:
        writer.writerow([])
        writer.writerow([title])

    pa = premium_analysis or {}
    summary = pa.get("model_summary") if isinstance(pa.get("model_summary"), dict) else {}
    if summary:
        _section("Model Summary")
        writer.writerow(["Field", "Value"])
        writer.writerow([
            "Overall Quality",
            f"{summary.get('overall_stars_display') or ''} {summary.get('overall_quality') or ''}".strip(),
        ])
        writer.writerow(["Typical prediction quality", summary.get("typical_prediction_quality") or ""])
        writer.writerow(["Extreme outlier handling", summary.get("extreme_outlier_handling") or ""])
        writer.writerow(["Main weakness", summary.get("main_weakness") or ""])

    root = pa.get("root_cause") if isinstance(pa.get("root_cause"), dict) else {}
    _section("Root Cause Summary")
    writer.writerow(["Field", "Value"])
    writer.writerow(["Primary Cause", root.get("primary_cause") or ""])
    if root.get("rmse_note"):
        writer.writerow(["RMSE Note", root.get("rmse_note")])
    for item in root.get("checklist") or []:
        if isinstance(item, dict) and item.get("text"):
            writer.writerow(["Check", item.get("text")])
    for item in root.get("warnings") or []:
        writer.writerow(["Warning", item])
    for item in root.get("detail_bullets") or []:
        writer.writerow(["Detail", item])
    for idx, bullet in enumerate(root.get("bullets") or [], start=1):
        writer.writerow([f"Evidence {idx}", bullet])
    writer.writerow(["Conclusion", root.get("conclusion") or ""])

    quality = pa.get("quality_summary") if isinstance(pa.get("quality_summary"), dict) else {}
    rel = quality.get("relative_error") if isinstance(quality.get("relative_error"), dict) else {}
    if quality:
        _section("Prediction Quality")
        writer.writerow(["Metric", "Value"])
        for key, label in (
            ("median", "Median Relative Error"),
            ("p90", "90th Percentile"),
            ("p95", "95th Percentile"),
            ("p99", "99th Percentile"),
        ):
            val = rel.get(key)
            writer.writerow([label, f"{float(val):.1f}%" if val is not None else ""])
        prmse = quality.get("premium_rmse_pct")
        writer.writerow(["Premium RMSE", f"{float(prmse):.1f}%" if prmse is not None else ""])
        excl = quality.get("premium_rmse_excl_top_pct")
        excl_pct = quality.get("exclude_top_pct") or 1
        writer.writerow([
            f"Premium RMSE (excluding top {int(excl_pct)}%)",
            f"{float(excl):.1f}%" if excl is not None else "",
        ])

    impact = pa.get("outlier_impact") if isinstance(pa.get("outlier_impact"), dict) else {}
    if impact:
        _section("Outlier Impact")
        writer.writerow(["Field", "Value"])
        writer.writerow(["Tier", impact.get("label") or ""])
        writer.writerow(["Rows", impact.get("row_count") or ""])
        writer.writerow(["Contribution to Σ(rel²) %", impact.get("contribution_pct") or ""])
        writer.writerow(["Status", impact.get("status") or ""])

    _section("Premium RMSE Breakdown")
    writer.writerow(["Premium Band", "Rows", "RMSE %", "Contribution %"])
    for row in pa.get("band_breakdown") or []:
        if not isinstance(row, dict):
            continue
        writer.writerow([
            row.get("band_label") or row.get("band") or "",
            row.get("rows") or "",
            row.get("premium_rmse_pct") if row.get("premium_rmse_pct") is not None else "",
            row.get("contribution_pct") if row.get("contribution_pct") is not None else "",
        ])

    _section("Outlier Contribution")
    writer.writerow(["Largest Errors", "Contribution to Σ(rel²) %"])
    for row in pa.get("outlier_contribution") or []:
        if not isinstance(row, dict):
            continue
        writer.writerow([row.get("label") or "", row.get("contribution_pct") if row.get("contribution_pct") is not None else ""])

    _section("Error Distribution")
    writer.writerow(["Metric", "WF", "Holdout"])
    dist = pa.get("error_distribution") if isinstance(pa.get("error_distribution"), dict) else {}
    wf_d = dist.get("wf") if isinstance(dist.get("wf"), dict) else {}
    ho_d = dist.get("holdout") if isinstance(dist.get("holdout"), dict) else {}
    for key, label in (
        ("median", "Median Error"),
        ("p90", "90th Percentile"),
        ("p95", "95th Percentile"),
        ("p99", "99th Percentile"),
        ("max", "Max Error"),
    ):
        writer.writerow([label, wf_d.get(key) if wf_d.get(key) is not None else "", ho_d.get(key) if ho_d.get(key) is not None else ""])

    _section("Worst Trading Days")
    writer.writerow(["Trading Day", "MAE", "Direction Acc %"])
    for row in pa.get("worst_trading_days") or []:
        if not isinstance(row, dict):
            continue
        writer.writerow([
            row.get("trading_day") or "",
            row.get("mae") if row.get("mae") is not None else "",
            row.get("directional_accuracy_pct") if row.get("directional_accuracy_pct") is not None else "",
        ])

    _section("Top Error Samples")
    writer.writerow(["Time", "Strike", "Actual", "Predicted", "Error", "Reason"])
    for row in pa.get("top_error_samples") or []:
        if not isinstance(row, dict):
            continue
        writer.writerow([
            row.get("time") or "",
            row.get("strike") or "",
            row.get("actual") if row.get("actual") is not None else "",
            row.get("predicted") if row.get("predicted") is not None else "",
            row.get("error") if row.get("error") is not None else "",
            row.get("reason") or "",
        ])

    if pa.get("holdout_premium_rmse_recomputed") is not None:
        _section("Summary")
        writer.writerow(["Holdout Premium RMSE Recomputed %", pa.get("holdout_premium_rmse_recomputed")])

    return buf.getvalue().lstrip("\n")

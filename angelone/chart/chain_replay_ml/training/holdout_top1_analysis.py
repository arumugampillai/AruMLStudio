"""Top 1% holdout error investigation — patterns, comparisons, knowledge export."""

from __future__ import annotations

import csv
import io
import json
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from .evaluator import PREMIUM_METRIC_BANDS, premium_rmse_pct

_PREMIUM_EPS = 1e-9
_TOP1_PCT = 1.0
_GAMMA_THRESHOLD = 0.05
_IV_ZSCORE_THRESHOLD = 2.0

_COMPARE_METRICS: list[tuple[str, str, list[str]]] = [
    ("current_iv", "Current IV", ["current_iv", "roll_iv"]),
    ("gamma", "Gamma", ["gamma", "gamma_x_spot"]),
    ("theta", "Theta", ["theta", "theta_per_min"]),
    ("delta", "Delta", ["delta", "abs_delta"]),
    ("vega", "Vega", ["vega", "vega_per_ivpt"]),
    ("spot_move_5m", "Spot Move 5m", ["spot_change_5m", "spot_change_1m", "spot_change"]),
    ("premium", "Premium", ["ltp"]),
    ("time_to_expiry", "Time to Expiry", ["minutes_to_expiry", "days_to_expiry"]),
    ("distance_from_atm", "Distance from ATM", ["strike_distance_from_atm", "moneyness"]),
]

_DIST_FEATURES: list[tuple[str, str, list[str]]] = [
    ("current_iv", "Current IV", ["current_iv"]),
    ("gamma", "Gamma", ["gamma"]),
    ("theta", "Theta", ["theta"]),
    ("delta", "Delta", ["delta"]),
    ("vega", "Vega", ["vega"]),
    ("premium", "Premium", ["ltp"]),
    ("minutes_to_expiry", "Time to Expiry", ["minutes_to_expiry"]),
]

_GREEK_METRICS = ("gamma", "current_iv", "theta", "delta", "vega")


def _resolve_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def _numeric_series(df: pd.DataFrame, col: str | None) -> pd.Series:
    if not col or col not in df.columns:
        return pd.Series(dtype=float)
    return pd.to_numeric(df[col], errors="coerce")


def _premium_band_label(premium: float) -> str:
    for label, lo, hi in PREMIUM_METRIC_BANDS:
        if hi is None:
            if premium >= lo:
                return f"₹{label}"
        elif lo <= premium < hi:
            return f"₹{label}"
    return "—"


def _strike_distance_bucket(dist: float | None) -> str:
    if dist is None or not np.isfinite(dist):
        return "—"
    ad = abs(float(dist))
    if ad <= 0.5:
        return "ATM"
    if ad <= 1.5:
        return "±1"
    if ad <= 2.5:
        return "±2"
    if ad <= 3.5:
        return "±3"
    if ad <= 5.5:
        return "±5"
    return ">±5"


def _pct_of(mask: np.ndarray, total: int) -> float:
    if total <= 0:
        return 0.0
    return round(float(mask.sum()) / float(total) * 100.0, 1)


def _mean_safe(s: pd.Series) -> float | None:
    if s is None or len(s) == 0:
        return None
    v = s.dropna()
    if v.empty:
        return None
    return float(v.mean())


def _rel_pct_diff(top_mean: float | None, rest_mean: float | None) -> float | None:
    if top_mean is None or rest_mean is None:
        return None
    denom = abs(rest_mean) if abs(rest_mean) > _PREMIUM_EPS else 1.0
    return round((top_mean - rest_mean) / denom * 100.0, 1)


def _percentile_row(s: pd.Series) -> dict[str, float | None]:
    v = s.dropna()
    if v.empty:
        return {"p25": None, "median": None, "p75": None, "p95": None}
    return {
        "p25": round(float(np.percentile(v, 25)), 4),
        "median": round(float(np.percentile(v, 50)), 4),
        "p75": round(float(np.percentile(v, 75)), 4),
        "p95": round(float(np.percentile(v, 95)), 4),
    }


def _star_display(score: float) -> str:
    filled = max(1, min(5, int(round(score))))
    return "★" * filled + "☆" * (5 - filled)


def _rank_stars(rank: int, total: int) -> str:
    if total <= 0:
        return ""
    if rank <= max(1, total // 5):
        return "⭐⭐⭐⭐⭐"
    if rank <= max(1, (2 * total) // 5):
        return "⭐⭐⭐⭐"
    if rank <= max(1, (3 * total) // 5):
        return "⭐⭐⭐"
    if rank <= max(1, (4 * total) // 5):
        return "⭐⭐"
    return "⭐"


def _split_top1(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int]:
    mask = np.isfinite(y_true) & np.isfinite(y_pred) & (np.abs(y_true) > _PREMIUM_EPS)
    yt = np.asarray(y_true, dtype=float)[mask]
    yp = np.asarray(y_pred, dtype=float)[mask]
    rel_sq = np.square((yp - yt) / np.abs(yt))
    n = len(rel_sq)
    if n <= 0:
        return np.array([], dtype=bool), np.array([], dtype=bool), 0
    k = max(1, int(np.ceil(n * _TOP1_PCT / 100.0)))
    order = np.argsort(rel_sq)[::-1]
    top_local = np.zeros(n, dtype=bool)
    top_local[order[:k]] = True
    full_top = np.zeros(len(y_true), dtype=bool)
    full_rest = np.zeros(len(y_true), dtype=bool)
    idxs = np.where(mask)[0]
    full_top[idxs[top_local]] = True
    full_rest[idxs[~top_local]] = True
    return full_top, full_rest, k


def build_top1_error_analysis(
    *,
    ho_df: pd.DataFrame,
    y_ho: pd.Series,
    pred_ho: np.ndarray,
    baseline_ho: pd.Series | np.ndarray | None = None,
    feature_drift_ranking: list[dict[str, Any]] | None = None,
    model_name: str = "",
    model: Any | None = None,
    use_features: list[str] | None = None,
) -> dict[str, Any]:
    yt = pd.to_numeric(y_ho, errors="coerce").to_numpy()
    yp = np.asarray(pred_ho, dtype=float)
    n = min(len(yt), len(yp), len(ho_df))
    if n <= 0:
        return {"ok": False, "error": "no holdout rows"}

    df = ho_df.iloc[:n].reset_index(drop=True)
    yt = yt[:n]
    yp = yp[:n]

    top_mask, rest_mask, top_k = _split_top1(yt, yp)
    if top_k <= 0:
        return {"ok": False, "error": "no valid premium rows for top 1% analysis"}

    top_df = df.loc[top_mask].copy()
    rest_df = df.loc[rest_mask].copy()

    rel_err_pct = np.abs((yp[top_mask] - yt[top_mask]) / np.abs(yt[top_mask])) * 100.0
    avg_premium_error = round(float(np.mean(rel_err_pct)), 1) if len(rel_err_pct) else None

    # --- pattern flags on top 1% ---
    premium_col = _resolve_column(df, ["ltp"]) or "ltp"
    prem_top = _numeric_series(top_df, premium_col if premium_col in top_df.columns else None)
    if prem_top.empty and len(yt[top_mask]):
        prem_top = pd.Series(yt[top_mask])

    expiry_day_pct = 0.0
    if "is_expiry_day" in top_df.columns:
        expiry_day_pct = _pct_of(
            pd.to_numeric(top_df["is_expiry_day"], errors="coerce").fillna(0).to_numpy() >= 0.5,
            len(top_df),
        )

    low_premium_pct = 0.0
    if not prem_top.empty:
        low_premium_pct = _pct_of((prem_top >= 15) & (prem_top < 30), len(prem_top))

    iv_spike_pct = 0.0
    for col in ("iv_zscore_5m", "iv_zscore_1m", "iv_zscore_15m", "current_iv"):
        if col in top_df.columns:
            iv_vals = _numeric_series(top_df, col)
            if col.startswith("iv_zscore"):
                iv_spike_pct = _pct_of(iv_vals.abs().to_numpy() >= _IV_ZSCORE_THRESHOLD, len(iv_vals))
            else:
                rest_iv = _numeric_series(rest_df, col)
                thr = float(rest_iv.quantile(0.90)) if not rest_iv.empty else 0.0
                if thr > 0:
                    iv_spike_pct = _pct_of(iv_vals.to_numpy() >= thr, len(iv_vals))
            break

    gamma_high_pct = 0.0
    for col in ("gamma", "gamma_x_spot"):
        if col in top_df.columns:
            g = _numeric_series(top_df, col)
            gamma_high_pct = _pct_of(g.to_numpy() >= _GAMMA_THRESHOLD, len(g))
            break

    patterns = []
    if expiry_day_pct >= 50:
        patterns.append(f"{expiry_day_pct:.0f}% are Expiry Day")
    if low_premium_pct >= 50:
        patterns.append(f"{low_premium_pct:.0f}% are ₹15–30 premium")
    if iv_spike_pct >= 50:
        patterns.append(f"{iv_spike_pct:.0f}% occur after IV spike")
    if gamma_high_pct >= 50:
        patterns.append(f"{gamma_high_pct:.0f}% occur when Gamma > threshold")

    clarity = min(5.0, 2.0 + len(patterns) * 0.75 + (0.5 if top_k >= 100 else 0))
    executive = {
        "title": "Top 1% Error Investigation",
        "stars_display": _star_display(clarity),
        "rows_analyzed": top_k,
        "avg_premium_error_pct": avg_premium_error,
        "patterns": patterns,
        "expiry_day_pct": expiry_day_pct,
        "low_premium_pct": low_premium_pct,
        "iv_spike_pct": iv_spike_pct,
        "gamma_high_pct": gamma_high_pct,
    }

    # --- metric comparison top vs rest ---
    compare_rows: list[dict[str, Any]] = []
    for key, label, candidates in _COMPARE_METRICS:
        col = _resolve_column(df, candidates)
        if key == "premium" and (col is None or col not in df.columns):
            top_mean = _mean_safe(pd.Series(yt[top_mask]))
            rest_mean = _mean_safe(pd.Series(yt[rest_mask]))
        elif col:
            top_mean = _mean_safe(_numeric_series(top_df, col))
            rest_mean = _mean_safe(_numeric_series(rest_df, col))
        else:
            top_mean = rest_mean = None
        diff = _rel_pct_diff(top_mean, rest_mean)
        compare_rows.append({
            "key": key,
            "metric": label,
            "top1_mean": round(top_mean, 4) if top_mean is not None else None,
            "rest_mean": round(rest_mean, 4) if rest_mean is not None else None,
            "difference_pct": diff,
        })

    # --- distribution comparison ---
    dist_rows: list[dict[str, Any]] = []
    for key, label, candidates in _DIST_FEATURES:
        col = _resolve_column(df, candidates)
        if key == "premium" and (col is None or col not in df.columns):
            top_s = pd.Series(yt[top_mask])
            rest_s = pd.Series(yt[rest_mask])
        elif col:
            top_s = _numeric_series(top_df, col)
            rest_s = _numeric_series(rest_df, col)
        else:
            continue
        dist_rows.append({
            "feature": label,
            "top1": _percentile_row(top_s),
            "rest": _percentile_row(rest_s),
        })

    # --- time of day ---
    hour_rows: list[dict[str, Any]] = []
    if "timestamp" in top_df.columns:
        ts = pd.to_datetime(top_df["timestamp"], errors="coerce", unit="s")
        if ts.isna().all():
            ts = pd.to_datetime(top_df["timestamp"], errors="coerce")
        hours = ts.dt.hour
        buckets = [
            ("9–10", 9, 10), ("10–11", 10, 11), ("11–12", 11, 12),
            ("12–1", 12, 13), ("1–2", 13, 14), ("2–3", 14, 15),
        ]
        for label, lo, hi in buckets:
            cnt = int(((hours >= lo) & (hours < hi)).sum())
            hour_rows.append({"hour": label, "count": cnt})
        late_pct = _pct_of(hours >= 14, len(hours.dropna()))
        if late_pct >= 60:
            executive["late_session_pct"] = late_pct

    # --- expiry categories ---
    expiry_rows: list[dict[str, Any]] = []
    n_top = len(top_df)
    if "is_expiry_day" in top_df.columns:
        day_m = pd.to_numeric(top_df["is_expiry_day"], errors="coerce").fillna(0).to_numpy() >= 0.5
        expiry_rows.append({"category": "Expiry Day", "percentage": _pct_of(day_m, n_top)})
    if "is_expiry_week" in top_df.columns:
        week_m = pd.to_numeric(top_df["is_expiry_week"], errors="coerce").fillna(0).to_numpy() >= 0.5
        day_set = day_m if "is_expiry_day" in top_df.columns else np.zeros(n_top, dtype=bool)
        weekly_only = week_m & ~day_set
        expiry_rows.append({"category": "Weekly Expiry", "percentage": _pct_of(weekly_only, n_top)})
    non_exp = np.ones(n_top, dtype=bool)
    if expiry_rows:
        if "is_expiry_day" in top_df.columns:
            non_exp &= ~day_m
        if "is_expiry_week" in top_df.columns:
            non_exp &= ~week_m
    expiry_rows.append({"category": "Non-expiry", "percentage": _pct_of(non_exp, n_top)})

    # --- premium bands ---
    premium_band_rows: list[dict[str, Any]] = []
    if not prem_top.empty:
        for label, lo, hi in PREMIUM_METRIC_BANDS:
            if int(lo) >= 100:
                continue
            if hi is None:
                band_m = prem_top >= lo
            else:
                band_m = (prem_top >= lo) & (prem_top < hi)
            premium_band_rows.append({
                "band": f"₹{label}",
                "percentage": _pct_of(band_m.to_numpy(), len(prem_top)),
            })

    # --- strike distance ---
    strike_rows: list[dict[str, Any]] = []
    dist_col = _resolve_column(df, ["strike_distance_from_atm", "moneyness"])
    if dist_col:
        dists = _numeric_series(top_df, dist_col)
        buckets: dict[str, int] = {}
        for val in dists.dropna():
            b = _strike_distance_bucket(float(val))
            buckets[b] = buckets.get(b, 0) + 1
        for label in ("ATM", "±1", "±2", "±3", "±5", ">±5"):
            if label in buckets:
                strike_rows.append({"distance": label, "count": buckets[label]})

    # --- greeks ranking ---
    greek_rows: list[dict[str, Any]] = []
    for gkey in _GREEK_METRICS:
        row = next((r for r in compare_rows if r["key"] == gkey or gkey in r["key"]), None)
        if row and row.get("difference_pct") is not None:
            greek_rows.append({
                "feature": row["metric"],
                "difference_pct": row["difference_pct"],
            })
    greek_rows.sort(key=lambda r: abs(float(r.get("difference_pct") or 0)), reverse=True)
    for i, row in enumerate(greek_rows, start=1):
        row["rank"] = i
        row["rank_stars"] = _rank_stars(i, len(greek_rows))

    # --- feature importance intersection ---
    top1_feat_diff: dict[str, float] = {}
    for feat in df.columns:
        if feat in ("timestamp", "trading_day", "token", "option_type"):
            continue
        try:
            top_m = _mean_safe(_numeric_series(top_df, feat))
            rest_m = _mean_safe(_numeric_series(rest_df, feat))
            diff = _rel_pct_diff(top_m, rest_m)
            if diff is not None and abs(diff) >= 15:
                top1_feat_diff[str(feat)] = diff
        except (TypeError, ValueError):
            continue

    feat_rows: list[dict[str, Any]] = []
    drift_map = {
        str(r.get("feature") or ""): r
        for r in (feature_drift_ranking or [])
        if isinstance(r, dict)
    }
    for feat, diff in sorted(top1_feat_diff.items(), key=lambda x: abs(x[1]), reverse=True)[:15]:
        drift_row = drift_map.get(feat, {})
        drift_pct = drift_row.get("drift_pct")
        importance = drift_row.get("importance")
        risk = "🔴" if (
            drift_pct is not None and abs(float(drift_pct)) >= 10
            and importance is not None and float(importance) >= 0.1
            and abs(diff) >= 50
        ) else ("🟡" if abs(diff) >= 30 else "—")
        feat_rows.append({
            "feature": feat,
            "drift_pct": drift_pct,
            "importance": importance,
            "top1_difference_pct": diff,
            "risk": risk,
        })

    # --- conclusion ---
    root_causes: list[str] = []
    if expiry_day_pct >= 60:
        root_causes.append("Expiry Gamma")
    if low_premium_pct >= 60:
        root_causes.append("Low Premium")
    if iv_spike_pct >= 60:
        root_causes.append("High IV")
    spot_row = next((r for r in compare_rows if r["key"] == "spot_move_5m"), None)
    if spot_row and spot_row.get("difference_pct") is not None and abs(float(spot_row["difference_pct"])) >= 40:
        root_causes.append("Large Spot Movement")

    confidence = min(99.0, 55.0 + len(root_causes) * 12.0 + len(patterns) * 4.0)
    if not root_causes and patterns:
        root_causes.append("Extreme relative errors on difficult samples")

    recommendation = "Monitor top-error patterns and consider targeted feature engineering."
    if "Expiry Gamma" in root_causes and "Low Premium" in root_causes:
        recommendation = "Train a dedicated model for expiry gamma situations in low-premium options."
    elif "Expiry Gamma" in root_causes:
        recommendation = "Train a dedicated model for expiry gamma conditions."
    elif "High IV" in root_causes:
        recommendation = "Add IV-regime features or a separate high-IV specialist model."

    conclusion = {
        "title": "Top 1% Investigation",
        "confidence_pct": round(confidence, 0),
        "root_causes": root_causes,
        "recommendation": recommendation,
        "finding_text": (
            f"Model struggles during {', '.join(c.lower() for c in root_causes)} conditions."
            if root_causes else "Model shows concentrated errors in top 1% of holdout rows."
        ),
        "knowledge_category": "Model Weakness",
        "evidence_rows": top_k,
    }

    holdout_rmse = premium_rmse_pct(yt, yp)

    from .holdout_top1_drivers import analyze_top1_error_drivers

    driver_analysis = analyze_top1_error_drivers(
        top_df=top_df,
        rest_df=rest_df,
        compare_rows=compare_rows,
        model=model,
        X_top1=df.loc[top_mask, [f for f in (use_features or []) if f in df.columns]] if use_features else None,
        y_top1=yt[top_mask],
        features=use_features,
    )
    if driver_analysis.get("primary_driver"):
        conclusion["primary_driver"] = driver_analysis["primary_driver"]
        if driver_analysis.get("primary_driver_by_error"):
            conclusion["primary_driver_by_error"] = driver_analysis["primary_driver_by_error"]
        rec = driver_analysis.get("feature_recommendations") if isinstance(
            driver_analysis.get("feature_recommendations"), dict,
        ) else {}
        if rec.get("notes"):
            conclusion["driver_notes"] = rec["notes"]
        if not rec.get("recommend_new_features"):
            conclusion["feature_engineering"] = (
                "No new features recommended beyond Gamma, IV, and Time-to-Expiry; "
                "prioritize expiry-gamma regime handling."
            )

    return {
        "ok": True,
        "model_name": model_name,
        "executive_summary": executive,
        "metric_comparison": compare_rows,
        "distribution_comparison": dist_rows,
        "time_analysis": hour_rows,
        "expiry_analysis": expiry_rows,
        "premium_band_analysis": premium_band_rows,
        "strike_distance": strike_rows,
        "greeks_ranking": greek_rows,
        "feature_risk_matrix": feat_rows,
        "driver_analysis": driver_analysis,
        "conclusion": conclusion,
        "holdout_premium_rmse_pct": holdout_rmse,
        "total_holdout_rows": n,
        "top1_row_count": top_k,
    }


def save_top1_investigation_knowledge(
    data_dir: str,
    analysis: dict[str, Any],
    *,
    model_name: str,
) -> dict[str, Any]:
    """Persist investigation conclusion to the knowledge base."""
    from chain_replay_ml.fold_research.knowledge_store import KnowledgeStore

    conclusion = analysis.get("conclusion") if isinstance(analysis.get("conclusion"), dict) else {}
    if not conclusion:
        return {"ok": False, "error": "no conclusion to save"}

    finding_text = str(conclusion.get("finding_text") or "Holdout top 1% error pattern")
    category = str(conclusion.get("knowledge_category") or "Model Weakness")
    conf_pct = float(conclusion.get("confidence_pct") or 0)
    root_causes = conclusion.get("root_causes") or []
    key_slug = "_".join(c.lower().replace(" ", "_") for c in root_causes[:2]) or "top1_errors"
    finding_key = f"holdout_top1:{model_name}:{key_slug}"

    metadata = {
        "source": "holdout_top1_investigation",
        "model_name": model_name,
        "confidence_pct": conf_pct,
        "root_causes": root_causes,
        "recommendation": conclusion.get("recommendation"),
        "rows_analyzed": analysis.get("top1_row_count"),
        "avg_premium_error_pct": (analysis.get("executive_summary") or {}).get("avg_premium_error_pct"),
        "patterns": (analysis.get("executive_summary") or {}).get("patterns"),
    }

    with KnowledgeStore(data_dir) as store:
        doc = store.get_finding_by_key(finding_key)
        if not doc:
            doc = store.upsert_finding(
                finding_key=finding_key,
                finding=finding_text,
                category=category,
                metadata=metadata,
            )
        finding_id = str(doc.get("finding_id") or "")
        if not finding_id:
            return {"ok": False, "error": "could not create finding"}

        store.add_evidence(
            finding_id,
            {
                "model_id": model_name,
                "trade_count": int(analysis.get("top1_row_count") or 0),
                "supports_finding": True,
                "evidence_quality": "high" if conf_pct >= 80 else "moderate",
                "notes": (
                    f"Holdout top 1% investigation — {int(analysis.get('top1_row_count') or 0)} catastrophic rows, "
                    f"confidence {conf_pct:.0f}%"
                ),
                "evidence_json": {
                    "confidence_pct": conf_pct,
                    "root_causes": root_causes,
                    "recommendation": conclusion.get("recommendation"),
                    "executive_summary": analysis.get("executive_summary"),
                },
            },
        )
        store.add_link(finding_id, link_type="model", link_ref=model_name, link_label=model_name)

        meta = dict(doc.get("metadata") or {})
        meta.update(metadata)
        store.conn.execute(
            "UPDATE knowledge_findings SET metadata_json = ?, updated_at = ? WHERE finding_id = ?",
            (json.dumps(meta, default=str), datetime.now(timezone.utc).isoformat(), finding_id),
        )
        store.conn.commit()
        refreshed = store.get_finding(finding_id)

    return {"ok": True, "finding": refreshed, "finding_key": finding_key}


def build_top1_analysis_csv(analysis: dict[str, Any]) -> str:
    buf = io.StringIO()
    w = csv.writer(buf)

    def section(title: str) -> None:
        w.writerow([])
        w.writerow([title])

    ex = analysis.get("executive_summary") or {}
    section("Executive Summary")
    w.writerow(["Field", "Value"])
    w.writerow(["Title", ex.get("title") or ""])
    w.writerow(["Clarity", ex.get("stars_display") or ""])
    w.writerow(["Rows analyzed", ex.get("rows_analyzed") or ""])
    w.writerow(["Average Premium Error %", ex.get("avg_premium_error_pct") or ""])
    for p in ex.get("patterns") or []:
        w.writerow(["Pattern", p])

    section("Top 1% vs Remaining 99%")
    w.writerow(["Metric", "Top 1%", "Remaining 99%", "Difference %"])
    for row in analysis.get("metric_comparison") or []:
        w.writerow([
            row.get("metric"), row.get("top1_mean"), row.get("rest_mean"), row.get("difference_pct"),
        ])

    section("Distribution Comparison")
    w.writerow(["Feature", "Group", "P25", "Median", "P75", "P95"])
    for row in analysis.get("distribution_comparison") or []:
        for grp, label in (("top1", "Top 1%"), ("rest", "Remaining 99%")):
            pct = row.get(grp) or {}
            w.writerow([
                row.get("feature"), label,
                pct.get("p25"), pct.get("median"), pct.get("p75"), pct.get("p95"),
            ])

    drivers = analysis.get("driver_analysis") if isinstance(analysis.get("driver_analysis"), dict) else {}
    section("Primary Driver Analysis")
    w.writerow(["Primary Driver", drivers.get("primary_driver") or ""])
    w.writerow(["Method", drivers.get("importance_method") or "separation"])
    w.writerow(["Driver", "Separation Score", "Error Contribution %", "Top Median", "Rest Median"])
    for row in drivers.get("driver_separation_ranking") or []:
        w.writerow([
            row.get("driver"),
            row.get("separation_score"),
            row.get("error_contribution_pct"),
            row.get("top_median"),
            row.get("rest_median"),
        ])

    section("Top 1% Error Feature Importance")
    w.writerow(["Feature", "Error Contribution %", "Method"])
    for row in drivers.get("feature_error_importance") or []:
        w.writerow([
            row.get("feature"),
            row.get("error_contribution_pct"),
            row.get("method"),
        ])

    rec = drivers.get("feature_recommendations") if isinstance(drivers.get("feature_recommendations"), dict) else {}
    if rec:
        section("Feature Recommendations")
        w.writerow(["Recommend New Features", rec.get("recommend_new_features")])
        w.writerow(["Baseline Trio %", rec.get("baseline_trio_pct")])
        w.writerow(["Notes", rec.get("notes") or ""])
        for cand in rec.get("candidates") or []:
            w.writerow([
                "Candidate",
                cand.get("feature"),
                cand.get("error_contribution_pct"),
            ])

    conc = analysis.get("conclusion") or {}
    section("Conclusion")
    w.writerow(["Confidence %", conc.get("confidence_pct") or ""])
    w.writerow(["Primary Driver", conc.get("primary_driver") or drivers.get("primary_driver") or ""])
    for rc in conc.get("root_causes") or []:
        w.writerow(["Root Cause", rc])
    w.writerow(["Recommendation", conc.get("recommendation") or ""])
    if conc.get("feature_engineering"):
        w.writerow(["Feature Engineering", conc.get("feature_engineering")])
    if conc.get("driver_notes"):
        w.writerow(["Driver Notes", conc.get("driver_notes")])
    w.writerow(["Finding", conc.get("finding_text") or ""])

    return buf.getvalue().lstrip("\n")

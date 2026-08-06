"""Why-driven fold diagnostics — context, distribution shift, outliers, narratives."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .dataset_loader import DatasetLoaderError, load_dataset_frame, parquet_column_names
from .holdout_days_analysis import (
    _day_volatility_stats,
    _resolve_col,
    _numeric,
)

# Context / distribution columns we try to load (plus model features).
_CONTEXT_WANT = (
    "trading_day",
    "timestamp",
    "spot",
    "underlying_spot",
    "spot_ltp",
    "ltp",
    "current_iv",
    "implied_vol",
    "iv_zscore_1m",
    "iv_zscore_5m",
    "iv_zscore_15m",
    "delta",
    "abs_delta",
    "volume",
    "oi",
    "open_interest",
    "days_to_expiry",
    "is_expiry_day",
    "minute_of_day",
    "is_first_hour",
    "is_last_hour",
    "spot_change_5m",
    "spot_change_1m",
)

_DIST_PRIORITY = (
    "iv_zscore_5m",
    "iv_zscore_1m",
    "iv_zscore_15m",
    "current_iv",
    "implied_vol",
    "ltp",
    "delta",
    "abs_delta",
    "volume",
    "oi",
    "spot",
    "spot_change_5m",
    "days_to_expiry",
)

_SEVERITY_HUGE = 0.50   # |pct Δ| ≥ 50%
_SEVERITY_LARGE = 0.25  # |pct Δ| ≥ 25%
_Z_NOTEWORTHY = 1.5


def _num(v: Any) -> float | None:
    if v is None or v == "":
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if f != f:
        return None
    return f


def _pct_change(a: float | None, b: float | None) -> float | None:
    if a is None or b is None:
        return None
    if abs(a) < 1e-12:
        return None
    return (b - a) / abs(a) * 100.0


def _severity(pct: float | None, *, abs_z: float | None = None) -> str:
    if abs_z is not None and abs_z >= 2.5:
        return "huge"
    if pct is not None and abs(pct) >= _SEVERITY_HUGE * 100:
        return "huge"
    if abs_z is not None and abs_z >= 2.0:
        return "large"
    if pct is not None and abs(pct) >= _SEVERITY_LARGE * 100:
        return "large"
    if abs_z is not None and abs_z >= _Z_NOTEWORTHY:
        return "moderate"
    if pct is not None and abs(pct) >= 10.0:
        return "moderate"
    return "mild"


def _severity_emoji(sev: str) -> str:
    return {"huge": "🔴", "large": "🟠", "moderate": "🟡", "mild": "⚪"}.get(sev, "⚪")


def _slice_bounds(fold_def: dict[str, Any], which: str) -> tuple[int | None, int | None, int | None]:
    block = fold_def.get(which) if isinstance(fold_def.get(which), dict) else {}
    start = block.get("start")
    stop = block.get("stop")
    try:
        s = int(start) if start is not None else None
        e = int(stop) if stop is not None else None
    except (TypeError, ValueError):
        return None, None, None
    if s is None or e is None or e <= s:
        return s, e, None
    return s, e, e - s


def _fmt_pts(v: float | None) -> str:
    if v is None:
        return "—"
    return f"{v:,.0f}" if abs(v) >= 10 else f"{v:,.2f}"


def _fmt_rupee_range(lo: float | None, hi: float | None) -> str:
    if lo is None or hi is None:
        return "—"
    return f"₹{lo:,.0f} → ₹{hi:,.0f}"


def _fmt_spot_range(lo: float | None, hi: float | None) -> str:
    if lo is None or hi is None:
        return "—"
    pts = hi - lo
    return f"{lo:,.0f} → {hi:,.0f} ({_fmt_pts(pts)} pts)"


def _session_bucket(day_df: pd.DataFrame) -> str | None:
    """Rough session label from timestamps / minute_of_day."""
    hour_means: list[float] = []
    if "minute_of_day" in day_df.columns:
        m = pd.to_numeric(day_df["minute_of_day"], errors="coerce").dropna()
        if len(m):
            hour_means.append(float(m.mean()) / 60.0)
    if "timestamp" in day_df.columns:
        ts = pd.to_numeric(day_df["timestamp"], errors="coerce").dropna()
        if len(ts):
            # Epoch seconds → approximate IST hour (UTC+5:30)
            hours = ((ts / 3600.0) + 5.5) % 24
            hour_means.append(float(hours.mean()))
    if not hour_means:
        if "is_first_hour" in day_df.columns:
            if pd.to_numeric(day_df["is_first_hour"], errors="coerce").fillna(0).mean() >= 0.5:
                return "Opening"
        if "is_last_hour" in day_df.columns:
            if pd.to_numeric(day_df["is_last_hour"], errors="coerce").fillna(0).mean() >= 0.5:
                return "Closing"
        return None
    h = float(np.mean(hour_means))
    if h < 10.5:
        return "Opening"
    if h < 12.5:
        return "Morning"
    if h < 14.0:
        return "Lunch"
    if h < 15.0:
        return "Afternoon"
    return "Closing"


def _regime_label(vol: dict[str, Any], *, session: str | None, dte: float | None) -> str:
    parts: list[str] = []
    ret = _num(vol.get("spot_return_pct"))
    spot_range_pct = _num(vol.get("spot_range_pct"))
    iv_abs = _num(vol.get("iv_abs_mean"))
    iv_mean = _num(vol.get("iv_mean"))

    if ret is not None and abs(ret) >= 0.35:
        parts.append("Trending Up" if ret > 0 else "Trending Down")
    else:
        parts.append("Sideways")

    high_vol = False
    if spot_range_pct is not None and spot_range_pct >= 0.8:
        high_vol = True
    if iv_abs is not None and iv_abs >= 1.2:
        high_vol = True
    if iv_mean is not None and abs(iv_mean) >= 20 and "iv_zscore" not in str(vol.get("iv_column") or ""):
        # raw IV in percent-ish
        high_vol = high_vol or iv_mean >= 18
    parts.append("High Volatility" if high_vol else "Low Volatility")

    if vol.get("is_expiry_day") or (dte is not None and dte <= 0.5):
        if session in ("Opening", "Morning"):
            parts.append("Expiry Morning")
        elif session in ("Afternoon", "Closing"):
            parts.append("Expiry Afternoon")
        else:
            parts.append("Expiry Day")
    elif session == "Opening":
        parts.append("Opening Volatility")
    elif session == "Lunch":
        parts.append("Lunch Consolidation")
    elif session == "Closing":
        parts.append("Closing Momentum")

    return " · ".join(parts)


def _gap_from_previous(
    full_df: pd.DataFrame,
    trading_days: list[str],
    spot_col: str | None,
) -> dict[str, Any]:
    if not trading_days or not spot_col or "trading_day" not in full_df.columns:
        return {"gap_pts": None, "gap_pct": None, "prior_day": None, "display": "—"}
    days_sorted = sorted(full_df["trading_day"].dropna().astype(str).unique().tolist())
    first = trading_days[0]
    try:
        idx = days_sorted.index(first)
    except ValueError:
        return {"gap_pts": None, "gap_pct": None, "prior_day": None, "display": "—"}
    if idx <= 0:
        return {"gap_pts": None, "gap_pct": None, "prior_day": None, "display": "—"}
    prior = days_sorted[idx - 1]
    prior_spot = _numeric(full_df[full_df["trading_day"].astype(str) == prior], spot_col).dropna()
    cur_spot = _numeric(full_df[full_df["trading_day"].astype(str) == first], spot_col).dropna()
    if not len(prior_spot) or not len(cur_spot):
        return {"gap_pts": None, "gap_pct": None, "prior_day": prior, "display": "—"}
    prior_close = float(prior_spot.iloc[-1])
    cur_open = float(cur_spot.iloc[0])
    gap = cur_open - prior_close
    gap_pct = (gap / abs(prior_close) * 100.0) if abs(prior_close) > 1e-9 else None
    return {
        "gap_pts": round(gap, 2),
        "gap_pct": round(gap_pct, 4) if gap_pct is not None else None,
        "prior_day": prior,
        "display": f"{gap:+.0f} pts" if abs(gap) >= 1 else f"{gap:+.2f} pts",
    }


def _iv_change(day_df: pd.DataFrame, iv_col: str | None) -> float | None:
    if not iv_col:
        return None
    iv = _numeric(day_df, iv_col).dropna()
    if len(iv) < 2:
        return None
    return float(iv.iloc[-1] - iv.iloc[0])


def _expiry_distance(day_df: pd.DataFrame) -> dict[str, Any]:
    dte = None
    if "days_to_expiry" in day_df.columns:
        s = pd.to_numeric(day_df["days_to_expiry"], errors="coerce").dropna()
        if len(s):
            dte = float(s.median())
    is_exp = None
    if "is_expiry_day" in day_df.columns:
        is_exp = bool(pd.to_numeric(day_df["is_expiry_day"], errors="coerce").fillna(0).max() >= 1)
    if dte is not None:
        # Prefer integer-ish T-n label
        n = int(round(dte))
        label = f"T-{n}" if n > 0 else "T-0"
    elif is_exp:
        label = "T-0"
    else:
        label = "—"
    return {"days_to_expiry": dte, "is_expiry_day": is_exp, "display": label}


def build_fold_context(
    fold_df: pd.DataFrame,
    *,
    fold: int,
    label: str,
    trading_days: list[str] | None = None,
    full_df: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Market / sample summary for one fold validation window."""
    if fold_df is None or len(fold_df) == 0:
        return {
            "available": False,
            "fold": fold,
            "label": label,
            "message": "No validation rows for this fold.",
            "rows": [],
        }

    days = trading_days or []
    if not days and "trading_day" in fold_df.columns:
        days = sorted(fold_df["trading_day"].dropna().astype(str).unique().tolist())

    vol = _day_volatility_stats(fold_df)
    session = _session_bucket(fold_df)
    expiry = _expiry_distance(fold_df)
    spot_col = vol.get("spot_column")
    gap = _gap_from_previous(full_df if full_df is not None else fold_df, days, spot_col)
    iv_chg = _iv_change(fold_df, vol.get("iv_column"))
    regime = _regime_label(vol, session=session, dte=_num(expiry.get("days_to_expiry")))

    delta_col = _resolve_col(fold_df, ("delta", "abs_delta"))
    vol_col = _resolve_col(fold_df, ("volume",))
    oi_col = _resolve_col(fold_df, ("oi", "open_interest"))
    delta_mean = float(_numeric(fold_df, delta_col).mean()) if delta_col else None
    volume_mean = float(_numeric(fold_df, vol_col).mean()) if vol_col else None
    oi_mean = float(_numeric(fold_df, oi_col).mean()) if oi_col else None

    # Premium lo/hi for display
    prem_col = vol.get("premium_column")
    prem = _numeric(fold_df, prem_col).dropna() if prem_col else pd.Series(dtype=float)
    prem_lo = float(prem.min()) if len(prem) else None
    prem_hi = float(prem.max()) if len(prem) else None

    rows = [
        {"key": "validation_days", "label": "Validation day(s)", "value": ", ".join(days) if days else "—"},
        {"key": "market_regime", "label": "Market regime", "value": regime},
        {
            "key": "spot_range",
            "label": "Spot range",
            "value": _fmt_spot_range(vol.get("spot_low"), vol.get("spot_high")),
        },
        {
            "key": "premium_range",
            "label": "Premium range",
            "value": _fmt_rupee_range(prem_lo, prem_hi),
        },
        {
            "key": "iv_avg",
            "label": "IV Avg",
            "value": f"{vol['iv_mean']:.2f}" if vol.get("iv_mean") is not None else "—",
        },
        {
            "key": "iv_change",
            "label": "IV Change",
            "value": f"{iv_chg:+.2f}" if iv_chg is not None else "—",
        },
        {
            "key": "gap",
            "label": "Gap from previous day",
            "value": gap.get("display") or "—",
        },
        {
            "key": "expiry_distance",
            "label": "Expiry distance",
            "value": expiry.get("display") or "—",
        },
        {
            "key": "sample_count",
            "label": "Sample count",
            "value": f"{len(fold_df):,}",
        },
        {
            "key": "avg_delta",
            "label": "Avg option delta",
            "value": f"{delta_mean:.3f}" if delta_mean is not None else "—",
        },
        {
            "key": "avg_volume",
            "label": "Avg volume",
            "value": f"{volume_mean:,.0f}" if volume_mean is not None else "—",
        },
        {
            "key": "avg_oi",
            "label": "Avg OI",
            "value": f"{oi_mean:,.0f}" if oi_mean is not None else "—",
        },
        {
            "key": "session",
            "label": "Session",
            "value": session or "—",
        },
    ]

    return {
        "available": True,
        "fold": fold,
        "label": label,
        "trading_days": days,
        "market_regime": regime,
        "session": session,
        "sample_count": len(fold_df),
        "volatility": vol,
        "gap": gap,
        "expiry": expiry,
        "iv_change": round(iv_chg, 4) if iv_chg is not None else None,
        "premium_lo": prem_lo,
        "premium_hi": prem_hi,
        "avg_delta": round(delta_mean, 4) if delta_mean is not None else None,
        "avg_volume": round(volume_mean, 2) if volume_mean is not None else None,
        "avg_oi": round(oi_mean, 2) if oi_mean is not None else None,
        "rows": rows,
        "message": None,
    }


def _series_stats(s: pd.Series) -> dict[str, float | None]:
    v = pd.to_numeric(s, errors="coerce").dropna()
    if not len(v):
        return {"mean": None, "std": None, "p10": None, "p50": None, "p90": None, "n": 0}
    return {
        "mean": float(v.mean()),
        "std": float(v.std()) if len(v) > 1 else 0.0,
        "p10": float(v.quantile(0.10)),
        "p50": float(v.quantile(0.50)),
        "p90": float(v.quantile(0.90)),
        "n": int(len(v)),
    }


def build_feature_distribution_shift(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    features: list[str],
    *,
    label_a: str,
    label_b: str,
    limit: int = 25,
) -> dict[str, Any]:
    """Compare input means Fold A vs Fold B (actionable distribution drift)."""
    cols = [c for c in features if c in df_a.columns and c in df_b.columns]
    # Prefer known market features first, then rest by abs pct change
    preferred = [c for c in _DIST_PRIORITY if c in cols]
    rest = [c for c in cols if c not in preferred]
    ordered = preferred + rest

    rows: list[dict[str, Any]] = []
    for feat in ordered:
        sa = _series_stats(df_a[feat])
        sb = _series_stats(df_b[feat])
        ma, mb = sa["mean"], sb["mean"]
        if ma is None or mb is None:
            continue
        pct = _pct_change(ma, mb)
        delta = mb - ma
        # pooled std for crude z of mean difference
        std_a = sa["std"] or 0.0
        std_b = sb["std"] or 0.0
        pooled = ((std_a ** 2 + std_b ** 2) / 2.0) ** 0.5
        z = abs(delta) / pooled if pooled and pooled > 1e-12 else None
        sev = _severity(pct, abs_z=z)
        rows.append({
            "feature": feat,
            "fold_a": round(ma, 6),
            "fold_b": round(mb, 6),
            "delta": round(delta, 6),
            "pct_change": round(pct, 2) if pct is not None else None,
            "abs_pct": abs(pct) if pct is not None else None,
            "z_diff": round(z, 3) if z is not None else None,
            "severity": sev,
            "severity_emoji": _severity_emoji(sev),
            "display_delta": (
                f"{pct:+.0f}%" if pct is not None and abs(pct) >= 1
                else (f"{delta:+.3g}" if abs(delta) < 1 else f"{delta:+.2f}")
            ),
        })

    rows.sort(key=lambda r: (r.get("abs_pct") is None, -(r.get("abs_pct") or 0), -(r.get("z_diff") or 0)))
    top = rows[:limit]
    return {
        "available": bool(top),
        "message": None if top else "No overlapping numeric features to compare.",
        "label_a": label_a,
        "label_b": label_b,
        "rows": top,
        "largest_shifts": [r for r in top if r.get("severity") in ("huge", "large")][:8],
    }


def build_fold_outlier_scores(
    fold_df: pd.DataFrame,
    train_df: pd.DataFrame,
    features: list[str],
    *,
    fold: int,
    label: str,
    limit: int = 12,
) -> dict[str, Any]:
    """What’s unique about this fold vs its own training window (z-scores)."""
    cols = [c for c in features if c in fold_df.columns and c in train_df.columns]
    rows: list[dict[str, Any]] = []
    for feat in cols:
        fstats = _series_stats(fold_df[feat])
        tstats = _series_stats(train_df[feat])
        fm, tm, tstd = fstats["mean"], tstats["mean"], tstats["std"]
        if fm is None or tm is None:
            continue
        z = None
        if tstd is not None and tstd > 1e-12:
            z = (fm - tm) / tstd
        # percentile of fold-mean among training values
        train_vals = pd.to_numeric(train_df[feat], errors="coerce").dropna()
        pctile = None
        if len(train_vals):
            pctile = float((train_vals <= fm).mean() * 100.0)
        abs_z = abs(z) if z is not None else None
        if abs_z is None or abs_z < 0.75:
            continue
        rows.append({
            "feature": feat,
            "fold_mean": round(fm, 6),
            "train_mean": round(tm, 6),
            "difference": round(fm - tm, 6),
            "z_score": round(z, 3) if z is not None else None,
            "abs_z": abs_z,
            "percentile": round(pctile, 1) if pctile is not None else None,
            "display": f"{z:+.1f}σ" if z is not None else "—",
            "severity_emoji": _severity_emoji(_severity(None, abs_z=abs_z)),
        })
    rows.sort(key=lambda r: -(r.get("abs_z") or 0))
    top = rows[:limit]
    return {
        "available": bool(top),
        "fold": fold,
        "label": label,
        "rows": top,
        "message": None if top else "No strong outliers vs training distribution.",
    }


def _context_why_bullets(
    ctx_a: dict[str, Any],
    ctx_b: dict[str, Any],
    *,
    worse_is_b: bool,
) -> list[str]:
    """Bullets describing how the worse fold differs on market context."""
    bullets: list[str] = []
    better, worse = (ctx_a, ctx_b) if worse_is_b else (ctx_b, ctx_a)

    va = better.get("volatility") if isinstance(better.get("volatility"), dict) else {}
    vb = worse.get("volatility") if isinstance(worse.get("volatility"), dict) else {}

    def add_pct(a_key: str, nice: str) -> None:
        a, b = _num(va.get(a_key)), _num(vb.get(a_key))
        pct = _pct_change(a, b)
        if pct is not None and abs(pct) >= 15:
            bullets.append(f"{nice} {pct:+.0f}%")

    add_pct("iv_mean", "IV average")
    if not any(b.startswith("IV average") for b in bullets):
        add_pct("iv_abs_mean", "IV |z|")

    add_pct("premium_mean", "Premium")
    add_pct("premium_std", "Premium volatility")
    add_pct("spot_range_pct", "Spot volatility")
    add_pct("spot_std", "Spot std")

    pa = better.get("premium_lo"), better.get("premium_hi")
    pb = worse.get("premium_lo"), worse.get("premium_hi")
    if all(x is not None for x in (*pa, *pb)):
        mid_a = (float(pa[0]) + float(pa[1])) / 2.0
        mid_b = (float(pb[0]) + float(pb[1])) / 2.0
        pct = _pct_change(mid_a, mid_b)
        if pct is not None and abs(pct) >= 25 and not any("Premium" in b for b in bullets):
            if mid_b >= mid_a * 1.8:
                bullets.append(f"Premiums {mid_b / mid_a:.1f}× larger")
            else:
                bullets.append(f"Premium mid {pct:+.0f}%")

    gap = worse.get("gap") if isinstance(worse.get("gap"), dict) else {}
    gap_pts = _num(gap.get("gap_pts"))
    if gap_pts is not None and abs(gap_pts) >= 40:
        bullets.append(f"Gap {gap_pts:+.0f} pts from prior day")

    exp = worse.get("expiry") if isinstance(worse.get("expiry"), dict) else {}
    session = worse.get("session")
    dte = _num(exp.get("days_to_expiry"))
    if exp.get("is_expiry_day") or (dte is not None and dte <= 0.5):
        sess = session or ""
        if sess in ("Afternoon", "Closing"):
            bullets.append("Expiry T-0 afternoon")
        elif sess in ("Opening", "Morning"):
            bullets.append("Expiry T-0 morning")
        else:
            bullets.append(f"Expiry {exp.get('display') or 'T-0'}")
    elif exp.get("display") and exp.get("display") != "—":
        bullets.append(f"Expiry {exp['display']}")

    regime = worse.get("market_regime")
    if regime:
        bullets.append(f"Regime: {regime}")

    return bullets


def build_why_explanations(
    *,
    label_a: str,
    label_b: str,
    metrics_a: dict[str, Any],
    metrics_b: dict[str, Any],
    context_a: dict[str, Any],
    context_b: dict[str, Any],
    distribution: dict[str, Any],
    outliers_a: dict[str, Any],
    outliers_b: dict[str, Any],
    error_histograms: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Attach Why? bullets so the user rarely needs to compare tables manually."""
    mae_a, mae_b = _num(metrics_a.get("mae")), _num(metrics_b.get("mae"))
    dir_a = _num(metrics_a.get("directional_accuracy_pct"))
    dir_b = _num(metrics_b.get("directional_accuracy_pct"))

    worse_is_b = True
    if mae_a is not None and mae_b is not None:
        worse_is_b = mae_b >= mae_a
    elif dir_a is not None and dir_b is not None:
        worse_is_b = dir_b <= dir_a

    worse_label = label_b if worse_is_b else label_a
    better_label = label_a if worse_is_b else label_b
    worse_metrics = metrics_b if worse_is_b else metrics_a
    better_metrics = metrics_a if worse_is_b else metrics_b
    worse_ctx = context_b if worse_is_b else context_a
    worse_outliers = outliers_b if worse_is_b else outliers_a
    _ = worse_ctx  # available for future per-context metric wiring


    context_bullets = _context_why_bullets(context_a, context_b, worse_is_b=worse_is_b)

    # Feature drift
    drift_bullets: list[str] = []
    for row in (distribution.get("largest_shifts") or distribution.get("rows") or [])[:5]:
        if not isinstance(row, dict):
            continue
        if row.get("severity") not in ("huge", "large", "moderate"):
            continue
        feat = row.get("feature")
        disp = row.get("display_delta")
        emoji = row.get("severity_emoji") or ""
        # When worse is A, flip sign interpretation of A→B display
        if worse_is_b:
            drift_bullets.append(f"{emoji} {feat} {disp}".strip())
        else:
            pct = row.get("pct_change")
            if pct is not None:
                drift_bullets.append(f"{emoji} {feat} {-pct:+.0f}% (vs {label_b})".strip())
            else:
                drift_bullets.append(f"{emoji} {feat} shifted vs {label_b}".strip())

    top_z = None
    for row in (worse_outliers.get("rows") or [])[:1]:
        if isinstance(row, dict) and row.get("z_score") is not None:
            top_z = row
            break
    outlier_bullets: list[str] = []
    if top_z:
        outlier_bullets.append(
            f"Feature drift {top_z.get('display')} on {top_z.get('feature')}"
        )
    for row in (worse_outliers.get("rows") or [])[1:4]:
        if isinstance(row, dict) and (row.get("abs_z") or 0) >= 2.0:
            outlier_bullets.append(f"{row.get('feature')} {row.get('display')}")

    # Error tail
    err_bullets: list[str] = []
    hist = error_histograms if isinstance(error_histograms, dict) else {}
    if hist.get("available"):
        side = hist.get("fold_b" if worse_is_b else "fold_a") or {}
        other = hist.get("fold_a" if worse_is_b else "fold_b") or {}
        tail = _num(side.get("tail_gt5_pct"))
        tail_o = _num(other.get("tail_gt5_pct"))
        if tail is not None:
            err_bullets.append(f"{tail:.0f}% predictions have >₹5 error")
            if tail_o is not None and tail > tail_o + 5:
                err_bullets.append(f"vs {tail_o:.0f}% on {better_label}")

    # Direction collapse
    metric_bullets: list[str] = []
    da = _num(better_metrics.get("directional_accuracy_pct"))
    db = _num(worse_metrics.get("directional_accuracy_pct"))
    if da is not None and db is not None and db < da - 2:
        metric_bullets.append(f"Direction accuracy fell from {da:.0f}% to {db:.0f}%")
    ma = _num(better_metrics.get("mae"))
    mb = _num(worse_metrics.get("mae"))
    if ma is not None and mb is not None and mb > ma * 1.1:
        metric_bullets.append(f"MAE rose from ₹{ma:.2f} to ₹{mb:.2f}")

    why_all: list[str] = []
    # Performance evidence first, then market/feature causes (never bury MAE/Direction).
    for group in (metric_bullets, err_bullets, context_bullets, outlier_bullets, drift_bullets[:3]):
        for b in group:
            if b and b not in why_all:
                why_all.append(b)

    # Prefer non-regime lines; keep regime if room
    core = [b for b in why_all if not b.startswith("Regime:")]
    regime_lines = [b for b in why_all if b.startswith("Regime:")]
    why_primary = core[:10]
    if regime_lines and len(why_primary) < 10:
        why_primary.append(regime_lines[0])

    headline = f"{worse_label} differs from {better_label} mainly because:"
    if not why_primary:
        headline = f"{worse_label} vs {better_label}: no dominant market/feature drivers detected."
        why_primary = ["Inspect distributions and error samples for residual causes."]

    # Per-metric why cards — market/feature causes (not the metric itself restating)
    card_why = [
        b for b in why_primary
        if not b.startswith("Direction accuracy") and not b.startswith("MAE rose")
    ][:6]

    def metric_card(
        key: str,
        title: str,
        better_v: float | None,
        worse_v: float | None,
        *,
        rupee: bool = False,
        pct: bool = False,
        higher_better: bool = False,
    ) -> dict[str, Any]:
        worse_display = "—"
        arrow = ""
        if worse_v is not None:
            if rupee:
                worse_display = f"₹{worse_v:.2f}"
            elif pct:
                worse_display = f"{worse_v:.0f}%"
            else:
                worse_display = f"{worse_v:.3g}"
            if better_v is not None:
                deteriorated = (
                    (worse_v < better_v) if higher_better else (worse_v > better_v)
                )
                if higher_better:
                    arrow = " ↓" if deteriorated else " ↑"
                else:
                    arrow = " ↑" if deteriorated else " ↓"
        why = list(card_why) if (
            better_v is not None and worse_v is not None and (
                (worse_v > better_v * 1.05 and not higher_better)
                or (worse_v < better_v * 0.95 and higher_better)
                or (higher_better and worse_v < better_v - 2)
                or (not higher_better and worse_v > better_v + 0.05)
            )
        ) else []
        return {
            "key": key,
            "title": title,
            "fold_label": worse_label,
            "value_display": f"{worse_display}{arrow}".strip(),
            "better_value": better_v,
            "worse_value": worse_v,
            "why": why,
        }

    mae_card = metric_card(
        "mae", "MAE",
        _num(better_metrics.get("mae")),
        _num(worse_metrics.get("mae")),
        rupee=True,
        higher_better=False,
    )
    dir_card = metric_card(
        "direction", "Direction",
        _num(better_metrics.get("directional_accuracy_pct")),
        _num(worse_metrics.get("directional_accuracy_pct")),
        pct=True,
        higher_better=True,
    )

    unique_about = []
    for row in (worse_outliers.get("rows") or [])[:5]:
        if isinstance(row, dict):
            unique_about.append({
                "feature": row.get("feature"),
                "display": row.get("display"),
                "z_score": row.get("z_score"),
                "emoji": row.get("severity_emoji"),
            })

    return {
        "available": True,
        "worse_label": worse_label,
        "better_label": better_label,
        "headline": headline,
        "bullets": why_primary,
        "metric_cards": [mae_card, dir_card],
        "what_is_unique": {
            "fold_label": worse_label,
            "rows": unique_about,
            "available": bool(unique_about),
        },
        "context_bullets": context_bullets,
        "drift_bullets": drift_bullets,
    }


def _pick_feature_list(df: pd.DataFrame, model_features: list[str] | None) -> list[str]:
    cols: list[str] = []
    for c in _DIST_PRIORITY:
        if c in df.columns and c not in cols:
            cols.append(c)
    for c in (model_features or []):
        if c in df.columns and c not in cols and pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    # Cap to keep UI readable / IO light analysis
    if len(cols) > 80:
        cols = cols[:80]
    return cols


def load_fold_diagnostics_frame(
    data_dir: str,
    dataset: str,
    *,
    model_features: list[str] | None = None,
) -> tuple[pd.DataFrame | None, str | None]:
    """Load the minimal column set needed for fold context + distribution analysis."""
    if not dataset:
        return None, "Dataset name missing"
    try:
        from .dataset_loader import datasets_dir
        from chain_replay_ml.dataset_builder.writer import _safe_filename
        import os
        path = os.path.join(datasets_dir(data_dir), f"{_safe_filename(dataset)}.parquet")
        available = parquet_column_names(path) or set()
    except Exception:
        available = set()

    want = list(_CONTEXT_WANT)
    for f in (model_features or [])[:120]:
        want.append(f)
    if available:
        want = [c for c in want if c in available]
    if "trading_day" not in want and (not available or "trading_day" in available):
        want = ["trading_day", *want]

    try:
        df, _, _ = load_dataset_frame(data_dir, dataset, columns=list(dict.fromkeys(want)))
    except DatasetLoaderError as exc:
        return None, str(exc)
    return df, None


def build_fold_pair_diagnostics(
    data_dir: str,
    *,
    dataset: str,
    fold_a: int,
    fold_b: int,
    fold_def_a: dict[str, Any],
    fold_def_b: dict[str, Any],
    metrics_a: dict[str, Any],
    metrics_b: dict[str, Any],
    label_a: str,
    label_b: str,
    days_a: list[str],
    days_b: list[str],
    error_histograms: dict[str, Any] | None = None,
    model_features: list[str] | None = None,
) -> dict[str, Any]:
    """Full Why-driven diagnostics payload for Fold Comparison."""
    features = list(model_features or [])
    df, err = load_fold_diagnostics_frame(data_dir, dataset, model_features=features)
    if df is None:
        return {
            "available": False,
            "message": err or "Could not load dataset for fold diagnostics.",
            "context_a": {"available": False},
            "context_b": {"available": False},
            "distribution_shift": {"available": False},
            "outliers_a": {"available": False},
            "outliers_b": {"available": False},
            "why": {"available": False, "bullets": []},
        }

    def _window(fold_def: dict[str, Any], which: str) -> pd.DataFrame:
        start, stop, _ = _slice_bounds(fold_def, which)
        if start is None or stop is None or not (0 <= start < stop <= len(df)):
            return df.iloc[0:0]
        return df.iloc[start:stop]

    val_a = _window(fold_def_a, "validation")
    val_b = _window(fold_def_b, "validation")
    train_a = _window(fold_def_a, "train")
    train_b = _window(fold_def_b, "train")

    ctx_a = build_fold_context(
        val_a, fold=fold_a, label=label_a, trading_days=days_a, full_df=df,
    )
    ctx_b = build_fold_context(
        val_b, fold=fold_b, label=label_b, trading_days=days_b, full_df=df,
    )

    feat_list = _pick_feature_list(df, features)
    dist = build_feature_distribution_shift(
        val_a, val_b, feat_list, label_a=label_a, label_b=label_b,
    )
    out_a = build_fold_outlier_scores(
        val_a, train_a if len(train_a) else val_a, feat_list, fold=fold_a, label=label_a,
    )
    out_b = build_fold_outlier_scores(
        val_b, train_b if len(train_b) else val_b, feat_list, fold=fold_b, label=label_b,
    )

    why = build_why_explanations(
        label_a=label_a,
        label_b=label_b,
        metrics_a=metrics_a,
        metrics_b=metrics_b,
        context_a=ctx_a,
        context_b=ctx_b,
        distribution=dist,
        outliers_a=out_a,
        outliers_b=out_b,
        error_histograms=error_histograms,
    )

    return {
        "available": bool(ctx_a.get("available") or ctx_b.get("available")),
        "message": None,
        "context_a": ctx_a,
        "context_b": ctx_b,
        "distribution_shift": dist,
        "outliers_a": out_a,
        "outliers_b": out_b,
        "why": why,
    }

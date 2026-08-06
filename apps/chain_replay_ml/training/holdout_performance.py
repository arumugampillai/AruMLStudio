"""Walk-forward holdout vs WF-region comparison and degradation diagnosis."""

from __future__ import annotations

import csv
import io
from typing import Any

import numpy as np
import pandas as pd

from .config import normalize_training_config
from .dataset_loader import DatasetLoaderError, load_dataset_frame
from .evaluator import (
    PREMIUM_METRIC_BANDS,
    evaluate_regression,
    premium_band_performance,
    resolve_ltp_baseline,
    resolve_ltp_baseline_from_frames,
)
from .model_runtime import load_prediction_model, resolve_production_model_path
from .paths import model_artifact_paths, model_package_dir
from .registry import _selected_feature_names
from .split import WalkForwardSplitError, normalize_walk_forward_config, walk_forward_fold_slices

_VOL_COLUMNS = ("implied_vol", "iv_zscore_1m", "iv_zscore_5m")

_IDENTITY_COLUMNS = (
    "trading_day",
    "timestamp",
    "ltp",
    "ltp_to_spot_ratio",
    "spot",
)


class HoldoutPerformanceError(Exception):
    pass


def _wf_summary_data(doc: dict[str, Any]) -> dict[str, Any]:
    wf = doc.get("walk_forward") if isinstance(doc.get("walk_forward"), dict) else {}
    summary_art = wf.get("summary") or {}
    data = summary_art.get("data") if isinstance(summary_art.get("data"), dict) else {}
    return data


def _holdout_from_block(block: Any) -> tuple[int, int] | None:
    if not isinstance(block, dict):
        return None
    start = block.get("start")
    stop = block.get("stop")
    if start is None or stop is None:
        return None
    try:
        s, e = int(start), int(stop)
    except (TypeError, ValueError):
        return None
    if e <= s:
        return None
    return s, e


def resolve_holdout_slice(doc: dict[str, Any], n_rows: int) -> tuple[int, int]:
    """Return (start, stop) row indices for the untouched holdout region."""
    if n_rows <= 0:
        raise HoldoutPerformanceError("Dataset has no rows")

    for source in (
        _wf_summary_data(doc).get("test_holdout"),
        (
            ((doc.get("walk_forward") or {}).get("champion_aggregate") or {}).get("data") or {}
        ).get("test_holdout"),
    ):
        parsed = _holdout_from_block(source)
        if parsed is not None:
            start, stop = parsed
            if 0 <= start < stop <= n_rows:
                return start, stop

    config = doc.get("config") if isinstance(doc.get("config"), dict) else {}
    split_cfg = dict(config.get("split") or config.get("split_info") or {})
    if not split_cfg.get("walk_forward"):
        wf_in = config.get("walk_forward")
        if isinstance(wf_in, dict):
            split_cfg["walk_forward"] = wf_in
    try:
        wf_cfg = normalize_walk_forward_config(split_cfg, n_rows)
        _, test_sl = walk_forward_fold_slices(n_rows, wf_cfg)
    except (WalkForwardSplitError, ValueError, KeyError) as exc:
        raise HoldoutPerformanceError(f"Could not resolve holdout slice: {exc}") from exc
    return test_sl.start, test_sl.stop


def resolve_holdout_slice_with_fallback(
    doc: dict[str, Any],
    n_rows: int,
    training_cfg: Any,
) -> tuple[int, int]:
    """Holdout [start, stop) using package metadata, else WF/test split, else last 15%."""
    try:
        start, stop = resolve_holdout_slice(doc, n_rows)
        return int(start), int(stop)
    except (HoldoutPerformanceError, AttributeError, TypeError):
        pass
    split = getattr(training_cfg, "split", None) or {}
    if not isinstance(split, dict):
        split = {}
    if str(split.get("strategy") or "").lower() in ("walk_forward", "rolling_window"):
        try:
            from chain_replay_ml.training.split import (
                WalkForwardSplitError,
                normalize_walk_forward_config,
                walk_forward_fold_slices,
            )

            wf_cfg = normalize_walk_forward_config(split, n_rows)
            _, test_sl = walk_forward_fold_slices(n_rows, wf_cfg)
            return int(test_sl.start), int(test_sl.stop)
        except (WalkForwardSplitError, ValueError, KeyError):
            return int(n_rows * 0.85), int(n_rows)
    try:
        test_pct = float(split.get("test", 15))
    except (TypeError, ValueError):
        test_pct = 15.0
    return int(n_rows * (1.0 - test_pct / 100.0)), int(n_rows)


def distribution_summary(values: pd.Series | np.ndarray) -> dict[str, float | None]:
    try:
        from chain_replay_ml.frame_backend.studio_stats import (
            distribution_summary_via_polars,
        )

        return distribution_summary_via_polars(values)
    except Exception:
        pass
    arr = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return {"mean": None, "std": None, "p25": None, "p50": None, "p75": None, "min": None, "max": None, "count": 0}
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "p25": float(np.percentile(arr, 25)),
        "p50": float(np.percentile(arr, 50)),
        "p75": float(np.percentile(arr, 75)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "count": int(len(arr)),
    }


def normalized_mean_shift(wf_mean: float | None, holdout_mean: float | None, pooled_std: float | None) -> float | None:
    if wf_mean is None or holdout_mean is None:
        return None
    denom = pooled_std if pooled_std and pooled_std > 1e-9 else max(abs(wf_mean), abs(holdout_mean), 1e-9)
    return float((holdout_mean - wf_mean) / denom)


def premium_band_pct(baseline: pd.Series | np.ndarray) -> dict[str, float]:
    base = pd.to_numeric(pd.Series(baseline), errors="coerce").to_numpy(dtype=float)
    finite = base[np.isfinite(base)]
    total = len(finite)
    if total <= 0:
        return {label: 0.0 for label, _, _ in PREMIUM_METRIC_BANDS}
    out: dict[str, float] = {}
    for label, lo, hi in PREMIUM_METRIC_BANDS:
        if hi is None:
            mask = finite >= lo
        else:
            mask = (finite >= lo) & (finite < hi)
        out[label] = float(mask.sum() / total * 100.0)
    return out


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def pct_change(wf_val: float | None, holdout_val: float | None) -> float | None:
    if wf_val is None or holdout_val is None:
        return None
    try:
        wf_f, ho_f = float(wf_val), float(holdout_val)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(wf_f) or not np.isfinite(ho_f):
        return None
    denom = max(abs(wf_f), 1e-9)
    return float((ho_f - wf_f) / denom * 100.0)


def feature_drift_score(wf_series: pd.Series, holdout_series: pd.Series) -> float:
    """Return 0–1 drift score for a single feature (higher = more drift)."""
    wf_stats = distribution_summary(wf_series)
    ho_stats = distribution_summary(holdout_series)
    if wf_stats.get("count", 0) == 0 or ho_stats.get("count", 0) == 0:
        return 0.0
    pooled = None
    if wf_stats.get("std") is not None and ho_stats.get("std") is not None:
        pooled = float(np.sqrt((float(wf_stats["std"]) ** 2 + float(ho_stats["std"]) ** 2) / 2.0))
    mean_shift = normalized_mean_shift(
        float(wf_stats["mean"]) if wf_stats.get("mean") is not None else None,
        float(ho_stats["mean"]) if ho_stats.get("mean") is not None else None,
        pooled,
    )
    parts: list[float] = []
    if mean_shift is not None:
        parts.append(_clamp01(abs(mean_shift) / 1.5))
    wf_std = float(wf_stats["std"]) if wf_stats.get("std") is not None else None
    ho_std = float(ho_stats["std"]) if ho_stats.get("std") is not None else None
    if wf_std is not None and ho_std is not None and wf_std > 1e-9:
        parts.append(_clamp01(abs(ho_std / wf_std - 1.0) / 0.5))
    if not parts:
        return 0.0
    return float(np.mean(parts))


_WASSERSTEIN_EPS = 1e-9
# Cap for mapping wasserstein_normalized → [0, 1] in composite risk (2σ transport → 1).
_WASSERSTEIN_NORM_SCALE = 2.0
# Cap for mapping |Δ null%| → [0, 1] (20 percentage points → 1).
_NULL_DRIFT_PP_SCALE = 20.0
# Cap for mapping unit-sum importance → [0, 1] (10% share → 1).
_IMPORTANCE_SHARE_SCALE = 0.10


def _finite_numeric(series: pd.Series | np.ndarray) -> np.ndarray:
    arr = pd.to_numeric(pd.Series(series), errors="coerce").to_numpy(dtype=float)
    return arr[np.isfinite(arr)]


def series_null_pct(series: pd.Series | np.ndarray) -> float:
    """Percent of non-finite / null values (0–100)."""
    raw = pd.to_numeric(pd.Series(series), errors="coerce").to_numpy(dtype=float)
    n = int(len(raw))
    if n <= 0:
        return 0.0
    n_finite = int(np.isfinite(raw).sum())
    return float(100.0 * (n - n_finite) / n)


def ks_drift_metrics(
    wf_series: pd.Series | np.ndarray,
    holdout_series: pd.Series | np.ndarray,
) -> tuple[float | None, float | None]:
    """Two-sample KS statistic and p-value (shape / CDF shift)."""
    wf = _finite_numeric(wf_series)
    ho = _finite_numeric(holdout_series)
    if len(wf) < 2 or len(ho) < 2:
        return None, None
    try:
        from scipy.stats import ks_2samp
    except ImportError:
        return None, None
    try:
        res = ks_2samp(wf, ho, method="auto")
        return float(res.statistic), float(res.pvalue)
    except Exception:
        return None, None


def wasserstein_drift_metrics(
    wf_series: pd.Series | np.ndarray,
    holdout_series: pd.Series | np.ndarray,
    *,
    wf_std: float | None = None,
) -> tuple[float | None, float | None]:
    """Raw Wasserstein-1 distance and scale-free normalized distance.

    wasserstein_normalized = W / max(wf_std, ε) so risk is comparable across features.
    """
    wf = _finite_numeric(wf_series)
    ho = _finite_numeric(holdout_series)
    if len(wf) < 1 or len(ho) < 1:
        return None, None
    try:
        from scipy.stats import wasserstein_distance
    except ImportError:
        return None, None
    try:
        raw = float(wasserstein_distance(wf, ho))
    except Exception:
        return None, None
    if wf_std is None:
        wf_std = float(np.std(wf)) if len(wf) else None
    denom = max(float(wf_std) if wf_std is not None else 0.0, _WASSERSTEIN_EPS)
    return raw, float(raw / denom)


def composite_drift_risk_components(
    *,
    mean_drift: float,
    ks_statistic: float | None,
    wasserstein_normalized: float | None,
    null_drift_pp: float | None,
    importance: float,
) -> dict[str, Any]:
    """Phase 5.2 clamp01 terms, risk score, and % shares of the composite.

    Equal-weight mean of five [0, 1] components, then ×100:

      mean_drift_n = clamp01(mean_drift)                         # existing 0–1 mean+std score
      ks_n         = clamp01(ks_statistic)                       # KS ∈ [0, 1]
      w_n          = clamp01(wasserstein_normalized / 2.0)       # 2σ transport → 1
      null_n       = clamp01(|Δnull%| / 20.0)                    # 20pp null shift → 1
      imp_n        = clamp01(importance / 0.10)                  # 10% unit-sum share → 1

    ``shares_pct`` is each term's fraction of the sum of the five clamp01
    values (sums to ~100% when any term is non-zero). Uses
    ``wasserstein_normalized`` (not raw W) so scales stay comparable.
    """
    terms = {
        "mean_drift": _clamp01(float(mean_drift or 0.0)),
        "ks": _clamp01(float(ks_statistic) if ks_statistic is not None else 0.0),
        "wasserstein_normalized": _clamp01(
            (float(wasserstein_normalized) if wasserstein_normalized is not None else 0.0)
            / _WASSERSTEIN_NORM_SCALE
        ),
        "null_drift": _clamp01(
            (abs(float(null_drift_pp)) if null_drift_pp is not None else 0.0)
            / _NULL_DRIFT_PP_SCALE
        ),
        "importance": _clamp01(float(importance or 0.0) / _IMPORTANCE_SHARE_SCALE),
    }
    values = list(terms.values())
    risk_score = float(round(100.0 * float(np.mean(values)), 2))
    total = float(sum(values))
    if total > 0.0:
        shares_pct = {k: float(round(100.0 * v / total, 1)) for k, v in terms.items()}
    else:
        shares_pct = {k: 0.0 for k in terms}
    return {
        "terms": terms,
        "shares_pct": shares_pct,
        "risk_score": risk_score,
    }


def composite_drift_risk_score(
    *,
    mean_drift: float,
    ks_statistic: float | None,
    wasserstein_normalized: float | None,
    null_drift_pp: float | None,
    importance: float,
) -> float:
    """Phase 5.2 composite risk on 0–100 (see ``composite_drift_risk_components``)."""
    return float(
        composite_drift_risk_components(
            mean_drift=mean_drift,
            ks_statistic=ks_statistic,
            wasserstein_normalized=wasserstein_normalized,
            null_drift_pp=null_drift_pp,
            importance=importance,
        )["risk_score"]
    )

def feature_drift_risk(drift: float, importance: float) -> str:
    """Return high | medium | low based on drift magnitude and model importance.

    Legacy label helper (mean/std drift × importance). Prefer
    ``composite_drift_risk_label`` for Phase 5.2 risk_score (0–100).
    """
    if drift >= 0.50 and importance >= 0.10:
        return "high"
    if drift >= 0.50 and importance >= 0.03:
        return "medium"
    if drift * importance >= 0.05:
        return "medium"
    return "low"


def composite_drift_risk_label(risk_score: float) -> str:
    """Map Phase 5.2 composite risk_score (0–100) to high | medium | low."""
    if risk_score >= 55.0:
        return "high"
    if risk_score >= 30.0:
        return "medium"
    return "low"


_RISK_LABELS = {
    "high": "🔴 High",
    "medium": "🟡 Medium",
    "low": "🟢 Low",
}


def _importance_map_from_doc(doc: dict[str, Any]) -> dict[str, float]:
    imp = doc.get("feature_importance") if isinstance(doc.get("feature_importance"), list) else []
    raw: dict[str, float] = {}
    for row in imp:
        if not isinstance(row, dict):
            continue
        feat = str(row.get("feature") or "").strip()
        if not feat:
            continue
        try:
            val = float(row.get("importance_pct") or row.get("importance") or 0)
        except (TypeError, ValueError):
            val = 0.0
        raw[feat] = val
    total = sum(raw.values()) or 1.0
    return {k: v / total for k, v in raw.items()}


def build_feature_drift_ranking(
    wf_df: pd.DataFrame,
    ho_df: pd.DataFrame,
    features: list[str],
    *,
    importance_map: dict[str, float] | None = None,
    limit: int = 20,
) -> list[dict[str, Any]]:
    """Per-feature WF vs holdout drift rows (JSON v2 / Phase 5.2).

    Includes mean/std drift, KS, Wasserstein (raw + normalized), null drift,
    and composite ``risk_score`` on 0–100 (see ``composite_drift_risk_score``).
    """
    importance_map = importance_map or {}
    rows: list[dict[str, Any]] = []
    for feat in features:
        if feat not in wf_df.columns or feat not in ho_df.columns:
            continue
        wf_col = wf_df[feat]
        ho_col = ho_df[feat]
        wf_stats = distribution_summary(wf_col)
        ho_stats = distribution_summary(ho_col)
        wf_mean = wf_stats.get("mean")
        ho_mean = ho_stats.get("mean")
        wf_std = wf_stats.get("std")
        ho_std = ho_stats.get("std")
        drift = feature_drift_score(wf_col, ho_col)
        drift_pct = pct_change(
            float(wf_mean) if wf_mean is not None else None,
            float(ho_mean) if ho_mean is not None else None,
        )
        importance = float(importance_map.get(feat) or 0.0)
        ks_stat, ks_p = ks_drift_metrics(wf_col, ho_col)
        w_raw, w_norm = wasserstein_drift_metrics(
            wf_col,
            ho_col,
            wf_std=float(wf_std) if wf_std is not None else None,
        )
        null_wf = series_null_pct(wf_col)
        null_ho = series_null_pct(ho_col)
        null_drift_pp = float(null_ho - null_wf)
        risk_score = composite_drift_risk_score(
            mean_drift=drift,
            ks_statistic=ks_stat,
            wasserstein_normalized=w_norm,
            null_drift_pp=null_drift_pp,
            importance=importance,
        )
        risk = composite_drift_risk_label(risk_score)
        rows.append({
            "feature": feat,
            "wf_mean": wf_mean,
            "holdout_mean": ho_mean,
            "wf_std": wf_std,
            "holdout_std": ho_std,
            "drift_pct": round(drift_pct, 1) if drift_pct is not None else None,
            "drift": round(drift, 4),
            "ks_statistic": round(ks_stat, 6) if ks_stat is not None else None,
            "ks_pvalue": round(ks_p, 6) if ks_p is not None else None,
            "wasserstein_distance": round(w_raw, 6) if w_raw is not None else None,
            "wasserstein_normalized": round(w_norm, 6) if w_norm is not None else None,
            "null_pct_wf": round(null_wf, 4),
            "null_pct_ho": round(null_ho, 4),
            "null_drift_pp": round(null_drift_pp, 4),
            "importance": round(importance, 4),
            "risk": risk,
            "risk_label": _RISK_LABELS.get(risk, risk),
            # Phase 5.2: composite 0–100 (not legacy drift×importance).
            "risk_score": risk_score,
        })
    rows.sort(key=lambda r: (float(r.get("risk_score") or 0), float(r.get("drift") or 0)), reverse=True)
    return rows[:limit]


def compute_drift_scores(
    *,
    target_wf: pd.Series,
    target_holdout: pd.Series,
    baseline_wf: pd.Series | None,
    baseline_holdout: pd.Series | None,
    vol_wf: pd.Series | None,
    vol_holdout: pd.Series | None,
    feature_ranking: list[dict[str, Any]],
) -> dict[str, float]:
    wf_tgt = distribution_summary(target_wf)
    ho_tgt = distribution_summary(target_holdout)
    pooled_tgt = None
    if wf_tgt.get("std") is not None and ho_tgt.get("std") is not None:
        pooled_tgt = float(np.sqrt((float(wf_tgt["std"]) ** 2 + float(ho_tgt["std"]) ** 2) / 2.0))

    target_parts: list[float] = []
    for wf_v, ho_v in (
        (wf_tgt.get("mean"), ho_tgt.get("mean")),
        (wf_tgt.get("p50"), ho_tgt.get("p50")),
    ):
        shift = normalized_mean_shift(
            float(wf_v) if wf_v is not None else None,
            float(ho_v) if ho_v is not None else None,
            pooled_tgt,
        )
        if shift is not None:
            target_parts.append(_clamp01(abs(shift) / 1.5))
    if wf_tgt.get("std") is not None and ho_tgt.get("std") is not None and float(wf_tgt["std"]) > 1e-9:
        target_parts.append(_clamp01(abs(float(ho_tgt["std"]) / float(wf_tgt["std"]) - 1.0) / 0.5))
    target_drift = round(100.0 * float(np.mean(target_parts)) if target_parts else 0.0, 1)

    premium_drift = 0.0
    if baseline_wf is not None and baseline_holdout is not None:
        wf_bands = premium_band_pct(baseline_wf)
        ho_bands = premium_band_pct(baseline_holdout)
        total_shift = sum(abs(ho_bands.get(label, 0.0) - wf_bands.get(label, 0.0)) for label, _, _ in PREMIUM_METRIC_BANDS)
        premium_drift = round(_clamp01(total_shift / 100.0) * 100.0, 1)

    volatility_drift = 0.0
    if vol_wf is not None and vol_holdout is not None:
        wf_vol = distribution_summary(vol_wf)
        ho_vol = distribution_summary(vol_holdout)
        pooled_vol = None
        if wf_vol.get("std") is not None and ho_vol.get("std") is not None:
            pooled_vol = float(np.sqrt((float(wf_vol["std"]) ** 2 + float(ho_vol["std"]) ** 2) / 2.0))
        vol_parts: list[float] = []
        shift = normalized_mean_shift(
            float(wf_vol["mean"]) if wf_vol.get("mean") is not None else None,
            float(ho_vol["mean"]) if ho_vol.get("mean") is not None else None,
            pooled_vol,
        )
        if shift is not None:
            vol_parts.append(_clamp01(abs(shift) / 1.5))
        if wf_vol.get("std") is not None and ho_vol.get("std") is not None and float(wf_vol["std"]) > 1e-9:
            vol_parts.append(_clamp01(abs(float(ho_vol["std"]) / float(wf_vol["std"]) - 1.0) / 0.5))
        volatility_drift = round(100.0 * float(np.mean(vol_parts)) if vol_parts else 0.0, 1)

    feat_drifts = [float(r.get("drift") or 0) for r in feature_ranking if r.get("drift") is not None]
    if feat_drifts:
        top_n = feat_drifts[: min(10, len(feat_drifts))]
        feature_drift = round(100.0 * float(np.mean(top_n)), 1)
    else:
        feature_drift = 0.0

    return {
        "target": target_drift,
        "feature": feature_drift,
        "premium": premium_drift,
        "volatility": volatility_drift,
    }


def compute_similarity_score(drift_scores: dict[str, float]) -> float:
    weights = {"target": 0.30, "feature": 0.35, "premium": 0.20, "volatility": 0.15}
    overall_drift = sum(float(drift_scores.get(k) or 0) * w for k, w in weights.items())
    return round(max(0.0, min(100.0, 100.0 - overall_drift)), 1)


def _comparison_row(
    *,
    category: str,
    metric: str,
    wf_val: float | None,
    holdout_val: float | None,
    shift: float | None = None,
) -> dict[str, Any]:
    return {
        "category": category,
        "metric": metric,
        "wf": wf_val,
        "holdout": holdout_val,
        "shift": shift,
    }


def build_region_comparison_rows(
    *,
    target_wf: pd.Series,
    target_holdout: pd.Series,
    baseline_wf: pd.Series | None,
    baseline_holdout: pd.Series | None,
    vol_wf: pd.Series | None,
    vol_holdout: pd.Series | None,
    trading_day_wf: pd.Series | None,
    trading_day_holdout: pd.Series | None,
    feature_frames: list[tuple[str, pd.Series, pd.Series]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    wf_tgt = distribution_summary(target_wf)
    ho_tgt = distribution_summary(target_holdout)
    pooled_tgt_std = None
    if wf_tgt.get("std") is not None and ho_tgt.get("std") is not None:
        pooled_tgt_std = float(np.sqrt((float(wf_tgt["std"]) ** 2 + float(ho_tgt["std"]) ** 2) / 2.0))

    for key, label in (
        ("mean", "Mean"),
        ("std", "Std"),
        ("p25", "P25"),
        ("p50", "Median"),
        ("p75", "P75"),
        ("min", "Min"),
        ("max", "Max"),
    ):
        wf_v = wf_tgt.get(key)
        ho_v = ho_tgt.get(key)
        shift = normalized_mean_shift(
            float(wf_v) if wf_v is not None else None,
            float(ho_v) if ho_v is not None else None,
            pooled_tgt_std if key == "mean" else float(wf_tgt.get("std") or ho_tgt.get("std") or 1.0),
        ) if key == "mean" else (
            (float(ho_v) - float(wf_v)) if wf_v is not None and ho_v is not None else None
        )
        rows.append(_comparison_row(category="Target distribution", metric=label, wf_val=wf_v, holdout_val=ho_v, shift=shift))

    if baseline_wf is not None and baseline_holdout is not None:
        wf_bands = premium_band_pct(baseline_wf)
        ho_bands = premium_band_pct(baseline_holdout)
        for label, _, _ in PREMIUM_METRIC_BANDS:
            wf_pct = wf_bands.get(label, 0.0)
            ho_pct = ho_bands.get(label, 0.0)
            rows.append(
                _comparison_row(
                    category="Premium bands",
                    metric=f"₹{label} %",
                    wf_val=round(wf_pct, 4),
                    holdout_val=round(ho_pct, 4),
                    shift=round(ho_pct - wf_pct, 4),
                )
            )

    if vol_wf is not None and vol_holdout is not None:
        wf_vol = distribution_summary(vol_wf)
        ho_vol = distribution_summary(vol_holdout)
        pooled_vol = None
        if wf_vol.get("std") is not None and ho_vol.get("std") is not None:
            pooled_vol = float(np.sqrt((float(wf_vol["std"]) ** 2 + float(ho_vol["std"]) ** 2) / 2.0))
        for key, label in (("mean", "Mean"), ("std", "Std")):
            wf_v = wf_vol.get(key)
            ho_v = ho_vol.get(key)
            shift = normalized_mean_shift(
                float(wf_v) if wf_v is not None else None,
                float(ho_v) if ho_v is not None else None,
                pooled_vol,
            ) if key == "mean" else (
                (float(ho_v) - float(wf_v)) if wf_v is not None and ho_v is not None else None
            )
            rows.append(_comparison_row(category="Volatility", metric=label, wf_val=wf_v, holdout_val=ho_v, shift=shift))

    if trading_day_wf is not None and trading_day_holdout is not None:
        wf_days = trading_day_wf.dropna().astype(str).nunique()
        ho_days = trading_day_holdout.dropna().astype(str).nunique()
        wf_rpd = len(trading_day_wf) / max(wf_days, 1)
        ho_rpd = len(trading_day_holdout) / max(ho_days, 1)
        rows.append(_comparison_row(category="Trading days", metric="Unique days", wf_val=float(wf_days), holdout_val=float(ho_days), shift=float(ho_days - wf_days)))
        rows.append(_comparison_row(category="Trading days", metric="Rows / day", wf_val=round(wf_rpd, 4), holdout_val=round(ho_rpd, 4), shift=round(ho_rpd - wf_rpd, 4)))

    for feat, wf_series, ho_series in feature_frames:
        wf_stats = distribution_summary(wf_series)
        ho_stats = distribution_summary(ho_series)
        pooled = None
        if wf_stats.get("std") is not None and ho_stats.get("std") is not None:
            pooled = float(np.sqrt((float(wf_stats["std"]) ** 2 + float(ho_stats["std"]) ** 2) / 2.0))
        shift = normalized_mean_shift(
            float(wf_stats["mean"]) if wf_stats.get("mean") is not None else None,
            float(ho_stats["mean"]) if ho_stats.get("mean") is not None else None,
            pooled,
        )
        rows.append(
            _comparison_row(
                category="Feature distributions",
                metric=feat,
                wf_val=wf_stats.get("mean"),
                holdout_val=ho_stats.get("mean"),
                shift=shift,
            )
        )

    return rows


def _distribution_shift_score(comparison_rows: list[dict[str, Any]]) -> float:
    shifts: list[float] = []
    for row in comparison_rows:
        if row.get("category") in ("Target distribution", "Premium bands", "Volatility", "Feature distributions"):
            shift = row.get("shift")
            if shift is None:
                continue
            try:
                val = abs(float(shift))
            except (TypeError, ValueError):
                continue
            if val != val:
                continue
            shifts.append(val)
    if not shifts:
        return 0.0
    return float(np.mean(shifts))


_CAUSE_LABELS = {
    "overfitting": "Overfitting",
    "data_drift": "Data drift",
    "difficult_market": "Difficult market conditions",
    "stable": "Stable / minor shift",
    "unknown": "Unclear",
}


_LIKELY_REASONS = {
    "overfitting": "Model may have memorized walk-forward validation patterns rather than generalizing.",
    "data_drift": "High probability of market regime change between training and holdout periods.",
    "difficult_market": "Holdout period shows harder-to-predict conditions without major input distribution shift.",
    "stable": "Holdout distribution and errors are aligned with walk-forward training.",
    "unknown": "Insufficient signal to classify degradation cause confidently.",
}


def _build_evidence(
    *,
    drift_scores: dict[str, float],
    feature_ranking: list[dict[str, Any]],
    wf_tgt_stats: dict[str, float | None],
    ho_tgt_stats: dict[str, float | None],
    wf_vol_stats: dict[str, float | None] | None,
    ho_vol_stats: dict[str, float | None] | None,
    premium_pct_change: float | None,
    holdout_unique_days: int | None,
    wf_unique_days: int | None,
    error_ratio: float | None,
) -> list[str]:
    evidence: list[str] = []

    if premium_pct_change is not None and abs(premium_pct_change) >= 1.0:
        evidence.append(f"Premium distribution changed {abs(premium_pct_change):.0f}%")
    elif drift_scores.get("premium", 0) >= 5:
        evidence.append(f"Premium distribution drift score {drift_scores['premium']:.0f}%")

    if wf_vol_stats and ho_vol_stats:
        vol_chg = pct_change(wf_vol_stats.get("std"), ho_vol_stats.get("std"))
        if vol_chg is not None and abs(vol_chg) >= 1.0:
            direction = "increased" if vol_chg > 0 else "decreased"
            evidence.append(f"Volatility std {direction} {abs(vol_chg):.0f}%")
        elif drift_scores.get("volatility", 0) >= 10:
            evidence.append(f"Volatility drift score {drift_scores['volatility']:.0f}%")

    median_chg = pct_change(wf_tgt_stats.get("p50"), ho_tgt_stats.get("p50"))
    if median_chg is not None and abs(median_chg) >= 1.0:
        evidence.append(f"Target median changed {abs(median_chg):.0f}%")
    elif drift_scores.get("target", 0) >= 10:
        evidence.append(f"Target drift score {drift_scores['target']:.0f}%")

    top_feats = [
        str(r.get("feature"))
        for r in sorted(
            feature_ranking,
            key=lambda r: float(r.get("risk_score") or r.get("drift") or 0),
            reverse=True,
        )[:5]
        if r.get("feature") and float(r.get("drift") or 0) >= 0.15
    ]
    if top_feats:
        evidence.append("Feature drift: " + ", ".join(top_feats))

    if holdout_unique_days is not None and holdout_unique_days <= 5:
        day_note = f"Holdout only has {holdout_unique_days} trading day{'s' if holdout_unique_days != 1 else ''}"
        if wf_unique_days is not None and wf_unique_days > holdout_unique_days * 3:
            day_note += f" (WF region has {wf_unique_days})"
        evidence.append(day_note)

    if error_ratio is not None and error_ratio >= 1.15:
        evidence.append(f"Holdout MAE is {error_ratio:.0%} of WF validation MAE")

    if drift_scores.get("feature", 0) >= 20 and not top_feats:
        evidence.append(f"Feature drift score {drift_scores['feature']:.0f}%")

    return evidence


def _diagnosis_confidence(
    primary: str,
    *,
    drift_scores: dict[str, float],
    error_ratio: float | None,
    evidence: list[str],
) -> float:
    overall_drift = float(np.mean([float(drift_scores.get(k) or 0) for k in ("target", "feature", "premium", "volatility")]))
    base = 55.0 + min(25.0, len(evidence) * 6.0)
    if primary == "data_drift":
        base += min(20.0, overall_drift * 0.25)
    elif primary == "overfitting" and error_ratio is not None:
        base += min(25.0, (error_ratio - 1.0) * 40.0)
        base += max(0.0, 15.0 - overall_drift * 0.3)
    elif primary == "difficult_market":
        base += min(15.0, float(drift_scores.get("target") or 0) * 0.15)
    elif primary == "stable":
        base += max(0.0, 20.0 - overall_drift * 0.4)
    return round(max(40.0, min(99.0, base)), 0)


def diagnose_degradation(
    *,
    wf_validation_mae: float | None,
    holdout_mae: float | None,
    comparison_rows: list[dict[str, Any]],
    wf_target_std: float | None,
    holdout_target_std: float | None,
    drift_scores: dict[str, float] | None = None,
    feature_ranking: list[dict[str, Any]] | None = None,
    wf_tgt_stats: dict[str, float | None] | None = None,
    ho_tgt_stats: dict[str, float | None] | None = None,
    wf_vol_stats: dict[str, float | None] | None = None,
    ho_vol_stats: dict[str, float | None] | None = None,
    premium_pct_change: float | None = None,
    holdout_unique_days: int | None = None,
    wf_unique_days: int | None = None,
    similarity_pct: float | None = None,
) -> dict[str, Any]:
    """Heuristic primary-cause diagnosis for holdout degradation."""
    drift_scores = drift_scores or {}
    feature_ranking = feature_ranking or []
    shift_score = _distribution_shift_score(comparison_rows)

    error_ratio = None
    if wf_validation_mae is not None and holdout_mae is not None and wf_validation_mae > 1e-9:
        error_ratio = float(holdout_mae / wf_validation_mae)

    target_vol_ratio = None
    if wf_target_std is not None and holdout_target_std is not None and wf_target_std > 1e-9:
        target_vol_ratio = float(holdout_target_std / wf_target_std)

    primary = "unknown"
    if error_ratio is not None and error_ratio >= 1.25 and shift_score < 0.15:
        primary = "overfitting"
    elif shift_score >= 0.25 or float(drift_scores.get("feature") or 0) >= 35 or float(drift_scores.get("target") or 0) >= 30:
        primary = "data_drift"
    elif target_vol_ratio is not None and target_vol_ratio >= 1.2 and shift_score < 0.25:
        primary = "difficult_market"
    elif error_ratio is not None and error_ratio < 1.1 and shift_score < 0.15:
        primary = "stable"
    elif float(drift_scores.get("target") or 0) >= 20 or float(drift_scores.get("premium") or 0) >= 20:
        primary = "data_drift"

    evidence = _build_evidence(
        drift_scores=drift_scores,
        feature_ranking=feature_ranking,
        wf_tgt_stats=wf_tgt_stats or {},
        ho_tgt_stats=ho_tgt_stats or {},
        wf_vol_stats=wf_vol_stats,
        ho_vol_stats=ho_vol_stats,
        premium_pct_change=premium_pct_change,
        holdout_unique_days=holdout_unique_days,
        wf_unique_days=wf_unique_days,
        error_ratio=error_ratio,
    )
    confidence = _diagnosis_confidence(primary, drift_scores=drift_scores, error_ratio=error_ratio, evidence=evidence)

    return {
        "primary_cause": primary,
        "label": _CAUSE_LABELS.get(primary, primary),
        "confidence_pct": confidence,
        "evidence": evidence,
        "likely_reason": _LIKELY_REASONS.get(primary, _LIKELY_REASONS["unknown"]),
        "signals": evidence,
        "error_ratio": error_ratio,
        "distribution_shift_score": round(shift_score, 4),
        "target_vol_ratio": target_vol_ratio,
        "drift_scores": drift_scores,
        "similarity_pct": similarity_pct,
    }


def holdout_performance_by_trading_day(
    y_true: pd.Series,
    y_pred: np.ndarray,
    trading_days: pd.Series,
    *,
    baseline: pd.Series | np.ndarray | None = None,
) -> list[dict[str, Any]]:
    days = trading_days.reset_index(drop=True)
    yt = pd.to_numeric(y_true, errors="coerce").reset_index(drop=True)
    yp = np.asarray(y_pred, dtype=float)
    base = baseline.reset_index(drop=True) if baseline is not None else None
    rows: list[dict[str, Any]] = []
    for day in sorted(days.dropna().astype(str).unique()):
        mask = days.astype(str) == day
        if not mask.any():
            continue
        idx = mask.to_numpy()
        b_slice = base.iloc[idx] if base is not None else None
        metrics = evaluate_regression(yt.iloc[idx], yp[idx], baseline=b_slice)
        rows.append({
            "trading_day": day,
            "rows": int(mask.sum()),
            "mae": metrics.get("mae"),
            "rmse": metrics.get("rmse"),
            "premium_mae_pct": metrics.get("premium_mae_pct"),
            "directional_accuracy_pct": metrics.get("directional_accuracy_pct"),
        })
    return rows


def _model_feature_names(data_dir: str, model_name: str, doc: dict[str, Any]) -> list[str]:
    paths = model_artifact_paths(data_dir, model_name)
    names = _selected_feature_names(data_dir, model_name, paths)
    if names:
        return names
    cfg = doc.get("config") if isinstance(doc.get("config"), dict) else {}
    return [str(f) for f in (cfg.get("selected_features") or cfg.get("features") or []) if f]


def _top_features_for_comparison(doc: dict[str, Any], available: list[str], *, limit: int = 8) -> list[str]:
    imp = doc.get("feature_importance") if isinstance(doc.get("feature_importance"), list) else []
    from_imp = [str(r.get("feature")) for r in imp if isinstance(r, dict) and r.get("feature")]
    ordered = [f for f in from_imp if f in available]
    if len(ordered) < limit:
        for f in available:
            if f not in ordered:
                ordered.append(f)
            if len(ordered) >= limit:
                break
    return ordered[:limit]


def _first_vol_column(columns: list[str]) -> str | None:
    for col in _VOL_COLUMNS:
        if col in columns:
            return col
    return None


_METRIC_KEYS = ("mae", "rmse", "premium_mae_pct", "premium_rmse_pct", "directional_accuracy_pct")


def extract_saved_prediction_metrics(doc: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Authoritative training-time metrics — matches Validation Metrics tab."""
    metrics = doc.get("metrics") if isinstance(doc.get("metrics"), dict) else {}
    prod = doc.get("production_metrics") if isinstance(doc.get("production_metrics"), dict) else {}
    val = metrics.get("validation") if isinstance(metrics.get("validation"), dict) else {}
    test = metrics.get("test") if isinstance(metrics.get("test"), dict) else {}

    def _pick(*sources: dict[str, Any]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for key in _METRIC_KEYS:
            for src in sources:
                if src.get(key) is not None:
                    out[key] = src[key]
                    break
        return out

    return {
        "production_wf": _pick(prod, val),
        "holdout_test": _pick(test),
    }


def _relative_pct_change(baseline: float | None, current: float | None) -> float | None:
    if baseline is None or current is None:
        return None
    try:
        base = float(baseline)
        cur = float(current)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(base) or not np.isfinite(cur) or abs(base) < 1e-12:
        return None
    return float((cur - base) / abs(base) * 100.0)


def build_prediction_error_change_row(
    production_wf: dict[str, Any],
    holdout_test: dict[str, Any],
) -> dict[str, Any]:
    """Percent change for error metrics; point change for direction accuracy."""
    mae_chg = _relative_pct_change(production_wf.get("mae"), holdout_test.get("mae"))
    rmse_chg = _relative_pct_change(production_wf.get("rmse"), holdout_test.get("rmse"))
    pmae_chg = _relative_pct_change(production_wf.get("premium_mae_pct"), holdout_test.get("premium_mae_pct"))
    prmse_chg = _relative_pct_change(production_wf.get("premium_rmse_pct"), holdout_test.get("premium_rmse_pct"))
    dir_delta = None
    wf_dir = production_wf.get("directional_accuracy_pct")
    ho_dir = holdout_test.get("directional_accuracy_pct")
    if wf_dir is not None and ho_dir is not None:
        try:
            dir_delta = float(ho_dir) - float(wf_dir)
        except (TypeError, ValueError):
            dir_delta = None
    return {
        "label": "Change",
        "mae_pct_change": mae_chg,
        "rmse_pct_change": rmse_chg,
        "premium_mae_pct_change": pmae_chg,
        "premium_rmse_pct_change": prmse_chg,
        "direction_pts_change": dir_delta,
    }


def _day_range(series: pd.Series | None) -> tuple[str | None, str | None]:
    if series is None or len(series) == 0:
        return None, None
    vals = series.dropna().astype(str)
    if vals.empty:
        return None, None
    sorted_vals = sorted(vals.unique())
    return sorted_vals[0], sorted_vals[-1]


_TOP1_ANALYSIS_COLUMNS = (
    "theta",
    "delta",
    "vega",
    "abs_delta",
    "minutes_to_expiry",
    "days_to_expiry",
    "strike_distance_from_atm",
    "moneyness",
    "spot_change_5m",
    "spot_change_1m",
    "spot_change",
    "strike",
    "option_type",
)


_CONTEXT_HINT_COLUMNS = (
    "is_expiry_day",
    "is_expiry_week",
    "is_first_hour",
    "iv_zscore_1m",
    "iv_zscore_5m",
    "iv_zscore_15m",
    "gamma",
    "gamma_x_spot",
    "current_iv",
)


def build_holdout_performance_report(data_dir: str, doc: dict[str, Any]) -> dict[str, Any]:
    if not doc.get("is_walk_forward"):
        return {"ok": False, "error": "Holdout performance analysis requires a walk-forward model"}

    model_name = str(doc.get("model_name") or "").strip()
    if not model_name:
        return {"ok": False, "error": "Model name missing from document"}

    config_raw = doc.get("config") if isinstance(doc.get("config"), dict) else {}
    try:
        training_cfg = normalize_training_config(config_raw)
    except Exception as exc:
        return {"ok": False, "error": f"Invalid training config: {exc}"}

    dataset = training_cfg.dataset
    target = training_cfg.target
    features = _model_feature_names(data_dir, model_name, doc)
    if not features:
        return {"ok": False, "error": "No feature list available for this model"}

    vol_col = _first_vol_column(features)
    wanted = list(dict.fromkeys([
        target, *features, *_IDENTITY_COLUMNS,
        *([vol_col] if vol_col else list(_VOL_COLUMNS)),
        *_CONTEXT_HINT_COLUMNS,
        *_TOP1_ANALYSIS_COLUMNS,
    ]))
    try:
        df, _, _ = load_dataset_frame(data_dir, dataset, columns=wanted)
    except DatasetLoaderError as exc:
        return {"ok": False, "error": str(exc)}

    n_rows = len(df)
    try:
        holdout_start, holdout_stop = resolve_holdout_slice(doc, n_rows)
    except HoldoutPerformanceError as exc:
        return {"ok": False, "error": str(exc)}

    if holdout_stop - holdout_start < 1:
        return {"ok": False, "error": "Holdout region is empty"}

    wf_df = df.iloc[:holdout_start]
    ho_df = df.iloc[holdout_start:holdout_stop]

    use_features = [f for f in features if f in df.columns]
    if not use_features:
        return {"ok": False, "error": "Model features not found in dataset parquet"}

    baseline_wf = resolve_ltp_baseline_from_frames(wf_df)
    baseline_ho = resolve_ltp_baseline_from_frames(ho_df)

    vol_name = vol_col or _first_vol_column(list(df.columns))
    vol_wf = wf_df[vol_name] if vol_name and vol_name in wf_df.columns else None
    vol_ho = ho_df[vol_name] if vol_name and vol_name in ho_df.columns else None

    td_wf = wf_df["trading_day"] if "trading_day" in wf_df.columns else None
    td_ho = ho_df["trading_day"] if "trading_day" in ho_df.columns else None

    feat_frames: list[tuple[str, pd.Series, pd.Series]] = []
    ranked_features = _top_features_for_comparison(doc, use_features, limit=len(use_features))
    importance_map = _importance_map_from_doc(doc)
    feature_ranking = build_feature_drift_ranking(
        wf_df, ho_df, ranked_features, importance_map=importance_map, limit=20,
    )

    comparison_rows = build_region_comparison_rows(
        target_wf=wf_df[target],
        target_holdout=ho_df[target],
        baseline_wf=baseline_wf,
        baseline_holdout=baseline_ho,
        vol_wf=vol_wf,
        vol_holdout=vol_ho,
        trading_day_wf=td_wf,
        trading_day_holdout=td_ho,
        feature_frames=feat_frames,
    )

    wf_tgt_stats = distribution_summary(wf_df[target])
    ho_tgt_stats = distribution_summary(ho_df[target])
    wf_vol_stats = distribution_summary(vol_wf) if vol_wf is not None else None
    ho_vol_stats = distribution_summary(vol_ho) if vol_ho is not None else None

    drift_scores = compute_drift_scores(
        target_wf=wf_df[target],
        target_holdout=ho_df[target],
        baseline_wf=baseline_wf,
        baseline_holdout=baseline_ho,
        vol_wf=vol_wf,
        vol_holdout=vol_ho,
        feature_ranking=feature_ranking,
    )
    similarity_pct = compute_similarity_score(drift_scores)

    wf_bands = premium_band_pct(baseline_wf) if baseline_wf is not None else {}
    ho_bands = premium_band_pct(baseline_ho) if baseline_ho is not None else {}
    premium_pct_change = None
    if wf_bands and ho_bands:
        premium_pct_change = sum(abs(ho_bands.get(label, 0.0) - wf_bands.get(label, 0.0)) for label, _, _ in PREMIUM_METRIC_BANDS)

    ho_unique_days = int(td_ho.dropna().astype(str).nunique()) if td_ho is not None else None
    wf_unique_days = int(td_wf.dropna().astype(str).nunique()) if td_wf is not None else None

    pkg = model_package_dir(data_dir, model_name)
    model_path = resolve_production_model_path(pkg, algorithm=training_cfg.algorithm)
    if not model_path:
        return {"ok": False, "error": "Production model file not found in model package"}

    try:
        model = load_prediction_model(model_path, training_cfg.algorithm)
    except Exception as exc:
        return {"ok": False, "error": f"Could not load model: {exc}"}

    X_wf = wf_df[use_features]
    X_ho = ho_df[use_features]
    try:
        pred_wf = np.asarray(model.predict(X_wf), dtype=float)
        pred_ho = np.asarray(model.predict(X_ho), dtype=float)
    except Exception as exc:
        return {"ok": False, "error": f"Model prediction failed: {exc}"}

    y_wf = pd.to_numeric(wf_df[target], errors="coerce")
    y_ho = pd.to_numeric(ho_df[target], errors="coerce")

    wf_err = evaluate_regression(y_wf, pred_wf, baseline=baseline_wf)
    ho_err = evaluate_regression(y_ho, pred_ho, baseline=baseline_ho)

    saved_metrics = extract_saved_prediction_metrics(doc)
    production_wf = saved_metrics["production_wf"]
    holdout_test = saved_metrics["holdout_test"]

    metrics_doc = doc.get("metrics") if isinstance(doc.get("metrics"), dict) else {}
    val_doc = metrics_doc.get("validation") if isinstance(metrics_doc.get("validation"), dict) else {}
    wf_ref_mae = production_wf.get("mae") or val_doc.get("mae")
    holdout_ref_mae = holdout_test.get("mae")

    wf_tgt_stats = distribution_summary(y_wf)
    ho_tgt_stats = distribution_summary(y_ho)
    diagnosis = diagnose_degradation(
        wf_validation_mae=float(wf_ref_mae) if wf_ref_mae is not None else None,
        holdout_mae=float(holdout_ref_mae) if holdout_ref_mae is not None else (
            float(ho_err.get("mae")) if ho_err.get("mae") is not None else None
        ),
        comparison_rows=comparison_rows,
        wf_target_std=float(wf_tgt_stats["std"]) if wf_tgt_stats.get("std") is not None else None,
        holdout_target_std=float(ho_tgt_stats["std"]) if ho_tgt_stats.get("std") is not None else None,
        drift_scores=drift_scores,
        feature_ranking=feature_ranking,
        wf_tgt_stats=wf_tgt_stats,
        ho_tgt_stats=ho_tgt_stats,
        wf_vol_stats=wf_vol_stats,
        ho_vol_stats=ho_vol_stats,
        premium_pct_change=premium_pct_change,
        holdout_unique_days=ho_unique_days,
        wf_unique_days=wf_unique_days,
        similarity_pct=similarity_pct,
    )

    comparison_rows.extend([
        _comparison_row(category="Prediction errors", metric="MAE", wf_val=production_wf.get("mae"), holdout_val=holdout_test.get("mae"),
                        shift=(float(holdout_test["mae"]) - float(production_wf["mae"])) if holdout_test.get("mae") is not None and production_wf.get("mae") is not None else None),
        _comparison_row(category="Prediction errors", metric="RMSE", wf_val=production_wf.get("rmse"), holdout_val=holdout_test.get("rmse"),
                        shift=(float(holdout_test["rmse"]) - float(production_wf["rmse"])) if holdout_test.get("rmse") is not None and production_wf.get("rmse") is not None else None),
        _comparison_row(category="Prediction errors", metric="Premium MAE %", wf_val=production_wf.get("premium_mae_pct"), holdout_val=holdout_test.get("premium_mae_pct"),
                        shift=(float(holdout_test["premium_mae_pct"]) - float(production_wf["premium_mae_pct"]))
                        if holdout_test.get("premium_mae_pct") is not None and production_wf.get("premium_mae_pct") is not None else None),
        _comparison_row(category="Prediction errors", metric="Direction %", wf_val=production_wf.get("directional_accuracy_pct"), holdout_val=holdout_test.get("directional_accuracy_pct"),
                        shift=(float(holdout_test["directional_accuracy_pct"]) - float(production_wf["directional_accuracy_pct"]))
                        if holdout_test.get("directional_accuracy_pct") is not None and production_wf.get("directional_accuracy_pct") is not None else None),
    ])

    ho_bands = premium_band_performance(
        y_ho.to_numpy(),
        pred_ho,
        baseline=baseline_ho.to_numpy() if baseline_ho is not None else None,
    )

    by_day: list[dict[str, Any]] = []
    if td_ho is not None:
        by_day = holdout_performance_by_trading_day(
            y_ho,
            pred_ho,
            td_ho,
            baseline=baseline_ho,
        )

    wf_day_lo, wf_day_hi = _day_range(td_wf)
    ho_day_lo, ho_day_hi = _day_range(td_ho)

    from .holdout_premium_analysis import build_premium_analysis
    from .holdout_days_analysis import build_holdout_days_analysis

    premium_analysis = build_premium_analysis(
        y_wf=y_wf,
        pred_wf=pred_wf,
        y_ho=y_ho,
        pred_ho=pred_ho,
        ho_df=ho_df,
        baseline_ho=baseline_ho,
        drift_scores=drift_scores,
        similarity_pct=similarity_pct,
        production_wf=production_wf,
        holdout_test=holdout_test,
        by_trading_day=by_day,
        feature_drift_ranking=feature_ranking,
        model_name=model_name,
        model=model,
        use_features=use_features,
    )

    holdout_days_analysis = build_holdout_days_analysis(
        holdout_df=ho_df,
        y_holdout=y_ho,
        pred_holdout=pred_ho,
        train_df=wf_df,
        y_train=y_wf,
        pred_train=pred_wf,
        model_name=model_name,
    )

    return {
        "ok": True,
        "overview": {
            "wf_rows": int(len(wf_df)),
            "holdout_rows": int(len(ho_df)),
            "holdout_start": holdout_start,
            "holdout_stop": holdout_stop,
            "wf_day_start": wf_day_lo,
            "wf_day_end": wf_day_hi,
            "holdout_day_start": ho_day_lo,
            "holdout_day_end": ho_day_hi,
            "volatility_column": vol_name,
        },
        "region_comparison": comparison_rows,
        "drift_scores": drift_scores,
        "similarity_pct": similarity_pct,
        "feature_drift_ranking": feature_ranking,
        "diagnosis": diagnosis,
        "prediction_errors": {
            "production_wf": production_wf,
            "holdout_test": holdout_test,
            "change": build_prediction_error_change_row(production_wf, holdout_test),
            "wf_region_recomputed": wf_err,
            "holdout_recomputed": ho_err,
        },
        "holdout_by_premium_band": ho_bands,
        "holdout_by_trading_day": by_day,
        "premium_analysis": premium_analysis,
        "holdout_days_analysis": holdout_days_analysis,
    }


def build_holdout_overview_csv(report: dict[str, Any]) -> str:
    """Serialize Overview tab sections to CSV text."""
    buf = io.StringIO()
    writer = csv.writer(buf)

    def _section(title: str) -> None:
        writer.writerow([])
        writer.writerow([title])

    similarity = report.get("similarity_pct")
    if similarity is not None:
        _section("Similarity Score")
        writer.writerow(["Training vs Holdout Similarity %", f"{float(similarity):.1f}"])

    drift_scores = report.get("drift_scores") if isinstance(report.get("drift_scores"), dict) else {}
    pred = report.get("prediction_errors") if isinstance(report.get("prediction_errors"), dict) else {}
    production_wf = pred.get("production_wf") if isinstance(pred.get("production_wf"), dict) else {}
    holdout_test = pred.get("holdout_test") if isinstance(pred.get("holdout_test"), dict) else {}
    if drift_scores or production_wf or holdout_test:
        _section("Drift Score")
        writer.writerow(["Metric", "Score"])
        for key, label in (
            ("target", "Target Drift"),
            ("feature", "Feature Drift"),
            ("premium", "Premium Drift"),
            ("volatility", "Volatility Drift"),
        ):
            val = drift_scores.get(key)
            writer.writerow([label, f"{float(val):.3f}" if val is not None else ""])

        if production_wf or holdout_test:
            _section("Prediction Errors")
            writer.writerow(["Region", "MAE", "RMSE", "Premium MAE %", "Premium RMSE %", "Direction Acc %"])
            for label, metrics in (("Production WF", production_wf), ("Holdout Test", holdout_test)):
                if not metrics:
                    continue
                writer.writerow([
                    label,
                    metrics.get("mae") if metrics.get("mae") is not None else "",
                    metrics.get("rmse") if metrics.get("rmse") is not None else "",
                    metrics.get("premium_mae_pct") if metrics.get("premium_mae_pct") is not None else "",
                    metrics.get("premium_rmse_pct") if metrics.get("premium_rmse_pct") is not None else "",
                    metrics.get("directional_accuracy_pct") if metrics.get("directional_accuracy_pct") is not None else "",
                ])
            change = pred.get("change") if isinstance(pred.get("change"), dict) else {}
            if change:
                writer.writerow([
                    "Change (Holdout − WF)",
                    change.get("mae_pct_change") if change.get("mae_pct_change") is not None else "",
                    change.get("rmse_pct_change") if change.get("rmse_pct_change") is not None else "",
                    change.get("premium_mae_pct_change") if change.get("premium_mae_pct_change") is not None else "",
                    change.get("premium_rmse_pct_change") if change.get("premium_rmse_pct_change") is not None else "",
                    change.get("direction_pts_change") if change.get("direction_pts_change") is not None else "",
                ])

    diagnosis = report.get("diagnosis") if isinstance(report.get("diagnosis"), dict) else {}
    if diagnosis:
        _section("Degradation Diagnosis")
        writer.writerow(["Field", "Value"])
        writer.writerow(["Primary Cause", diagnosis.get("label") or ""])
        if diagnosis.get("confidence_pct") is not None:
            writer.writerow(["Confidence %", f"{float(diagnosis['confidence_pct']):.0f}"])
        if diagnosis.get("likely_reason"):
            writer.writerow(["Likely Reason", diagnosis.get("likely_reason")])
        for item in diagnosis.get("evidence") or diagnosis.get("signals") or []:
            writer.writerow(["Evidence", item])

    feat_rank = report.get("feature_drift_ranking") or []
    if feat_rank:
        _section("Feature Drift Ranking")
        writer.writerow([
            "Feature", "WF Mean", "Holdout Mean", "Drift %", "Drift Score", "Importance", "Risk",
        ])
        for row in feat_rank:
            if not isinstance(row, dict):
                continue
            writer.writerow([
                row.get("feature") or "",
                row.get("wf_mean") if row.get("wf_mean") is not None else "",
                row.get("holdout_mean") if row.get("holdout_mean") is not None else "",
                row.get("drift_pct") if row.get("drift_pct") is not None else "",
                row.get("drift") if row.get("drift") is not None else "",
                row.get("importance") if row.get("importance") is not None else "",
                row.get("risk_label") or row.get("risk") or "",
            ])

    overview = report.get("overview") if isinstance(report.get("overview"), dict) else {}
    if overview:
        _section("Region Overview")
        writer.writerow(["Field", "Value"])
        for label, key in (
            ("WF region rows", "wf_rows"),
            ("Holdout rows", "holdout_rows"),
            ("Holdout indices", "holdout_range"),
            ("WF trading days", "wf_days"),
            ("Holdout trading days", "holdout_days"),
            ("Volatility column", "volatility_column"),
        ):
            if key == "holdout_range":
                val = f"{overview.get('holdout_start')} – {overview.get('holdout_stop')}"
            elif key == "wf_days":
                val = f"{overview.get('wf_day_start') or '—'} → {overview.get('wf_day_end') or '—'}"
            elif key == "holdout_days":
                val = f"{overview.get('holdout_day_start') or '—'} → {overview.get('holdout_day_end') or '—'}"
            else:
                val = overview.get(key) or ""
            writer.writerow([label, val])

    comp_rows = report.get("region_comparison") or []
    if comp_rows:
        _section("Region Comparison")
        writer.writerow(["Dimension", "Metric", "WF Region", "Holdout", "Shift"])
        for row in comp_rows:
            if not isinstance(row, dict):
                continue
            writer.writerow([
                row.get("category") or "",
                row.get("metric") or "",
                row.get("wf") if row.get("wf") is not None else "",
                row.get("holdout") if row.get("holdout") is not None else "",
                row.get("shift") if row.get("shift") is not None else "",
            ])

    band_rows = report.get("holdout_by_premium_band") or []
    if band_rows:
        _section("Holdout by Premium Band")
        writer.writerow(["Band", "Rows", "MAE", "RMSE", "Premium MAE %", "Direction Acc %"])
        for row in band_rows:
            if not isinstance(row, dict):
                continue
            writer.writerow([
                row.get("band_label") or row.get("band") or "",
                row.get("samples") if row.get("samples") is not None else "",
                row.get("mae") if row.get("mae") is not None else "",
                row.get("rmse") if row.get("rmse") is not None else "",
                row.get("premium_mae_pct") if row.get("premium_mae_pct") is not None else "",
                row.get("directional_accuracy_pct") if row.get("directional_accuracy_pct") is not None else "",
            ])

    day_rows = report.get("holdout_by_trading_day") or []
    if day_rows:
        _section("Holdout by Trading Day")
        writer.writerow(["Trading Day", "Rows", "MAE", "RMSE", "Premium MAE %", "Direction Acc %"])
        for row in day_rows:
            if not isinstance(row, dict):
                continue
            writer.writerow([
                row.get("trading_day") or "",
                row.get("rows") if row.get("rows") is not None else "",
                row.get("mae") if row.get("mae") is not None else "",
                row.get("rmse") if row.get("rmse") is not None else "",
                row.get("premium_mae_pct") if row.get("premium_mae_pct") is not None else "",
                row.get("directional_accuracy_pct") if row.get("directional_accuracy_pct") is not None else "",
            ])

    return buf.getvalue().lstrip("\n")


def build_holdout_performance_csv(report: dict[str, Any]) -> str:
    """Serialize all Holdout Performance tabs (Overview, Premium, Top 1%, Days) to one CSV."""
    parts: list[str] = []

    parts.append("TAB: Overview")
    parts.append(build_holdout_overview_csv(report))

    pa = report.get("premium_analysis") if isinstance(report.get("premium_analysis"), dict) else {}
    if pa:
        from .holdout_premium_analysis import build_premium_analysis_csv

        parts.append("")
        parts.append("TAB: Premium Analysis")
        parts.append(build_premium_analysis_csv(pa))

    top1 = pa.get("top1_analysis") if isinstance(pa.get("top1_analysis"), dict) else {}
    if top1.get("ok"):
        from .holdout_top1_analysis import build_top1_analysis_csv

        parts.append("")
        parts.append("TAB: Top 1% Error Analysis")
        parts.append(build_top1_analysis_csv(top1))

    days = report.get("holdout_days_analysis") if isinstance(report.get("holdout_days_analysis"), dict) else {}
    if days.get("ok"):
        from .holdout_days_analysis import build_holdout_days_analysis_csv

        parts.append("")
        parts.append("TAB: Holdout Days Analysis")
        parts.append(build_holdout_days_analysis_csv(days))

    return "\n".join(parts).lstrip("\n")

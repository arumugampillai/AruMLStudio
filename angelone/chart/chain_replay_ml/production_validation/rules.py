"""Rank-based Holdout vs Unseen comparison + multi-signal recommendation.

Replaces misleading Importance Collapse % with:
  Holdout Rank / Unseen Rank / Rank Change / Importance Difference
plus optional Drift Studio signals (feature drift, KS, Wasserstein).
"""

from __future__ import annotations

from typing import Any

# --- Ranking / summary -------------------------------------------------------
# Rank Change = Holdout Rank − Unseen Rank.
# Negative ⇒ feature fell on unseen (less important); positive ⇒ rose.
_NEAR_ZERO = 1e-12
_STABLE_RANK_ABS = 1  # |ΔRank| ≤ this ⇒ "stable"

# --- Recommendation thresholds (frozen note mirrors these) -------------------
# Large / medium rank *drop* (fell on unseen ⇒ Rank Change negative).
_REMOVE_RANK_DROP = 5  # Rank Change ≤ −5
_WATCH_RANK_DROP = 2  # Rank Change ≤ −2

# Relative importance drop: (−ImpDiff) / |Holdout| when ImpDiff = Unseen − Holdout.
_REMOVE_IMP_DROP_FRAC = 0.50
_WATCH_IMP_DROP_FRAC = 0.25

# Drift Studio signals (WF→Holdout artifacts when present for this model).
_REMOVE_DRIFT = 0.50
_WATCH_DRIFT = 0.25
_REMOVE_KS = 0.35
_WATCH_KS = 0.20
_REMOVE_WASSERSTEIN_NORM = 1.0
_WATCH_WASSERSTEIN_NORM = 0.50
# Raw Wasserstein fallback when normalized is missing.
_REMOVE_WASSERSTEIN_RAW = 2.0
_WATCH_WASSERSTEIN_RAW = 1.0

RECOMMENDATION_THRESHOLDS: dict[str, Any] = {
    "stable_rank_abs": _STABLE_RANK_ABS,
    "remove_rank_drop": _REMOVE_RANK_DROP,
    "watch_rank_drop": _WATCH_RANK_DROP,
    "remove_imp_drop_frac": _REMOVE_IMP_DROP_FRAC,
    "watch_imp_drop_frac": _WATCH_IMP_DROP_FRAC,
    "remove_drift": _REMOVE_DRIFT,
    "watch_drift": _WATCH_DRIFT,
    "remove_ks": _REMOVE_KS,
    "watch_ks": _WATCH_KS,
    "remove_wasserstein_normalized": _REMOVE_WASSERSTEIN_NORM,
    "watch_wasserstein_normalized": _WATCH_WASSERSTEIN_NORM,
    "remove_wasserstein_raw": _REMOVE_WASSERSTEIN_RAW,
    "watch_wasserstein_raw": _WATCH_WASSERSTEIN_RAW,
}


def rank_by_abs_importance(importance: dict[str, float]) -> dict[str, int]:
    """Rank features by ``abs(importance)``; 1 = most important.

    Ties broken by feature name for determinism.
    """
    ordered = sorted(importance.items(), key=lambda kv: (-abs(float(kv[1])), str(kv[0])))
    return {feat: i + 1 for i, (feat, _) in enumerate(ordered)}


def importance_difference(holdout: float | None, unseen: float | None) -> float | None:
    """Unseen − Holdout. Negative = importance fell on unseen."""
    if holdout is None or unseen is None:
        return None
    try:
        return float(unseen) - float(holdout)
    except (TypeError, ValueError):
        return None


def relative_importance_drop(holdout: float | None, imp_diff: float | None) -> float | None:
    """How much importance fell relative to |holdout|: ``max(0, −Δimp) / |holdout|``.

    0 = no drop; 1 = full drop of holdout magnitude. None if inputs missing / near-zero holdout
    with a non-zero drop (treated as full drop = 1.0).
    """
    if holdout is None or imp_diff is None:
        return None
    try:
        h = float(holdout)
        d = float(imp_diff)
    except (TypeError, ValueError):
        return None
    drop = max(0.0, -d)
    denom = abs(h)
    if denom <= _NEAR_ZERO:
        return 1.0 if drop > _NEAR_ZERO else 0.0
    return drop / denom


def _sev_rank(rank_change: int | None) -> int:
    """0/1/2 severity from rank drop (negative Rank Change)."""
    if rank_change is None:
        return 0
    if rank_change <= -_REMOVE_RANK_DROP:
        return 2
    if rank_change <= -_WATCH_RANK_DROP:
        return 1
    return 0


def _sev_imp(rel_drop: float | None) -> int:
    if rel_drop is None:
        return 0
    if rel_drop >= _REMOVE_IMP_DROP_FRAC:
        return 2
    if rel_drop >= _WATCH_IMP_DROP_FRAC:
        return 1
    return 0


def _sev_drift_signals(
    *,
    feature_drift: float | None,
    ks_statistic: float | None,
    wasserstein_distance: float | None,
    wasserstein_normalized: float | None,
) -> tuple[int, bool]:
    """Return (severity 0/1/2, signals_available).

    Uses the max severity across available Drift/KS/W signals.
    """
    sevs: list[int] = []
    if feature_drift is not None:
        try:
            d = float(feature_drift)
        except (TypeError, ValueError):
            d = None
        else:
            if d >= _REMOVE_DRIFT:
                sevs.append(2)
            elif d >= _WATCH_DRIFT:
                sevs.append(1)
            else:
                sevs.append(0)
    if ks_statistic is not None:
        try:
            ks = float(ks_statistic)
        except (TypeError, ValueError):
            ks = None
        else:
            if ks >= _REMOVE_KS:
                sevs.append(2)
            elif ks >= _WATCH_KS:
                sevs.append(1)
            else:
                sevs.append(0)
    w_norm = wasserstein_normalized
    if w_norm is not None:
        try:
            wn = float(w_norm)
        except (TypeError, ValueError):
            wn = None
        else:
            if wn >= _REMOVE_WASSERSTEIN_NORM:
                sevs.append(2)
            elif wn >= _WATCH_WASSERSTEIN_NORM:
                sevs.append(1)
            else:
                sevs.append(0)
    elif wasserstein_distance is not None:
        try:
            wr = float(wasserstein_distance)
        except (TypeError, ValueError):
            wr = None
        else:
            if wr >= _REMOVE_WASSERSTEIN_RAW:
                sevs.append(2)
            elif wr >= _WATCH_WASSERSTEIN_RAW:
                sevs.append(1)
            else:
                sevs.append(0)
    if not sevs:
        return 0, False
    return max(sevs), True


def recommend_feature(
    *,
    rank_change: int | None,
    holdout_importance: float | None,
    importance_diff: float | None,
    feature_drift: float | None = None,
    ks_statistic: float | None = None,
    wasserstein_distance: float | None = None,
    wasserstein_normalized: float | None = None,
) -> tuple[str, dict[str, Any]]:
    """Multi-signal ``KEEP`` / ``WATCH`` / ``REMOVE``.

    REMOVE: large rank drop AND large |Δimp| drop AND (high drift/KS/W if available).
    If Drift/KS/W unavailable → degrade to rank + Δimp only.
    WATCH: any medium signal (and not REMOVE).
    KEEP: otherwise.
    """
    rel_drop = relative_importance_drop(holdout_importance, importance_diff)
    s_rank = _sev_rank(rank_change)
    s_imp = _sev_imp(rel_drop)
    s_drift, drift_available = _sev_drift_signals(
        feature_drift=feature_drift,
        ks_statistic=ks_statistic,
        wasserstein_distance=wasserstein_distance,
        wasserstein_normalized=wasserstein_normalized,
    )
    detail = {
        "rank_severity": s_rank,
        "imp_severity": s_imp,
        "drift_severity": s_drift if drift_available else None,
        "drift_signals_available": drift_available,
        "relative_imp_drop": round(rel_drop, 6) if rel_drop is not None else None,
    }
    if drift_available:
        if s_rank >= 2 and s_imp >= 2 and s_drift >= 1:
            return "REMOVE", detail
        if s_rank >= 1 or s_imp >= 1 or s_drift >= 1:
            return "WATCH", detail
        return "KEEP", detail

    # Degrade: rank + Δimp only.
    if s_rank >= 2 and s_imp >= 2:
        return "REMOVE", detail
    if s_rank >= 1 or s_imp >= 1:
        return "WATCH", detail
    return "KEEP", detail


def build_feature_rows(
    *,
    holdout_importance: dict[str, float],
    unseen_importance: dict[str, float],
    drift_by_feature: dict[str, dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build ranked comparison rows for all features present in both maps.

    Returns ``(rows, join_meta)`` where ``join_meta`` notes drift join coverage.
    """
    drift_map = drift_by_feature or {}
    features = [f for f in holdout_importance if f in unseen_importance]
    features = sorted(features)
    ho_ranks = rank_by_abs_importance({f: holdout_importance[f] for f in features})
    un_ranks = rank_by_abs_importance({f: unseen_importance[f] for f in features})

    rows: list[dict[str, Any]] = []
    drift_joined = 0
    for feat in features:
        ho = float(holdout_importance[feat])
        un = float(unseen_importance[feat])
        ho_rank = int(ho_ranks[feat])
        un_rank = int(un_ranks[feat])
        rank_change = ho_rank - un_rank
        imp_diff = importance_difference(ho, un)
        assert imp_diff is not None

        drow = drift_map.get(feat) if isinstance(drift_map.get(feat), dict) else None
        feature_drift = _opt_float(drow, "drift", "feature_drift") if drow else None
        ks = _opt_float(drow, "ks_statistic") if drow else None
        w_raw = _opt_float(drow, "wasserstein_distance") if drow else None
        w_norm = _opt_float(drow, "wasserstein_normalized") if drow else None
        if drow and any(v is not None for v in (feature_drift, ks, w_raw, w_norm)):
            drift_joined += 1

        rec, detail = recommend_feature(
            rank_change=rank_change,
            holdout_importance=ho,
            importance_diff=imp_diff,
            feature_drift=feature_drift,
            ks_statistic=ks,
            wasserstein_distance=w_raw,
            wasserstein_normalized=w_norm,
        )
        rows.append(
            {
                "feature": feat,
                "holdout_rank": ho_rank,
                "unseen_rank": un_rank,
                "rank_change": rank_change,
                "holdout_importance": ho,
                "unseen_importance": un,
                "importance_difference": round(imp_diff, 8),
                "feature_drift": feature_drift,
                "ks_statistic": ks,
                "wasserstein_distance": w_raw,
                "wasserstein_normalized": w_norm,
                "recommendation": rec,
                "recommendation_detail": detail,
            }
        )

    # Worst rank drops first, then largest relative importance drop.
    rows.sort(
        key=lambda r: (
            int(r.get("rank_change") or 0),
            float(r.get("importance_difference") or 0.0),
            str(r.get("feature") or ""),
        )
    )

    n = len(rows)
    join_meta = {
        "drift_join_attempted": bool(drift_map),
        "drift_features_available": len(drift_map),
        "drift_features_joined": drift_joined,
        "drift_signals_used": drift_joined > 0,
        "degraded_to_rank_imp_only": drift_joined == 0,
        "note": (
            "Recommendations used Drift Studio Feature Drift / KS / Wasserstein where joined."
            if drift_joined > 0
            else (
                "Drift Studio artifacts missing or unmatched — "
                "recommendations used Rank Change + Importance Difference only."
            )
        ),
        "feature_count": n,
    }
    return rows, join_meta


def comparison_rows_need_rank_enrichment(rows: list[dict[str, Any]]) -> bool:
    """True when rows have importances but lack rank / Δimportance fields.

    Covers pre-v1.1 artifacts that stored ``collapse_pct`` instead of ranks.
    """
    if not rows:
        return False
    for row in rows:
        if not isinstance(row, dict):
            continue
        ho = row.get("holdout_importance")
        un = row.get("unseen_importance")
        if ho is None or un is None:
            continue
        if (
            row.get("holdout_rank") is None
            or row.get("unseen_rank") is None
            or row.get("rank_change") is None
            or row.get("importance_difference") is None
        ):
            return True
    return False


def enrich_comparison_rows_from_importances(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], bool]:
    """Derive ranks / Rank Change / Importance Difference from stored importances.

    Does **not** re-run permutation. Rebuilds recommendations under the current
    rank-based rules (and preserves Drift/KS/W fields when already on the row).

    Returns ``(rows, enriched)`` where ``enriched`` is True only when ranks were
    derived (caller should refresh summary strip / dual confidence).
    """
    if not comparison_rows_need_rank_enrichment(rows):
        return rows, False

    holdout: dict[str, float] = {}
    unseen: dict[str, float] = {}
    drift_by_feature: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        feat = str(row.get("feature") or "").strip()
        if not feat:
            continue
        try:
            ho = float(row["holdout_importance"])
            un = float(row["unseen_importance"])
        except (KeyError, TypeError, ValueError):
            continue
        holdout[feat] = ho
        unseen[feat] = un
        drow: dict[str, Any] = {}
        for src, dst in (
            ("feature_drift", "drift"),
            ("drift", "drift"),
            ("ks_statistic", "ks_statistic"),
            ("wasserstein_distance", "wasserstein_distance"),
            ("wasserstein_normalized", "wasserstein_normalized"),
        ):
            val = row.get(src)
            if val is not None and dst not in drow:
                drow[dst] = val
        if drow:
            drift_by_feature[feat] = drow

    if not holdout:
        return rows, False

    enriched, _join = build_feature_rows(
        holdout_importance=holdout,
        unseen_importance=unseen,
        drift_by_feature=drift_by_feature or None,
    )
    return enriched, True


def build_feature_validation_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Summary strip metrics over all scored model features."""
    n = len(rows)
    recs = [str(r.get("recommendation") or "") for r in rows]
    keep_n = sum(1 for r in recs if r == "KEEP")
    watch_n = sum(1 for r in recs if r == "WATCH")
    remove_n = sum(1 for r in recs if r == "REMOVE")
    changes = [int(r["rank_change"]) for r in rows if r.get("rank_change") is not None]
    avg_rc = (sum(changes) / len(changes)) if changes else 0.0
    med_rc = _median([float(c) for c in changes]) if changes else 0.0
    stable_n = sum(1 for c in changes if abs(c) <= _STABLE_RANK_ABS)
    stable_pct = (100.0 * stable_n / n) if n else 0.0
    return {
        "keep_count": keep_n,
        "watch_count": watch_n,
        "remove_count": remove_n,
        "feature_count": n,
        "average_rank_change": round(avg_rc, 4),
        "median_rank_change": round(med_rc, 4),
        "stable_features_count": stable_n,
        "stable_features_pct": round(stable_pct, 2),
        "stable_rank_abs_threshold": _STABLE_RANK_ABS,
    }


def build_dual_confidence(
    rows: list[dict[str, Any]],
    *,
    unseen_day_count: int,
    feature_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Diagnosis + Production Confirmation from rank-based recommendations."""
    n = len(rows)
    feat_sum = feature_summary or build_feature_validation_summary(rows)
    keep_n = int(feat_sum.get("keep_count") or 0)
    watch_n = int(feat_sum.get("watch_count") or 0)
    remove_n = int(feat_sum.get("remove_count") or 0)
    remove_frac = (remove_n / n) if n else 0.0
    keep_frac = (keep_n / n) if n else 0.0
    median_rc = float(feat_sum.get("median_rank_change") or 0.0)
    mean_rc = float(feat_sum.get("average_rank_change") or 0.0)
    # Negative median Rank Change ⇒ typical feature fell on unseen.
    median_drop = max(0.0, -median_rc)
    mean_drop = max(0.0, -mean_rc)

    if n == 0:
        label = "Insufficient inputs"
        primary = "insufficient_inputs"
        diag_conf = 0
    elif remove_frac >= 0.40 or median_drop >= float(_REMOVE_RANK_DROP):
        label = "Overfit"
        primary = "overfit"
        diag_conf = _clamp_pct(40 + remove_frac * 50 + median_drop * 4.0)
    elif remove_frac + watch_n / n >= 0.50 or median_drop >= float(_WATCH_RANK_DROP):
        label = "Fragile"
        primary = "fragile"
        diag_conf = _clamp_pct(35 + (remove_frac + watch_n / max(n, 1)) * 40)
    else:
        label = "Stable"
        primary = "stable"
        diag_conf = _clamp_pct(55 + keep_frac * 40 - mean_drop * 3.0)

    days = max(0, int(unseen_day_count))
    confirmed = bool(days >= 1 and n > 0 and keep_frac >= 0.60 and remove_frac < 0.25)
    if days < 1 or n == 0:
        conf_status = "Not Confirmed"
        prod_conf = 0
        explanation = "No unseen days tested."
    elif confirmed:
        conf_status = "Confirmed"
        prod_conf = _clamp_pct(50 + keep_frac * 40 + min(days, 5) * 2)
        explanation = f"Tested {days} unseen day{'s' if days != 1 else ''}."
    else:
        conf_status = "Not Confirmed"
        prod_conf = _clamp_pct(20 + keep_frac * 30 + min(days, 5) * 2)
        explanation = (
            f"Tested {days} unseen day{'s' if days != 1 else ''} — "
            f"rank instability too high for confirmation "
            f"(KEEP {keep_n}/{n}, REMOVE {remove_n}/{n})."
        )

    return {
        "diagnosis": {
            "primary_cause": primary,
            "label": label,
            "confidence_pct": int(diag_conf),
            "median_rank_change": round(median_rc, 4),
            "average_rank_change": round(mean_rc, 4),
            "keep_count": keep_n,
            "watch_count": watch_n,
            "remove_count": remove_n,
            "feature_count": n,
            "stable_features_pct": feat_sum.get("stable_features_pct"),
        },
        "production_confirmation": {
            "status": conf_status,
            "confirmed": confirmed,
            "confidence_pct": int(prod_conf),
            "unseen_days_tested": days,
            "explanation": explanation,
        },
        "thresholds": dict(RECOMMENDATION_THRESHOLDS),
    }


def _opt_float(row: dict[str, Any] | None, *keys: str) -> float | None:
    if not row:
        return None
    for key in keys:
        val = row.get(key)
        if val is None:
            continue
        try:
            return float(val)
        except (TypeError, ValueError):
            continue
    return None


def _median(vals: list[float]) -> float:
    ordered = sorted(vals)
    m = len(ordered)
    if m == 0:
        return 0.0
    mid = m // 2
    if m % 2:
        return float(ordered[mid])
    return float(ordered[mid - 1] + ordered[mid]) / 2.0


def _clamp_pct(val: float) -> int:
    return int(round(max(0.0, min(100.0, float(val)))))

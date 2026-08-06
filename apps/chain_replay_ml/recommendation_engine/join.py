"""Soft-join Importance / Distribution / Drift (+ optional Diagnostics) rows."""

from __future__ import annotations

from typing import Any


def index_by_feature(rows: list[dict[str, Any]] | None) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in rows or []:
        if not isinstance(row, dict):
            continue
        feat = str(row.get("feature") or "").strip()
        if feat:
            out[feat] = row
    return out


def _num(row: dict[str, Any] | None, *keys: str) -> float | None:
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


def build_feature_rows(
    *,
    importance: dict[str, dict[str, Any]],
    drift: dict[str, dict[str, Any]],
    distribution: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """One joined row per feature for rule evaluation."""
    features = sorted(set(importance) | set(drift) | set(distribution))
    rows: list[dict[str, Any]] = []
    for feat in features:
        i = importance.get(feat)
        d = drift.get(feat)
        s = distribution.get(feat)
        risk = str(d.get("risk")).lower() if d and d.get("risk") is not None else None
        null_pct = _num(s, "null_pct")
        if null_pct is None:
            null_pct = _num(d, "null_pct_ho")
        null_drift_pp = _num(d, "null_drift_pp")
        rank_gain = _num(i, "rank_gain")
        if rank_gain is None:
            rank_gain = _num(d, "rank_gain")
        gain = _num(i, "gain")
        if gain is None:
            gain = _num(d, "importance")
        rows.append(
            {
                "feature": feat,
                "rank_gain": rank_gain,
                "gain": gain,
                "shap_mean_abs": _num(i, "shap_mean_abs"),
                "risk": risk,
                "risk_score": _num(d, "risk_score"),
                "drift": _num(d, "drift"),
                "drift_pct": _num(d, "drift_pct"),
                "ks_statistic": _num(d, "ks_statistic"),
                "wasserstein_distance": _num(d, "wasserstein_distance"),
                "wasserstein_normalized": _num(d, "wasserstein_normalized"),
                "null_pct": null_pct,
                "null_drift_pp": null_drift_pp,
                "skew": _num(s, "skew") if _num(s, "skew") is not None else _num(d, "skew_ho"),
            }
        )
    return rows

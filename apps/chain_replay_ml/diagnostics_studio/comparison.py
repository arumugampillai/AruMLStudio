"""Join Importance / Drift / Distribution rows for Diagnostics Studio."""

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


def _num(row: dict[str, Any] | None, key: str) -> float | None:
    if not row:
        return None
    val = row.get(key)
    if val is None:
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def diagnostic_flag(
    *,
    risk: str | None,
    risk_score: float | None,
    null_pct: float | None,
    rank_gain: float | None,
    drift: float | None,
) -> str:
    # Phase 5.2 risk_score is composite 0–100; treat ≥50 as high.
    if risk == "high" or (risk_score is not None and risk_score >= 50.0):
        return "high_risk"
    if null_pct is not None and null_pct >= 5.0:
        return "high_null"
    if (
        rank_gain is not None
        and rank_gain <= 20
        and drift is not None
        and drift >= 0.35
    ):
        return "rank_drift_conflict"
    return "ok"


def build_comparison_rows(
    *,
    importance: dict[str, dict[str, Any]],
    drift: dict[str, dict[str, Any]],
    distribution: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    features = sorted(set(importance) | set(drift) | set(distribution))
    rows: list[dict[str, Any]] = []
    for feat in features:
        i = importance.get(feat)
        d = drift.get(feat)
        s = distribution.get(feat)
        risk = str(d.get("risk")) if d and d.get("risk") is not None else None
        risk_score = _num(d, "risk_score")
        null_pct = _num(s, "null_pct")
        if null_pct is None:
            null_pct = _num(d, "null_pct_ho")
        skew = _num(s, "skew")
        if skew is None:
            skew = _num(d, "skew_ho")
        rank_gain = _num(i, "rank_gain")
        if rank_gain is None:
            rank_gain = _num(d, "rank_gain")
        drift_score = _num(d, "drift")
        rows.append(
            {
                "feature": feat,
                "rank_gain": rank_gain,
                "gain": _num(i, "gain"),
                "shap_mean_abs": _num(i, "shap_mean_abs"),
                "risk": risk,
                "risk_score": risk_score,
                "drift": drift_score,
                "drift_pct": _num(d, "drift_pct"),
                "wf_mean": _num(d, "wf_mean"),
                "holdout_mean": _num(d, "holdout_mean"),
                "null_pct": null_pct,
                "skew": skew,
                "diagnostic_flag": diagnostic_flag(
                    risk=risk,
                    risk_score=risk_score,
                    null_pct=null_pct,
                    rank_gain=rank_gain,
                    drift=drift_score,
                ),
            }
        )

    def sort_key(r: dict[str, Any]) -> tuple:
        rs = r.get("risk_score")
        rg = r.get("rank_gain")
        return (
            -(float(rs) if rs is not None else -1.0),
            float(rg) if rg is not None else 10**9,
            str(r.get("feature") or ""),
        )

    rows.sort(key=sort_key)
    return rows

"""Join two models' studio comparison rows into a side-by-side table."""

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


def _str(row: dict[str, Any] | None, key: str) -> str | None:
    if not row:
        return None
    val = row.get(key)
    if val is None:
        return None
    return str(val)


def _delta(a: float | None, b: float | None) -> float | None:
    if a is None or b is None:
        return None
    return b - a


def build_comparison_rows(
    *,
    imp_a: dict[str, dict[str, Any]],
    imp_b: dict[str, dict[str, Any]],
    drift_a: dict[str, dict[str, Any]],
    drift_b: dict[str, dict[str, Any]],
    dist_a: dict[str, dict[str, Any]],
    dist_b: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    features = sorted(
        set(imp_a)
        | set(imp_b)
        | set(drift_a)
        | set(drift_b)
        | set(dist_a)
        | set(dist_b)
    )
    rows: list[dict[str, Any]] = []
    for feat in features:
        ia, ib = imp_a.get(feat), imp_b.get(feat)
        da, db = drift_a.get(feat), drift_b.get(feat)
        sa, sb = dist_a.get(feat), dist_b.get(feat)
        in_a = any(x is not None for x in (ia, da, sa))
        in_b = any(x is not None for x in (ib, db, sb))
        rank_a = _num(ia, "rank_gain")
        rank_b = _num(ib, "rank_gain")
        risk_a = _num(da, "risk_score")
        risk_b = _num(db, "risk_score")
        null_a = _num(sa, "null_pct")
        null_b = _num(sb, "null_pct")
        rows.append(
            {
                "feature": feat,
                "in_a": in_a,
                "in_b": in_b,
                "rank_gain_a": rank_a,
                "rank_gain_b": rank_b,
                "rank_gain_delta": _delta(rank_a, rank_b),
                "shap_mean_abs_a": _num(ia, "shap_mean_abs"),
                "shap_mean_abs_b": _num(ib, "shap_mean_abs"),
                "gain_a": _num(ia, "gain"),
                "gain_b": _num(ib, "gain"),
                "risk_a": _str(da, "risk"),
                "risk_b": _str(db, "risk"),
                "risk_score_a": risk_a,
                "risk_score_b": risk_b,
                "risk_score_delta": _delta(risk_a, risk_b),
                "drift_a": _num(da, "drift"),
                "drift_b": _num(db, "drift"),
                "null_pct_a": null_a,
                "null_pct_b": null_b,
                "null_pct_delta": _delta(null_a, null_b),
                "skew_a": _num(sa, "skew"),
                "skew_b": _num(sb, "skew"),
                "mean_a": _num(sa, "mean"),
                "mean_b": _num(sb, "mean"),
                "p50_a": _num(sa, "p50"),
                "p50_b": _num(sb, "p50"),
            }
        )

    def sort_key(r: dict[str, Any]) -> tuple:
        rd = r.get("rank_gain_delta")
        rs = r.get("risk_score_delta")
        return (
            -(abs(float(rd)) if rd is not None else -1.0),
            -(abs(float(rs)) if rs is not None else -1.0),
            str(r.get("feature") or ""),
        )

    rows.sort(key=sort_key)
    return rows

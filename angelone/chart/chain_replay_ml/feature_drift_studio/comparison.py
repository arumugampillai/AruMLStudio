"""Join drift ranking with optional Importance / Distribution columns."""

from __future__ import annotations

from typing import Any


_IMPORTANCE_KEYS = ("rank_gain", "rank_shap", "rank_delta_gain_shap", "shap_mean_abs", "gain")
_DIST_KEYS = ("null_pct", "skew", "p50", "std")


def index_rows(rows: list[dict[str, Any]] | None) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in rows or []:
        if not isinstance(row, dict):
            continue
        feat = str(row.get("feature") or "").strip()
        if feat:
            out[feat] = row
    return out


def importance_weights_from_rows(
    rows: list[dict[str, Any]] | None,
) -> dict[str, float]:
    """Normalize Gain (fallback SHAP) to a unit-sum importance map."""
    raw: dict[str, float] = {}
    for row in rows or []:
        if not isinstance(row, dict):
            continue
        feat = str(row.get("feature") or "").strip()
        if not feat:
            continue
        try:
            val = float(row.get("gain") if row.get("gain") is not None else row.get("shap_mean_abs") or 0)
        except (TypeError, ValueError):
            val = 0.0
        if val < 0:
            val = 0.0
        raw[feat] = val
    total = sum(raw.values()) or 1.0
    return {k: v / total for k, v in raw.items()}


def build_comparison_rows(
    drift_rows: list[dict[str, Any]],
    *,
    importance_by_feature: dict[str, dict[str, Any]] | None = None,
    distribution_by_feature: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    imp = importance_by_feature or {}
    dist = distribution_by_feature or {}
    out: list[dict[str, Any]] = []
    for row in drift_rows:
        if not isinstance(row, dict):
            continue
        feat = str(row.get("feature") or "")
        merged = dict(row)
        src_i = imp.get(feat)
        if isinstance(src_i, dict):
            for key in _IMPORTANCE_KEYS:
                if key in src_i:
                    merged[key] = src_i[key]
            merged["importance_joined"] = True
        else:
            merged["importance_joined"] = False
        src_d = dist.get(feat)
        if isinstance(src_d, dict):
            for key in _DIST_KEYS:
                if key in src_d:
                    # prefix holdout distribution fields clearly when not already present
                    if key == "null_pct":
                        merged["null_pct_ho"] = src_d[key]
                    elif key == "skew":
                        merged["skew_ho"] = src_d[key]
                    else:
                        merged[f"{key}_ho"] = src_d[key]
            merged["distribution_joined"] = True
        else:
            merged["distribution_joined"] = False
        out.append(merged)
    return out

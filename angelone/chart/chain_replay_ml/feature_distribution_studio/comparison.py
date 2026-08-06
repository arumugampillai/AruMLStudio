"""Join holdout stats (+ optional Importance ranks) into UI comparison rows."""

from __future__ import annotations

from typing import Any


_IMPORTANCE_KEYS = (
    "rank_gain",
    "rank_shap",
    "rank_delta_gain_shap",
    "shap_mean_abs",
    "gain",
    "permutation_mean",
)


def build_comparison_rows(
    holdout_stats: list[dict[str, Any]],
    *,
    importance_by_feature: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Build UI table rows from holdout stats, optionally joined to Importance."""
    imp = importance_by_feature or {}
    out: list[dict[str, Any]] = []
    for row in holdout_stats:
        if not isinstance(row, dict):
            continue
        feat = str(row.get("feature") or "")
        merged = dict(row)
        src = imp.get(feat)
        if isinstance(src, dict):
            for key in _IMPORTANCE_KEYS:
                if key in src:
                    merged[key] = src[key]
            merged["importance_joined"] = True
        else:
            merged["importance_joined"] = False
        out.append(merged)
    return out


def index_importance_rows(rows: list[dict[str, Any]] | None) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in rows or []:
        if not isinstance(row, dict):
            continue
        feat = str(row.get("feature") or "").strip()
        if feat:
            out[feat] = row
    return out

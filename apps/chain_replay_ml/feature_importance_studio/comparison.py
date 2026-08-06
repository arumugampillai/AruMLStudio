"""Join native / permutation / SHAP into one comparison schema."""

from __future__ import annotations

from typing import Any


def build_comparison_rows(
    *,
    features: list[str],
    native: list[dict[str, Any]],
    permutation: list[dict[str, Any]],
    shap: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Normalize all methods into the UI contract schema."""
    nat = {str(r.get("feature")): r for r in native if r.get("feature")}
    perm = {str(r.get("feature")): r for r in permutation if r.get("feature")}
    shp = {str(r.get("feature")): r for r in shap if r.get("feature")}

    rows: list[dict[str, Any]] = []
    for feat in features:
        n = nat.get(feat) or {}
        p = perm.get(feat) or {}
        s = shp.get(feat) or {}
        rows.append(
            {
                "feature": feat,
                "gain": n.get("gain"),
                "weight": n.get("weight"),
                "cover": n.get("cover"),
                "permutation_mean": p.get("permutation_mean"),
                "permutation_std": p.get("permutation_std"),
                "shap_mean_abs": s.get("shap_mean_abs"),
                "rank_gain": n.get("rank_gain"),
                "rank_permutation": p.get("rank_permutation"),
                "rank_shap": s.get("rank_shap"),
                "rank_delta_gain_shap": (
                    abs(int(n["rank_gain"]) - int(s["rank_shap"]))
                    if n.get("rank_gain") is not None and s.get("rank_shap") is not None
                    else None
                ),
            }
        )
    return rows

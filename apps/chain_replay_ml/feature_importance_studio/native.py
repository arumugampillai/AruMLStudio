"""Native XGBoost importance: gain, weight, cover."""

from __future__ import annotations

from typing import Any


def _booster_from_model(model: Any) -> Any:
    if hasattr(model, "get_booster"):
        return model.get_booster()
    if hasattr(model, "_bst"):
        return model._bst
    return model


def compute_native_xgb_importance(model: Any, features: list[str]) -> list[dict[str, Any]]:
    """Return per-feature gain / weight / cover from a trained XGBoost model."""
    booster = _booster_from_model(model)
    gain = dict(booster.get_score(importance_type="gain") or {})
    weight = dict(booster.get_score(importance_type="weight") or {})
    cover = dict(booster.get_score(importance_type="cover") or {})

    rows: list[dict[str, Any]] = []
    for feat in features:
        rows.append(
            {
                "feature": feat,
                "gain": float(gain.get(feat, 0.0)),
                "weight": float(weight.get(feat, 0.0)),
                "cover": float(cover.get(feat, 0.0)),
            }
        )

    # Rank by gain (primary native metric)
    ranked = sorted(rows, key=lambda r: r["gain"], reverse=True)
    for i, row in enumerate(ranked, start=1):
        row["rank_gain"] = i
    by_feat = {r["feature"]: r for r in ranked}
    return [by_feat[f] for f in features if f in by_feat]

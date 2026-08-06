"""Train vs validation feature drift for a fold."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .feature_rehydration import slice_feature_matrix


def _severity(shift_pct: float | None) -> str:
    if shift_pct is None:
        return "—"
    a = abs(shift_pct)
    if a >= 15:
        return "High"
    if a >= 5:
        return "Medium"
    if a >= 2:
        return "Low"
    return "Stable"


def compute_fold_feature_drift(
    data_dir: str,
    *,
    run: dict[str, Any],
    fold: dict[str, Any],
    top_n: int = 50,
) -> dict[str, Any]:
    sliced = slice_feature_matrix(data_dir, run=run, fold=fold)
    if not sliced.get("ok"):
        return {
            "available": False,
            "note": sliced.get("error") or "Could not load feature matrix for drift analysis.",
            "features": [],
        }

    train: pd.DataFrame = sliced["train"]
    val: pd.DataFrame = sliced["validation"]
    features: list[str] = sliced["features"]
    rows: list[dict[str, Any]] = []

    for feat in features:
        t = pd.to_numeric(train[feat], errors="coerce")
        v = pd.to_numeric(val[feat], errors="coerce")
        t_mean = float(t.mean()) if t.notna().any() else None
        v_mean = float(v.mean()) if v.notna().any() else None
        shift_pct = None
        if t_mean is not None and v_mean is not None and t_mean != 0:
            shift_pct = round((v_mean - t_mean) / abs(t_mean) * 100.0, 2)
        rows.append({
            "feature": feat,
            "train_mean": round(t_mean, 6) if t_mean is not None else None,
            "validation_mean": round(v_mean, 6) if v_mean is not None else None,
            "shift_pct": shift_pct,
            "severity": _severity(shift_pct),
        })

    rows.sort(key=lambda r: abs(r.get("shift_pct") or 0), reverse=True)

    return {
        "available": True,
        "feature_count": len(features),
        "top_drifted": rows[:top_n],
        "note": f"Compared train [{fold.get('train_start')}:{fold.get('train_end')}] vs validation [{fold.get('validation_start')}:{fold.get('validation_end')}] feature means.",
    }

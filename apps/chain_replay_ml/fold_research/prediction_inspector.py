"""Per-row SHAP / feature contributions for Prediction Inspector."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from chain_replay_ml.training.model_runtime import load_prediction_model_cached, resolve_production_model_path

from .feature_rehydration import load_package_config, rehydrate_feature_row


def _booster(model: Any) -> Any:
    if hasattr(model, "get_booster"):
        return model.get_booster()
    return model


def compute_row_contributions(
    model: Any,
    features: list[str],
    feature_values: dict[str, Any],
    *,
    top_n: int = 20,
) -> tuple[list[dict[str, Any]], float | None]:
    try:
        import shap
    except ImportError:
        return [], None

    row = {f: feature_values.get(f) for f in features}
    df = pd.DataFrame([row], columns=features)
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    try:
        explainer = shap.TreeExplainer(_booster(model))
        shap_vals = explainer.shap_values(df)
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[0]
        impacts = np.asarray(shap_vals).reshape(-1)
        base = explainer.expected_value
        if isinstance(base, (list, np.ndarray)):
            base = float(np.asarray(base).reshape(-1)[0])
        else:
            base = float(base)
    except Exception:
        return [], None

    out: list[dict[str, Any]] = []
    for feat, impact in zip(features, impacts):
        val = feature_values.get(feat)
        try:
            v = round(float(val), 4) if val is not None else None
        except (TypeError, ValueError):
            v = None
        out.append({
            "feature": feat,
            "value": v,
            "impact": round(float(impact), 4),
        })
    out.sort(key=lambda r: abs(r.get("impact") or 0), reverse=True)
    return out[:top_n], base


def get_prediction_inspector(
    data_dir: str,
    *,
    prediction_run_id: str,
    fold_id: str,
    prediction_id: str | None = None,
    sequence: int | None = None,
) -> dict[str, Any]:
    from chain_replay_ml.prediction_runs.store import PredictionRunStore

    with PredictionRunStore(data_dir) as store:
        run = store.get_run(prediction_run_id)
        if not run:
            return {"ok": False, "error": "prediction run not found"}
        folds = {f["fold_id"]: f for f in store.list_folds(prediction_run_id)}
        fold = folds.get(fold_id)
        if not fold:
            return {"ok": False, "error": "fold not found"}

        pred_row = store.get_prediction_row(prediction_id) if prediction_id else None
        if not pred_row and sequence is not None:
            rows = store.list_all_rows(prediction_run_id, fold_id=fold_id)
            if 1 <= sequence <= len(rows):
                pred_row = rows[sequence - 1]
        if not pred_row:
            return {"ok": False, "error": "prediction row not found"}

    rehyd = rehydrate_feature_row(data_dir, run=run, fold=fold, prediction_row=pred_row)
    contributions: list[dict[str, Any]] = []
    base_value: float | None = None
    model_prediction: float | None = None

    if rehyd.get("ok") and run.get("package_dir"):
        cfg = load_package_config(run["package_dir"])
        algo = cfg.get("algorithm") or rehyd.get("algorithm")
        model_path = resolve_production_model_path(run["package_dir"], algorithm=algo)
        if model_path:
            try:
                model, _ms, _disk = load_prediction_model_cached(model_path, algo)
                feats = rehyd.get("features") or []
                vals = rehyd.get("feature_values") or {}
                contributions, base_value = compute_row_contributions(model, feats, vals)
                if feats:
                    row_df = pd.DataFrame([{f: vals.get(f) for f in feats}], columns=feats)
                    row_df = row_df.replace([np.inf, -np.inf], np.nan).fillna(0.0)
                    model_prediction = round(float(model.predict(row_df)[0]), 4)
            except Exception:
                pass

    predicted = pred_row.get("predicted_ltp")
    actual = pred_row.get("actual_ltp")
    error = pred_row.get("prediction_error")
    direction = pred_row.get("direction_correct")

    return {
        "ok": True,
        "prediction_id": pred_row.get("prediction_id"),
        "timestamp": pred_row.get("timestamp"),
        "trading_day": pred_row.get("trading_day"),
        "token": pred_row.get("token"),
        "strike": pred_row.get("strike"),
        "option_type": pred_row.get("option_type"),
        "spot": pred_row.get("spot"),
        "ltp": pred_row.get("ltp"),
        "predicted_ltp": predicted,
        "actual_ltp": actual,
        "prediction_error": error,
        "direction_correct": direction,
        "confidence": pred_row.get("confidence"),
        "contributions": contributions,
        "base_value": round(base_value, 4) if base_value is not None else None,
        "model_prediction": model_prediction,
        "final_prediction": predicted,
        "rehydration": {
            "ok": bool(rehyd.get("ok")),
            "error": rehyd.get("error"),
            "global_index": rehyd.get("global_index"),
        },
    }

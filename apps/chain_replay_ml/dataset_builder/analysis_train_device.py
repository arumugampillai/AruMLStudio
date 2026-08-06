"""GPU-first XGBoost + SHAP helpers for Experiment Manager training.

Uses the shared ``training.model_device`` probe so Analysis experiments follow
the same CUDA policy as production trainers, with automatic CPU fallback.
"""
from __future__ import annotations

from typing import Any

import numpy as np

DEVICE_GPU = "GPU"
DEVICE_CPU = "CPU"


def resolve_experiment_xgb_plan() -> Any:
    """Resolve XGBoost device plan (GPU preferred when CUDA probe succeeds)."""
    from chain_replay_ml.training.model_device import resolve_training_device

    return resolve_training_device("xgboost", allow_cpu_fallback=True)


def _cpu_library_params() -> dict[str, Any]:
    return {
        "tree_method": "hist",
        "device": "cpu",
        "predictor": "cpu_predictor",
    }


def _gpu_library_params(plan: Any | None = None) -> dict[str, Any]:
    if plan is not None and getattr(plan, "library_params", None):
        return dict(plan.library_params)
    return {
        "tree_method": "hist",
        "device": "cuda",
        "predictor": "gpu_predictor",
    }


def _build_xgb_regressor(base: dict[str, Any], device_params: dict[str, Any]) -> Any:
    """Construct XGBRegressor; tolerate older installs without ``device=``."""
    from xgboost import XGBRegressor

    kwargs = dict(base)
    kwargs.update(device_params)
    try:
        return XGBRegressor(**kwargs)
    except TypeError:
        # Legacy: drop modern keys, try gpu_hist / hist
        legacy = dict(base)
        device = str(device_params.get("device") or "").lower()
        if device.startswith("cuda") or device == "gpu":
            legacy["tree_method"] = "gpu_hist"
        else:
            legacy["tree_method"] = str(
                device_params.get("tree_method") or "hist"
            )
        legacy.pop("device", None)
        legacy.pop("predictor", None)
        try:
            return XGBRegressor(**legacy)
        except TypeError:
            legacy.pop("tree_method", None)
            return XGBRegressor(**legacy)


def _verify_model_device(model: Any) -> str:
    try:
        from chain_replay_ml.training.model_device import (
            verify_xgboost_booster_device,
        )

        booster = model.get_booster() if hasattr(model, "get_booster") else None
        if booster is None:
            return "cpu"
        return str(verify_xgboost_booster_device(booster) or "cpu")
    except Exception:
        return "cpu"


def fit_xgb_regressor_gpu_first(
    X: Any,
    y: Any,
    *,
    base_params: dict[str, Any] | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Fit XGBRegressor preferring CUDA; fall back to CPU on failure.

    Returns ``(model, device_info)`` where device_info includes:
    ``train_device`` (GPU|CPU), ``executed_device``, ``gpu_name``,
    ``fallback_reason``, ``library_params``.
    """
    base = {
        "n_estimators": 40,
        "max_depth": 3,
        "learning_rate": 0.1,
        "objective": "reg:squarederror",
        "verbosity": 0,
        **dict(base_params or {}),
    }
    plan = resolve_experiment_xgb_plan()
    prefer_gpu = bool(getattr(plan, "use_gpu", False))
    gpu_name = getattr(plan, "gpu_name", None)
    fallback_reason = getattr(plan, "fallback_reason", None)
    probe_notes = list(getattr(plan, "probe_notes", ()) or ())

    if prefer_gpu:
        model = _build_xgb_regressor(base, _gpu_library_params(plan))
        try:
            model.fit(X, y)
            executed = _verify_model_device(model)
            if str(executed).startswith("cuda"):
                return model, {
                    "train_device": DEVICE_GPU,
                    "executed_device": executed,
                    "requested_device": "cuda",
                    "gpu_name": gpu_name,
                    "fallback_reason": None,
                    "library_params": _gpu_library_params(plan),
                    "probe_notes": probe_notes,
                }
            # Params said GPU but booster reports CPU — keep model, label CPU
            fallback_reason = (
                f"XGBoost reported device={executed} after GPU request"
            )
        except Exception as exc:
            fallback_reason = f"{exc.__class__.__name__}: {str(exc)[:160]}"

    # CPU path (requested, or GPU failed / not actually used)
    model = _build_xgb_regressor(base, _cpu_library_params())
    model.fit(X, y)
    executed = _verify_model_device(model)
    reason = fallback_reason
    if prefer_gpu and not reason:
        reason = "CUDA-compatible GPU not available"
    elif not prefer_gpu and not reason:
        hw_reason = None
        try:
            from chain_replay_ml.training.model_device import detect_gpu_hardware

            hw = detect_gpu_hardware()
            if not hw.get("gpu_detected"):
                hw_reason = "No NVIDIA GPU detected"
        except Exception:
            hw_reason = "GPU probe unavailable"
        reason = hw_reason or "CPU selected"
    return model, {
        "train_device": DEVICE_CPU,
        "executed_device": executed if executed else "cpu",
        "requested_device": "cuda" if prefer_gpu else "cpu",
        "gpu_name": gpu_name,
        "fallback_reason": reason,
        "library_params": _cpu_library_params(),
        "probe_notes": probe_notes,
    }


def _explainer_target(model: Any) -> Any:
    if hasattr(model, "_booster") and getattr(model, "_booster") is not None:
        return model._booster
    getter = getattr(model, "get_booster", None)
    if callable(getter):
        booster = getter()
        if booster is not None:
            return booster
    return model


def compute_shap_mean_abs_gpu_first(
    model: Any,
    X: Any,
    features: list[str],
    *,
    prefer_gpu: bool = True,
) -> tuple[list[dict[str, Any]], str, str]:
    """Mean |SHAP| per feature; prefer GPU paths when the model is CUDA.

    Returns ``(rows, shap_device, error)``.
    """
    feats = [str(f) for f in features]
    if not feats or X is None or len(X) == 0:
        return [], DEVICE_CPU, "empty sample"

    # 1) Native XGBoost pred_contribs — uses booster device (GPU when trained so)
    if prefer_gpu:
        try:
            import xgboost as xgb

            booster = _explainer_target(model)
            dmat = xgb.DMatrix(X, feature_names=list(feats))
            contribs = np.asarray(
                booster.predict(dmat, pred_contribs=True), dtype=float
            )
            if contribs.ndim == 2 and contribs.shape[1] >= len(feats) + 1:
                mean_abs = np.abs(contribs[:, : len(feats)]).mean(axis=0)
                order = np.argsort(-mean_abs)
                rows = []
                for rank, idx in enumerate(order, start=1):
                    rows.append(
                        {
                            "feature": feats[int(idx)],
                            "importance": float(mean_abs[int(idx)]),
                            "rank": rank,
                            "percentile": 100.0
                            * (1.0 - (rank - 1) / max(len(feats), 1)),
                        }
                    )
                executed = _verify_model_device(model)
                device = (
                    DEVICE_GPU
                    if prefer_gpu and str(executed).startswith("cuda")
                    else DEVICE_CPU
                )
                return rows, device, ""
        except Exception:
            pass

        # 2) shap.GPUTreeExplainer when available
        try:
            import shap  # type: ignore

            gpu_cls = getattr(shap, "GPUTreeExplainer", None)
            if gpu_cls is not None:
                explainer = gpu_cls(_explainer_target(model))
                values = explainer.shap_values(X)
                arr = np.asarray(values)
                if arr.ndim == 3:
                    arr = arr[:, :, 0]
                mean_abs = np.mean(np.abs(arr), axis=0)
                order = np.argsort(-mean_abs)
                rows = []
                for rank, idx in enumerate(order, start=1):
                    rows.append(
                        {
                            "feature": feats[int(idx)],
                            "importance": float(mean_abs[int(idx)]),
                            "rank": rank,
                            "percentile": 100.0
                            * (1.0 - (rank - 1) / max(len(feats), 1)),
                        }
                    )
                return rows, DEVICE_GPU, ""
        except Exception:
            pass

    # 3) CPU TreeExplainer
    try:
        import shap  # type: ignore

        explainer = shap.TreeExplainer(_explainer_target(model))
        values = explainer.shap_values(X)
        arr = np.asarray(values)
        if arr.ndim == 3:
            arr = arr[:, :, 0]
        mean_abs = np.mean(np.abs(arr), axis=0)
        order = np.argsort(-mean_abs)
        rows = []
        for rank, idx in enumerate(order, start=1):
            rows.append(
                {
                    "feature": feats[int(idx)],
                    "importance": float(mean_abs[int(idx)]),
                    "rank": rank,
                    "percentile": 100.0
                    * (1.0 - (rank - 1) / max(len(feats), 1)),
                }
            )
        return rows, DEVICE_CPU, ""
    except Exception as exc:
        return [], DEVICE_CPU, str(exc)


def format_device_label(
    train_device: str | None,
    *,
    shap_device: str | None = None,
    fallback_reason: str | None = None,
) -> str:
    """Short UI label, e.g. ``GPU`` or ``CPU (no CUDA)``."""
    td = str(train_device or DEVICE_CPU).upper()
    if td not in (DEVICE_GPU, DEVICE_CPU):
        td = DEVICE_GPU if "cuda" in td.lower() or td == "GPU" else DEVICE_CPU
    if td == DEVICE_GPU:
        if shap_device and str(shap_device).upper() == DEVICE_CPU:
            return "GPU (SHAP CPU)"
        return DEVICE_GPU
    if fallback_reason:
        short = str(fallback_reason).split(":")[0][:40]
        return f"CPU ({short})" if short else DEVICE_CPU
    return DEVICE_CPU


__all__ = [
    "DEVICE_CPU",
    "DEVICE_GPU",
    "compute_shap_mean_abs_gpu_first",
    "fit_xgb_regressor_gpu_first",
    "format_device_label",
    "resolve_experiment_xgb_plan",
]

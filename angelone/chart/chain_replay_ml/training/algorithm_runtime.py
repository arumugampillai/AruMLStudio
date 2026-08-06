"""Algorithm runtime metadata — parameters, device, implementation, prediction latency."""

from __future__ import annotations

import time
from typing import Any

import pandas as pd

from .trainers import algorithm_display_label, normalize_algorithm_id


def detect_gpu_name() -> str | None:
    """Best-effort GPU name for training logs (NVIDIA)."""
    try:
        from .model_device import detect_gpu_hardware

        hw = detect_gpu_hardware()
        if hw.get("gpu_detected") and hw.get("gpu_name"):
            return str(hw["gpu_name"])
    except Exception:
        pass
    try:
        import subprocess

        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            stderr=subprocess.DEVNULL,
            timeout=3,
            text=True,
        )
        name = (out or "").strip().splitlines()[0].strip()
        return name or None
    except Exception:
        return None


def measure_prediction_time_ms(model: Any, X: pd.DataFrame, *, warmup: bool = True) -> float:
    """Wall-clock ms for one full ``model.predict(X)`` pass (validation-sized batch)."""
    if X is None or len(X) == 0:
        return 0.0
    if warmup:
        try:
            model.predict(X.iloc[: min(32, len(X))])
        except Exception:
            pass
    t0 = time.perf_counter()
    model.predict(X)
    return round((time.perf_counter() - t0) * 1000.0, 3)


def format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "—"
    try:
        sec = float(seconds)
    except (TypeError, ValueError):
        return "—"
    if sec < 0:
        sec = 0.0
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = int(sec % 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def device_display(device: str | None) -> str:
    d = str(device or "").strip().lower()
    if d in ("cuda", "gpu", "cuml"):
        return "GPU"
    if d in ("cpu",):
        return "CPU"
    return str(device or "—") or "—"


def build_algorithm_runtime(
    *,
    algorithm: str | None,
    implementation: str,
    device: str,
    algorithm_parameters: dict[str, Any],
    gpu_name: str | None = None,
    fallback_reason: str | None = None,
    training_time_sec: float | None = None,
    prediction_time_ms: float | None = None,
) -> dict[str, Any]:
    algo = normalize_algorithm_id(algorithm)
    used_gpu = str(device).strip().lower() in ("cuda", "gpu", "cuml")
    return {
        "algorithm": algo,
        "algorithm_label": algorithm_display_label(algo),
        "implementation": implementation,
        "device": "cuda" if used_gpu else "cpu",
        "device_label": "GPU" if used_gpu else "CPU",
        "gpu": bool(used_gpu),
        "gpu_name": gpu_name if used_gpu else None,
        "fallback_reason": fallback_reason,
        "algorithm_parameters": dict(algorithm_parameters or {}),
        "training_time_sec": round(float(training_time_sec), 2) if training_time_sec is not None else None,
        "training_time_display": format_duration(training_time_sec),
        "prediction_time_ms": prediction_time_ms,
    }


def format_runtime_log_block(runtime: dict[str, Any] | None) -> list[str]:
    """Human-readable Training Log lines for reproducibility."""
    if not isinstance(runtime, dict) or not runtime:
        return []
    params = runtime.get("algorithm_parameters") or {}
    lines = [
        "--- Algorithm runtime ---",
        f"Algorithm: {runtime.get('algorithm_label') or runtime.get('algorithm') or '—'}",
        f"Implementation: {runtime.get('implementation') or '—'}",
        f"Device: {runtime.get('device_label') or device_display(runtime.get('device'))}",
    ]
    if runtime.get("gpu_name"):
        lines.append(f"GPU: {runtime['gpu_name']}")
    if runtime.get("fallback_reason"):
        lines.append(f"Reason: {runtime['fallback_reason']}")
    if runtime.get("training_time_sec") is not None:
        lines.append(
            f"Training Time: {runtime.get('training_time_display') or format_duration(runtime.get('training_time_sec'))}"
            f" ({runtime['training_time_sec']} s)"
        )
    if runtime.get("prediction_time_ms") is not None:
        lines.append(f"Prediction Time: {runtime['prediction_time_ms']} ms (validation batch)")
    if params:
        lines.append("Parameters:")
        for key, val in params.items():
            lines.append(f"  {key} = {val}")
        if params.get("gpu_params_passed") is not None:
            lines.append(f"  gpu_params_passed = {params.get('gpu_params_passed')}")
        if params.get("executed_device") is not None:
            lines.append(f"  executed_device (verified) = {params.get('executed_device')}")
    return lines


def merge_runtime_into_training_meta(
    training_meta: dict[str, Any] | None,
    runtime: dict[str, Any],
) -> dict[str, Any]:
    meta = dict(training_meta or {})
    meta["algorithm_runtime"] = runtime
    meta["algorithm_parameters"] = dict(runtime.get("algorithm_parameters") or {})
    meta["implementation"] = runtime.get("implementation")
    meta["device"] = runtime.get("device")
    meta["device_label"] = runtime.get("device_label")
    meta["gpu_name"] = runtime.get("gpu_name")
    meta["fallback_reason"] = runtime.get("fallback_reason")
    if runtime.get("prediction_time_ms") is not None:
        meta["prediction_time_ms"] = runtime["prediction_time_ms"]
    return meta


def attach_prediction_latency(
    result: dict[str, Any],
    *,
    val_X: pd.DataFrame,
    features: list[str] | None = None,
) -> dict[str, Any]:
    """Measure validation-batch predict latency and fold into ``training_meta``."""
    from .xgb_trainer import select_feature_columns

    model = result.get("model")
    if model is None:
        return result
    use_features = list(result.get("features") or features or [])
    try:
        X_feat, _ = select_feature_columns(val_X, use_features) if use_features else (val_X, use_features)
        pred_ms = measure_prediction_time_ms(model, X_feat)
    except Exception:
        pred_ms = None

    meta = dict(result.get("training_meta") or {})
    runtime = dict(meta.get("algorithm_runtime") or {})
    if pred_ms is not None:
        meta["prediction_time_ms"] = pred_ms
        if runtime:
            runtime["prediction_time_ms"] = pred_ms
            meta["algorithm_runtime"] = runtime
    result["training_meta"] = meta
    if pred_ms is not None:
        result["prediction_time_ms"] = pred_ms
    return result

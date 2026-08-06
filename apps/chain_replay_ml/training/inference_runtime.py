"""
Verified GPU/CPU inference for prediction builds.

Training ``device=cuda`` does NOT guarantee GPU inference after ``load_model``.
Call ``configure_prediction_model_for_inference`` once at worker startup, then
``batch_predict_day`` for each trading day.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import numpy as np

from .model_runtime import normalize_algorithm


@dataclass(frozen=True)
class InferenceRuntimeInfo:
    """Resolved inference backend after an explicit smoke test."""

    algorithm: str
    device_label: str  # "CUDA" | "CPU"
    device_param: str  # "cuda" | "cpu"
    gpu_requested: bool
    gpu_active: bool
    fallback_reason: str | None = None
    predict_api: str = "predict"  # predict | inplace_predict


def prefer_gpu_inference() -> bool:
    """
    Prefer CUDA for inference unless explicitly forced to CPU.

    Env ``XGB_INFER_DEVICE`` / ``PREDICTION_INFER_DEVICE``: cuda|gpu|cpu
    Default: cuda (with runtime fallback).
    """
    raw = (
        os.environ.get("PREDICTION_INFER_DEVICE")
        or os.environ.get("XGB_INFER_DEVICE")
        or "cuda"
    )
    key = str(raw).strip().lower()
    if key in ("cpu", "host"):
        return False
    return key.startswith("cuda") or key in ("gpu", "auto", "")


def infer_chunk_rows() -> int:
    """
    Max rows per GPU/CPU predict chunk. 0 = try entire day first.
    Env: ``PREDICTION_INFER_CHUNK_ROWS``.
    """
    raw = os.environ.get("PREDICTION_INFER_CHUNK_ROWS", "0")
    try:
        return max(0, int(raw))
    except (TypeError, ValueError):
        return 0


def _is_oom_error(exc: BaseException) -> bool:
    msg = str(exc).lower()
    needles = (
        "out of memory",
        "oom",
        "cudaerror",
        "cudamemcpy",
        "resource exhausted",
        "failed to allocate",
    )
    return any(n in msg for n in needles)


def _set_xgb_device(model: Any, device: str) -> None:
    device = "cuda" if device.startswith("cuda") or device == "gpu" else "cpu"
    params = {"device": device, "tree_method": "hist"}
    if hasattr(model, "set_params"):
        model.set_params(**params)
    booster = getattr(model, "get_booster", lambda: None)()
    if booster is not None and hasattr(booster, "set_param"):
        try:
            booster.set_param({"device": device})
        except Exception:
            pass


def force_cpu_inference(model: Any, algorithm: str | None = None) -> None:
    """Force a side-scorer model to CPU-only inference. Best-effort, never raises.

    A saved XGBoost booster restores whatever ``device`` it was *trained*
    with (often ``cuda``) as soon as it is loaded — regardless of how the
    fresh sklearn wrapper was constructed. Secondary/side-scorer models
    (Triple Barrier, prediction-package ladder members) run inside the same
    worker process as the primary GPU-configured regression model; letting
    them silently re-acquire a CUDA context/DMatrix just to score a handful
    of extra columns risks GPU memory contention with the primary model and
    can crash the whole worker process (a native CUDA fault is not a
    catchable Python exception). Side-scorers are cheap on CPU, so always
    pin them there instead.
    """
    algo = normalize_algorithm(algorithm)
    if algo != "xgboost":
        return
    try:
        _set_xgb_device(model, "cpu")
    except Exception:
        pass


def _xgb_smoke_predict(model: Any, n_features: int) -> str:
    """Return which predict API succeeded on a tiny matrix."""
    X = np.zeros((8, max(1, n_features)), dtype=np.float32)
    booster = getattr(model, "get_booster", lambda: None)()
    if booster is not None and hasattr(booster, "inplace_predict"):
        try:
            out = booster.inplace_predict(X)
            _ = np.asarray(out)
            return "inplace_predict"
        except Exception:
            pass
    out = model.predict(X)
    _ = np.asarray(out)
    return "predict"


def _feature_count_hint(model: Any) -> int:
    names = getattr(model, "feature_names_in_", None)
    if names is not None:
        try:
            return int(len(names))
        except TypeError:
            pass
    booster = getattr(model, "get_booster", lambda: None)()
    if booster is not None:
        try:
            n = int(booster.num_features())
            if n > 0:
                return n
        except Exception:
            pass
    return 8


def configure_prediction_model_for_inference(
    model: Any,
    algorithm: str | None = None,
    *,
    prefer_gpu: bool | None = None,
) -> InferenceRuntimeInfo:
    """
    Force an explicit inference device and smoke-test it once.

    Prefer CUDA when requested; on any failure fall back to CPU cleanly.
    """
    algo = normalize_algorithm(algorithm)
    want_gpu = prefer_gpu_inference() if prefer_gpu is None else bool(prefer_gpu)

    if algo != "xgboost":
        # LGB/CatBoost: leave as loaded; report CPU unless trainer set GPU.
        label = "CPU"
        if algo == "catboost":
            try:
                task = str(getattr(model, "get_param", lambda _k: "")("task_type") or "")
                if task.upper() == "GPU":
                    label = "CUDA"
            except Exception:
                pass
        return InferenceRuntimeInfo(
            algorithm=algo,
            device_label=label,
            device_param="cuda" if label == "CUDA" else "cpu",
            gpu_requested=want_gpu,
            gpu_active=label == "CUDA",
            fallback_reason=None if label == "CUDA" else "non-xgboost uses host path",
            predict_api="predict",
        )

    n_feat = _feature_count_hint(model)
    fallback_reason: str | None = None
    device_param = "cpu"
    api = "predict"

    if want_gpu:
        try:
            _set_xgb_device(model, "cuda")
            api = _xgb_smoke_predict(model, n_feat)
            device_param = "cuda"
        except Exception as exc:
            fallback_reason = f"GPU inference unavailable: {exc}"
            _set_xgb_device(model, "cpu")
            api = _xgb_smoke_predict(model, n_feat)
            device_param = "cpu"
    else:
        _set_xgb_device(model, "cpu")
        api = _xgb_smoke_predict(model, n_feat)
        device_param = "cpu"
        fallback_reason = "PREDICTION_INFER_DEVICE/XGB_INFER_DEVICE=cpu"

    label = "CUDA" if device_param.startswith("cuda") else "CPU"
    return InferenceRuntimeInfo(
        algorithm=algo,
        device_label=label,
        device_param=device_param,
        gpu_requested=want_gpu,
        gpu_active=label == "CUDA",
        fallback_reason=fallback_reason,
        predict_api=api,
    )


def load_prediction_model_for_inference(
    model_path: str,
    algorithm: str | None,
    *,
    prefer_gpu: bool | None = None,
) -> tuple[Any, InferenceRuntimeInfo]:
    from .model_runtime import load_prediction_model

    model = load_prediction_model(model_path, algorithm)
    info = configure_prediction_model_for_inference(
        model, algorithm, prefer_gpu=prefer_gpu
    )
    return model, info


def _predict_chunk(model: Any, X: np.ndarray, *, predict_api: str) -> np.ndarray:
    if predict_api == "inplace_predict":
        booster = getattr(model, "get_booster", lambda: None)()
        if booster is not None and hasattr(booster, "inplace_predict"):
            return np.asarray(booster.inplace_predict(X), dtype=np.float64)
    return np.asarray(model.predict(X), dtype=np.float64)


def batch_predict_day(
    model: Any,
    X_df: Any,
    *,
    info: InferenceRuntimeInfo | None = None,
    chunk_rows: int | None = None,
) -> np.ndarray:
    """
    Predict an entire trading-day feature matrix.

    Prefer one GPU/CPU batch for the whole day. If memory fails, bisect into
    chunks (or honor ``PREDICTION_INFER_CHUNK_ROWS``).
    """
    api = (info.predict_api if info else "predict") or "predict"
    X = np.ascontiguousarray(
        X_df.to_numpy(dtype=np.float32, copy=False)
        if hasattr(X_df, "to_numpy")
        else np.asarray(X_df, dtype=np.float32)
    )
    n = int(X.shape[0])
    if n == 0:
        return np.asarray([], dtype=np.float64)

    preferred = infer_chunk_rows() if chunk_rows is None else max(0, int(chunk_rows))
    if preferred <= 0:
        preferred = n

    def run_chunks(size: int) -> np.ndarray:
        size = max(1, int(size))
        parts: list[np.ndarray] = []
        for start in range(0, n, size):
            parts.append(_predict_chunk(model, X[start : start + size], predict_api=api))
        return np.concatenate(parts, axis=0) if len(parts) > 1 else parts[0]

    size = preferred
    while True:
        try:
            return run_chunks(size)
        except Exception as exc:
            if size <= 1024 or not _is_oom_error(exc):
                # Last resort: CPU path if we were on CUDA
                if info and info.gpu_active:
                    _set_xgb_device(model, "cpu")
                    return run_chunks(min(size, max(1024, n // 8 or 1)))
                raise
            size = max(1024, size // 2)


def format_day_stage_timings(
    timings: dict[str, float],
    *,
    device_label: str = "CPU",
    algorithm: str = "xgboost",
) -> str:
    """Human-readable per-day stage log block (with outcome sub-breakdown)."""
    algo = normalize_algorithm(algorithm)
    predict_name = {
        "xgboost": "XGBoost Predict",
        "lightgbm": "LightGBM Predict",
        "catboost": "CatBoost Predict",
    }.get(algo, "Model Predict")
    load_m = float(timings.get("load_master") or 0.0)
    load_t = float(timings.get("load_timeline") or 0.0)
    prep = float(timings.get("prepare_matrix") or 0.0)
    pred = float(timings.get("predict") or 0.0)
    outcomes = float(timings.get("outcomes") or 0.0)
    write = float(timings.get("sqlite_write") or 0.0)
    total = float(
        timings.get("total")
        or (load_m + load_t + prep + pred + outcomes + write)
    )
    pad = 26
    lines = [
        f"{'Load Master Dataset':<{pad}}: {load_m:.2f} s",
        f"{'Load Tick Timeline':<{pad}}: {load_t:.2f} s",
        f"{'Prepare Feature Matrix':<{pad}}: {prep:.2f} s",
        f"{predict_name:<{pad}}: {pred:.2f} s ({device_label})",
        f"{'CPU Outcome Metrics':<{pad}}: {outcomes:.2f} s",
    ]
    sub = (
        ("outcomes_read", "  Read row (iterrows)"),
        ("outcomes_path", "  Path outcome calculation"),
        ("outcomes_build", "  Build Python dict"),
        ("outcomes_append", "  Append to list"),
        ("outcomes_checkpoint", "  Checkpoint batching"),
    )
    if any(timings.get(k) is not None for k, _ in sub):
        for key, label in sub:
            lines.append(f"{label:<{pad}}: {float(timings.get(key) or 0.0):.2f} s")
    lines.extend(
        [
            f"{'SQLite Write':<{pad}}: {write:.2f} s",
            f"{'Total':<{pad}}: {total:.2f} s",
        ]
    )
    return "\n".join(lines)


def resolve_outcome_profile_rows(
    *,
    row_limit: int | None = None,
    profile_outcome_rows: int | None = None,
) -> int:
    """
    How many per-prediction outcome timings to record.

    Priority:
      1) explicit profile_outcome_rows
      2) PREDICTION_OUTCOME_PROFILE_ROWS env
      3) when Test row_limit is set → min(row_limit, 1000)
      4) else 0 (disabled for full-day builds)
    """
    if profile_outcome_rows is not None:
        try:
            return max(0, int(profile_outcome_rows))
        except (TypeError, ValueError):
            return 0
    raw = os.environ.get("PREDICTION_OUTCOME_PROFILE_ROWS")
    if raw is not None and str(raw).strip() != "":
        try:
            return max(0, int(raw))
        except (TypeError, ValueError):
            return 0
    if row_limit is not None and int(row_limit) > 0:
        # Test / sample builds: profile every processed row (CSV + log summary).
        return int(row_limit)
    return 0


def format_per_prediction_outcome_timings(
    times_ms: list[float],
    *,
    warn_mult: float = 3.0,
    max_log_rows: int = 1000,
) -> str:
    """
    Tabular per-prediction outcome timings.

    Prediction	Outcome Time
    1	3.12 ms
    3	18.45 ms ⚠
    """
    if not times_ms:
        return "(no per-prediction outcome timings)"
    sorted_ms = sorted(float(x) for x in times_ms)
    n = len(sorted_ms)
    median = sorted_ms[n // 2]
    p95 = sorted_ms[min(n - 1, int(0.95 * (n - 1)))]
    mean = sum(sorted_ms) / n
    thresh = max(median * float(warn_mult), p95) if median > 0 else float("inf")
    warn_n = sum(1 for x in times_ms if float(x) > thresh)

    lines = [
        f"Per-prediction outcome timings · n={n} · "
        f"mean={mean:.2f} ms · median={median:.2f} ms · p95={p95:.2f} ms · "
        f"warn>{thresh:.2f} ms ({warn_n})",
        "Prediction\tOutcome Time",
    ]
    limit = min(n, max(0, int(max_log_rows)))
    for i in range(limit):
        ms = float(times_ms[i])
        flag = " WARN" if ms > thresh else ""
        lines.append(f"{i + 1}\t{ms:.2f} ms{flag}")
    if n > limit:
        lines.append(f"… ({n - limit} more rows in CSV)")
    return "\n".join(lines)


def write_outcome_profile_csv(path: str, times_ms: list[float], *, warn_mult: float = 3.0) -> str:
    """Write prediction_idx,outcome_ms,warn CSV. Returns path."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    sorted_ms = sorted(float(x) for x in times_ms) if times_ms else [0.0]
    n = len(sorted_ms)
    median = sorted_ms[n // 2] if times_ms else 0.0
    p95 = sorted_ms[min(n - 1, int(0.95 * (n - 1)))] if times_ms else 0.0
    thresh = max(median * float(warn_mult), p95) if times_ms and median > 0 else float("inf")
    with open(path, "w", encoding="utf-8", newline="") as fh:
        fh.write("prediction,outcome_ms,warn\n")
        for i, ms in enumerate(times_ms, start=1):
            v = float(ms)
            warn = 1 if v > thresh else 0
            fh.write(f"{i},{v:.4f},{warn}\n")
    return path


_PATH_MICRO_LABELS: dict[str, str] = {
    "timeline_lookup": "Timeline array access (cached)",
    "future_window_index": "Future window index (searchsorted)",
    "future_window_slice": "Future window slice",
    "future_tick_scan": "Future tick scan (max/min/counts)",
    "mfe_mae_update": "MFE / MAE update",
    "target_hit_detection": "Target-hit detection",
    "dd_before_target_update": "DD-before-target update",
    "timestamp_tracking": "Timestamp tracking",
    "result_construction": "Result construction",
}


def format_path_outcome_microprofile(
    samples_ms: dict[str, list[float]],
    *,
    n_predictions: int | None = None,
) -> str:
    """
    Average / median / % of total for each operation inside compute_path_outcomes.
    Identifies the single most expensive op and estimates speedup if it were free.
    """
    keys = [k for k in _PATH_MICRO_LABELS if samples_ms.get(k)]
    if not keys:
        # still allow arbitrary keys
        keys = [k for k, v in samples_ms.items() if v]
    if not keys:
        return "(no path-outcome micro-profile samples)"

    n = int(n_predictions or max(len(samples_ms[k]) for k in keys))
    rows: list[tuple[str, float, float, float]] = []
    total_mean = 0.0
    for key in keys:
        vals = [float(x) for x in (samples_ms.get(key) or [])]
        if not vals:
            continue
        vals_sorted = sorted(vals)
        mean = sum(vals) / len(vals)
        median = vals_sorted[len(vals_sorted) // 2]
        total_mean += mean
        label = _PATH_MICRO_LABELS.get(key, key)
        rows.append((label, mean, median, mean))

    if total_mean <= 0:
        return "(path-outcome micro-profile empty)"

    lines = [
        f"Path-outcome micro-profile · first {n} predictions "
        f"(avg total {total_mean:.2f} ms / prediction)",
        f"{'Operation':<42} {'avg_ms':>8} {'med_ms':>8} {'%total':>8}",
        "-" * 70,
    ]
    ranked = sorted(rows, key=lambda r: r[1], reverse=True)
    for label, mean, median, _ in ranked:
        pct = 100.0 * mean / total_mean
        lines.append(f"{label:<42} {mean:8.3f} {median:8.3f} {pct:7.1f}%")

    top_label, top_mean, _, _ = ranked[0]
    top_pct = 100.0 * top_mean / total_mean
    remaining = max(0.0, total_mean - top_mean)
    speedup = (total_mean / remaining) if remaining > 1e-9 else float("inf")
    lines.append("-" * 70)
    lines.append(
        f"MOST EXPENSIVE: {top_label}  ({top_mean:.3f} ms, {top_pct:.1f}% of path time)"
    )
    if remaining > 1e-9:
        lines.append(
            f"If that op were free: ~{remaining:.2f} ms/pred remaining "
            f"(~{speedup:.1f}x vs current {total_mean:.2f} ms)"
        )
    else:
        lines.append("If that op were free: ~0 ms remaining (would dominate entirely)")
    return "\n".join(lines)

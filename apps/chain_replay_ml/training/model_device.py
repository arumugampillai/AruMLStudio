"""Shared GPU/CPU device factory for every trainer.

All training code should resolve device policy through this module so GPU
defaults, probes, warnings, and hard-fail rules live in one place.

Policy
------
* Prefer GPU for XGBoost, LightGBM, CatBoost, Random Forest, Extra Trees.
* Do **not** silently fall back to CPU — always record / warn with a reason.
* LightGBM is special: if GPU was requested and the installed build has no
  GPU support, raise ``LightGBMGpuUnavailableError`` instead of training on CPU.
* After XGBoost trains, verify the booster config actually references CUDA —
  setting ``device=cuda`` alone is not treated as proof.
"""

from __future__ import annotations

import logging
import os
import subprocess
import threading
import warnings
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

from .trainers.base import (
    ALGORITHM_CATBOOST,
    ALGORITHM_EXTRA_TREES,
    ALGORITHM_LIGHTGBM,
    ALGORITHM_RANDOM_FOREST,
    ALGORITHM_XGBOOST,
    algorithm_display_label,
    normalize_algorithm_id,
)

logger = logging.getLogger(__name__)

# Process-wide probe cache — avoid re-probing on every fold / HPO trial.
_PROBE_LOCK = threading.Lock()
_PROBE_CACHE: dict[str, Any] = {}
_STARTUP_LOGGED = False


@contextmanager
def _suppress_c_stderr():
    """Temporarily suppress C-level stderr (fd 2) during native GPU capability probes.
    
    Prevents third-party C++ libraries (e.g. LightGBM C++ CUDA/GPU probe) from
    printing spurious '[Fatal] CUDA/GPU Tree Learner was not enabled' messages
    to the console during routine capability discovery. Genuine training errors
    outside of probes remain fully visible and unsuppressed.
    """
    try:
        devnull_fd = os.open(os.devnull, os.O_WRONLY)
        old_stderr_fd = os.dup(2)
        os.dup2(devnull_fd, 2)
        os.close(devnull_fd)
        try:
            yield
        finally:
            os.dup2(old_stderr_fd, 2)
            os.close(old_stderr_fd)
    except Exception:
        yield


class LightGBMGpuUnavailableError(RuntimeError):
    """Raised when LightGBM GPU was requested but the install cannot use a GPU."""


@dataclass(frozen=True)
class DevicePlan:
    """Resolved training device for one algorithm invocation."""

    algorithm: str
    prefer_gpu: bool
    use_gpu: bool
    device: str  # "cuda" | "cpu"
    device_label: str  # "GPU" | "CPU"
    requested: str
    library_params: dict[str, Any] = field(default_factory=dict)
    fallback_reason: str | None = None
    gpu_name: str | None = None
    library_version: str | None = None
    probe_notes: tuple[str, ...] = ()
    hard_fail_on_gpu_miss: bool = False

    def warning_lines(self) -> list[str]:
        if self.use_gpu or not self.fallback_reason:
            return []
        return [
            f"WARNING: {algorithm_display_label(self.algorithm)} training on CPU — "
            f"{self.fallback_reason}"
        ]

    def log_lines(self) -> list[str]:
        lines = [
            f"Algorithm: {algorithm_display_label(self.algorithm)}",
            f"Training device selected: {self.device_label}",
            f"Requested: {self.requested}",
        ]
        if self.gpu_name:
            lines.append(f"GPU model: {self.gpu_name}")
        if self.library_version:
            lines.append(f"Library version: {self.library_version}")
        if self.library_params:
            lines.append(f"GPU/device params: {self.library_params}")
        if self.fallback_reason and not self.use_gpu:
            lines.append(f"CPU reason: {self.fallback_reason}")
        for note in self.probe_notes:
            lines.append(f"Probe: {note}")
        return lines


def _env_or(params: dict[str, Any], *keys: str, default: str = "cuda") -> str:
    for key in keys:
        if key in params and params.get(key) not in (None, ""):
            return str(params.get(key)).strip()
        env_key = key.upper()
        if os.environ.get(env_key):
            return str(os.environ.get(env_key)).strip()
    # Common shared override.
    if os.environ.get("ML_TRAIN_DEVICE"):
        return str(os.environ.get("ML_TRAIN_DEVICE")).strip()
    return default


def _wants_gpu(raw: str) -> bool:
    text = str(raw or "").strip().lower()
    if text in ("cpu", "none", "off", "disable", "disabled"):
        return False
    if text in ("cuda", "gpu", "cuml", "auto", "default") or text.startswith("cuda"):
        return True
    return True


def detect_gpu_hardware() -> dict[str, Any]:
    """Return NVIDIA GPU presence + name (cached)."""
    with _PROBE_LOCK:
        cached = _PROBE_CACHE.get("hardware")
        if isinstance(cached, dict):
            return dict(cached)

    info: dict[str, Any] = {
        "gpu_detected": False,
        "gpu_name": None,
        "driver_version": None,
        "memory_total": None,
        "source": None,
        "error": None,
    }
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader",
            ],
            stderr=subprocess.DEVNULL,
            timeout=5,
            text=True,
        )
        line = (out or "").strip().splitlines()[0].strip()
        parts = [p.strip() for p in line.split(",")]
        if parts:
            info["gpu_detected"] = True
            info["gpu_name"] = parts[0] or None
            info["driver_version"] = parts[1] if len(parts) > 1 else None
            info["memory_total"] = parts[2] if len(parts) > 2 else None
            info["source"] = "nvidia-smi"
    except Exception as exc:
        info["error"] = f"nvidia-smi unavailable ({exc.__class__.__name__})"
        try:
            import cupy  # type: ignore

            props = cupy.cuda.runtime.getDeviceProperties(0)
            raw = props.get("name") if isinstance(props, dict) else getattr(props, "name", None)
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8", errors="ignore")
            if raw:
                info["gpu_detected"] = True
                info["gpu_name"] = str(raw).strip()
                info["source"] = "cupy"
                info["error"] = None
        except Exception:
            pass

    with _PROBE_LOCK:
        _PROBE_CACHE["hardware"] = dict(info)
    return info


def _library_version(module_name: str) -> str | None:
    try:
        mod = __import__(module_name)
        return str(getattr(mod, "__version__", None) or "?")
    except Exception:
        return None


def probe_xgboost_gpu() -> dict[str, Any]:
    """Tiny XGBoost CUDA train to verify GPU execution at runtime."""
    with _PROBE_LOCK:
        cached = _PROBE_CACHE.get("xgboost_gpu")
        if isinstance(cached, dict):
            return dict(cached)

    result: dict[str, Any] = {
        "supported": False,
        "version": _library_version("xgboost"),
        "detail": None,
        "executed_device": None,
    }
    try:
        import numpy as np
        import xgboost as xgb

        X = np.random.randn(64, 4).astype("float32")
        y = X[:, 0].astype("float32")
        dtrain = xgb.DMatrix(X, label=y)
        booster = xgb.train(
            {
                "objective": "reg:squarederror",
                "tree_method": "hist",
                "device": "cuda",
                "verbosity": 0,
            },
            dtrain,
            num_boost_round=2,
        )
        executed = verify_xgboost_booster_device(booster)
        result["executed_device"] = executed
        result["supported"] = executed.startswith("cuda")
        result["detail"] = (
            f"probe trained on {executed}"
            if result["supported"]
            else f"probe requested cuda but booster reported {executed}"
        )
    except Exception as exc:
        result["detail"] = f"{exc.__class__.__name__}: {exc}"

    with _PROBE_LOCK:
        _PROBE_CACHE["xgboost_gpu"] = dict(result)
    return result


def probe_lightgbm_gpu() -> dict[str, Any]:
    """Detect whether the installed LightGBM build can train on GPU/CUDA."""
    with _PROBE_LOCK:
        cached = _PROBE_CACHE.get("lightgbm_gpu")
        if isinstance(cached, dict):
            return dict(cached)

    result: dict[str, Any] = {
        "supported": False,
        "version": _library_version("lightgbm"),
        "backend": None,
        "detail": None,
        "installed": False,
    }
    try:
        import lightgbm as lgb  # noqa: F401
        import numpy as np

        result["installed"] = True
        result["version"] = _library_version("lightgbm")
        X = np.random.randn(64, 4).astype("float32")
        y = (X[:, 0] > 0).astype("float32")
        dataset = lgb.Dataset(X, label=y)
        errors: list[str] = []
        with _suppress_c_stderr():
            for device in ("cuda", "gpu"):
                try:
                    lgb.train(
                        {
                            "objective": "regression",
                            "device": device,
                            "verbosity": -1,
                            "num_leaves": 7,
                            "gpu_use_dp": False,
                        },
                        dataset,
                        num_boost_round=2,
                    )
                    result["supported"] = True
                    result["backend"] = device
                    result["detail"] = f"probe succeeded with device={device}"
                    break
                except Exception as exc:
                    errors.append(f"device={device}: {exc.__class__.__name__}: {exc}")
        if not result["supported"]:
            result["detail"] = "Installed LightGBM build has no CUDA/OpenCL GPU learner (CPU/OpenMP only)"
    except ImportError:
        result["detail"] = "lightgbm is not installed"
    except Exception as exc:
        result["detail"] = f"{exc.__class__.__name__}: {exc}"

    with _PROBE_LOCK:
        _PROBE_CACHE["lightgbm_gpu"] = dict(result)
    return result


def probe_catboost_gpu() -> dict[str, Any]:
    with _PROBE_LOCK:
        cached = _PROBE_CACHE.get("catboost_gpu")
        if isinstance(cached, dict):
            return dict(cached)

    result: dict[str, Any] = {
        "supported": False,
        "version": _library_version("catboost"),
        "detail": None,
        "installed": False,
    }
    try:
        import numpy as np
        from catboost import CatBoostRegressor

        result["installed"] = True
        result["version"] = _library_version("catboost")
        X = np.random.randn(64, 4).astype("float32")
        y = X[:, 0].astype("float32")
        model = CatBoostRegressor(iterations=2, depth=2, task_type="GPU", verbose=False)
        model.fit(X, y)
        result["supported"] = True
        result["detail"] = "probe succeeded with task_type=GPU"
    except ImportError:
        result["detail"] = "catboost is not installed"
    except Exception as exc:
        result["detail"] = f"{exc.__class__.__name__}: {exc}"

    with _PROBE_LOCK:
        _PROBE_CACHE["catboost_gpu"] = dict(result)
    return result


def probe_cuml() -> dict[str, Any]:
    with _PROBE_LOCK:
        cached = _PROBE_CACHE.get("cuml")
        if isinstance(cached, dict):
            return dict(cached)
    result: dict[str, Any] = {
        "supported": False,
        "version": _library_version("cuml"),
        "detail": None,
    }
    try:
        import cuml  # noqa: F401

        result["supported"] = True
        result["version"] = _library_version("cuml")
        result["detail"] = "cuML importable"
    except Exception as exc:
        result["detail"] = f"cuML unavailable ({exc.__class__.__name__})"
    with _PROBE_LOCK:
        _PROBE_CACHE["cuml"] = dict(result)
    return result


def verify_xgboost_booster_device(booster: Any) -> str:
    """Return actual device string from a fitted XGBoost booster config."""
    try:
        cfg = booster.save_config()
    except Exception:
        return "unknown"
    text = str(cfg or "")
    lowered = text.lower()
    if "cuda:" in lowered or '"device": "cuda' in lowered or '"device":"cuda' in lowered:
        # Prefer concrete cuda:N if present.
        import re

        match = re.search(r'"device(?:_name)?"\s*:\s*"(cuda[^"]*)"', text, flags=re.I)
        if match:
            return match.group(1)
        return "cuda"
    if '"device": "cpu"' in lowered or '"device":"cpu"' in lowered:
        return "cpu"
    return "unknown"


def get_startup_diagnostics() -> dict[str, Any]:
    """Full GPU / library snapshot for training startup logs."""
    hw = detect_gpu_hardware()
    xgb = probe_xgboost_gpu() if _library_version("xgboost") else {
        "supported": False, "version": None, "detail": "xgboost not installed",
    }
    # LightGBM / CatBoost probes are expensive and may be missing — still report.
    lgb = probe_lightgbm_gpu()
    cat = probe_catboost_gpu()
    cuml = probe_cuml()
    return {
        "gpu_detected": bool(hw.get("gpu_detected")),
        "gpu_name": hw.get("gpu_name"),
        "driver_version": hw.get("driver_version"),
        "memory_total": hw.get("memory_total"),
        "hardware_source": hw.get("source"),
        "hardware_error": hw.get("error"),
        "libraries": {
            "xgboost": xgb,
            "lightgbm": lgb,
            "catboost": cat,
            "cuml": cuml,
            "sklearn": {"version": _library_version("sklearn")},
        },
    }


def format_startup_diagnostics(diag: dict[str, Any] | None = None) -> list[str]:
    doc = diag or get_startup_diagnostics()
    libs = doc.get("libraries") or {}
    lines = [
        "=== GPU / training device diagnostics ===",
        f"GPU detected: {'Yes' if doc.get('gpu_detected') else 'No'}",
    ]
    if doc.get("gpu_name"):
        lines.append(f"GPU model: {doc['gpu_name']}")
    if doc.get("driver_version"):
        lines.append(f"Driver: {doc['driver_version']}")
    if doc.get("memory_total"):
        lines.append(f"VRAM: {doc['memory_total']}")
    if doc.get("hardware_error") and not doc.get("gpu_detected"):
        lines.append(f"Hardware note: {doc['hardware_error']}")

    def _lib_line(name: str, payload: dict[str, Any] | None) -> str:
        payload = payload or {}
        ver = payload.get("version") or "—"
        if name == "sklearn":
            return f"  {name}: {ver}"
        supported = payload.get("supported")
        detail = payload.get("detail") or ""
        flag = "GPU OK" if supported else "GPU NO"
        return f"  {name}: {ver} · {flag}" + (f" — {detail}" if detail and not supported else "")

    lines.append("GPU-enabled library status:")
    for key in ("xgboost", "lightgbm", "catboost", "cuml", "sklearn"):
        lines.append(_lib_line(key, libs.get(key) if isinstance(libs.get(key), dict) else {}))

    try:
        from .algorithm_capabilities import format_algorithm_support_report

        lines.append("")
        lines.extend(format_algorithm_support_report())
    except Exception as exc:
        lines.append(f"Algorithm Support: unavailable ({exc.__class__.__name__}: {exc})")

    lines.append("=== end diagnostics ===")
    return lines


def emit_startup_diagnostics_once(
    log_fn: Any | None = None,
) -> dict[str, Any]:
    """Log diagnostics once per process. ``log_fn`` receives one string per line."""
    global _STARTUP_LOGGED
    diag = get_startup_diagnostics()
    lines = format_startup_diagnostics(diag)
    if not _STARTUP_LOGGED:
        _STARTUP_LOGGED = True
        for line in lines:
            if log_fn is not None:
                try:
                    log_fn(line)
                except Exception:
                    logger.info(line)
            else:
                logger.info(line)
    return diag


def resolve_training_device(
    algorithm: str | None,
    parameters: dict[str, Any] | None = None,
    *,
    allow_cpu_fallback: bool | None = None,
) -> DevicePlan:
    """Single entry point: decide GPU vs CPU for ``algorithm``.

    LightGBM never silently falls back when GPU is requested — it raises
    ``LightGBMGpuUnavailableError`` unless ``allow_cpu_fallback=True`` was
    passed explicitly (tests / emergency override).
    """
    params = dict(parameters or {})
    algo = normalize_algorithm_id(algorithm)
    hw = detect_gpu_hardware()
    gpu_name = hw.get("gpu_name") if hw.get("gpu_detected") else None

    if algo == ALGORITHM_XGBOOST:
        requested = _env_or(
            params,
            "xgb_device",
            "device",
            default=os.environ.get("XGB_TRAIN_DEVICE") or "cuda",
        )
        prefer = _wants_gpu(requested)
        version = _library_version("xgboost")
        notes: list[str] = []
        if not prefer:
            return DevicePlan(
                algorithm=algo,
                prefer_gpu=False,
                use_gpu=False,
                device="cpu",
                device_label="CPU",
                requested=requested,
                library_params={
                    "tree_method": "hist",
                    "device": "cpu",
                    "predictor": "cpu_predictor",
                },
                fallback_reason="CPU requested via xgb_device / XGB_TRAIN_DEVICE",
                gpu_name=None,
                library_version=version,
            )
        probe = probe_xgboost_gpu()
        notes.append(str(probe.get("detail") or ""))
        if probe.get("supported"):
            return DevicePlan(
                algorithm=algo,
                prefer_gpu=True,
                use_gpu=True,
                device="cuda",
                device_label="GPU",
                requested=requested,
                library_params={
                    "tree_method": "hist",
                    "device": "cuda",
                    "predictor": "gpu_predictor",
                },
                gpu_name=gpu_name,
                library_version=version,
                probe_notes=tuple(n for n in notes if n),
            )
        reason = (
            "XGBoost GPU probe failed — "
            f"{probe.get('detail') or 'unknown'}. Will attempt GPU train and "
            "fall back to CPU only with an explicit warning if CUDA errors."
        )
        # Still *attempt* GPU at train time; trainer may catch XGBoostError.
        return DevicePlan(
            algorithm=algo,
            prefer_gpu=True,
            use_gpu=True,
            device="cuda",
            device_label="GPU",
            requested=requested,
            library_params={
                "tree_method": "hist",
                "device": "cuda",
                "predictor": "gpu_predictor",
            },
            fallback_reason=None,
            gpu_name=gpu_name,
            library_version=version,
            probe_notes=(reason,),
        )

    if algo == ALGORITHM_LIGHTGBM:
        requested = _env_or(
            params,
            "lgb_device",
            "device",
            default=(
                os.environ.get("LGB_TRAIN_DEVICE")
                or os.environ.get("XGB_TRAIN_DEVICE")
                or "cuda"
            ),
        )
        prefer = _wants_gpu(requested)
        version = _library_version("lightgbm")
        fallback_allowed = (
            bool(allow_cpu_fallback)
            if allow_cpu_fallback is not None
            else False  # LightGBM: no silent CPU when GPU requested
        )
        if not prefer:
            return DevicePlan(
                algorithm=algo,
                prefer_gpu=False,
                use_gpu=False,
                device="cpu",
                device_label="CPU",
                requested=requested,
                library_params={"num_threads": -1},
                fallback_reason="CPU requested via lgb_device / LGB_TRAIN_DEVICE",
                library_version=version,
            )
        probe = probe_lightgbm_gpu()
        if probe.get("supported"):
            backend = str(probe.get("backend") or "cuda")
            lib_params: dict[str, Any] = {"device": backend, "gpu_use_dp": False}
            return DevicePlan(
                algorithm=algo,
                prefer_gpu=True,
                use_gpu=True,
                device="cuda",
                device_label="GPU",
                requested=requested,
                library_params=lib_params,
                gpu_name=gpu_name,
                library_version=version,
                probe_notes=(str(probe.get("detail") or ""),),
                hard_fail_on_gpu_miss=True,
            )
        detail = str(probe.get("detail") or "Installed LightGBM build has no CUDA/OpenCL GPU learner (CPU/OpenMP active)")
        return DevicePlan(
            algorithm=algo,
            prefer_gpu=True,
            use_gpu=False,
            device="cpu",
            device_label="CPU",
            requested=requested,
            library_params={"num_threads": -1},
            fallback_reason=detail,
            library_version=version,
            probe_notes=(detail,),
            hard_fail_on_gpu_miss=False,
        )

    if algo == ALGORITHM_CATBOOST:
        requested = _env_or(
            params,
            "catboost_device",
            "device",
            default=os.environ.get("CATBOOST_DEVICE") or "cuda",
        )
        prefer = _wants_gpu(requested)
        version = _library_version("catboost")
        if not prefer:
            return DevicePlan(
                algorithm=algo,
                prefer_gpu=False,
                use_gpu=False,
                device="cpu",
                device_label="CPU",
                requested=requested,
                library_params={"task_type": "CPU"},
                fallback_reason="CPU requested via catboost_device / CATBOOST_DEVICE",
                library_version=version,
            )
        probe = probe_catboost_gpu()
        if probe.get("supported"):
            return DevicePlan(
                algorithm=algo,
                prefer_gpu=True,
                use_gpu=True,
                device="cuda",
                device_label="GPU",
                requested=requested,
                library_params={"task_type": "GPU"},
                gpu_name=gpu_name,
                library_version=version,
                probe_notes=(str(probe.get("detail") or ""),),
            )
        # Attempt GPU at fit-time anyway; trainer warns + falls back on failure.
        return DevicePlan(
            algorithm=algo,
            prefer_gpu=True,
            use_gpu=True,
            device="cuda",
            device_label="GPU",
            requested=requested,
            library_params={"task_type": "GPU"},
            gpu_name=gpu_name,
            library_version=version,
            probe_notes=(
                f"CatBoost GPU probe failed ({probe.get('detail')}); "
                "will attempt GPU fit and warn if falling back to CPU",
            ),
        )

    if algo in (ALGORITHM_RANDOM_FOREST, ALGORITHM_EXTRA_TREES):
        key = "rf_device" if algo == ALGORITHM_RANDOM_FOREST else "et_device"
        env_key = "RF_TRAIN_DEVICE" if algo == ALGORITHM_RANDOM_FOREST else "ET_TRAIN_DEVICE"
        requested = _env_or(
            params,
            key,
            "rf_device",
            "device",
            default=os.environ.get(env_key) or os.environ.get("RF_TRAIN_DEVICE") or "cuda",
        )
        prefer = _wants_gpu(requested)
        cuml = probe_cuml()
        if prefer and cuml.get("supported"):
            return DevicePlan(
                algorithm=algo,
                prefer_gpu=True,
                use_gpu=True,
                device="cuda",
                device_label="GPU",
                requested=requested,
                library_params={"backend": "cuml"},
                gpu_name=gpu_name,
                library_version=cuml.get("version"),
                probe_notes=(str(cuml.get("detail") or ""),),
            )
        reason = (
            "CPU requested"
            if not prefer
            else f"cuML unavailable ({cuml.get('detail') or 'not installed'}); using scikit-learn CPU"
        )
        if prefer:
            warnings.warn(reason, UserWarning, stacklevel=2)
            logger.warning(reason)
        return DevicePlan(
            algorithm=algo,
            prefer_gpu=prefer,
            use_gpu=False,
            device="cpu",
            device_label="CPU",
            requested=requested,
            library_params={"backend": "sklearn"},
            fallback_reason=reason,
            library_version=_library_version("sklearn"),
        )

    # Unknown → treat like XGBoost defaults (registry already normalizes to xgb).
    return resolve_training_device(ALGORITHM_XGBOOST, params, allow_cpu_fallback=allow_cpu_fallback)


def announce_device_plan(plan: DevicePlan, *, log_fn: Any | None = None) -> None:
    """Emit device selection lines (and CPU warnings) to logger / training log."""
    for line in plan.log_lines():
        msg = f"[device] {line}"
        if log_fn is not None:
            try:
                log_fn(msg)
            except Exception:
                logger.info(msg)
        else:
            logger.info(msg)
    for warn in plan.warning_lines():
        warnings.warn(warn, UserWarning, stacklevel=2)
        logger.warning(warn)
        if log_fn is not None:
            try:
                log_fn(warn)
            except Exception:
                pass


def clear_device_probe_cache() -> None:
    """Test helper — reset cached probes."""
    global _STARTUP_LOGGED
    with _PROBE_LOCK:
        _PROBE_CACHE.clear()
    _STARTUP_LOGGED = False

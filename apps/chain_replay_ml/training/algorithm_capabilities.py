"""Central algorithm capability registry.

UI, config validation, and trainers all consult this module — never duplicate
``if algorithm == "xgboost"`` special cases for prediction-type support.

Static ``supports_*`` flags describe what each algorithm *can* do when the
library is installed correctly. Runtime GPU availability is probed via
``model_device`` and folded into diagnostics / device selection separately.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .trainers.base import (
    ALGORITHM_CATBOOST,
    ALGORITHM_EXTRA_TREES,
    ALGORITHM_LABELS,
    ALGORITHM_LIGHTGBM,
    ALGORITHM_RANDOM_FOREST,
    ALGORITHM_XGBOOST,
    algorithm_display_label,
    normalize_algorithm_id,
    supported_algorithms,
)


@dataclass(frozen=True)
class AlgorithmCapabilities:
    algorithm_id: str
    label: str
    supports_regression: bool = True
    supports_binary_classification: bool = True
    supports_multiclass: bool = True
    # Declared library ability (not current install state).
    supports_gpu_training: bool = True
    supports_gpu_prediction: bool = True
    # Human-readable GPU backend name for diagnostics.
    gpu_backend: str = "native"
    notes: str = ""


# Declared capabilities — independent of whether the GPU library is installed.
_CAPABILITIES: dict[str, AlgorithmCapabilities] = {
    ALGORITHM_XGBOOST: AlgorithmCapabilities(
        algorithm_id=ALGORITHM_XGBOOST,
        label=ALGORITHM_LABELS[ALGORITHM_XGBOOST],
        supports_gpu_training=True,
        supports_gpu_prediction=True,
        gpu_backend="xgboost cuda",
    ),
    ALGORITHM_LIGHTGBM: AlgorithmCapabilities(
        algorithm_id=ALGORITHM_LIGHTGBM,
        label=ALGORITHM_LABELS[ALGORITHM_LIGHTGBM],
        supports_gpu_training=True,
        supports_gpu_prediction=True,
        gpu_backend="lightgbm cuda/gpu build",
        notes="GPU requires a CUDA/OpenCL-enabled LightGBM build",
    ),
    ALGORITHM_CATBOOST: AlgorithmCapabilities(
        algorithm_id=ALGORITHM_CATBOOST,
        label=ALGORITHM_LABELS[ALGORITHM_CATBOOST],
        supports_gpu_training=True,
        supports_gpu_prediction=True,
        gpu_backend="catboost task_type=GPU",
    ),
    ALGORITHM_RANDOM_FOREST: AlgorithmCapabilities(
        algorithm_id=ALGORITHM_RANDOM_FOREST,
        label=ALGORITHM_LABELS[ALGORITHM_RANDOM_FOREST],
        supports_gpu_training=True,
        supports_gpu_prediction=True,
        gpu_backend="cuML",
        notes="GPU via cuML; sklearn CPU fallback when cuML missing",
    ),
    ALGORITHM_EXTRA_TREES: AlgorithmCapabilities(
        algorithm_id=ALGORITHM_EXTRA_TREES,
        label=ALGORITHM_LABELS[ALGORITHM_EXTRA_TREES],
        supports_gpu_training=True,
        supports_gpu_prediction=True,
        gpu_backend="cuML",
        notes="GPU via cuML; sklearn CPU fallback when cuML missing",
    ),
}


def get_algorithm_capabilities(algorithm: str | None) -> AlgorithmCapabilities:
    algo = normalize_algorithm_id(algorithm)
    caps = _CAPABILITIES.get(algo)
    if caps is not None:
        return caps
    return AlgorithmCapabilities(
        algorithm_id=algo,
        label=algorithm_display_label(algo),
        supports_regression=True,
        supports_binary_classification=False,
        supports_multiclass=False,
        supports_gpu_training=False,
        supports_gpu_prediction=False,
        gpu_backend="none",
        notes="Unknown algorithm — conservative defaults",
    )


def list_algorithm_capabilities() -> list[AlgorithmCapabilities]:
    return [get_algorithm_capabilities(aid) for aid, _label in supported_algorithms()]


def normalize_prediction_type(prediction_type: str | None) -> str:
    pred = str(prediction_type or "regression").strip().lower()
    if pred in ("binary", "classification", "multiclass", "regression"):
        return pred
    return "regression"


def algorithm_supports_prediction_type(
    algorithm: str | None,
    prediction_type: str | None,
) -> bool:
    caps = get_algorithm_capabilities(algorithm)
    pred = normalize_prediction_type(prediction_type)
    if pred == "regression":
        return bool(caps.supports_regression)
    if pred == "binary":
        return bool(caps.supports_binary_classification)
    if pred in ("classification", "multiclass"):
        # "classification" covers ORMP direction / general class labels.
        return bool(caps.supports_multiclass or caps.supports_binary_classification)
    return False


def assert_algorithm_supports_prediction_type(
    algorithm: str | None,
    prediction_type: str | None,
) -> None:
    if algorithm_supports_prediction_type(algorithm, prediction_type):
        return
    caps = get_algorithm_capabilities(algorithm)
    pred = normalize_prediction_type(prediction_type)
    raise ValueError(
        f"{caps.label} does not support prediction type '{pred}'. "
        f"Capabilities: regression={caps.supports_regression}, "
        f"binary={caps.supports_binary_classification}, "
        f"multiclass={caps.supports_multiclass}."
    )


def algorithms_for_prediction_type(prediction_type: str | None) -> list[tuple[str, str]]:
    """Return ``(algorithm_id, label)`` pairs that support ``prediction_type``."""
    return [
        (aid, label)
        for aid, label in supported_algorithms()
        if algorithm_supports_prediction_type(aid, prediction_type)
    ]


def runtime_gpu_status(algorithm: str | None) -> dict[str, Any]:
    """Detect whether GPU training is actually available for ``algorithm`` now."""
    from .model_device import (
        detect_gpu_hardware,
        probe_catboost_gpu,
        probe_cuml,
        probe_lightgbm_gpu,
        probe_xgboost_gpu,
    )

    algo = normalize_algorithm_id(algorithm)
    caps = get_algorithm_capabilities(algo)
    hw = detect_gpu_hardware()
    out: dict[str, Any] = {
        "algorithm": algo,
        "label": caps.label,
        "declared_gpu_training": caps.supports_gpu_training,
        "declared_gpu_prediction": caps.supports_gpu_prediction,
        "gpu_backend": caps.gpu_backend,
        "hardware_gpu_detected": bool(hw.get("gpu_detected")),
        "gpu_name": hw.get("gpu_name"),
        "gpu_training_available": False,
        "reason": None,
    }
    if not caps.supports_gpu_training:
        out["reason"] = "algorithm has no GPU training path"
        return out
    if not hw.get("gpu_detected"):
        out["reason"] = hw.get("error") or "no NVIDIA GPU detected"
        return out

    if algo == ALGORITHM_XGBOOST:
        probe = probe_xgboost_gpu()
        out["gpu_training_available"] = bool(probe.get("supported"))
        out["reason"] = None if probe.get("supported") else (probe.get("detail") or "XGBoost GPU probe failed")
        out["library_version"] = probe.get("version")
        return out
    if algo == ALGORITHM_LIGHTGBM:
        probe = probe_lightgbm_gpu()
        out["gpu_training_available"] = bool(probe.get("supported"))
        out["reason"] = None if probe.get("supported") else (probe.get("detail") or "LightGBM GPU unavailable")
        out["library_version"] = probe.get("version")
        return out
    if algo == ALGORITHM_CATBOOST:
        probe = probe_catboost_gpu()
        out["gpu_training_available"] = bool(probe.get("supported"))
        out["reason"] = None if probe.get("supported") else (probe.get("detail") or "CatBoost GPU unavailable")
        out["library_version"] = probe.get("version")
        return out
    if algo in (ALGORITHM_RANDOM_FOREST, ALGORITHM_EXTRA_TREES):
        probe = probe_cuml()
        out["gpu_training_available"] = bool(probe.get("supported"))
        out["reason"] = None if probe.get("supported") else (probe.get("detail") or "cuML unavailable")
        out["library_version"] = probe.get("version")
        return out

    out["reason"] = "no GPU probe registered for algorithm"
    return out


def format_algorithm_support_report() -> list[str]:
    """Human-readable Algorithm Support block for startup diagnostics."""
    lines = ["Algorithm Support"]
    for caps in list_algorithm_capabilities():
        gpu = runtime_gpu_status(caps.algorithm_id)
        gpu_flag = "✓" if gpu.get("gpu_training_available") else "✗"
        gpu_note = ""
        if not gpu.get("gpu_training_available"):
            why = gpu.get("reason") or "unavailable"
            gpu_note = f" ({why})"
        lines.append(caps.label)
        lines.append(f"  Regression {'✓' if caps.supports_regression else '✗'}")
        lines.append(f"  Binary {'✓' if caps.supports_binary_classification else '✗'}")
        lines.append(f"  Multiclass {'✓' if caps.supports_multiclass else '✗'}")
        lines.append(f"  GPU {gpu_flag}{gpu_note}")
    return lines


def capability_matrix_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for caps in list_algorithm_capabilities():
        gpu = runtime_gpu_status(caps.algorithm_id)
        rows.append({
            "algorithm": caps.algorithm_id,
            "label": caps.label,
            "regression": caps.supports_regression,
            "binary": caps.supports_binary_classification,
            "multiclass": caps.supports_multiclass,
            "gpu_train_declared": caps.supports_gpu_training,
            "gpu_predict_declared": caps.supports_gpu_prediction,
            "gpu_train_available": bool(gpu.get("gpu_training_available")),
            "gpu_reason": gpu.get("reason"),
            "gpu_backend": caps.gpu_backend,
            "notes": caps.notes,
        })
    return rows

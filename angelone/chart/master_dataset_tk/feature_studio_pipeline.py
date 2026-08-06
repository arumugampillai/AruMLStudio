"""Feature Studio unified load/compute pipeline (no Tk).

Owns sequential Load Artifacts / Compute for Imp → Dist → Drift → Diagnostics → Planner.
Studio Compare is intentionally excluded from this pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

ProgressCb = Callable[[dict[str, Any]], None]

# Dependency order; Experiment Planner must be last.
PIPELINE_ORDER: tuple[str, ...] = (
    "importance",
    "distribution",
    "drift",
    "diagnostics",
    "planner",
)

STUDIO_LABELS: dict[str, str] = {
    "importance": "Importance",
    "distribution": "Distribution",
    "drift": "Drift",
    "diagnostics": "Diagnostics",
    "planner": "Experiment Planner",
}

# Matches recommendation_engine.compute: need ≥1 of Imp/Dist/Drift; diagnostics optional.
PLANNER_SOFT_INPUTS: tuple[str, ...] = ("importance", "distribution", "drift")

PLANNER_SKIP_MSG = "Planner skipped — required artifacts unavailable."


@dataclass
class StudioStageResult:
    key: str
    available: bool = False
    error: str | None = None
    skipped: bool = False
    skip_reason: str | None = None
    payload: dict[str, Any] | None = None


@dataclass
class PipelineResult:
    model_name: str
    stages: dict[str, StudioStageResult] = field(default_factory=dict)

    def status_marks(self) -> str:
        """Subtle readiness marks: Importance ✓ Distribution ✗ …"""
        parts: list[str] = []
        for key in PIPELINE_ORDER:
            label = STUDIO_LABELS[key]
            stage = self.stages.get(key)
            if stage is None:
                parts.append(f"{label} —")
            elif stage.skipped:
                parts.append(f"{label} skipped")
            elif stage.available:
                parts.append(f"{label} ✓")
            elif stage.error:
                parts.append(f"{label} ✗")
            else:
                parts.append(f"{label} ✗")
        return " · ".join(parts)

    def errors_summary(self) -> str:
        bits: list[str] = []
        for key in PIPELINE_ORDER:
            stage = self.stages.get(key)
            if stage is None:
                continue
            if stage.skipped and stage.skip_reason:
                bits.append(stage.skip_reason)
            elif stage.error:
                bits.append(f"{STUDIO_LABELS[key]}: {stage.error}")
        return "; ".join(bits)


def planner_inputs_available(available: dict[str, bool]) -> bool:
    """True when Recommendation Engine has enough inputs to run."""
    return any(bool(available.get(k)) for k in PLANNER_SOFT_INPUTS)


def _load_importance(package_dir: str) -> dict[str, Any] | None:
    from chain_replay_ml.feature_importance_studio.writer import load_studio_artifacts

    return load_studio_artifacts(package_dir)


def _load_distribution(package_dir: str) -> dict[str, Any] | None:
    from chain_replay_ml.feature_distribution_studio.writer import load_studio_artifacts

    return load_studio_artifacts(package_dir)


def _load_drift(package_dir: str) -> dict[str, Any] | None:
    from chain_replay_ml.feature_drift_studio.writer import load_studio_artifacts

    return load_studio_artifacts(package_dir)


def _load_diagnostics(package_dir: str) -> dict[str, Any] | None:
    from chain_replay_ml.diagnostics_studio.writer import load_studio_artifacts

    return load_studio_artifacts(package_dir)


def _load_planner(package_dir: str) -> dict[str, Any] | None:
    from chain_replay_ml.recommendation_engine.writer import load_studio_artifacts

    return load_studio_artifacts(package_dir)


_LOADERS: dict[str, Callable[[str], dict[str, Any] | None]] = {
    "importance": _load_importance,
    "distribution": _load_distribution,
    "drift": _load_drift,
    "diagnostics": _load_diagnostics,
    "planner": _load_planner,
}


def load_stage(*, package_dir: str, key: str) -> StudioStageResult:
    """Load one studio's artifacts. Missing artifacts → unavailable (not an error abort)."""
    loader = _LOADERS.get(key)
    if loader is None:
        return StudioStageResult(key=key, error=f"Unknown studio: {key}")
    try:
        payload = loader(package_dir)
    except Exception as exc:
        return StudioStageResult(key=key, error=str(exc))
    if not payload:
        return StudioStageResult(key=key, available=False)
    return StudioStageResult(key=key, available=True, payload=payload)


def run_load_pipeline(
    *,
    data_dir: str,
    model_name: str,
    progress: ProgressCb | None = None,
) -> PipelineResult:
    """Sequential Load Artifacts for the selected model (Planner last)."""
    from chain_replay_ml.training.paths import model_package_dir, safe_model_name

    safe = safe_model_name(model_name)
    pkg = model_package_dir(data_dir, safe)
    result = PipelineResult(model_name=safe)

    for key in PIPELINE_ORDER:
        if progress:
            progress({"stage": "load", "studio": key})
        stage = load_stage(package_dir=pkg, key=key)
        result.stages[key] = stage
    return result


def _compute_importance(data_dir: str, model_name: str) -> tuple[bool, str | None]:
    from chain_replay_ml.feature_importance_studio import run_feature_importance_studio

    result = run_feature_importance_studio(
        data_dir=data_dir,
        model_name=model_name,
        holdout_max_rows=20_000,
        permutation_n_repeats=3,
        shap_sample_size=400,
    )
    if not result.ok:
        return False, result.error or "Compute failed"
    return True, None


def _compute_distribution(data_dir: str, model_name: str) -> tuple[bool, str | None]:
    from chain_replay_ml.feature_distribution_studio import (
        run_feature_distribution_studio,
    )

    result = run_feature_distribution_studio(
        data_dir=data_dir,
        model_name=model_name,
        holdout_max_rows=20_000,
    )
    if not result.ok:
        return False, result.error or "Compute failed"
    return True, None


def _compute_drift(data_dir: str, model_name: str) -> tuple[bool, str | None]:
    from chain_replay_ml.feature_drift_studio import run_feature_drift_studio

    result = run_feature_drift_studio(
        data_dir=data_dir,
        model_name=model_name,
        holdout_max_rows=20_000,
        wf_max_rows=50_000,
    )
    if not result.ok:
        return False, result.error or "Compute failed"
    return True, None


def _compute_diagnostics(data_dir: str, model_name: str) -> tuple[bool, str | None]:
    from chain_replay_ml.diagnostics_studio import run_diagnostics_studio

    result = run_diagnostics_studio(data_dir=data_dir, model_name=model_name)
    if not result.ok:
        return False, result.error or "Compute failed"
    return True, None


def _compute_planner(data_dir: str, model_name: str) -> tuple[bool, str | None]:
    from chain_replay_ml.recommendation_engine import run_recommendation_engine

    result = run_recommendation_engine(data_dir=data_dir, model_name=model_name)
    if not result.ok:
        return False, result.error or "Compute failed"
    return True, None


_COMPUTERS: dict[str, Callable[[str, str], tuple[bool, str | None]]] = {
    "importance": _compute_importance,
    "distribution": _compute_distribution,
    "drift": _compute_drift,
    "diagnostics": _compute_diagnostics,
    "planner": _compute_planner,
}


def run_compute_pipeline(
    *,
    data_dir: str,
    model_name: str,
    progress: ProgressCb | None = None,
) -> PipelineResult:
    """Sequential Compute → Persist. Planner last; skipped if soft inputs missing.

    Does not populate UI payloads — call ``run_load_pipeline`` after this
    (Compute → Persist → Load → Populate).
    """
    from chain_replay_ml.training.paths import model_package_dir, safe_model_name

    safe = safe_model_name(model_name)
    pkg = model_package_dir(data_dir, safe)
    result = PipelineResult(model_name=safe)
    available: dict[str, bool] = {}

    for key in PIPELINE_ORDER:
        if progress:
            progress({"stage": "compute", "studio": key})

        if key == "planner":
            # Prefer freshly computed availability; fall back to on-disk artifacts.
            if not planner_inputs_available(available):
                disk = {
                    k: load_stage(package_dir=pkg, key=k).available
                    for k in PLANNER_SOFT_INPUTS
                }
                if not planner_inputs_available(disk):
                    result.stages[key] = StudioStageResult(
                        key=key,
                        skipped=True,
                        skip_reason=PLANNER_SKIP_MSG,
                    )
                    continue

        computer = _COMPUTERS.get(key)
        if computer is None:
            result.stages[key] = StudioStageResult(
                key=key, error=f"Unknown studio: {key}"
            )
            continue
        try:
            ok, err = computer(data_dir, safe)
        except Exception as exc:
            ok, err = False, str(exc)

        stage = StudioStageResult(key=key, available=ok, error=None if ok else err)
        result.stages[key] = stage
        if key != "planner":
            available[key] = ok

    return result

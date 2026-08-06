"""Sequential Feature Studio post-training orchestrator.

Order: Importance → Distribution → Drift.
Persists ``feature_studio_status.json`` under the model package (M2).
Config switches + richer telemetry (M3).
Failures are warnings only — never invalidate a trained model package.
"""

from __future__ import annotations

import logging
import os
import time
from datetime import datetime, timezone
from typing import Any, Callable

from .config import STAGE_KEYS as _STAGE_ORDER, resolve_post_training_config
from .status import empty_stages, write_feature_studio_status

logger = logging.getLogger(__name__)

ProgressCb = Callable[[dict[str, Any]], None]


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _stage_runner(stage: str):
    if stage == "importance":
        from chain_replay_ml.feature_importance_studio import (
            run_feature_importance_studio,
        )

        return run_feature_importance_studio
    if stage == "distribution":
        from chain_replay_ml.feature_distribution_studio import (
            run_feature_distribution_studio,
        )

        return run_feature_distribution_studio
    if stage == "drift":
        from chain_replay_ml.feature_drift_studio import run_feature_drift_studio

        return run_feature_drift_studio
    raise KeyError(stage)


def _overall_status(stages: dict[str, dict[str, Any]], *, active: list[str]) -> str:
    if not active:
        return "skipped"
    values = [str((stages.get(s) or {}).get("status") or "skipped") for s in active]
    if all(v == "completed" for v in values):
        return "completed"
    if all(v in ("failed", "skipped") for v in values):
        return "failed"
    if any(v == "completed" for v in values):
        return "partial"
    return "failed"


def _persist(package_dir: str, payload: dict[str, Any]) -> None:
    try:
        write_feature_studio_status(package_dir, payload)
    except Exception:
        logger.warning(
            "[PostTraining] Failed to write feature_studio_status.json",
            exc_info=True,
        )


def _snapshot(
    *,
    status: str,
    name: str,
    package_dir: str,
    stages: dict[str, dict[str, Any]],
    warnings: list[str],
    started_at: str,
    config: dict[str, Any],
    finished_at: str | None = None,
    duration_sec: float = 0.0,
    error: str | None = None,
) -> dict[str, Any]:
    active = list(config.get("active_stages") or [])
    skipped = [s for s in _STAGE_ORDER if s not in active]
    out: dict[str, Any] = {
        "status": status,
        "model_name": name,
        "package_dir": package_dir,
        "stages": stages,
        "importance": (stages.get("importance") or {}).get("status", "pending"),
        "distribution": (stages.get("distribution") or {}).get("status", "pending"),
        "drift": (stages.get("drift") or {}).get("status", "pending"),
        "warnings": list(warnings),
        "started_at": started_at,
        "finished_at": finished_at,
        "duration_sec": duration_sec,
        "timings_sec": {
            k: float((stages.get(k) or {}).get("duration_sec") or 0.0)
            for k in _STAGE_ORDER
        },
        "config": {
            "enabled": bool(config.get("enabled")),
            "importance": bool(config.get("importance", True)),
            "distribution": bool(config.get("distribution", True)),
            "drift": bool(config.get("drift", True)),
            "env_disabled": bool(config.get("env_disabled")),
        },
        "telemetry": {
            "started_at": started_at,
            "finished_at": finished_at,
            "duration_sec": duration_sec,
            "stages_run": active,
            "stages_skipped": skipped,
            "per_stage_sec": {
                k: float((stages.get(k) or {}).get("duration_sec") or 0.0)
                for k in active
            },
        },
    }
    if error:
        out["error"] = error
    return out


def run(
    model_package_path: str,
    data_dir: str,
    *,
    model_name: str | None = None,
    progress: ProgressCb | None = None,
    config: dict[str, Any] | None = None,
    X: Any = None,
    y: Any = None,
) -> dict[str, Any]:
    """Run Importance → Distribution → Drift after a successful Create Model.

    Parameters
    ----------
    model_package_path:
        Absolute/relative path to ``models/<ModelName>/``.
    data_dir:
        Chart data directory (datasets, models root).
    model_name:
        Optional; defaults to the package directory basename.
    progress:
        Optional callback ``{"stage", "status", "message", ...}``.
    config:
        Optional post-training switches (``enabled`` + per-studio flags).
        Env ``ARUNEO_POST_TRAINING=off`` still hard-disables the pipeline.
    X, y:
        Optional in-memory training matrices from Create Model. When both are
        provided, studios skip reloading the dataset parquet.

    Returns
    -------
    dict
        ``status``, per-stage statuses/timings/telemetry, warnings, timestamps.
        Also persisted as ``feature_studio_status.json`` in the package.
    """
    package_dir = os.path.abspath(str(model_package_path or "").strip())
    data_dir = os.path.abspath(str(data_dir or "").strip())
    name = str(model_name or "").strip() or os.path.basename(package_dir.rstrip("\\/"))
    resolved = resolve_post_training_config(config)

    started_at = _iso_now()
    t0 = time.perf_counter()
    stages: dict[str, dict[str, Any]] = empty_stages(status="pending")
    warnings: list[str] = []

    def _emit(payload: dict[str, Any]) -> None:
        if progress is None:
            return
        try:
            progress(payload)
        except Exception:
            pass

    if not resolved["enabled"]:
        reason = (
            "ARUNEO_POST_TRAINING=off"
            if resolved.get("env_disabled")
            else "disabled_by_config"
        )
        logger.info("[PostTraining] Skipped (%s)", reason)
        for s in _STAGE_ORDER:
            stages[s] = {
                "status": "skipped",
                "duration_sec": 0.0,
                "error": None,
                "skipped_reason": reason,
            }
        finished_at = _iso_now()
        result = _snapshot(
            status="skipped",
            name=name,
            package_dir=package_dir,
            stages=stages,
            warnings=[],
            started_at=started_at,
            finished_at=finished_at,
            duration_sec=round(max(time.perf_counter() - t0, 0.0), 4),
            config=resolved,
        )
        _persist(package_dir, result)
        _emit(
            {
                "stage": "post_training",
                "status": "skipped",
                "message": f"[PostTraining] Skipped ({reason})",
                "model_name": name,
                "post_training_status": "skipped",
                "timings_sec": result["timings_sec"],
                "duration_sec": result["duration_sec"],
            }
        )
        return result

    if not package_dir or not os.path.isdir(package_dir):
        msg = f"Model package not found: {package_dir}"
        logger.warning("[PostTraining] %s", msg)
        warnings.append(msg)
        finished_at = _iso_now()
        for s in _STAGE_ORDER:
            stages[s] = {
                "status": "skipped",
                "duration_sec": 0.0,
                "error": None,
                "skipped_reason": "package_missing",
            }
        return _snapshot(
            status="failed",
            name=name,
            package_dir=package_dir,
            stages=stages,
            warnings=warnings,
            started_at=started_at,
            finished_at=finished_at,
            duration_sec=round(max(time.perf_counter() - t0, 0.0), 4),
            config=resolved,
            error=msg,
        )

    active = list(resolved["active_stages"])
    for s in _STAGE_ORDER:
        if s not in active:
            stages[s] = {
                "status": "skipped",
                "duration_sec": 0.0,
                "error": None,
                "skipped_reason": "disabled_by_config",
            }

    if not active:
        logger.info("[PostTraining] Skipped (no stages enabled)")
        finished_at = _iso_now()
        result = _snapshot(
            status="skipped",
            name=name,
            package_dir=package_dir,
            stages=stages,
            warnings=[],
            started_at=started_at,
            finished_at=finished_at,
            duration_sec=round(max(time.perf_counter() - t0, 0.0), 4),
            config=resolved,
        )
        _persist(package_dir, result)
        return result

    logger.info(
        "[PostTraining] Start model=%s package=%s stages=%s",
        name,
        package_dir,
        ",".join(active),
    )
    _persist(
        package_dir,
        _snapshot(
            status="running",
            name=name,
            package_dir=package_dir,
            stages=stages,
            warnings=warnings,
            started_at=started_at,
            duration_sec=0.0,
            config=resolved,
        ),
    )
    _emit(
        {
            "stage": "post_training",
            "status": "started",
            "message": f"Post-training Feature Studio for {name} ({', '.join(active)})",
            "model_name": name,
            "active_stages": active,
            "post_training_config": {
                k: resolved[k]
                for k in ("enabled", "importance", "distribution", "drift")
            },
        }
    )

    for stage in _STAGE_ORDER:
        if stage not in active:
            _emit(
                {
                    "stage": f"post_training_{stage}",
                    "status": "skipped",
                    "message": f"[PostTraining] {stage.capitalize()} skipped (disabled)",
                    "model_name": name,
                    "duration_sec": 0.0,
                }
            )
            continue

        label = stage.capitalize()
        stage_started = _iso_now()
        logger.info("[PostTraining] %s started", label)
        stages[stage] = {
            "status": "running",
            "duration_sec": 0.0,
            "error": None,
            "started_at": stage_started,
        }
        _persist(
            package_dir,
            _snapshot(
                status="running",
                name=name,
                package_dir=package_dir,
                stages=stages,
                warnings=warnings,
                started_at=started_at,
                duration_sec=round(max(time.perf_counter() - t0, 0.0), 4),
                config=resolved,
            ),
        )
        _emit(
            {
                "stage": f"post_training_{stage}",
                "status": "started",
                "message": f"[PostTraining] {label} started",
                "model_name": name,
                "started_at": stage_started,
            }
        )
        stage_t0 = time.perf_counter()
        stage_status = "failed"
        stage_error: str | None = None
        artifacts_dir: str | None = None
        try:
            runner = _stage_runner(stage)

            def _stage_progress(evt: dict[str, Any], *, _stage=stage) -> None:
                payload = dict(evt or {})
                payload.setdefault("stage", f"post_training_{_stage}")
                payload.setdefault("model_name", name)
                _emit(payload)

            result = runner(
                data_dir=data_dir,
                model_name=name,
                package_dir=package_dir,
                progress=_stage_progress,
                X=X,
                y=y,
            )
            ok = bool(getattr(result, "ok", False))
            err = getattr(result, "error", None)
            art = getattr(result, "artifacts_dir", None)
            if art:
                artifacts_dir = str(art)
            if ok:
                stage_status = "completed"
            else:
                stage_status = "failed"
                stage_error = str(err or f"{label} returned ok=False")
                warnings.append(f"{label} failed: {stage_error}")
                logger.warning(
                    "[PostTraining] %s failed: %s",
                    label,
                    stage_error,
                )
        except Exception as exc:
            stage_status = "failed"
            stage_error = str(exc)
            warnings.append(f"{label} failed: {stage_error}")
            logger.warning(
                "[PostTraining] %s failed: %s",
                label,
                stage_error,
                exc_info=True,
            )

        duration = round(max(time.perf_counter() - stage_t0, 0.0), 4)
        stage_finished = _iso_now()
        stage_row: dict[str, Any] = {
            "status": stage_status,
            "duration_sec": duration,
            "error": stage_error,
            "started_at": stage_started,
            "finished_at": stage_finished,
        }
        if artifacts_dir:
            stage_row["artifacts_dir"] = artifacts_dir
        stages[stage] = stage_row
        _persist(
            package_dir,
            _snapshot(
                status="running",
                name=name,
                package_dir=package_dir,
                stages=stages,
                warnings=warnings,
                started_at=started_at,
                duration_sec=round(max(time.perf_counter() - t0, 0.0), 4),
                config=resolved,
            ),
        )
        if stage_status == "completed":
            logger.info(
                "[PostTraining] %s completed (%.1f s)",
                label,
                duration,
            )
            _emit(
                {
                    "stage": f"post_training_{stage}",
                    "status": "done",
                    "message": f"[PostTraining] {label} completed ({duration:.1f} s)",
                    "model_name": name,
                    "duration_sec": duration,
                    "finished_at": stage_finished,
                    "artifacts_dir": artifacts_dir,
                }
            )
        else:
            _emit(
                {
                    "stage": f"post_training_{stage}",
                    "status": "error",
                    "message": f"[PostTraining] {label} failed ({duration:.1f} s)",
                    "model_name": name,
                    "error": stage_error,
                    "duration_sec": duration,
                    "finished_at": stage_finished,
                }
            )

    finished_at = _iso_now()
    total = round(max(time.perf_counter() - t0, 0.0), 4)
    status = _overall_status(stages, active=active)
    logger.info(
        "[PostTraining] Complete total=%.1f s status=%s%s",
        total,
        status,
        f" warnings={len(warnings)}" if warnings else "",
    )
    if warnings:
        for w in warnings:
            logger.warning("[PostTraining] Warning: %s", w)

    final = _snapshot(
        status=status,
        name=name,
        package_dir=package_dir,
        stages=stages,
        warnings=warnings,
        started_at=started_at,
        finished_at=finished_at,
        duration_sec=total,
        config=resolved,
    )
    _persist(package_dir, final)

    _emit(
        {
            "stage": "post_training",
            "status": "done" if status == "completed" else status,
            "message": f"[PostTraining] Complete · {status} · {total:.1f} s",
            "model_name": name,
            "duration_sec": total,
            "post_training_status": status,
            "timings_sec": final["timings_sec"],
            "telemetry": final["telemetry"],
        }
    )

    return final


def run_safe(
    model_package_path: str,
    data_dir: str,
    *,
    model_name: str | None = None,
    progress: ProgressCb | None = None,
    config: dict[str, Any] | None = None,
    X: Any = None,
    y: Any = None,
) -> dict[str, Any]:
    """Like :func:`run` but never raises (Create Model must stay successful)."""
    try:
        return run(
            model_package_path,
            data_dir,
            model_name=model_name,
            progress=progress,
            config=config,
            X=X,
            y=y,
        )
    except Exception as exc:
        logger.warning(
            "[PostTraining] Unexpected failure (model package still valid): %s",
            exc,
            exc_info=True,
        )
        now = _iso_now()
        package_dir = os.path.abspath(str(model_package_path or "").strip())
        name = str(model_name or "").strip() or os.path.basename(package_dir.rstrip("\\/"))
        resolved = resolve_post_training_config(config)
        stages = empty_stages(status="skipped")
        for s in _STAGE_ORDER:
            stages[s] = {
                "status": "skipped",
                "duration_sec": 0.0,
                "error": None,
                "skipped_reason": "unexpected_error",
            }
        result = _snapshot(
            status="failed",
            name=name,
            package_dir=package_dir,
            stages=stages,
            warnings=[str(exc)],
            started_at=now,
            finished_at=now,
            duration_sec=0.0,
            config=resolved,
            error=str(exc),
        )
        _persist(package_dir, result)
        return result

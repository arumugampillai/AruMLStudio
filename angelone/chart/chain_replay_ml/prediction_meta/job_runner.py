"""Background prediction meta dataset build jobs."""

from __future__ import annotations

import threading
from typing import Any

_lock = threading.Lock()
_jobs: dict[str, dict[str, Any]] = {}


def job_key_for_project(project_id: str) -> str:
    return f"project:{str(project_id)}"


def is_build_running(project_id: str | None = None, *, market: str = "NIFTY", interval_sec: int = 3) -> bool:
    key = job_key_for_project(project_id) if project_id else f"legacy:{market}|{interval_sec}"
    with _lock:
        job = _jobs.get(key)
        if not job:
            return False
        thread = job.get("thread")
        return bool(thread and thread.is_alive())


def get_job_state(project_id: str | None = None, *, market: str = "NIFTY", interval_sec: int = 3) -> dict[str, Any]:
    key = job_key_for_project(project_id) if project_id else f"legacy:{market}|{interval_sec}"
    with _lock:
        job = dict(_jobs.get(key) or {})
    thread = job.get("thread")
    job["running"] = bool(thread and thread.is_alive())
    job["job_key"] = key
    return job


def start_build_job(
    *,
    data_dir: str,
    project_id: str | None = None,
    market: str = "NIFTY",
    sampling_interval_sec: int = 3,
    master_db_path: str | None = None,
    output_db_path: str | None = None,
    batch_size: int = 1000,
    resume: bool = True,
    enrich_path_outcomes: bool = True,
    selected_models: list[str] | None = None,
    trading_days_filter: list[str] | None = None,
    project_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    from .builder import build_prediction_meta_dataset
    from .projects import get_project, resolve_master_path_from_project

    proj = get_project(data_dir, project_id) if project_id else None
    if proj:
        master_db_path = resolve_master_path_from_project(data_dir, proj)
        output_db_path = proj.db_path(data_dir)
        market = proj.market
        sampling_interval_sec = proj.sampling_interval_sec
        batch_size = proj.batch_size
        enrich_path_outcomes = proj.enrich_path_outcomes
        selected_models = proj.selected_models
        trading_days_filter = proj.trading_days_filter
        project_config = proj.project_config_blob()

    key = job_key_for_project(project_id) if project_id else f"legacy:{market}|{sampling_interval_sec}"
    with _lock:
        existing = _jobs.get(key)
        if existing and existing.get("thread") and existing["thread"].is_alive():
            return {
                "status": "already_running",
                "job_key": key,
                "project_id": project_id,
                "message": "Prediction build already in progress",
            }

        result_holder: dict[str, Any] = {"result": None, "error": None}

        def _run() -> None:
            try:
                result_holder["result"] = build_prediction_meta_dataset(
                    data_dir,
                    market=market,
                    sampling_interval_sec=sampling_interval_sec,
                    master_db_path=master_db_path,
                    output_db_path=output_db_path,
                    batch_size=batch_size,
                    resume=resume,
                    enrich_path_outcomes=enrich_path_outcomes,
                    selected_models=selected_models,
                    trading_days_filter=trading_days_filter,
                    project_config=project_config,
                    project_id=project_id,
                )
            except Exception as exc:
                result_holder["error"] = str(exc)
            finally:
                with _lock:
                    if key in _jobs:
                        _jobs[key]["finished"] = True
                        _jobs[key]["result"] = result_holder.get("result")
                        _jobs[key]["error"] = result_holder.get("error")

        thread = threading.Thread(target=_run, name=f"prediction-meta-{key}", daemon=True)
        _jobs[key] = {
            "thread": thread,
            "finished": False,
            "result": None,
            "error": None,
            "project_id": project_id,
            "market": market,
            "interval_sec": sampling_interval_sec,
        }
        thread.start()

    return {
        "status": "started",
        "job_key": key,
        "project_id": project_id,
        "market": market,
        "interval_sec": sampling_interval_sec,
    }

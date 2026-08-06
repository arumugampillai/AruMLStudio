"""Read prediction meta dataset build status."""

from __future__ import annotations

import os
from typing import Any

from .builder import resolve_prediction_meta_db_path
from .model_registry import read_prediction_versions
from .store import PredictionMetaStore

try:
    from .job_runner import get_job_state, is_build_running
except ImportError:
    def is_build_running(**_kw: object) -> bool:
        return False

    def get_job_state(**_kw: object) -> dict:
        return {}


def read_prediction_meta_dashboard(
    data_dir: str,
    *,
    market: str = "NIFTY",
    sampling_interval_sec: int = 3,
    db_path: str | None = None,
    project_id: str | None = None,
) -> dict[str, Any]:
    """Full dashboard payload: progress, timings, caches, quality metrics."""
    if project_id and not db_path:
        from .projects import enrich_project_stats, get_project

        proj = get_project(data_dir, project_id)
        if proj:
            db_path = proj.db_path(data_dir)
            market = proj.market
            sampling_interval_sec = proj.sampling_interval_sec
            enriched = enrich_project_stats(data_dir, proj)
            project_meta = enriched
            build_fingerprint = enriched.get("build_fingerprint")
        else:
            build_fingerprint = None
    else:
        build_fingerprint = None

    base = read_prediction_meta_status(
        data_dir,
        market=market,
        sampling_interval_sec=sampling_interval_sec,
        db_path=db_path,
        project_id=project_id,
    )
    live = base.get("live_dashboard") if isinstance(base.get("live_dashboard"), dict) else None
    if not live and base.get("exists"):
        path = base.get("db_path")
        if path:
            with PredictionMetaStore(str(path)) as store:
                live = store.get_meta("live_dashboard")
                if isinstance(live, str):
                    import json
                    try:
                        live = json.loads(live)
                    except json.JSONDecodeError:
                        live = None

    job = get_job_state(project_id)
    status = str(base.get("status") or "missing")
    if job.get("running"):
        status = "running"
    elif job.get("error") and status != "complete":
        status = "failed"
        base["error_message"] = job.get("error")

    rows_done = int(base.get("rows_done") or 0)
    rows_total = base.get("rows_total")
    pct = round(100.0 * rows_done / max(rows_total or 1, 1), 1) if rows_total else 0.0

    dashboard = {
        **base,
        "status": status,
        "job_running": bool(job.get("running")),
        "build_fingerprint": build_fingerprint or base.get("build_fingerprint"),
        "progress_pct": live.get("progress_pct") if live else pct,
        "eta_label": live.get("eta_label") if live else "—",
        "eta_sec": live.get("eta_sec") if live else None,
        "rows_per_sec": live.get("rows_per_sec") if live else 0,
        "predictions_per_sec": live.get("predictions_per_sec") if live else 0,
        "timing_ms": (live or {}).get("timing_ms") or {"feature_build": 0, "prediction": 0, "sqlite": 0},
        "cache_pct": (live or {}).get("cache_pct") or {"registry_cache": 0, "model_cache": 0, "feature_cache": 0},
        "outcomes": (live or {}).get("outcomes") or {"completed": 0, "pending": 0},
        "failed_models": (live or {}).get("failed_models", 0),
        "skipped_rows": (live or {}).get("skipped_rows", 0),
        "quality": (live or {}).get("quality") or {},
        "live_dashboard": live,
    }
    return dashboard


def read_prediction_meta_status(
    data_dir: str,
    *,
    market: str = "NIFTY",
    sampling_interval_sec: int = 3,
    db_path: str | None = None,
    project_id: str | None = None,
) -> dict[str, Any]:
    project_meta = None
    if project_id and not db_path:
        from .projects import get_project

        proj = get_project(data_dir, project_id)
        if proj:
            db_path = proj.db_path(data_dir)
            project_meta = proj.to_dict()
            market = proj.market
            sampling_interval_sec = proj.sampling_interval_sec

    path = db_path or resolve_prediction_meta_db_path(
        data_dir, market=market, sampling_interval_sec=sampling_interval_sec,
    )
    if not os.path.isfile(path):
        return {
            "exists": False,
            "db_path": path,
            "project_id": project_id,
            "project": project_meta,
            "status": "missing",
            "rows_done": 0,
            "rows_total": None,
            "row_count": 0,
        }

    with PredictionMetaStore(path) as store:
        prog = store.read_progress()
        cfg = store.get_meta("project_config") or project_meta
        fp = store.get_meta("build_fingerprint")
        if not isinstance(fp, dict) and isinstance(cfg, dict):
            fp = cfg.get("build_fingerprint")
        return {
            "exists": True,
            "db_path": path,
            "project_id": project_id or (cfg or {}).get("project_id"),
            "project": cfg,
            "build_fingerprint": fp if isinstance(fp, dict) else None,
            "status": prog.status,
            "rows_done": prog.rows_done,
            "rows_total": prog.rows_total,
            "row_count": store.row_count(),
            "last_trading_day": prog.last_trading_day,
            "last_timestamp": prog.last_timestamp,
            "last_token": prog.last_token,
            "started_at": prog.started_at,
            "finished_at": prog.finished_at,
            "error_message": prog.error_message,
            "active_prediction_version": store.get_meta("active_prediction_version"),
            "model_registry_version": store.get_meta("model_registry_version"),
            "model_catalog": store.get_meta("model_catalog"),
            "source_master_db": store.get_meta("source_master_db"),
            "prediction_versions": read_prediction_versions(store.conn),
            "live_dashboard": store.get_meta("live_dashboard"),
        }

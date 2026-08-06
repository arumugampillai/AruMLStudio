"""Adapt MasterDatasetBuildOrchestrator progress → Create Dataset pipeline shape."""

from __future__ import annotations

from typing import Any

from chain_replay_ml.dataset_builder.progress import STAGE_NAMES

from .progress_panel import seed_pipeline_waiting


def _effective_substage_status(
    substage_id: int,
    timer_status: str,
    *,
    active_stage: int,
    running: bool,
) -> str:
    st = str(timer_status or "waiting").lower()
    if st in ("done", "skipped", "failed", "running"):
        return st
    if not running:
        return st
    if substage_id < active_stage:
        return "done"
    if substage_id == active_stage:
        return "running"
    return "waiting"


def enrich_master_build_payload(raw: dict[str, Any]) -> dict[str, Any]:
    """Map master-build progress events for the shared progress panel."""
    out = dict(raw)
    timer_pl = raw.get("pipeline") or {}
    stage = int(raw.get("stage") or 1)
    running = str(raw.get("status") or "running").lower() == "running"

    if timer_pl.get("stages"):
        substages: list[dict[str, Any]] = []
        by_id = {
            int(st.get("id") or 0): st
            for st in (timer_pl.get("stages") or [])
            if int(st.get("id") or 0) > 0
        }
        for sid in range(1, 9):
            st = by_id.get(sid) or {
                "id": sid,
                "name": STAGE_NAMES.get(sid, f"Stage {sid}"),
                "status": "waiting",
            }
            status = _effective_substage_status(
                sid,
                str(st.get("status") or "waiting"),
                active_stage=stage,
                running=running,
            )
            substages.append({
                "id": sid,
                "label": st.get("name") or STAGE_NAMES.get(sid, f"Stage {sid}"),
                "name": st.get("name") or STAGE_NAMES.get(sid, f"Stage {sid}"),
                "status": status,
                "elapsed_sec": st.get("elapsed_sec"),
                "elapsed_label": st.get("elapsed_label"),
                "progress_current": st.get("progress_current"),
                "progress_total": st.get("progress_total"),
                "progress_unit": st.get("progress_unit"),
                "parent_stage": "build",
            })

        for sub in timer_pl.get("substages") or []:
            substages.append({
                "id": sub.get("id"),
                "label": sub.get("label") or sub.get("id"),
                "name": sub.get("label") or sub.get("id"),
                "status": sub.get("status") or "waiting",
                "elapsed_sec": sub.get("elapsed_sec"),
                "elapsed_label": sub.get("elapsed_label"),
                "parent_stage": 6,
            })

        build_status = (
            "done" if out.get("status") == "completed"
            else "failed" if out.get("status") in ("failed", "cancelled")
            else "running" if running
            else "waiting"
        )
        out["pipeline"] = {
            "stages": [{
                "id": "build",
                "name": "Create Dataset",
                "status": build_status,
                "elapsed_label": timer_pl.get("total_elapsed_label"),
                "elapsed_sec": timer_pl.get("total_elapsed_sec"),
            }],
            "substages": substages,
            "total_elapsed_sec": timer_pl.get("total_elapsed_sec"),
            "total_elapsed_label": timer_pl.get("total_elapsed_label"),
            "rows_per_sec": timer_pl.get("rows_per_sec"),
            "eta_label": timer_pl.get("eta_label"),
            "eta_sec": timer_pl.get("eta_sec"),
            "active_stage": timer_pl.get("active_stage"),
            "slowest_stage": timer_pl.get("slowest_stage"),
        }
    else:
        pl = seed_pipeline_waiting()
        substages = pl["substages"]
        for sub in substages:
            sid = int(sub["id"])
            if sid < stage:
                sub["status"] = "done"
            elif sid == stage:
                sub["status"] = "running"
            else:
                sub["status"] = "waiting"
        pl["stages"] = [{
            "id": "build",
            "name": "Create Dataset",
            "status": "running" if running else "done",
            "elapsed_label": timer_pl.get("total_elapsed_label"),
        }]
        pl["substages"] = substages
        out["pipeline"] = pl

    out.setdefault("stage_name", STAGE_NAMES.get(stage, ""))
    return out


def master_build_done_payload(result_dict: dict[str, Any], master_db_path: str) -> dict[str, Any]:
    stats = result_dict.get("dataset_stats") or {}
    timer_pl = result_dict.get("pipeline") or {}
    pl = seed_pipeline_waiting()
    for sub in pl["substages"]:
        sub["status"] = "done"
    pl["stages"] = [{
        "id": "build",
        "name": "Create Dataset",
        "status": "done",
        "elapsed_label": timer_pl.get("total_elapsed_label"),
        "elapsed_sec": timer_pl.get("total_elapsed_sec"),
    }]
    pl["total_elapsed_label"] = timer_pl.get("total_elapsed_label")
    pl["total_elapsed_sec"] = timer_pl.get("total_elapsed_sec")
    out = {
        "status": result_dict.get("status") or "completed",
        "master_dataset_only": True,
        "master_db_path": master_db_path,
        "dataset_stats": stats,
        "message": (
            f"Master Dataset updated — {int(stats.get('rows') or 0):,} rows"
        ),
        "pipeline": pl,
    }
    if stats.get("feature_policy_report"):
        out["feature_policy_report"] = stats["feature_policy_report"]
    if stats.get("build_profiler_report"):
        out["build_profiler_report"] = stats["build_profiler_report"]
    return out


def _debug_pipeline(
    *,
    stage1_status: str = "running",
    build_status: str = "running",
    stage1_current: int | None = None,
    stage1_total: int | None = None,
) -> dict[str, Any]:
    pl = seed_pipeline_waiting()
    pl["stages"][0]["status"] = build_status
    for sub in pl["substages"]:
        sid = int(sub["id"])
        if sid == 1:
            sub["status"] = stage1_status
            if stage1_total is not None and stage1_total > 0:
                sub["progress_current"] = max(0, int(stage1_current or 0))
                sub["progress_total"] = int(stage1_total)
                sub["progress_unit"] = "days"
        elif sid < 1:
            sub["status"] = "done"
        elif stage1_status == "done":
            sub["status"] = "waiting"
        else:
            sub["status"] = "waiting"
    return pl


def debug_load_running_payload(
    *,
    message: str,
    current: int = 0,
    total: int = 1,
    source_day_index: int | None = None,
    source_day_total: int | None = None,
    ticks_in_memory: int = 0,
    spot_ticks: int = 0,
    chain_ticks: int = 0,
    elapsed_sec: float | None = None,
) -> dict[str, Any]:
    pl = _debug_pipeline(
        stage1_status="running",
        build_status="running",
        stage1_current=current,
        stage1_total=total,
    )
    if elapsed_sec is not None:
        pl["total_elapsed_sec"] = round(elapsed_sec, 2)
        pl["total_elapsed_label"] = f"{elapsed_sec:.2f} s"
        pl["stages"][0]["elapsed_sec"] = pl["total_elapsed_sec"]
        pl["stages"][0]["elapsed_label"] = pl["total_elapsed_label"]
        for sub in pl["substages"]:
            if int(sub["id"]) == 1:
                sub["elapsed_sec"] = pl["total_elapsed_sec"]
                sub["elapsed_label"] = pl["total_elapsed_label"]
    return {
        "status": "running",
        "debug_load": True,
        "stage": 1,
        "stage_name": "Load Database",
        "current": current,
        "total": total,
        "message": message,
        "source_day_index": source_day_index,
        "source_day_total": source_day_total,
        "ticks_in_memory": ticks_in_memory,
        "spot_ticks": spot_ticks,
        "chain_ticks": chain_ticks,
        "pipeline": pl,
    }


def debug_load_done_payload(
    *,
    ticks_in_memory: int,
    spot_ticks: int,
    chain_ticks: int,
    sources_loaded: int,
    elapsed_sec: float,
    load_results: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    pl = _debug_pipeline(
        stage1_status="done",
        build_status="done",
        stage1_current=sources_loaded,
        stage1_total=max(1, sources_loaded),
    )
    pl["total_elapsed_sec"] = round(elapsed_sec, 2)
    pl["total_elapsed_label"] = f"{elapsed_sec:.2f} s"
    pl["stages"][0]["elapsed_sec"] = pl["total_elapsed_sec"]
    pl["stages"][0]["elapsed_label"] = pl["total_elapsed_label"]
    for sub in pl["substages"]:
        if int(sub["id"]) == 1:
            sub["elapsed_sec"] = pl["total_elapsed_sec"]
            sub["elapsed_label"] = pl["total_elapsed_label"]
    msg = (
        f"Debug load done — {ticks_in_memory:,} ticks in memory "
        f"({spot_ticks:,} spot + {chain_ticks:,} chain)"
    )
    return {
        "status": "completed",
        "debug_load": True,
        "message": msg,
        "ticks_in_memory": ticks_in_memory,
        "spot_ticks": spot_ticks,
        "chain_ticks": chain_ticks,
        "sources_loaded": sources_loaded,
        "load_results": load_results or [],
        "pipeline": pl,
    }


def debug_feature_done_payload(
    *,
    rows: int,
    feature_count: int,
    groups_run: int,
    sources_loaded: int,
    elapsed_sec: float,
    ticks_in_memory: int,
    spot_ticks: int = 0,
    chain_ticks: int = 0,
    pipeline: dict[str, Any],
    feature_readiness: dict[str, Any] | None = None,
) -> dict[str, Any]:
    pl = dict(pipeline)
    pl["total_elapsed_sec"] = round(elapsed_sec, 2)
    pl["total_elapsed_label"] = f"{elapsed_sec:.2f} s"
    msg = (
        f"Debug features done — {rows:,} rows in memory "
        f"({feature_count} features, {groups_run} groups)"
    )
    out: dict[str, Any] = {
        "status": "completed",
        "debug_features": True,
        "message": msg,
        "rows": rows,
        "feature_count": feature_count,
        "groups_run": groups_run,
        "sources_loaded": sources_loaded,
        "ticks_in_memory": ticks_in_memory,
        "spot_ticks": spot_ticks,
        "chain_ticks": chain_ticks,
        "pipeline": enrich_master_build_payload({
            "status": "completed",
            "stage": 6,
            "rows": rows,
            "pipeline": pl,
        })["pipeline"],
    }
    if feature_readiness:
        out["dataset_stats"] = {"feature_readiness": feature_readiness}
    return out

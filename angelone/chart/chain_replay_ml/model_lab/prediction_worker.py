"""
External prediction worker process.

Usage:
  python -m chain_replay_ml.model_lab.prediction_worker \\
      --job-id <id> --worker-id <n> --lab-db <path> [--log-file <path>]

Fully stateless: resume state comes only from job / worker / checkpoint tables.
Each worker owns whole trading day(s) — never a partial day shared with peers.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import threading
import time
import traceback
from typing import Any

from .prediction_job_schema import (
    CHECKPOINT_BATCH_ROWS,
    DAY_CP_CANCELLED,
    DAY_CP_COMPLETED,
    DAY_CP_FAILED,
    DAY_CP_IN_PROGRESS,
    JOB_STATUS_CANCELLED,
    JOB_STATUS_PAUSED,
    JOB_STATUS_RUNNING,
    WORKER_POLL_SEC,
    WORKER_STATUS_DONE,
    WORKER_STATUS_FAILED,
    WORKER_STATUS_RUNNING,
    WORKER_STATUS_WAITING,
)
from .prediction_job_store import (
    get_job,
    refresh_job_day_counts,
    update_checkpoint,
    update_worker,
    worker_pending_days,
)
from .prediction_parallel import (
    DayJobContext,
    process_trading_day,
    tb_all_null_warning,
)
from .prediction_schema import (
    DAY_FAILED,
    DAY_RUNNING,
    DAY_WAITING,
    LAB_PHASE_PREDICTION,
    LAB_SCHEMA_VERSION_PREDICTION,
    PRED_STATUS_PAUSED,
    PRED_STATUS_READY,
    resolve_day_completion_status,
)
from .store import ModelLabStore


def _setup_logging(log_file: str | None, worker_id: int) -> logging.Logger:
    logger = logging.getLogger(f"prediction_worker_{worker_id}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter(
        "%(asctime)s [worker-%(worker_id)s] %(levelname)s %(message)s"
    )
    # Inject worker_id into records
    old_factory = logging.getLogRecordFactory()

    def record_factory(*args: Any, **kwargs: Any):
        record = old_factory(*args, **kwargs)
        record.worker_id = worker_id  # type: ignore[attr-defined]
        return record

    logging.setLogRecordFactory(record_factory)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    if log_file:
        os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


def _read_job_status(lab_db: str, job_id: str) -> str:
    job = get_job(lab_db, job_id)
    return str((job or {}).get("status") or "")


def _wait_if_paused(lab_db: str, job_id: str, worker_id: int, log: logging.Logger) -> str:
    """Block while job is paused (between days only). Return status when leaving wait."""
    while True:
        status = _read_job_status(lab_db, job_id)
        if status == JOB_STATUS_PAUSED:
            update_worker(
                lab_db,
                job_id,
                worker_id,
                status=WORKER_STATUS_WAITING,
                last_message="Paused — waiting for resume",
            )
            time.sleep(WORKER_POLL_SEC)
            continue
        return status


_HEARTBEAT_INTERVAL_SEC = 15.0


def _start_heartbeat_thread(
    lab_db: str,
    job_id: str,
    worker_id: int,
    log: logging.Logger,
    *,
    interval_sec: float = _HEARTBEAT_INTERVAL_SEC,
) -> tuple[threading.Event, threading.Thread]:
    """
    Background liveness ping — independent of day-processing progress.

    Some trading days spend 60-200+ seconds inside a single blocking call
    (e.g. loading tick timelines for path-outcome enrichment) with no
    intermediate progress callback. The manager (``prediction_manager``)
    treats a worker as dead once its ``heartbeat_at`` column goes stale
    (``_STALE_HEARTBEAT_SEC``), even though the OS process is healthy and
    simply mid-computation. Without this thread, a slow-but-legitimate day
    can be misdiagnosed as "Worker process exited before day completed" —
    the job/day get marked failed while the orphaned worker keeps running
    and finishes normally moments later. Ping on a fixed wall-clock cadence
    so liveness detection never depends on how granular any one stage's
    progress reporting happens to be.
    """
    stop_event = threading.Event()

    def _loop() -> None:
        while not stop_event.wait(interval_sec):
            try:
                update_worker(lab_db, job_id, worker_id, heartbeat=True)
            except Exception as exc:  # never let heartbeat noise kill the worker
                log.debug("Heartbeat ping failed (non-fatal): %s", exc)

    thread = threading.Thread(
        target=_loop, name=f"pred-worker-heartbeat-{worker_id}", daemon=True
    )
    thread.start()
    return stop_event, thread


def _stop_heartbeat_thread(stop_event: threading.Event, thread: threading.Thread) -> None:
    stop_event.set()
    thread.join(timeout=2.0)


def _ctx_from_config(cfg: dict[str, Any]) -> DayJobContext:
    return DayJobContext(
        data_dir=str(cfg["data_dir"]),
        parquet_path=str(cfg["parquet_path"]),
        features=list(cfg["features"]),
        target=str(cfg["target"]),
        wanted_columns=list(cfg["wanted_columns"]),
        lab_uuid=str(cfg["lab_uuid"]),
        feat_map=dict(cfg["feat_map"]),
        horizon_sec=float(cfg["horizon_sec"]),
        enrich_path_outcomes=bool(cfg.get("enrich_path_outcomes", True)),
        model_path=str(cfg["model_path"]),
        algorithm=cfg.get("algorithm"),
        days=list(cfg.get("days") or []),
        row_limit=cfg.get("row_limit"),
        mark_day_complete=bool(cfg.get("mark_day_complete", True)),
        embed_features=bool(cfg.get("embed_features", True)),
        master_db_path=cfg.get("master_db_path"),
        stamp_master_row_ids=bool(cfg.get("stamp_master_row_ids", False)),
        master_filter=dict(cfg.get("master_filter") or {}) or None,
        profile_outcome_rows=cfg.get("profile_outcome_rows"),
        package_members=list(cfg.get("package_members") or []),
        transformation_config=(
            dict(cfg["transformation_config"])
            if isinstance(cfg.get("transformation_config"), dict)
            else None
        ),
        sample_interval_sec=(
            float(cfg["sample_interval_sec"])
            if cfg.get("sample_interval_sec") is not None
            else None
        ),
        parent_dataset=(
            str(cfg.get("parent_dataset") or "").strip() or None
        ),
        tb_model_name=str(cfg.get("tb_model_name") or "").strip() or None,
    )


def _write_day_batches(
    *,
    lab_db: str,
    job_id: str,
    worker_id: int,
    lab_uuid: str,
    day: str,
    rows: list[dict[str, Any]],
    feature_columns: list[str],
    start_at: int,
    log: logging.Logger,
) -> tuple[int, str | None]:
    """
    Insert rows in checkpoint batches. Returns (rows_committed, stop_reason).

    Pause does not interrupt an in-flight day (finish the day). Cancel stops
    after the current batch.
    """
    total = len(rows)
    committed = max(0, int(start_at))
    batch = max(1, int(CHECKPOINT_BATCH_ROWS))
    while committed < total:
        status = _read_job_status(lab_db, job_id)
        if status == JOB_STATUS_CANCELLED:
            return committed, "cancelled"

        end = min(committed + batch, total)
        chunk = rows[committed:end]
        with ModelLabStore(lab_db) as store:
            if committed == 0:
                store.delete_predictions_for_day(lab_uuid, day)
            store.insert_prediction_rows(chunk, feature_columns=feature_columns)
        committed = end
        update_checkpoint(
            lab_db,
            job_id,
            day,
            status=DAY_CP_IN_PROGRESS,
            rows_committed=committed,
            rows_expected=total,
        )
        pct = 100.0 * committed / max(total, 1)
        update_worker(
            lab_db,
            job_id,
            worker_id,
            assigned_day=day,
            current_row=committed,
            total_rows=total,
            percent=pct,
            status=WORKER_STATUS_RUNNING,
            last_message=f"Writing {day}: {committed:,}/{total:,}",
        )
        log.info("Checkpoint %s rows_committed=%s/%s", day, committed, total)
    return committed, None


def run_worker(
    *,
    job_id: str,
    worker_id: int,
    lab_db: str,
    log_file: str | None = None,
) -> int:
    log = _setup_logging(log_file, worker_id)
    lab_db = os.path.abspath(lab_db)
    log.info("Starting job=%s worker=%s lab=%s", job_id, worker_id, lab_db)

    job = get_job(lab_db, job_id)
    if not job:
        log.error("Job not found: %s", job_id)
        return 2

    update_worker(
        lab_db,
        job_id,
        worker_id,
        pid=os.getpid(),
        status=WORKER_STATUS_RUNNING,
        log_path=log_file,
        last_message="Worker started",
        started=True,
    )

    cfg = dict(job.get("config") or {})
    if not cfg:
        update_worker(
            lab_db,
            job_id,
            worker_id,
            status=WORKER_STATUS_FAILED,
            last_message="Missing job config",
            finished=True,
        )
        return 3

    from chain_replay_ml.training.inference_runtime import (
        format_day_stage_timings,
        load_prediction_model_for_inference,
    )

    try:
        model, infer_info = load_prediction_model_for_inference(
            str(cfg["model_path"]), cfg.get("algorithm")
        )
    except Exception as exc:
        log.exception("Failed to load model")
        update_worker(
            lab_db,
            job_id,
            worker_id,
            status=WORKER_STATUS_FAILED,
            last_message=f"Model load failed: {exc}",
            finished=True,
        )
        return 4

    log.info(
        "Inference device=%s algorithm=%s api=%s gpu_requested=%s%s",
        infer_info.device_label,
        infer_info.algorithm,
        infer_info.predict_api,
        infer_info.gpu_requested,
        f" fallback={infer_info.fallback_reason}" if infer_info.fallback_reason else "",
    )

    ctx = _ctx_from_config(cfg)
    ctx.inference_device = infer_info.device_label
    ctx.inference_algorithm = infer_info.algorithm
    feature_columns = list(cfg.get("feature_columns") or [])
    exit_code = 0

    # Independent liveness ping — see _start_heartbeat_thread docstring. Must
    # cover the whole day-processing loop, including long single blocking
    # calls (tick-timeline load) that have no intermediate progress callback.
    heartbeat_stop, heartbeat_thread = _start_heartbeat_thread(
        lab_db, job_id, worker_id, log
    )

    try:
        while True:
            status = _wait_if_paused(lab_db, job_id, worker_id, log)
            if status == JOB_STATUS_CANCELLED:
                log.info("Job cancelled — exiting")
                # Mark remaining pending days owned by this worker as cancelled
                for item in worker_pending_days(lab_db, job_id, worker_id):
                    update_checkpoint(
                        lab_db,
                        job_id,
                        item["trading_day"],
                        status=DAY_CP_CANCELLED,
                        finished=True,
                    )
                break

            pending = worker_pending_days(lab_db, job_id, worker_id)
            if not pending:
                log.info("No pending days — worker done")
                break

            day_info = pending[0]
            day = str(day_info["trading_day"])
            already = int(day_info.get("rows_committed") or 0)
            log.info(
                "Processing day=%s resume_from_row=%s", day, already
            )

            update_checkpoint(
                lab_db,
                job_id,
                day,
                status=DAY_CP_IN_PROGRESS,
                started=True,
            )
            update_worker(
                lab_db,
                job_id,
                worker_id,
                assigned_day=day,
                current_row=already,
                total_rows=0,
                percent=0.0,
                status=WORKER_STATUS_RUNNING,
                last_message=f"Predicting {day} ({infer_info.device_label})",
            )

            with ModelLabStore(lab_db) as store:
                store.set_build_day_status(
                    ctx.lab_uuid,
                    day,
                    status=DAY_RUNNING,
                    started=True,
                    progress_pct=0.0,
                    error_message="",
                )

            day_index = list(ctx.days).index(day) if day in ctx.days else 0

            def _on_status(state: str) -> None:
                update_worker(
                    lab_db,
                    job_id,
                    worker_id,
                    assigned_day=day,
                    status=WORKER_STATUS_RUNNING,
                    last_message=f"{day}: {state}",
                )

            def _on_day_progress(fields: dict[str, Any]) -> None:
                update_worker(
                    lab_db,
                    job_id,
                    worker_id,
                    assigned_day=day,
                    current_row=fields.get("current_row"),
                    total_rows=fields.get("total_rows"),
                    percent=fields.get("percent"),
                    status=WORKER_STATUS_RUNNING,
                    last_message=str(fields.get("message") or ""),
                )

            batch = process_trading_day(
                ctx=ctx,
                day=day,
                day_index=day_index,
                worker_id=worker_id,
                model=model,
                hub=None,
                on_status=_on_status,
                inference_info=infer_info,
                on_day_progress=_on_day_progress,
            )

            if not batch.ok:
                err = batch.error or "day failed"
                log.error("Day %s failed: %s", day, err)
                if batch.stage_timing_log:
                    log.info("Day %s timings (failed):\n%s", day, batch.stage_timing_log)
                update_checkpoint(
                    lab_db,
                    job_id,
                    day,
                    status=DAY_CP_FAILED,
                    error_message=err,
                    finished=True,
                )
                with ModelLabStore(lab_db) as store:
                    store.set_build_day_status(
                        ctx.lab_uuid,
                        day,
                        status=DAY_FAILED,
                        finished=True,
                        error_message=err,
                    )
                exit_code = 5
                continue

            rows = list(batch.rows or [])
            update_checkpoint(
                lab_db,
                job_id,
                day,
                rows_expected=len(rows),
            )

            t_write = time.perf_counter()
            committed, stop = _write_day_batches(
                lab_db=lab_db,
                job_id=job_id,
                worker_id=worker_id,
                lab_uuid=ctx.lab_uuid,
                day=day,
                rows=rows,
                feature_columns=feature_columns,
                start_at=already,
                log=log,
            )
            write_sec = time.perf_counter() - t_write
            timings = dict(batch.timings or {})
            timings["sqlite_write"] = write_sec
            timings["total"] = (
                float(timings.get("load_master") or 0.0)
                + float(timings.get("load_timeline") or 0.0)
                + float(timings.get("prepare_matrix") or 0.0)
                + float(timings.get("predict") or 0.0)
                + float(timings.get("predict_members") or 0.0)
                + float(timings.get("outcomes") or 0.0)
                + write_sec
            )
            timing_block = format_day_stage_timings(
                timings,
                device_label=str(
                    batch.inference_device or infer_info.device_label
                ),
                algorithm=str(infer_info.algorithm),
            )
            log.info("Day %s stage timings:\n%s", day, timing_block)
            if batch.per_prediction_timing_log:
                log.info(
                    "Day %s per-prediction outcome timings:\n%s",
                    day,
                    batch.per_prediction_timing_log,
                )
            if getattr(batch, "path_microprofile_log", ""):
                log.info(
                    "Day %s path-outcome micro-profile:\n%s",
                    day,
                    batch.path_microprofile_log,
                )
            if batch.outcome_profile_ms and log_file:
                from chain_replay_ml.training.inference_runtime import (
                    write_outcome_profile_csv,
                )

                safe_day = str(day).replace("-", "")
                csv_path = os.path.join(
                    os.path.dirname(os.path.abspath(log_file)),
                    f"outcome_profile_{job_id}_{worker_id}_{safe_day}.csv",
                )
                try:
                    write_outcome_profile_csv(csv_path, list(batch.outcome_profile_ms))
                    log.info("Day %s outcome profile CSV: %s", day, csv_path)
                except Exception as exc:
                    log.warning("Failed to write outcome profile CSV: %s", exc)

            if stop == "cancelled":
                update_checkpoint(
                    lab_db,
                    job_id,
                    day,
                    status=DAY_CP_CANCELLED,
                    rows_committed=committed,
                    finished=True,
                )
                with ModelLabStore(lab_db) as store:
                    store.set_build_day_status(
                        ctx.lab_uuid,
                        day,
                        status=DAY_WAITING,
                        progress_pct=100.0 * committed / max(len(rows), 1),
                    )
                break

            if stop == "paused" or committed < len(rows):
                # Left mid-day for pause — keep in_progress for resume
                update_checkpoint(
                    lab_db,
                    job_id,
                    day,
                    status=DAY_CP_IN_PROGRESS,
                    rows_committed=committed,
                )
                continue

            # Day complete
            update_checkpoint(
                lab_db,
                job_id,
                day,
                status=DAY_CP_COMPLETED,
                rows_committed=committed,
                finished=True,
            )
            tb_warn = tb_all_null_warning(tb_model_name=ctx.tb_model_name, rows=rows)
            if tb_warn:
                log.warning("Day %s: %s", day, tb_warn)
            with ModelLabStore(lab_db) as store:
                if ctx.mark_day_complete:
                    # Preserve the true parent-dataset/Master expected count —
                    # never clobber it with the just-committed row count, or
                    # Complete vs Partial can never be told apart later.
                    existing_expected = store.day_rows_expected(ctx.lab_uuid, day)
                    expected = (
                        existing_expected
                        if existing_expected and existing_expected > 0
                        else committed
                    )
                    day_status = resolve_day_completion_status(committed, expected)
                    store.set_build_day_status(
                        ctx.lab_uuid,
                        day,
                        status=day_status,
                        row_count=committed,
                        rows_expected=expected,
                        finished=True,
                        progress_pct=100.0,
                    )
                else:
                    store.set_build_day_status(
                        ctx.lab_uuid,
                        day,
                        status=DAY_WAITING,
                        row_count=committed,
                        progress_pct=100.0,
                    )
            refresh_job_day_counts(lab_db, job_id)
            log.info("Day %s completed (%s rows)", day, committed)

        # Finalize lab summary if this was the last work
        refresh_job_day_counts(lab_db, job_id)
        pending_any = worker_pending_days(lab_db, job_id, worker_id)
        job_status = _read_job_status(lab_db, job_id)
        with ModelLabStore(lab_db) as store:
            done = store.prediction_row_count()
            n_days = len(cfg.get("days") or [])
            if job_status == JOB_STATUS_CANCELLED:
                store.write_prediction_summary(
                    lab_uuid=str(cfg["lab_uuid"]),
                    status=PRED_STATUS_PAUSED if done > 0 else "error",
                    row_count=done,
                    trading_days=n_days,
                    parent_model_name=cfg.get("parent_model_name"),
                    parent_dataset=cfg.get("parent_dataset"),
                    target_column=cfg.get("target"),
                    selected_feature_count=len(cfg.get("features") or []),
                    feature_columns_json=__import__("json").dumps(
                        cfg.get("feat_map") or {}, ensure_ascii=False
                    ),
                    feature_storage_mode=cfg.get("storage_mode"),
                    master_dataset_id=cfg.get("master_dataset_id"),
                    master_db_path=cfg.get("master_db_path_store"),
                    error_message="Cancelled",
                )
            elif job_status == JOB_STATUS_PAUSED or pending_any:
                store.write_prediction_summary(
                    lab_uuid=str(cfg["lab_uuid"]),
                    status=PRED_STATUS_PAUSED,
                    row_count=done,
                    trading_days=n_days,
                    parent_model_name=cfg.get("parent_model_name"),
                    parent_dataset=cfg.get("parent_dataset"),
                    target_column=cfg.get("target"),
                    selected_feature_count=len(cfg.get("features") or []),
                    feature_columns_json=__import__("json").dumps(
                        cfg.get("feat_map") or {}, ensure_ascii=False
                    ),
                    feature_storage_mode=cfg.get("storage_mode"),
                    master_dataset_id=cfg.get("master_dataset_id"),
                    master_db_path=cfg.get("master_db_path_store"),
                )
            else:
                store.write_prediction_summary(
                    lab_uuid=str(cfg["lab_uuid"]),
                    status=PRED_STATUS_READY if done > 0 else "error",
                    row_count=done,
                    trading_days=n_days,
                    start_day=(cfg.get("days") or [None])[0],
                    end_day=(cfg.get("days") or [None])[-1],
                    parent_model_name=cfg.get("parent_model_name"),
                    parent_dataset=cfg.get("parent_dataset"),
                    target_column=cfg.get("target"),
                    selected_feature_count=len(cfg.get("features") or []),
                    feature_columns_json=__import__("json").dumps(
                        cfg.get("feat_map") or {}, ensure_ascii=False
                    ),
                    feature_storage_mode=cfg.get("storage_mode"),
                    master_dataset_id=cfg.get("master_dataset_id"),
                    master_db_path=cfg.get("master_db_path_store"),
                )
                if done > 0:
                    store.update_lab_phase(
                        phase=LAB_PHASE_PREDICTION,
                        lab_schema_version=LAB_SCHEMA_VERSION_PREDICTION,
                    )

        update_worker(
            lab_db,
            job_id,
            worker_id,
            status=WORKER_STATUS_DONE if exit_code == 0 else WORKER_STATUS_FAILED,
            assigned_day="",
            last_message="Worker finished",
            finished=True,
        )
        log.info("Worker exit code=%s", exit_code)
        return exit_code
    except Exception as exc:
        log.error("Worker crashed: %s\n%s", exc, traceback.format_exc())
        update_worker(
            lab_db,
            job_id,
            worker_id,
            status=WORKER_STATUS_FAILED,
            last_message=str(exc),
            finished=True,
        )
        return 1
    finally:
        _stop_heartbeat_thread(heartbeat_stop, heartbeat_thread)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prediction Dataset external worker")
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--worker-id", type=int, required=True)
    parser.add_argument("--lab-db", required=True)
    parser.add_argument("--log-file", default=None)
    args = parser.parse_args(argv)
    return run_worker(
        job_id=args.job_id,
        worker_id=int(args.worker_id),
        lab_db=args.lab_db,
        log_file=args.log_file,
    )


if __name__ == "__main__":
    raise SystemExit(main())

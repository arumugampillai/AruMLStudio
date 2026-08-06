"""Prediction Manager — spawn/monitor/control external worker processes.

GUI uses this only. Prediction generation never runs in the GUI process.
"""

from __future__ import annotations

import os
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any

from .prediction_job_prepare import prepare_prediction_exec_config
from .prediction_job_schema import (
    DAY_CP_FAILED,
    DEFAULT_EXEC_WORKERS,
    JOB_STATUS_CANCELLED,
    JOB_STATUS_COMPLETED,
    JOB_STATUS_FAILED,
    JOB_STATUS_PAUSED,
    JOB_STATUS_PENDING,
    JOB_STATUS_RUNNING,
    MAX_EXEC_WORKERS,
    WORKER_STATUS_DONE,
    WORKER_STATUS_FAILED,
    WORKER_STATUS_RUNNING,
)
from .prediction_job_store import (
    assign_days_round_robin,
    create_job,
    get_job,
    job_progress_snapshot,
    list_active_jobs,
    list_workers,
    request_job_cancel,
    request_job_pause,
    request_job_resume,
    set_job_status,
    update_checkpoint,
    update_worker,
)


def _worker_module() -> str:
    return "chain_replay_ml.model_lab.prediction_worker"


def _logs_dir(lab_db_path: str) -> str:
    base = os.path.dirname(os.path.abspath(lab_db_path))
    path = os.path.join(base, "logs")
    os.makedirs(path, exist_ok=True)
    return path


# After this many seconds without a heartbeat, treat the worker as dead even if
# the OS PID still exists (PID reuse / hung process after GUI relaunch).
#
# The worker pings a background heartbeat every ~15s
# (prediction_worker._HEARTBEAT_INTERVAL_SEC) independent of day-processing
# progress, so this threshold only needs to tolerate a handful of
# missed/delayed pings (e.g.
# transient SQLite lock contention from sibling workers) — not the duration
# of any single processing stage. Do not lower this without confirming the
# heartbeat thread is still running; before that thread existed, a single
# ~90-180s blocking stage (tick-timeline load for a heavy trading day) could
# starve the heartbeat and cause a false "Worker process exited before day
# completed" verdict while the worker was still healthy and finished on its
# own moments later.
_STALE_HEARTBEAT_SEC = 120.0
_SPAWN_GRACE_SEC = 45.0


def _pid_alive(pid: int | None) -> bool:
    if not pid or int(pid) <= 0:
        return False
    pid = int(pid)
    try:
        if sys.platform == "win32":
            import ctypes

            PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
            handle = ctypes.windll.kernel32.OpenProcess(  # type: ignore[attr-defined]
                PROCESS_QUERY_LIMITED_INFORMATION, False, pid
            )
            if handle:
                ctypes.windll.kernel32.CloseHandle(handle)  # type: ignore[attr-defined]
                return True
            return False
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _job_age_seconds(started_at: str | None) -> float | None:
    text = str(started_at or "").strip()
    if not text:
        return None
    try:
        from datetime import datetime, timezone

        raw = text.replace("Z", "+00:00")
        dt = datetime.fromisoformat(raw)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return max(0.0, (datetime.now(timezone.utc) - dt).total_seconds())
    except (TypeError, ValueError):
        return None


def _worker_effectively_alive(
    worker: dict[str, Any],
    *,
    job_age_sec: float | None,
) -> bool:
    """PID alive + fresh heartbeat (or still inside spawn grace)."""
    status = str(worker.get("status") or "")
    if status in (WORKER_STATUS_DONE, WORKER_STATUS_FAILED):
        return False
    if not _pid_alive(worker.get("pid")):
        return False
    hb_age = _job_age_seconds(worker.get("heartbeat_at"))
    if hb_age is None:
        # Spawned but no heartbeat yet — only trust during grace window.
        return job_age_sec is not None and job_age_sec < _SPAWN_GRACE_SEC
    return hb_age < _STALE_HEARTBEAT_SEC


def _terminate_pid(pid: int | None) -> None:
    if not pid or int(pid) <= 0 or not _pid_alive(pid):
        return
    pid = int(pid)
    try:
        if sys.platform == "win32":
            import ctypes

            PROCESS_TERMINATE = 0x0001
            handle = ctypes.windll.kernel32.OpenProcess(  # type: ignore[attr-defined]
                PROCESS_TERMINATE, False, pid
            )
            if handle:
                ctypes.windll.kernel32.TerminateProcess(handle, 1)  # type: ignore[attr-defined]
                ctypes.windll.kernel32.CloseHandle(handle)  # type: ignore[attr-defined]
        else:
            os.kill(pid, 15)
    except OSError:
        pass


class PredictionManager:
    """Create jobs, launch OS workers, pause/cancel via SQLite."""

    def __init__(self, lab_db_path: str) -> None:
        self.lab_db_path = os.path.abspath(lab_db_path)
        self._procs: dict[str, list[subprocess.Popen]] = {}

    def create_and_start(
        self,
        data_dir: str,
        *,
        overwrite: bool = False,
        resume: bool = True,
        selected_days: list[str] | None = None,
        enrich_path_outcomes: bool = True,
        row_limit: int | None = None,
        mark_day_complete: bool = True,
        tb_model_name: str | None = None,
        worker_count: int = DEFAULT_EXEC_WORKERS,
        python_exe: str | None = None,
        on_stage: Any = None,
    ) -> dict[str, Any]:
        """Prepare durable job + spawn worker processes. Returns job snapshot."""
        n_workers = max(1, min(int(worker_count), MAX_EXEC_WORKERS))
        prepared = prepare_prediction_exec_config(
            data_dir,
            self.lab_db_path,
            overwrite=overwrite,
            resume=resume,
            selected_days=selected_days,
            enrich_path_outcomes=enrich_path_outcomes,
            row_limit=row_limit,
            mark_day_complete=mark_day_complete,
            tb_model_name=tb_model_name,
            on_stage=on_stage,
        )
        if not prepared.get("ok"):
            return prepared

        if on_stage:
            try:
                on_stage("Spawning prediction worker process(es)…")
            except Exception:
                pass
        days_to_run = list(prepared.get("days_to_run") or [])
        if not days_to_run:
            return {"ok": False, "error": "No days selected to build"}

        n_workers = max(1, min(n_workers, len(days_to_run)))
        assignments = assign_days_round_robin(days_to_run, n_workers)
        job_id = uuid.uuid4().hex[:12]
        job = create_job(
            self.lab_db_path,
            job_id=job_id,
            lab_uuid=str(prepared["lab_uuid"]),
            data_dir=data_dir,
            worker_count=n_workers,
            config=dict(prepared["config"]),
            day_assignments=assignments,
            overwrite=overwrite,
            resume=resume,
            row_limit=row_limit,
            mark_day_complete=mark_day_complete,
        )
        set_job_status(self.lab_db_path, job_id, JOB_STATUS_RUNNING, started=True)
        spawn = self.spawn_workers(job_id, python_exe=python_exe)
        return {
            "ok": True,
            "job_id": job_id,
            "job": job,
            "worker_count": n_workers,
            "days_to_run": days_to_run,
            "spawned": spawn,
            "skipped_days": list(prepared.get("skipped_days") or []),
        }

    def spawn_workers(
        self,
        job_id: str,
        *,
        python_exe: str | None = None,
        worker_ids: list[int] | None = None,
    ) -> list[dict[str, Any]]:
        job = get_job(self.lab_db_path, job_id)
        if not job:
            return [{"ok": False, "error": "job not found"}]
        exe = python_exe or sys.executable
        logs = _logs_dir(self.lab_db_path)
        # Ensure chart package root on PYTHONPATH
        chart_root = str(Path(__file__).resolve().parents[2])
        env = os.environ.copy()
        prev = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = (
            chart_root + os.pathsep + prev if prev else chart_root
        )

        ids = worker_ids or list(range(1, int(job["worker_count"]) + 1))
        launched: list[dict[str, Any]] = []
        procs: list[subprocess.Popen] = []
        for wid in ids:
            log_path = os.path.join(logs, f"prediction_worker_{job_id}_{wid}.log")
            cmd = [
                exe,
                "-m",
                _worker_module(),
                "--job-id",
                job_id,
                "--worker-id",
                str(wid),
                "--lab-db",
                self.lab_db_path,
                "--log-file",
                log_path,
            ]
            # Detached-ish: workers survive GUI exit on Windows via CREATE_NEW_PROCESS_GROUP
            creationflags = 0
            if sys.platform == "win32":
                creationflags = subprocess.CREATE_NEW_PROCESS_GROUP  # type: ignore[attr-defined]
            log_fh = open(log_path, "a", encoding="utf-8")
            try:
                proc = subprocess.Popen(
                    cmd,
                    cwd=chart_root,
                    env=env,
                    stdout=log_fh,
                    stderr=subprocess.STDOUT,
                    creationflags=creationflags,
                    close_fds=False if sys.platform == "win32" else True,
                )
            except Exception as exc:
                log_fh.close()
                launched.append({"ok": False, "worker_id": wid, "error": str(exc)})
                continue
            # Register OS PID immediately so progress polls don't treat the
            # worker as dead before it finishes importing / writing its heartbeat.
            try:
                update_worker(
                    self.lab_db_path,
                    job_id,
                    int(wid),
                    pid=int(proc.pid),
                    status=WORKER_STATUS_RUNNING,
                    last_message="Spawned",
                    started=True,
                )
            except Exception:
                pass
            procs.append(proc)
            launched.append(
                {
                    "ok": True,
                    "worker_id": wid,
                    "pid": proc.pid,
                    "log_path": log_path,
                }
            )
            # Keep log_fh alive with process (Popen holds it)
        self._procs[job_id] = procs
        return launched

    def pause(self, job_id: str) -> None:
        request_job_pause(self.lab_db_path, job_id)

    def resume(self, job_id: str, *, python_exe: str | None = None) -> dict[str, Any]:
        """Mark job running and re-spawn workers that are not alive."""
        request_job_resume(self.lab_db_path, job_id)
        workers = list_workers(self.lab_db_path, job_id)
        need = [
            int(w["worker_id"])
            for w in workers
            if w.get("status") not in (WORKER_STATUS_DONE,)
            and not _pid_alive(w.get("pid"))
        ]
        spawned = self.spawn_workers(job_id, python_exe=python_exe, worker_ids=need) if need else []
        return {"ok": True, "job_id": job_id, "respawned": need, "spawned": spawned}

    def cancel(self, job_id: str) -> None:
        request_job_cancel(self.lab_db_path, job_id)

    def _count_alive_workers(
        self,
        workers: list[dict[str, Any]],
        *,
        job_age_sec: float | None,
    ) -> int:
        return sum(
            1
            for w in workers
            if _worker_effectively_alive(w, job_age_sec=job_age_sec)
        )

    def _finalize_dead_job(
        self,
        job_id: str,
        snap: dict[str, Any],
        *,
        reason: str,
    ) -> dict[str, Any]:
        """Mark remaining days/workers failed and close the job as failed."""
        workers = list(snap.get("workers") or [])
        checkpoints = list(snap.get("checkpoints") or [])
        remaining = [
            c
            for c in checkpoints
            if c.get("status") in ("pending", "in_progress")
        ]
        for c in remaining:
            day = str(c.get("trading_day") or "")
            if not day:
                continue
            update_checkpoint(
                self.lab_db_path,
                job_id,
                day,
                status=DAY_CP_FAILED,
                error_message=reason,
                finished=True,
            )
        for w in workers:
            if w.get("status") not in (WORKER_STATUS_DONE, WORKER_STATUS_FAILED):
                update_worker(
                    self.lab_db_path,
                    job_id,
                    int(w["worker_id"]),
                    status=WORKER_STATUS_FAILED,
                    last_message=reason,
                    finished=True,
                )
        any_fail = bool(remaining) or any(
            c.get("status") == "failed" for c in checkpoints
        )
        # Prefer a specific per-day failure already recorded by the worker
        # (e.g. "No rows for <day> in parent parquet or Master Dataset
        # (target=...)") over the generic `reason` used to detect a dead
        # worker. A worker that fails a day and then exits cleanly (no more
        # pending days) looks identical, from here, to a worker that crashed
        # with zero diagnostics — but only the latter should surface the
        # opaque "Worker process exited before day completed" message. Using
        # the day-level reason keeps the real cause visible in the UI.
        day_reason = next(
            (
                str(c.get("error_message") or "").strip()
                for c in checkpoints
                if c.get("status") == "failed"
                and str(c.get("error_message") or "").strip()
            ),
            "",
        )
        error_message = day_reason or reason
        set_job_status(
            self.lab_db_path,
            job_id,
            JOB_STATUS_FAILED if any_fail or remaining else JOB_STATUS_COMPLETED,
            error_message=error_message if (any_fail or remaining) else None,
            finished=True,
        )
        return job_progress_snapshot(self.lab_db_path, job_id)

    def progress(self, job_id: str, *, timeout_sec: float = 1.5) -> dict[str, Any]:
        snap = job_progress_snapshot(
            self.lab_db_path, job_id, timeout_sec=timeout_sec
        )
        workers = list(snap.get("workers") or [])
        started_at = str(
            (snap.get("job") or {}).get("started_at")
            or snap.get("started_at")
            or ""
        )
        age_sec = _job_age_seconds(started_at)
        alive = self._count_alive_workers(workers, job_age_sec=age_sec)
        snap["workers_alive"] = alive
        status = str(snap.get("status") or "")
        # Auto-finalize when no effective workers remain and job is still open.
        if status in (JOB_STATUS_RUNNING, JOB_STATUS_PENDING) and alive == 0:
            # Grace: freshly spawned workers may not have heartbeats yet.
            never_bound = any(
                not w.get("pid")
                and w.get("status")
                not in (WORKER_STATUS_DONE, WORKER_STATUS_FAILED)
                for w in workers
            )
            if age_sec is not None and age_sec < 20.0:
                return snap
            if never_bound and age_sec is not None and age_sec < _SPAWN_GRACE_SEC:
                return snap
            # Empty worker table (crash before spawn register) or all dead/stale.
            if not workers or all(
                w.get("status") in (WORKER_STATUS_DONE, WORKER_STATUS_FAILED)
                or not _worker_effectively_alive(w, job_age_sec=age_sec)
                for w in workers
            ):
                snap = self._finalize_dead_job(
                    job_id,
                    snap,
                    reason="Worker process exited before day completed",
                )
        snap["workers_alive"] = self._count_alive_workers(
            list(snap.get("workers") or []),
            job_age_sec=_job_age_seconds(
                str(
                    (snap.get("job") or {}).get("started_at")
                    or snap.get("started_at")
                    or ""
                )
            ),
        )
        return snap

    def reclaim_stale_active_jobs(self) -> list[dict[str, Any]]:
        """Finalize orphaned active jobs after GUI relaunch / worker death.

        Returns one progress snapshot per previously-active job (post-reclaim).
        """
        jobs = list_active_jobs(self.lab_db_path)
        out: list[dict[str, Any]] = []
        for job in jobs:
            jid = str(job["job_id"])
            snap = self.progress(jid)
            status = str(snap.get("status") or "")
            alive = int(snap.get("workers_alive") or 0)
            # Still open with zero effective workers → force-fail (covers empty
            # worker rows and heartbeat-stale PID reuse after relaunch).
            if status in (JOB_STATUS_RUNNING, JOB_STATUS_PENDING, JOB_STATUS_PAUSED) and alive <= 0:
                snap = self._finalize_dead_job(
                    jid,
                    snap,
                    reason="Stale job reclaimed after relaunch (no live workers)",
                )
                snap["workers_alive"] = 0
                snap["reclaimed"] = True
            snap["job_id"] = jid
            out.append(snap)
        return out

    def abandon_job(self, job_id: str, *, reason: str | None = None) -> dict[str, Any]:
        """Terminate worker PIDs and mark the job failed so Build unlocks."""
        msg = reason or "Abandoned by user after relaunch"
        workers = list_workers(self.lab_db_path, job_id)
        for w in workers:
            _terminate_pid(w.get("pid"))
        snap = job_progress_snapshot(self.lab_db_path, job_id)
        snap = self._finalize_dead_job(job_id, snap, reason=msg)
        # Prefer cancelled semantics for explicit user abandon.
        set_job_status(
            self.lab_db_path,
            job_id,
            JOB_STATUS_CANCELLED,
            error_message=msg,
            finished=True,
        )
        self._procs.pop(job_id, None)
        out = job_progress_snapshot(self.lab_db_path, job_id)
        out["workers_alive"] = 0
        out["abandoned"] = True
        return out

    def reattach(self) -> list[dict[str, Any]]:
        """Find active jobs for this lab and reclaim orphans / report alive state."""
        return self.reclaim_stale_active_jobs()

    def active_job_id(self) -> str | None:
        jobs = list_active_jobs(self.lab_db_path)
        for job in jobs:
            if job.get("status") in (
                JOB_STATUS_PENDING,
                JOB_STATUS_RUNNING,
                JOB_STATUS_PAUSED,
            ):
                return str(job["job_id"])
        return None

    def is_busy(self) -> bool:
        """True only while worker OS processes are actually alive.

        A job row left as ``running`` after a crash must not lock Research Lab
        Build buttons forever.
        """
        jid = self.active_job_id()
        if not jid:
            return False
        snap = self.progress(jid, timeout_sec=0.5)
        status = str(snap.get("status") or "")
        if status in (
            JOB_STATUS_COMPLETED,
            JOB_STATUS_FAILED,
            JOB_STATUS_CANCELLED,
            JOB_STATUS_PAUSED,
        ):
            return False
        return int(snap.get("workers_alive") or 0) > 0

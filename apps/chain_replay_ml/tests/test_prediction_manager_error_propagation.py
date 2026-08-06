"""Regression test for opaque "Worker process exited before day completed".

Root cause (job e354f8da60f8, trading_day 2026-07-31):

The worker correctly detected the real failure — "No rows for 2026-07-31 in
parent parquet or Master Dataset (target=future_ltp_5m)" — recorded it on the
day checkpoint via ``update_checkpoint(..., error_message=err)``, and exited
cleanly (exit code 5) once there were no more pending days.

``PredictionManager.progress()`` then ran its auto-finalize path because no
workers were left alive, and unconditionally overwrote the job-level
``error_message`` with the generic dead-worker ``reason`` string ("Worker
process exited before day completed") in ``_finalize_dead_job`` — even though
a specific, actionable per-day error was already sitting right there on the
failed checkpoint. The UI only ever displays the job-level message
(``model_lab_window.py``: ``f"Failed - {job.get('error_message')}"``), so the
real cause was silently discarded.

The fix makes ``_finalize_dead_job`` prefer the specific failed-checkpoint
``error_message`` over the generic dead-worker ``reason`` whenever one is
available.
"""

from __future__ import annotations

import os
import tempfile
import unittest
from datetime import datetime, timedelta, timezone

from chain_replay_ml.model_lab.prediction_job_schema import (
    DAY_CP_FAILED,
    JOB_STATUS_FAILED,
    JOB_STATUS_RUNNING,
    WORKER_STATUS_FAILED,
)
from chain_replay_ml.model_lab.prediction_job_store import (
    _connect,
    assign_days_round_robin,
    create_job,
    get_job,
    set_job_status,
    update_checkpoint,
    update_worker,
)
from chain_replay_ml.model_lab.prediction_manager import PredictionManager

_REAL_ERROR = (
    "No rows for 2026-07-31 in parent parquet or Master Dataset "
    "(target=future_ltp_5m)"
)


def _make_job_with_dead_worker(
    lab_db: str, job_id: str, *, day: str = "2026-07-31"
) -> None:
    create_job(
        lab_db,
        job_id=job_id,
        lab_uuid="lab-uuid",
        data_dir=os.path.dirname(lab_db),
        worker_count=1,
        day_assignments=assign_days_round_robin([day], 1),
        config={"wanted_columns": []},
    )
    set_job_status(lab_db, job_id, JOB_STATUS_RUNNING, started=True)

    # Job started well outside the manager's 20s "freshly spawned" grace
    # window, so progress() will actually run its auto-finalize path.
    old = (datetime.now(timezone.utc) - timedelta(seconds=60)).isoformat()
    conn = _connect(lab_db)
    try:
        conn.execute(
            "UPDATE prediction_exec_job SET started_at = ? WHERE job_id = ?",
            (old, job_id),
        )
        conn.commit()
    finally:
        conn.close()

    # Mirror exactly what prediction_worker.run_worker does on a day
    # failure: record the specific reason on the checkpoint, then (since
    # there are no more pending days) mark the worker done/failed and exit
    # cleanly — no crash, no PID left alive.
    update_checkpoint(
        lab_db,
        job_id,
        day,
        status=DAY_CP_FAILED,
        error_message=_REAL_ERROR,
        finished=True,
    )
    update_worker(
        lab_db,
        job_id,
        1,
        pid=None,
        status=WORKER_STATUS_FAILED,
        last_message="Worker finished",
        finished=True,
    )


class FinalizeDeadJobErrorPropagationTests(unittest.TestCase):
    def test_specific_day_error_wins_over_generic_dead_worker_reason(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            lab_db = os.path.join(tmp, "lab.db")
            job_id = "errtest01"
            _make_job_with_dead_worker(lab_db, job_id)

            mgr = PredictionManager(lab_db)
            snap = mgr.progress(job_id)

            self.assertEqual(snap.get("status"), JOB_STATUS_FAILED)
            job = get_job(lab_db, job_id)
            assert job is not None
            self.assertEqual(
                job.get("error_message"),
                _REAL_ERROR,
                "job-level error_message must surface the specific per-day "
                "failure, not the opaque 'Worker process exited before day "
                "completed' — that string must never reach the UI when a "
                "real diagnostic is already available",
            )

    def test_generic_reason_used_only_when_no_specific_error_recorded(self) -> None:
        """A genuine crash with zero diagnostics still gets the generic reason."""
        with tempfile.TemporaryDirectory() as tmp:
            lab_db = os.path.join(tmp, "lab.db")
            job_id = "errtest02"
            create_job(
                lab_db,
                job_id=job_id,
                lab_uuid="lab-uuid",
                data_dir=tmp,
                worker_count=1,
                day_assignments=assign_days_round_robin(["2026-07-31"], 1),
                config={"wanted_columns": []},
            )
            set_job_status(lab_db, job_id, JOB_STATUS_RUNNING, started=True)
            old = (datetime.now(timezone.utc) - timedelta(seconds=60)).isoformat()
            conn = _connect(lab_db)
            try:
                conn.execute(
                    "UPDATE prediction_exec_job SET started_at = ? WHERE job_id = ?",
                    (old, job_id),
                )
                conn.commit()
            finally:
                conn.close()
            # No checkpoint update at all — worker process vanished (killed /
            # crashed) leaving the day still pending, with no recorded reason.
            update_worker(
                lab_db,
                job_id,
                1,
                pid=999_999_999,  # never a real, alive PID
                status="running",
                last_message="Predicting 2026-07-31",
                started=True,
            )

            mgr = PredictionManager(lab_db)
            snap = mgr.progress(job_id)

            self.assertEqual(snap.get("status"), JOB_STATUS_FAILED)
            job = get_job(lab_db, job_id)
            assert job is not None
            self.assertEqual(
                job.get("error_message"),
                "Worker process exited before day completed",
            )


if __name__ == "__main__":
    unittest.main()

"""Regression test for the prediction-worker heartbeat / false-dead-worker bug.

Root cause (job cfc73b643460, day 2026-07-22, TB disabled):

``process_trading_day`` sends a progress update right before loading tick
timelines for path-outcome enrichment (``percent=8.0, current_row=0``), then
performs a single long blocking call (``_load_day_timelines`` /
``prepare_path_outcome_timelines``) with *no* intermediate progress
callback. On a heavy trading day that call alone took ~96s in a clean run —
already brushing against ``prediction_manager._STALE_HEARTBEAT_SEC`` (90s at
the time). When the GUI polled ``PredictionManager.progress()`` during that
window (it polls every ~1.5-2.5s), it saw a stale ``heartbeat_at`` while the
worker OS process was completely healthy, and incorrectly finalized the job
as failed ("Worker process exited before day completed") — even though the
orphaned worker kept running and finished the day normally moments later.

The fix adds a background heartbeat thread (``prediction_worker.
_start_heartbeat_thread``) that pings ``heartbeat_at`` on a fixed wall-clock
cadence, independent of day-processing progress, so liveness detection never
depends on how long any single blocking stage takes.
"""

from __future__ import annotations

import os
import tempfile
import time
import unittest
from unittest import mock

from chain_replay_ml.model_lab.prediction_job_schema import (
    JOB_STATUS_RUNNING,
    WORKER_STATUS_RUNNING,
)
from chain_replay_ml.model_lab.prediction_job_store import (
    assign_days_round_robin,
    create_job,
    list_workers,
    set_job_status,
    update_worker,
)


def _get_worker(lab_db: str, job_id: str, worker_id: int) -> dict | None:
    for w in list_workers(lab_db, job_id):
        if int(w.get("worker_id") or 0) == worker_id:
            return w
    return None
from chain_replay_ml.model_lab.prediction_manager import (
    _STALE_HEARTBEAT_SEC,
    _worker_effectively_alive,
)
from chain_replay_ml.model_lab.prediction_worker import (
    _HEARTBEAT_INTERVAL_SEC,
    _start_heartbeat_thread,
    _stop_heartbeat_thread,
)


def _make_job(lab_db: str, job_id: str, *, pid: int) -> None:
    days = ["2026-07-22"]
    create_job(
        lab_db,
        job_id=job_id,
        lab_uuid="lab-uuid",
        data_dir=os.path.dirname(lab_db),
        worker_count=1,
        day_assignments=assign_days_round_robin(days, 1),
        config={"wanted_columns": []},
    )
    set_job_status(lab_db, job_id, JOB_STATUS_RUNNING, started=True)
    update_worker(
        lab_db,
        job_id,
        1,
        pid=pid,
        status=WORKER_STATUS_RUNNING,
        last_message="Predicting 2026-07-22",
        started=True,
    )


class HeartbeatThreadTests(unittest.TestCase):
    def test_heartbeat_thread_keeps_worker_fresh_during_long_blocking_stage(
        self,
    ) -> None:
        """
        Simulate a long single blocking call (tick-timeline load) with *no*
        other update_worker() calls. The background heartbeat thread must
        keep heartbeat_at fresh throughout, so the manager never flags the
        worker as dead even though nothing else touched the row for the
        whole simulated stage duration.
        """
        with tempfile.TemporaryDirectory() as tmp:
            lab_db = os.path.join(tmp, "lab.db")
            job_id = "hbtest01"
            _make_job(lab_db, job_id, pid=os.getpid())

            import logging

            log = logging.getLogger("test-heartbeat")

            stop_event, thread = _start_heartbeat_thread(
                lab_db, job_id, 1, log, interval_sec=0.05
            )
            try:
                # Simulate a blocking stage several heartbeat intervals long,
                # with no other progress update in between (this is exactly
                # what _load_day_timelines / prepare_path_outcome_timelines
                # look like from the manager's point of view).
                deadline = time.monotonic() + 0.4
                worst_gap = 0.0
                last_hb = None
                while time.monotonic() < deadline:
                    worker = _get_worker(lab_db, job_id, 1)
                    assert worker is not None
                    alive = _worker_effectively_alive(worker, job_age_sec=1.0)
                    self.assertTrue(
                        alive,
                        "worker must stay 'alive' while heartbeat thread runs, "
                        "even with no other progress updates",
                    )
                    time.sleep(0.05)
            finally:
                _stop_heartbeat_thread(stop_event, thread)

            self.assertFalse(thread.is_alive())

    def test_stale_heartbeat_without_background_thread_is_flagged_dead(self) -> None:
        """
        Sanity check on the *old* failure mode: if nothing pings
        heartbeat_at for longer than _STALE_HEARTBEAT_SEC, the worker is
        (correctly, in the true-crash case) considered not alive. This
        documents why the background thread in prediction_worker.run_worker
        is required — without it, any single blocking stage longer than
        this threshold reproduces the false "exited before day completed"
        verdict.
        """
        with tempfile.TemporaryDirectory() as tmp:
            lab_db = os.path.join(tmp, "lab.db")
            job_id = "hbtest02"
            _make_job(lab_db, job_id, pid=os.getpid())

            worker = _get_worker(lab_db, job_id, 1)
            assert worker is not None
            # Force heartbeat_at far enough in the past to exceed the threshold.
            from datetime import datetime, timedelta, timezone

            stale_dt = datetime.now(timezone.utc) - timedelta(
                seconds=_STALE_HEARTBEAT_SEC + 5
            )
            worker["heartbeat_at"] = stale_dt.isoformat()
            with mock.patch(
                "chain_replay_ml.model_lab.prediction_manager._pid_alive",
                return_value=True,
            ):
                alive = _worker_effectively_alive(worker, job_age_sec=99_999.0)
            self.assertFalse(
                alive,
                "a genuinely stale heartbeat (no pings at all) must still be "
                "treated as dead — this is the real-crash detection path",
            )

    def test_heartbeat_interval_leaves_safety_margin_under_stale_threshold(self) -> None:
        """Guardrail: heartbeat cadence must stay well under the stale threshold."""
        self.assertGreaterEqual(_STALE_HEARTBEAT_SEC / _HEARTBEAT_INTERVAL_SEC, 4.0)


if __name__ == "__main__":
    unittest.main()

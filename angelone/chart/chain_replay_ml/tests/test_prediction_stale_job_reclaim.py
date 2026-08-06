"""Stale prediction job reclaim after GUI relaunch."""

from __future__ import annotations

import os
import tempfile
import unittest
from unittest import mock

from chain_replay_ml.model_lab.prediction_job_schema import (
    DAY_CP_PENDING,
    JOB_STATUS_FAILED,
    JOB_STATUS_RUNNING,
    WORKER_STATUS_RUNNING,
)
from chain_replay_ml.model_lab.prediction_job_store import (
    assign_days_round_robin,
    create_job,
    get_job,
    list_active_jobs,
    set_job_status,
    update_checkpoint,
    update_worker,
)
from chain_replay_ml.model_lab.prediction_manager import PredictionManager


class StaleJobReclaimTests(unittest.TestCase):
    def test_reclaim_running_job_with_dead_worker(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            lab_db = os.path.join(tmp, "lab.db")
            days = ["2026-05-26"]
            create_job(
                lab_db,
                job_id="stalejob01",
                lab_uuid="lab-uuid",
                data_dir=tmp,
                worker_count=1,
                day_assignments=assign_days_round_robin(days, 1),
                config={"wanted_columns": []},
            )
            set_job_status(lab_db, "stalejob01", JOB_STATUS_RUNNING, started=True)
            update_worker(
                lab_db,
                "stalejob01",
                1,
                pid=9_999_999,  # almost certainly not alive
                status=WORKER_STATUS_RUNNING,
                last_message="orphaned",
                started=True,
            )
            update_checkpoint(
                lab_db,
                "stalejob01",
                "2026-05-26",
                status=DAY_CP_PENDING,
            )

            mgr = PredictionManager(lab_db)
            with mock.patch(
                "chain_replay_ml.model_lab.prediction_manager._pid_alive",
                return_value=False,
            ):
                snaps = mgr.reclaim_stale_active_jobs()

            self.assertEqual(len(snaps), 1)
            self.assertTrue(snaps[0].get("reclaimed"))
            job = get_job(lab_db, "stalejob01")
            assert job is not None
            self.assertEqual(job["status"], JOB_STATUS_FAILED)
            self.assertEqual(list_active_jobs(lab_db), [])
            self.assertFalse(mgr.is_busy())

    def test_abandon_terminates_and_cancels(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            lab_db = os.path.join(tmp, "lab.db")
            create_job(
                lab_db,
                job_id="abandon01",
                lab_uuid="lab-uuid",
                data_dir=tmp,
                worker_count=1,
                day_assignments=assign_days_round_robin(["2026-05-26"], 1),
                config={},
            )
            set_job_status(lab_db, "abandon01", JOB_STATUS_RUNNING, started=True)
            update_worker(
                lab_db,
                "abandon01",
                1,
                pid=12345,
                status=WORKER_STATUS_RUNNING,
                started=True,
            )
            mgr = PredictionManager(lab_db)
            with mock.patch(
                "chain_replay_ml.model_lab.prediction_manager._terminate_pid"
            ) as term:
                snap = mgr.abandon_job("abandon01")
            term.assert_called()
            self.assertTrue(snap.get("abandoned"))
            job = get_job(lab_db, "abandon01")
            assert job is not None
            self.assertEqual(job["status"], "cancelled")
            self.assertEqual(list_active_jobs(lab_db), [])


if __name__ == "__main__":
    unittest.main()

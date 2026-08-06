"""Tests for external prediction job assignment + checkpoint resume."""

from __future__ import annotations

import os
import tempfile
import unittest

from chain_replay_ml.model_lab.prediction_job_schema import (
    DAY_CP_COMPLETED,
    DAY_CP_IN_PROGRESS,
    DAY_CP_PENDING,
)
from chain_replay_ml.model_lab.prediction_job_store import (
    assign_days_round_robin,
    create_job,
    get_job,
    list_checkpoints,
    update_checkpoint,
    worker_pending_days,
)


class AssignDaysRoundRobinTests(unittest.TestCase):
    def test_each_day_owned_by_exactly_one_worker(self) -> None:
        days = [f"2026-06-{i:02d}" for i in range(1, 10)]
        assignments = assign_days_round_robin(days, 3)
        self.assertEqual(len(assignments), len(days))
        by_day = {d: w for d, w in assignments}
        self.assertEqual(sorted(by_day), sorted(days))
        # Never split: one worker id per day
        self.assertEqual(len(by_day), len(days))
        # Round-robin pattern
        self.assertEqual(by_day["2026-06-01"], 1)
        self.assertEqual(by_day["2026-06-02"], 2)
        self.assertEqual(by_day["2026-06-03"], 3)
        self.assertEqual(by_day["2026-06-04"], 1)

    def test_worker_count_capped_by_days_not_required_here(self) -> None:
        assignments = assign_days_round_robin(["2026-06-01"], 4)
        self.assertEqual(assignments, [("2026-06-01", 1)])


class CheckpointResumeTests(unittest.TestCase):
    def test_create_job_and_pending_resume(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            lab_db = os.path.join(tmp, "lab.db")
            days = ["2026-06-01", "2026-06-02", "2026-06-03"]
            assignments = assign_days_round_robin(days, 3)
            # One day per worker for 3 days / 3 workers
            self.assertEqual(
                {(d, w) for d, w in assignments},
                {("2026-06-01", 1), ("2026-06-02", 2), ("2026-06-03", 3)},
            )
            job = create_job(
                lab_db,
                job_id="jobtest01",
                lab_uuid="lab-uuid",
                data_dir=tmp,
                worker_count=3,
                config={"data_dir": tmp, "features": []},
                day_assignments=assignments,
            )
            self.assertEqual(job["job_id"], "jobtest01")
            self.assertEqual(int(job["days_total"]), 3)

            cps = list_checkpoints(lab_db, "jobtest01")
            self.assertEqual(len(cps), 3)
            owners = {c["trading_day"]: c["worker_id"] for c in cps}
            self.assertEqual(owners["2026-06-01"], 1)
            self.assertEqual(owners["2026-06-02"], 2)
            self.assertEqual(owners["2026-06-03"], 3)

            # Worker 1 mid-day checkpoint
            update_checkpoint(
                lab_db,
                "jobtest01",
                "2026-06-01",
                status=DAY_CP_IN_PROGRESS,
                rows_committed=2000,
                rows_expected=5000,
                started=True,
            )
            pending = worker_pending_days(lab_db, "jobtest01", 1)
            self.assertEqual(len(pending), 1)
            self.assertEqual(pending[0]["trading_day"], "2026-06-01")
            self.assertEqual(pending[0]["rows_committed"], 2000)

            # Worker 2 has not started — full pending day
            pending2 = worker_pending_days(lab_db, "jobtest01", 2)
            self.assertEqual(len(pending2), 1)
            self.assertEqual(pending2[0]["status"], DAY_CP_PENDING)

            # Complete day 1 → worker 1 has nothing left
            update_checkpoint(
                lab_db,
                "jobtest01",
                "2026-06-01",
                status=DAY_CP_COMPLETED,
                rows_committed=5000,
                finished=True,
            )
            self.assertEqual(worker_pending_days(lab_db, "jobtest01", 1), [])

            # Workers never share days
            for wid in (1, 2, 3):
                for other in worker_pending_days(lab_db, "jobtest01", wid):
                    cp = next(
                        c
                        for c in list_checkpoints(lab_db, "jobtest01")
                        if c["trading_day"] == other["trading_day"]
                    )
                    self.assertEqual(cp["worker_id"], wid)

            self.assertIsNotNone(get_job(lab_db, "jobtest01"))


if __name__ == "__main__":
    unittest.main()

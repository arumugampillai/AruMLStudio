"""Pre-flight day-eligibility regression test.

Root cause (job e354f8da60f8, trading_day 2026-07-31):

The parent analysis parquet was built at 04:47:47 on 2026-07-31 itself
(before that day's samples existed), so it has zero rows for that day. The
worker's Master-Dataset fallback *does* find rows for 2026-07-31, but
Master's ``samples`` table only stores near-term forward labels
(``future_ltp_10s`` / ``future_ltp_1m``) — never the analysis-time
``future_ltp_5m`` target this job needed. So the day was offered/queued for
Build (it exists in Master's day catalog), but had zero usable rows for the
configured target in *either* source. The worker only discovered this ~40s
into the job, after spawning a whole worker process.

``_filter_days_with_target_coverage`` (``prediction_job_prepare.py``) runs
the same parent-parquet / Master-with-target checks *before* a job is
created, so an unbuildable day is excluded from ``days_to_run`` (and marked
``DAY_SKIPPED`` with a specific reason) instead of silently burning worker
time before failing.
"""

from __future__ import annotations

import os
import tempfile
import unittest

import pandas as pd

from chain_replay_ml.dataset_builder.master_store import MasterStore
from chain_replay_ml.model_lab.prediction_job_prepare import (
    _filter_days_with_target_coverage,
)

TARGET = "future_ltp_5m"


def _write_parquet(path: str, days_rows: dict[str, int]) -> None:
    rows: list[dict] = []
    ts = 1_700_000_000.0
    for day, n in days_rows.items():
        for i in range(n):
            rows.append(
                {
                    "trading_day": day,
                    "timestamp": ts + i,
                    "token": "t1",
                    TARGET: 100.0 + i,
                }
            )
    pd.DataFrame(rows).to_parquet(path)


class DayEligibilityTests(unittest.TestCase):
    def test_day_missing_everywhere_is_unbuildable_with_specific_reason(self) -> None:
        """2026-07-31-style case: 0 rows in parquet, Master lacks target col."""
        with tempfile.TemporaryDirectory() as tmp:
            parquet_path = os.path.join(tmp, "analysis.parquet")
            _write_parquet(parquet_path, {"2026-07-22": 5})  # 07-31 absent

            master_path = os.path.join(tmp, "master.db")
            with MasterStore(master_path) as store:
                # Master carries only near-term labels — never future_ltp_5m.
                cols = ["trading_day", "timestamp", "token", "future_ltp_1m"]
                store.begin_day("2026-07-31", cols)
                store.insert_rows(
                    [
                        {
                            "trading_day": "2026-07-31",
                            "timestamp": 1.0,
                            "token": "t1",
                            "future_ltp_1m": 1.0,
                        }
                    ]
                )
                store.commit_day("2026-07-31")

            buildable, unbuildable = _filter_days_with_target_coverage(
                ["2026-07-22", "2026-07-31"],
                parquet_path=parquet_path,
                pq_cols={"trading_day", "timestamp", "token", TARGET},
                target=TARGET,
                master_abs=master_path,
                master_filter=None,
            )

            self.assertEqual(buildable, ["2026-07-22"])
            self.assertEqual(len(unbuildable), 1)
            day, reason = unbuildable[0]
            self.assertEqual(day, "2026-07-31")
            self.assertIn("future_ltp_5m", reason)
            self.assertIn("Master", reason)
            self.assertIn("rebuild", reason.lower())

    def test_day_available_via_master_fallback_when_target_present(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            parquet_path = os.path.join(tmp, "analysis.parquet")
            _write_parquet(parquet_path, {"2026-07-22": 5})  # target day absent

            master_path = os.path.join(tmp, "master.db")
            with MasterStore(master_path) as store:
                cols = ["trading_day", "timestamp", "token", TARGET]
                store.begin_day("2026-07-31", cols)
                store.insert_rows(
                    [
                        {
                            "trading_day": "2026-07-31",
                            "timestamp": 1.0,
                            "token": "t1",
                            TARGET: 42.0,
                        }
                    ]
                )
                store.commit_day("2026-07-31")

            buildable, unbuildable = _filter_days_with_target_coverage(
                ["2026-07-31"],
                parquet_path=parquet_path,
                pq_cols={"trading_day", "timestamp", "token", TARGET},
                target=TARGET,
                master_abs=master_path,
                master_filter=None,
            )

            self.assertEqual(buildable, ["2026-07-31"])
            self.assertEqual(unbuildable, [])

    def test_day_present_in_parent_parquet_is_buildable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            parquet_path = os.path.join(tmp, "analysis.parquet")
            _write_parquet(parquet_path, {"2026-07-22": 5})

            buildable, unbuildable = _filter_days_with_target_coverage(
                ["2026-07-22"],
                parquet_path=parquet_path,
                pq_cols={"trading_day", "timestamp", "token", TARGET},
                target=TARGET,
                master_abs=None,
                master_filter=None,
            )
            self.assertEqual(buildable, ["2026-07-22"])
            self.assertEqual(unbuildable, [])

    def test_day_missing_with_no_master_configured_at_all(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            parquet_path = os.path.join(tmp, "analysis.parquet")
            _write_parquet(parquet_path, {"2026-07-22": 5})

            buildable, unbuildable = _filter_days_with_target_coverage(
                ["2026-07-31"],
                parquet_path=parquet_path,
                pq_cols={"trading_day", "timestamp", "token", TARGET},
                target=TARGET,
                master_abs=None,
                master_filter=None,
            )
            self.assertEqual(buildable, [])
            self.assertEqual(len(unbuildable), 1)
            day, reason = unbuildable[0]
            self.assertEqual(day, "2026-07-31")
            self.assertNotIn("Master Dataset", reason)


if __name__ == "__main__":
    unittest.main()

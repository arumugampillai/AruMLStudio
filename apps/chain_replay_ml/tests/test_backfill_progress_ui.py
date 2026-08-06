"""Tests for backfill progress → UI wiring.

Regression coverage for a bug where a completed ~8-minute Master Dataset
Create looked stuck: the status bar kept showing stage name "Load Database"
(with a false "100%") while target-horizon backfill was running, "Day"
raced through row-scale numbers (e.g. "Day 310000/310002"), and "Samples"
stayed pinned at 0 the whole time. See `make_backfill_progress_handler` in
`master_build.py` and the `backfill_active`/`backfill_*` fields on
`BuildProgress` for the fix.
"""

from __future__ import annotations

import unittest

from chain_replay_ml.dataset_builder.master_build import (
    BACKFILL_STAGE_NAME,
    make_backfill_progress_handler,
)
from chain_replay_ml.dataset_builder.progress import BuildProgress, STAGE_NAMES
from chain_replay_ml.dataset_builder.timing import PipelineTimer


class BackfillProgressHandlerTests(unittest.TestCase):
    """Exercises the exact on_progress wiring used by MasterDatasetBuildOrchestrator."""

    def setUp(self) -> None:
        self.timer = PipelineTimer([])
        self.progress = BuildProgress(job_id="job-1")
        self.timer.start_stage(1)
        self.timer.set_stage_name(1, BACKFILL_STAGE_NAME)
        self.handler = make_backfill_progress_handler(self.timer, self.progress.emit)

    def test_stage_name_is_not_load_database_during_backfill(self) -> None:
        self.handler("2026-07-21", 0, 3, "days")
        self.assertEqual(self.progress.stage_name, BACKFILL_STAGE_NAME)
        self.assertNotEqual(self.progress.stage_name, "Load Database")
        # The pipeline timer's own stage-1 label must also be renamed — this
        # is what the "Load Database" tree/table row actually renders.
        snap = self.timer.snapshot(rows=0, estimated_total_rows=0)
        stage1 = next(s for s in snap["stages"] if s["id"] == 1)
        self.assertEqual(stage1["name"], BACKFILL_STAGE_NAME)

    def test_day_counter_reflects_days_not_row_scale_numbers(self) -> None:
        # Outer "days" loop: day 1 of 3.
        self.handler("2026-07-20", 0, 3, "days")
        self.assertEqual(self.progress.backfill_days_current, 1)
        self.assertEqual(self.progress.backfill_days_total, 3)
        # Inner "rows" loop for that same day must NOT bump the day counter,
        # even when the row-scale numbers are huge (this used to leak into
        # the "Day" display as e.g. "Day 310000/310002").
        self.handler("2026-07-20", 250_000, 310_002, "rows")
        self.assertEqual(self.progress.backfill_days_current, 1)
        self.assertEqual(self.progress.backfill_days_total, 3)
        self.assertEqual(self.progress.backfill_rows_current, 250_000)
        self.assertEqual(self.progress.backfill_rows_total, 310_002)

    def test_samples_counter_moves_during_row_backfill_not_stuck_at_zero(self) -> None:
        self.handler("2026-07-20", 0, 1, "days")
        self.handler("2026-07-20", 0, 620, "rows")
        self.assertEqual(self.progress.backfill_rows_current, 0)
        self.handler("2026-07-20", 500, 620, "rows")
        self.assertEqual(self.progress.backfill_rows_current, 500)
        self.assertEqual(self.progress.backfill_rows_total, 620)
        self.handler("2026-07-20", 620, 620, "rows")
        self.assertEqual(self.progress.backfill_rows_current, 620)

    def test_new_day_resets_row_counter(self) -> None:
        self.handler("2026-07-20", 0, 2, "days")
        self.handler("2026-07-20", 400, 500, "rows")
        self.assertEqual(self.progress.backfill_rows_total, 500)
        # Moving to day 2 must reset the per-day row counter to 0, not leave
        # day 1's stale total/current lingering.
        self.handler("2026-07-21", 1, 2, "days")
        self.assertEqual(self.progress.backfill_rows_current, 0)
        self.assertEqual(self.progress.backfill_rows_total, 0)
        self.assertEqual(self.progress.backfill_days_current, 2)

    def test_percent_never_falsely_reports_100_mid_backfill(self) -> None:
        self.handler("2026-07-20", 0, 5, "days")
        self.handler("2026-07-20", 309_999, 310_002, "rows")
        # The per-day row loop is nearly done (309999/310002 ~ 99.999%), but
        # this is day 1 of 5 — the overall job must not be reported as done.
        self.assertLess(self.progress.backfill_percent, 25.0)
        self.assertEqual(self.progress.percent, 0.0)

    def test_message_prefixed_with_backfilling_existing_master_data(self) -> None:
        self.handler("2026-07-21", 0, 1, "days")
        self.assertTrue(self.progress.message.startswith("Backfilling existing master data — "))

    def test_backfill_active_cleared_after_backfill_ends(self) -> None:
        self.handler("2026-07-20", 0, 1, "days")
        self.handler("2026-07-20", 620, 620, "rows")
        self.assertTrue(self.progress.backfill_active)
        self.timer.set_stage_name(1, None)
        self.progress.emit(
            stage=1,
            current=0,
            total=2,
            message="Phase 1: Master build — master.db",
            backfill_active=False,
            clear_substage=True,
        )
        self.assertFalse(self.progress.backfill_active)
        self.assertIsNone(self.progress.backfill_days_current)
        self.assertIsNone(self.progress.backfill_rows_current)
        self.assertEqual(self.progress.stage_name, "Load Database")
        snap = self.timer.snapshot(rows=0, estimated_total_rows=0)
        stage1 = next(s for s in snap["stages"] if s["id"] == 1)
        self.assertEqual(stage1["name"], "Load Database")


class PipelineTimerStageNameOverrideTests(unittest.TestCase):
    def test_default_stage_names_unaffected_when_no_override(self) -> None:
        timer = PipelineTimer([])
        snap = timer.snapshot(rows=0, estimated_total_rows=0)
        names = {s["id"]: s["name"] for s in snap["stages"]}
        self.assertEqual(names[1], STAGE_NAMES[1])

    def test_override_and_reset_round_trip(self) -> None:
        timer = PipelineTimer([])
        timer.set_stage_name(1, "Backfilling existing master")
        snap = timer.snapshot(rows=0, estimated_total_rows=0)
        stage1 = next(s for s in snap["stages"] if s["id"] == 1)
        self.assertEqual(stage1["name"], "Backfilling existing master")

        timer.set_stage_name(1, None)
        snap2 = timer.snapshot(rows=0, estimated_total_rows=0)
        stage1b = next(s for s in snap2["stages"] if s["id"] == 1)
        self.assertEqual(stage1b["name"], STAGE_NAMES[1])

    def test_override_does_not_affect_other_stages(self) -> None:
        timer = PipelineTimer([])
        timer.set_stage_name(1, "Backfilling existing master")
        snap = timer.snapshot(rows=0, estimated_total_rows=0)
        names = {s["id"]: s["name"] for s in snap["stages"]}
        self.assertEqual(names[6], STAGE_NAMES[6])


class BuildProgressStageNameOverrideTests(unittest.TestCase):
    def test_stage_name_override_takes_precedence_over_default(self) -> None:
        progress = BuildProgress(job_id="job-2")
        payload = progress.emit(stage=1, stage_name="Backfilling existing master")
        self.assertEqual(payload["stage_name"], "Backfilling existing master")

    def test_stage_name_defaults_when_not_overridden(self) -> None:
        progress = BuildProgress(job_id="job-3")
        payload = progress.emit(stage=1)
        self.assertEqual(payload["stage_name"], "Load Database")

    def test_stage_name_reverts_once_override_omitted_on_next_emit(self) -> None:
        progress = BuildProgress(job_id="job-4")
        progress.emit(stage=1, stage_name="Backfilling existing master")
        payload = progress.emit(stage=1, message="Loading tick DB")
        self.assertEqual(payload["stage_name"], "Load Database")


if __name__ == "__main__":
    unittest.main()

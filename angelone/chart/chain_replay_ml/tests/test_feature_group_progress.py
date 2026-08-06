"""Tests for feature-group progress throttling during large builds."""

from __future__ import annotations

import time
import unittest

from chain_replay_ml.dataset_builder.stages import _feature_progress_due


class FeatureGroupProgressTests(unittest.TestCase):
    def test_first_and_last_rows_always_emit(self) -> None:
        self.assertTrue(_feature_progress_due(0, 310_002, progress_every=100, last_emit_at=time.perf_counter()))
        self.assertTrue(
            _feature_progress_due(310_001, 310_002, progress_every=100, last_emit_at=time.perf_counter())
        )

    def test_row_interval_emit(self) -> None:
        self.assertTrue(_feature_progress_due(100, 310_002, progress_every=100, last_emit_at=time.perf_counter()))
        self.assertFalse(_feature_progress_due(50, 310_002, progress_every=100, last_emit_at=time.perf_counter()))

    def test_time_interval_emit_between_row_milestones(self) -> None:
        old = time.perf_counter() - 2.0
        self.assertTrue(_feature_progress_due(1, 310_002, progress_every=100, last_emit_at=old))


if __name__ == "__main__":
    unittest.main()

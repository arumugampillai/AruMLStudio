"""Smoke tests for Phase 2 baseline regression runner."""

from __future__ import annotations

import unittest

from chain_replay_ml.dataset_builder.phase2_baseline_report import (
    PHASE2_BASELINE_TAG,
    build_phase2_baseline_suite,
    format_phase2_baseline_report,
    run_phase2_baseline,
)


class Phase2BaselineReportTests(unittest.TestCase):
    def test_suite_has_expected_scale(self) -> None:
        suite = build_phase2_baseline_suite()
        count = suite.countTestCases()
        self.assertGreaterEqual(count, 59)
        self.assertLessEqual(count, 70)

    def test_baseline_run_all_pass_or_documented_skip(self) -> None:
        run = run_phase2_baseline()
        report = format_phase2_baseline_report(run)
        self.assertIn(PHASE2_BASELINE_TAG, report)
        self.assertIn("Controller Migration Status", report)
        self.assertIn("EMA          PASS", report)
        self.assertIn("IV History   PASS", report)
        self.assertIn("Regression:", report)
        failures = run.count_status("FAIL") + run.count_status("ERROR")
        self.assertEqual(failures, 0, run.stdout)


if __name__ == "__main__":
    unittest.main()

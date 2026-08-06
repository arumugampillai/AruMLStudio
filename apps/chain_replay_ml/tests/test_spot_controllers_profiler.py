"""Tests for SpotControllers.update() path profiler."""

from __future__ import annotations

import unittest

from chain_replay_ml.dataset_builder.build_profiler import BuildProfiler, set_profiler
from chain_replay_ml.dataset_builder.rolling_controllers import SpotControllers
from chain_replay_ml.dataset_builder.spot_controllers_profiler import (
    reset_spot_controllers_profiler,
    snapshot_spot_controllers_profiler,
)


class SpotControllersProfilerTests(unittest.TestCase):
    def setUp(self) -> None:
        self._profiler = BuildProfiler()
        set_profiler(self._profiler)
        reset_spot_controllers_profiler()
        self.ctrl = SpotControllers()

    def tearDown(self) -> None:
        set_profiler(None)

    def test_counts_duplicate_invalid_and_full_paths(self) -> None:
        ctrl = self.ctrl
        ctrl.update(None, ts=1.0)
        ctrl.update(0.0, ts=2.0)
        ctrl.update(100.0, ts=10.0)
        ctrl.update(101.0, ts=10.0)
        ctrl.update(102.0, ts=20.0)

        stats = snapshot_spot_controllers_profiler()
        assert stats is not None
        doc = stats.to_dict()
        self.assertEqual(doc["total_calls"], 5)
        self.assertEqual(doc["early_returns_invalid_spot"], 2)
        self.assertEqual(doc["early_returns_duplicate_timestamp"], 1)
        self.assertEqual(doc["full_updates_executed"], 2)

        table = {row["path"]: row for row in doc["summary_table"]}
        self.assertIn("Duplicate timestamp", table)
        self.assertIn("Invalid spot", table)
        self.assertIn("Full update", table)
        self.assertIn("├─ EMA", table)
        self.assertIn("├─ RV", table)
        self.assertIn("├─ Momentum", table)
        self.assertEqual(table["Full update"]["calls"], 2)
        self.assertGreaterEqual(table["Duplicate timestamp"]["calls"], 1)

        breakdown = doc["full_update_breakdown"]
        self.assertGreaterEqual(len(breakdown), 4)
        sections = {row["section"]: row for row in breakdown}
        self.assertIn("EMA9–200", sections)
        self.assertIn("RV", sections)
        self.assertIn("Momentum", sections)
        self.assertIn("Spot HL", sections)
        pct_sum = sum(float(r["pct_of_full_update"]) for r in breakdown)
        self.assertAlmostEqual(pct_sum, 100.0, delta=0.5)
        self.assertEqual(doc["case"], "case_1_full_updates_dominate")


if __name__ == "__main__":
    unittest.main()

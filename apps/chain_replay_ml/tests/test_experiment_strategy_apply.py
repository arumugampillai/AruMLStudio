"""Tests for filter mapping in experiment strategy apply."""

from __future__ import annotations

import unittest

from chain_replay_ml.fold_research.experiment_strategy_apply import (
    filters_to_config_overrides,
    merge_strategy_filters,
)


class ExperimentStrategyApplyTests(unittest.TestCase):
    def test_merge_and_map_filters(self) -> None:
        changes = [
            {"filters": {"min_premium": 26.0, "stop_pct": 7.0}},
            {"filters": {"min_confidence": 70.0}},
        ]
        merged = merge_strategy_filters(changes)
        self.assertEqual(merged["min_premium"], 26.0)
        self.assertEqual(merged["stop_pct"], 7.0)
        self.assertEqual(merged["min_confidence"], 70.0)

        overrides, notes = filters_to_config_overrides(merged)
        self.assertEqual(overrides["entry"]["premium_min"], 26.0)
        self.assertEqual(overrides["stop"]["stop_loss_pct"], 7.0)
        self.assertAlmostEqual(overrides["confidence"]["min_signal_strength"], 0.7)
        self.assertTrue(overrides["confidence"]["use_model_confidence"])
        self.assertTrue(notes)


if __name__ == "__main__":
    unittest.main()

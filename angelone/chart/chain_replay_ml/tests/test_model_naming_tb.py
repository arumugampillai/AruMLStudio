"""Auto model naming — Triple Barrier TB_ prefix."""

from __future__ import annotations

import unittest
from datetime import datetime
from zoneinfo import ZoneInfo

from chain_replay_ml.training.naming import suggest_model_name
from master_dataset_tk.model_builder.state import ModelBuilderState

_IST = ZoneInfo("Asia/Kolkata")


class TripleBarrierNamingTests(unittest.TestCase):
    def test_tb_prefix_for_triple_barrier(self) -> None:
        when = datetime(2026, 8, 1, 12, 34, tzinfo=_IST)
        name = suggest_model_name(
            "label_id",
            "xgboost",
            validation_strategy="walk_forward",
            feature_count=395,
            when=when,
            label_strategy="triple_barrier",
            label_strategy_params={"tp_value": 20, "sl_value": 10, "barrier_type": "percentage"},
        )
        self.assertTrue(name.startswith("TB_tp_20_sl_10_"))
        self.assertIn("WF", name)
        self.assertIn("395f", name)
        self.assertIn("XGB", name)

    def test_fixed_horizon_unchanged(self) -> None:
        when = datetime(2026, 8, 1, 12, 34, tzinfo=_IST)
        name = suggest_model_name(
            "future_ltp_5m",
            "xgboost",
            validation_strategy="time_series_split",
            feature_count=50,
            when=when,
            label_strategy="fixed_horizon",
        )
        self.assertTrue(name.startswith("Future_LTP_5m_"))
        self.assertFalse(name.startswith("TB_"))

    def test_state_suggests_tb(self) -> None:
        state = ModelBuilderState(
            dataset="ds",
            target="label_up_2pct_5m",
            algorithm="xgboost",
            label_strategy_id="triple_barrier",
            label_strategy_params={"tp_value": 20.0, "sl_value": 10.0},
            features={"a", "b"},
            validation_strategy="walk_forward",
        )
        name = state.suggest_model_name()
        self.assertTrue(name.startswith("TB_tp_20_sl_10_"))
        self.assertNotIn("2pct", name)
        self.assertEqual(state.build_training_config()["target"], "label_id")


if __name__ == "__main__":
    unittest.main()

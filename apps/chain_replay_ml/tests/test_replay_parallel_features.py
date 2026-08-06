"""Tests for replay parallel feature build (report path)."""

from __future__ import annotations

import os
import unittest

from chain_replay_ml.dataset_builder.stages_parallel import _group_rows_by_token


class TestReplayParallelFeatures(unittest.TestCase):
    def test_group_rows_by_token_sorts_timestamps(self):
        rows = [
            {"token": "A", "timestamp": 30.0, "strike": 1, "option_type": "CE", "_atm": 1},
            {"token": "B", "timestamp": 10.0, "strike": 2, "option_type": "PE", "_atm": 2},
            {"token": "A", "timestamp": 10.0, "strike": 1, "option_type": "CE", "_atm": 1},
        ]
        grouped = _group_rows_by_token(rows)
        self.assertEqual([snap["timestamp"] for _, snap in grouped["A"]], [10.0, 30.0])
        self.assertEqual(len(grouped["B"]), 1)
        self.assertEqual(grouped["A"][0][1]["token"], "A")
        self.assertEqual(grouped["B"][0][1]["token"], "B")


@unittest.skipUnless(
    os.getenv("RUN_REPLAY_PARITY_TEST") == "1",
    "Set RUN_REPLAY_PARITY_TEST=1 with tick DB + model to run integration parity test",
)
class TestReplayParallelParity(unittest.TestCase):
    def test_parallel_matches_serial_on_replay_day(self):
        import numpy as np
        import pandas as pd

        from chain_replay_ml.replay_feature_scoring import build_replay_day_frame, load_model_inference_config
        from chain_replay_ml.replay_scoring_cache import clear_replay_scoring_cache
        from chain_replay_ml.training.default_model import resolve_default_model_name

        data_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "data",
        )
        model_name = os.getenv("REPLAY_PARITY_MODEL") or resolve_default_model_name(data_dir)
        date_str = os.getenv("REPLAY_PARITY_DATE") or "2026-05-27"
        expiry = os.getenv("REPLAY_PARITY_EXPIRY") or "2026-06-02"
        if not model_name:
            self.skipTest("no model in registry")

        loaded = load_model_inference_config(data_dir, model_name)
        if not loaded:
            self.skipTest(f"model not found: {model_name}")
        replay_config = loaded.get("replay_config") or {}
        if not replay_config:
            self.skipTest("model has no replay_config")

        clear_replay_scoring_cache()
        serial_df, err_s, _, _ = build_replay_day_frame(
            data_dir,
            replay_config,
            date_str,
            expiry_hint=expiry,
            target=loaded["target"],
            parallel_features=False,
        )
        self.assertIsNone(err_s, err_s)
        self.assertFalse(serial_df.empty)

        clear_replay_scoring_cache()
        parallel_df, err_p, _, stats_p = build_replay_day_frame(
            data_dir,
            replay_config,
            date_str,
            expiry_hint=expiry,
            target=loaded["target"],
            parallel_features=True,
        )
        self.assertIsNone(err_p, err_p)
        self.assertFalse(parallel_df.empty)
        self.assertEqual(stats_p.get("parallel_mode"), "token")

        features = list(loaded["features"])
        key = ["timestamp", "token", "strike", "option_type"]
        serial_df = serial_df.sort_values(key).reset_index(drop=True)
        parallel_df = parallel_df.sort_values(key).reset_index(drop=True)
        self.assertEqual(len(serial_df), len(parallel_df))
        for col in features:
            if col not in serial_df.columns:
                continue
            np.testing.assert_allclose(
                serial_df[col].astype(float),
                parallel_df[col].astype(float),
                rtol=1e-9,
                atol=1e-6,
                err_msg=f"mismatch on {col}",
            )


if __name__ == "__main__":
    unittest.main()

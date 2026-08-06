"""Tests for binary TP-hit adaptation of Triple Barrier label_id."""

from __future__ import annotations

import unittest

import pandas as pd

from chain_replay_ml.training.label_prep import adapt_target_for_prediction_type


class LabelPrepTests(unittest.TestCase):
    def test_binary_tp_hit_from_label_id(self) -> None:
        y = pd.Series([0, 1, 2, 0, 1], dtype="float32")
        out, meta = adapt_target_for_prediction_type(
            y,
            prediction_type="binary",
            target="label_id",
            label_encoding={"TP": 0, "SL": 1, "TIME": 2},
        )
        self.assertEqual(meta["mode"], "binary_tp_hit")
        self.assertListEqual(out.tolist(), [1.0, 0.0, 0.0, 1.0, 0.0])
        # Idempotent on already-binary labels.
        out2, meta2 = adapt_target_for_prediction_type(
            out, prediction_type="binary", target="label_id"
        )
        self.assertEqual(meta2["mode"], "binary_passthrough")
        self.assertListEqual(out2.tolist(), out.tolist())

    def test_multiclass_keeps_three_way(self) -> None:
        y = pd.Series([0, 1, 2, 0], dtype="float32")
        out, meta = adapt_target_for_prediction_type(
            y, prediction_type="classification", target="label_id"
        )
        self.assertEqual(meta["mode"], "multiclass_remap")
        self.assertEqual(meta["n_classes"], 3)
        self.assertListEqual(out.tolist(), [0.0, 1.0, 2.0, 0.0])


if __name__ == "__main__":
    unittest.main()

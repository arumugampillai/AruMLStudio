"""Tests for Master Dataset export classification labels (up-% at 5m)."""

from __future__ import annotations

import unittest

import pandas as pd

from chain_replay_ml.dataset_builder.classification_labels import (
    CLASSIFICATION_LABEL_COLUMNS_5M,
    attach_up_pct_classification_labels_5m,
    can_generate_up_pct_labels_5m,
    merge_classification_targets,
)
from chain_replay_ml.training.target_kinds import (
    is_binary_hit_target,
    is_label_up_target,
    prediction_type_for_target,
    target_prediction_type_compatible,
)


class TestUpPctLabels(unittest.TestCase):
    def test_catalog_size(self) -> None:
        self.assertEqual(len(CLASSIFICATION_LABEL_COLUMNS_5M), 6)

    def test_requires_ltp_and_future(self) -> None:
        self.assertFalse(can_generate_up_pct_labels_5m(["future_ltp_5m"]))
        self.assertTrue(can_generate_up_pct_labels_5m(["ltp", "future_ltp_5m"]))
        self.assertTrue(can_generate_up_pct_labels_5m(["current_ltp", "future_ltp_5m"]))

    def test_thresholds(self) -> None:
        # current=100 → 2%=102, 3%=103, … 6%=106
        df = pd.DataFrame(
            {
                "ltp": [100.0, 100.0, 100.0, 100.0],
                "future_ltp_5m": [101.0, 102.0, 106.0, 107.0],
            }
        )
        added = attach_up_pct_classification_labels_5m(df)
        self.assertEqual(added, list(CLASSIFICATION_LABEL_COLUMNS_5M))

        # +1% → none
        self.assertEqual(df.loc[0, "label_up_2pct_5m"], 0.0)
        self.assertEqual(df.loc[0, "label_up_gt6pct_5m"], 0.0)

        # exactly +2% → up_2 yes; others no
        self.assertEqual(df.loc[1, "label_up_2pct_5m"], 1.0)
        self.assertEqual(df.loc[1, "label_up_3pct_5m"], 0.0)

        # exactly +6% → up_2..up_6 yes; gt6 no
        self.assertEqual(df.loc[2, "label_up_6pct_5m"], 1.0)
        self.assertEqual(df.loc[2, "label_up_gt6pct_5m"], 0.0)

        # +7% → all including gt6
        self.assertEqual(df.loc[3, "label_up_6pct_5m"], 1.0)
        self.assertEqual(df.loc[3, "label_up_gt6pct_5m"], 1.0)

    def test_null_when_missing_future(self) -> None:
        df = pd.DataFrame({"ltp": [100.0], "future_ltp_5m": [float("nan")]})
        attach_up_pct_classification_labels_5m(df)
        self.assertTrue(pd.isna(df.loc[0, "label_up_2pct_5m"]))

    def test_merge_targets(self) -> None:
        merged = merge_classification_targets(["future_ltp_1m", "future_ltp_5m"])
        self.assertIn("future_ltp_5m", merged)
        self.assertEqual(merged[-6:], list(CLASSIFICATION_LABEL_COLUMNS_5M))


class TestTargetKindsLabelUp(unittest.TestCase):
    def test_binary_compat(self) -> None:
        self.assertTrue(is_label_up_target("label_up_2pct_5m"))
        self.assertTrue(is_binary_hit_target("label_up_5pct_5m"))
        self.assertEqual(prediction_type_for_target("label_up_4pct_5m"), "binary")
        self.assertTrue(target_prediction_type_compatible("binary", "label_up_gt6pct_5m"))
        self.assertFalse(target_prediction_type_compatible("regression", "label_up_2pct_5m"))
        self.assertTrue(target_prediction_type_compatible("regression", "future_ltp_5m"))


if __name__ == "__main__":
    unittest.main()

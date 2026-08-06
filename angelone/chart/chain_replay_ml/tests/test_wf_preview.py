"""Tests for walk-forward preview planner."""

from __future__ import annotations

import unittest

from master_dataset_tk.model_builder.wf_preview import compute_walk_forward_preview_plan
from chain_replay_ml.training.split import normalize_walk_forward_config, walk_forward_fold_slices


class TestWalkForwardPreview(unittest.TestCase):
    def test_preview_matches_walk_forward_fold_slices(self) -> None:
        row_count = 1_300_000
        preview = compute_walk_forward_preview_plan(
            row_count=row_count,
            n_folds=5,
            train_window=5000,
            val_window=1000,
            window_mode="expanding",
            fold_placement="distributed",
            test_holdout_pct=15,
            validation_strategy="walk_forward",
        )
        self.assertTrue(preview.get("ok"))

        split_cfg = {
            "test": 15,
            "walk_forward": {
                "n_folds": 5,
                "train_window_size": 5000,
                "validation_window_size": 1000,
                "window_mode": "expanding",
                "fold_placement": "distributed",
            },
        }
        wf_cfg = normalize_walk_forward_config(split_cfg, row_count)
        folds, test_sl = walk_forward_fold_slices(row_count, wf_cfg)

        self.assertEqual(preview["test_slice"]["start"], test_sl.start)
        self.assertEqual(preview["test_slice"]["stop"], test_sl.stop)
        self.assertEqual(len(preview["folds"]), len(folds))
        for preview_fold, fold in zip(preview["folds"], folds):
            self.assertEqual(preview_fold["train_start"], fold["train"]["start"])
            self.assertEqual(preview_fold["train_end"], fold["train"]["stop"] - 1)
            self.assertEqual(preview_fold["train_rows"], fold["train"]["rows"])
            self.assertEqual(preview_fold["val_start"], fold["validation"]["start"])
            self.assertEqual(preview_fold["val_end"], fold["validation"]["stop"] - 1)
            self.assertEqual(preview_fold["val_rows"], fold["validation"]["rows"])

    def test_preview_without_rows_reports_error(self) -> None:
        preview = compute_walk_forward_preview_plan(
            row_count=None,
            n_folds=5,
            train_window=5000,
            val_window=1000,
            window_mode="expanding",
            fold_placement="anchored",
            test_holdout_pct=15,
        )
        self.assertFalse(preview.get("ok"))
        self.assertIn("row count", str(preview.get("error")).lower())


if __name__ == "__main__":
    unittest.main()

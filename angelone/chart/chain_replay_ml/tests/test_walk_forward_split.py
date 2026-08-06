"""Walk-forward fold placement tests."""

from __future__ import annotations

import unittest

from chain_replay_ml.training.split import (
    FOLD_PLACEMENT_ANCHORED,
    FOLD_PLACEMENT_DISTRIBUTED,
    normalize_walk_forward_config,
    validation_strategy_fields_from_split,
    validation_strategy_label_from_ui,
    walk_forward_fold_slices,
    walk_forward_meta_from_config,
)


def _wf_cfg(
    *,
    n_rows: int,
    placement: str = "anchored",
    window_mode: str = "expanding",
    train_window: int = 5000,
    val_window: int = 1000,
    n_folds: int = 5,
    test_pct: int = 15,
) -> dict:
    split_cfg = {
        "test": test_pct,
        "walk_forward": {
            "train_window_size": train_window,
            "validation_window_size": val_window,
            "n_folds": n_folds,
            "window_mode": window_mode,
            "fold_placement": placement,
        },
    }
    return normalize_walk_forward_config(split_cfg, n_rows)


class TestWalkForwardSplit(unittest.TestCase):
    def test_anchored_matches_legacy_expanding_layout(self) -> None:
        n_rows = 1_300_000
        wf = _wf_cfg(n_rows=n_rows, placement="anchored")
        folds, test_sl = walk_forward_fold_slices(n_rows, wf)

        self.assertEqual(len(folds), 5)
        self.assertEqual(folds[0]["train"]["start"], 0)
        self.assertEqual(folds[0]["train"]["stop"], 5000)
        self.assertEqual(folds[0]["validation"]["start"], 5000)
        self.assertEqual(folds[0]["validation"]["stop"], 6000)
        self.assertEqual(folds[4]["validation"]["stop"], 10_000)
        self.assertEqual(test_sl.start, 1_105_000)
        self.assertEqual(test_sl.stop, n_rows)
        self.assertTrue(all(f["fold_placement"] == FOLD_PLACEMENT_ANCHORED for f in folds))

    def test_distributed_spans_walk_forward_region(self) -> None:
        n_rows = 1_300_000
        wf = _wf_cfg(n_rows=n_rows, placement="distributed")
        folds, test_sl = walk_forward_fold_slices(n_rows, wf)

        self.assertEqual(len(folds), 5)
        self.assertEqual(test_sl.start, 1_105_000)
        self.assertEqual(folds[0]["validation"]["stop"], 6000)
        self.assertEqual(folds[-1]["validation"]["stop"], 1_105_000)
        self.assertTrue(all(f["fold_placement"] == FOLD_PLACEMENT_DISTRIBUTED for f in folds))

        val_starts = [f["validation"]["start"] for f in folds]
        val_ends = [f["validation"]["stop"] for f in folds]
        self.assertEqual(val_starts, sorted(val_starts))
        self.assertEqual(val_ends, sorted(val_ends))
        self.assertTrue(all(val_ends[i] <= val_starts[i + 1] for i in range(len(folds) - 1)))

    def test_distributed_preserves_chronology_and_no_leakage(self) -> None:
        n_rows = 1_300_000
        wf = _wf_cfg(n_rows=n_rows, placement="distributed", window_mode="expanding")
        folds, _test_sl = walk_forward_fold_slices(n_rows, wf)

        for fold in folds:
            tr = fold["train"]
            va = fold["validation"]
            self.assertEqual(tr["start"], 0)
            self.assertEqual(tr["stop"], va["start"])
            self.assertLess(tr["stop"], va["stop"])
            self.assertEqual(va["stop"] - va["start"], 1000)
            self.assertGreaterEqual(tr["stop"], 5000)

    def test_distributed_rolling_uses_fixed_train_window(self) -> None:
        n_rows = 1_300_000
        wf = _wf_cfg(n_rows=n_rows, placement="distributed", window_mode="rolling")
        folds, _test_sl = walk_forward_fold_slices(n_rows, wf)

        for fold in folds:
            tr = fold["train"]
            va = fold["validation"]
            self.assertEqual(tr["stop"], va["start"])
            self.assertEqual(tr["stop"] - tr["start"], 5000)
            self.assertGreaterEqual(tr["start"], 0)

    def test_distributed_validation_ends_are_evenly_spaced(self) -> None:
        n_rows = 1_300_000
        wf = _wf_cfg(n_rows=n_rows, placement="distributed")
        folds, _test_sl = walk_forward_fold_slices(n_rows, wf)
        val_ends = [f["validation"]["stop"] for f in folds]
        self.assertEqual(val_ends, [6000, 280_750, 555_500, 830_250, 1_105_000])

    def test_distributed_covers_far_more_rows_than_anchored(self) -> None:
        n_rows = 1_300_000
        anchored, _ = walk_forward_fold_slices(n_rows, _wf_cfg(n_rows=n_rows, placement="anchored"))
        distributed, _ = walk_forward_fold_slices(n_rows, _wf_cfg(n_rows=n_rows, placement="distributed"))

        def _covered(fold_list: list[dict]) -> set[int]:
            used: set[int] = set()
            for fold in fold_list:
                for key in ("train", "validation"):
                    part = fold[key]
                    used.update(range(part["start"], part["stop"]))
            return used

        anchored_used = _covered(anchored)
        distributed_used = _covered(distributed)
        self.assertEqual(max(anchored_used), 9_999)
        self.assertEqual(max(distributed_used), 1_104_999)
        self.assertGreater(len(distributed_used), len(anchored_used))


class WalkForwardMetaTests(unittest.TestCase):
    def test_meta_includes_fold_placement_label(self) -> None:
        wf = normalize_walk_forward_config(
            {"walk_forward": {"fold_placement": "distributed", "n_folds": 10}, "test": 15},
            1_000_000,
        )
        meta = walk_forward_meta_from_config(wf)
        self.assertEqual(meta["fold_placement"], FOLD_PLACEMENT_DISTRIBUTED)
        self.assertEqual(meta["fold_placement_label"], "Distributed")
        self.assertEqual(meta["n_folds"], 10)

    def test_validation_strategy_label_mapping(self) -> None:
        self.assertEqual(validation_strategy_label_from_ui("time_series_split"), "Time Series Split")
        self.assertEqual(validation_strategy_label_from_ui("walk_forward"), "Walk Forward")
        self.assertEqual(validation_strategy_label_from_ui("walk_forward", window_mode="rolling"), "Rolling Window")
        self.assertEqual(validation_strategy_label_from_ui("rolling_window"), "Rolling Window")

    def test_meta_includes_validation_strategy_from_split_cfg(self) -> None:
        wf = normalize_walk_forward_config(
            {"walk_forward": {"window_mode": "expanding", "n_folds": 5}, "test": 15},
            1_000_000,
        )
        split_cfg = {
            "strategy": "walk_forward",
            "validation_strategy_ui": "walk_forward",
            "walk_forward": wf,
        }
        meta = walk_forward_meta_from_config(wf, split_cfg=split_cfg)
        self.assertEqual(meta["validation_strategy_ui"], "walk_forward")
        self.assertEqual(meta["validation_strategy_label"], "Walk Forward")

    def test_meta_rolling_window_from_split_cfg(self) -> None:
        wf = normalize_walk_forward_config(
            {"walk_forward": {"window_mode": "rolling", "n_folds": 5}, "test": 15},
            1_000_000,
        )
        split_cfg = {
            "strategy": "walk_forward",
            "validation_strategy_ui": "rolling_window",
            "walk_forward": wf,
        }
        meta = walk_forward_meta_from_config(wf, split_cfg=split_cfg)
        self.assertEqual(meta["validation_strategy_ui"], "rolling_window")
        self.assertEqual(meta["validation_strategy_label"], "Rolling Window")

    def test_validation_strategy_fields_inferred_from_strategy(self) -> None:
        fields = validation_strategy_fields_from_split({
            "strategy": "time_series",
            "train": 70,
            "validation": 15,
            "test": 15,
        })
        self.assertEqual(fields["validation_strategy_ui"], "time_series_split")
        self.assertEqual(fields["validation_strategy_label"], "Time Series Split")


if __name__ == "__main__":
    unittest.main()

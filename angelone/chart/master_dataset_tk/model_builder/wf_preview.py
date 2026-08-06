"""Walk-forward execution plan preview — mirrors training split logic."""

from __future__ import annotations

from typing import Any

from chain_replay_ml.training.split import (
    WalkForwardSplitError,
    normalize_walk_forward_config,
    walk_forward_fold_slices,
)


def compute_walk_forward_preview_plan(
    *,
    row_count: int | None,
    n_folds: int,
    train_window: int,
    val_window: int,
    window_mode: str,
    fold_placement: str,
    test_holdout_pct: int,
    validation_strategy: str = "walk_forward",
) -> dict[str, Any]:
    """Build the same fold plan used at training time (walk_forward_fold_slices)."""
    if row_count is None or int(row_count) <= 0:
        return {
            "ok": False,
            "error": "Select a dataset with a known row count to preview walk-forward folds.",
        }

    n_rows = int(row_count)
    resolved_window_mode = "rolling" if validation_strategy == "rolling_window" else str(window_mode or "expanding")
    split_cfg = {
        "test": int(test_holdout_pct),
        "walk_forward": {
            "n_folds": int(n_folds),
            "train_window_size": int(train_window),
            "validation_window_size": int(val_window),
            "window_mode": resolved_window_mode,
            "fold_placement": str(fold_placement or "anchored"),
        },
    }
    try:
        wf_cfg = normalize_walk_forward_config(split_cfg, n_rows)
        folds, test_sl = walk_forward_fold_slices(n_rows, wf_cfg)
    except WalkForwardSplitError as exc:
        return {"ok": False, "error": str(exc), "row_count": n_rows}

    wf_end = int(test_sl.start)
    holdout_rows = int(test_sl.stop - test_sl.start)
    fold_rows: list[dict[str, Any]] = []
    for fold in folds:
        tr = fold["train"]
        va = fold["validation"]
        fold_rows.append({
            "fold": int(fold["fold"]),
            "train_start": int(tr["start"]),
            "train_end": int(tr["stop"] - 1),
            "train_rows": int(tr["rows"]),
            "val_start": int(va["start"]),
            "val_end": int(va["stop"] - 1),
            "val_rows": int(va["rows"]),
            "train_stop_exclusive": int(tr["stop"]),
            "val_stop_exclusive": int(va["stop"]),
        })

    return {
        "ok": True,
        "summary": {
            "total_rows": n_rows,
            "walk_forward_region_rows": wf_end,
            "walk_forward_region_start": 0,
            "walk_forward_region_end": max(0, wf_end - 1),
            "test_holdout_rows": holdout_rows,
            "test_holdout_start": int(test_sl.start),
            "test_holdout_end": int(test_sl.stop - 1),
            "test_holdout_pct": int(test_holdout_pct),
            "fold_placement": str(wf_cfg.get("fold_placement") or "anchored"),
            "window_mode": str(wf_cfg.get("window_mode") or resolved_window_mode),
            "train_window": int(wf_cfg.get("train_window_size") or train_window),
            "validation_window": int(wf_cfg.get("validation_window_size") or val_window),
            "n_folds": len(fold_rows),
            "n_folds_requested": int(wf_cfg.get("n_folds") or n_folds),
        },
        "folds": fold_rows,
        "test_slice": {"start": int(test_sl.start), "stop": int(test_sl.stop)},
    }

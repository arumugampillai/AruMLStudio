"""Chronological train / validation / test splits — never shuffle."""

from __future__ import annotations

from typing import Any

import pandas as pd

from .config import TrainingConfig


class WalkForwardSplitError(ValueError):
    pass


FOLD_PLACEMENT_ANCHORED = "anchored"
FOLD_PLACEMENT_DISTRIBUTED = "distributed"
_VALID_FOLD_PLACEMENTS = (FOLD_PLACEMENT_ANCHORED, FOLD_PLACEMENT_DISTRIBUTED)
_VALID_WINDOW_MODES = ("expanding", "rolling")


def _normalize_fold_placement(raw: Any) -> str:
    key = str(raw or FOLD_PLACEMENT_ANCHORED).strip().lower()
    if key in (FOLD_PLACEMENT_DISTRIBUTED, "spread", "even"):
        return FOLD_PLACEMENT_DISTRIBUTED
    return FOLD_PLACEMENT_ANCHORED


def _normalize_window_mode(raw: Any) -> str:
    key = str(raw or "expanding").strip().lower()
    return key if key in _VALID_WINDOW_MODES else "expanding"


def fold_placement_label(placement: str | None) -> str:
    key = _normalize_fold_placement(placement)
    return "Distributed" if key == FOLD_PLACEMENT_DISTRIBUTED else "Anchored"


def validation_strategy_label_from_ui(ui_key: str, *, window_mode: str | None = None) -> str:
    """Map UI validation strategy key to display label."""
    key = str(ui_key or "").strip().lower()
    if key == "rolling_window":
        return "Rolling Window"
    if key == "walk_forward":
        if _normalize_window_mode(window_mode) == "rolling":
            return "Rolling Window"
        return "Walk Forward"
    return "Time Series Split"


def validation_strategy_fields_from_split(
    split_cfg: dict[str, Any] | None,
    wf_cfg: dict[str, Any] | None = None,
) -> dict[str, str]:
    """Derive persisted validation strategy UI key and display label from split config."""
    split = dict(split_cfg or {})
    wf = dict(wf_cfg or split.get("walk_forward") or {})
    window_mode = _normalize_window_mode(wf.get("window_mode"))
    ui = str(split.get("validation_strategy_ui") or "").strip().lower()
    if not ui:
        strategy = str(split.get("strategy") or "time_series").strip().lower()
        if strategy == "walk_forward" and window_mode == "rolling":
            ui = "rolling_window"
        elif strategy == "walk_forward":
            ui = "walk_forward"
        else:
            ui = "time_series_split"
    return {
        "validation_strategy_ui": ui,
        "validation_strategy_label": validation_strategy_label_from_ui(ui, window_mode=window_mode),
    }


def walk_forward_meta_from_config(
    wf_cfg: dict[str, Any] | None,
    *,
    split_cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Canonical walk-forward configuration block for summary / registry display."""
    cfg = dict(wf_cfg or {})
    if not cfg:
        return validation_strategy_fields_from_split(split_cfg, cfg) if split_cfg else {}
    placement = _normalize_fold_placement(cfg.get("fold_placement"))
    window_mode = _normalize_window_mode(cfg.get("window_mode"))
    hpo = dict(cfg.get("hyperparameter_optimization") or {})
    test_holdout_rows = cfg.get("test_holdout_rows")
    test_holdout_pct = cfg.get("test_holdout_pct")
    meta: dict[str, Any] = {
        "n_folds": int(cfg["n_folds"]) if cfg.get("n_folds") is not None else None,
        "window_mode": window_mode,
        "fold_placement": placement,
        "fold_placement_label": fold_placement_label(placement),
        "train_window_size": int(cfg["train_window_size"]) if cfg.get("train_window_size") is not None else None,
        "validation_window_size": int(cfg["validation_window_size"]) if cfg.get("validation_window_size") is not None else None,
        "test_holdout_rows": int(test_holdout_rows) if test_holdout_rows is not None else None,
        "test_holdout_pct": int(test_holdout_pct) if test_holdout_pct is not None else None,
        "feature_selection_method": cfg.get("feature_selection_method"),
        "optimization_metric": cfg.get("optimization_metric"),
        "hyperparameter_optimization_enabled": bool(hpo.get("enabled")) if hpo else None,
        "hpo_n_trials": int(hpo["n_trials"]) if hpo.get("n_trials") is not None else None,
    }
    if split_cfg is not None:
        meta.update(validation_strategy_fields_from_split(split_cfg, cfg))
    return meta


def _distributed_validation_ends(*, n_folds: int, min_val_end: int, max_val_end: int) -> list[int]:
    """Evenly space validation-window end indices across the walk-forward region."""
    if n_folds <= 0:
        return []
    if max_val_end < min_val_end:
        raise WalkForwardSplitError(
            f"Walk-forward region too small for distributed folds "
            f"(need val_end >= {min_val_end}, wf_end={max_val_end})"
        )
    if n_folds == 1:
        return [max_val_end]
    span = max_val_end - min_val_end
    ends: list[int] = []
    for fold_idx in range(n_folds):
        if fold_idx == n_folds - 1:
            ends.append(max_val_end)
        else:
            ends.append(min_val_end + (span * fold_idx) // (n_folds - 1))
    return ends


def _fold_payload(
    *,
    fold_number: int,
    train_start: int,
    train_end: int,
    val_start: int,
    val_end: int,
    window_mode: str,
    fold_placement: str,
) -> dict[str, Any]:
    return {
        "fold": fold_number,
        "train": {"start": train_start, "stop": train_end, "rows": train_end - train_start},
        "validation": {"start": val_start, "stop": val_end, "rows": val_end - val_start},
        "window_mode": window_mode,
        "fold_placement": fold_placement,
    }


def _build_anchored_folds(
    *,
    n_folds: int,
    train_window: int,
    val_window: int,
    wf_end: int,
    window_mode: str,
) -> list[dict[str, Any]]:
    folds: list[dict[str, Any]] = []
    for fold_idx in range(n_folds):
        if window_mode == "expanding":
            train_start = 0
            train_end = train_window + fold_idx * val_window
        else:
            train_start = fold_idx * val_window
            train_end = train_start + train_window
        val_start = train_end
        val_end = val_start + val_window
        if val_end > wf_end:
            break
        if train_end <= train_start or val_end <= val_start:
            break
        folds.append(_fold_payload(
            fold_number=fold_idx + 1,
            train_start=train_start,
            train_end=train_end,
            val_start=val_start,
            val_end=val_end,
            window_mode=window_mode,
            fold_placement=FOLD_PLACEMENT_ANCHORED,
        ))
    return folds


def _build_distributed_folds(
    *,
    n_folds: int,
    train_window: int,
    val_window: int,
    wf_end: int,
    window_mode: str,
) -> list[dict[str, Any]]:
    min_val_end = train_window + val_window
    val_ends = _distributed_validation_ends(
        n_folds=n_folds,
        min_val_end=min_val_end,
        max_val_end=wf_end,
    )
    folds: list[dict[str, Any]] = []
    for fold_idx, val_end in enumerate(val_ends):
        val_start = val_end - val_window
        if val_start < 0 or val_end > wf_end:
            continue
        if window_mode == "expanding":
            train_start = 0
            train_end = val_start
        else:
            train_end = val_start
            train_start = val_start - train_window
        if train_end < 0 or train_end <= train_start or val_end <= val_start:
            continue
        if window_mode == "expanding" and train_end < train_window:
            continue
        folds.append(_fold_payload(
            fold_number=fold_idx + 1,
            train_start=train_start,
            train_end=train_end,
            val_start=val_start,
            val_end=val_end,
            window_mode=window_mode,
            fold_placement=FOLD_PLACEMENT_DISTRIBUTED,
        ))
    return folds


def chronological_split_indices(
    n_rows: int,
    *,
    train_pct: int,
    val_pct: int,
    test_pct: int,
) -> dict[str, slice]:
    if n_rows <= 0:
        raise ValueError("Cannot split empty dataset")
    total_pct = train_pct + val_pct + test_pct
    if total_pct <= 0:
        raise ValueError("Split percentages must sum to a positive value")

    train_n = max(1, int(n_rows * train_pct / total_pct))
    val_n = max(1, int(n_rows * val_pct / total_pct))
    test_n = n_rows - train_n - val_n
    if test_n < 1:
        test_n = 1
        val_n = max(1, n_rows - train_n - test_n)

    train_end = train_n
    val_end = train_n + val_n
    return {
        "train": slice(0, train_end),
        "validation": slice(train_end, val_end),
        "test": slice(val_end, n_rows),
    }


def normalize_walk_forward_config(split_cfg: dict[str, Any], n_rows: int) -> dict[str, Any]:
    wf_in = dict(split_cfg.get("walk_forward") or {})
    test_pct = int(split_cfg.get("test", split_cfg.get("test_pct", 15)))
    test_holdout_rows = int(wf_in.get("test_holdout_rows") or 0)
    if test_holdout_rows <= 0:
        test_holdout_rows = max(1, int(n_rows * test_pct / 100))

    train_window = int(wf_in.get("train_window_size") or wf_in.get("train_window") or max(500, n_rows // 10))
    val_window = int(wf_in.get("validation_window_size") or wf_in.get("val_window") or max(100, n_rows // 50))
    n_folds = int(wf_in.get("n_folds") or wf_in.get("folds") or 5)
    window_mode = _normalize_window_mode(wf_in.get("window_mode"))
    fold_placement = _normalize_fold_placement(wf_in.get("fold_placement"))
    feature_selection_method = str(wf_in.get("feature_selection_method") or "rfe").strip().lower()
    if feature_selection_method not in ("none", "shap", "rfe", "permutation"):
        feature_selection_method = "rfe"
    optimization_metric = str(wf_in.get("optimization_metric") or "auto").strip().lower()
    if optimization_metric not in ("auto", "rmse", "mae", "directional_accuracy", "accuracy", "f1", "composite", "custom"):
        optimization_metric = "auto"
    min_selected_features = int(wf_in.get("min_selected_features") or 3)
    hpo_in = dict(wf_in.get("hyperparameter_optimization") or {})
    hpo_seeds_raw = hpo_in.get("validation_seeds") or hpo_in.get("seeds") or [42, 123, 999]

    return {
        "n_folds": max(1, n_folds),
        "window_mode": window_mode,
        "fold_placement": fold_placement,
        "train_window_size": max(50, train_window),
        "validation_window_size": max(20, val_window),
        "test_holdout_rows": max(1, test_holdout_rows),
        "feature_selection_method": feature_selection_method,
        "optimization_metric": optimization_metric,
        "min_selected_features": max(1, min_selected_features),
        "hyperparameter_optimization": {
            "enabled": bool(dict(wf_in.get("hyperparameter_optimization") or {}).get("enabled", wf_in.get("hpo_enabled", True))),
            "n_trials": max(1, int(dict(wf_in.get("hyperparameter_optimization") or {}).get("n_trials") or wf_in.get("hpo_n_trials") or 25)),
            "validation_seeds": [int(s) for s in hpo_seeds_raw],
            "resume": bool(hpo_in.get("resume", True)),
        },
    }


def walk_forward_fold_slices(n_rows: int, wf_cfg: dict[str, Any]) -> tuple[list[dict[str, Any]], slice]:
    """Build walk-forward fold index ranges plus final untouched test holdout."""
    if n_rows <= 0:
        raise WalkForwardSplitError("Cannot walk-forward split empty dataset")

    test_holdout = int(wf_cfg["test_holdout_rows"])
    train_window = int(wf_cfg["train_window_size"])
    val_window = int(wf_cfg["validation_window_size"])
    n_folds = int(wf_cfg["n_folds"])
    window_mode = _normalize_window_mode(wf_cfg.get("window_mode"))
    fold_placement = _normalize_fold_placement(wf_cfg.get("fold_placement"))

    if test_holdout >= n_rows:
        raise WalkForwardSplitError("Test holdout must be smaller than dataset row count")
    wf_end = n_rows - test_holdout
    min_region = train_window + val_window
    if fold_placement == FOLD_PLACEMENT_ANCHORED:
        required = train_window + n_folds * val_window
        if wf_end < required:
            raise WalkForwardSplitError(
                f"Need at least {required + test_holdout} rows for {n_folds} anchored folds "
                f"(train_window={train_window}, val_window={val_window}, test_holdout={test_holdout}), "
                f"but only {n_rows} available"
            )
    elif wf_end < min_region:
        raise WalkForwardSplitError(
            f"Need at least {min_region + test_holdout} rows for distributed folds "
            f"(train_window={train_window}, val_window={val_window}, test_holdout={test_holdout}), "
            f"but only {n_rows} available"
        )

    if fold_placement == FOLD_PLACEMENT_DISTRIBUTED:
        folds = _build_distributed_folds(
            n_folds=n_folds,
            train_window=train_window,
            val_window=val_window,
            wf_end=wf_end,
            window_mode=window_mode,
        )
    else:
        folds = _build_anchored_folds(
            n_folds=n_folds,
            train_window=train_window,
            val_window=val_window,
            wf_end=wf_end,
            window_mode=window_mode,
        )

    if not folds:
        raise WalkForwardSplitError("Could not construct any walk-forward folds with current parameters")

    test_slice = slice(wf_end, n_rows)
    return folds, test_slice


def split_xy(
    X: pd.DataFrame,
    y: pd.Series,
    config: TrainingConfig,
) -> dict[str, Any]:
    """Split X and y in chronological order (assumes rows are time-ordered)."""
    split_cfg = config.split
    strategy = str(split_cfg.get("strategy") or "time_series")

    if strategy == "walk_forward":
        wf_cfg = normalize_walk_forward_config(split_cfg, len(X))
        fold_defs, test_sl = walk_forward_fold_slices(len(X), wf_cfg)
        parts: dict[str, Any] = {
            "strategy": strategy,
            "walk_forward": wf_cfg,
            "folds": fold_defs,
            "slices": {
                "test": {"start": test_sl.start, "stop": test_sl.stop, "rows": test_sl.stop - test_sl.start},
            },
            "test": {
                "X": X.iloc[test_sl].reset_index(drop=True),
                "y": y.iloc[test_sl].reset_index(drop=True),
                "rows": int(test_sl.stop - test_sl.start),
                "slice": (test_sl.start, test_sl.stop),
            },
        }
        return parts

    train_pct = int(split_cfg.get("train", 70))
    val_pct = int(split_cfg.get("validation", 15))
    test_pct = int(split_cfg.get("test", 15))

    indices = chronological_split_indices(
        len(X),
        train_pct=train_pct,
        val_pct=val_pct,
        test_pct=test_pct,
    )

    parts = {"strategy": strategy, "slices": {}}
    for key, sl in indices.items():
        parts[key] = {
            "X": X.iloc[sl].reset_index(drop=True),
            "y": y.iloc[sl].reset_index(drop=True),
            "rows": int(sl.stop - sl.start),
            "slice": (sl.start, sl.stop),
        }
        parts["slices"][key] = {"start": sl.start, "stop": sl.stop, "rows": parts[key]["rows"]}
    return parts

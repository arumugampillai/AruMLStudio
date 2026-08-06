"""One-shot Feature Dataset ⟕ Label Run join for Create Model load."""

from __future__ import annotations

from typing import Any

import pandas as pd

from .registry import load_label_run_frame, load_label_run_meta


def resolve_join_keys(
    feature_columns: list[str] | set[str],
    label_columns: list[str] | set[str],
    *,
    preferred: list[str] | None = None,
) -> list[str]:
    """Pick a stable join key set present on both sides."""
    feat = set(feature_columns)
    lab = set(label_columns)
    if preferred:
        keys = [k for k in preferred if k in feat and k in lab]
        if keys:
            return keys
    if "master_row_id" in feat and "master_row_id" in lab:
        return ["master_row_id"]
    if "sample_id" in feat and "sample_id" in lab:
        return ["sample_id"]
    composite = [c for c in ("trading_day", "timestamp", "token") if c in feat and c in lab]
    if len(composite) == 3:
        return composite
    raise ValueError(
        "Cannot join Feature Dataset to Label Run — need shared master_row_id "
        "or (trading_day, timestamp, token)."
    )


def join_feature_frame_with_label_run(
    feature_df: pd.DataFrame,
    data_dir: str,
    label_run_id: str,
    *,
    drop_invalid: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Left-identity join: keep feature rows that match the Label Run.

    Attaches ``primary_target`` (and optional display columns) once.
    """
    meta = load_label_run_meta(data_dir, label_run_id)
    primary = str(meta.get("primary_target") or "label_id")
    labels = load_label_run_frame(data_dir, label_run_id)
    preferred = list(meta.get("join_keys") or [])
    keys = resolve_join_keys(list(feature_df.columns), list(labels.columns), preferred=preferred)

    label_cols = list(dict.fromkeys([*keys, primary, "label_name", "is_valid", "invalid_reason"]))
    label_cols = [c for c in label_cols if c in labels.columns]
    lab = labels[label_cols].copy()

    if drop_invalid and "is_valid" in lab.columns:
        before = len(lab)
        lab = lab[lab["is_valid"].fillna(True).astype(bool)]
        dropped_invalid = before - len(lab)
    else:
        dropped_invalid = 0

    # Detect duplicate keys on label side (would fan out features).
    dup = int(lab.duplicated(subset=keys).sum())
    if dup:
        raise ValueError(
            f"Label Run {label_run_id!r} has {dup} duplicate join key rows "
            f"on {keys}; refuse to train."
        )

    before_feat = len(feature_df)
    merged = feature_df.merge(lab, on=keys, how="inner", suffixes=("", "_label"))
    # If primary already existed on features (legacy), prefer Label Run column.
    if f"{primary}_label" in merged.columns:
        merged[primary] = merged[f"{primary}_label"]
        merged.drop(columns=[f"{primary}_label"], inplace=True)

    info = {
        "label_run_id": label_run_id,
        "join_keys": keys,
        "primary_target": primary,
        "rows_features_before": before_feat,
        "rows_labels": int(len(lab)),
        "rows_after_join": int(len(merged)),
        "dropped_invalid_labels": int(dropped_invalid),
        "strategy": meta.get("strategy"),
        "parameters": dict(meta.get("parameters") or {}),
    }
    if merged.empty:
        raise ValueError(
            f"Join Feature Dataset ⟕ Label Run {label_run_id!r} produced 0 rows "
            f"(keys={keys})."
        )
    return merged, info

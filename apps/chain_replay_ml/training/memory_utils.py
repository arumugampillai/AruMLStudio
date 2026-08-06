"""Memory helpers for large training matrices."""

from __future__ import annotations

import gc
from typing import Any

import numpy as np
import pandas as pd

# Chronological tail cap — keeps most recent rows when dataset exceeds this.
MAX_TRAINING_ROWS = 750_000


def compact_numeric_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Downcast float64/int64 columns to float32/int32 to halve RAM use."""
    out = df.copy()
    for col in out.columns:
        s = out[col]
        if pd.api.types.is_float_dtype(s):
            out[col] = s.astype(np.float32)
        elif pd.api.types.is_integer_dtype(s):
            out[col] = pd.to_numeric(s, downcast="integer")
    return out


def cap_training_rows(
    df: pd.DataFrame,
    *,
    max_rows: int = MAX_TRAINING_ROWS,
) -> tuple[pd.DataFrame, dict[str, Any] | None]:
    """Keep the most recent rows when the frame is too large for RAM."""
    n = len(df)
    if n <= max_rows:
        return df, None
    trimmed = df.iloc[-max_rows:].reset_index(drop=True)
    return trimmed, {
        "training_rows_capped": True,
        "original_rows": n,
        "training_rows": len(trimmed),
        "max_training_rows": max_rows,
        "note": (
            f"Training used the most recent {max_rows:,} of {n:,} rows to reduce memory pressure. "
            "Rebuild with fewer trading days or increase system RAM for full-dataset training."
        ),
    }


def release_memory() -> None:
    gc.collect()

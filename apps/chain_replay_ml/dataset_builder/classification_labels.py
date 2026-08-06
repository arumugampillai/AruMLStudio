"""Binary classification labels derived at Master Dataset export time.

Uses existing ``ltp`` (alias: current_ltp) and ``future_ltp_5m`` only.
Does not modify the Master Dataset schema.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

# (column_name, multiplier, compare)  compare: "ge" (>=) or "gt" (>)
_UP_PCT_5M_SPECS: tuple[tuple[str, float, str], ...] = (
    ("label_up_2pct_5m", 1.02, "ge"),
    ("label_up_3pct_5m", 1.03, "ge"),
    ("label_up_4pct_5m", 1.04, "ge"),
    ("label_up_5pct_5m", 1.05, "ge"),
    ("label_up_6pct_5m", 1.06, "ge"),
    ("label_up_gt6pct_5m", 1.06, "gt"),
)

CLASSIFICATION_LABEL_COLUMNS_5M: tuple[str, ...] = tuple(s[0] for s in _UP_PCT_5M_SPECS)

FUTURE_LTP_5M = "future_ltp_5m"
_CURRENT_CANDIDATES = ("ltp", "current_ltp")


def resolve_current_ltp_column(columns: list[str] | set[str] | pd.Index) -> str | None:
    cols = set(str(c) for c in columns)
    for name in _CURRENT_CANDIDATES:
        if name in cols:
            return name
    return None


def can_generate_up_pct_labels_5m(columns: list[str] | set[str] | pd.Index) -> bool:
    cols = set(str(c) for c in columns)
    return FUTURE_LTP_5M in cols and resolve_current_ltp_column(cols) is not None


def attach_up_pct_classification_labels_5m(df: pd.DataFrame) -> list[str]:
    """Add the six binary up-% labels in-place. Returns names of columns written.

    Values are ``1.0`` / ``0.0`` when both prices are present; ``NaN`` otherwise.
    No-op (returns []) if required source columns are missing.
    """
    if df is None or not len(df.columns):
        return []
    if not can_generate_up_pct_labels_5m(df.columns):
        return []

    current_col = resolve_current_ltp_column(df.columns)
    assert current_col is not None
    current = pd.to_numeric(df[current_col], errors="coerce")
    future = pd.to_numeric(df[FUTURE_LTP_5M], errors="coerce")
    valid = current.notna() & future.notna() & (current > 0)

    added: list[str] = []
    for col, mult, cmp_op in _UP_PCT_5M_SPECS:
        thresh = current * float(mult)
        if cmp_op == "ge":
            hit = future >= thresh
        else:
            hit = future > thresh
        values = np.where(valid, hit.astype(float), np.nan)
        df[col] = values
        added.append(col)
    return added


def merge_classification_targets(
    target_columns: list[str],
    *,
    generated: list[str] | tuple[str, ...] | None = None,
) -> list[str]:
    """Append generated classification labels to regression target list (stable, unique)."""
    out = list(dict.fromkeys(str(c) for c in target_columns if c))
    for col in generated or CLASSIFICATION_LABEL_COLUMNS_5M:
        c = str(col)
        if c and c not in out:
            out.append(c)
    return out


def classification_label_meta() -> dict[str, Any]:
    """Small audit blob for dataset JSON."""
    return {
        "source_current": "ltp|current_ltp",
        "source_future": FUTURE_LTP_5M,
        "columns": list(CLASSIFICATION_LABEL_COLUMNS_5M),
        "definitions": {
            "label_up_2pct_5m": "future_ltp_5m >= current_ltp * 1.02",
            "label_up_3pct_5m": "future_ltp_5m >= current_ltp * 1.03",
            "label_up_4pct_5m": "future_ltp_5m >= current_ltp * 1.04",
            "label_up_5pct_5m": "future_ltp_5m >= current_ltp * 1.05",
            "label_up_6pct_5m": "future_ltp_5m >= current_ltp * 1.06",
            "label_up_gt6pct_5m": "future_ltp_5m > current_ltp * 1.06",
        },
    }

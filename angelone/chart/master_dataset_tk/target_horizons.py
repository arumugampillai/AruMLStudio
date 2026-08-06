"""Prediction target horizons — shared with web Create Dataset UI."""

from __future__ import annotations

from typing import Any

from chain_replay_ml.dataset_builder.feature_plugins import horizon_column_name, horizon_label
from chain_replay_ml.dataset_builder.schema_feature_meta import TARGET_DEFINITIONS

ALL_HORIZON_SEC: list[int] = sorted(
    int(t["prediction_horizon_sec"]) for t in TARGET_DEFINITIONS
)
DEFAULT_HORIZON_SEC: list[int] = list(ALL_HORIZON_SEC)
TARGET_TYPE_LABEL = "Future LTP"
_SECONDS_MAX = 59


def compact_target_label(sec: int) -> str:
    """Tk target tab label, e.g. Future LTP (3 s) or Future LTP (1 m)."""
    if sec >= 60 and sec % 60 == 0:
        return f"Future LTP ({sec // 60} m)"
    return f"Future LTP ({sec} s)"


def target_horizon_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw in TARGET_DEFINITIONS:
        sec = int(raw["prediction_horizon_sec"])
        rows.append({
            "sec": sec,
            "name": str(raw.get("name") or horizon_column_name(sec)),
            "display_name": compact_target_label(sec),
            "description": str(raw.get("description") or ""),
            "interpretation": str(raw.get("interpretation") or ""),
            "short": horizon_label(sec),
            "unit": "minute" if sec > _SECONDS_MAX else "second",
        })
    return sorted(rows, key=lambda r: r["sec"])


def target_horizon_columns() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    seconds: list[dict[str, Any]] = []
    minutes: list[dict[str, Any]] = []
    for row in target_horizon_rows():
        if row["unit"] == "minute":
            minutes.append(row)
        else:
            seconds.append(row)
    return seconds, minutes


def horizons_summary_labels(horizons_sec: list[int]) -> str:
    selected = {int(h) for h in horizons_sec}
    labels = [
        row["display_name"]
        for row in target_horizon_rows()
        if row["sec"] in selected
    ]
    return ", ".join(labels) if labels else "—"


def default_horizon_selection() -> dict[int, bool]:
    return {sec: True for sec in ALL_HORIZON_SEC}

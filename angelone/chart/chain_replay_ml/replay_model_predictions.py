"""Registry inference — delegates to live_inference layered pipeline."""

from __future__ import annotations

from typing import Any

from live_inference.grid import (
    align_replay_clock_ts,
    resolve_feature_grid_ts_legacy as resolve_feature_grid_ts,
)
from live_inference.pipeline import run_registry_inference_at_clock as predict_registry_models_at_clock

__all__ = [
    "align_replay_clock_ts",
    "predict_registry_models_at_clock",
    "resolve_feature_grid_ts",
]

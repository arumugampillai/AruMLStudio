"""ML inference status and replay validation helpers (registry models)."""

from __future__ import annotations

import json
import os
import sqlite3
import sys
from typing import Any

_CHART_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _CHART_DIR not in sys.path:
    sys.path.insert(0, _CHART_DIR)

from storage.chain_replay_export import (
    ChainReplayError,
    bootstrap_provider_for_underlying,
    ist_market_session_bounds,
    normalize_expiry_param,
    require_v1_ticks_schema,
    resolve_chain_tokens,
)

from .bs import expiry_close_ts, format_time_hhmm, normalize_strike_rupees
from .constants import DEFAULT_TARGET, FEATURE_COLUMNS_SIDE, SUPPORTED_TARGETS
from .features import build_option_rows
from .pipeline import INDEX_CONFIG, normalize_index_name, replay_db_path
from .reanchor import ReanchorThresholds
from .ticks import load_tick_timelines
from .training.default_model import resolve_default_model_name

HERE = os.path.dirname(os.path.abspath(__file__))
CHART_DIR = os.path.dirname(HERE)
DEFAULT_DATA_DIR = os.path.join(CHART_DIR, "data")


def default_data_dir() -> str:
    return DEFAULT_DATA_DIR


def ml_status(data_dir: str = DEFAULT_DATA_DIR) -> dict[str, Any]:
    """Return the active or latest registry model package."""
    name = resolve_default_model_name(data_dir)
    if not name:
        return {"enabled": False, "data_dir": data_dir}
    row = get_trained_model(data_dir, name) or {}
    paths = os.path.join(data_dir, "models", name)
    model_path = os.path.join(paths, "model.json")
    tmeta_path = os.path.join(paths, "training_metadata.json")
    if os.path.isfile(tmeta_path):
        try:
            with open(tmeta_path, encoding="utf-8") as fh:
                tmeta = json.load(fh)
            prod = str(tmeta.get("production_model") or "").strip()
            if prod:
                cand = os.path.join(paths, prod)
                if os.path.isfile(cand):
                    model_path = cand
        except Exception:
            pass
    return {
        "enabled": True,
        "data_dir": data_dir,
        "model_name": name,
        "stamp": name,
        "target": row.get("target"),
        "dataset": row.get("dataset"),
        "algorithm": row.get("algorithm"),
        "trained_at": row.get("trained_at"),
        "model_path": model_path,
    }


def predict_batch_replay(
    *,
    underlying: str,
    expiry: str,
    date: str,
    token: str,
    option_type: str,
    strike_rupees: float,
    row_times: list[float],
    iv_threshold_pct: float = 2.0,
    spot_threshold_pct: float = 0.3,
    max_roll_age_min: float = 15.0,
    data_dir: str = DEFAULT_DATA_DIR,
    model_name: str | None = None,
    model_stamp: str | None = None,
    target: str | None = None,
) -> dict[str, Any]:
    """Residual CE/PE batch predict is retired — use registry model analysis instead."""
    del (
        underlying, expiry, date, token, option_type, strike_rupees, row_times,
        iv_threshold_pct, spot_threshold_pct, max_roll_age_min, data_dir,
        model_name, model_stamp, target,
    )
    raise ChainReplayError(
        "Legacy CE/PE ml_models batch predict is removed. "
        "Use Model Builder registry models and /api/replay/trades for analysis."
    )

"""Version stamps for immutable inference snapshots."""

from __future__ import annotations

from chain_replay_ml.replay_engine_version import replay_engine_version

# Bump when LiveMarketState / tick ingestion semantics change.
MARKET_STATE_VERSION = "1.0"

# Bump when feature engineering logic or feature catalog changes.
FEATURE_ENGINE_VERSION = f"fe-{replay_engine_version()}"

# Bump when PredictionResult / Meta Engine contract changes.
PREDICTION_ENGINE_VERSION = "1.0"


def feature_version() -> str:
    return FEATURE_ENGINE_VERSION


def market_state_version() -> str:
    return MARKET_STATE_VERSION


def prediction_version() -> str:
    return PREDICTION_ENGINE_VERSION

"""Replay engine version — bump when replay/scoring logic changes (invalidates disk cache)."""

from __future__ import annotations

from chain_replay_ml.replay_scoring_cache import _CACHE_VERSION as _SCORING_CACHE_VERSION

# Keep in sync with _REPLAY_TRADES_CACHE_VERSION in main.py when replay pipeline changes.
_REPLAY_TRADES_PIPELINE_VERSION = 42


def replay_engine_version() -> str:
    return f"sc{_SCORING_CACHE_VERSION}-tp{_REPLAY_TRADES_PIPELINE_VERSION}"

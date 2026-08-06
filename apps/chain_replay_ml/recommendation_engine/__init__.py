"""Recommendation Engine (Phase 5.3) — deterministic experiment suggestions.

Compute package name: ``recommendation_engine``.
Artifact / UI folder: ``experiment_planner/`` (Experiment Planner tab).

See docs/project-main/feature-studio/EXPERIMENT_PLANNER.md
"""

from __future__ import annotations

from .api import run_recommendation_engine
from .types import RecommendationEngineResult

__all__ = [
    "RecommendationEngineResult",
    "run_recommendation_engine",
]

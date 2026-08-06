"""Lifecycle reset hooks for the Feature Policy Engine."""

from __future__ import annotations

from .types import FeatureCategory, FeatureLifecycle


def should_reset_on_session_start(category: FeatureCategory, lifecycle: FeatureLifecycle) -> bool:
    return lifecycle in (FeatureLifecycle.SESSION, FeatureLifecycle.DAY)


def should_reset_on_gap(category: FeatureCategory, *, reset_on_gap: bool) -> bool:
    return category == FeatureCategory.ROLLING and reset_on_gap

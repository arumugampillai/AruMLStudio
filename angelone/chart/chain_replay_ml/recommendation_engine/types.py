"""Public types for Recommendation Engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class RecommendationEngineResult:
    ok: bool
    model_name: str
    package_dir: str
    artifacts_dir: str
    suggestions: list[dict[str, Any]] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

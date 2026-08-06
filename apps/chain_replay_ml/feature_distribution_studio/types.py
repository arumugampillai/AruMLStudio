"""Public types for Feature Distribution Studio."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class DistributionStudioResult:
    """v1 result: holdout stats + comparison rows for UI."""

    ok: bool
    model_name: str
    package_dir: str
    artifacts_dir: str
    holdout_stats: list[dict[str, Any]] = field(default_factory=list)
    comparison: list[dict[str, Any]] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

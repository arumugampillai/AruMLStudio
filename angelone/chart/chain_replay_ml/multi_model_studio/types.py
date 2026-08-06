"""Public types for Multi-model Feature Studio."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class MultiModelStudioResult:
    ok: bool
    model_a: str
    model_b: str
    pair_dir: str
    artifacts_dir: str
    comparison: list[dict[str, Any]] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

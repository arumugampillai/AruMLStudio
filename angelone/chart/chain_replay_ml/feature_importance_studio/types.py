"""Public types for Feature Importance Studio."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ImportanceStudioResult:
    """v1 result: native + permutation + SHAP + comparison rows."""

    ok: bool
    model_name: str
    package_dir: str
    artifacts_dir: str
    native: list[dict[str, Any]] = field(default_factory=list)
    permutation: list[dict[str, Any]] = field(default_factory=list)
    shap: list[dict[str, Any]] = field(default_factory=list)
    comparison: list[dict[str, Any]] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

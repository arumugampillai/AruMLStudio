"""Public types for Diagnostics Studio."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class DiagnosticsStudioResult:
    ok: bool
    model_name: str
    package_dir: str
    artifacts_dir: str
    summary: dict[str, Any] = field(default_factory=dict)
    narrative: list[str] = field(default_factory=list)
    comparison: list[dict[str, Any]] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

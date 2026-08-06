"""Diagnostics Studio (Phase 4.5) — join studios + metrics into a short diagnosis.

Does not recompute Holdout Performance. See docs/project-main/feature-studio/DIAGNOSTICS_STUDIO.md
"""

from __future__ import annotations

from .api import run_diagnostics_studio
from .types import DiagnosticsStudioResult

__all__ = [
    "DiagnosticsStudioResult",
    "run_diagnostics_studio",
]

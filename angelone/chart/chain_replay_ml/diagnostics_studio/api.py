"""Diagnostics Studio API — Phase 4.5 (join + summarize)."""

from __future__ import annotations

from typing import Any, Callable

from .compute import run_compute
from .types import DiagnosticsStudioResult

ProgressCb = Callable[[dict[str, Any]], None]


def run_diagnostics_studio(
    *,
    data_dir: str,
    model_name: str,
    package_dir: str | None = None,
    require: tuple[str, ...] = (),
    progress: ProgressCb | None = None,
) -> DiagnosticsStudioResult:
    """Join 4.1–4.3 artifacts + saved metrics into a short holdout diagnosis.

    Artifacts under ``diagnostics_studio/``. Does not run full Holdout Performance.
    """
    return run_compute(
        data_dir=data_dir,
        model_name=model_name,
        package_dir=package_dir,
        require=require,
        progress=progress,
    )

"""Multi-model Feature Studio API — Phase 4.4 (join-only)."""

from __future__ import annotations

from typing import Any, Callable

from .compute import run_compute
from .types import MultiModelStudioResult

ProgressCb = Callable[[dict[str, Any]], None]


def run_multi_model_studio(
    *,
    data_dir: str,
    model_a: str,
    model_b: str,
    package_dir_a: str | None = None,
    package_dir_b: str | None = None,
    require: tuple[str, ...] = (),
    progress: ProgressCb | None = None,
) -> MultiModelStudioResult:
    """Join precomputed Importance / Distribution / Drift artifacts for two models.

    Does not recompute studios. Artifacts under ``{data_dir}/multi_model_studio/``.
    """
    return run_compute(
        data_dir=data_dir,
        model_a=model_a,
        model_b=model_b,
        package_dir_a=package_dir_a,
        package_dir_b=package_dir_b,
        require=require,
        progress=progress,
    )

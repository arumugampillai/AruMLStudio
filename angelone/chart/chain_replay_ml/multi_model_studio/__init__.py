"""Multi-model Feature Studio (Phase 4.4) — join Importance/Dist/Drift across 2 models.

Does not recompute studios. Complements metrics Model Comparison.
See docs/project-main/feature-studio/MULTI_MODEL_STUDIO.md
"""

from __future__ import annotations

from .api import run_multi_model_studio
from .types import MultiModelStudioResult

__all__ = [
    "MultiModelStudioResult",
    "run_multi_model_studio",
]

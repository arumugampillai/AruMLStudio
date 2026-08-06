"""Feature Importance Studio (Phase 4.1) — analyze existing model packages.

No retraining. Inputs: model package + holdout via Dataset Engine.
v1: native XGB (gain/weight/cover) → permutation → TreeSHAP → comparison table.

See docs/project-main/feature-studio/FEATURE_IMPORTANCE_STUDIO.md
"""

from __future__ import annotations

from .api import run_feature_importance_studio
from .types import ImportanceStudioResult

__all__ = [
    "ImportanceStudioResult",
    "run_feature_importance_studio",
]

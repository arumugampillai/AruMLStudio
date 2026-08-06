"""Feature Distribution Studio (Phase 4.2) — holdout univariate stats.

No retraining. Complements Feature Importance Studio.
See docs/project-main/feature-studio/FEATURE_DISTRIBUTION_STUDIO.md
"""

from __future__ import annotations

from .api import run_feature_distribution_studio
from .types import DistributionStudioResult

__all__ = [
    "DistributionStudioResult",
    "run_feature_distribution_studio",
]

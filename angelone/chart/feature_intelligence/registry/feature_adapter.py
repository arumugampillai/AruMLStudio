"""One-way pull adapter: legacy Feature Registry → FIC (Sprint 2).

Delegates to :mod:`feature_registry_synchronizer` (FEAT preservation + SyncSummary).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from feature_intelligence.registry.feature_models import SyncReport
from feature_intelligence.registry.feature_registry_synchronizer import (
    synchronize_feature_registry,
)
from feature_intelligence.registry.primitive_mapping import PrimitiveMappingProvider

if TYPE_CHECKING:
    from feature_intelligence.registry.feature_service import FeatureRegistryService


def sync_from_legacy(
    service: FeatureRegistryService,
    data_dir: str | Path,
    *,
    mode: str = "strict",
    force: bool = False,
    mapping_provider: PrimitiveMappingProvider | None = None,
    research_sync: bool = False,
) -> SyncReport:
    """Pull from ``build_feature_registry_catalog`` into FIC.

    Returns legacy :class:`SyncReport`. Prefer
    :func:`synchronize_feature_registry` for the full :class:`SyncSummary`.
    """
    summary = synchronize_feature_registry(
        service,
        data_dir,
        mode=mode,
        force=force,
        research_sync=research_sync,
        mapping_provider=mapping_provider,
    )
    return summary.to_sync_report()

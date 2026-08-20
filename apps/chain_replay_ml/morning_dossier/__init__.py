"""Phase 4F.6: Morning Research Dossier & Model Research Lab UI Presentation."""

from .feature_intelligence import extract_discovered_feature_intelligence
from .generator import (
    export_morning_dossier_markdown,
    generate_morning_research_dossier,
)
from .types import (
    CandidateFeatureDeltaView,
    DiscoveredFeatureRecord,
    DiscoveredFeatureStatus,
    DiscoveredFeatureSynergy,
    FeatureGovernanceAuditSummary,
    LineageNodeView,
    MorningResearchDossier,
)

__all__ = [
    "CandidateFeatureDeltaView",
    "DiscoveredFeatureRecord",
    "DiscoveredFeatureStatus",
    "DiscoveredFeatureSynergy",
    "FeatureGovernanceAuditSummary",
    "LineageNodeView",
    "MorningResearchDossier",
    "export_morning_dossier_markdown",
    "extract_discovered_feature_intelligence",
    "generate_morning_research_dossier",
]


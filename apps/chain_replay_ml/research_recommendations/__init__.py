"""Automated Project Recommendations Subsystem (Phase 4E).

Advisory research hypothesis generator and project recommendation engine.
Synthesizes research memory, evidence density, champion vulnerabilities,
feature intelligence, negative pruning, and multi-objective scoring into
actionable, explainable research recommendation dossiers.
"""

from __future__ import annotations

from .coverage import (
    ContextCoverage,
    CoverageClass,
    CoverageMatrix,
    analyze_context_coverage,
    build_coverage_matrix,
    classify_coverage,
    compute_evidence_density_score,
)
from .dossier import (
    RecommendationDossier,
    build_recommendation_dossier,
    generate_context_recommendation_dossiers,
)
from .feature_affinity import (
    ContextFeatureAffinityReport,
    FeatureAffinityResult,
    FeatureInteractionResult,
    FeatureRecommendationClass,
    analyze_feature_affinity,
    compute_feature_affinity_score,
    compute_feature_confidence,
    compute_interaction_synergy_score,
    recommend_features_for_context,
)
from .negative_pruning import (
    ContextPruningAgenda,
    ExclusionReason,
    ExclusionVerdict,
    PruningAuditResult,
    audit_experiment_exclusion,
    audit_signature_exclusion,
    build_context_pruning_agenda,
    is_search_path_excluded,
)
from .priority_scoring import (
    ComponentScoreBreakdown,
    ContextPriorityAgendaReport,
    EvidenceConfidenceLevel,
    OpportunityType,
    ResearchOpportunity,
    ResearchPriorityClass,
    build_context_priority_agenda,
    classify_evidence_confidence,
    classify_priority_score,
    compute_research_priority_score,
    evaluate_research_opportunity,
)
from .vulnerability import (
    ChallengerStatus,
    ChampionVulnerabilityResult,
    ResearchPriority,
    VulnerabilityAuditReport,
    VulnerabilityClass,
    audit_all_context_vulnerabilities,
    audit_champion_vulnerability,
    compute_champion_vulnerability_score,
)

__all__ = [
    "ChallengerStatus",
    "ChampionVulnerabilityResult",
    "ComponentScoreBreakdown",
    "ContextCoverage",
    "ContextFeatureAffinityReport",
    "ContextPriorityAgendaReport",
    "ContextPruningAgenda",
    "CoverageClass",
    "CoverageMatrix",
    "EvidenceConfidenceLevel",
    "ExclusionReason",
    "ExclusionVerdict",
    "FeatureAffinityResult",
    "FeatureInteractionResult",
    "FeatureRecommendationClass",
    "OpportunityType",
    "PruningAuditResult",
    "RecommendationDossier",
    "ResearchOpportunity",
    "ResearchPriority",
    "ResearchPriorityClass",
    "VulnerabilityAuditReport",
    "VulnerabilityClass",
    "analyze_context_coverage",
    "analyze_feature_affinity",
    "audit_all_context_vulnerabilities",
    "audit_champion_vulnerability",
    "audit_experiment_exclusion",
    "audit_signature_exclusion",
    "build_context_priority_agenda",
    "build_context_pruning_agenda",
    "build_coverage_matrix",
    "build_recommendation_dossier",
    "classify_coverage",
    "classify_evidence_confidence",
    "classify_priority_score",
    "compute_champion_vulnerability_score",
    "compute_evidence_density_score",
    "compute_feature_affinity_score",
    "compute_feature_confidence",
    "compute_interaction_synergy_score",
    "compute_research_priority_score",
    "evaluate_research_opportunity",
    "generate_context_recommendation_dossiers",
    "is_search_path_excluded",
    "recommend_features_for_context",
]

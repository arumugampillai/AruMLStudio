"""Research Recommendation Dossier Generator (Phase 4E.6).

Transforms raw multi-objective research priority scores into human-interpretable,
governance-ready research dossiers explaining the rationale, missing evidence,
caution factors, and suggested next steps for each candidate research opportunity.

Invariants:
1. Strictly Advisory: Zero automated model promotion, demotion, deployment, or training.
2. Dynamic Synthesis: Derives dossiers on-demand from existing Phase 4E services without redundant DB tables.
3. Transparent Explainability: Fully exposes component score contributions and empirical evidence traces.
4. Human Governance Boundary: Clear separation between Production Champion and Advisory Candidate.
"""

from __future__ import annotations

import json
import math
import os
import sqlite3
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Sequence

from chain_replay_ml.dataset_builder.schema_registry import load_schema_registry
from chain_replay_ml.model_taxonomy.specs import ModelContextKey
from chain_replay_ml.research_memory.benchmarks import get_model_benchmarks_for_context
from chain_replay_ml.research_memory.db import connect_analysis_db, init_analysis_db
from chain_replay_ml.research_recommendations.coverage import (
    CoverageClass,
    analyze_context_coverage,
)
from chain_replay_ml.research_recommendations.feature_affinity import (
    FeatureRecommendationClass,
    analyze_feature_affinity,
    recommend_features_for_context,
)
from chain_replay_ml.research_recommendations.negative_pruning import (
    ExclusionReason,
    ExclusionVerdict,
    audit_experiment_exclusion,
)
from chain_replay_ml.research_recommendations.priority_scoring import (
    ComponentScoreBreakdown,
    ContextPriorityAgendaReport,
    EvidenceConfidenceLevel,
    OpportunityType,
    ResearchOpportunity,
    ResearchPriorityClass,
    build_context_priority_agenda,
    evaluate_research_opportunity,
)
from chain_replay_ml.research_recommendations.vulnerability import (
    ChallengerStatus,
    ChampionVulnerabilityResult,
    VulnerabilityClass,
    audit_champion_vulnerability,
)
from chain_replay_ml.training.lifecycle_store import get_champion_for_context


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class RecommendationDossier:
    """Human-readable, explainable research recommendation dossier for a single opportunity."""

    context_key: str
    opportunity_id: str
    priority_class: ResearchPriorityClass
    priority_score: float
    opportunity_type: OpportunityType
    evidence_confidence: EvidenceConfidenceLevel
    confidence_value: float
    exclusion_verdict: ExclusionVerdict
    exclusion_reason: ExclusionReason
    caution_warnings: list[str]
    champion_vulnerability_contrib: float
    challenger_gap_contrib: float
    coverage_gap_contrib: float
    feature_affinity_contrib: float
    interaction_synergy_contrib: float
    caution_penalty: float
    candidate_features: list[str]
    target_algorithm: str
    why_recommended: str
    missing_evidence_summary: str
    suggested_next_steps: list[str]
    supporting_empirical_evidence: dict[str, Any]
    production_champion_context: dict[str, Any]
    generated_at: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "context_key": self.context_key,
            "opportunity_id": self.opportunity_id,
            "priority_class": self.priority_class.value,
            "priority_score": self.priority_score,
            "opportunity_type": self.opportunity_type.value,
            "evidence_confidence": self.evidence_confidence.value,
            "confidence_value": self.confidence_value,
            "exclusion_verdict": self.exclusion_verdict.value,
            "exclusion_reason": self.exclusion_reason.value,
            "caution_warnings": self.caution_warnings,
            "champion_vulnerability_contrib": self.champion_vulnerability_contrib,
            "challenger_gap_contrib": self.challenger_gap_contrib,
            "coverage_gap_contrib": self.coverage_gap_contrib,
            "feature_affinity_contrib": self.feature_affinity_contrib,
            "interaction_synergy_contrib": self.interaction_synergy_contrib,
            "caution_penalty": self.caution_penalty,
            "candidate_features": self.candidate_features,
            "target_algorithm": self.target_algorithm,
            "why_recommended": self.why_recommended,
            "missing_evidence_summary": self.missing_evidence_summary,
            "suggested_next_steps": self.suggested_next_steps,
            "supporting_empirical_evidence": self.supporting_empirical_evidence,
            "production_champion_context": self.production_champion_context,
            "generated_at": self.generated_at,
        }


def build_recommendation_dossier(
    data_dir: str,
    opportunity: ResearchOpportunity,
    *,
    schema: dict[str, Any] | None = None,
) -> RecommendationDossier:
    """Build a detailed, explainable research recommendation dossier from an opportunity."""
    init_analysis_db(data_dir)
    c_key = opportunity.context_key
    reg_schema = schema if schema is not None else load_schema_registry()

    # 1. Gather Context Diagnostics
    cov = analyze_context_coverage(data_dir, c_key)
    vuln = audit_champion_vulnerability(data_dir, c_key)
    prod_doc = get_champion_for_context(data_dir, c_key)

    champ_name = prod_doc.get("champion_model_name") if prod_doc else vuln.champion_model_name
    chall_name = prod_doc.get("challenger_model_name") if prod_doc else vuln.production_challenger_name

    prod_context_info = {
        "champion_model_name": champ_name,
        "challenger_model_name": chall_name,
        "champion_robustness_score": vuln.champion_robustness_score,
        "champion_degradation_pct": vuln.champion_degradation_pct,
        "champion_ece": vuln.champion_ece,
        "vulnerability_class": vuln.vulnerability_class.value,
        "vulnerability_score": vuln.vulnerability_score,
        "challenger_status": vuln.challenger_status.value,
    }

    # 2. Extract Warnings and Caution Factors
    caution_warnings: list[str] = []
    if opportunity.exclusion_verdict == ExclusionVerdict.CAUTION:
        caution_warnings.append(f"Caution Flag: {opportunity.exclusion_reason.value} ({opportunity.rationale})")
    if vuln.vulnerability_class == VulnerabilityClass.FRAGILE:
        caution_warnings.append(f"Production Champion is fragile (Degradation: {vuln.champion_degradation_pct or 0.0:.1f}%, ECE: {vuln.champion_ece or 0.0:.4f}).")
    if cov.coverage_class == CoverageClass.COLD_START:
        caution_warnings.append("Cold-start context: Baseline evidence is unobserved.")

    # 3. Construct "Why Recommended" Narrative
    breakdown = opportunity.component_breakdown
    why_parts: list[str] = []

    if opportunity.opportunity_type == OpportunityType.CHAMPION_VULNERABILITY_DEFENSE:
        why_parts.append(f"Production Champion '{champ_name or 'None'}' exhibits operational vulnerability (Score: {vuln.vulnerability_score*100.0:.1f}).")
        why_parts.append(f"Testing missing or candidate features [{', '.join(opportunity.candidate_features)}] can defend model robustness.")
    elif opportunity.opportunity_type == OpportunityType.CHALLENGER_OPPORTUNITY:
        why_parts.append(f"Research Challenger leads Production Champion by {vuln.challenger_gap or 0.0:+.2f} pts in robustness.")
        why_parts.append("Validating this candidate under cross-regime stress will mature the challenger pipeline.")
    elif opportunity.opportunity_type == OpportunityType.FEATURE_EXPLORATION:
        why_parts.append(f"Targeting candidate features [{', '.join(opportunity.candidate_features)}] with mean empirical score {breakdown.feature_affinity_score:.1f}.")
    elif opportunity.opportunity_type == OpportunityType.INTERACTION_VALIDATION:
        why_parts.append(f"Empirical feature pair [{', '.join(opportunity.candidate_features)}] demonstrates interaction lift score {breakdown.interaction_synergy_score:.1f}.")
    elif opportunity.opportunity_type == OpportunityType.COVERAGE_EXPANSION:
        why_parts.append(f"Context '{c_key}' has sparse evidence (Density: {cov.evidence_density_score:.1f}/100). Baseline expansion recommended.")

    why_recommended = " ".join(why_parts)

    # 4. Construct "Missing Evidence" Narrative
    missing_parts: list[str] = []
    if cov.benchmark_count < 5:
        missing_parts.append(f"Benchmark sample size is low ({cov.benchmark_count}/5 minimum for mature confidence).")
    if cov.unique_features_count < 10:
        missing_parts.append(f"Feature exploration is sparse ({cov.unique_features_count} unique features tested).")
    if opportunity.evidence_confidence in (EvidenceConfidenceLevel.WEAK, EvidenceConfidenceLevel.INSUFFICIENT):
        missing_parts.append("Confidence is constrained by limited historical experiment volume.")

    missing_evidence_summary = " ".join(missing_parts) if missing_parts else "Evidence base is mature across standard benchmarks."

    # 5. Formulate Suggested Next Steps
    feats_arg = ", ".join(opportunity.candidate_features)
    next_steps: list[str] = [
        f"Design research experiment using algorithm '{opportunity.target_algorithm}' and features [{feats_arg}].",
        "Execute cross-regime evaluation across all 7 baseline market regimes (R001-R007).",
        "Audit Expected Calibration Error (ECE) and fold variance before considering candidate promotion.",
    ]

    # 6. Supporting Empirical Evidence Summary
    supporting_evidence = {
        "context_benchmark_count": cov.benchmark_count,
        "evidence_density_score": cov.evidence_density_score,
        "coverage_class": cov.coverage_class.value,
        "feature_affinity_score": breakdown.feature_affinity_score,
        "interaction_synergy_score": breakdown.interaction_synergy_score,
        "champion_vulnerability_score": breakdown.champion_vulnerability_score,
        "challenger_gap_score": breakdown.challenger_gap_score,
        "coverage_gap_score": breakdown.coverage_gap_score,
        "caution_penalty": breakdown.caution_penalty,
    }

    return RecommendationDossier(
        context_key=c_key,
        opportunity_id=opportunity.opportunity_id,
        priority_class=opportunity.priority_class,
        priority_score=opportunity.priority_score,
        opportunity_type=opportunity.opportunity_type,
        evidence_confidence=opportunity.evidence_confidence,
        confidence_value=opportunity.confidence_value,
        exclusion_verdict=opportunity.exclusion_verdict,
        exclusion_reason=opportunity.exclusion_reason,
        caution_warnings=caution_warnings,
        champion_vulnerability_contrib=breakdown.champion_vulnerability_score * 0.30,
        challenger_gap_contrib=breakdown.challenger_gap_score * 0.25,
        coverage_gap_contrib=breakdown.coverage_gap_score * 0.15,
        feature_affinity_contrib=breakdown.feature_affinity_score * 0.20,
        interaction_synergy_contrib=breakdown.interaction_synergy_score * 0.10,
        caution_penalty=breakdown.caution_penalty,
        candidate_features=opportunity.candidate_features,
        target_algorithm=opportunity.target_algorithm,
        why_recommended=why_recommended,
        missing_evidence_summary=missing_evidence_summary,
        suggested_next_steps=next_steps,
        supporting_empirical_evidence=supporting_evidence,
        production_champion_context=prod_context_info,
        generated_at=_utc_now_iso(),
    )


def generate_context_recommendation_dossiers(
    data_dir: str,
    context_key: str,
    *,
    schema: dict[str, Any] | None = None,
) -> list[RecommendationDossier]:
    """Generate complete list of explainable recommendation dossiers for all eligible & caution opportunities."""
    agenda = build_context_priority_agenda(data_dir, context_key, schema=schema)
    all_opps = agenda.eligible_opportunities + agenda.caution_opportunities

    dossiers: list[RecommendationDossier] = []
    for opp in all_opps:
        dossier = build_recommendation_dossier(data_dir, opp, schema=schema)
        dossiers.append(dossier)

    # Sort deterministically by priority score descending
    dossiers.sort(key=lambda d: (-d.priority_score, d.context_key, d.opportunity_id))
    return dossiers

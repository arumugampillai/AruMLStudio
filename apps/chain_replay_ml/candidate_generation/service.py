"""High-level Automated Candidate Generation Service (Phase 4F.2).

Orchestrates candidate generation, duplicate detection (via Phase 4D experiment_signatures),
negative-evidence pruning (via Phase 4E.4 negative_pruning), and multi-objective Phase 4E agenda ingestion.
"""

from __future__ import annotations

from typing import Any, Sequence

from chain_replay_ml.research_memory.db import init_analysis_db
from chain_replay_ml.research_memory.signature import check_experiment_exists
from chain_replay_ml.research_recommendations.dossier import (
    RecommendationDossier,
    generate_context_recommendation_dossiers,
)
from chain_replay_ml.research_recommendations.feature_affinity import (
    analyze_feature_affinity,
    recommend_features_for_context,
)
from chain_replay_ml.research_recommendations.negative_pruning import (
    ExclusionVerdict,
    audit_experiment_exclusion,
)
from chain_replay_ml.research_recommendations.priority_scoring import (
    ContextPriorityAgendaReport,
    ResearchOpportunity,
    build_context_priority_agenda,
)
from .generator import create_candidate_spec, generate_cold_start_candidates
from .mutator import generate_descendant_mutations
from .types import (
    CandidateEligibility,
    CandidateGenerationBudget,
    CandidateGenerationResult,
    CandidateSpec,
    MutationType,
)


def evaluate_candidate_eligibility(
    data_dir: str,
    candidate: CandidateSpec,
    *,
    schema: dict[str, Any] | None = None,
) -> CandidateSpec:
    """Evaluate candidate eligibility against Phase 4D signature registry and Phase 4E negative pruning."""
    init_analysis_db(data_dir)

    # 1. Check if experiment signature was already evaluated
    existing_record = check_experiment_exists(data_dir, candidate.signature_hash)
    if existing_record is not None:
        candidate.eligibility = CandidateEligibility.ALREADY_EVALUATED
        candidate.exclusion_reasons.append("ALREADY_EVALUATED: Experiment signature already exists in research memory.")
        return candidate

    # 2. Check Phase 4E.4 Negative Pruning
    spec_dict = candidate.to_experiment_spec()
    prune_res = audit_experiment_exclusion(data_dir, spec_dict, schema=schema)

    if prune_res.verdict == ExclusionVerdict.EXCLUDED:
        candidate.eligibility = CandidateEligibility.EXCLUDED
        candidate.exclusion_reasons.extend([r.value for r in prune_res.exclusion_reasons])
        if prune_res.evidence_summary:
            candidate.exclusion_reasons.append(prune_res.evidence_summary)
    elif prune_res.verdict == ExclusionVerdict.CAUTION:
        candidate.eligibility = CandidateEligibility.CAUTION
        candidate.caution_warnings.extend([r.value for r in prune_res.exclusion_reasons])
        if prune_res.evidence_summary:
            candidate.caution_warnings.append(prune_res.evidence_summary)
    else:
        candidate.eligibility = CandidateEligibility.ELIGIBLE

    return candidate


def generate_candidate_batch(
    data_dir: str,
    context_key: str,
    base_features: Sequence[str],
    *,
    seed_candidates: Sequence[CandidateSpec] | None = None,
    budget: CandidateGenerationBudget | None = None,
    campaign_id: str | None = None,
    regime_definition_hash: str = "regime_hash_universal",
    dataset_snapshot_hash: str = "dataset_snapshot_v1",
    schema: dict[str, Any] | None = None,
) -> CandidateGenerationResult:
    """Generate a batch of candidate specifications with eligibility validation and budget enforcement."""
    b = budget or CandidateGenerationBudget()
    raw_candidates: list[CandidateSpec] = []

    # 1. If seed candidates provided, generate descendant mutations; otherwise cold-start
    if seed_candidates and len(seed_candidates) > 0:
        # Query Phase 4E top recommended affinity features and interaction synergy pairs
        try:
            aff_report = analyze_feature_affinity(data_dir, context_key)
            top_aff_feats = [r.feature_name for r in aff_report.univariate_recommendations]
            interaction_pairs = [(r.feature_a, r.feature_b) for r in aff_report.interactions if r.synergy_lift > 0.0]
        except Exception:
            top_aff_feats = []
            interaction_pairs = []

        for parent in seed_candidates:
            if len(raw_candidates) >= b.max_candidates_per_campaign:
                break
            mutations = generate_descendant_mutations(
                parent,
                top_affinity_features=top_aff_feats,
                interaction_pairs=interaction_pairs,
                budget=b,
                campaign_id=campaign_id,
            )
            raw_candidates.extend(mutations)
    else:
        raw_candidates = generate_cold_start_candidates(
            context_key=context_key,
            base_features=base_features,
            regime_definition_hash=regime_definition_hash,
            dataset_snapshot_hash=dataset_snapshot_hash,
            campaign_id=campaign_id,
            budget=b,
        )

    # 2. Validate Eligibility and Budget Limits
    final_candidates: list[CandidateSpec] = []
    eligible_cnt = 0
    caution_cnt = 0
    excluded_cnt = 0
    evaluated_cnt = 0

    for cand in raw_candidates:
        if len(final_candidates) >= b.max_candidates_per_campaign:
            break

        cand_validated = evaluate_candidate_eligibility(data_dir, cand, schema=schema)
        final_candidates.append(cand_validated)

        if cand_validated.eligibility == CandidateEligibility.ELIGIBLE:
            eligible_cnt += 1
        elif cand_validated.eligibility == CandidateEligibility.CAUTION:
            caution_cnt += 1
        elif cand_validated.eligibility == CandidateEligibility.EXCLUDED:
            excluded_cnt += 1
        elif cand_validated.eligibility == CandidateEligibility.ALREADY_EVALUATED:
            evaluated_cnt += 1

    budget_exhausted = len(final_candidates) >= b.max_candidates_per_campaign

    return CandidateGenerationResult(
        context_key=context_key,
        campaign_id=campaign_id,
        candidates=final_candidates,
        total_generated=len(final_candidates),
        eligible_count=eligible_cnt,
        caution_count=caution_cnt,
        excluded_count=excluded_cnt,
        already_evaluated_count=evaluated_cnt,
        budget_exhausted=budget_exhausted,
    )


def generate_candidates_from_priority_agenda(
    data_dir: str,
    context_key: str,
    *,
    base_features: Sequence[str] | None = None,
    agenda: ContextPriorityAgendaReport | list[RecommendationDossier] | None = None,
    budget: CandidateGenerationBudget | None = None,
    campaign_id: str | None = None,
    regime_definition_hash: str = "regime_hash_universal",
    dataset_snapshot_hash: str = "dataset_snapshot_v1",
    schema: dict[str, Any] | None = None,
) -> CandidateGenerationResult:
    """Directly formulate CandidateSpec batches from authoritative Phase 4E Research Priority Agenda."""
    b = budget or CandidateGenerationBudget()
    init_analysis_db(data_dir)

    # 1. Retrieve Phase 4E Priority Opportunities
    items: list[tuple[str, str, float, list[str], str]] = []  # (opp_id, opp_type_str, score, features, target_algo)
    if agenda is None:
        try:
            report = build_context_priority_agenda(data_dir, context_key, schema=schema)
            for opp in (report.eligible_opportunities + report.caution_opportunities):
                items.append((opp.opportunity_id, opp.opportunity_type.value, opp.priority_score, opp.candidate_features, opp.target_algorithm))
        except Exception:
            items = []
    elif isinstance(agenda, ContextPriorityAgendaReport):
        for opp in (agenda.eligible_opportunities + agenda.caution_opportunities):
            items.append((opp.opportunity_id, opp.opportunity_type.value, opp.priority_score, opp.candidate_features, opp.target_algorithm))
    elif isinstance(agenda, list):
        # List of RecommendationDossier
        for d in agenda:
            items.append((d.opportunity_id, d.opportunity_type.value, d.priority_score, d.candidate_features, d.target_algorithm))

    raw_candidates: list[CandidateSpec] = []
    default_base = list(base_features or ["adx_14", "rsi_14", "macd_diff"])

    for opp_id, opp_type_str, score, feats, target_algo in items:
        if len(raw_candidates) >= b.max_candidates_per_campaign:
            break

        cand_features = feats if feats else default_base
        capped_features = list(cand_features)[:b.max_features_per_candidate]

        cand = create_candidate_spec(
            context_key=context_key,
            algorithm=target_algo or "xgboost",
            features=capped_features,
            regime_definition_hash=regime_definition_hash,
            dataset_snapshot_hash=dataset_snapshot_hash,
            mutation_type=MutationType.FEATURE_SUBSET_MUTATION,
            mutation_description=f"Phase 4E Opportunity: {opp_type_str} ({opp_id})",
            campaign_id=campaign_id,
            opportunity_id=opp_id,
            opportunity_type=opp_type_str,
            priority_score=score,
            candidate_id_suffix=f"_{opp_id[-4:]}" if len(opp_id) >= 4 else "_OPP",
        )
        raw_candidates.append(cand)

    # 2. Validate Eligibility and Budget Limits
    final_candidates: list[CandidateSpec] = []
    eligible_cnt = 0
    caution_cnt = 0
    excluded_cnt = 0
    evaluated_cnt = 0

    for cand in raw_candidates:
        if len(final_candidates) >= b.max_candidates_per_campaign:
            break

        cand_validated = evaluate_candidate_eligibility(data_dir, cand, schema=schema)
        final_candidates.append(cand_validated)

        if cand_validated.eligibility == CandidateEligibility.ELIGIBLE:
            eligible_cnt += 1
        elif cand_validated.eligibility == CandidateEligibility.CAUTION:
            caution_cnt += 1
        elif cand_validated.eligibility == CandidateEligibility.EXCLUDED:
            excluded_cnt += 1
        elif cand_validated.eligibility == CandidateEligibility.ALREADY_EVALUATED:
            evaluated_cnt += 1

    budget_exhausted = len(final_candidates) >= b.max_candidates_per_campaign

    return CandidateGenerationResult(
        context_key=context_key,
        campaign_id=campaign_id,
        candidates=final_candidates,
        total_generated=len(final_candidates),
        eligible_count=eligible_cnt,
        caution_count=caution_cnt,
        excluded_count=excluded_cnt,
        already_evaluated_count=evaluated_cnt,
        budget_exhausted=budget_exhausted,
    )

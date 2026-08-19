"""Evidence-guided descendant mutation generator for fine-tuning (Phase 4F.4).

Consumes Phase 4E Feature Affinity, Interaction Synergy, and Priority Dossiers
to generate high-value, pruning-compliant descendant CandidateSpec instances.
"""

from __future__ import annotations

from typing import Any, Sequence

from chain_replay_ml.candidate_generation.generator import create_candidate_spec
from chain_replay_ml.candidate_generation.mutator import (
    generate_descendant_mutations,
    mutate_algorithm,
    mutate_feature_subset,
    mutate_hyperparameters,
)
from chain_replay_ml.candidate_generation.service import evaluate_candidate_eligibility
from chain_replay_ml.candidate_generation.types import (
    CandidateEligibility,
    CandidateGenerationBudget,
    CandidateSpec,
    MutationType,
)
from chain_replay_ml.model_ranking.types import CandidateEvidenceScore
from chain_replay_ml.research_recommendations.feature_affinity import (
    analyze_feature_affinity,
    recommend_features_for_context,
)
from chain_replay_ml.research_recommendations.priority_scoring import build_context_priority_agenda
from .types import FineTuningBudget


def generate_fine_tuning_descendants(
    data_dir: str,
    parent_candidate: CandidateSpec,
    parent_score: CandidateEvidenceScore | None = None,
    *,
    budget: FineTuningBudget | None = None,
    campaign_id: str | None = None,
    schema: dict[str, Any] | None = None,
) -> list[CandidateSpec]:
    """Generate a batch of evidence-guided, pruning-validated descendant candidates for a parent."""
    b = budget or FineTuningBudget()

    # If generation depth ceiling reached, return empty
    gen_num = parent_candidate.lineage.generation_number if parent_candidate.lineage else 0
    if gen_num >= b.max_generations:
        return []

    # 1. Ingest Phase 4E Feature Affinity & Interaction Synergy Evidence
    try:
        aff_report = analyze_feature_affinity(data_dir, parent_candidate.context_key)
        top_affinity = [r.feature_name for r in aff_report.univariate_recommendations]
        interaction_pairs = [(r.feature_a, r.feature_b) for r in aff_report.interactions if r.synergy_lift > 0.0]
    except Exception:
        top_affinity = []
        interaction_pairs = []

    c_budget = CandidateGenerationBudget(
        max_candidates_per_campaign=b.max_candidates_total,
        max_generations=b.max_generations,
        max_descendants_per_parent=b.max_descendants_per_parent,
        max_features_per_candidate=b.max_features_per_candidate,
    )

    raw_descendants = generate_descendant_mutations(
        parent_candidate,
        top_affinity_features=top_affinity,
        interaction_pairs=interaction_pairs,
        budget=c_budget,
        campaign_id=campaign_id,
    )

    # 2. Validate Eligibility and Negative Pruning for all generated descendants
    eligible_descendants: list[CandidateSpec] = []
    for d in raw_descendants:
        if len(eligible_descendants) >= b.max_descendants_per_parent:
            break
        validated = evaluate_candidate_eligibility(data_dir, d, schema=schema)
        # Only schedule non-excluded candidates
        if validated.eligibility != CandidateEligibility.EXCLUDED:
            eligible_descendants.append(validated)

    return eligible_descendants

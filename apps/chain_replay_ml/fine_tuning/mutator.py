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
from .feature_elimination import apply_feature_elimination
from .types import FineTuningBudget


def generate_fine_tuning_descendants(
    data_dir: str,
    parent_candidate: CandidateSpec,
    parent_score: CandidateEvidenceScore | None = None,
    *,
    budget: FineTuningBudget | None = None,
    campaign_id: str | None = None,
    schema: dict[str, Any] | None = None,
    feature_elimination_strategy: str = "NONE",
) -> list[CandidateSpec]:
    """Generate a batch of evidence-guided, pruning-validated descendant candidates for a parent."""
    b = budget or FineTuningBudget()

    # If generation depth ceiling reached, return empty
    gen_num = parent_candidate.lineage.generation_number if parent_candidate.lineage else 0
    if gen_num >= b.max_generations:
        return []

    strat = str(feature_elimination_strategy or "NONE").strip().upper()
    raw_descendants: list[CandidateSpec] = []

    # 1. If an active feature elimination strategy is selected, generate pruned descendant candidates
    if strat not in ("NONE", ""):
        retained, eliminated, desc = apply_feature_elimination(
            data_dir=data_dir,
            context_key=parent_candidate.context_key,
            current_features=parent_candidate.features,
            strategy=strat,
            generation_number=gen_num + 1,
        )
        if eliminated:
            elim_spec = create_candidate_spec(
                context_key=parent_candidate.context_key,
                algorithm=parent_candidate.algorithm,
                features=retained,
                hyperparameters=parent_candidate.hyperparameters,
                walk_forward_config=parent_candidate.walk_forward_config,
                regime_definition_hash=parent_candidate.regime_definition_hash,
                dataset_snapshot_hash=parent_candidate.dataset_snapshot_hash,
                random_seed=parent_candidate.random_seed,
                parent_spec=parent_candidate,
                mutation_type=MutationType.FEATURE_ELIMINATION,
                mutation_description=desc,
                campaign_id=campaign_id or (parent_candidate.lineage.campaign_id if parent_candidate.lineage else None),
                candidate_id_suffix=f"_{strat[:4]}",
                feature_elimination_strategy=strat,
            )
            raw_descendants.append(elim_spec)

    # 2. Ingest Phase 4E Feature Affinity & Interaction Synergy Evidence
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

    other_descendants = generate_descendant_mutations(
        parent_candidate,
        top_affinity_features=top_affinity,
        interaction_pairs=interaction_pairs,
        budget=c_budget,
        campaign_id=campaign_id,
    )
    for od in other_descendants:
        od.feature_elimination_strategy = strat
    raw_descendants.extend(other_descendants)

    # 3. Validate Eligibility and Negative Pruning for all generated descendants
    eligible_descendants: list[CandidateSpec] = []
    for d in raw_descendants:
        if len(eligible_descendants) >= b.max_descendants_per_parent:
            break
        validated = evaluate_candidate_eligibility(data_dir, d, schema=schema)
        # Only schedule non-excluded candidates
        if validated.eligibility != CandidateEligibility.EXCLUDED:
            eligible_descendants.append(validated)

    return eligible_descendants

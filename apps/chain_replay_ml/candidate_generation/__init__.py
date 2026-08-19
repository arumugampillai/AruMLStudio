"""Phase 4F.2: Automated Candidate Generator & Lineage Tracker."""

from .generator import (
    create_candidate_spec,
    generate_cold_start_candidates,
    validate_hyperparameters_for_algorithm,
)
from .lineage import (
    reconstruct_lineage_graph,
    trace_ancestors,
)
from .mutator import (
    generate_descendant_mutations,
    mutate_algorithm,
    mutate_feature_subset,
    mutate_hyperparameters,
)
from .service import (
    evaluate_candidate_eligibility,
    generate_candidate_batch,
    generate_candidates_from_priority_agenda,
)
from .types import (
    CandidateEligibility,
    CandidateGenerationBudget,
    CandidateGenerationResult,
    CandidateLineageRecord,
    CandidateSpec,
    MutationType,
)

__all__ = [
    "CandidateEligibility",
    "CandidateGenerationBudget",
    "CandidateGenerationResult",
    "CandidateLineageRecord",
    "CandidateSpec",
    "MutationType",
    "create_candidate_spec",
    "evaluate_candidate_eligibility",
    "generate_candidate_batch",
    "generate_candidates_from_priority_agenda",
    "generate_cold_start_candidates",
    "generate_descendant_mutations",
    "mutate_algorithm",
    "mutate_feature_subset",
    "mutate_hyperparameters",
    "reconstruct_lineage_graph",
    "trace_ancestors",
    "validate_hyperparameters_for_algorithm",
]

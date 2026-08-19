"""Phase 4F.3: Model + Trading Evidence Ranking Engine."""

from .persistence import (
    init_candidate_rankings_table,
    load_candidate_rankings_for_context,
    persist_candidate_rankings,
)
from .ranker import rank_candidates_in_context
from .scorer import (
    compute_composite_candidate_score,
    compute_trading_evidence_score,
    evaluate_candidate_evidence,
    normalize_mfe_mae_ratio,
    normalize_profit_factor,
    normalize_win_rate,
)
from .types import (
    CandidateEvidenceScore,
    CandidateRankingPolicy,
    ContextRankingReport,
    RANK_POLICY_v1_0,
    RecommendationClass,
)

__all__ = [
    "CandidateEvidenceScore",
    "CandidateRankingPolicy",
    "ContextRankingReport",
    "RANK_POLICY_v1_0",
    "RecommendationClass",
    "compute_composite_candidate_score",
    "compute_trading_evidence_score",
    "evaluate_candidate_evidence",
    "init_candidate_rankings_table",
    "load_candidate_rankings_for_context",
    "normalize_mfe_mae_ratio",
    "normalize_profit_factor",
    "normalize_win_rate",
    "persist_candidate_rankings",
    "rank_candidates_in_context",
]

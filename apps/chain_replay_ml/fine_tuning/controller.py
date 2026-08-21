"""Fine-Tuning & Descendant Mutation Controller (Phase 4F.4).

Coordinates parent selection, descendant mutation proposing, child-vs-parent delta evaluation,
and evolutionary search tree tracking.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Sequence

from chain_replay_ml.candidate_generation.types import CandidateSpec
from chain_replay_ml.model_ranking.types import (
    CandidateEvidenceScore,
    ContextRankingReport,
    RecommendationClass,
)
from .evaluator import evaluate_child_vs_parent
from .mutator import generate_fine_tuning_descendants
from .types import (
    DescendantEvaluationRecord,
    FineTuningBudget,
    FineTuningCampaignResult,
    FineTuningDecision,
)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class FineTuningController:
    """Autonomous mutation and fine-tuning controller."""

    def __init__(self, budget: FineTuningBudget | None = None):
        self.budget = budget or FineTuningBudget()

    def select_promising_parents(
        self,
        ranking_report: ContextRankingReport,
    ) -> list[CandidateEvidenceScore]:
        """Select top-ranked candidates eligible for fine-tuning mutations."""
        eligible_parents: list[CandidateEvidenceScore] = []
        for c in ranking_report.ranked_candidates:
            if c.recommendation_class != RecommendationClass.REJECTED:
                eligible_parents.append(c)
                if len(eligible_parents) >= 3:
                    break
        return eligible_parents

    def propose_fine_tuning_batch(
        self,
        data_dir: str,
        parents: Sequence[CandidateSpec],
        parent_scores: dict[str, CandidateEvidenceScore] | None = None,
        *,
        campaign_id: str | None = None,
        schema: dict[str, Any] | None = None,
        feature_elimination_strategy: str = "NONE",
    ) -> list[CandidateSpec]:
        """Propose a batch of descendant candidates from eligible parents."""
        proposed: list[CandidateSpec] = []
        p_scores = parent_scores or {}

        for p in parents:
            if len(proposed) >= self.budget.max_candidates_total:
                break
            p_score = p_scores.get(p.candidate_id)
            descendants = generate_fine_tuning_descendants(
                data_dir=data_dir,
                parent_candidate=p,
                parent_score=p_score,
                budget=self.budget,
                campaign_id=campaign_id,
                schema=schema,
                feature_elimination_strategy=feature_elimination_strategy,
            )
            proposed.extend(descendants)

        return proposed[:self.budget.max_candidates_total]

    def evaluate_and_record_campaign(
        self,
        context_key: str,
        child_scores: Sequence[CandidateEvidenceScore],
        parent_scores: dict[str, CandidateEvidenceScore],
        *,
        campaign_id: str | None = None,
    ) -> FineTuningCampaignResult:
        """Evaluate all executed child candidates against their parents and compile campaign result."""
        trials: list[DescendantEvaluationRecord] = []
        confirmed_cnt = 0
        neutral_cnt = 0
        regr_cnt = 0
        pruned_cnt = 0
        active_roots: set[str] = set()

        for c in child_scores:
            p_id = c.parent_candidate_id
            if p_id and p_id in parent_scores:
                p_score = parent_scores[p_id]
                rec = evaluate_child_vs_parent(
                    child_score=c,
                    parent_score=p_score,
                    generation_number=int(c.candidate_id.split("_G")[1][0]) if "_G" in c.candidate_id else 1,
                    opportunity_id=c.opportunity_id,
                    budget=self.budget,
                )
                trials.append(rec)

                if rec.decision_verdict == FineTuningDecision.CONFIRMED_MUTATION_LIFT:
                    confirmed_cnt += 1
                    active_roots.add(c.candidate_id)
                elif rec.decision_verdict == FineTuningDecision.NEUTRAL_MUTATION:
                    neutral_cnt += 1
                    active_roots.add(c.candidate_id)
                elif rec.decision_verdict == FineTuningDecision.REGRESSION:
                    regr_cnt += 1
                elif rec.decision_verdict == FineTuningDecision.PRUNED_MUTATION_PATH:
                    pruned_cnt += 1

        # Sort trials by delta composite score descending
        trials.sort(key=lambda t: -t.delta_composite_score)
        best_trial = trials[0] if trials else None

        return FineTuningCampaignResult(
            context_key=context_key,
            campaign_id=campaign_id,
            total_descendants_generated=len(trials),
            confirmed_lifts_count=confirmed_cnt,
            neutral_count=neutral_cnt,
            regression_count=regr_cnt,
            pruned_paths_count=pruned_cnt,
            active_lineage_roots=sorted(list(active_roots)),
            best_descendant=best_trial,
            trial_records=trials,
            generated_at=_utc_now_iso(),
        )

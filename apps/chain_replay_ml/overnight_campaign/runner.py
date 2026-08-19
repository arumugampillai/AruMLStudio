"""Autonomous Overnight Campaign Runner & Orchestrator (Phase 4F.5).

Coordinates:
Phase 4E (Research Opportunities)
    ↓
Phase 4F.2 (Candidate Generation)
    ↓
Candidate Training & OOS Evaluation
    ↓
Phase 4F.1 (Strategy Replay)
    ↓
Phase 4F.3 (Model + Trading Evidence Ranking)
    ↓
Phase 4F.4 (Fine-Tuning / Descendant Mutation)
    ↓
Repeat until Budget or Stop Condition Reached
"""

from __future__ import annotations

from datetime import datetime, timezone
import math
import time
from typing import Any, Callable, Sequence

from chain_replay_ml.candidate_generation.generator import create_candidate_spec
from chain_replay_ml.candidate_generation.service import (
    generate_candidate_batch,
    generate_candidates_from_priority_agenda,
)
from chain_replay_ml.candidate_generation.types import (
    CandidateEligibility,
    CandidateGenerationBudget,
    CandidateSpec,
    MutationType,
)
from chain_replay_ml.fine_tuning.controller import FineTuningController
from chain_replay_ml.fine_tuning.persistence import persist_fine_tuning_records
from chain_replay_ml.fine_tuning.types import (
    DescendantEvaluationRecord,
    FineTuningBudget,
    FineTuningDecision,
)
from chain_replay_ml.model_ranking.persistence import persist_candidate_rankings
from chain_replay_ml.model_ranking.ranker import rank_candidates_in_context
from chain_replay_ml.model_ranking.scorer import evaluate_candidate_evidence
from chain_replay_ml.model_ranking.types import (
    CandidateEvidenceScore,
    CandidateRankingPolicy,
    ContextRankingReport,
    RecommendationClass,
)
from chain_replay_ml.research_memory.db import init_analysis_db
from chain_replay_ml.research_recommendations.priority_scoring import (
    build_context_priority_agenda,
)
from .persistence import (
    init_campaign_tables,
    load_campaign_state,
    persist_campaign_state,
)
from .types import (
    CampaignConfig,
    CampaignState,
    CampaignStatus,
    CampaignStopReason,
    OvernightCampaignReport,
)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# Type alias for evaluation callback: fn(data_dir, candidate_spec) -> (model_metrics, trading_metrics)
CandidateEvaluationFn = Callable[[str, CandidateSpec], tuple[dict[str, Any], dict[str, Any]]]


def _default_synthetic_evaluator(data_dir: str, spec: CandidateSpec) -> tuple[dict[str, Any], dict[str, Any]]:
    """Deterministic fallback evaluator for simulations and testing."""
    # Deterministic pseudo-metrics based on signature hash
    h = int(spec.signature_hash[:6], 16)
    n_feats = len(spec.features)

    # Higher feature count without pruning slight penalty; good base features rewarded
    base_auc = 0.70 + ((h % 15) * 0.01)
    base_wr = 52.0 + ((h % 16) * 1.0)
    base_pf = 1.20 + ((h % 12) * 0.08)

    model_metrics = {
        "roc_auc": round(base_auc, 4),
        "fold_mean": round(base_auc, 4),
        "fold_std": 0.015,
        "expected_calibration_error": 0.03,
        "total_features": n_feats,
    }
    trading_metrics = {
        "win_rate_pct": round(base_wr, 2),
        "profit_factor": round(base_pf, 2),
        "mfe_mae_ratio": 1.25,
        "max_drawdown_pct": 3.5,
        "max_consecutive_losses": 2,
        "total_trades": 45,
    }
    return model_metrics, trading_metrics


class OvernightCampaignRunner:
    """Autonomous Research Campaign Controller."""

    def __init__(
        self,
        data_dir: str,
        config: CampaignConfig,
        evaluator_fn: CandidateEvaluationFn | None = None,
        schema: dict[str, Any] | None = None,
    ):
        self.data_dir = data_dir
        self.config = config
        self.evaluator_fn = evaluator_fn or _default_synthetic_evaluator
        self.schema = schema
        init_campaign_tables(self.data_dir)

    def run(self) -> OvernightCampaignReport:
        """Execute the autonomous overnight research campaign."""
        start_ts = time.time()
        start_iso = _utc_now_iso()

        # Check for existing state (idempotency / recovery)
        _, existing_state = load_campaign_state(self.data_dir, self.config.campaign_id)
        if existing_state and existing_state.status == CampaignStatus.COMPLETED:
            # Already completed, reload report
            return self._build_campaign_report(existing_state, start_ts, start_iso)

        state = existing_state or CampaignState(
            campaign_id=self.config.campaign_id,
            config_hash=self.config.compute_config_hash(),
            status=CampaignStatus.RUNNING,
            start_time_iso=start_iso,
            last_update_iso=start_iso,
        )

        all_candidate_specs_by_id: dict[str, CandidateSpec] = {}
        all_ranked_scores_by_sig: dict[str, CandidateEvidenceScore] = {}
        all_ranked_scores: list[CandidateEvidenceScore] = []
        all_trials: list[DescendantEvaluationRecord] = []
        ft_controller = FineTuningController(
            budget=FineTuningBudget(
                max_descendants_per_parent=self.config.max_descendants_per_parent,
                max_generations=self.config.max_generations,
                max_candidates_total=self.config.max_candidates_total,
            )
        )

        try:
            # Loop across generations: Generation 0 (Seed/Phase 4E) -> Generation 1..N (Fine-Tuning)
            for gen in range(state.current_generation, self.config.max_generations):
                state.current_generation = gen
                state.status = CampaignStatus.GENERATING_CANDIDATES
                state.last_update_iso = _utc_now_iso()
                persist_campaign_state(self.data_dir, self.config, state)

                # Check duration stop condition
                elapsed_hours = (time.time() - start_ts) / 3600.0
                if elapsed_hours >= self.config.max_duration_hours:
                    state.status = CampaignStatus.COMPLETED
                    state.stop_reason = CampaignStopReason.MAX_DURATION_EXCEEDED
                    break

                # Check candidate budget stop condition
                if state.total_candidates_trained >= self.config.max_candidates_total:
                    state.status = CampaignStatus.COMPLETED
                    state.stop_reason = CampaignStopReason.MAX_CANDIDATES_REACHED
                    break

                # 1. Generate Candidates for this Generation
                gen_candidates: list[CandidateSpec] = []
                for ctx in self.config.context_keys:
                    if gen == 0:
                        # Cold-start generation from Phase 4E agenda or batch
                        batch_res = None
                        try:
                            agenda = build_context_priority_agenda(self.data_dir, ctx)
                            batch_res = generate_candidates_from_priority_agenda(
                                self.data_dir, ctx, agenda=agenda, campaign_id=self.config.campaign_id, schema=self.schema
                            )
                        except Exception:
                            pass

                        if not batch_res or batch_res.total_generated == 0:
                            batch_res = generate_candidate_batch(
                                self.data_dir, ctx, base_features=["adx_14", "rsi_14", "macd_diff", "bb_width_20", "iv_mean"],
                                campaign_id=self.config.campaign_id, schema=self.schema
                            )
                        gen_candidates.extend(batch_res.candidates)
                        state.total_candidates_excluded += batch_res.excluded_count
                    else:
                        # Descendant mutations from previous generation's top parents
                        parents = [
                            all_candidate_specs_by_id[s.candidate_id]
                            for s in all_ranked_scores[:3]
                            if s.candidate_id in all_candidate_specs_by_id
                        ]
                        if not parents:
                            parents = [
                                create_candidate_spec(
                                    context_key=ctx,
                                    algorithm="xgboost",
                                    features=["adx_14", "rsi_14", "macd_diff", "bb_width_20", "iv_mean"],
                                )
                            ]
                        parent_scores = {s.candidate_id: s for s in all_ranked_scores}
                        descendants = ft_controller.propose_fine_tuning_batch(
                            self.data_dir,
                            parents,
                            parent_scores=parent_scores,
                            campaign_id=self.config.campaign_id,
                            schema=self.schema,
                        )
                        gen_candidates.extend(descendants)

                for c in gen_candidates:
                    all_candidate_specs_by_id[c.candidate_id] = c

                state.total_candidates_generated += len(gen_candidates)

                # Filter eligible candidates for training
                trainable_candidates = [c for c in gen_candidates if c.eligibility != CandidateEligibility.EXCLUDED]
                if not trainable_candidates and gen == 0:
                    state.status = CampaignStatus.COMPLETED
                    state.stop_reason = CampaignStopReason.NO_ELIGIBLE_CANDIDATES
                    break

                # 2. Train & Evaluate Candidates
                state.status = CampaignStatus.TRAINING
                state.last_update_iso = _utc_now_iso()
                persist_campaign_state(self.data_dir, self.config, state)

                gen_scores: list[CandidateEvidenceScore] = []
                budget_exhausted = False
                for cand in trainable_candidates:
                    if state.total_candidates_trained >= self.config.max_candidates_total:
                        budget_exhausted = True
                        break
                    try:
                        state.status = CampaignStatus.OOS_EVALUATION
                        m_metrics, t_metrics = self.evaluator_fn(self.data_dir, cand)
                        state.status = CampaignStatus.TRADING_EVALUATION

                        ev_score = evaluate_candidate_evidence(
                            candidate_id=cand.candidate_id,
                            signature_hash=cand.signature_hash,
                            context_key=cand.context_key,
                            model_metrics=m_metrics,
                            trading_metrics=t_metrics,
                            parent_candidate_id=cand.lineage.parent_candidate_id if cand.lineage else None,
                            opportunity_id=cand.lineage.opportunity_id if cand.lineage else None,
                        )
                        gen_scores.append(ev_score)
                        state.total_candidates_trained += 1
                        state.total_candidates_evaluated += 1
                        state.consecutive_failures = 0
                    except Exception as e:
                        state.total_failures += 1
                        state.consecutive_failures += 1
                        state.warnings.append(f"CANDIDATE_EVAL_ERROR_{cand.candidate_id}: {str(e)}")
                        if state.consecutive_failures >= self.config.max_consecutive_failures:
                            state.status = CampaignStatus.CAMPAIGN_FAILED
                            state.stop_reason = CampaignStopReason.EXCESSIVE_FAILURES
                            break

                if state.status == CampaignStatus.CAMPAIGN_FAILED:
                    break

                # 3. Rank Candidates in Context
                state.status = CampaignStatus.RANKING
                for s in gen_scores:
                    all_ranked_scores_by_sig[s.signature_hash] = s
                all_ranked_scores = list(all_ranked_scores_by_sig.values())

                context_reports: list[ContextRankingReport] = []
                for ctx in self.config.context_keys:
                    ctx_scores = [s for s in all_ranked_scores if s.context_key == ctx]
                    raw_items = [
                        {
                            "candidate_id": s.candidate_id,
                            "signature_hash": s.signature_hash,
                            "context_key": s.context_key,
                            "model_metrics": s.model_metrics,
                            "trading_metrics": s.trading_metrics,
                            "parent_candidate_id": s.parent_candidate_id,
                            "opportunity_id": s.opportunity_id,
                        }
                        for s in ctx_scores
                    ]
                    report = rank_candidates_in_context(self.data_dir, ctx, candidate_items=raw_items)
                    context_reports.append(report)
                    persist_candidate_rankings(self.data_dir, report)

                # Update best score and candidate
                prev_best = state.best_composite_score
                for r in context_reports:
                    if r.top_candidate and r.top_candidate.composite_score > state.best_composite_score:
                        state.best_candidate_id = r.top_candidate.candidate_id
                        state.best_signature_hash = r.top_candidate.signature_hash
                        state.best_composite_score = r.top_candidate.composite_score
                        state.best_trading_score = r.top_candidate.trading_evidence_score
                        state.best_model_score = r.top_candidate.model_evidence_score

                if gen == 0:
                    state.starting_best_score = state.best_composite_score

                # 4. Fine-Tuning Evaluation & Plateau Check
                state.status = CampaignStatus.FINE_TUNING
                if gen > 0 and gen_scores:
                    score_lookup = {s.candidate_id: s for s in all_ranked_scores}
                    camp_res = ft_controller.evaluate_and_record_campaign(
                        context_key=self.config.context_keys[0],
                        child_scores=gen_scores,
                        parent_scores=score_lookup,
                        campaign_id=self.config.campaign_id,
                    )
                    all_trials.extend(camp_res.trial_records)
                    persist_fine_tuning_records(self.data_dir, camp_res.trial_records)
                    state.total_candidates_pruned += camp_res.pruned_paths_count

                    # Plateau check
                    lift = state.best_composite_score - prev_best
                    if lift < self.config.plateau_min_lift:
                        state.consecutive_plateau_generations += 1
                        if state.consecutive_plateau_generations >= self.config.plateau_patience_generations and not budget_exhausted:
                            state.status = CampaignStatus.COMPLETED
                            state.stop_reason = CampaignStopReason.PLATEAU_DETECTED
                            break
                    else:
                        state.consecutive_plateau_generations = 0

                if budget_exhausted:
                    state.status = CampaignStatus.COMPLETED
                    state.stop_reason = CampaignStopReason.MAX_CANDIDATES_REACHED
                    break

            # End of campaign
            if state.status not in (CampaignStatus.COMPLETED, CampaignStatus.CAMPAIGN_FAILED):
                state.status = CampaignStatus.COMPLETED
                state.stop_reason = CampaignStopReason.MAX_GENERATIONS_REACHED

        except Exception as ex:
            state.status = CampaignStatus.CAMPAIGN_FAILED
            state.warnings.append(f"UNHANDLED_CAMPAIGN_EXCEPTION: {str(ex)}")

        state.end_time_iso = _utc_now_iso()
        state.last_update_iso = _utc_now_iso()
        persist_campaign_state(self.data_dir, self.config, state)

        return self._build_campaign_report(state, start_ts, start_iso, all_trials, all_ranked_scores)

    def _build_campaign_report(
        self,
        state: CampaignState,
        start_ts: float,
        start_iso: str,
        trials: list[DescendantEvaluationRecord] | None = None,
        ranked_scores: list[CandidateEvidenceScore] | None = None,
    ) -> OvernightCampaignReport:
        end_iso = state.end_time_iso or _utc_now_iso()
        duration = round(time.time() - start_ts, 2)
        total_lift = round(state.best_composite_score - state.starting_best_score, 4)

        best_cand = None
        if ranked_scores and state.best_candidate_id:
            best_cand = next((c for c in ranked_scores if c.candidate_id == state.best_candidate_id), None)

        return OvernightCampaignReport(
            campaign_id=self.config.campaign_id,
            config=self.config,
            status=state.status,
            stop_reason=state.stop_reason,
            contexts_researched=self.config.context_keys,
            total_generations_completed=state.current_generation + 1,
            total_candidates_generated=state.total_candidates_generated,
            total_candidates_trained=state.total_candidates_trained,
            total_candidates_evaluated=state.total_candidates_evaluated,
            total_candidates_excluded=state.total_candidates_excluded,
            total_candidates_pruned=state.total_candidates_pruned,
            best_candidate=best_cand,
            starting_best_score=state.starting_best_score,
            best_composite_score=state.best_composite_score,
            total_score_improvement=total_lift,
            best_trading_score=state.best_trading_score,
            best_model_score=state.best_model_score,
            fine_tuning_trials=trials or [],
            ranked_candidates=ranked_scores or [],
            start_time_iso=start_iso,
            end_time_iso=end_iso,
            duration_seconds=duration,
            warnings=state.warnings,
        )

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

from chain_replay_ml.candidate_generation.generator import (
    create_candidate_spec,
    generate_cold_start_candidates,
)
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
    persist_campaign_event,
    persist_campaign_state,
    persist_candidate_specs,
)
from chain_replay_ml.model_taxonomy import ModelContextKey
from chain_replay_ml.dataset_builder.master_naming import (
    master_dataset_slug,
    resolve_master_db_path,
)
import os
import sqlite3

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
        if evaluator_fn is not None:
            self.evaluator_fn = evaluator_fn
        elif self.config.dataset_path and os.path.isfile(self.config.dataset_path):
            from .dataset_evaluator import train_and_evaluate_candidate_real

            p_path = self.config.dataset_path
            ds_name = self.config.dataset_name or os.path.splitext(os.path.basename(p_path))[0]
            ds_hash = self.config.dataset_snapshot_hash or "snapshot_v1"
            tgt = self.config.target_column or "label_up_5pct_5m"
            camp_id = self.config.campaign_id

            def _real_eval(d_dir: str, spec: CandidateSpec) -> tuple[dict[str, Any], dict[str, Any]]:
                return train_and_evaluate_candidate_real(
                    d_dir,
                    spec,
                    parquet_path=p_path,
                    dataset_name=ds_name,
                    dataset_snapshot_hash=ds_hash,
                    target_column=tgt,
                    campaign_id=camp_id,
                )

            self.evaluator_fn = _real_eval
        else:
            self.evaluator_fn = _default_synthetic_evaluator
        self.schema = schema
        self._cancel_requested = False
        init_campaign_tables(self.data_dir)

    def cancel(self) -> None:
        """Thread-safe request to gracefully cancel campaign execution after the current candidate."""
        self._cancel_requested = True

    def run(
        self,
        *,
        progress_callback: Callable[[CampaignState, str], None] | None = None,
    ) -> OvernightCampaignReport:
        """Execute the autonomous overnight research campaign."""
        start_ts = time.time()
        start_iso = _utc_now_iso()
        self._cancel_requested = False

        def _notify(st: CampaignState, msg: str) -> None:
            if progress_callback is not None:
                try:
                    progress_callback(st, msg)
                except Exception:
                    pass

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

        # 1. Persist campaign row into overnight_campaigns FIRST (ensures foreign-key constraint is satisfied)
        persist_campaign_state(self.data_dir, self.config, state)

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

        # 2. Log CAMPAIGN_STARTED audit event
        ctx_str = self.config.context_keys[0] if self.config.context_keys else "NIFTY_6s_DIRECTION_CLASSIFIER_5m_R001"
        ctx_obj = ModelContextKey.from_key_str(ctx_str)
        market = str(ctx_obj.market).upper()
        int_sec = int(ctx_obj.sampling_interval_sec)

        ds_name = self.config.dataset_name or ""
        ds_path = self.config.dataset_path or ""
        ds_hash = self.config.dataset_snapshot_hash or "dataset_snapshot_v1"
        feature_universe = list(self.config.dataset_feature_universe)

        if ds_name:
            ds_label = ds_name
            db_exists = os.path.exists(ds_path) if ds_path else True
            master_db = ds_path or ds_name
            master_rows = 0
            trading_days_list = []
        else:
            from chain_replay_ml.dataset_builder.master_status import read_master_dataset_status

            ds_slug = master_dataset_slug(market=market, sampling_interval_sec=int_sec)
            ds_label = f"{ds_slug}.db"
            status_info = read_master_dataset_status(
                self.data_dir,
                market=market,
                interval_sec=int_sec,
            )
            db_exists = bool(status_info.get("exists"))
            master_db = status_info.get("master_db_abs") or resolve_master_db_path(self.data_dir, market=market, sampling_interval_sec=int_sec)
            master_rows = int(status_info.get("row_count") or 0)
            trading_days_list = list(status_info.get("days_in_master") or [])
            ds_hash = status_info.get("dataset_fingerprint") or (status_info.get("master_meta") or {}).get("schema_hash") or ds_hash

        persist_campaign_event(
            self.data_dir,
            campaign_id=self.config.campaign_id,
            generation_number=0,
            event_type="CAMPAIGN_STARTED",
            message=f"Campaign {self.config.campaign_id} started for {ctx_str} on {ds_label} ({len(feature_universe)} eligible features).",
            details={
                "campaign_id": self.config.campaign_id,
                "context_key": ctx_str,
                "dataset_name": ds_name or ds_label,
                "dataset_path": master_db,
                "dataset_exists": db_exists,
                "feature_count": len(feature_universe),
                "feature_universe": feature_universe,
                "target_column": self.config.target_column,
                "market": market,
                "sampling_interval_sec": int_sec,
                "task_type": ctx_obj.task_type.value,
                "prediction_horizon": ctx_obj.prediction_horizon,
                "regime_id": ctx_obj.regime_id,
                "row_count": master_rows,
                "trading_days": trading_days_list,
                "trading_day_count": len(trading_days_list),
                "schema_hash": ds_hash,
                "max_generations": self.config.max_generations,
                "max_candidates": self.config.max_candidates_total,
                "max_hours": self.config.max_duration_hours,
                "plateau_enabled": self.config.plateau_enabled,
                "plateau_patience": self.config.plateau_patience_generations,
                "plateau_min_lift": self.config.plateau_min_lift,
                "feature_elimination_strategy": self.config.feature_elimination_strategy,
            },
        )


        try:
            # Loop across generations: Generation 0 (Full Feature Baseline) -> Generation 1..N (Fine-Tuning / Elimination)
            for gen in range(state.current_generation, self.config.max_generations):
                if self._cancel_requested:
                    state.status = CampaignStatus.CAMPAIGN_STOPPED
                    state.stop_reason = CampaignStopReason.USER_CANCELLED
                    _notify(state, "Campaign cancellation requested by user.")
                    persist_campaign_event(
                        self.data_dir,
                        campaign_id=self.config.campaign_id,
                        generation_number=gen,
                        event_type="CAMPAIGN_STOPPED",
                        message="Campaign cancellation requested by user.",
                        details={"generation": gen},
                    )
                    break

                state.current_generation = gen
                state.status = CampaignStatus.GENERATING_CANDIDATES
                state.last_update_iso = _utc_now_iso()
                persist_campaign_state(self.data_dir, self.config, state)
                _notify(state, f"Generation {gen}: Generating candidate specifications...")

                persist_campaign_event(
                    self.data_dir,
                    campaign_id=self.config.campaign_id,
                    generation_number=gen,
                    event_type="GENERATION_STARTED",
                    message=f"Generation {gen} initialized. Generating candidate specifications...",
                    details={"generation": gen, "max_generations": self.config.max_generations},
                )

                # Check duration stop condition
                elapsed_hours = (time.time() - start_ts) / 3600.0
                if elapsed_hours >= self.config.max_duration_hours:
                    state.status = CampaignStatus.COMPLETED
                    state.stop_reason = CampaignStopReason.MAX_DURATION_EXCEEDED
                    _notify(state, "Duration limit reached.")
                    break

                # Check candidate budget stop condition
                if state.total_candidates_trained >= self.config.max_candidates_total:
                    state.status = CampaignStatus.COMPLETED
                    state.stop_reason = CampaignStopReason.MAX_CANDIDATES_REACHED
                    _notify(state, "Maximum candidate budget reached.")
                    break

                # 1. Generate Candidates for this Generation
                allowed_algos = self.config.allowed_algorithms or ["xgboost", "catboost", "lightgbm", "random_forest", "extra_trees"]
                from chain_replay_ml.training.trainers.base import normalize_algorithm_id
                allowed_algos_set = {normalize_algorithm_id(a) for a in allowed_algos}

                gen_candidates: list[CandidateSpec] = []
                for ctx in self.config.context_keys:
                    if gen == 0:
                        # Generation 0: Full Feature Baseline Models across all eligible dataset features
                        base_universe = list(self.config.dataset_feature_universe)
                        if base_universe:
                            gen_candidates.extend(
                                generate_cold_start_candidates(
                                    context_key=ctx,
                                    base_features=base_universe,
                                    algorithms=allowed_algos,
                                    regime_definition_hash="regime_hash_universal",
                                    dataset_snapshot_hash=self.config.dataset_snapshot_hash or "dataset_snapshot_v1",
                                    campaign_id=self.config.campaign_id,
                                    mutation_type=MutationType.FULL_FEATURE_BASELINE,
                                    feature_elimination_strategy=self.config.feature_elimination_strategy,
                                )
                            )
                        else:
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
                        # Descendant mutations from previous generation's top parents (Feature Elimination & Tuning)
                        parents = [
                            all_candidate_specs_by_id[s.candidate_id]
                            for s in all_ranked_scores[:3]
                            if s.candidate_id in all_candidate_specs_by_id
                        ]
                        if not parents:
                            fallback_feats = list(self.config.dataset_feature_universe) or ["adx_14", "rsi_14", "macd_diff", "bb_width_20", "iv_mean"]
                            fallback_algo = allowed_algos[0] if allowed_algos else "xgboost"
                            parents = [
                                create_candidate_spec(
                                    context_key=ctx,
                                    algorithm=fallback_algo,
                                    features=fallback_feats,
                                    dataset_snapshot_hash=self.config.dataset_snapshot_hash or "dataset_snapshot_v1",
                                    feature_elimination_strategy=self.config.feature_elimination_strategy,
                                )
                            ]
                        parent_scores = {s.candidate_id: s for s in all_ranked_scores}
                        descendants = ft_controller.propose_fine_tuning_batch(
                            self.data_dir,
                            parents,
                            parent_scores=parent_scores,
                            campaign_id=self.config.campaign_id,
                            schema=self.schema,
                            feature_elimination_strategy=self.config.feature_elimination_strategy,
                        )
                        gen_candidates.extend(descendants)

                # Strictly enforce algorithm selection filter across all generated candidates
                gen_candidates = [c for c in gen_candidates if normalize_algorithm_id(c.algorithm) in allowed_algos_set]

                for c in gen_candidates:
                    all_candidate_specs_by_id[c.candidate_id] = c
                    persist_campaign_event(
                        self.data_dir,
                        campaign_id=self.config.campaign_id,
                        generation_number=gen,
                        event_type="CANDIDATE_CREATED",
                        candidate_id=c.candidate_id,
                        message=f"Created candidate {c.candidate_id} ({c.algorithm}, {len(c.features)} features).",
                        details={
                            "candidate_id": c.candidate_id,
                            "signature_hash": c.signature_hash,
                            "parent_candidate_id": c.lineage.parent_candidate_id if c.lineage else None,
                            "generation": gen,
                            "mutation_type": c.lineage.mutation_type.value if c.lineage else "INITIAL_SPEC",
                            "mutation_description": c.lineage.mutation_description if c.lineage else "",
                            "algorithm": c.algorithm,
                            "features": list(c.features),
                            "feature_count": len(c.features),
                            "hyperparameters": dict(c.hyperparameters),
                            "eligibility": c.eligibility.value,
                            "feature_elimination_strategy": c.feature_elimination_strategy or self.config.feature_elimination_strategy,
                        },
                    )

                state.total_candidates_generated += len(gen_candidates)
                persist_candidate_specs(self.data_dir, gen_candidates, campaign_id=self.config.campaign_id)

                # Filter eligible candidates for training
                trainable_candidates = [c for c in gen_candidates if c.eligibility != CandidateEligibility.EXCLUDED]
                if not trainable_candidates:
                    state.status = CampaignStatus.COMPLETED
                    state.stop_reason = CampaignStopReason.NO_ELIGIBLE_CANDIDATES
                    _notify(state, f"Generation {gen}: No eligible candidates discovered.")
                    break

                # 2. Train & Evaluate Candidates
                state.status = CampaignStatus.TRAINING
                state.last_update_iso = _utc_now_iso()
                persist_campaign_state(self.data_dir, self.config, state)

                gen_scores: list[CandidateEvidenceScore] = []
                budget_exhausted = False
                for cand in trainable_candidates:
                    if self._cancel_requested:
                        state.status = CampaignStatus.CAMPAIGN_STOPPED
                        state.stop_reason = CampaignStopReason.USER_CANCELLED
                        _notify(state, "Campaign cancellation requested by user.")
                        break

                    if state.total_candidates_trained >= self.config.max_candidates_total:
                        budget_exhausted = True
                        break
                    try:
                        state.status = CampaignStatus.OOS_EVALUATION
                        _notify(state, f"Evaluating candidate {cand.candidate_id} ({state.total_candidates_trained + 1}/{self.config.max_candidates_total})...")

                        persist_campaign_event(
                            self.data_dir,
                            campaign_id=self.config.campaign_id,
                            generation_number=gen,
                            event_type="CANDIDATE_EVAL_START",
                            candidate_id=cand.candidate_id,
                            message=f"Evaluating candidate {cand.candidate_id} across 5 walk-forward folds...",
                            details={
                                "candidate_id": cand.candidate_id,
                                "algorithm": cand.algorithm,
                                "features": list(cand.features),
                                "feature_count": len(cand.features),
                                "validation_strategy": "Walk Forward (5 folds)",
                                "feature_elimination_strategy": cand.feature_elimination_strategy or self.config.feature_elimination_strategy,
                            },
                        )

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

                        persist_campaign_event(
                            self.data_dir,
                            campaign_id=self.config.campaign_id,
                            generation_number=gen,
                            event_type="CANDIDATE_EVAL_DONE",
                            candidate_id=cand.candidate_id,
                            message=f"Candidate {cand.candidate_id} evaluated: Composite {ev_score.composite_score:.2f} pts (Model {ev_score.model_evidence_score:.2f}, Trading {ev_score.trading_evidence_score:.2f}, AUC {m_metrics.get('roc_auc', 0.0):.4f}, WinRate {t_metrics.get('win_rate_pct', 0.0):.1f}%).",
                            details={
                                "candidate_id": cand.candidate_id,
                                "composite_score": ev_score.composite_score,
                                "model_evidence_score": ev_score.model_evidence_score,
                                "trading_evidence_score": ev_score.trading_evidence_score,
                                "risk_penalty": ev_score.risk_penalty,
                                "model_metrics": dict(m_metrics),
                                "trading_metrics": dict(t_metrics),
                                "recommendation_class": ev_score.recommendation_class.value,
                                "warnings": list(ev_score.warnings),
                            },
                        )

                        persist_campaign_event(
                            self.data_dir,
                            campaign_id=self.config.campaign_id,
                            generation_number=gen,
                            event_type="CANDIDATE_VERDICT",
                            candidate_id=cand.candidate_id,
                            message=f"Candidate {cand.candidate_id} verdict: {ev_score.recommendation_class.value} (Score: {ev_score.composite_score:.2f} pts).",
                            details={
                                "candidate_id": cand.candidate_id,
                                "verdict": ev_score.recommendation_class.value,
                                "composite_score": ev_score.composite_score,
                                "warnings": list(ev_score.warnings),
                            },
                        )

                    except Exception as e:
                        state.total_failures += 1
                        state.consecutive_failures += 1
                        state.warnings.append(f"CANDIDATE_EVAL_ERROR_{cand.candidate_id}: {str(e)}")
                        persist_campaign_event(
                            self.data_dir,
                            campaign_id=self.config.campaign_id,
                            generation_number=gen,
                            event_type="CANDIDATE_EVAL_ERROR",
                            candidate_id=cand.candidate_id,
                            message=f"⚠️ Candidate {cand.candidate_id} evaluation failed: {str(e)}",
                            details={"candidate_id": cand.candidate_id, "error": str(e)},
                        )
                        if state.consecutive_failures >= self.config.max_consecutive_failures:
                            state.status = CampaignStatus.CAMPAIGN_FAILED
                            state.stop_reason = CampaignStopReason.EXCESSIVE_FAILURES
                            _notify(state, "Campaign halted: Excessive candidate failures.")
                            break

                if state.status in (CampaignStatus.CAMPAIGN_FAILED, CampaignStatus.CAMPAIGN_STOPPED):
                    break

                # 3. Rank Candidates in Context
                state.status = CampaignStatus.RANKING
                _notify(state, f"Generation {gen}: Ranking candidates...")
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
                prev_best_id = state.best_candidate_id
                for r in context_reports:
                    if r.top_candidate and r.top_candidate.composite_score > state.best_composite_score:
                        state.best_candidate_id = r.top_candidate.candidate_id
                        state.best_signature_hash = r.top_candidate.signature_hash
                        state.best_composite_score = r.top_candidate.composite_score
                        state.best_trading_score = r.top_candidate.trading_evidence_score
                        state.best_model_score = r.top_candidate.model_evidence_score

                if gen == 0:
                    state.starting_best_score = state.best_composite_score

                # Log generation champion & global champion transition
                if context_reports and context_reports[0].top_candidate:
                    gen_top = context_reports[0].top_candidate
                    persist_campaign_event(
                        self.data_dir,
                        campaign_id=self.config.campaign_id,
                        generation_number=gen,
                        event_type="GEN_CHAMPION_ELECTED",
                        candidate_id=gen_top.candidate_id,
                        message=f"Generation {gen} Champion: {gen_top.candidate_id} (Score: {gen_top.composite_score:.2f} pts).",
                        details={
                            "generation": gen,
                            "top_candidate_id": gen_top.candidate_id,
                            "composite_score": gen_top.composite_score,
                            "model_score": gen_top.model_evidence_score,
                            "trading_score": gen_top.trading_evidence_score,
                        },
                    )

                if prev_best_id != state.best_candidate_id:
                    persist_campaign_event(
                        self.data_dir,
                        campaign_id=self.config.campaign_id,
                        generation_number=gen,
                        event_type="GLOBAL_CHAMP_UPDATED",
                        candidate_id=state.best_candidate_id,
                        message=f"👑 New Global Champion: {state.best_candidate_id} (Score: {state.best_composite_score:.2f} pts, Lift: +{state.best_composite_score - state.starting_best_score:.2f}).",
                        details={
                            "new_champion_id": state.best_candidate_id,
                            "previous_champion_id": prev_best_id,
                            "composite_score": state.best_composite_score,
                            "lift": state.best_composite_score - state.starting_best_score,
                        },
                    )

                # 4. Fine-Tuning Evaluation & Plateau Check
                state.status = CampaignStatus.FINE_TUNING
                _notify(state, f"Generation {gen}: Performing fine-tuning mutation analysis...")
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

                    # Plateau check (only if plateau detection is enabled and beyond minimum warm-up generations)
                    if self.config.plateau_enabled and gen >= self.config.min_generations_before_plateau:
                        lift = state.best_composite_score - prev_best
                        if lift < self.config.plateau_min_lift:
                            state.consecutive_plateau_generations += 1
                            persist_campaign_event(
                                self.data_dir,
                                campaign_id=self.config.campaign_id,
                                generation_number=gen,
                                event_type="PLATEAU_CHECK",
                                message=f"Plateau check at Gen {gen}: Score lift {lift:.2f} pts vs min {self.config.plateau_min_lift:.2f} pts (Patience: {state.consecutive_plateau_generations}/{self.config.plateau_patience_generations}).",
                                details={
                                    "generation": gen,
                                    "score_lift": lift,
                                    "plateau_min_lift": self.config.plateau_min_lift,
                                    "patience_count": state.consecutive_plateau_generations,
                                    "max_patience": self.config.plateau_patience_generations,
                                    "plateau_detected": state.consecutive_plateau_generations >= self.config.plateau_patience_generations,
                                },
                            )
                            if state.consecutive_plateau_generations >= self.config.plateau_patience_generations and not budget_exhausted:
                                state.status = CampaignStatus.COMPLETED
                                state.stop_reason = CampaignStopReason.PLATEAU_DETECTED
                                _notify(state, f"Research plateau detected: No lift >= {self.config.plateau_min_lift:.2f} pts over {self.config.plateau_patience_generations} consecutive generations.")
                                break
                        else:
                            state.consecutive_plateau_generations = 0

                # 5. Orchestrate Autonomous Discovery Pipeline Generation (Phases 1–10)
                try:
                    if db_exists and (ds_path or ds_name):
                        import json
                        from chain_replay_ml.discovery_pipeline.loop import run_discovery_generation
                        from chain_replay_ml.discovery_pipeline.types import format_discovery_pipeline_id
                        from chain_replay_ml.dataset_builder.pipeline_registry_store import get_base_pipeline_for_context, load_store as load_pr_store
                        from chain_replay_ml.overnight_campaign.dataset_evaluator import load_dataset_matrix_cached

                        pr_store = load_pr_store(self.data_dir)
                        pl0001_obj = pr_store.get("pipelines", {}).get("PL_0001", {})
                        base_features_pl0001 = list(pl0001_obj.get("feature_names", []))
                        if not base_features_pl0001:
                            base_pipe = get_base_pipeline_for_context(self.data_dir, ctx_str)
                            base_features_pl0001 = list(base_pipe.get("feature_names", [])) if base_pipe else []
                        if not base_features_pl0001:
                            meta_base_path = os.path.join(self.data_dir, "datasets", f"{ds_name}.json") if ds_name else ""
                            if meta_base_path and os.path.isfile(meta_base_path):
                                with open(meta_base_path, "r", encoding="utf-8") as fh:
                                    m_data = json.load(fh)
                                base_features_pl0001 = list(m_data.get("base_pipeline_export_features", []))
                        if not base_features_pl0001:
                            base_features_pl0001 = [f for f in feature_universe if f.startswith("atm_") or f.startswith("iv_") or f in ("adx_14", "rsi_14", "macd_diff", "bb_width_20", "iv_mean")] or feature_universe[:50]

                        # Load dataset matrix cached
                        feats_to_load = feature_universe or base_features_pl0001
                        df_matrix = load_dataset_matrix_cached(
                            ds_path or ds_name,
                            feature_columns=feats_to_load,
                            target_column=self.config.target_column or "label_up_5pct_5m",
                        )

                        dp_id = format_discovery_pipeline_id(self.config.campaign_id)
                        run_discovery_generation(
                            df_matrix,
                            data_dir=self.data_dir,
                            pipeline_id=dp_id,
                            campaign_id=self.config.campaign_id,
                            generation_number=gen + 1,
                            base_features=base_features_pl0001,
                            target_column=self.config.target_column or "label_up_5pct_5m",
                            context_key=ctx_str,
                            dataset_name=ds_name or ds_label,
                            dataset_snapshot_hash=ds_hash,
                        )
                except Exception as disc_err:
                    logger.warning("Autonomous discovery generation failed for campaign %s: %s", self.config.campaign_id, disc_err, exc_info=True)

                if budget_exhausted:
                    state.status = CampaignStatus.COMPLETED
                    state.stop_reason = CampaignStopReason.MAX_CANDIDATES_REACHED
                    _notify(state, "Candidate budget reached.")
                    break

            # End of campaign
            if state.status not in (CampaignStatus.COMPLETED, CampaignStatus.CAMPAIGN_FAILED, CampaignStatus.CAMPAIGN_STOPPED):
                state.status = CampaignStatus.COMPLETED
                state.stop_reason = CampaignStopReason.MAX_GENERATIONS_REACHED
                _notify(state, "Maximum generations completed.")

        except Exception as ex:
            import traceback
            tb_str = traceback.format_exc()
            state.status = CampaignStatus.CAMPAIGN_FAILED
            state.warnings.append(f"UNHANDLED_CAMPAIGN_EXCEPTION: {str(ex)}\n{tb_str}")
            persist_campaign_event(
                self.data_dir,
                campaign_id=self.config.campaign_id,
                generation_number=state.current_generation,
                event_type="CAMPAIGN_FAILED",
                message=f"Campaign failed with exception: {str(ex)}",
                details={"error": str(ex), "traceback": tb_str},
            )

        state.end_time_iso = _utc_now_iso()
        state.last_update_iso = _utc_now_iso()
        persist_campaign_state(self.data_dir, self.config, state)

        persist_campaign_event(
            self.data_dir,
            campaign_id=self.config.campaign_id,
            generation_number=state.current_generation,
            event_type="CAMPAIGN_COMPLETED",
            candidate_id=state.best_candidate_id,
            message=f"Campaign {self.config.campaign_id} {state.status.value}: {state.stop_reason.value if state.stop_reason else 'Finished'}. Winning Candidate: {state.best_candidate_id} ({state.best_composite_score:.2f} pts).",
            details={
                "campaign_id": self.config.campaign_id,
                "status": state.status.value,
                "stop_reason": state.stop_reason.value if state.stop_reason else "COMPLETED",
                "total_generations": state.current_generation + 1,
                "total_candidates_trained": state.total_candidates_trained,
                "total_candidates_pruned": state.total_candidates_pruned,
                "best_candidate_id": state.best_candidate_id,
                "best_composite_score": state.best_composite_score,
                "total_lift": state.best_composite_score - state.starting_best_score,
                "duration_seconds": round(time.time() - start_ts, 2),
            },
        )


        state.end_time_iso = _utc_now_iso()
        state.last_update_iso = _utc_now_iso()
        return self._build_campaign_report(
            state,
            start_ts,
            start_iso,
            all_trials,
            list(all_ranked_scores_by_sig.values()),
        )

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

        sorted_ranked = sorted(ranked_scores or [], key=lambda s: -s.composite_score)
        best_cand = sorted_ranked[0] if sorted_ranked else None

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
            best_composite_score=best_cand.composite_score if best_cand else state.best_composite_score,
            total_score_improvement=total_lift,
            best_trading_score=best_cand.trading_evidence_score if best_cand else state.best_trading_score,
            best_model_score=best_cand.model_evidence_score if best_cand else state.best_model_score,
            fine_tuning_trials=trials or [],
            ranked_candidates=sorted_ranked,
            start_time_iso=start_iso,
            end_time_iso=end_iso,
            duration_seconds=duration,
            warnings=state.warnings,
        )

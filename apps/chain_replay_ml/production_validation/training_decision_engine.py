"""Recommendation-to-Training Decision Engine (Phase 3A Core).

Deterministic, context-scoped policy evaluator converting historical feature
validation evidence and intelligence into four canonical decision states:
- TRAIN_CANDIDATE: Approved for Model Builder training candidate set.
- REVIEW: Not training-approved; requires human inspection.
- NEW_UNSEEN: Allowed for initial feature validation/transformation (total_runs == 0), but not training-approved.
- EXCLUDE: Blocked from candidate generation.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from .dataset_context import DatasetContext, resolve_context_or_legacy
from .evidence_store import get_connection as get_evidence_connection
from .recommendation_policy import (
    BasePipelinePolicy,
    ExperimentalLifecyclePolicy,
    FeatureRegistryPolicy,
    RecommendationPolicy,
    ScoringPolicy,
    TrainingDecisionPolicy,
    compute_context_generalization,
    compute_evidence_confidence,
    compute_evidence_score,
    compute_model_consensus,
    compute_recency_staleness,
    compute_score_volatility,
    derive_risk_badges,
    load_recommendation_policy,
)


class TrainingDecisionState:
    """Canonical decision state constants."""

    TRAIN_CANDIDATE = "TRAIN_CANDIDATE"
    REVIEW = "REVIEW"
    NEW_UNSEEN = "NEW_UNSEEN"
    EXCLUDE = "EXCLUDE"


@dataclass(frozen=True)
class TrainingDecisionResult:
    """Structured decision output with explicit 4-boolean eligibility contract and full provenance."""

    feature_name: str
    context_id: str
    feature_source: str  # "registry" | "base_pipeline" | "experimental"
    decision: str  # TRAIN_CANDIDATE | REVIEW | NEW_UNSEEN | EXCLUDE

    # 4-Boolean Semantic Contract
    is_excluded: bool
    is_candidate_generation_allowed: bool
    is_training_candidate: bool
    requires_review: bool

    # Decision Provenance & Explainability
    primary_reason: str  # Highest-precedence rule code (e.g. CONTEXT_BLOCKED, LOW_EVIDENCE)
    reason_precedence_tier: int  # 1=Exclusion, 2=Review, 3=Zero Runs (Unseen), 4=Candidate
    all_triggered_rules: tuple[str, ...] = field(default_factory=tuple)
    failed_checks: tuple[str, ...] = field(default_factory=tuple)
    passed_checks: tuple[str, ...] = field(default_factory=tuple)
    reason_badges: tuple[str, ...] = field(default_factory=tuple)
    explanation_bullets: tuple[str, ...] = field(default_factory=tuple)

    # Optional Lineage & Ranking Context
    pipeline_id: str | None = None
    pipeline_snapshot_id: str | None = None
    priority_rank: int | None = None
    advisory_rank: int | None = None
    operational_priority_score: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature_name": self.feature_name,
            "context_id": self.context_id,
            "feature_source": self.feature_source,
            "decision": self.decision,
            "is_excluded": self.is_excluded,
            "is_candidate_generation_allowed": self.is_candidate_generation_allowed,
            "is_training_candidate": self.is_training_candidate,
            "requires_review": self.requires_review,
            "primary_reason": self.primary_reason,
            "reason_precedence_tier": self.reason_precedence_tier,
            "all_triggered_rules": list(self.all_triggered_rules),
            "failed_checks": list(self.failed_checks),
            "passed_checks": list(self.passed_checks),
            "reason_badges": list(self.reason_badges),
            "explanation_bullets": list(self.explanation_bullets),
            "pipeline_id": self.pipeline_id,
            "pipeline_snapshot_id": self.pipeline_snapshot_id,
            "priority_rank": self.priority_rank,
            "advisory_rank": self.advisory_rank,
            "operational_priority_score": self.operational_priority_score,
        }


def evaluate_training_decision(
    *,
    feature_name: str,
    context_id: str,
    feature_source: str,
    total_runs: int = 0,
    unique_models_count: int = 0,
    evidence_score: float = 0.0,
    lifecycle_status: str = "active",
    context_status: str = "active",
    consecutive_remove_count: int = 0,
    remove_runs: int = 0,
    consecutive_keep_count: int = 0,
    evidence_confidence: float | None = None,
    dominant_recommendation: str | None = None,
    is_consensus_tie: bool = False,
    freshness_label: str | None = None,
    score_volatility: float | None = None,
    generalization_label: str | None = None,
    generalization_score: float | None = None,
    is_promotion_candidate: bool = False,
    implementation_status: str | None = None,
    policy: RecommendationPolicy | None = None,
    pipeline_id: str | None = None,
    pipeline_snapshot_id: str | None = None,
    priority_rank: int | None = None,
    advisory_rank: int | None = None,
    operational_priority_score: float | None = None,
) -> TrainingDecisionResult:
    """Pure deterministic evaluator deriving the 4-state Training Decision.

    Follows strict deterministic precedence:
    - Tier 1: Hard Exclusion (DEPRECATED_FEATURE, E1-E4) [Non-immune experimental / deprecated features]
    - Tier 2: Review Required (R1-R10)
    - Tier 3: Zero Runs / Unseen (NEW_UNSEEN for total_runs == 0)
    - Tier 4: Training Candidate Eligibility (T1-T2)
    """
    if policy is None:
        policy = RecommendationPolicy()

    s_pol = policy.scoring
    base_pol = policy.base_pipeline
    exp_pol = policy.experimental_lifecycle
    td_pol = policy.training_decision

    source = str(feature_source or "experimental").lower().strip()
    is_deprecated = (
        str(lifecycle_status or "").lower().strip() == "deprecated"
        or str(implementation_status or "").lower().strip() == "deprecated"
    )
    is_immune = source in ("registry", "base_pipeline", "feature_registry") and not is_deprecated

    # If confidence was not pre-computed, calculate it
    if evidence_confidence is None:
        evidence_confidence = compute_evidence_confidence(
            total_runs,
            unique_models_count,
            policy=policy,
        )

    # Track check results and explanation bullets
    passed_checks: list[str] = []
    failed_checks: list[str] = []
    bullets: list[str] = []
    badges: list[str] = []
    triggered_rules: list[str] = []

    # -------------------------------------------------------------------------
    # STAGE 1: HARD EXCLUSION EVALUATION (Precedence Tier 1)
    # -------------------------------------------------------------------------
    exclusion_reasons: list[str] = []

    if is_deprecated:
        exclusion_reasons.append("DEPRECATED_FEATURE")
        failed_checks.append("non_deprecated")
        badges.append("[DEPRECATED]")
        bullets.append("✗ Deprecation Gate: Feature has been permanently deprecated in the Feature Registry.")
    elif is_immune:
        passed_checks.append("hard_exclusion_immunity")
        bullets.append(f"✓ Population Immunity: {source} is immune from automated hard exclusion.")
    else:
        # Check E1: Context / Lifecycle Blocked
        if str(context_status).upper() == "BLOCKED" or str(lifecycle_status).lower() == "blocked":
            exclusion_reasons.append("CONTEXT_BLOCKED")
            failed_checks.append("context_clear")
            badges.append("[BLOCKED]")
            bullets.append("✗ Context Gate: BLOCKED (Feature has been blocked in this dataset context).")
        else:
            passed_checks.append("context_clear")

        # Check E2: Consecutive REMOVE streak
        if consecutive_remove_count >= exp_pol.remove_block_consecutive_threshold:
            exclusion_reasons.append("REMOVE_STREAK_EXCEEDED")
            failed_checks.append("remove_streak_clear")
            bullets.append(
                f"✗ Remove Streak: {consecutive_remove_count} consecutive REMOVEs "
                f"(>= threshold of {exp_pol.remove_block_consecutive_threshold})."
            )
        else:
            passed_checks.append("remove_streak_clear")

        # Check E3: Total REMOVE runs
        if remove_runs >= exp_pol.remove_block_total_threshold:
            exclusion_reasons.append("TOTAL_REMOVE_EXCEEDED")
            failed_checks.append("total_remove_clear")
            bullets.append(
                f"✗ Total Removes: {remove_runs} total REMOVE runs "
                f"(>= threshold of {exp_pol.remove_block_total_threshold})."
            )
        else:
            passed_checks.append("total_remove_clear")

        # Check E4: Severe degradation score
        if evidence_score <= base_pol.negative_alert_score_threshold:
            exclusion_reasons.append("SEVERE_DEGRADATION")
            failed_checks.append("non_negative_alert_score")
            badges.append("[DEGRADED]")
            bullets.append(
                f"✗ Evidence Score: {evidence_score:.1f} "
                f"(<= severe degradation threshold of {base_pol.negative_alert_score_threshold:.1f})."
            )
        else:
            passed_checks.append("non_negative_alert_score")

    if exclusion_reasons:
        triggered_rules.extend(exclusion_reasons)
        return TrainingDecisionResult(
            feature_name=feature_name,
            context_id=context_id,
            feature_source=source,
            decision=TrainingDecisionState.EXCLUDE,
            is_excluded=True,
            is_candidate_generation_allowed=False,
            is_training_candidate=False,
            requires_review=False,
            primary_reason=exclusion_reasons[0],
            reason_precedence_tier=1,
            all_triggered_rules=tuple(triggered_rules),
            failed_checks=tuple(failed_checks),
            passed_checks=tuple(passed_checks),
            reason_badges=tuple(dict.fromkeys(badges)),  # Preserve order, deduplicate
            explanation_bullets=tuple(bullets),
            pipeline_id=pipeline_id,
            pipeline_snapshot_id=pipeline_snapshot_id,
            priority_rank=priority_rank,
            advisory_rank=advisory_rank,
            operational_priority_score=operational_priority_score,
        )

    # -------------------------------------------------------------------------
    # STAGE 2: REVIEW CHECKS (Precedence Tier 2)
    # (Evaluated only for features with total_runs >= 1 or immune degraded features)
    # -------------------------------------------------------------------------
    review_reasons: list[str] = []

    if total_runs >= 1:
        # R8: Health alert on Registry / Base pipeline
        if is_immune and (
            str(lifecycle_status).lower() == "alert"
            or evidence_score <= base_pol.negative_alert_score_threshold
        ):
            review_reasons.append("HEALTH_ALERT")
            failed_checks.append("health_alert_clear")
            badges.append("[ALERT]")
            bullets.append(
                f"✗ Health Status: ALERT (Score {evidence_score:.1f} <= {base_pol.negative_alert_score_threshold:.1f})."
            )
        else:
            passed_checks.append("health_alert_clear")

        # R4: Model consensus dominant REMOVE
        if str(dominant_recommendation).upper() == "REMOVE":
            review_reasons.append("MODEL_DISAGREEMENT_REMOVE")
            failed_checks.append("consensus_non_remove")
            bullets.append("✗ Model Consensus: REMOVE (Model consensus dominant recommendation is REMOVE).")
        else:
            passed_checks.append("consensus_non_remove")

        # R2: Consensus Split / Tie
        if is_consensus_tie:
            review_reasons.append("CONSENSUS_SPLIT")
            failed_checks.append("consensus_non_split")
            badges.append("[SPLIT]")
            bullets.append("✗ Model Consensus: SPLIT (50/50 tie across validating models).")
        else:
            passed_checks.append("consensus_non_split")

        # R3: Dominant WATCH
        if str(dominant_recommendation).upper() == "WATCH":
            review_reasons.append("DOMINANT_WATCH")
            failed_checks.append("consensus_non_watch")
            bullets.append("✗ Model Consensus: WATCH (Feature flagged for monitoring).")
        else:
            passed_checks.append("consensus_non_watch")

        # R5: Stale evidence (> 30 days)
        if str(freshness_label).lower() == "stale":
            review_reasons.append("STALE_EVIDENCE")
            failed_checks.append("freshness_clear")
            badges.append("[STALE]")
            bullets.append("✗ Freshness: STALE (> 30 days elapsed since last validation).")
        else:
            passed_checks.append("freshness_clear")

        # R6: High volatility (Only evaluated if total_runs >= 3; lack of data does NOT cause review)
        if total_runs >= 3 and score_volatility is not None:
            if score_volatility >= td_pol.max_volatility_for_candidate:
                review_reasons.append("HIGH_VOLATILITY")
                failed_checks.append("volatility_acceptable")
                badges.append("[UNSTABLE]")
                bullets.append(
                    f"✗ Score Volatility: σ = {score_volatility:.1f} "
                    f"(>= volatile threshold of {td_pol.max_volatility_for_candidate:.1f})."
                )
            else:
                passed_checks.append("volatility_acceptable")
                bullets.append(f"✓ Score Volatility: σ = {score_volatility:.1f} (Stable).")
        else:
            passed_checks.append("volatility_acceptable")
            bullets.append("✓ Score Volatility: N/A (< 3 runs; insufficient runs to measure volatility).")

        # R7: Scale-specific generalization (Only evaluated if K >= 2; lack of data does NOT cause review)
        if generalization_score is not None:
            if generalization_score < td_pol.min_level1_generalization:
                review_reasons.append("SCALE_SPECIFIC_DIVERGENCE")
                failed_checks.append("generalization_acceptable")
                bullets.append(
                    f"✗ Generalization: G = {generalization_score:.2f} "
                    f"(< minimum cross-context threshold of {td_pol.min_level1_generalization:.2f})."
                )
            else:
                passed_checks.append("generalization_acceptable")
                bullets.append(f"✓ Generalization: G = {generalization_score:.2f} (Cross-context consistent).")
        else:
            passed_checks.append("generalization_acceptable")
            bullets.append("✓ Generalization: Single Context (Only 1 sampling interval evaluated).")

        # R9: Score below candidate minimum
        if evidence_score < td_pol.train_candidate_min_score:
            review_reasons.append("SCORE_BELOW_CANDIDATE_MIN")
            failed_checks.append("score_sufficient")
            bullets.append(
                f"✗ Evidence Score: {evidence_score:.1f} "
                f"(< training candidate minimum score of {td_pol.train_candidate_min_score:.1f})."
            )
        else:
            passed_checks.append("score_sufficient")
            bullets.append(
                f"✓ Evidence Score: {evidence_score:.1f} "
                f"(>= minimum of {td_pol.train_candidate_min_score:.1f})."
            )

        # R1: Low evidence confidence (< 0.30)
        if evidence_confidence < td_pol.train_candidate_min_confidence:
            review_reasons.append("LOW_EVIDENCE")
            failed_checks.append("confidence_sufficient")
            bullets.append(
                f"✗ Evidence Confidence: {evidence_confidence * 100:.1f}% "
                f"(< minimum confidence of {td_pol.train_candidate_min_confidence * 100:.1f}%)."
            )
        else:
            passed_checks.append("confidence_sufficient")
            bullets.append(
                f"✓ Evidence Confidence: {evidence_confidence * 100:.1f}% "
                f"(>= minimum of {td_pol.train_candidate_min_confidence * 100:.1f}%)."
            )

        # R10: Negative votes detected (optional policy toggle)
        if td_pol.require_zero_negative_votes and remove_runs > 0:
            review_reasons.append("NEGATIVE_VOTES_DETECTED")
            failed_checks.append("zero_negative_votes")
            bullets.append(f"✗ Negative Votes: {remove_runs} REMOVE votes recorded (Policy requires 0).")
        else:
            passed_checks.append("zero_negative_votes")

    if review_reasons:
        triggered_rules.extend(review_reasons)
        return TrainingDecisionResult(
            feature_name=feature_name,
            context_id=context_id,
            feature_source=source,
            decision=TrainingDecisionState.REVIEW,
            is_excluded=False,
            is_candidate_generation_allowed=True,
            is_training_candidate=False,
            requires_review=True,
            primary_reason=review_reasons[0],
            reason_precedence_tier=2,
            all_triggered_rules=tuple(triggered_rules),
            failed_checks=tuple(failed_checks),
            passed_checks=tuple(passed_checks),
            reason_badges=tuple(dict.fromkeys(badges)),
            explanation_bullets=tuple(bullets),
            pipeline_id=pipeline_id,
            pipeline_snapshot_id=pipeline_snapshot_id,
            priority_rank=priority_rank,
            advisory_rank=advisory_rank,
            operational_priority_score=operational_priority_score,
        )

    # -------------------------------------------------------------------------
    # STAGE 3: ZERO RUNS / UNSEEN (Precedence Tier 3)
    # (Invariant: total_runs == 0 strictly yields NEW_UNSEEN)
    # -------------------------------------------------------------------------
    if total_runs == 0:
        triggered_rules.append("NEW_UNSEEN_CLEAR")
        passed_checks.extend(["context_clear", "unseen_candidate_cleared"])
        bullets.append("✓ Zero Historical Runs (N = 0): Cleared for candidate generation and initial forward validation.")
        return TrainingDecisionResult(
            feature_name=feature_name,
            context_id=context_id,
            feature_source=source,
            decision=TrainingDecisionState.NEW_UNSEEN,
            is_excluded=False,
            is_candidate_generation_allowed=True,
            is_training_candidate=False,
            requires_review=False,
            primary_reason="NEW_UNSEEN_CLEAR",
            reason_precedence_tier=3,
            all_triggered_rules=tuple(triggered_rules),
            failed_checks=tuple(failed_checks),
            passed_checks=tuple(passed_checks),
            reason_badges=tuple(dict.fromkeys(badges)),
            explanation_bullets=tuple(bullets),
            pipeline_id=pipeline_id,
            pipeline_snapshot_id=pipeline_snapshot_id,
            priority_rank=priority_rank,
            advisory_rank=advisory_rank,
            operational_priority_score=operational_priority_score,
        )

    # -------------------------------------------------------------------------
    # STAGE 4: TRAINING CANDIDATE ELIGIBILITY (Precedence Tier 4)
    # -------------------------------------------------------------------------
    if is_promotion_candidate:
        primary_candidate_reason = "PROMOTION_CANDIDATE_QUALIFIED"
        badges.append("[PROMOTION]")
        bullets.append(
            f"★ Promotion Candidate: Consecutive KEEP streak of {consecutive_keep_count} "
            f"across {unique_models_count} unique models with Score {evidence_score:.1f}."
        )
    else:
        primary_candidate_reason = "TRAINING_CANDIDATE_ELIGIBLE"
        bullets.append(
            f"✓ Training Candidate Approved: Evidence score {evidence_score:.1f} "
            f"and confidence {evidence_confidence * 100:.1f}% meet training requirements."
        )

    triggered_rules.append(primary_candidate_reason)

    return TrainingDecisionResult(
        feature_name=feature_name,
        context_id=context_id,
        feature_source=source,
        decision=TrainingDecisionState.TRAIN_CANDIDATE,
        is_excluded=False,
        is_candidate_generation_allowed=True,
        is_training_candidate=True,
        requires_review=False,
        primary_reason=primary_candidate_reason,
        reason_precedence_tier=4,
        all_triggered_rules=tuple(triggered_rules),
        failed_checks=tuple(failed_checks),
        passed_checks=tuple(passed_checks),
        reason_badges=tuple(dict.fromkeys(badges)),
        explanation_bullets=tuple(bullets),
        pipeline_id=pipeline_id,
        pipeline_snapshot_id=pipeline_snapshot_id,
        priority_rank=priority_rank,
        advisory_rank=advisory_rank,
        operational_priority_score=operational_priority_score,
    )


def evaluate_candidate_training_eligibility(
    data_dir_or_conn: str | sqlite3.Connection,
    context_id: str,
    candidate_names: Sequence[str],
    policy: RecommendationPolicy | None = None,
) -> dict[str, TrainingDecisionResult]:
    """Batch evaluator for Auto Candidate Generation and Pre-Training Elimination Gates.

    Queries the SQLite Evidence DB and derives full deterministic TrainingDecisionResult
    for every candidate in `candidate_names`.

    Features with zero historical records in the DB return `NEW_UNSEEN` (unless blocked).
    """
    if not candidate_names:
        return {}

    should_close = False
    if isinstance(data_dir_or_conn, str):
        conn = get_evidence_connection(data_dir_or_conn)
        should_close = True
    else:
        conn = data_dir_or_conn

    cid = str(context_id or "").strip()

    if policy is None and isinstance(data_dir_or_conn, str):
        policy = load_recommendation_policy(data_dir_or_conn, context_id=cid)
    elif policy is None:
        policy = RecommendationPolicy()

    results: dict[str, TrainingDecisionResult] = {}

    try:
        # 1. Fetch feature_context_summary records for requested candidates
        placeholders = ",".join("?" for _ in candidate_names)
        cursor = conn.cursor()
        query = f"""
            SELECT
                feature_name,
                feature_source,
                total_runs,
                unique_models_count,
                keep_runs,
                watch_runs,
                remove_runs,
                consecutive_keep_count,
                consecutive_remove_count,
                evidence_score,
                lifecycle_status,
                last_validated_at
            FROM feature_context_summary
            WHERE context_id = ? AND feature_name IN ({placeholders})
        """
        cursor.execute(query, [cid, *candidate_names])
        db_rows = {row[0]: row for row in cursor.fetchall()}

        # 2. Fetch raw evidence for multi-model consensus & volatility calculation
        evidence_query = f"""
            SELECT
                feature_name,
                model_name,
                recommendation,
                run_timestamp
            FROM recommendation_evidence
            WHERE context_id = ? AND feature_name IN ({placeholders})
            ORDER BY run_timestamp ASC
        """
        cursor.execute(evidence_query, [cid, *candidate_names])
        raw_evidence_by_feature: dict[str, list[dict[str, Any]]] = {}
        for fname, mname, rec, rts in cursor.fetchall():
            raw_evidence_by_feature.setdefault(fname, []).append({
                "model_name": mname,
                "recommendation": rec,
                "run_timestamp": rts,
            })

        for name in candidate_names:
            row = db_rows.get(name)
            if row is None:
                # Brand-new feature with zero historical runs
                results[name] = evaluate_training_decision(
                    feature_name=name,
                    context_id=cid,
                    feature_source="experimental",
                    total_runs=0,
                    policy=policy,
                )
                continue

            (
                fname,
                fsource,
                tot_runs,
                uniq_models,
                k_runs,
                w_runs,
                r_runs,
                c_keep,
                c_remove,
                e_score,
                l_status,
                last_val_at,
            ) = row

            # Intelligence metrics
            evidence_rows = raw_evidence_by_feature.get(fname, [])
            consensus_info = compute_model_consensus(evidence_rows)
            freshness_info = compute_recency_staleness(last_val_at)
            volatility_info = compute_score_volatility(evidence_rows, policy=policy)

            # Promotion candidate check
            is_promo = (
                c_keep >= policy.experimental_lifecycle.promotion_candidate_consecutive_keep
                and uniq_models >= policy.experimental_lifecycle.experimental_promotion_min_unique_models
                and e_score >= policy.experimental_lifecycle.promotion_candidate_min_score
            )

            results[name] = evaluate_training_decision(
                feature_name=fname,
                context_id=cid,
                feature_source=fsource,
                total_runs=tot_runs,
                unique_models_count=uniq_models,
                evidence_score=e_score,
                lifecycle_status=l_status,
                consecutive_remove_count=c_remove,
                remove_runs=r_runs,
                consecutive_keep_count=c_keep,
                dominant_recommendation=consensus_info.get("dominant_recommendation"),
                is_consensus_tie=bool(consensus_info.get("is_consensus_tie", False)),
                freshness_label=freshness_info.get("freshness_label"),
                score_volatility=volatility_info.get("volatility_score"),
                is_promotion_candidate=is_promo,
                policy=policy,
            )
    finally:
        if should_close:
            conn.close()

    return results


def evaluate_population_training_decisions(
    population_rows: Sequence[dict[str, Any]],
    policy: RecommendationPolicy | None = None,
    context_id: str | None = None,
) -> list[dict[str, Any]]:
    """Enriches a sequence of population records (e.g. from recommendation_store) with decision results.

    Appends 'decision_result' (TrainingDecisionResult) and 'training_decision' (str) to each record.
    """
    if not population_rows:
        return []

    enriched_rows: list[dict[str, Any]] = []

    for row in population_rows:
        row_dict = dict(row)
        fname = str(row_dict.get("feature_name") or "")
        cid = str(row_dict.get("context_id") or context_id or "ctx_unknown")
        fsource = str(row_dict.get("feature_source") or "experimental")

        tot_runs = int(row_dict.get("total_runs") or row_dict.get("validation_runs_count") or 0)
        uniq_models = int(row_dict.get("unique_models_count") or 0)
        e_score_raw = row_dict.get("evidence_score") if row_dict.get("evidence_score") is not None else row_dict.get("lineage_evidence_score")
        e_score = float(e_score_raw or 0.0)
        l_status = str(row_dict.get("lifecycle_status") or row_dict.get("health_status") or "active")
        c_status = str(row_dict.get("context_status") or "active")
        c_remove = int(row_dict.get("consecutive_remove_count") or 0)
        r_runs = int(row_dict.get("remove_runs") or 0)
        c_keep = int(row_dict.get("consecutive_keep_count") or 0)

        conf = row_dict.get("evidence_confidence")
        if conf is not None:
            conf = float(conf)

        dom_rec = row_dict.get("dominant_recommendation")
        is_tie = bool(row_dict.get("is_consensus_tie", False))
        fresh_lbl = row_dict.get("freshness_label")
        vol_score = row_dict.get("score_volatility")
        if vol_score is not None:
            vol_score = float(vol_score)
        gen_lbl = row_dict.get("generalization_label")
        gen_score = row_dict.get("generalization_score")
        if gen_score is not None:
            gen_score = float(gen_score)

        is_promo = bool(
            row_dict.get("is_promotion_candidate")
            or str(row_dict.get("lifecycle_status") or "").upper() == "PROMOTION_CANDIDATE"
            or str(row_dict.get("lineage_status") or "").upper() == "PROMOTION_CANDIDATE"
        )

        p_rank = row_dict.get("priority_rank")
        if p_rank is not None:
            p_rank = int(p_rank)
        a_rank = row_dict.get("advisory_rank")
        if a_rank is not None:
            a_rank = int(a_rank)
        op_score = row_dict.get("operational_priority_score")
        if op_score is not None:
            op_score = float(op_score)

        decision_res = evaluate_training_decision(
            feature_name=fname,
            context_id=cid,
            feature_source=fsource,
            total_runs=tot_runs,
            unique_models_count=uniq_models,
            evidence_score=e_score,
            lifecycle_status=l_status,
            context_status=c_status,
            consecutive_remove_count=c_remove,
            remove_runs=r_runs,
            consecutive_keep_count=c_keep,
            evidence_confidence=conf,
            dominant_recommendation=dom_rec,
            is_consensus_tie=is_tie,
            freshness_label=fresh_lbl,
            score_volatility=vol_score,
            generalization_label=gen_lbl,
            generalization_score=gen_score,
            is_promotion_candidate=is_promo,
            policy=policy,
            pipeline_id=row_dict.get("pipeline_id"),
            pipeline_snapshot_id=row_dict.get("pipeline_snapshot_id"),
            priority_rank=p_rank,
            advisory_rank=a_rank,
            operational_priority_score=op_score,
        )

        row_dict["decision_result"] = decision_res
        row_dict["training_decision"] = decision_res.decision
        row_dict["is_training_candidate"] = decision_res.is_training_candidate
        row_dict["requires_review"] = decision_res.requires_review
        row_dict["is_excluded"] = decision_res.is_excluded
        row_dict["is_candidate_generation_allowed"] = decision_res.is_candidate_generation_allowed
        enriched_rows.append(row_dict)

    return enriched_rows

"""Phase 3B: Model Builder Handoff & Training Candidate Selection Bridge.

Constructs context-isolated Model Builder preset bundles from Phase 3A decisions,
ensuring strict user approval boundaries, full decision provenance, and zero automated training.
"""

from __future__ import annotations

import os
import time
from datetime import datetime, timezone
from typing import Any, Sequence

from .dataset_context import DatasetContext, resolve_context_or_legacy
from .evidence_store import get_connection
from .recommendation_policy import RecommendationPolicy, load_recommendation_policy
from .recommendation_store import get_population_recommendations
from .training_decision_engine import (
    TrainingDecisionResult,
    TrainingDecisionState,
    evaluate_population_training_decisions,
)

DECISION_ENGINE_VERSION = "3B.1"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _candidate_sort_key(item: dict[str, Any]) -> tuple[int, int, float, int, str]:
    """Deterministic ordering for candidate ranking:

    1. Promotion candidate qualified (0 for promo, 1 for regular)
    2. priority_rank (lower is higher priority; None defaults to 999999)
    3. operational_priority_score (negated for descending sort; None defaults to 0.0)
    4. advisory_rank (lower is higher priority; None defaults to 999999)
    5. feature_name (alphabetical ascending)
    """
    res: TrainingDecisionResult | None = item.get("decision_result")
    is_promo = False
    p_rank = 999999
    op_score = 0.0
    a_rank = 999999
    fname = str(item.get("feature_name") or "")

    if res:
        is_promo = res.primary_reason == "PROMOTION_CANDIDATE_QUALIFIED" or "[PROMOTION]" in res.reason_badges
        if res.priority_rank is not None:
            p_rank = int(res.priority_rank)
        if res.operational_priority_score is not None:
            op_score = float(res.operational_priority_score)
        if res.advisory_rank is not None:
            a_rank = int(res.advisory_rank)
    else:
        if item.get("priority_rank") is not None:
            p_rank = int(item["priority_rank"])
        if item.get("operational_priority_score") is not None:
            op_score = float(item["operational_priority_score"])
        if item.get("advisory_rank") is not None:
            a_rank = int(item["advisory_rank"])

    promo_tier = 0 if is_promo else 1
    return (promo_tier, p_rank, -op_score, a_rank, fname)


def build_model_builder_training_bundle(
    data_dir: str,
    *,
    context_id: str | None = None,
    context: DatasetContext | None = None,
    policy: RecommendationPolicy | None = None,
    selected_features: Sequence[str] | None = None,
    include_review_features: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Build a context-scoped Model Builder training preset bundle from Phase 3A decisions.

    - Evaluates all features in context across Registry, Base Pipeline, and Experimental populations.
    - Categorizes into TRAIN_CANDIDATE, REVIEW, NEW_UNSEEN, and EXCLUDE.
    - Filters and orders the approved candidate subset.
    - Generates full structured decision provenance.
    - Does NOT mutate any database rows or start model training.
    """
    root = str(data_dir or "").strip()
    cid = str(context_id or (context.context_id if context else "")).strip() or None

    if policy is None:
        policy = load_recommendation_policy(root, context_id=cid)

    # 1. Fetch population rows across all 3 populations
    all_raw_rows: list[dict[str, Any]] = []
    for pop in ("registry", "base_pipeline", "experimental"):
        pop_rows = get_population_recommendations(root, population=pop, context_id=cid)
        for r in pop_rows:
            r["_pop_source"] = pop
            all_raw_rows.append(r)

    # 2. Enrich with Phase 3A Decisions
    evaluated_rows = evaluate_population_training_decisions(all_raw_rows, policy=policy, context_id=cid)

    # Deduplicate by feature_name (keeping highest precedence population / evidence)
    features_by_name: dict[str, dict[str, Any]] = {}
    for r in evaluated_rows:
        fn = str(r.get("feature_name") or "")
        if not fn:
            continue
        if fn not in features_by_name:
            features_by_name[fn] = r
        else:
            # If feature exists in multiple populations (e.g. experimental promoted to base),
            # preserve the one with higher tier decision
            cur = features_by_name[fn]
            cur_dec = cur["training_decision"]
            new_dec = r["training_decision"]
            dec_rank = {
                TrainingDecisionState.TRAIN_CANDIDATE: 4,
                TrainingDecisionState.REVIEW: 3,
                TrainingDecisionState.NEW_UNSEEN: 2,
                TrainingDecisionState.EXCLUDE: 1,
            }
            if dec_rank.get(new_dec, 0) > dec_rank.get(cur_dec, 0):
                features_by_name[fn] = r

    unique_evaluated = list(features_by_name.values())

    # 3. Categorize candidates
    train_candidates: list[dict[str, Any]] = []
    review_items: list[dict[str, Any]] = []
    unseen_items: list[dict[str, Any]] = []
    exclude_items: list[dict[str, Any]] = []

    for item in unique_evaluated:
        dec = item["training_decision"]
        if dec == TrainingDecisionState.TRAIN_CANDIDATE:
            train_candidates.append(item)
        elif dec == TrainingDecisionState.REVIEW:
            review_items.append(item)
        elif dec == TrainingDecisionState.NEW_UNSEEN:
            unseen_items.append(item)
        elif dec == TrainingDecisionState.EXCLUDE:
            exclude_items.append(item)

    # Sort train candidates deterministically
    train_candidates.sort(key=_candidate_sort_key)

    # 4. Resolve final selected features
    default_candidate_names = [str(item.get("feature_name") or "") for item in train_candidates]

    if selected_features is not None:
        # User customized selection from dialog
        allowed_train_set = set(default_candidate_names)
        # Only allow user-selected features that are genuine TRAIN_CANDIDATEs (or explicitly permitted review items)
        permitted_review_set = set(include_review_features or [])
        final_selected = [
            f for f in selected_features
            if f in allowed_train_set or f in permitted_review_set
        ]
        user_deselected_count = max(0, len(default_candidate_names) - len([f for f in final_selected if f in allowed_train_set]))
    else:
        final_selected = list(default_candidate_names)
        user_deselected_count = 0

    # Ensure no EXCLUDE features can ever be selected
    exclude_set = {str(item.get("feature_name") or "") for item in exclude_items}
    final_selected = [f for f in final_selected if f not in exclude_set]

    # 5. Extract Context Metadata
    market = "UNKNOWN"
    interval_sec = 0
    sliding_window = "standard"
    feature_project_id = "all"

    if context:
        market = context.market
        interval_sec = context.sampling_interval_sec
        sliding_window = context.sliding_window
        feature_project_id = context.feature_project_id
    elif cid:
        try:
            conn = get_connection(root)
            cur = conn.cursor()
            cur.execute(
                "SELECT market, sampling_interval_sec, sliding_window, feature_project_id FROM dataset_contexts WHERE context_id = ?",
                (cid,),
            )
            c_row = cur.fetchone()
            conn.close()
            if c_row:
                market = c_row[0]
                interval_sec = c_row[1]
                sliding_window = c_row[2]
                feature_project_id = c_row[3]
        except Exception:
            pass

    # 6. Build Provenance Map for all features
    provenance_map: dict[str, dict[str, Any]] = {}
    promo_count = 0
    for item in unique_evaluated:
        fn = str(item.get("feature_name") or "")
        res: TrainingDecisionResult = item["decision_result"]
        is_promo = res.primary_reason == "PROMOTION_CANDIDATE_QUALIFIED" or "[PROMOTION]" in res.reason_badges
        if is_promo and res.decision == TrainingDecisionState.TRAIN_CANDIDATE:
            promo_count += 1

        provenance_map[fn] = {
            "feature_source": res.feature_source,
            "decision": res.decision,
            "primary_reason": res.primary_reason,
            "reason_badges": list(res.reason_badges),
            "evidence_score": float(item.get("evidence_score") or item.get("lineage_evidence_score") or 0.0),
            "evidence_confidence": float(item.get("evidence_confidence") or 0.0),
            "dominant_recommendation": item.get("dominant_recommendation"),
            "freshness_label": item.get("freshness_label"),
            "score_volatility": item.get("score_volatility") or item.get("volatility_score"),
            "generalization_score": item.get("generalization_score"),
            "priority_rank": res.priority_rank,
            "advisory_rank": res.advisory_rank,
            "operational_priority_score": res.operational_priority_score,
            "all_triggered_rules": list(res.all_triggered_rules),
            "failed_checks": list(res.failed_checks),
            "passed_checks": list(res.passed_checks),
        }

    bundle: dict[str, Any] = {
        "features": final_selected,
        "dataset": None,
        "source_model": f"RecommendationDecisionEngine:{cid or 'global'}",
        "at": int(time.time() * 1000),
        "recommendation_decision_bundle": {
            "context_id": cid,
            "market": market,
            "sampling_interval_sec": interval_sec,
            "sliding_window": sliding_window,
            "feature_project_id": feature_project_id,
            "policy_id": policy.policy_id,
            "policy_version": policy.policy_version,
            "decision_engine_version": DECISION_ENGINE_VERSION,
            "generated_at_utc": _utc_now_iso(),
            "total_evaluated_count": len(unique_evaluated),
            "selected_candidates_count": len(final_selected),
            "eligible_candidates_count": len(train_candidates),
            "review_count": len(review_items),
            "excluded_count": len(exclude_items),
            "unseen_count": len(unseen_items),
            "selection_summary": {
                "promotion_qualified_count": promo_count,
                "standard_candidate_count": max(0, len(train_candidates) - promo_count),
                "user_deselected_count": user_deselected_count,
            },
            "feature_provenance": provenance_map,
        },
    }

    return bundle


def export_training_candidates_preset(
    chart_dir: str,
    data_dir: str,
    *,
    context_id: str | None = None,
    context: DatasetContext | None = None,
    policy: RecommendationPolicy | None = None,
    selected_features: Sequence[str] | None = None,
    include_review_features: Sequence[str] | None = None,
    dataset_name: str | None = None,
) -> dict[str, Any]:
    """Export training candidates directly to Model Builder feature preset storage.

    Calls `save_feature_preset()` under `chart_dir/data/ml_model_builder_feature_preset_tk.json`.
    Does NOT mutate SQLite evidence tables and does NOT start training.
    """
    from master_dataset_tk.model_builder.feature_preset import save_feature_preset

    bundle = build_model_builder_training_bundle(
        data_dir=data_dir,
        context_id=context_id,
        context=context,
        policy=policy,
        selected_features=selected_features,
        include_review_features=include_review_features,
    )

    if dataset_name:
        bundle["dataset"] = str(dataset_name).strip()

    preset_doc = save_feature_preset(
        chart_dir,
        features=bundle["features"],
        dataset=bundle.get("dataset"),
        source_model=bundle.get("source_model"),
        recommendation_decision_bundle=bundle.get("recommendation_decision_bundle"),
    )

    return preset_doc

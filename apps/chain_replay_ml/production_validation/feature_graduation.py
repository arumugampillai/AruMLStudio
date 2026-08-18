"""Phase 3D.1 — Evidence Dossier Compiler & Qualification Verifier.

Pure read-only auditing and dossier compilation for Feature Registry Graduation.
Does NOT modify Evidence DB, SQLite schema, feature files, or registry stores.
"""

from __future__ import annotations

import json
import os
import sqlite3
import tempfile
from datetime import datetime, timezone
from typing import Any

from ..dataset_builder.feature_registry_store import (
    default_owner_for_group,
    load_store,
    store_path,
    _next_free_feature_id,
)
from ..dataset_builder.pipeline_registry_store import (
    ensure_default_existing_pipeline,
    is_base_pipeline_record,
    load_store as load_pipeline_store,
    store_path as pipeline_store_path,
)
from .dataset_context import DatasetContext, LEGACY_UNKNOWN_CONTEXT_ID
from .evidence_store import evidence_db_path
from .recommendation_policy import (
    RecommendationPolicy,
    compute_context_generalization,
    compute_evidence_confidence,
    compute_model_consensus,
    compute_recency_staleness,
    compute_score_volatility,
    load_recommendation_policy,
)
from .training_decision_engine import (
    TrainingDecisionState,
    evaluate_training_decision,
)


def _get_read_only_connection(data_dir: str) -> sqlite3.Connection | None:
    path = evidence_db_path(data_dir)
    if not os.path.isfile(path):
        return None
    try:
        uri = f"file:{os.path.abspath(path)}?mode=ro"
        conn = sqlite3.connect(uri, uri=True, timeout=10.0)
        conn.row_factory = sqlite3.Row
        return conn
    except Exception:
        return None


def evaluate_graduation_prerequisites(
    data_dir: str,
    feature_name: str,
    context_id: str | None = None,
    policy: RecommendationPolicy | None = None,
    precompiled_dossier: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Deterministically evaluates whether an experimental feature meets the 7 graduation criteria.

    Classifies status as:
    - NOT_READY: Any core qualification check failed.
    - CONTEXT_SCOPED_READY: All 7 checks passed, but K == 1 or G < 0.50.
    - UNIVERSAL_READY: All 7 checks passed, and K >= 2 with G >= 0.50.

    Pure read-only; never mutates databases or JSON files.
    """
    dossier = precompiled_dossier or compile_feature_evidence_dossier(
        data_dir, feature_name, context_id=context_id, policy=policy
    )

    checks: dict[str, dict[str, Any]] = {}
    failed_checks: list[str] = []
    passed_checks: list[str] = []

    # 1. Consecutive KEEP Count >= 3
    c_keep = int(dossier.get("consecutive_keep_count") or 0)
    req_keep = 3
    pass_keep = c_keep >= req_keep
    checks["consecutive_keep_streak"] = {
        "passed": pass_keep,
        "actual": c_keep,
        "required": f">= {req_keep}",
        "description": f"Consecutive KEEP streak (Actual: {c_keep}, Required: >= {req_keep})",
    }
    if pass_keep:
        passed_checks.append("consecutive_keep_streak")
    else:
        failed_checks.append("consecutive_keep_streak")

    # 2. Unique Model Architectures >= 3
    u_models = int(dossier.get("unique_model_count") or 0)
    req_models = 3
    pass_models = u_models >= req_models
    checks["unique_models_count"] = {
        "passed": pass_models,
        "actual": u_models,
        "required": f">= {req_models}",
        "description": f"Unique model architectures tested (Actual: {u_models}, Required: >= {req_models})",
    }
    if pass_models:
        passed_checks.append("unique_models_count")
    else:
        failed_checks.append("unique_models_count")

    # 3. Lineage Evidence Score >= 75.0
    ev_score = float(dossier.get("lineage_evidence_score") or 0.0)
    req_score = 75.0
    pass_score = ev_score >= req_score
    checks["evidence_score"] = {
        "passed": pass_score,
        "actual": ev_score,
        "required": f">= {req_score}",
        "description": f"Cumulative evidence score (Actual: {ev_score:.1f}, Required: >= {req_score})",
    }
    if pass_score:
        passed_checks.append("evidence_score")
    else:
        failed_checks.append("evidence_score")

    # 4. Evidence Confidence >= 0.70
    ev_conf = float(dossier.get("evidence_confidence") or 0.0)
    req_conf = 0.70
    pass_conf = ev_conf >= req_conf
    checks["evidence_confidence"] = {
        "passed": pass_conf,
        "actual": ev_conf,
        "required": f">= {req_conf}",
        "description": f"Evidence volume confidence saturation (Actual: {ev_conf:.2f}, Required: >= {req_conf})",
    }
    if pass_conf:
        passed_checks.append("evidence_confidence")
    else:
        failed_checks.append("evidence_confidence")

    # 5. Score Volatility <= 25.0 (and available, i.e. N >= 3)
    vol = dossier.get("score_volatility")
    req_vol = 25.0
    tot_runs = int(dossier.get("total_validation_runs") or 0)
    if tot_runs < 3 or vol is None:
        pass_vol = False
        vol_desc = "Score volatility unavailable (Insufficient data: N < 3 runs)"
    else:
        pass_vol = float(vol) <= req_vol
        vol_desc = f"Temporal score volatility standard deviation (Actual: σ={float(vol):.1f}, Required: <= {req_vol})"

    checks["score_volatility"] = {
        "passed": pass_vol,
        "actual": vol,
        "required": f"<= {req_vol} (N >= 3)",
        "description": vol_desc,
    }
    if pass_vol:
        passed_checks.append("score_volatility")
    else:
        failed_checks.append("score_volatility")

    # 6. No Active Extreme Health Alert
    health_status = str(dossier.get("health_status") or "HEALTHY").upper()
    has_health_alert = bool(dossier.get("has_health_alert", False))
    pass_health = (health_status not in ("BLOCKED", "EXTREME_DROP", "DEGRADED")) and not has_health_alert
    checks["health_integrity"] = {
        "passed": pass_health,
        "actual": health_status,
        "required": "No active extreme health degradation alerts",
        "description": f"Health status check (Actual: {health_status}, Passed: {pass_health})",
    }
    if pass_health:
        passed_checks.append("health_integrity")
    else:
        failed_checks.append("health_integrity")

    # 7. Phase 3A Promotion Candidate Qualified
    is_promo_qual = bool(dossier.get("is_phase_3a_promotion_qualified", False))
    checks["phase_3a_promotion_qualified"] = {
        "passed": is_promo_qual,
        "actual": is_promo_qual,
        "required": "Phase 3A PROMOTION_CANDIDATE_QUALIFIED must be True",
        "description": f"Phase 3A Promotion Candidate standing (Actual: {is_promo_qual})",
    }
    if is_promo_qual:
        passed_checks.append("phase_3a_promotion_qualified")
    else:
        failed_checks.append("phase_3a_promotion_qualified")

    all_core_passed = len(failed_checks) == 0

    # Generalization Analysis (Universal vs Context-Scoped)
    k_contexts = int(dossier.get("comparable_context_count") or 1)
    gen_score = dossier.get("generalization_score")
    gen_val = float(gen_score) if gen_score is not None else None

    is_universal_ready = False
    is_context_scoped_ready = False
    is_base_pipeline_eligible = False
    allowed_contexts: list[str] = []
    scope_classification = "NOT_READY"

    explanations: list[str] = []

    cid = dossier.get("context_id") or LEGACY_UNKNOWN_CONTEXT_ID

    if not all_core_passed:
        scope_classification = "NOT_READY"
        explanations.append(f"Graduation not approved: {len(failed_checks)} prerequisite check(s) failed ({', '.join(failed_checks)}).")
    else:
        # All 7 core checks passed! Determine Scope:
        if k_contexts >= 2 and gen_val is not None and gen_val >= 0.50:
            scope_classification = "UNIVERSAL_READY"
            is_universal_ready = True
            is_context_scoped_ready = True
            is_base_pipeline_eligible = True
            allowed_contexts = ["ALL"]
            explanations.append(
                f"Universal Graduation Approved: Qualified across {k_contexts} contexts with strong generalization (G={gen_val:.2f} >= 0.50). Eligible for Base Pipeline promotion review."
            )
        else:
            scope_classification = "CONTEXT_SCOPED_READY"
            is_universal_ready = False
            is_context_scoped_ready = True
            is_base_pipeline_eligible = False
            allowed_contexts = [cid]
            if k_contexts < 2:
                explanations.append(
                    f"Context-Scoped Graduation Approved: Qualified for context '{cid}' only (K=1, cross-context generalization unavailable). Blocked from Base Pipeline."
                )
            else:
                explanations.append(
                    f"Context-Scoped Graduation Approved: Qualified for context '{cid}' (Scale-specific behavior with G={gen_val:.2f} < 0.50). Blocked from Universal Base Pipeline."
                )

    return {
        "feature_name": feature_name,
        "context_id": cid,
        "status": scope_classification,
        "is_universal_ready": is_universal_ready,
        "is_context_scoped_ready": is_context_scoped_ready,
        "is_base_pipeline_eligible": is_base_pipeline_eligible,
        "allowed_contexts": allowed_contexts,
        "checks": checks,
        "passed_checks": passed_checks,
        "failed_checks": failed_checks,
        "total_checks_count": len(checks),
        "passed_checks_count": len(passed_checks),
        "failed_checks_count": len(failed_checks),
        "comparable_context_count": k_contexts,
        "generalization_score": gen_val,
        "explanations": explanations,
    }


def compile_feature_evidence_dossier(
    data_dir: str,
    feature_name: str,
    context_id: str | None = None,
    policy: RecommendationPolicy | None = None,
) -> dict[str, Any]:
    """Compiles the complete, multi-model Evidence Dossier for a feature.

    Gathers metrics across:
    - Lineage summary (runs, keeps, removes, models, lineage score)
    - Context summary & metadata (market, interval, window, project)
    - Evidence intelligence (confidence saturation, consensus, freshness)
    - Stability & Volatility (trajectory, sigma_s, direction flips)
    - Cross-Context Generalization (K contexts, agreement ratio, G index)
    - Phase 3A decision standing

    Pure read-only. Returns comprehensive dossier dictionary.
    """
    pol = policy or load_recommendation_policy(data_dir, context_id=context_id)
    conn = _get_read_only_connection(data_dir)

    dossier: dict[str, Any] = {
        "feature_name": feature_name,
        "feature_source": "experimental",
        "context_id": context_id or LEGACY_UNKNOWN_CONTEXT_ID,
        "market": "UNKNOWN",
        "sampling_interval_sec": 0,
        "sliding_window": "standard",
        "feature_project_id": "all",
        "total_validation_runs": 0,
        "unique_model_count": 0,
        "consecutive_keep_count": 0,
        "consecutive_remove_count": 0,
        "keep_runs": 0,
        "watch_runs": 0,
        "remove_runs": 0,
        "lineage_evidence_score": 0.0,
        "lifecycle_status": "active",
        "evidence_confidence": 0.0,
        "model_consensus": {},
        "dominant_recommendation": None,
        "freshness": {},
        "score_volatility": None,
        "score_range": None,
        "direction_flips": None,
        "stability_label": "Insufficient Data",
        "generalization_score": None,
        "comparable_context_count": 1,
        "is_phase_3a_promotion_qualified": False,
        "phase_3a_decision": None,
        "health_status": "HEALTHY",
        "has_health_alert": False,
        "raw_evidence_runs_count": 0,
        "validation_history": [],
    }

    if conn is None:
        # Evaluate prerequisites with empty dossier
        dossier["prerequisites_evaluation"] = evaluate_graduation_prerequisites(
            data_dir, feature_name, context_id=context_id, precompiled_dossier=dossier
        )
        return dossier

    try:
        cur = conn.cursor()

        # 1. Resolve Context and Context Metadata
        target_cid = context_id
        if not target_cid:
            # Find the context with the most runs for this feature
            cur.execute(
                """
                SELECT context_id, total_runs FROM feature_context_summary
                WHERE feature_name = ?
                ORDER BY total_runs DESC LIMIT 1;
                """,
                (feature_name,),
            )
            r = cur.fetchone()
            if r:
                target_cid = str(r["context_id"])
            else:
                target_cid = LEGACY_UNKNOWN_CONTEXT_ID

        dossier["context_id"] = target_cid

        # Context details
        cur.execute(
            "SELECT market, sampling_interval_sec, sliding_window, feature_project_id FROM dataset_contexts WHERE context_id = ?",
            (target_cid,),
        )
        c_row = cur.fetchone()
        if c_row:
            dossier["market"] = str(c_row["market"] or "UNKNOWN")
            dossier["sampling_interval_sec"] = int(c_row["sampling_interval_sec"] or 0)
            dossier["sliding_window"] = str(c_row["sliding_window"] or "standard")
            dossier["feature_project_id"] = str(c_row["feature_project_id"] or "all")

        # 2. Query Lineage Summary
        cur.execute(
            """
            SELECT total_runs, unique_models_count, consecutive_keep_count,
                   consecutive_remove_count, keep_runs, watch_runs, remove_runs,
                   lineage_evidence_score, lifecycle_status, last_validated_at
            FROM experimental_lineage_summary
            WHERE feature_name = ? AND context_id = ?
            ORDER BY total_runs DESC LIMIT 1;
            """,
            (feature_name, target_cid),
        )
        lin_row = cur.fetchone()
        if lin_row:
            dossier["feature_source"] = "experimental"
            dossier["total_validation_runs"] = int(lin_row["total_runs"] or 0)
            dossier["unique_model_count"] = int(lin_row["unique_models_count"] or 0)
            dossier["consecutive_keep_count"] = int(lin_row["consecutive_keep_count"] or 0)
            dossier["consecutive_remove_count"] = int(lin_row["consecutive_remove_count"] or 0)
            dossier["keep_runs"] = int(lin_row["keep_runs"] or 0)
            dossier["watch_runs"] = int(lin_row["watch_runs"] or 0)
            dossier["remove_runs"] = int(lin_row["remove_runs"] or 0)
            dossier["lineage_evidence_score"] = float(lin_row["lineage_evidence_score"] or 0.0)
            dossier["lifecycle_status"] = str(lin_row["lifecycle_status"] or "active")
            last_val_at = lin_row["last_validated_at"]
        else:
            # Fallback to feature_context_summary
            cur.execute(
                """
                SELECT feature_source, total_runs, unique_models_count, consecutive_keep_count,
                       consecutive_remove_count, keep_runs, watch_runs, remove_runs,
                       evidence_score, lifecycle_status, last_validated_at
                FROM feature_context_summary
                WHERE feature_name = ? AND context_id = ?
                LIMIT 1;
                """,
                (feature_name, target_cid),
            )
            ctx_row = cur.fetchone()
            if ctx_row:
                dossier["feature_source"] = str(ctx_row["feature_source"] or "experimental")
                dossier["total_validation_runs"] = int(ctx_row["total_runs"] or 0)
                dossier["unique_model_count"] = int(ctx_row["unique_models_count"] or 0)
                dossier["consecutive_keep_count"] = int(ctx_row["consecutive_keep_count"] or 0)
                dossier["consecutive_remove_count"] = int(ctx_row["consecutive_remove_count"] or 0)
                dossier["keep_runs"] = int(ctx_row["keep_runs"] or 0)
                dossier["watch_runs"] = int(ctx_row["watch_runs"] or 0)
                dossier["remove_runs"] = int(ctx_row["remove_runs"] or 0)
                dossier["lineage_evidence_score"] = float(ctx_row["evidence_score"] or 0.0)
                dossier["lifecycle_status"] = str(ctx_row["lifecycle_status"] or "active")
                last_val_at = ctx_row["last_validated_at"]
            else:
                last_val_at = None

        # 3. Query Raw Validation Evidence for this Feature in Target Context
        cur.execute(
            """
            SELECT evidence_id, model_name, recommendation,
                   holdout_rank, unseen_rank, rank_change,
                   relative_imp_drop, drift_severity, run_timestamp, evidence_detail_json
            FROM recommendation_evidence
            WHERE feature_name = ? AND context_id = ?
            ORDER BY run_timestamp ASC, evidence_id ASC;
            """,
            (feature_name, target_cid),
        )
        ev_rows = [dict(r) for r in cur.fetchall()]
        dossier["raw_evidence_runs_count"] = len(ev_rows)
        dossier["validation_history"] = ev_rows

        # Calculate Intelligence Metrics
        conf = compute_evidence_confidence(
            dossier["total_validation_runs"],
            dossier["unique_model_count"],
            policy=pol,
        )
        dossier["evidence_confidence"] = conf

        consensus = compute_model_consensus(ev_rows)
        dossier["model_consensus"] = consensus
        dossier["dominant_recommendation"] = consensus.get("dominant_recommendation")

        freshness = compute_recency_staleness(last_val_at)
        dossier["freshness"] = freshness

        volatility = compute_score_volatility(ev_rows, policy=pol)
        dossier["score_volatility"] = volatility.get("volatility_score")
        dossier["score_range"] = volatility.get("score_range")
        dossier["direction_flips"] = volatility.get("direction_flips")
        dossier["stability_label"] = volatility.get("stability_label", "Insufficient Data")

        # 4. Cross-Context Generalization Query
        cur.execute(
            """
            SELECT context_id, feature_name, evidence_score,
                   last_recommendation, total_runs
            FROM feature_context_summary
            WHERE feature_name = ? AND total_runs > 0;
            """,
            (feature_name,),
        )
        all_ctx_rows = cur.fetchall()
        ctx_dict: dict[str, dict[str, Any]] = {}
        for r in all_ctx_rows:
            cid_item = str(r["context_id"])
            if cid_item not in ctx_dict:
                ctx_dict[cid_item] = {}
            ctx_dict[cid_item][feature_name] = dict(r)

        # Comparable contexts: all distinct contexts where this feature was tested
        if target_cid not in ctx_dict:
            ctx_dict[target_cid] = {
                feature_name: {
                    "context_id": target_cid,
                    "feature_name": feature_name,
                    "evidence_score": dossier["lineage_evidence_score"],
                    "last_recommendation": dossier.get("dominant_recommendation") or "NONE",
                    "total_runs": dossier["total_validation_runs"],
                }
            }
        comparable_cids = list(ctx_dict.keys())
        gen_res = compute_context_generalization(
            target_cid,
            feature_name,
            ctx_dict,
            comparable_cids,
        )
        dossier["comparable_context_count"] = gen_res.get("comparable_context_count", 1)
        dossier["generalization_score"] = gen_res.get("generalization_score")
        dossier["generalization_details"] = gen_res

        # 5. Phase 3A Decision Standing Evaluation
        is_promo_qual = (
            dossier["consecutive_keep_count"] >= pol.experimental_lifecycle.promotion_candidate_consecutive_keep
            and dossier["unique_model_count"] >= pol.experimental_lifecycle.experimental_promotion_min_unique_models
            and dossier["lineage_evidence_score"] >= pol.experimental_lifecycle.promotion_candidate_min_score
        )

        p3a_res = evaluate_training_decision(
            feature_name=feature_name,
            context_id=target_cid,
            feature_source=dossier["feature_source"],
            total_runs=dossier["total_validation_runs"],
            unique_models_count=dossier["unique_model_count"],
            evidence_score=dossier["lineage_evidence_score"],
            lifecycle_status=dossier["lifecycle_status"],
            consecutive_remove_count=dossier["consecutive_remove_count"],
            remove_runs=dossier["remove_runs"],
            consecutive_keep_count=dossier["consecutive_keep_count"],
            dominant_recommendation=dossier["dominant_recommendation"],
            is_consensus_tie=bool(consensus.get("is_tie", False)),
            freshness_label=freshness.get("freshness_label"),
            score_volatility=dossier["score_volatility"],
            is_promotion_candidate=is_promo_qual,
            policy=pol,
        )

        decision_str = str(getattr(p3a_res.decision, "value", p3a_res.decision))
        dossier["phase_3a_decision"] = decision_str
        dossier["phase_3a_primary_reason"] = str(p3a_res.primary_reason or "")
        dossier["is_phase_3a_promotion_qualified"] = (
            decision_str == str(TrainingDecisionState.TRAIN_CANDIDATE)
            and is_promo_qual
        )

        # Health status check
        if dossier["lifecycle_status"] in ("blocked", "extreme_drop", "degraded"):
            dossier["health_status"] = dossier["lifecycle_status"].upper()
            dossier["has_health_alert"] = True
        else:
            dossier["health_status"] = "HEALTHY"
            dossier["has_health_alert"] = False

    finally:
        conn.close()

    # 6. Evaluate Prerequisites on Compiled Dossier
    dossier["prerequisites_evaluation"] = evaluate_graduation_prerequisites(
        data_dir, feature_name, context_id=dossier["context_id"], policy=pol, precompiled_dossier=dossier
    )

    return dossier


def feature_graduation_audit_log_path(data_dir: str) -> str:
    """Returns the path to the append-only graduation audit log JSON."""
    return os.path.join(data_dir, "feature_graduation_audit_log.json")


def _atomic_save_json(path: str, data: Any) -> None:
    """Atomically writes JSON document to disk using temporary file and os.replace."""
    dir_name = os.path.dirname(os.path.abspath(path))
    os.makedirs(dir_name, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=dir_name, prefix=".tmp_grad_", suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2)
            fh.flush()
            try:
                os.fsync(fh.fileno())
            except OSError:
                pass
        os.replace(tmp_path, path)
    except Exception:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass
        raise


def get_feature_graduation_audit_log(data_dir: str) -> list[dict[str, Any]]:
    """Pure read-only retrieval of graduation audit events."""
    path = feature_graduation_audit_log_path(data_dir)
    if not os.path.isfile(path):
        return []
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, list) else []
    except Exception:
        return []


def execute_registry_graduation(
    data_dir: str,
    feature_name: str,
    approval_payload: dict[str, Any],
    policy: RecommendationPolicy | None = None,
) -> dict[str, Any]:
    """Executes atomic Feature Registry Graduation (Phase 3D.3).

    Steps:
    1. Validates human approval payload structure and required metadata.
    2. Recompiles live evidence dossier and re-verifies qualification status.
    3. Verifies scope alignment and context isolation.
    4. Protects against duplicates and prior graduation.
    5. Allocates sequential FRxxxx identity.
    6. Atomically updates feature_registry_store.json.
    7. Atomically appends event to feature_graduation_audit_log.json.
    8. Preserves recommendation_evidence historical rows immutable (zero modifications).

    Returns a structured status result dictionary.
    """
    pol = policy or load_recommendation_policy(data_dir)

    # 1. Validate Approval Payload
    if not isinstance(approval_payload, dict):
        return {
            "status": "INVALID_APPROVAL",
            "message": "Approval payload must be a non-empty dictionary.",
        }

    if approval_payload.get("action") != "APPROVE":
        return {
            "status": "INVALID_APPROVAL",
            "message": f"Payload action is '{approval_payload.get('action')}', expected 'APPROVE'.",
        }

    p_feat = str(approval_payload.get("feature_name") or "").strip()
    if p_feat != feature_name:
        return {
            "status": "INVALID_APPROVAL",
            "message": f"Payload feature_name '{p_feat}' does not match target feature '{feature_name}'.",
        }

    domain = str(approval_payload.get("domain") or "").strip()
    group = str(approval_payload.get("group") or "").strip()
    desc = str(approval_payload.get("description") or "").strip()

    if not domain or not group or not desc:
        return {
            "status": "INVALID_APPROVAL",
            "message": "Approval payload missing required fields: domain, group, or description.",
        }

    target_cid = approval_payload.get("context_id")

    # 2. Live Re-Verification
    fresh_dossier = compile_feature_evidence_dossier(
        data_dir, feature_name, context_id=target_cid, policy=pol
    )
    fresh_eval = fresh_dossier.get("prerequisites_evaluation") or {}
    fresh_status = str(fresh_eval.get("status") or "NOT_READY")

    if fresh_status == "NOT_READY":
        return {
            "status": "NOT_QUALIFIED",
            "message": f"Feature '{feature_name}' failed graduation prerequisites upon live re-verification.",
            "failed_checks": fresh_eval.get("failed_checks", []),
        }

    # 3. Verify Scope & Context Alignment
    is_universal_approved = bool(approval_payload.get("is_universal_ready", False))
    if is_universal_approved and fresh_status != "UNIVERSAL_READY":
        return {
            "status": "CONTEXT_MISMATCH",
            "message": (
                f"Approval requested UNIVERSAL_READY scope, but live evidence only qualifies "
                f"for '{fresh_status}' (K={fresh_dossier.get('comparable_context_count')}, "
                f"G={fresh_dossier.get('generalization_score')})."
            ),
        }

    # 4. Check Duplicate / Already Graduated
    store = load_store(data_dir)
    ids_map = store.get("feature_ids") or {}
    if feature_name in ids_map:
        existing_fid = ids_map[feature_name]
        return {
            "status": "ALREADY_GRADUATED",
            "feature_id": existing_fid,
            "message": f"Feature '{feature_name}' is already graduated as {existing_fid}.",
        }

    # 5. Allocate Sequential FR ID
    fid = _next_free_feature_id(store)
    now_iso = datetime.now(timezone.utc).isoformat()
    is_univ = (fresh_status == "UNIVERSAL_READY")
    resolved_allowed_contexts = ["ALL"] if is_univ else [str(fresh_dossier.get("context_id") or "UNKNOWN")]

    # 6. Construct Permanent Identity
    identities = store.setdefault("feature_identities", {})
    overrides = store.setdefault("overrides", {})

    identity_record: dict[str, Any] = {
        "feature_id": fid,
        "name": feature_name,
        "display_name": approval_payload.get("display_name") or feature_name,
        "previous_names": [],
        "created_at": now_iso,
        "created_by": approval_payload.get("reviewer") or approval_payload.get("created_by") or "Human Reviewer",
        "updated_at": now_iso,
        "version": "1.0",
        "owner": approval_payload.get("owner") or default_owner_for_group(group),
        "group_id": group,
        "domain": domain,
        "formula": approval_payload.get("formula") or f"calc_{feature_name}(ohlcv)",
        "description": desc,
        "expected_data_type": approval_payload.get("expected_data_type") or "float",
        "implementation_status": "implemented",
        "scope": "universal" if is_univ else "context_scoped",
        "allowed_contexts": resolved_allowed_contexts,
        "originating_context": fresh_dossier.get("context_id"),
        "graduation_classification": fresh_status,
        "reviewer_notes": approval_payload.get("reviewer_notes") or "",
        "dossier_snapshot": {
            "total_validation_runs": fresh_dossier.get("total_validation_runs"),
            "unique_model_count": fresh_dossier.get("unique_model_count"),
            "consecutive_keep_count": fresh_dossier.get("consecutive_keep_count"),
            "lineage_evidence_score": fresh_dossier.get("lineage_evidence_score"),
            "evidence_confidence": fresh_dossier.get("evidence_confidence"),
            "score_volatility": fresh_dossier.get("score_volatility"),
            "generalization_score": fresh_dossier.get("generalization_score"),
            "comparable_context_count": fresh_dossier.get("comparable_context_count"),
        },
    }

    identities[fid] = identity_record
    ids_map[feature_name] = fid
    store["feature_ids"] = ids_map
    overrides[feature_name] = {
        "group": group,
        "formula": identity_record["formula"],
        "description": desc,
        "domain": domain,
    }

    # 7. Audit Log Entry
    audit_path = feature_graduation_audit_log_path(data_dir)
    audit_log = get_feature_graduation_audit_log(data_dir)

    event_id = f"grad_evt_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{fid.lower()}"
    audit_entry: dict[str, Any] = {
        "event_id": event_id,
        "feature_name": feature_name,
        "assigned_feature_id": fid,
        "context_id": fresh_dossier.get("context_id"),
        "graduation_scope": "UNIVERSAL" if is_univ else "CONTEXT_SCOPED",
        "previous_source": fresh_dossier.get("feature_source", "experimental"),
        "new_source": "registry",
        "reviewer_information": approval_payload.get("reviewer") or "Human Reviewer",
        "reviewer_notes": approval_payload.get("reviewer_notes") or "",
        "timestamp": now_iso,
        "dossier_snapshot": identity_record["dossier_snapshot"],
        "feature_definition": {
            "domain": domain,
            "group": group,
            "expected_data_type": identity_record["expected_data_type"],
            "formula": identity_record["formula"],
            "description": desc,
            "allowed_contexts": resolved_allowed_contexts,
        },
        "decision_engine_version": "3D.3",
    }
    audit_log.append(audit_entry)

    # 8. Atomic Cross-File Writes
    try:
        _atomic_save_json(store_path(data_dir), store)
        _atomic_save_json(audit_path, audit_log)
    except Exception as exc:
        return {
            "status": "WRITE_FAILURE",
            "message": f"Atomic write failed during store or audit commit: {exc}",
        }

    # 9. Update Lineage Summary Projection (if graduated_feature_id exists)
    db_p = evidence_db_path(data_dir)
    if os.path.isfile(db_p):
        try:
            with sqlite3.connect(db_p, timeout=10.0) as conn:
                cur = conn.execute("PRAGMA table_info(experimental_lineage_summary);")
                cols = [r[1] for r in cur.fetchall()]
                if "graduated_feature_id" in cols:
                    conn.execute(
                        "UPDATE experimental_lineage_summary SET graduated_feature_id = ? WHERE feature_name = ?;",
                        (fid, feature_name),
                    )
        except Exception:
            pass

    return {
        "status": "SUCCESS",
        "feature_name": feature_name,
        "assigned_feature_id": fid,
        "scope_classification": fresh_status,
        "allowed_contexts": resolved_allowed_contexts,
        "is_base_pipeline_eligible": is_univ,
        "audit_event_id": event_id,
        "message": f"Feature '{feature_name}' successfully graduated as {fid} ({fresh_status}).",
    }


def evaluate_base_pipeline_eligibility(
    data_dir: str,
    feature_ref: str,
    policy: RecommendationPolicy | None = None,
    precompiled_dossier: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluates whether an already graduated Universal Registry feature meets Base Pipeline promotion criteria.

    Prerequisites:
    1. Registry Identity: Exists in feature_registry_store.json with valid FRxxxx identity and implementation_status != 'deprecated'.
    2. Universal Scope: Scope is 'universal', allowed_contexts is ['ALL'], K >= 2, G >= 0.50.
    3. Validation Volume: total_runs >= 5, unique_model_count >= 3.
    4. Strong Standing: evidence_score >= 80.0, evidence_confidence >= 0.75, score_volatility <= 20.0 (with N >= 3).
    5. Clean Health: No active health alert / degraded / blocked state.
    6. Graduation Standing: Phase 3A/3D prerequisites satisfied.

    Context-scoped features (K=1 or G < 0.50) are strictly rejected with CONTEXT_SCOPED_PROHIBITED.
    """
    pol = policy or load_recommendation_policy(data_dir)

    # 1. Resolve Feature in Registry
    store = load_store(data_dir)
    ids_map = store.get("feature_ids") or {}
    identities = store.get("feature_identities") or {}

    feature_id: str | None = None
    feature_name: str | None = None

    ref_str = str(feature_ref or "").strip()
    if ref_str in identities:
        feature_id = ref_str
        feature_name = str(identities[feature_id].get("name") or ref_str)
    elif ref_str in ids_map:
        feature_name = ref_str
        feature_id = str(ids_map[feature_name])
    else:
        # Check if ref is a name in identities
        for fid, ident in identities.items():
            if isinstance(ident, dict) and ident.get("name") == ref_str:
                feature_id = fid
                feature_name = ref_str
                break

    if not feature_id or not feature_name:
        return {
            "status": "NOT_GRADUATED",
            "is_eligible": False,
            "feature_ref": feature_ref,
            "feature_id": None,
            "feature_name": feature_ref,
            "failed_checks": ["registry_identity"],
            "checks": {
                "registry_identity": {
                    "passed": False,
                    "description": f"Feature '{feature_ref}' is not registered in feature_registry_store.json (must graduate first).",
                }
            },
            "explanations": [f"Feature '{feature_ref}' must be graduated into Feature Registry before Base Pipeline promotion."],
        }

    ident = identities.get(feature_id) or {}
    impl_status = str(ident.get("implementation_status") or "implemented").lower()
    if impl_status == "deprecated":
        return {
            "status": "NOT_ELIGIBLE",
            "is_eligible": False,
            "feature_ref": feature_ref,
            "feature_id": feature_id,
            "feature_name": feature_name,
            "failed_checks": ["active_registry_status"],
            "checks": {
                "active_registry_status": {
                    "passed": False,
                    "description": f"Feature '{feature_name}' ({feature_id}) is deprecated in the Feature Registry.",
                }
            },
            "explanations": [f"Deprecated feature '{feature_name}' cannot be promoted to Base Pipeline."],
        }

    # 2. Compile Live Dossier
    if precompiled_dossier:
        dossier = dict(precompiled_dossier)
    else:
        dossier = compile_feature_evidence_dossier(data_dir, feature_name, policy=pol)

    checks: dict[str, dict[str, Any]] = {}
    failed_checks: list[str] = []

    # Check 1: Registry Identity
    checks["registry_identity"] = {
        "passed": True,
        "description": f"Valid Feature Registry identity: {feature_id} ({feature_name})",
        "value": feature_id,
    }

    # Check 2: Universal Scope (K >= 2, G >= 0.50, allowed_contexts == ["ALL"])
    scope = str(ident.get("scope") or "universal" if dossier.get("prerequisites_evaluation", {}).get("is_universal_ready") else "context_scoped")
    allowed_ctxs = list(ident.get("allowed_contexts") or (["ALL"] if scope == "universal" else [dossier.get("context_id")]))
    K = int(dossier.get("comparable_context_count") or 1)
    G = dossier.get("generalization_score")
    G_val = float(G) if G is not None else -1.0

    is_scope_universal = (
        scope == "universal"
        and (allowed_ctxs == ["ALL"] or "ALL" in allowed_ctxs)
        and K >= 2
        and G_val >= 0.50
    )

    checks["universal_scope"] = {
        "passed": is_scope_universal,
        "description": f"Universal Scope verified: K={K} (req >= 2), G={G_val:.2f} (req >= 0.50), allowed_contexts={allowed_ctxs}",
        "value": {"K": K, "G": G, "allowed_contexts": allowed_ctxs},
    }
    if not is_scope_universal:
        failed_checks.append("universal_scope")

    # Check 3: Total Validation Runs >= 5
    n_runs = int(dossier.get("total_validation_runs") or 0)
    c_runs = (n_runs >= 5)
    checks["validation_runs_count"] = {
        "passed": c_runs,
        "description": f"Validation runs count: {n_runs} (req >= 5)",
        "value": n_runs,
    }
    if not c_runs:
        failed_checks.append("validation_runs_count")

    # Check 4: Unique Model Count >= 3
    u_models = int(dossier.get("unique_model_count") or 0)
    c_models = (u_models >= 3)
    checks["unique_models_count"] = {
        "passed": c_models,
        "description": f"Unique model architectures tested: {u_models} (req >= 3)",
        "value": u_models,
    }
    if not c_models:
        failed_checks.append("unique_models_count")

    # Check 5: Evidence Score >= 80.0
    ev_score = float(dossier.get("lineage_evidence_score") or 0.0)
    c_score = (ev_score >= 80.0)
    checks["evidence_score"] = {
        "passed": c_score,
        "description": f"Lineage evidence score: {ev_score:+.1f} (req >= +80.0)",
        "value": ev_score,
    }
    if not c_score:
        failed_checks.append("evidence_score")

    # Check 6: Evidence Confidence >= 0.75
    ev_conf = float(dossier.get("evidence_confidence") or 0.0)
    c_conf = (ev_conf >= 0.75)
    checks["evidence_confidence"] = {
        "passed": c_conf,
        "description": f"Evidence confidence saturation: {ev_conf:.2%} (req >= 75.0%)",
        "value": ev_conf,
    }
    if not c_conf:
        failed_checks.append("evidence_confidence")

    # Check 7: Score Volatility <= 20.0 (requires N >= 3)
    vol = dossier.get("score_volatility")
    c_vol = (vol is not None and float(vol) <= 20.0)
    checks["score_volatility"] = {
        "passed": c_vol,
        "description": f"Temporal score volatility: {vol if vol is not None else 'N/A (<3 runs)'} (req <= 20.0)",
        "value": vol,
    }
    if not c_vol:
        failed_checks.append("score_volatility")

    # Check 8: Health & Degradation Integrity
    h_stat = str(dossier.get("health_status") or "HEALTHY")
    has_alert = bool(dossier.get("has_health_alert", False))
    c_health = (h_stat not in ("BLOCKED", "EXTREME_DROP", "DEGRADED") and not has_alert)
    checks["health_integrity"] = {
        "passed": c_health,
        "description": f"Health integrity check: {h_stat} (alert={has_alert})",
        "value": {"health_status": h_stat, "has_health_alert": has_alert},
    }
    if not c_health:
        failed_checks.append("health_integrity")

    # Final Classification
    explanations: list[str] = []
    if not is_scope_universal:
        status = "CONTEXT_SCOPED_PROHIBITED"
        is_eligible = False
        explanations.append(
            f"Feature '{feature_name}' ({feature_id}) is context-scoped and permanently prohibited from Universal Base Pipeline."
        )
    elif failed_checks:
        status = "NOT_READY"
        is_eligible = False
        explanations.append(
            f"Feature '{feature_name}' ({feature_id}) is not ready for Base Pipeline. Failed checks: {', '.join(failed_checks)}."
        )
    else:
        status = "ELIGIBLE"
        is_eligible = True
        explanations.append(
            f"Feature '{feature_name}' ({feature_id}) meets all Base Pipeline promotion criteria. Ready for engineering review."
        )

    return {
        "status": status,
        "is_eligible": is_eligible,
        "feature_ref": feature_ref,
        "feature_id": feature_id,
        "feature_name": feature_name,
        "checks": checks,
        "failed_checks": failed_checks,
        "passed_checks_count": sum(1 for c in checks.values() if c["passed"]),
        "failed_checks_count": len(failed_checks),
        "explanations": explanations,
        "dossier_snapshot": {
            "total_validation_runs": n_runs,
            "unique_model_count": u_models,
            "consecutive_keep_count": dossier.get("consecutive_keep_count"),
            "lineage_evidence_score": ev_score,
            "evidence_confidence": ev_conf,
            "score_volatility": vol,
            "generalization_score": G,
            "comparable_context_count": K,
        },
    }


def execute_base_pipeline_promotion(
    data_dir: str,
    feature_ref: str,
    approval_payload: dict[str, Any],
    policy: RecommendationPolicy | None = None,
) -> dict[str, Any]:
    """Executes atomic Base Pipeline Promotion (Phase 3D.4A).

    Steps:
    1. Validates human approval payload:
       - base_pipeline_promotion == True
       - latency_budget_compliant == True
       - reviewer and reviewer_notes present
    2. Live Re-Verification:
       - Re-evaluates Base Pipeline eligibility.
       - Aborts if feature is not ELIGIBLE.
    3. Duplicate Check:
       - Checks if feature_id is already in PL_0001['registry_feature_ids'].
    4. Atomic Store Commit:
       - Appends FRxxxx identity to PL_0001['registry_feature_ids'].
       - Updates updated_at timestamp.
       - Marks feature_identities[fid]['is_base_pipeline'] = True in feature_registry_store.json.
       - Atomically writes pipeline_registry_store.json and feature_registry_store.json.
    5. Append Immutable Audit Log:
       - Writes BASE_PIPELINE_PROMOTION event to data/feature_graduation_audit_log.json.

    Returns a structured status result dictionary.
    """
    pol = policy or load_recommendation_policy(data_dir)

    # 1. Validate Approval Payload
    if not isinstance(approval_payload, dict):
        return {
            "status": "INVALID_APPROVAL",
            "message": "Approval payload must be a non-empty dictionary.",
        }

    if not approval_payload.get("base_pipeline_promotion"):
        return {
            "status": "INVALID_APPROVAL",
            "message": "Approval payload must explicitly declare 'base_pipeline_promotion': True.",
        }

    if not approval_payload.get("latency_budget_compliant"):
        return {
            "status": "LATENCY_APPROVAL_REQUIRED",
            "message": "Base Pipeline promotion requires explicit 'latency_budget_compliant': True confirmation.",
        }

    reviewer = str(approval_payload.get("reviewer") or approval_payload.get("reviewer_information") or "").strip()
    notes = str(approval_payload.get("reviewer_notes") or "").strip()
    if not reviewer or not notes:
        return {
            "status": "INVALID_APPROVAL",
            "message": "Approval payload must include reviewer identity and reviewer_notes.",
        }

    # 2. Live Re-Verification
    elig = evaluate_base_pipeline_eligibility(data_dir, feature_ref, policy=pol)
    if not elig["is_eligible"]:
        return {
            "status": "NOT_QUALIFIED" if elig["status"] != "CONTEXT_SCOPED_PROHIBITED" else "CONTEXT_MISMATCH",
            "eligibility_status": elig["status"],
            "message": f"Feature '{feature_ref}' failed Base Pipeline qualification upon live re-verification.",
            "failed_checks": elig.get("failed_checks", []),
        }

    feature_id = elig["feature_id"]
    feature_name = elig["feature_name"]

    # 3. Duplicate Check in Pipeline Store
    pipe_doc = load_pipeline_store(data_dir)
    pipelines = pipe_doc.setdefault("pipelines", {})
    if not pipelines:
        pipe_doc = ensure_default_existing_pipeline(data_dir)
        pipelines = pipe_doc.get("pipelines") or {}

    base_pid: str | None = None
    for pid, rec in pipelines.items():
        if isinstance(rec, dict) and is_base_pipeline_record(rec):
            base_pid = str(pid)
            break

    if not base_pid:
        pipe_doc = ensure_default_existing_pipeline(data_dir)
        pipelines = pipe_doc.get("pipelines") or {}
        for pid, rec in pipelines.items():
            if isinstance(rec, dict) and is_base_pipeline_record(rec):
                base_pid = str(pid)
                break

    base_rec = pipelines[base_pid]
    existing_reg_ids = list(base_rec.get("registry_feature_ids") or [])

    if feature_id in existing_reg_ids:
        return {
            "status": "ALREADY_IN_BASE_PIPELINE",
            "pipeline_id": base_pid,
            "feature_id": feature_id,
            "feature_name": feature_name,
            "message": f"Feature {feature_id} ({feature_name}) is already active in Base Pipeline {base_pid}.",
        }

    # 4. Atomic Mutation
    now_iso = datetime.now(timezone.utc).isoformat()
    existing_reg_ids.append(feature_id)
    base_rec["registry_feature_ids"] = existing_reg_ids
    base_rec["updated_at"] = now_iso

    pipe_doc.setdefault("history", []).append({
        "ts": now_iso,
        "action": "base_pipeline_promotion",
        "pipeline_id": base_pid,
        "feature_id": feature_id,
        "feature_name": feature_name,
        "reviewer": reviewer,
    })

    # Update Registry Store metadata flag
    feat_store = load_store(data_dir)
    if feature_id in (feat_store.get("feature_identities") or {}):
        feat_store["feature_identities"][feature_id]["is_base_pipeline"] = True
        feat_store["feature_identities"][feature_id]["updated_at"] = now_iso

    # 5. Append Audit Log
    audit_path = feature_graduation_audit_log_path(data_dir)
    audit_log = get_feature_graduation_audit_log(data_dir)

    event_id = f"base_prom_evt_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{feature_id.lower()}"
    audit_entry: dict[str, Any] = {
        "event_id": event_id,
        "event_type": "BASE_PIPELINE_PROMOTION",
        "feature_name": feature_name,
        "assigned_feature_id": feature_id,
        "context_id": "ALL",
        "previous_source": "registry",
        "new_source": "base_pipeline",
        "pipeline_id": base_pid,
        "reviewer_information": reviewer,
        "reviewer_notes": notes,
        "latency_budget_compliant": True,
        "timestamp": now_iso,
        "decision_engine_version": "3D.4A",
        "dossier_snapshot": elig.get("dossier_snapshot"),
        "promotion_qualification_snapshot": elig.get("checks"),
    }
    audit_log.append(audit_entry)

    # 6. Commit Transactions Atomically
    try:
        _atomic_save_json(pipeline_store_path(data_dir), pipe_doc)
        _atomic_save_json(store_path(data_dir), feat_store)
        _atomic_save_json(audit_path, audit_log)
    except Exception as exc:
        return {
            "status": "WRITE_FAILURE",
            "message": f"Atomic write failed during pipeline promotion commit: {exc}",
        }

    return {
        "status": "SUCCESS",
        "feature_name": feature_name,
        "assigned_feature_id": feature_id,
        "pipeline_id": base_pid,
        "audit_event_id": event_id,
        "message": f"Feature '{feature_name}' ({feature_id}) successfully promoted to Base Pipeline {base_pid}.",
    }


def is_feature_in_base_pipeline(data_dir: str, feature_id_or_name: str) -> bool:
    """Authoritative runtime check: returns True iff FRxxxx identity is in PL_0001.registry_feature_ids.

    Uses pipeline_registry_store.json as the sole ground truth.
    """
    ref = str(feature_id_or_name or "").strip()
    if not ref:
        return False

    fid: str | None = None
    if ref.upper().startswith("FR"):
        fid = ref.upper()
    else:
        feat_store = load_store(data_dir)
        ids_map = feat_store.get("feature_ids") or {}
        if ref in ids_map:
            fid = str(ids_map[ref])
        else:
            identities = feat_store.get("feature_identities") or {}
            for k, v in identities.items():
                if isinstance(v, dict) and v.get("name") == ref:
                    fid = k
                    break

    if not fid:
        return False

    pipe_doc = load_pipeline_store(data_dir)
    pipelines = pipe_doc.get("pipelines") or {}
    for pid, rec in pipelines.items():
        if isinstance(rec, dict) and is_base_pipeline_record(rec):
            reg_ids = [str(x).upper() for x in (rec.get("registry_feature_ids") or [])]
            return fid in reg_ids

    return False


def evaluate_deprecation_prerequisites(
    data_dir: str,
    feature_ref: str,
    policy: RecommendationPolicy | None = None,
) -> dict[str, Any]:
    """Evaluates whether a graduated Registry / Base Pipeline feature is eligible for governed deprecation.

    Checks:
    1. Feature existence in Feature Registry (feature_identities or feature_ids).
    2. Valid permanent FRxxxx identity format.
    3. Current deprecation status (returns ALREADY_DEPRECATED if already deprecated).
    4. Base Pipeline membership resolution via runtime authority.
    """
    ref_str = str(feature_ref or "").strip()
    if not ref_str:
        return {
            "status": "INVALID_FEATURE_ID",
            "is_eligible_for_deprecation": False,
            "feature_ref": feature_ref,
            "feature_id": None,
            "feature_name": None,
            "explanations": ["Empty or invalid feature reference."],
        }

    feat_store = load_store(data_dir)
    ids_map = feat_store.get("feature_ids") or {}
    identities = feat_store.get("feature_identities") or {}

    feature_id: str | None = None
    feature_name: str | None = None

    if ref_str in identities:
        feature_id = ref_str
        feature_name = str(identities[feature_id].get("name") or ref_str)
    elif ref_str in ids_map:
        feature_name = ref_str
        feature_id = str(ids_map[feature_name])
    else:
        for fid, ident in identities.items():
            if isinstance(ident, dict) and ident.get("name") == ref_str:
                feature_id = fid
                feature_name = ref_str
                break

    if not feature_id or not feature_name:
        return {
            "status": "FEATURE_NOT_FOUND",
            "is_eligible_for_deprecation": False,
            "feature_ref": feature_ref,
            "feature_id": None,
            "feature_name": ref_str,
            "explanations": [
                f"Feature '{ref_str}' is not registered in Feature Registry. "
                "Only graduated Registry/Base Pipeline features can be deprecated through this workflow."
            ],
        }

    if not feature_id.upper().startswith("FR"):
        return {
            "status": "INVALID_FEATURE_ID",
            "is_eligible_for_deprecation": False,
            "feature_ref": feature_ref,
            "feature_id": feature_id,
            "feature_name": feature_name,
            "explanations": [f"Feature ID '{feature_id}' is not a valid FRxxxx identity."],
        }

    ident = identities.get(feature_id) or {}
    impl_status = str(ident.get("implementation_status") or "implemented").lower()
    if impl_status == "deprecated":
        return {
            "status": "ALREADY_DEPRECATED",
            "is_eligible_for_deprecation": False,
            "feature_ref": feature_ref,
            "feature_id": feature_id,
            "feature_name": feature_name,
            "deprecated_at": ident.get("deprecated_at"),
            "deprecated_by": ident.get("deprecated_by"),
            "deprecation_reason": ident.get("deprecation_reason"),
            "explanations": [
                f"Feature '{feature_name}' ({feature_id}) has already been deprecated on {ident.get('deprecated_at')}."
            ],
        }

    in_base = is_feature_in_base_pipeline(data_dir, feature_id)
    scope = str(ident.get("scope") or "universal")

    # Optional dossier compilation for metadata snapshot
    pol = policy or load_recommendation_policy(data_dir)
    dossier: dict[str, Any] = {}
    try:
        dossier = compile_feature_evidence_dossier(data_dir, feature_name, policy=pol)
    except Exception:
        pass

    return {
        "status": "ELIGIBLE",
        "is_eligible_for_deprecation": True,
        "feature_ref": feature_ref,
        "feature_id": feature_id,
        "feature_name": feature_name,
        "current_scope": scope,
        "is_in_base_pipeline": in_base,
        "implementation_status": impl_status,
        "explanations": [
            f"Feature '{feature_name}' ({feature_id}) is active in Registry (in_base_pipeline={in_base}) "
            "and eligible for governed retirement/deprecation."
        ],
        "dossier_snapshot": {
            "total_validation_runs": dossier.get("total_validation_runs"),
            "unique_model_count": dossier.get("unique_model_count"),
            "consecutive_keep_count": dossier.get("consecutive_keep_count"),
            "lineage_evidence_score": dossier.get("lineage_evidence_score"),
            "evidence_confidence": dossier.get("evidence_confidence"),
            "score_volatility": dossier.get("score_volatility"),
            "generalization_score": dossier.get("generalization_score"),
            "comparable_context_count": dossier.get("comparable_context_count"),
        } if dossier else None,
    }


def execute_feature_deprecation(
    data_dir: str,
    feature_ref: str,
    deprecation_payload: dict[str, Any],
    policy: RecommendationPolicy | None = None,
) -> dict[str, Any]:
    """Executes atomic Feature Deprecation and Base Pipeline Demotion (Phase 3D.4B).

    Steps:
    1. Validates human approval payload:
       - action == "DEPRECATE"
       - reviewer_information, deprecation_reason, reviewer_notes present
    2. Live Re-Verification:
       - Checks evaluate_deprecation_prerequisites() to ensure feature is still registered and not yet deprecated.
    3. Atomic Multi-Store Modifications:
       - In feature_registry_store.json:
         * Sets feature_identities[fid]["implementation_status"] = "deprecated"
         * Sets feature_identities[fid]["is_base_pipeline"] = False
         * Records deprecated_at, deprecated_by, deprecation_reason
         * Adds feature_name to disabled_features map
         * Keeps feature_ids[name] and feature_identities[fid] intact (permanent identity)
         * Keeps deleted_feature_ids intact (no tombstone movement)
       - In pipeline_registry_store.json:
         * Removes fid from PL_0001["registry_feature_ids"] if present
         * Updates updated_at and history log
       - In feature_graduation_audit_log.json:
         * Appends immutable FEATURE_DEPRECATION event
    4. Commit Atomically:
       - Uses _atomic_save_json for all 3 store files.

    Returns structured result dictionary.
    """
    pol = policy or load_recommendation_policy(data_dir)

    # 1. Validate Deprecation Payload
    if not isinstance(deprecation_payload, dict) or not deprecation_payload:
        return {
            "status": "APPROVAL_REQUIRED",
            "message": "Feature deprecation requires an explicit human approval payload.",
        }

    action = str(deprecation_payload.get("action") or "").strip().upper()
    if action != "DEPRECATE":
        return {
            "status": "INVALID_APPROVAL",
            "message": f"Deprecation payload action must be 'DEPRECATE' (received '{action}').",
        }

    reviewer = str(
        deprecation_payload.get("reviewer_information")
        or deprecation_payload.get("reviewer")
        or ""
    ).strip()
    reason = str(
        deprecation_payload.get("deprecation_reason")
        or deprecation_payload.get("reason")
        or ""
    ).strip()
    notes = str(
        deprecation_payload.get("reviewer_notes")
        or deprecation_payload.get("notes")
        or ""
    ).strip()

    if not reviewer or not reason or not notes:
        return {
            "status": "INVALID_APPROVAL",
            "message": "Deprecation approval payload must include reviewer_information, deprecation_reason, and reviewer_notes.",
        }

    # 2. Live Re-Verification
    eval_res = evaluate_deprecation_prerequisites(data_dir, feature_ref, policy=pol)
    if not eval_res["is_eligible_for_deprecation"]:
        return {
            "status": eval_res["status"],
            "feature_ref": feature_ref,
            "feature_id": eval_res.get("feature_id"),
            "feature_name": eval_res.get("feature_name"),
            "message": eval_res.get("explanations", ["Feature is not eligible for deprecation."])[0],
        }

    feature_id = eval_res["feature_id"]
    feature_name = eval_res["feature_name"]
    was_in_base = bool(eval_res.get("is_in_base_pipeline", False))

    now_iso = datetime.now(timezone.utc).isoformat()

    # 3. Prepare Feature Registry Store Mutation
    feat_store = load_store(data_dir)
    identities = feat_store.setdefault("feature_identities", {})
    if feature_id in identities:
        ident = identities[feature_id]
        ident["implementation_status"] = "deprecated"
        ident["is_base_pipeline"] = False
        ident["deprecated_at"] = now_iso
        ident["deprecated_by"] = reviewer
        ident["deprecation_reason"] = reason
        ident["updated_at"] = now_iso

    # Add to disabled_features overlay
    disabled = feat_store.setdefault("disabled_features", {})
    disabled[feature_name] = {
        "feature_id": feature_id,
        "disabled_at": now_iso,
        "reason": reason,
        "disabled_by": reviewer,
    }

    # 4. Prepare Pipeline Registry Store Mutation
    pipe_doc = load_pipeline_store(data_dir)
    pipelines = pipe_doc.get("pipelines") or {}
    base_pid: str | None = None
    for pid, rec in pipelines.items():
        if isinstance(rec, dict) and is_base_pipeline_record(rec):
            base_pid = str(pid)
            reg_ids = list(rec.get("registry_feature_ids") or [])
            if feature_id in reg_ids:
                reg_ids = [fid for fid in reg_ids if fid != feature_id]
                rec["registry_feature_ids"] = reg_ids
                rec["updated_at"] = now_iso
                pipe_doc.setdefault("history", []).append({
                    "ts": now_iso,
                    "action": "base_pipeline_demotion",
                    "pipeline_id": base_pid,
                    "feature_id": feature_id,
                    "feature_name": feature_name,
                    "reason": reason,
                    "reviewer": reviewer,
                })
            break

    # 5. Prepare Audit Log Mutation
    audit_path = feature_graduation_audit_log_path(data_dir)
    audit_log = get_feature_graduation_audit_log(data_dir)

    event_id = f"feat_depr_evt_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{feature_id.lower()}"
    audit_entry: dict[str, Any] = {
        "event_id": event_id,
        "event_type": "FEATURE_DEPRECATION",
        "feature_name": feature_name,
        "assigned_feature_id": feature_id,
        "context_id": "ALL",
        "previous_source": "base_pipeline" if was_in_base else "registry",
        "new_source": "deprecated",
        "previous_base_pipeline_membership": was_in_base,
        "new_base_pipeline_membership": False,
        "reviewer_information": reviewer,
        "reviewer_notes": notes,
        "deprecation_reason": reason,
        "timestamp": now_iso,
        "decision_engine_version": "3D.4B",
        "dossier_snapshot": eval_res.get("dossier_snapshot"),
    }
    audit_log.append(audit_entry)

    # 6. Atomic Commit
    try:
        _atomic_save_json(store_path(data_dir), feat_store)
        _atomic_save_json(pipeline_store_path(data_dir), pipe_doc)
        _atomic_save_json(audit_path, audit_log)
    except Exception as exc:
        return {
            "status": "WRITE_FAILURE",
            "message": f"Atomic write failed during feature deprecation commit: {exc}",
        }

    return {
        "status": "DEPRECATED",
        "feature_name": feature_name,
        "assigned_feature_id": feature_id,
        "was_in_base_pipeline": was_in_base,
        "audit_event_id": event_id,
        "message": f"Feature '{feature_name}' ({feature_id}) successfully deprecated/retired from Registry and Base Pipeline.",
    }





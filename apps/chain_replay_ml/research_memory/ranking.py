"""Robustness Ranking Policy Engine & Pareto Multi-Objective Evaluation (Phase 4D.5).

Evaluates experimentally validated models within the same exact `ModelContextKey`,
ranking candidates according to multi-factor empirical robustness rather than peak
validation score alone.

Invariants:
1. Context-Scoped Isolation: Ranking occurs strictly within a single `ModelContextKey`.
   Cross-regime or cross-task models never enter the same ranking pool.
2. Robustness over Peak Score: Penalizes walk-forward fold variance, worst-fold drawdown,
   probability miscalibration (ECE), cross-regime degradation, experimental feature risk,
   deprecated feature exposure, and model complexity.
3. Policy Versioning & Immutability: Every ranking is calculated and stamped using a
   versioned, deterministically hashed `RobustnessRankingPolicy` (e.g. `ROB_POLICY_v1.0`).
4. Pure Ranking & Advisory Boundary: Outputs ranked candidate dossiers with recommendation
   statuses (e.g. `CHAMPION_CANDIDATE`). It strictly NEVER automatically modifies
   `.lifecycle_registry.db`, `active_model.json`, or promotes champions.
5. 16 GB Workstation Safety: Operates strictly on persisted scalar research memory records.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import sqlite3
from typing import Any

from .db import connect_analysis_db, init_analysis_db
from .feature_comp import get_feature_set_evaluation
from .regime_eval import get_regime_evaluations_for_model


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class RobustnessRankingPolicy:
    """Configurable and deterministically hashable robustness ranking policy."""

    policy_version: str = "ROB_POLICY_v1.0"
    lambda_sigma: float = 15.0          # Walk-forward fold variance penalty weight
    lambda_w: float = 10.0              # Worst-fold drawdown penalty weight
    tau_safe: float = 0.05              # Worst-fold safe drawdown threshold (5%)
    lambda_c: float = 10.0              # Expected Calibration Error (ECE) penalty weight
    lambda_deg: float = 0.15            # Cross-regime degradation penalty weight
    lambda_e: float = 12.0              # Experimental feature dependency penalty weight
    omega_dep: float = 25.0             # Deprecated feature exposure flat penalty
    lambda_N: float = 2.0               # Model parsimony / feature count penalty weight
    N_baseline: int = 30                # Parsimony baseline feature count
    tau_max_error: float = 1.0          # Normalization upper bound for lower-is-better metrics

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def compute_policy_hash(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# Canonical Default Policy instance
ROB_POLICY_v1_0 = RobustnessRankingPolicy()


def normalize_metric(
    metric_name: str,
    metric_value: float,
    *,
    tau_max: float = 1.0,
) -> float:
    """Normalize metric values to a standard [0.0, 1.0] scale where 1.0 is optimal."""
    try:
        val = float(metric_value)
    except (TypeError, ValueError):
        return 0.0

    if math.isnan(val) or math.isinf(val):
        return 0.0

    m_name = str(metric_name).lower().strip()

    # Percentage metrics (e.g. accuracy_pct, directional_acc_pct, win_rate)
    if "pct" in m_name or "percent" in m_name or "win_rate" in m_name or val > 1.0:
        if val <= 1.0 and ("auc" in m_name or "score" in m_name):
            return max(0.0, min(1.0, val))
        return max(0.0, min(1.0, val / 100.0))

    # Lower-is-better error metrics (e.g. rmse, mae, mse)
    if any(k in m_name for k in ("rmse", "mae", "mse", "error")):
        t_max = max(1e-6, float(tau_max))
        return max(0.0, min(1.0, 1.0 - (val / t_max)))

    # Lower-is-better loss/calibration metrics (e.g. log_loss, brier)
    if any(k in m_name for k in ("log_loss", "brier", "loss")):
        return max(0.0, min(1.0, 1.0 - val))

    # Standard higher-is-better bounded metrics (e.g. roc_auc, pr_auc, f1, precision, recall)
    return max(0.0, min(1.0, val))


def compute_robustness_score(
    primary_metric_name: str,
    primary_metric_value: float,
    *,
    fold_mean: float,
    fold_std: float,
    worst_fold_drawdown: float | None = None,
    ece: float | None = None,
    avg_regime_degradation_pct: float | None = None,
    total_features: int = 30,
    experimental_dependency_ratio: float = 0.0,
    deprecated_feature_count: int = 0,
    is_classification: bool = True,
    wf_folds_count: int = 5,
    policy: RobustnessRankingPolicy = ROB_POLICY_v1_0,
) -> tuple[float, dict[str, float], list[str]]:
    """Compute deterministic multi-factor robustness score and detailed penalty breakdown.
    
    Returns:
        (robustness_score, score_breakdown_dict, warnings_list)
    """
    warnings: list[str] = []

    # Check for NaN / Inf in inputs
    for name, val in [("primary_metric_value", primary_metric_value), ("fold_mean", fold_mean), ("fold_std", fold_std)]:
        if val is None or math.isnan(float(val)) or math.isinf(float(val)):
            return (
                0.0,
                {
                    "base_performance_contribution": 0.0,
                    "fold_variance_penalty": 0.0,
                    "worst_fold_penalty": 0.0,
                    "calibration_penalty": 0.0,
                    "regime_degradation_penalty": 0.0,
                    "experimental_risk_penalty": 0.0,
                    "parsimony_penalty": 0.0,
                },
                ["REJECTED_INVALID_METRICS"],
            )

    # 1. Base Score Contribution
    norm_metric = normalize_metric(primary_metric_name, primary_metric_value, tau_max=policy.tau_max_error)
    s_base = 100.0 * norm_metric

    # 2. Walk-Forward Fold Variance Penalty
    if int(wf_folds_count) < 2:
        p_sigma = 0.0
        warnings.append("MISSING_FOLDS_WARNING")
    else:
        denom = max(1e-6, abs(float(fold_mean)))
        p_sigma = policy.lambda_sigma * (float(fold_std) / denom)

    # 3. Worst-Fold Drawdown Penalty
    if worst_fold_drawdown is not None and not math.isnan(float(worst_fold_drawdown)):
        dd = float(worst_fold_drawdown)
        p_worst = policy.lambda_w * max(0.0, dd - policy.tau_safe)
    else:
        p_worst = 0.0

    # 4. Calibration Penalty (ECE)
    if is_classification and ece is not None and not math.isnan(float(ece)):
        p_calib = policy.lambda_c * max(0.0, float(ece))
    else:
        p_calib = 0.0

    # 5. Cross-Regime Degradation Penalty
    if avg_regime_degradation_pct is not None and not math.isnan(float(avg_regime_degradation_pct)):
        p_deg = policy.lambda_deg * max(0.0, float(avg_regime_degradation_pct))
    else:
        p_deg = 0.0
        warnings.append("NO_REGIME_STRESS_EVALUATION")

    # 6. Experimental Feature Risk & Deprecated Exposure Penalty
    exp_ratio = max(0.0, min(1.0, float(experimental_dependency_ratio)))
    dep_count = int(deprecated_feature_count)
    p_exp = (policy.lambda_e * exp_ratio) + (policy.omega_dep if dep_count > 0 else 0.0)
    if dep_count > 0:
        warnings.append(f"DEPRECATED_FEATURE_EXPOSURE_COUNT_{dep_count}")

    # 7. Parsimony / Complexity Penalty
    n_feats = max(1, int(total_features))
    n_base = max(1, int(policy.N_baseline))
    p_N = policy.lambda_N * math.log(max(1.0, n_feats / n_base))

    # Total Robustness Score Calculation
    raw_robustness = s_base - p_sigma - p_worst - p_calib - p_deg - p_exp - p_N
    clamped_score = max(0.0, min(100.0, raw_robustness))

    breakdown = {
        "base_performance_contribution": round(s_base, 4),
        "fold_variance_penalty": round(-p_sigma, 4),
        "worst_fold_penalty": round(-p_worst, 4),
        "calibration_penalty": round(-p_calib, 4),
        "regime_degradation_penalty": round(-p_deg, 4),
        "experimental_risk_penalty": round(-p_exp, 4),
        "parsimony_penalty": round(-p_N, 4),
    }

    return (round(clamped_score, 4), breakdown, warnings)


def compute_pareto_frontier(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Compute non-dominated Pareto ranking across 4 orthogonal objective dimensions:
    
    1. Base Performance: `s_base` (maximize)
    2. Negative Fold Variance: `-p_sigma` (maximize / minimize variance)
    3. Negative Regime Degradation: `-p_deg` (maximize / minimize degradation)
    4. Negative Feature Count: `-total_features` (maximize / minimize complexity)
    """
    if not candidates:
        return []

    # Extract objective vectors
    objs: list[tuple[float, float, float, float]] = []
    for c in candidates:
        sb = float(c.get("score_breakdown", {}).get("base_performance_contribution", 0.0))
        pv = float(c.get("score_breakdown", {}).get("fold_variance_penalty", 0.0)) # Already negative
        pd = float(c.get("score_breakdown", {}).get("regime_degradation_penalty", 0.0)) # Already negative
        nf = -float(c.get("raw_metrics_summary", {}).get("total_features", 30))
        objs.append((sb, pv, pd, nf))

    n = len(candidates)
    pareto_ranks = [0] * n
    current_rank = 1
    remaining_indices = set(range(n))

    while remaining_indices:
        non_dominated: set[int] = set()
        for i in remaining_indices:
            dominated = False
            for j in remaining_indices:
                if i == j:
                    continue
                # Check if j dominates i
                # j dominates i if j >= i on all 4 objectives and strictly > on at least one
                all_gte = (
                    objs[j][0] >= objs[i][0]
                    and objs[j][1] >= objs[i][1]
                    and objs[j][2] >= objs[i][2]
                    and objs[j][3] >= objs[i][3]
                )
                any_gt = (
                    objs[j][0] > objs[i][0]
                    or objs[j][1] > objs[i][1]
                    or objs[j][2] > objs[i][2]
                    or objs[j][3] > objs[i][3]
                )
                if all_gte and any_gt:
                    dominated = True
                    break
            if not dominated:
                non_dominated.add(i)

        if not non_dominated:
            # Fallback for remaining ties
            for idx in remaining_indices:
                pareto_ranks[idx] = current_rank
            break

        for idx in non_dominated:
            pareto_ranks[idx] = current_rank
        remaining_indices -= non_dominated
        current_rank += 1

    # Attach Pareto fields to candidates
    results: list[dict[str, Any]] = []
    for idx, c in enumerate(candidates):
        cand_copy = dict(c)
        r = pareto_ranks[idx]
        cand_copy["pareto_rank"] = r
        cand_copy["is_pareto_optimal"] = (r == 1)
        results.append(cand_copy)

    return results


def rank_models_in_context(
    data_dir: str,
    context_key: str,
    *,
    benchmark_run_id: str | None = None,
    policy: RobustnessRankingPolicy = ROB_POLICY_v1_0,
) -> list[dict[str, Any]]:
    """Evaluate and rank all models for a specific `ModelContextKey`.
    
    Returns a sorted list of explainable Ranking Dossiers.
    """
    init_analysis_db(data_dir)
    norm_key = str(context_key).strip()

    conn = connect_analysis_db(data_dir)
    try:
        query = "SELECT * FROM model_benchmarks WHERE context_key = ?"
        params: list[Any] = [norm_key]
        if benchmark_run_id:
            query += " AND benchmark_run_id = ?"
            params.append(str(benchmark_run_id).strip())

        bm_rows = conn.execute(query, tuple(params)).fetchall()
        if not bm_rows:
            return []

        raw_candidates: list[dict[str, Any]] = []
        is_cls = "classifier" in norm_key.lower() or "direction" in norm_key.lower() or "regime" in norm_key.lower()

        for bm in bm_rows:
            sig_hash = bm["signature_hash"]
            model_name = bm["model_name"]

            # Query feature composition
            f_eval = get_feature_set_evaluation(data_dir, sig_hash)
            tot_feats = f_eval["total_features"] if f_eval else bm["feature_count"]
            exp_ratio = f_eval["experimental_dependency_ratio"] if f_eval else 0.0
            dep_count = f_eval["deprecated_feature_count"] if f_eval else 0

            # Query regime evaluations for cross-regime degradation
            reg_evals = get_regime_evaluations_for_model(data_dir, sig_hash)
            stress_evals = [r for r in reg_evals if not r["is_native_regime"]]
            avg_deg = (
                sum(r["regime_degradation_pct"] for r in stress_evals) / len(stress_evals)
                if stress_evals
                else None
            )

            # Compute robustness score
            score, breakdown, warnings = compute_robustness_score(
                primary_metric_name=bm["primary_metric_name"],
                primary_metric_value=bm["primary_metric_value"],
                fold_mean=bm["fold_metric_mean"],
                fold_std=bm["fold_metric_std"],
                worst_fold_drawdown=bm["worst_fold_drawdown"],
                ece=bm["expected_calibration_error"],
                avg_regime_degradation_pct=avg_deg,
                total_features=tot_feats,
                experimental_dependency_ratio=exp_ratio,
                deprecated_feature_count=dep_count,
                is_classification=is_cls,
                wf_folds_count=bm["wf_folds_count"],
                policy=policy,
            )

            dossier: dict[str, Any] = {
                "benchmark_id": bm["benchmark_id"],
                "benchmark_run_id": bm["benchmark_run_id"],
                "context_key": norm_key,
                "model_name": model_name,
                "signature_hash": sig_hash,
                "algorithm": bm["algorithm"],
                "ranking_policy_version": policy.policy_version,
                "ranking_policy_hash": policy.compute_policy_hash(),
                "robustness_score": score,
                "score_breakdown": breakdown,
                "raw_metrics_summary": {
                    "primary_metric_name": bm["primary_metric_name"],
                    "primary_metric_value": bm["primary_metric_value"],
                    "fold_mean": bm["fold_metric_mean"],
                    "fold_std": bm["fold_metric_std"],
                    "worst_fold_drawdown": bm["worst_fold_drawdown"],
                    "expected_calibration_error": bm["expected_calibration_error"],
                    "avg_regime_degradation_pct": avg_deg,
                    "total_features": tot_feats,
                    "experimental_dependency_ratio": exp_ratio,
                    "deprecated_feature_count": dep_count,
                    "model_size_bytes": bm["model_size_bytes"],
                },
                "warnings": warnings,
            }
            raw_candidates.append(dossier)

        # Compute Pareto non-dominated sorting
        pareto_candidates = compute_pareto_frontier(raw_candidates)

        # Deterministic 5-Level Tie-Breaking Sort
        def _sort_key(c: dict[str, Any]) -> tuple:
            score = round(float(c["robustness_score"]), 3)
            p_rank = int(c.get("pareto_rank", 999))
            f_std = round(float(c["raw_metrics_summary"]["fold_std"] or 0.0), 6)
            exp_r = round(float(c["raw_metrics_summary"]["experimental_dependency_ratio"] or 0.0), 6)
            tot_f = int(c["raw_metrics_summary"]["total_features"] or 0)
            sz = int(c["raw_metrics_summary"]["model_size_bytes"] or 0)
            sig = str(c["signature_hash"])
            # Higher score first (-score), lower pareto rank first, lower fold std first,
            # lower exp ratio first, lower features count first, lower size first, stable signature hash
            return (-score, p_rank, f_std, exp_r, tot_f, sz, sig)

        sorted_candidates = sorted(pareto_candidates, key=_sort_key)

        # Assign final rank_in_context and recommendation_status
        for rank_idx, cand in enumerate(sorted_candidates, start=1):
            cand["rank_in_context"] = rank_idx
            if "REJECTED_INVALID_METRICS" in cand.get("warnings", []):
                cand["recommendation_status"] = "REJECTED_INVALID_METRICS"
            elif rank_idx == 1:
                cand["recommendation_status"] = "CHAMPION_CANDIDATE"
            elif rank_idx in (2, 3):
                cand["recommendation_status"] = "CHALLENGER_CANDIDATE"
            else:
                cand["recommendation_status"] = "BENCHMARKED"

        return sorted_candidates
    finally:
        conn.close()


def persist_context_rankings(
    data_dir: str,
    *,
    benchmark_run_id: str,
    ranked_dossiers: list[dict[str, Any]],
    policy: RobustnessRankingPolicy = ROB_POLICY_v1_0,
) -> int:
    """Persist calculated robustness scores, ranks, and recommendations into `analysis.db`.
    
    Returns:
        Number of model benchmarks updated.
    """
    init_analysis_db(data_dir)
    updated_count = 0
    top_model = ranked_dossiers[0]["model_name"] if ranked_dossiers else None

    conn = connect_analysis_db(data_dir)
    try:
        with conn:
            for d in ranked_dossiers:
                bm_id = d.get("benchmark_id")
                if not bm_id:
                    continue
                conn.execute(
                    """
                    UPDATE model_benchmarks
                    SET robustness_score = ?,
                        rank_in_context = ?,
                        recommendation_status = ?,
                        ranking_policy_version = ?
                    WHERE benchmark_id = ?;
                    """,
                    (
                        float(d["robustness_score"]),
                        int(d["rank_in_context"]),
                        str(d["recommendation_status"]),
                        str(policy.policy_version),
                        int(bm_id),
                    ),
                )
                updated_count += 1

            # Update benchmark_runs with top model name & policy details
            policy_criteria = {
                "ranking_policy_version": policy.policy_version,
                "ranking_policy_hash": policy.compute_policy_hash(),
                "policy_parameters": policy.to_dict(),
                "ranked_at": _utc_now_iso(),
            }
            conn.execute(
                """
                UPDATE benchmark_runs
                SET top_model_name = ?,
                    ranking_policy_version = ?,
                    evaluation_criteria_json = ?
                WHERE benchmark_run_id = ?;
                """,
                (
                    top_model,
                    str(policy.policy_version),
                    json.dumps(policy_criteria, sort_keys=True),
                    str(benchmark_run_id).strip(),
                ),
            )
        return updated_count
    finally:
        conn.close()

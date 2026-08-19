"""Negative Evidence Pruning & Search Space Exclusion Engine (Phase 4E.4).

Advisory research filter that identifies search paths, experiment configurations,
feature combinations, and algorithms to avoid or treat with caution based on
accumulated empirical evidence in Research Memory.

Invariants:
1. Strictly Advisory: Zero automated deletion of experiments or mutation of registries/pipelines.
2. Deduplication Authority: Uses canonical experiment signatures and get_experiment_by_signature().
3. Evidence-Backed Chronic Failure: Requires at least 3 historical trials with mean robustness < 40.0.
4. Non-Exclusion of Unknowns: Unmapped/unknown features remain ELIGIBLE by default.
5. Absence of Evidence != Negative Evidence: Unobserved/cold-start spaces default to ELIGIBLE.
6. Deterministic Precedence:
   DUPLICATE_EXPERIMENT -> DEPRECATED_FEATURE -> CHRONIC_LOW_ROBUSTNESS ->
   EXTREME_REGIME_FRAGILITY -> SEVERE_MISCALIBRATION -> NONE.
"""

from __future__ import annotations

import json
import math
import os
import sqlite3
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Sequence

from chain_replay_ml.dataset_builder.schema_registry import load_schema_registry
from chain_replay_ml.model_taxonomy.enums import (
    BASELINE_REGIME_CATALOG,
    DEFAULT_REGIME_ID,
    DEFAULT_REGIME_NAME,
)
from chain_replay_ml.model_taxonomy.specs import ModelContextKey
from chain_replay_ml.research_memory.benchmarks import get_model_benchmarks_for_context
from chain_replay_ml.research_memory.db import connect_analysis_db, init_analysis_db
from chain_replay_ml.research_memory.feature_comp import classify_feature_population
from chain_replay_ml.research_memory.regime_eval import get_regime_evaluations_for_model
from chain_replay_ml.research_memory.signature import (
    canonical_context_key,
    compute_experiment_signature,
    get_experiment_by_signature,
    list_experiments_for_context,
)
from chain_replay_ml.research_recommendations.coverage import (
    CoverageClass,
    analyze_context_coverage,
)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class ExclusionVerdict(str, Enum):
    """Advisory decision for a proposed experiment or search space candidate."""

    EXCLUDED = "EXCLUDED"  # Definitive recommendation to skip (saves compute / prevents known failure)
    CAUTION = "CAUTION"    # High risk (extreme fragility or miscalibration); proceed with safeguards
    ELIGIBLE = "ELIGIBLE"  # Valid search candidate (supported by positive evidence or unobserved)

    @classmethod
    def from_str(cls, value: str | Any) -> ExclusionVerdict:
        raw = str(value or "").strip().upper()
        try:
            return cls(raw)
        except ValueError:
            pass
        mapping = {
            "PRUNED": cls.EXCLUDED,
            "SKIP": cls.EXCLUDED,
            "REJECT": cls.EXCLUDED,
            "WARN": cls.CAUTION,
            "WARNING": cls.CAUTION,
            "RISK": cls.CAUTION,
            "VALID": cls.ELIGIBLE,
            "PASS": cls.ELIGIBLE,
            "ALLOW": cls.ELIGIBLE,
        }
        if raw in mapping:
            return mapping[raw]
        return cls.ELIGIBLE


class ExclusionReason(str, Enum):
    """Specific empirical negative-evidence rationale for pruning or caution."""

    DUPLICATE_EXPERIMENT = "DUPLICATE_EXPERIMENT"
    DEPRECATED_FEATURE = "DEPRECATED_FEATURE"
    CHRONIC_LOW_ROBUSTNESS = "CHRONIC_LOW_ROBUSTNESS"
    EXTREME_REGIME_FRAGILITY = "EXTREME_REGIME_FRAGILITY"
    SEVERE_MISCALIBRATION = "SEVERE_MISCALIBRATION"
    NONE = "NONE"

    @classmethod
    def from_str(cls, value: str | Any) -> ExclusionReason:
        raw = str(value or "").strip().upper()
        try:
            return cls(raw)
        except ValueError:
            pass
        return cls.NONE


@dataclass(frozen=True)
class PruningAuditResult:
    """Detailed negative-evidence audit outcome for a specific experiment specification or signature."""

    context_key: str
    signature_hash: str
    verdict: ExclusionVerdict
    primary_reason: ExclusionReason
    exclusion_reasons: list[ExclusionReason]
    flagged_features: list[str]
    evidence_summary: str
    historical_benchmark_count: int
    mean_historical_robustness: float | None
    max_historical_degradation: float | None
    worst_historical_ece: float | None
    generated_at: str

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["verdict"] = self.verdict.value
        d["primary_reason"] = self.primary_reason.value
        d["exclusion_reasons"] = [r.value for r in self.exclusion_reasons]
        return d


@dataclass(frozen=True)
class ContextPruningAgenda:
    """Aggregated context-wide negative evidence and search space pruning summary."""

    context_key: str
    market: str
    sampling_interval_sec: int
    task_type: str
    prediction_horizon: str
    regime_id: str
    coverage_class: str
    quarantined_features: list[str]
    fragile_features: list[str]
    chronically_failing_combinations: list[list[str]]
    quarantined_algorithms: list[str]
    total_explored_signatures: int
    pruned_signatures_count: int
    caution_signatures_count: int
    eligible_signatures_count: int
    generated_at: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "context_key": self.context_key,
            "market": self.market,
            "sampling_interval_sec": self.sampling_interval_sec,
            "task_type": self.task_type,
            "prediction_horizon": self.prediction_horizon,
            "regime_id": self.regime_id,
            "coverage_class": self.coverage_class,
            "quarantined_features": self.quarantined_features,
            "fragile_features": self.fragile_features,
            "chronically_failing_combinations": self.chronically_failing_combinations,
            "quarantined_algorithms": self.quarantined_algorithms,
            "total_explored_signatures": self.total_explored_signatures,
            "pruned_signatures_count": self.pruned_signatures_count,
            "caution_signatures_count": self.caution_signatures_count,
            "eligible_signatures_count": self.eligible_signatures_count,
            "generated_at": self.generated_at,
        }


def audit_experiment_exclusion(
    data_dir: str,
    experiment_spec: dict[str, Any],
    *,
    schema: dict[str, Any] | None = None,
) -> PruningAuditResult:
    """Audit a proposed experiment specification against negative evidence and duplication rules.
    
    Deterministic Precedence:
        1. DUPLICATE_EXPERIMENT (EXCLUDED)
        2. DEPRECATED_FEATURE (EXCLUDED)
        3. CHRONIC_LOW_ROBUSTNESS (EXCLUDED)
        4. EXTREME_REGIME_FRAGILITY (CAUTION)
        5. SEVERE_MISCALIBRATION (CAUTION)
        6. NONE (ELIGIBLE)
    """
    init_analysis_db(data_dir)
    sig_hash, canonical_json, norm_payload = compute_experiment_signature(experiment_spec)
    c_key_str = norm_payload["context_key"]
    features = norm_payload.get("features", [])
    algo = norm_payload.get("algorithm", "xgboost")

    reasons: list[ExclusionReason] = []
    flagged_features: list[str] = []

    # 1. Exact Duplicate Check (via authoritative get_experiment_by_signature)
    existing_rec = get_experiment_by_signature(data_dir, sig_hash)
    if existing_rec is not None:
        reasons.append(ExclusionReason.DUPLICATE_EXPERIMENT)

    # 2. Deprecated Feature Check (via Schema Registry)
    reg_schema = schema if schema is not None else load_schema_registry()
    for f in features:
        pop = classify_feature_population(f, schema=reg_schema)
        if pop == "DEPRECATED":
            reasons.append(ExclusionReason.DEPRECATED_FEATURE)
            flagged_features.append(f)

    # 3. Context Historical Benchmarks Query
    conn = connect_analysis_db(data_dir)
    context_experiments: list[dict[str, Any]] = []
    try:
        rows = conn.execute(
            """
            SELECT s.signature_hash, s.canonical_payload_json, s.algorithm,
                   b.robustness_score, b.expected_calibration_error
            FROM experiment_signatures s
            LEFT JOIN model_benchmarks b ON s.signature_hash = b.signature_hash
            WHERE s.context_key = ?;
            """,
            (c_key_str,),
        ).fetchall()
        for r in rows:
            p_json = json.loads(r["canonical_payload_json"]) if r["canonical_payload_json"] else {}
            context_experiments.append({
                "signature_hash": r["signature_hash"],
                "algorithm": r["algorithm"],
                "features": p_json.get("features", []),
                "robustness_score": float(r["robustness_score"]) if r["robustness_score"] is not None else None,
                "ece": float(r["expected_calibration_error"]) if r["expected_calibration_error"] is not None else None,
            })
    finally:
        conn.close()

    # Find matching experiments for this exact feature combination in context
    prop_feats_set = set(features)
    matching_trials = [
        exp for exp in context_experiments
        if set(exp.get("features", [])) == prop_feats_set
    ]

    hist_count = len(matching_trials)
    rob_scores = [t["robustness_score"] for t in matching_trials if t["robustness_score"] is not None]
    ece_scores = [t["ece"] for t in matching_trials if t["ece"] is not None]

    mean_rob = round(sum(rob_scores) / len(rob_scores), 4) if rob_scores else None
    worst_ece = max(ece_scores) if ece_scores else None

    # Check for Chronic Low Robustness (at least 3 relevant trials AND mean robustness < 40.0)
    if len(rob_scores) >= 3 and mean_rob is not None and mean_rob < 40.0:
        reasons.append(ExclusionReason.CHRONIC_LOW_ROBUSTNESS)

    # 4. Cross-Regime Degradation Analysis for these signatures
    degradations: list[float] = []
    for t in matching_trials:
        reg_evals = get_regime_evaluations_for_model(data_dir, t["signature_hash"])
        for r in reg_evals:
            if not r.get("is_native_regime", False):
                deg = r.get("regime_degradation_pct")
                if deg is not None:
                    degradations.append(float(deg))

    max_deg = max(degradations) if degradations else None

    # Check for Extreme Regime Fragility (> 30.0% degradation)
    if max_deg is not None and max_deg > 30.0:
        reasons.append(ExclusionReason.EXTREME_REGIME_FRAGILITY)

    # 5. Check for Severe Miscalibration (ECE >= 0.10)
    if worst_ece is not None and worst_ece >= 0.10:
        reasons.append(ExclusionReason.SEVERE_MISCALIBRATION)

    # Determine Verdict & Primary Reason via deterministic precedence
    if ExclusionReason.DUPLICATE_EXPERIMENT in reasons:
        verdict = ExclusionVerdict.EXCLUDED
        primary_reason = ExclusionReason.DUPLICATE_EXPERIMENT
        evidence_summary = f"Experiment signature '{sig_hash[:12]}' already exists in context '{c_key_str}' (Duplicate suppression)."
    elif ExclusionReason.DEPRECATED_FEATURE in reasons:
        verdict = ExclusionVerdict.EXCLUDED
        primary_reason = ExclusionReason.DEPRECATED_FEATURE
        evidence_summary = f"Contains deprecated/retired feature(s): {', '.join(sorted(list(set(flagged_features))))}."
    elif ExclusionReason.CHRONIC_LOW_ROBUSTNESS in reasons:
        verdict = ExclusionVerdict.EXCLUDED
        primary_reason = ExclusionReason.CHRONIC_LOW_ROBUSTNESS
        evidence_summary = f"Feature combination chronically weak across {len(rob_scores)} trials (Mean RobScore: {mean_rob:.2f} < 40.0)."
    elif ExclusionReason.EXTREME_REGIME_FRAGILITY in reasons:
        verdict = ExclusionVerdict.CAUTION
        primary_reason = ExclusionReason.EXTREME_REGIME_FRAGILITY
        evidence_summary = f"High cross-regime fragility detected (Max non-native degradation: {max_deg:.2f}% > 30%)."
    elif ExclusionReason.SEVERE_MISCALIBRATION in reasons:
        verdict = ExclusionVerdict.CAUTION
        primary_reason = ExclusionReason.SEVERE_MISCALIBRATION
        evidence_summary = f"Severe miscalibration detected (ECE: {worst_ece:.4f} >= 0.1000)."
    else:
        verdict = ExclusionVerdict.ELIGIBLE
        primary_reason = ExclusionReason.NONE
        evidence_summary = "Candidate specification is valid and eligible for exploration."

    return PruningAuditResult(
        context_key=c_key_str,
        signature_hash=sig_hash,
        verdict=verdict,
        primary_reason=primary_reason,
        exclusion_reasons=reasons if reasons else [ExclusionReason.NONE],
        flagged_features=sorted(list(set(flagged_features))),
        evidence_summary=evidence_summary,
        historical_benchmark_count=hist_count,
        mean_historical_robustness=mean_rob,
        max_historical_degradation=max_deg,
        worst_historical_ece=worst_ece,
        generated_at=_utc_now_iso(),
    )


def audit_signature_exclusion(
    data_dir: str,
    signature_hash: str,
    *,
    schema: dict[str, Any] | None = None,
) -> PruningAuditResult:
    """Audit an existing or proposed signature hash against negative evidence rules."""
    init_analysis_db(data_dir)
    sig_h = str(signature_hash).strip()
    rec = get_experiment_by_signature(data_dir, sig_h)
    if not rec:
        # Unknown signature without payload
        return PruningAuditResult(
            context_key="UNKNOWN",
            signature_hash=sig_h,
            verdict=ExclusionVerdict.ELIGIBLE,
            primary_reason=ExclusionReason.NONE,
            exclusion_reasons=[ExclusionReason.NONE],
            flagged_features=[],
            evidence_summary="Signature not found in database; eligible as unobserved candidate.",
            historical_benchmark_count=0,
            mean_historical_robustness=None,
            max_historical_degradation=None,
            worst_historical_ece=None,
            generated_at=_utc_now_iso(),
        )

    raw_json = rec.get("canonical_payload_json")
    spec = json.loads(raw_json) if raw_json else {}
    return audit_experiment_exclusion(data_dir, spec, schema=schema)


def is_search_path_excluded(
    data_dir: str,
    experiment_spec: dict[str, Any],
    *,
    allow_caution: bool = True,
    schema: dict[str, Any] | None = None,
) -> bool:
    """Convenience boolean helper indicating whether an experiment should be skipped.
    
    If allow_caution is True: Only EXCLUDED returns True (CAUTION is permitted).
    If allow_caution is False: Both EXCLUDED and CAUTION return True.
    """
    res = audit_experiment_exclusion(data_dir, experiment_spec, schema=schema)
    if res.verdict == ExclusionVerdict.EXCLUDED:
        return True
    if not allow_caution and res.verdict == ExclusionVerdict.CAUTION:
        return True
    return False


def build_context_pruning_agenda(
    data_dir: str,
    context_key: str,
    *,
    schema: dict[str, Any] | None = None,
) -> ContextPruningAgenda:
    """Construct an aggregated negative-evidence and search space pruning summary for a context."""
    init_analysis_db(data_dir)
    c_key_str = str(context_key).strip()

    ctx_obj = ModelContextKey.from_key_str(c_key_str)
    market = ctx_obj.market
    sampling_sec = ctx_obj.sampling_interval_sec
    task_type = ctx_obj.task_type.value if hasattr(ctx_obj.task_type, "value") else str(ctx_obj.task_type)
    horizon = ctx_obj.prediction_horizon
    regime_id = ctx_obj.regime_id

    # Coverage profile
    cov = analyze_context_coverage(data_dir, c_key_str)
    cov_class = cov.coverage_class.value

    # Query all signatures in this context
    context_sigs = list_experiments_for_context(data_dir, c_key_str)
    reg_schema = schema if schema is not None else load_schema_registry()
    cols = reg_schema.get("columns") or {}

    # Discover quarantined / deprecated features
    quarantined_feats: set[str] = set()
    for f, meta in cols.items():
        if isinstance(meta, dict):
            stat = str(meta.get("status") or "").upper().strip()
            if stat in ("DEPRECATED", "RETIRED", "DISABLED"):
                quarantined_feats.add(f)

    # Evaluate all context signatures
    pruned_cnt = 0
    caution_cnt = 0
    eligible_cnt = 0
    fragile_feats: set[str] = set()
    chronically_failing_combs: list[list[str]] = []
    quarantined_algos: set[str] = set()

    for sig_rec in context_sigs:
        raw_json = sig_rec.get("canonical_payload_json")
        if raw_json:
            spec = json.loads(raw_json)
            res = audit_experiment_exclusion(data_dir, spec, schema=reg_schema)
            if res.verdict == ExclusionVerdict.EXCLUDED:
                pruned_cnt += 1
                if res.primary_reason == ExclusionReason.CHRONIC_LOW_ROBUSTNESS:
                    feats = sorted(spec.get("features", []))
                    if feats and feats not in chronically_failing_combs:
                        chronically_failing_combs.append(feats)
            elif res.verdict == ExclusionVerdict.CAUTION:
                caution_cnt += 1
                if res.primary_reason == ExclusionReason.EXTREME_REGIME_FRAGILITY:
                    for f in spec.get("features", []):
                        fragile_feats.add(f)
            else:
                eligible_cnt += 1

    return ContextPruningAgenda(
        context_key=c_key_str,
        market=market,
        sampling_interval_sec=sampling_sec,
        task_type=task_type,
        prediction_horizon=horizon,
        regime_id=regime_id,
        coverage_class=cov_class,
        quarantined_features=sorted(list(quarantined_feats)),
        fragile_features=sorted(list(fragile_feats)),
        chronically_failing_combinations=chronically_failing_combs,
        quarantined_algorithms=sorted(list(quarantined_algos)),
        total_explored_signatures=len(context_sigs),
        pruned_signatures_count=pruned_cnt,
        caution_signatures_count=caution_cnt,
        eligible_signatures_count=eligible_cnt,
        generated_at=_utc_now_iso(),
    )

"""Multi-Objective Recommendation Priority Scoring Engine (Phase 4E.5).

Synthesizes context coverage, production champion vulnerability, feature affinity,
interaction evidence, challenger gaps, and negative pruning into a deterministic,
decomposable, and advisory Research Priority Agenda.

Invariants:
1. Strictly Advisory: Zero automated model promotion, demotion, deployment, or training.
2. Context Isolation: Scored strictly within canonical 5-dimensional ModelContextKey.
3. Negative Evidence Suppression: Suppresses EXCLUDED search spaces; propagates CAUTION.
4. Non-Causal Empirical Labeling: Treats all empirical lifts and historical metrics as observational evidence.
5. Deterministic Tie-Breaking: -priority_score, -champ_vuln, -chal_gap, -feat_aff, +cov_density, context_key, opp_id.
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
from chain_replay_ml.research_memory.ranking import rank_models_in_context
from chain_replay_ml.research_recommendations.coverage import (
    ContextCoverage,
    CoverageClass,
    analyze_context_coverage,
)
from chain_replay_ml.research_recommendations.feature_affinity import (
    ContextFeatureAffinityReport,
    FeatureAffinityResult,
    FeatureInteractionResult,
    FeatureRecommendationClass,
    analyze_feature_affinity,
    recommend_features_for_context,
)
from chain_replay_ml.research_recommendations.negative_pruning import (
    ExclusionReason,
    ExclusionVerdict,
    PruningAuditResult,
    audit_experiment_exclusion,
)
from chain_replay_ml.research_recommendations.vulnerability import (
    ChallengerStatus,
    ChampionVulnerabilityResult,
    VulnerabilityClass,
    audit_champion_vulnerability,
)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _clean_float(val: Any, default: float) -> float:
    if val is None:
        return default
    try:
        f = float(val)
        if math.isnan(f):
            return default
        if math.isinf(f):
            return 1e9 if f > 0 else -1e9
        return f
    except (ValueError, TypeError):
        return default


class ResearchPriorityClass(str, Enum):
    """Categorization of research urgency and potential impact."""

    CRITICAL = "CRITICAL"    # Immediate research needed (Fragile Champion, Leading Challenger, or severe gap)
    HIGH = "HIGH"            # Substantial potential impact (Confirmed feature upgrade or high-leverage context)
    MEDIUM = "MEDIUM"        # Standard research priority (Promising candidate exploration)
    LOW = "LOW"              # Minor exploration value (Established stability or marginal lift)
    NEGLIGIBLE = "NEGLIGIBLE"# Very low priority (Saturated coverage with no vulnerability)

    @classmethod
    def from_str(cls, value: str | Any) -> ResearchPriorityClass:
        raw = str(value or "").strip().upper()
        try:
            return cls(raw)
        except ValueError:
            pass
        mapping = {
            "URGENT": cls.CRITICAL,
            "VERY_HIGH": cls.CRITICAL,
            "DEFENSE": cls.CRITICAL,
            "MAJOR": cls.HIGH,
            "MODERATE": cls.MEDIUM,
            "NORMAL": cls.MEDIUM,
            "MINOR": cls.LOW,
            "MAINTENANCE": cls.LOW,
            "NONE": cls.NEGLIGIBLE,
            "ZERO": cls.NEGLIGIBLE,
        }
        if raw in mapping:
            return mapping[raw]
        return cls.MEDIUM


class OpportunityType(str, Enum):
    """Functional research intent of the opportunity."""

    CHAMPION_VULNERABILITY_DEFENSE = "CHAMPION_VULNERABILITY_DEFENSE"
    CHALLENGER_OPPORTUNITY = "CHALLENGER_OPPORTUNITY"
    FEATURE_EXPLORATION = "FEATURE_EXPLORATION"
    INTERACTION_VALIDATION = "INTERACTION_VALIDATION"
    COVERAGE_EXPANSION = "COVERAGE_EXPANSION"

    @classmethod
    def from_str(cls, value: str | Any) -> OpportunityType:
        raw = str(value or "").strip().upper()
        try:
            return cls(raw)
        except ValueError:
            pass
        return cls.FEATURE_EXPLORATION


class EvidenceConfidenceLevel(str, Enum):
    """Categorical confidence in the empirical research evidence base."""

    STRONG = "STRONG"              # C >= 0.70 (Mature empirical volume across multiple benchmarks)
    MODERATE = "MODERATE"          # 0.40 <= C < 0.70 (Developing empirical volume)
    WEAK = "WEAK"                  # 0.15 <= C < 0.40 (Sparse empirical volume)
    INSUFFICIENT = "INSUFFICIENT"  # C < 0.15 (Cold-start / unobserved)

    @classmethod
    def from_str(cls, value: str | Any) -> EvidenceConfidenceLevel:
        raw = str(value or "").strip().upper()
        try:
            return cls(raw)
        except ValueError:
            pass
        return cls.WEAK


@dataclass(frozen=True)
class ComponentScoreBreakdown:
    """Transparent multi-objective score breakdown."""

    coverage_gap_score: float
    champion_vulnerability_score: float
    challenger_gap_score: float
    feature_affinity_score: float
    interaction_synergy_score: float
    caution_penalty: float
    raw_composite_score: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ResearchOpportunity:
    """A prioritized, actionable research proposal for a ModelContextKey."""

    opportunity_id: str
    context_key: str
    opportunity_type: OpportunityType
    priority_class: ResearchPriorityClass
    priority_score: float
    evidence_confidence: EvidenceConfidenceLevel
    confidence_value: float
    exclusion_verdict: ExclusionVerdict
    exclusion_reason: ExclusionReason
    component_breakdown: ComponentScoreBreakdown
    candidate_features: list[str]
    target_algorithm: str
    rationale: str
    recommended_action: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "opportunity_id": self.opportunity_id,
            "context_key": self.context_key,
            "opportunity_type": self.opportunity_type.value,
            "priority_class": self.priority_class.value,
            "priority_score": self.priority_score,
            "evidence_confidence": self.evidence_confidence.value,
            "confidence_value": self.confidence_value,
            "exclusion_verdict": self.exclusion_verdict.value,
            "exclusion_reason": self.exclusion_reason.value,
            "component_breakdown": self.component_breakdown.to_dict(),
            "candidate_features": self.candidate_features,
            "target_algorithm": self.target_algorithm,
            "rationale": self.rationale,
            "recommended_action": self.recommended_action,
        }


@dataclass(frozen=True)
class ContextPriorityAgendaReport:
    """Aggregated, ranked research agenda report for a ModelContextKey."""

    context_key: str
    market: str
    sampling_interval_sec: int
    task_type: str
    prediction_horizon: str
    regime_id: str
    top_priority_class: ResearchPriorityClass
    total_opportunities_evaluated: int
    eligible_opportunities: list[ResearchOpportunity]
    caution_opportunities: list[ResearchOpportunity]
    suppressed_excluded_count: int
    generated_at: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "context_key": self.context_key,
            "market": self.market,
            "sampling_interval_sec": self.sampling_interval_sec,
            "task_type": self.task_type,
            "prediction_horizon": self.prediction_horizon,
            "regime_id": self.regime_id,
            "top_priority_class": self.top_priority_class.value,
            "total_opportunities_evaluated": self.total_opportunities_evaluated,
            "eligible_opportunities": [o.to_dict() for o in self.eligible_opportunities],
            "caution_opportunities": [o.to_dict() for o in self.caution_opportunities],
            "suppressed_excluded_count": self.suppressed_excluded_count,
            "generated_at": self.generated_at,
        }


def compute_research_priority_score(
    *,
    coverage_density_score: float = 0.0,
    champion_vulnerability_score: float = 0.0,
    challenger_gap_pct: float = 0.0,
    feature_affinity_score: float = 50.0,
    interaction_synergy_score: float = 50.0,
    is_caution: bool = False,
    is_excluded: bool = False,
) -> tuple[float, ComponentScoreBreakdown]:
    """Calculate the safe bounded Research Priority Score P in [0.0, 100.0]."""
    if is_excluded:
        breakdown = ComponentScoreBreakdown(
            coverage_gap_score=0.0,
            champion_vulnerability_score=0.0,
            challenger_gap_score=0.0,
            feature_affinity_score=0.0,
            interaction_synergy_score=0.0,
            caution_penalty=0.0,
            raw_composite_score=0.0,
        )
        return 0.0, breakdown

    s_cov_gap = min(100.0, max(0.0, 100.0 - _clean_float(coverage_density_score, 0.0)))
    s_champ_v = min(100.0, max(0.0, _clean_float(champion_vulnerability_score, 0.0)))
    
    c_gap = _clean_float(challenger_gap_pct, 0.0)
    s_chal_gap = min(100.0, max(0.0, 50.0 + (c_gap * 5.0)))

    s_feat_aff = min(100.0, max(0.0, _clean_float(feature_affinity_score, 50.0)))
    s_inter_s = min(100.0, max(0.0, _clean_float(interaction_synergy_score, 50.0)))

    raw_base = (
        (0.30 * s_champ_v)
        + (0.25 * s_chal_gap)
        + (0.20 * s_feat_aff)
        + (0.15 * s_cov_gap)
        + (0.10 * s_inter_s)
    )

    caution_pen = -15.0 if is_caution else 0.0
    final_p = round(min(100.0, max(0.0, raw_base + caution_pen)), 4)

    breakdown = ComponentScoreBreakdown(
        coverage_gap_score=round(s_cov_gap, 4),
        champion_vulnerability_score=round(s_champ_v, 4),
        challenger_gap_score=round(s_chal_gap, 4),
        feature_affinity_score=round(s_feat_aff, 4),
        interaction_synergy_score=round(s_inter_s, 4),
        caution_penalty=caution_pen,
        raw_composite_score=round(raw_base, 4),
    )
    return final_p, breakdown


def classify_priority_score(priority_score: float) -> ResearchPriorityClass:
    p = min(100.0, max(0.0, float(priority_score or 0.0)))
    if p >= 80.0:
        return ResearchPriorityClass.CRITICAL
    elif p >= 65.0:
        return ResearchPriorityClass.HIGH
    elif p >= 50.0:
        return ResearchPriorityClass.MEDIUM
    elif p >= 30.0:
        return ResearchPriorityClass.LOW
    else:
        return ResearchPriorityClass.NEGLIGIBLE


def classify_evidence_confidence(
    coverage_confidence: float,
    feature_confidence: float,
) -> tuple[EvidenceConfidenceLevel, float]:
    c_cov = min(1.0, max(0.0, _clean_float(coverage_confidence, 0.0)))
    c_feat = min(1.0, max(0.0, _clean_float(feature_confidence, 0.0)))

    comp_c = round((0.50 * c_cov) + (0.50 * c_feat), 4)

    if comp_c >= 0.70:
        level = EvidenceConfidenceLevel.STRONG
    elif comp_c >= 0.40:
        level = EvidenceConfidenceLevel.MODERATE
    elif comp_c >= 0.15:
        level = EvidenceConfidenceLevel.WEAK
    else:
        level = EvidenceConfidenceLevel.INSUFFICIENT

    return level, comp_c


def evaluate_research_opportunity(
    data_dir: str,
    context_key: str,
    *,
    opportunity_type: OpportunityType,
    candidate_features: list[str],
    target_algorithm: str = "xgboost",
    schema: dict[str, Any] | None = None,
) -> ResearchOpportunity:
    init_analysis_db(data_dir)
    c_key_str = str(context_key).strip()
    norm_feats = sorted(list(set(str(f).strip() for f in candidate_features if str(f).strip())))
    clean_algo = str(target_algorithm or "xgboost").lower().strip()

    ctx_obj = ModelContextKey.from_key_str(c_key_str)
    market = ctx_obj.market
    interval_sec = ctx_obj.sampling_interval_sec
    task_type = ctx_obj.task_type.value if hasattr(ctx_obj.task_type, "value") else str(ctx_obj.task_type)
    horizon = ctx_obj.prediction_horizon
    regime_id = ctx_obj.regime_id

    # 1. Check Negative Pruning (Phase 4E.4)
    experiment_spec = {
        "market": market,
        "sampling_interval_sec": interval_sec,
        "task_type": task_type,
        "prediction_horizon": horizon,
        "regime_id": regime_id,
        "features": norm_feats,
        "algorithm": clean_algo,
    }
    prune_res = audit_experiment_exclusion(data_dir, experiment_spec, schema=schema)
    is_exc = (prune_res.verdict == ExclusionVerdict.EXCLUDED)
    is_caut = (prune_res.verdict == ExclusionVerdict.CAUTION)

    # 2. Context Coverage (Phase 4E.1)
    cov = analyze_context_coverage(data_dir, c_key_str)
    cov_density = cov.evidence_density_score
    cov_conf = min(1.0, cov.benchmark_count / 10.0)

    # 3. Champion Vulnerability & Challenger Gap (Phase 4E.2)
    vuln = audit_champion_vulnerability(data_dir, c_key_str)
    champ_vuln_score = vuln.vulnerability_score * 100.0
    chal_gap_pct = vuln.challenger_gap or 0.0

    # 4. Feature Affinity & Interaction Lift (Phase 4E.3)
    feat_aff_scores: list[float] = []
    feat_confs: list[float] = []
    reg_schema = schema if schema is not None else load_schema_registry()

    for f in norm_feats:
        aff = analyze_feature_affinity(data_dir, c_key_str, f, schema=reg_schema)
        feat_aff_scores.append(aff.affinity_score)
        feat_confs.append(aff.confidence)

    mean_feat_aff = sum(feat_aff_scores) / len(feat_aff_scores) if feat_aff_scores else 50.0
    mean_feat_conf = sum(feat_confs) / len(feat_confs) if feat_confs else 0.0

    # Interaction score if multi-feature
    inter_score = 50.0
    if len(norm_feats) >= 2:
        aff_rep = recommend_features_for_context(data_dir, c_key_str, schema=reg_schema)
        prop_set = set(norm_feats)
        for inter in aff_rep.interaction_recommendations:
            if set(inter.feature_set).issubset(prop_set):
                inter_score = max(inter_score, inter.interaction_score)

    # 5. Composite Priority Scoring
    priority_score, breakdown = compute_research_priority_score(
        coverage_density_score=cov_density,
        champion_vulnerability_score=champ_vuln_score,
        challenger_gap_pct=chal_gap_pct,
        feature_affinity_score=mean_feat_aff,
        interaction_synergy_score=inter_score,
        is_caution=is_caut,
        is_excluded=is_exc,
    )

    priority_class = classify_priority_score(priority_score)
    conf_level, conf_val = classify_evidence_confidence(cov_conf, mean_feat_conf)

    # 6. Rationale & Recommendation
    feats_str = ", ".join(norm_feats) if norm_feats else "None"
    opp_id = f"OPP_{c_key_str}_{clean_algo}_{len(norm_feats)}F_{abs(hash(tuple(norm_feats))) % 100000:05d}"

    if is_exc:
        rationale = f"Suppressed due to negative evidence: {prune_res.primary_reason.value} ({prune_res.evidence_summary})"
        rec_action = "Do not schedule training. Excluded from research campaign."
    elif is_caut:
        rationale = f"Proceed with caution: {prune_res.primary_reason.value} ({prune_res.evidence_summary})"
        rec_action = "Schedule targeted stress-test campaign with regime safeguards."
    else:
        rationale = f"Eligible research proposal targeting {opportunity_type.value} with priority {priority_score:.2f}."
        rec_action = f"Schedule candidate evaluation campaign using algorithm {clean_algo} and features [{feats_str}]."

    return ResearchOpportunity(
        opportunity_id=opp_id,
        context_key=c_key_str,
        opportunity_type=opportunity_type,
        priority_class=priority_class,
        priority_score=priority_score,
        evidence_confidence=conf_level,
        confidence_value=conf_val,
        exclusion_verdict=prune_res.verdict,
        exclusion_reason=prune_res.primary_reason,
        component_breakdown=breakdown,
        candidate_features=norm_feats,
        target_algorithm=clean_algo,
        rationale=rationale,
        recommended_action=rec_action,
    )


def build_context_priority_agenda(
    data_dir: str,
    context_key: str,
    *,
    schema: dict[str, Any] | None = None,
) -> ContextPriorityAgendaReport:
    """Build a comprehensive ranked research priority agenda report for a ModelContextKey."""
    init_analysis_db(data_dir)
    c_key_str = str(context_key).strip()

    ctx_obj = ModelContextKey.from_key_str(c_key_str)
    market = ctx_obj.market
    interval_sec = ctx_obj.sampling_interval_sec
    task_type = ctx_obj.task_type.value if hasattr(ctx_obj.task_type, "value") else str(ctx_obj.task_type)
    horizon = ctx_obj.prediction_horizon
    regime_id = ctx_obj.regime_id

    reg_schema = schema if schema is not None else load_schema_registry()
    aff_rep = recommend_features_for_context(data_dir, c_key_str, schema=reg_schema)
    vuln = audit_champion_vulnerability(data_dir, c_key_str)
    cov = analyze_context_coverage(data_dir, c_key_str)

    candidates: list[ResearchOpportunity] = []

    # 1. Opportunity Type A: Missing Champion Features or Vulnerability Defense
    champ_defense_feats = list(aff_rep.missing_champion_feature_opportunities[:5])
    if not champ_defense_feats and vuln.vulnerability_score >= 0.40:
        # If champion is vulnerable, propose testing non-quarantined features
        avail = [
            f.feature_name for f in aff_rep.recommended_features
            if f.recommendation_class != FeatureRecommendationClass.QUARANTINED
        ][:5]
        if avail:
            champ_defense_feats = avail

    if champ_defense_feats:
        opp = evaluate_research_opportunity(
            data_dir,
            c_key_str,
            opportunity_type=OpportunityType.CHAMPION_VULNERABILITY_DEFENSE,
            candidate_features=champ_defense_feats,
            target_algorithm="xgboost",
            schema=reg_schema,
        )
        candidates.append(opp)

    # 2. Opportunity Type B: Top Confirmed/Promising/Active Features
    top_feats = [
        f.feature_name for f in aff_rep.recommended_features
        if f.recommendation_class in (FeatureRecommendationClass.CONFIRMED, FeatureRecommendationClass.PROMISING)
    ][:6]
    if not top_feats:
        top_feats = [
            f.feature_name for f in aff_rep.recommended_features
            if f.recommendation_class != FeatureRecommendationClass.QUARANTINED
        ][:4]

    if top_feats:
        opp = evaluate_research_opportunity(
            data_dir,
            c_key_str,
            opportunity_type=OpportunityType.FEATURE_EXPLORATION,
            candidate_features=top_feats,
            target_algorithm="xgboost",
            schema=reg_schema,
        )
        candidates.append(opp)

    # 3. Opportunity Type C: Top Interaction Pair
    if aff_rep.interaction_recommendations:
        top_inter = aff_rep.interaction_recommendations[0]
        opp = evaluate_research_opportunity(
            data_dir,
            c_key_str,
            opportunity_type=OpportunityType.INTERACTION_VALIDATION,
            candidate_features=top_inter.feature_set,
            target_algorithm="xgboost",
            schema=reg_schema,
        )
        candidates.append(opp)

    # 4. Opportunity Type D: Challenger Opportunity (if leading or active)
    if vuln.challenger_status in (ChallengerStatus.RESEARCH_CANDIDATE_LEADS, ChallengerStatus.PRODUCTION_CHALLENGER_ACTIVE):
        bms = get_model_benchmarks_for_context(data_dir, c_key_str)
        if bms:
            top_bm = max(bms, key=lambda b: _clean_float(b.get("robustness_score"), 0.0))
            if top_bm.get("signature_hash"):
                conn = connect_analysis_db(data_dir)
                try:
                    row = conn.execute(
                        "SELECT canonical_payload_json, algorithm FROM experiment_signatures WHERE signature_hash = ?;",
                        (top_bm["signature_hash"],),
                    ).fetchone()
                    if row and row["canonical_payload_json"]:
                        p = json.loads(row["canonical_payload_json"])
                        opp = evaluate_research_opportunity(
                            data_dir,
                            c_key_str,
                            opportunity_type=OpportunityType.CHALLENGER_OPPORTUNITY,
                            candidate_features=p.get("features", []),
                            target_algorithm=row["algorithm"] or "xgboost",
                            schema=reg_schema,
                        )
                        candidates.append(opp)
                finally:
                    conn.close()

    # 5. Opportunity Type E: Coverage Expansion (if cold or sparse)
    if cov.coverage_class in (CoverageClass.COLD_START, CoverageClass.SPARSE):
        cols = reg_schema.get("columns") or {}
        base_feats = [
            f for f, m in cols.items()
            if isinstance(m, dict) and classify_feature_population(f, schema=reg_schema) == "BASE"
        ][:8]
        if not base_feats:
            base_feats = [
                f for f, m in cols.items()
                if isinstance(m, dict) and str(m.get("status", "ACTIVE")).upper() != "DEPRECATED"
            ][:4]
        if base_feats:
            opp = evaluate_research_opportunity(
                data_dir,
                c_key_str,
                opportunity_type=OpportunityType.COVERAGE_EXPANSION,
                candidate_features=base_feats,
                target_algorithm="catboost" if len(candidates) > 0 else "xgboost",
                schema=reg_schema,
            )
            candidates.append(opp)

    # Deterministic Tie-Breaking Sort:
    def _sort_key(o: ResearchOpportunity):
        return (
            -o.priority_score,
            -o.component_breakdown.champion_vulnerability_score,
            -o.component_breakdown.challenger_gap_score,
            -o.component_breakdown.feature_affinity_score,
            -o.component_breakdown.coverage_gap_score,
            o.context_key,
            o.opportunity_id,
        )

    candidates.sort(key=_sort_key)

    eligible_opps = [o for o in candidates if o.exclusion_verdict == ExclusionVerdict.ELIGIBLE]
    caution_opps = [o for o in candidates if o.exclusion_verdict == ExclusionVerdict.CAUTION]
    suppressed_cnt = sum(1 for o in candidates if o.exclusion_verdict == ExclusionVerdict.EXCLUDED)

    top_class = candidates[0].priority_class if candidates else ResearchPriorityClass.NEGLIGIBLE

    return ContextPriorityAgendaReport(
        context_key=c_key_str,
        market=market,
        sampling_interval_sec=interval_sec,
        task_type=task_type,
        prediction_horizon=horizon,
        regime_id=regime_id,
        top_priority_class=top_class,
        total_opportunities_evaluated=len(candidates),
        eligible_opportunities=eligible_opps,
        caution_opportunities=caution_opps,
        suppressed_excluded_count=suppressed_cnt,
        generated_at=_utc_now_iso(),
    )

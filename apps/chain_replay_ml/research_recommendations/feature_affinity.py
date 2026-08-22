"""Empirical Feature Affinity & Interaction Recommender (Phase 4E.3).

Synthesizes feature evidence from Research Memory (analysis.db), Feature Evidence (feature_recommendation_evidence.db),
and the Feature Registry to recommend high-impact individual features and pairwise combinations for a ModelContextKey.

Invariants:
1. Strictly Advisory: Zero automated feature graduation, deprecation, or pipeline modifications.
2. Context Isolation: All benchmark metrics and interactions are partitioned by canonical ModelContextKey.
3. Disaggregated Confidence: Affinity scores are strictly separated from evidence-volume confidence.
4. Quarantined Deprecation: Deprecated/retired features receive score 0.0 and recommendation class QUARANTINED.
5. Non-Causal Interaction Modeling: Pairwise synergy is reported as empirical interaction lift, not causal proof.
"""

from __future__ import annotations

import json
import math
import os
import sqlite3
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from itertools import combinations
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
from chain_replay_ml.research_memory.regime_eval import get_regime_evaluations_for_model
from chain_replay_ml.research_memory.signature import canonical_context_key
from chain_replay_ml.research_recommendations.coverage import (
    CoverageClass,
    analyze_context_coverage,
)
from chain_replay_ml.research_recommendations.vulnerability import (
    ChampionVulnerabilityResult,
    audit_champion_vulnerability,
)
# get_champion_for_context decoupled to analysis.db


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class FeatureRecommendationClass(str, Enum):
    """Categorization of feature research viability and evidence maturity."""

    CONFIRMED = "CONFIRMED"      # High score (>=70.0) with strong empirical volume (C >= 0.60)
    PROMISING = "PROMISING"      # Solid score (>=65.0) with moderate empirical volume (C >= 0.30)
    EXPLORATORY = "EXPLORATORY"  # High/moderate score with sparse empirical volume (C < 0.30)
    MARGINAL = "MARGINAL"        # Low empirical affinity score (<50.0)
    QUARANTINED = "QUARANTINED"  # Deprecated in registry or severe negative evidence

    @classmethod
    def from_str(cls, value: str | Any) -> FeatureRecommendationClass:
        raw = str(value or "").strip().upper()
        try:
            return cls(raw)
        except ValueError:
            pass
        mapping = {
            "CONFIRM": cls.CONFIRMED,
            "STRONG": cls.CONFIRMED,
            "PROMISE": cls.PROMISING,
            "ACTIVE": cls.PROMISING,
            "EXPLORE": cls.EXPLORATORY,
            "SPARSE": cls.EXPLORATORY,
            "COLD": cls.EXPLORATORY,
            "WEAK": cls.MARGINAL,
            "LOW": cls.MARGINAL,
            "DEPRECATED": cls.QUARANTINED,
            "RETIRED": cls.QUARANTINED,
            "QUARANTINE": cls.QUARANTINED,
        }
        if raw in mapping:
            return mapping[raw]
        return cls.EXPLORATORY


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


def compute_feature_affinity_score(
    *,
    robustness_support: float | None = None,
    evidence_support: float | None = None,
    stability_support: float | None = None,
    population_support: float | None = None,
    is_deprecated: bool = False,
) -> float:
    """Calculate the safe bounded Feature Affinity Score A(f) in [0.0, 100.0].
    
    Mathematical Formulation:
        A(f) = 0.40 * S_rob + 0.30 * S_evid + 0.20 * S_stab + 0.10 * S_pop
        
    Component Formulations:
        S_rob  = clamp(robustness_support, 0.0, 100.0) (fallback 50.0 if unobserved)
        S_evid = clamp(evidence_support, 0.0, 100.0)   (fallback 50.0 if unobserved)
        S_stab = clamp(stability_support, 0.0, 100.0)  (fallback 100.0 if unobserved)
        S_pop  = clamp(population_support, 0.0, 100.0) (BASE=100, REG=85, EXP=70, UNK=20, DEP=0)
        
    Invariant:
        If is_deprecated is True: A(f) == 0.0 (Strict quarantine).
    """
    if is_deprecated:
        return 0.0

    s_rob = min(100.0, max(0.0, _clean_float(robustness_support, 50.0)))
    s_evid = min(100.0, max(0.0, _clean_float(evidence_support, 50.0)))
    s_stab = min(100.0, max(0.0, _clean_float(stability_support, 100.0)))
    s_pop = min(100.0, max(0.0, _clean_float(population_support, 70.0)))

    raw_a = (0.40 * s_rob) + (0.30 * s_evid) + (0.20 * s_stab) + (0.10 * s_pop)
    clamped_a = min(100.0, max(0.0, raw_a))
    return round(clamped_a, 4)


def compute_feature_confidence(experiment_count: int) -> float:
    """Calculate statistical evidence volume confidence C(f) in [0.0, 1.0].
    
    Formulation:
        C(f) = 1.0 - exp(-N_experiments / 5.0)
        
    Properties:
        - Exactly 0.0 when N <= 0
        - ~0.1813 at N = 1
        - ~0.6321 at N = 5
        - ~0.9502 at N = 15
    """
    n = max(0, int(experiment_count or 0))
    if n == 0:
        return 0.0
    c = 1.0 - math.exp(-float(n) / 5.0)
    return round(min(1.0, max(0.0, c)), 4)


def compute_interaction_synergy_score(
    pair_mean_robustness: float | None,
    max_individual_robustness: float | None,
    pair_experiment_count: int,
) -> tuple[float, float, float]:
    """Calculate empirical interaction lift, synergy score, and confidence for a feature pair.
    
    Formulations:
        interaction_lift = pair_mean_robustness - max_individual_robustness
        interaction_score = clamp(50.0 + (interaction_lift * 2.5), 0.0, 100.0)
        interaction_confidence = 1.0 - exp(-N_pair / 3.0)
        
    Returns:
        (interaction_lift, interaction_score, interaction_confidence)
    """
    if pair_mean_robustness is None or max_individual_robustness is None or pair_experiment_count <= 0:
        return (0.0, 50.0, 0.0)

    p_rob = _clean_float(pair_mean_robustness, 50.0)
    ind_rob = _clean_float(max_individual_robustness, 50.0)
    
    lift = round(p_rob - ind_rob, 4)
    raw_score = 50.0 + (lift * 2.5)
    score = round(min(100.0, max(0.0, raw_score)), 4)

    conf = round(min(1.0, max(0.0, 1.0 - math.exp(-float(pair_experiment_count) / 3.0))), 4)
    return (lift, score, conf)


def classify_feature_recommendation(
    affinity_score: float,
    confidence: float,
    *,
    is_deprecated: bool = False,
) -> FeatureRecommendationClass:
    """Deterministically categorize feature recommendation class."""
    if is_deprecated:
        return FeatureRecommendationClass.QUARANTINED
    
    score = min(100.0, max(0.0, float(affinity_score or 0.0)))
    conf = min(1.0, max(0.0, float(confidence or 0.0)))

    if score < 50.0:
        return FeatureRecommendationClass.MARGINAL
    elif score >= 70.0 and conf >= 0.60:
        return FeatureRecommendationClass.CONFIRMED
    elif score >= 65.0 and conf >= 0.30:
        return FeatureRecommendationClass.PROMISING
    else:
        return FeatureRecommendationClass.EXPLORATORY


def _lookup_feature_evidence_db(data_dir: str, feature_name: str) -> tuple[float, str | None, str | None]:
    """Safely query feature_recommendation_evidence.db for evidence_score, lifecycle_status, and graduation."""
    from chain_replay_ml.production_validation.evidence_store import evidence_db_path
    db_path = evidence_db_path(data_dir)
    if not os.path.isfile(db_path):
        return (50.0, None, None)

    ev_score = 50.0
    lifecycle_stat = None
    grad_stat = None

    try:
        conn = sqlite3.connect(db_path, timeout=5.0)
        conn.row_factory = sqlite3.Row
        try:
            # Check summary
            row = conn.execute(
                "SELECT evidence_score, lifecycle_status FROM feature_context_summary WHERE feature_name = ? LIMIT 1;",
                (feature_name,),
            ).fetchone()
            if row:
                if row["evidence_score"] is not None:
                    ev_score = min(100.0, max(0.0, float(row["evidence_score"])))
                lifecycle_stat = str(row["lifecycle_status"]) if row["lifecycle_status"] else None

            # Check graduation dossier
            row_g = conn.execute(
                "SELECT graduation_decision, verdict FROM feature_graduation_dossiers WHERE feature_name = ? ORDER BY created_at DESC LIMIT 1;",
                (feature_name,),
            ).fetchone()
            if row_g:
                grad_stat = str(row_g["graduation_decision"] or row_g["verdict"] or "")
        finally:
            conn.close()
    except Exception:
        pass

    return (round(ev_score, 4), lifecycle_stat, grad_stat)


@dataclass(frozen=True)
class FeatureAffinityResult:
    """Empirical affinity and evidence profile for a single feature in a ModelContextKey."""

    context_key: str
    feature_name: str
    feature_population: str
    evidence_count: int
    successful_experiment_count: int
    affinity_score: float
    confidence: float
    robustness_support: float
    evidence_support: float
    stability_support: float
    population_support: float
    graduation_status: str | None
    recommendation_class: FeatureRecommendationClass
    supporting_signature_hashes: list[str]

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["recommendation_class"] = self.recommendation_class.value
        return d


@dataclass(frozen=True)
class FeatureInteractionResult:
    """Observed combination synergy and interaction evidence between feature pairs."""

    context_key: str
    feature_set: list[str]
    interaction_size: int
    pair_experiment_count: int
    interaction_lift: float
    interaction_score: float
    interaction_confidence: float
    recommendation_class: FeatureRecommendationClass
    supporting_signature_hashes: list[str]

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["recommendation_class"] = self.recommendation_class.value
        return d


@dataclass(frozen=True)
class ContextFeatureAffinityReport:
    """Aggregated feature intelligence and recommendation agenda for a ModelContextKey."""

    context_key: str
    market: str
    sampling_interval_sec: int
    task_type: str
    prediction_horizon: str
    regime_id: str
    coverage_class: str
    champion_model_name: str | None
    recommended_features: list[FeatureAffinityResult]
    interaction_recommendations: list[FeatureInteractionResult]
    missing_champion_feature_opportunities: list[str]
    excluded_features: list[str]
    total_features_analyzed: int
    confirmed_feature_count: int
    promising_feature_count: int
    exploratory_feature_count: int
    quarantined_feature_count: int
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
            "champion_model_name": self.champion_model_name,
            "recommended_features": [f.to_dict() for f in self.recommended_features],
            "interaction_recommendations": [i.to_dict() for i in self.interaction_recommendations],
            "missing_champion_feature_opportunities": self.missing_champion_feature_opportunities,
            "excluded_features": self.excluded_features,
            "total_features_analyzed": self.total_features_analyzed,
            "confirmed_feature_count": self.confirmed_feature_count,
            "promising_feature_count": self.promising_feature_count,
            "exploratory_feature_count": self.exploratory_feature_count,
            "quarantined_feature_count": self.quarantined_feature_count,
            "generated_at": self.generated_at,
        }


def analyze_feature_affinity(
    data_dir: str,
    context_key: str,
    feature_name: str,
    *,
    schema: dict[str, Any] | None = None,
) -> FeatureAffinityResult:
    """Analyze empirical affinity and confidence for a single feature in a specific context."""
    init_analysis_db(data_dir)
    c_key_str = str(context_key).strip()
    f_name = str(feature_name).strip()

    # 1. Authoritative Population Classification
    pop_str = classify_feature_population(f_name, schema=schema)
    is_deprecated = (pop_str == "DEPRECATED")

    pop_scores = {
        "BASE": 100.0,
        "REGISTRY": 85.0,
        "EXPERIMENTAL": 70.0,
        "UNKNOWN": 20.0,
        "DEPRECATED": 0.0,
    }
    s_pop = pop_scores.get(pop_str, 20.0)

    # 2. Query Research Memory (analysis.db) for experiments containing feature f
    conn = connect_analysis_db(data_dir)
    exp_sig_rows: list[sqlite3.Row] = []
    try:
        exp_sig_rows = conn.execute(
            """
            SELECT signature_hash, canonical_payload_json 
            FROM experiment_signatures 
            WHERE context_key = ?;
            """,
            (c_key_str,),
        ).fetchall()
    finally:
        conn.close()

    matching_sig_hashes: list[str] = []
    for sr in exp_sig_rows:
        raw_json = sr["canonical_payload_json"]
        if raw_json:
            try:
                payload = json.loads(raw_json)
                feats = payload.get("features", [])
                if isinstance(feats, list) and f_name in feats:
                    matching_sig_hashes.append(str(sr["signature_hash"]))
            except (json.JSONDecodeError, TypeError):
                pass

    # 3. Retrieve Benchmarks for Matching Signatures
    benchmarks = get_model_benchmarks_for_context(data_dir, c_key_str)
    matching_bms = [b for b in benchmarks if str(b.get("signature_hash")) in matching_sig_hashes]

    n_exp = len(matching_bms)
    rob_scores: list[float] = []
    success_exp = 0
    for b in matching_bms:
        rs = b.get("robustness_score")
        if rs is not None:
            r_val = min(100.0, max(0.0, float(rs)))
            rob_scores.append(r_val)
            if r_val >= 60.0:
                success_exp += 1

    s_rob = round(sum(rob_scores) / len(rob_scores), 4) if rob_scores else 50.0

    # 4. Cross-Regime Degradation Analysis
    degradations: list[float] = []
    for sig_h in matching_sig_hashes:
        reg_evals = get_regime_evaluations_for_model(data_dir, sig_h)
        for r in reg_evals:
            if not r.get("is_native_regime", False):
                deg = r.get("regime_degradation_pct")
                if deg is not None:
                    degradations.append(min(100.0, max(0.0, float(deg))))

    if degradations:
        mean_deg = sum(degradations) / len(degradations)
        s_stab = round(max(0.0, 100.0 - (mean_deg * 3.3333)), 4)
    else:
        s_stab = 100.0

    # 5. Feature Evidence Store Lookups
    s_evid, lifecycle_status, grad_status = _lookup_feature_evidence_db(data_dir, f_name)
    if lifecycle_status in ("DEPRECATED", "RETIRED"):
        is_deprecated = True

    # 6. Compute Affinity Score & Confidence
    aff_score = compute_feature_affinity_score(
        robustness_support=s_rob,
        evidence_support=s_evid,
        stability_support=s_stab,
        population_support=s_pop,
        is_deprecated=is_deprecated,
    )
    conf = compute_feature_confidence(n_exp)
    rec_class = classify_feature_recommendation(aff_score, conf, is_deprecated=is_deprecated)

    return FeatureAffinityResult(
        context_key=c_key_str,
        feature_name=f_name,
        feature_population=pop_str,
        evidence_count=n_exp,
        successful_experiment_count=success_exp,
        affinity_score=aff_score,
        confidence=conf,
        robustness_support=s_rob,
        evidence_support=s_evid,
        stability_support=s_stab,
        population_support=s_pop,
        graduation_status=grad_status,
        recommendation_class=rec_class,
        supporting_signature_hashes=sorted(matching_sig_hashes),
    )


def recommend_features_for_context(
    data_dir: str,
    context_key: str,
    *,
    max_features: int = 50,
    max_interactions: int = 25,
    schema: dict[str, Any] | None = None,
) -> ContextFeatureAffinityReport:
    """Generate a complete empirical feature affinity and interaction recommendation report for a context."""
    init_analysis_db(data_dir)
    c_key_str = str(context_key).strip()

    ctx_obj = ModelContextKey.from_key_str(c_key_str)
    market = ctx_obj.market
    sampling_sec = ctx_obj.sampling_interval_sec
    task_type = ctx_obj.task_type.value if hasattr(ctx_obj.task_type, "value") else str(ctx_obj.task_type)
    horizon = ctx_obj.prediction_horizon
    regime_id = ctx_obj.regime_id

    # 1. Context Coverage & Production Champion Analysis
    cov_profile = analyze_context_coverage(data_dir, c_key_str)
    cov_class_str = cov_profile.coverage_class.value

    champ_audit = audit_champion_vulnerability(data_dir, c_key_str)
    champ_name = champ_audit.champion_model_name

    # 2. Discover all features tested in this context + canonical schema features
    reg_schema = schema if schema is not None else load_schema_registry()
    cols = reg_schema.get("columns") or {}

    conn = connect_analysis_db(data_dir)
    all_context_signatures: list[dict[str, Any]] = []
    try:
        rows = conn.execute(
            """
            SELECT s.signature_hash, s.feature_set_hash, s.canonical_payload_json, b.robustness_score
            FROM experiment_signatures s
            LEFT JOIN model_benchmarks b ON s.signature_hash = b.signature_hash
            WHERE s.context_key = ?;
            """,
            (c_key_str,),
        ).fetchall()
        for r in rows:
            all_context_signatures.append({
                "signature_hash": r["signature_hash"],
                "feature_set_hash": r["feature_set_hash"],
                "payload": json.loads(r["canonical_payload_json"]) if r["canonical_payload_json"] else {},
                "robustness_score": float(r["robustness_score"]) if r["robustness_score"] is not None else None,
            })
    finally:
        conn.close()

    discovered_features: set[str] = set()
    for item in all_context_signatures:
        for f in item["payload"].get("features", []):
            if f:
                discovered_features.add(str(f).strip())

    # Add registry features for comprehensive baseline exploration
    for f in cols.keys():
        if f and not str(f).startswith("_"):
            discovered_features.add(str(f).strip())

    # 3. Analyze Affinity for All Discovered Features
    feature_results: list[FeatureAffinityResult] = []
    excluded_features: list[str] = []

    for f_name in discovered_features:
        res = analyze_feature_affinity(data_dir, c_key_str, f_name, schema=reg_schema)
        if res.recommendation_class == FeatureRecommendationClass.QUARANTINED:
            excluded_features.append(f_name)
        feature_results.append(res)

    # Sort deterministically: -affinity_score, -confidence, feature_name
    feature_results.sort(key=lambda x: (-x.affinity_score, -x.confidence, x.feature_name))
    excluded_features.sort()

    # 4. Pairwise Interaction Synergy Analysis
    # Group experiments by feature combinations
    feature_exp_map: dict[str, list[float]] = {}
    pair_exp_map: dict[tuple[str, str], list[float]] = {}
    pair_sigs_map: dict[tuple[str, str], list[str]] = {}

    for item in all_context_signatures:
        rob = item["robustness_score"]
        sig_h = item["signature_hash"]
        feats = sorted(list(set(str(f).strip() for f in item["payload"].get("features", []) if f)))
        if rob is not None:
            for f in feats:
                feature_exp_map.setdefault(f, []).append(rob)
            for f1, f2 in combinations(feats, 2):
                pair_key = (f1, f2) if f1 < f2 else (f2, f1)
                pair_exp_map.setdefault(pair_key, []).append(rob)
                pair_sigs_map.setdefault(pair_key, []).append(sig_h)

    interaction_results: list[FeatureInteractionResult] = []
    for (f1, f2), pair_robs in pair_exp_map.items():
        # Check if either feature is quarantined
        pop1 = classify_feature_population(f1, schema=reg_schema)
        pop2 = classify_feature_population(f2, schema=reg_schema)
        is_dep = (pop1 == "DEPRECATED" or pop2 == "DEPRECATED")

        p_mean = sum(pair_robs) / len(pair_robs)
        r1_list = feature_exp_map.get(f1, [])
        r2_list = feature_exp_map.get(f2, [])
        max_ind = max(
            (sum(r1_list) / len(r1_list) if r1_list else 50.0),
            (sum(r2_list) / len(r2_list) if r2_list else 50.0),
        )

        lift, score, conf = compute_interaction_synergy_score(p_mean, max_ind, len(pair_robs))
        if is_dep:
            i_class = FeatureRecommendationClass.QUARANTINED
            score = 0.0
        else:
            i_class = classify_feature_recommendation(score, conf, is_deprecated=False)

        interaction_results.append(FeatureInteractionResult(
            context_key=c_key_str,
            feature_set=[f1, f2],
            interaction_size=2,
            pair_experiment_count=len(pair_robs),
            interaction_lift=lift,
            interaction_score=score,
            interaction_confidence=conf,
            recommendation_class=i_class,
            supporting_signature_hashes=sorted(list(set(pair_sigs_map.get((f1, f2), [])))),
        ))

    # Sort interactions deterministically: -interaction_score, -interaction_confidence, feature_pair
    interaction_results.sort(key=lambda x: (-x.interaction_score, -x.interaction_confidence, x.feature_set[0], x.feature_set[1]))

    # 5. Identify Missing Champion Feature Opportunities
    # Determine champion's feature set
    champ_features: set[str] = set()
    if champ_name:
        bms = get_model_benchmarks_for_context(data_dir, c_key_str)
        c_bm = next((b for b in bms if str(b.get("model_name")).strip() == str(champ_name).strip()), None)
        if c_bm and c_bm.get("signature_hash"):
            matching_item = next((it for it in all_context_signatures if it["signature_hash"] == c_bm["signature_hash"]), None)
            if matching_item:
                champ_features.update(matching_item["payload"].get("features", []))

    missing_champ_opps: list[str] = []
    for fr in feature_results:
        if fr.recommendation_class in (FeatureRecommendationClass.CONFIRMED, FeatureRecommendationClass.PROMISING):
            if fr.feature_name not in champ_features and fr.feature_name not in excluded_features:
                missing_champ_opps.append(fr.feature_name)

    # 6. Summary Counts
    conf_cnt = sum(1 for f in feature_results if f.recommendation_class == FeatureRecommendationClass.CONFIRMED)
    prom_cnt = sum(1 for f in feature_results if f.recommendation_class == FeatureRecommendationClass.PROMISING)
    expl_cnt = sum(1 for f in feature_results if f.recommendation_class == FeatureRecommendationClass.EXPLORATORY)
    quar_cnt = len(excluded_features)

    return ContextFeatureAffinityReport(
        context_key=c_key_str,
        market=market,
        sampling_interval_sec=sampling_sec,
        task_type=task_type,
        prediction_horizon=horizon,
        regime_id=regime_id,
        coverage_class=cov_class_str,
        champion_model_name=champ_name,
        recommended_features=feature_results[:max_features],
        interaction_recommendations=interaction_results[:max_interactions],
        missing_champion_feature_opportunities=missing_champ_opps[:20],
        excluded_features=excluded_features,
        total_features_analyzed=len(feature_results),
        confirmed_feature_count=conf_cnt,
        promising_feature_count=prom_cnt,
        exploratory_feature_count=expl_cnt,
        quarantined_feature_count=quar_cnt,
        generated_at=_utc_now_iso(),
    )

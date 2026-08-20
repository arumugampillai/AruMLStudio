"""Discovered Feature Intelligence aggregation for Phase 4F.6.

Correlates candidate feature mutations with empirical trading/model performance deltas
and intersects with Phase 4E Feature Intelligence to deliver actionable researcher guidance
without modifying production assets or feature registries.
"""

from __future__ import annotations

from typing import Any, Sequence

from chain_replay_ml.fine_tuning.types import DescendantEvaluationRecord
from chain_replay_ml.model_ranking.types import CandidateEvidenceScore
from chain_replay_ml.research_recommendations.feature_affinity import (
    analyze_feature_affinity,
)
from .types import (
    CandidateFeatureDeltaView,
    DiscoveredFeatureRecord,
    DiscoveredFeatureStatus,
    DiscoveredFeatureSynergy,
)


def extract_discovered_feature_intelligence(
    data_dir: str,
    context_key: str,
    ranked_candidates: Sequence[CandidateEvidenceScore],
    fine_tuning_trials: Sequence[DescendantEvaluationRecord],
    candidate_specs: dict[str, dict[str, Any]] | None = None,
) -> tuple[list[DiscoveredFeatureRecord], list[DiscoveredFeatureSynergy], list[CandidateFeatureDeltaView]]:
    """Aggregate empirical feature discovery evidence across all evaluated candidates in a campaign."""
    specs = candidate_specs or {}
    score_by_id = {c.candidate_id: c for c in ranked_candidates}
    trial_by_child = {t.child_candidate_id: t for t in fine_tuning_trials}

    # 1. Ingest Phase 4E Feature Affinity & Interaction Synergy Evidence
    affinity_map: dict[str, str] = {}
    synergy_map: set[tuple[str, str]] = set()
    try:
        aff_report = analyze_feature_affinity(data_dir, context_key)
        for u in aff_report.univariate_recommendations:
            affinity_map[u.feature_name] = "HIGH" if u.composite_affinity_score >= 0.70 else "MODERATE"
        for inter in aff_report.interactions:
            if inter.synergy_lift > 0:
                synergy_map.add((inter.feature_a, inter.feature_b))
                synergy_map.add((inter.feature_b, inter.feature_a))
    except Exception:
        pass

    # 2. Extract Candidate Feature Deltas (Drill-Down View)
    feature_deltas: list[CandidateFeatureDeltaView] = []
    features_by_cand: dict[str, list[str]] = {}

    for c in ranked_candidates:
        c_id = c.candidate_id
        c_spec = specs.get(c_id, {})
        c_feats = c_spec.get("features", [])
        if not c_feats:
            # Fallback to model_metrics keys if features not explicitly populated
            c_feats = [k for k in c.model_metrics.keys() if not k.startswith("fold_") and k not in ("roc_auc", "total_features")]
            if not c_feats:
                c_feats = ["adx_14", "rsi_14", "macd_diff", "bb_width_20", "iv_mean"]
        features_by_cand[c_id] = c_feats

    for c in ranked_candidates:
        c_id = c.candidate_id
        t = trial_by_child.get(c_id)
        p_id = c.parent_candidate_id or (t.parent_candidate_id if t else None)
        p_score = score_by_id.get(p_id) if p_id else None
        p_feats = features_by_cand.get(p_id, []) if p_id else []
        c_feats = features_by_cand.get(c_id, [])

        set_c = set(c_feats)
        set_p = set(p_feats) if p_id else set()

        added = sorted(list(set_c - set_p)) if p_id else sorted(list(set_c))
        removed = sorted(list(set_p - set_c)) if p_id else []
        interactions = [f for f in c_feats if "_syn" in f.lower() or "_x_" in f.lower()]

        delta_comp = round(c.composite_score - (p_score.composite_score if p_score else c.composite_score), 2) if p_id else 0.0
        delta_trade = round(c.trading_evidence_score - (p_score.trading_evidence_score if p_score else c.trading_evidence_score), 2) if p_id else 0.0
        delta_model = round(c.model_evidence_score - (p_score.model_evidence_score if p_score else c.model_evidence_score), 2) if p_id else 0.0

        if t:
            delta_comp = round(t.delta_composite_score, 2)
            delta_trade = round(t.delta_trading_score, 2)
            delta_model = round(t.delta_model_score, 2)

        view = CandidateFeatureDeltaView(
            candidate_id=c_id,
            parent_candidate_id=p_id,
            parent_features=p_feats,
            child_features=c_feats,
            added_features=added,
            removed_features=removed,
            interaction_features=interactions,
            delta_composite=delta_comp,
            delta_trading=delta_trade,
            delta_model=delta_model,
            model_score=c.model_evidence_score,
            trading_score=c.trading_evidence_score,
            composite_score=c.composite_score,
            win_rate_pct=c.trading_metrics.get("win_rate_pct", 0.0),
            profit_factor=c.trading_metrics.get("profit_factor", 0.0),
            max_drawdown_pct=c.trading_metrics.get("max_drawdown_pct", 0.0),
        )
        feature_deltas.append(view)

    # 3. Aggregate Individual Discovered Features
    all_features = set()
    for feats in features_by_cand.values():
        all_features.update(feats)

    feature_records: list[DiscoveredFeatureRecord] = []
    for f_name in sorted(list(all_features)):
        cands_with_f = [c for c in ranked_candidates if f_name in features_by_cand.get(c.candidate_id, [])]
        if not cands_with_f:
            continue

        times_tested = len(cands_with_f)
        top_cands = [c.candidate_id for c in cands_with_f[:3]]
        best_comp = max(c.composite_score for c in cands_with_f)
        best_trade = max(c.trading_evidence_score for c in cands_with_f)
        avg_comp = sum(c.composite_score for c in cands_with_f) / float(times_tested)
        avg_trade = sum(c.trading_evidence_score for c in cands_with_f) / float(times_tested)

        # Collect deltas when feature was present in child candidate
        deltas = []
        for d_view in feature_deltas:
            if f_name in d_view.child_features and d_view.parent_candidate_id:
                deltas.append(d_view.delta_composite)

        pos_count = sum(1 for d in deltas if d > 0.0)
        best_delta = max(deltas) if deltas else 0.0
        avg_delta = (sum(deltas) / float(len(deltas))) if deltas else 0.0

        p4e_ev = affinity_map.get(f_name, "MODERATE")
        cross_regime = "HIGH" if times_tested >= 3 and avg_comp >= 65.0 else ("MODERATE" if times_tested >= 2 else "LOW")
        lifecycle_stat = "ACTIVE"

        # Multi-evidence Status Classification
        if (best_delta > 0.0 or avg_delta > 0.0 or best_comp >= 75.0) and best_trade >= 60.0 and p4e_ev != "NEGATIVE":
            if pos_count >= 1 or best_comp >= 72.0:
                status = DiscoveredFeatureStatus.STRONG_DISCOVERED
                rec = "DISCOVERED — HUMAN REVIEW REQUIRED"
            else:
                status = DiscoveredFeatureStatus.PROMISING
                rec = "DISCOVERED — HUMAN REVIEW REQUIRED"
        elif avg_delta < -2.0 and best_delta <= 0.0:
            status = DiscoveredFeatureStatus.REJECTED_HARMFUL
            rec = "REJECTED — AVOID"
        else:
            status = DiscoveredFeatureStatus.PROMISING
            rec = "DISCOVERED — HUMAN REVIEW REQUIRED"

        feature_records.append(
            DiscoveredFeatureRecord(
                feature_name=f_name,
                times_tested=times_tested,
                positive_descendant_count=pos_count,
                top_candidates=top_cands,
                best_composite_score=best_comp,
                best_trading_score=best_trade,
                avg_composite_score=avg_comp,
                avg_trading_score=avg_trade,
                best_delta_vs_parent=best_delta,
                avg_delta_vs_parent=avg_delta,
                cross_regime_consistency=cross_regime,
                phase4e_evidence_level=p4e_ev,
                lifecycle_status=lifecycle_stat,
                status=status,
                recommendation=rec,
            )
        )

    # Sort records: STRONG_DISCOVERED first, then by best_composite_score descending
    status_order = {
        DiscoveredFeatureStatus.STRONG_DISCOVERED: 0,
        DiscoveredFeatureStatus.PROMISING: 1,
        DiscoveredFeatureStatus.REJECTED_HARMFUL: 2,
    }
    feature_records.sort(key=lambda r: (status_order[r.status], -r.best_composite_score, -r.best_delta_vs_parent))

    # 4. Aggregate Feature Synergy Discoveries
    synergy_pairs: dict[tuple[str, str], list[CandidateFeatureDeltaView]] = {}
    for d_view in feature_deltas:
        feats = d_view.child_features
        for i in range(len(feats)):
            for j in range(i + 1, len(feats)):
                pair = tuple(sorted([feats[i], feats[j]]))
                if pair not in synergy_pairs:
                    synergy_pairs[pair] = []
                synergy_pairs[pair].append(d_view)

    synergies: list[DiscoveredFeatureSynergy] = []
    for (fa, fb), views in synergy_pairs.items():
        if len(views) >= 1:
            best_d_comp = max(v.delta_composite for v in views)
            best_d_trade = max(v.delta_trading for v in views)
            is_p4e_pair = (fa, fb) in synergy_map or (fb, fa) in synergy_map
            cross_reg = "HIGH" if len(views) >= 3 and best_d_comp > 0.0 else ("MODERATE" if len(views) >= 2 else "EMERGING")
            stat = "VALIDATED_SYNERGY — REVIEW" if (best_d_comp > 0.0 or is_p4e_pair) else "CANDIDATE_SYNERGY"

            synergies.append(
                DiscoveredFeatureSynergy(
                    feature_a=fa,
                    feature_b=fb,
                    times_tested=len(views),
                    best_delta_composite=round(best_d_comp, 2),
                    best_delta_trading=round(best_d_trade, 2),
                    cross_regime_evidence=cross_reg,
                    status=stat,
                )
            )

    synergies.sort(key=lambda s: (-s.best_delta_composite, -s.times_tested))

    return feature_records, synergies, feature_deltas

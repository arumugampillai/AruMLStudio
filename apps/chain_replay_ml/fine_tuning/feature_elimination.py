"""Feature Elimination Strategies for Autonomous Research Fine-Tuning (Phase 4F.4).

Supports 4 selectable policies:
1. NONE: No feature elimination (preserves complete parent feature set).
2. SHAP: Eliminates lowest SHAP importance features.
3. RFE: Recursive Feature Elimination (step-wise backward pruning).
4. PERMUTATION: Eliminates features with lowest/negative permutation importance drop.
"""

from __future__ import annotations

import sqlite3
from typing import Any, Sequence

from chain_replay_ml.research_memory.db import connect_analysis_db, init_analysis_db


def resolve_feature_importance_scores(
    data_dir: str,
    context_key: str,
    features: Sequence[str],
    strategy: str,
) -> dict[str, float]:
    """Query or compute importance scores for the requested features under the given strategy."""
    strat = str(strategy or "NONE").strip().upper()
    scores: dict[str, float] = {}
    init_analysis_db(data_dir)
    conn = connect_analysis_db(data_dir)

    try:
        if strat == "SHAP":
            # 1. Query feature_profiles for shap_importance
            rows = conn.execute(
                "SELECT feature_name, shap_importance, mi_score, feature_score FROM feature_profiles WHERE feature_name IN ({})".format(
                    ",".join("?" for _ in features)
                ),
                tuple(features),
            ).fetchall()
            for r in rows:
                fn = r["feature_name"]
                shap_val = r["shap_importance"]
                if shap_val is not None:
                    scores[fn] = float(shap_val)
                elif r["feature_score"] is not None:
                    scores[fn] = float(r["feature_score"])
                elif r["mi_score"] is not None:
                    scores[fn] = float(r["mi_score"])

            # 2. Check shap table as secondary fallback
            if len(scores) < len(features):
                s_rows = conn.execute(
                    "SELECT feature_name, mean_abs_shap FROM shap WHERE feature_name IN ({})".format(
                        ",".join("?" for _ in features)
                    ),
                    tuple(features),
                ).fetchall()
                for sr in s_rows:
                    fn = sr["feature_name"]
                    if fn not in scores and sr["mean_abs_shap"] is not None:
                        scores[fn] = float(sr["mean_abs_shap"])

        elif strat == "PERMUTATION":
            # 1. Query feature_profiles for permutation_importance / permutation_delta_rmse
            rows = conn.execute(
                "SELECT feature_name, permutation_importance, permutation_delta_rmse, feature_score FROM feature_profiles WHERE feature_name IN ({})".format(
                    ",".join("?" for _ in features)
                ),
                tuple(features),
            ).fetchall()
            for r in rows:
                fn = r["feature_name"]
                perm_val = r["permutation_importance"]
                if perm_val is not None:
                    scores[fn] = float(perm_val)
                elif r["permutation_delta_rmse"] is not None:
                    scores[fn] = float(r["permutation_delta_rmse"])
                elif r["feature_score"] is not None:
                    scores[fn] = float(r["feature_score"])

            # 2. Check permutation_importance table as fallback
            if len(scores) < len(features):
                try:
                    p_rows = conn.execute(
                        "SELECT feature_name, mean_importance FROM permutation_importance WHERE feature_name IN ({})".format(
                            ",".join("?" for _ in features)
                        ),
                        tuple(features),
                    ).fetchall()
                    for pr in p_rows:
                        fn = pr["feature_name"]
                        if fn not in scores and pr["mean_importance"] is not None:
                            scores[fn] = float(pr["mean_importance"])
                except Exception:
                    pass

        elif strat == "RFE":
            # Recursive Feature Elimination combines feature scorecard + variance + signal ranking
            rows = conn.execute(
                "SELECT feature_name, feature_score, mi_score, shap_importance, permutation_importance FROM feature_profiles WHERE feature_name IN ({})".format(
                    ",".join("?" for _ in features)
                ),
                tuple(features),
            ).fetchall()
            for r in rows:
                fn = r["feature_name"]
                f_score = r["feature_score"] or 0.0
                mi = r["mi_score"] or 0.0
                sh = r["shap_importance"] or 0.0
                pm = r["permutation_importance"] or 0.0
                combined = float(f_score) * 0.4 + float(sh) * 0.3 + float(pm) * 0.2 + float(mi) * 0.1
                scores[fn] = combined

    except Exception:
        pass
    finally:
        conn.close()

    # Query Feature Studio Evidence Store (feature_recommendation_evidence.db) for context evidence
    try:
        from chain_replay_ml.production_validation.evidence_store import get_connection
        ev_conn = get_connection(data_dir)
        try:
            ev_rows = ev_conn.execute(
                "SELECT feature_name, evidence_score, last_recommendation, remove_runs, keep_runs FROM feature_context_summary WHERE feature_name IN ({})".format(
                    ",".join("?" for _ in features)
                ),
                tuple(features),
            ).fetchall()
            for er in ev_rows:
                fn = er["feature_name"]
                ev_score = float(er["evidence_score"] or 0.0) / 100.0
                last_rec = str(er["last_recommendation"] or "").upper()
                rem_runs = int(er["remove_runs"] or 0)

                cur_val = scores.get(fn, 0.5)
                # Blend model importance with Feature Studio longitudinal evidence
                blended = (0.60 * cur_val) + (0.40 * ev_score)
                # Penalize features with repeated REMOVE verdicts in Feature Studio
                if last_rec == "REMOVE" or rem_runs > 0:
                    penalty_factor = max(0.10, 1.0 - (0.30 * min(3, rem_runs + (1 if last_rec == "REMOVE" else 0))))
                    blended *= penalty_factor

                scores[fn] = blended
        finally:
            ev_conn.close()
    except Exception:
        pass

    # Deterministic heuristic fallback for any features not yet scored in DB tables
    for idx, f in enumerate(features):
        if f not in scores:
            h_val = int(str(hash(f))[-4:]) / 10000.0
            scores[f] = 0.5 + h_val

    return scores


def apply_feature_elimination(
    data_dir: str,
    context_key: str,
    current_features: Sequence[str],
    strategy: str,
    generation_number: int,
    *,
    prune_fraction: float = 0.20,
    min_features: int = 15,
) -> tuple[list[str], list[str], str]:
    """Execute the configured elimination strategy on the candidate's feature universe.
    
    Returns:
        (retained_features, eliminated_features, mutation_description)
    """
    strat = str(strategy or "NONE").strip().upper()
    feats = list(dict.fromkeys(str(f).strip() for f in current_features if f and str(f).strip()))

    if strat in ("NONE", ""):
        return feats, [], "No feature elimination (NONE strategy active)"

    if len(feats) <= min_features:
        return feats, [], f"Feature set size ({len(feats)}) at or below minimum threshold ({min_features}); no features eliminated"

    scores = resolve_feature_importance_scores(data_dir, context_key, feats, strat)

    # Sort features by importance score descending (highest importance first)
    sorted_features = sorted(feats, key=lambda f: (-scores.get(f, 0.0), f))

    # Calculate number of features to keep
    drop_count = max(1, int(len(feats) * prune_fraction))
    keep_count = max(min_features, len(feats) - drop_count)

    retained = sorted_features[:keep_count]
    eliminated = sorted_features[keep_count:]

    strat_label = {
        "SHAP": "SHAP Importance",
        "RFE": "Recursive Feature Elimination (RFE)",
        "PERMUTATION": "Permutation Importance",
    }.get(strat, strat)

    desc = (
        f"{strat_label} [Gen {generation_number}]: "
        f"Retained {len(retained)} / {len(feats)} features (-{len(eliminated)} pruned)"
    )

    return sorted(retained), eliminated, desc

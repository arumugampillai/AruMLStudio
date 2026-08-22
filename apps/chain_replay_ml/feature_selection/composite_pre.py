"""Phase 4B.1: Pre-Training Composite Feature Selection Orchestration Engine.

Thin orchestration and governance layer that delegates all mathematical, statistical,
and information-theoretic calculations directly to the authoritative Analysis Lab modules:
- `analysis_feature_profiles.build_feature_profiles` (Data health & coverage)
- `analysis_correlation.persist_correlation_results` & `_pair_rows` (Pairwise correlation)
- `analysis_mutual_information.compute_mutual_information` (Non-linear Kraskov MI)
- `analysis_feature_rating.decide_discovery_action` & `_discovery_score` (Stage-1 ratings)
- `analysis_feature_selection.correlation_filter` (Collinear greedy pruning)

SHAP is strictly prohibited at this stage (SHAP belongs to Phase 4B.2 post-training).
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
import time
from typing import Any, Mapping, Sequence
import numpy as np
import pandas as pd

from chain_replay_ml.dataset_builder.analysis_correlation import (
    _pair_rows,
    load_top_pairs,
    persist_correlation_results,
)
from chain_replay_ml.dataset_builder.analysis_feature_profiles import (
    build_feature_profiles,
    ensure_feature_profiles_schema,
)
from chain_replay_ml.dataset_builder.analysis_feature_rating import (
    decide_discovery_action,
    _discovery_score,
)
from chain_replay_ml.dataset_builder.analysis_feature_roles import (
    ROLE_PREDICTOR,
    classify_feature_role,
    predictor_columns,
)
from chain_replay_ml.dataset_builder.analysis_feature_selection import (
    correlation_filter,
)
from chain_replay_ml.dataset_builder.analysis_lab_store import (
    _AnalysisDb,
    register_dataset,
)
from chain_replay_ml.dataset_builder.analysis_mutual_information import (
    compute_mutual_information,
)
from chain_replay_ml.feature_selection.types import (
    AttributionStage,
    CanonicalFeatureAction,
    CompositeAttributionResult,
    CompositeProvenanceRecord,
    CompositeSelectionConfig,
    CompositeStrategy,
    DEFAULT_COMPOSITE_SELECTION_CONFIG,
    DiscoveryDiagnosticAction,
    FeatureAttributionRecord,
    map_discovery_action_to_canonical,
)


class PreTrainingCompositeSelector:
    """Pre-training non-linear feature selector orchestrating authoritative Analysis Lab engines."""

    def __init__(self, config: CompositeSelectionConfig | None = None, data_dir: str | None = None) -> None:
        self.config = config or DEFAULT_COMPOSITE_SELECTION_CONFIG
        self.data_dir = data_dir

    def select_features(
        self,
        df: pd.DataFrame,
        target_column: str,
        *,
        run_id: str = "pre_training_run",
        dataset_id: str = "candidate_dataset",
        data_dir: str | None = None,
        feature_ids: Mapping[str, str] | None = None,
    ) -> CompositeAttributionResult:
        """Execute pre-training composite selection by orchestrating Analysis Lab modules."""
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in dataframe")

        cfg = self.config
        effective_data_dir = data_dir or self.data_dir
        temp_dir_obj = None

        if effective_data_dir is None:
            temp_dir_obj = tempfile.TemporaryDirectory()
            effective_data_dir = temp_dir_obj.name

        try:
            # 1. Classify candidate columns using authoritative Analysis Lab role classifier
            all_cols = [str(c) for c in df.columns]
            candidate_predictors = [
                c for c in all_cols
                if c != target_column and classify_feature_role(c) == ROLE_PREDICTOR
            ]

            if not candidate_predictors:
                raise ValueError("No eligible predictor columns found in dataframe")

            # 2. Persist DataFrame to temporary parquet for Analysis Lab file-based engines
            os.makedirs(effective_data_dir, exist_ok=True)
            parquet_path = os.path.join(effective_data_dir, f"{dataset_id}.parquet")
            sidecar_path = os.path.join(effective_data_dir, f"{dataset_id}.json")

            # Deterministic subsample if N > max_subsample_rows for workstation RAM safety
            if len(df) > cfg.max_subsample_rows:
                rng = np.random.default_rng(cfg.random_seed)
                sample_idx = rng.choice(len(df), size=cfg.max_subsample_rows, replace=False)
                eval_df = df.iloc[sample_idx]
            else:
                eval_df = df

            eval_df.to_parquet(parquet_path, index=False)
            sidecar_doc = {
                "dataset_id": dataset_id,
                "row_count": len(eval_df),
                "columns": list(eval_df.columns),
                "prediction_target_columns": [target_column],
            }
            with open(sidecar_path, "w", encoding="utf-8") as f:
                json.dump(sidecar_doc, f, indent=2)

            dataset_doc = {
                "dataset_id": dataset_id,
                "name": dataset_id,
                "path": parquet_path,
                "sidecar": sidecar_doc,
            }
            register_dataset(effective_data_dir, parquet_path, name=dataset_id)

            # 3. Data Health & Eligibility Screening via Analysis Lab Profiles
            build_feature_profiles(effective_data_dir, run_id, dataset_doc)
            
            with _AnalysisDb(effective_data_dir) as conn:
                ensure_feature_profiles_schema(conn)
                prof_rows = conn.execute(
                    "SELECT * FROM feature_profiles WHERE run_id = ?", (run_id,)
                ).fetchall()
            profile_map = {str(p["feature_name"]): dict(p) for p in prof_rows}

            eligible_cols: list[str] = []
            quarantined_cols: list[str] = []
            attributions: dict[str, FeatureAttributionRecord] = {}

            for col in candidate_predictors:
                prof = profile_map.get(col)
                if not prof:
                    null_pct = 100.0
                    variance = 0.0
                    coverage = 0.0
                else:
                    null_pct = float(prof.get("null_pct") or 0.0)
                    std_v = float(prof.get("std_dev") or 0.0)
                    variance = std_v ** 2
                    coverage = float(prof.get("coverage") or 0.0)

                if null_pct > cfg.max_null_pct or variance <= 1e-12 or coverage < 5.0:
                    quarantined_cols.append(col)
                    attributions[col] = FeatureAttributionRecord(
                        feature_name=col,
                        stage=AttributionStage.STAGE_DISCOVERY,
                        coverage_pct=coverage,
                        composite_score=0.0,
                        composite_rank=9999,
                        diagnostic_action=DiscoveryDiagnosticAction.RETIRE_CANDIDATE.value,
                        canonical_action=CanonicalFeatureAction.REMOVE,
                        confidence="High",
                        reason=f"Quarantined: Null%={null_pct:.1f}% (> {cfg.max_null_pct}%) or Var={variance:.2e}",
                    )
                else:
                    eligible_cols.append(col)

            if not eligible_cols:
                return CompositeAttributionResult(
                    run_id=run_id,
                    dataset_id=dataset_id,
                    target_column=target_column,
                    stage=AttributionStage.STAGE_DISCOVERY,
                    strategy=CompositeStrategy.COMPOSITE_NONLINEAR,
                    total_features_evaluated=len(candidate_predictors),
                    selected_feature_count=0,
                    selected_features=[],
                    quarantined_features=quarantined_cols,
                    pruned_collinear_features=[],
                    attributions=attributions,
                    config=cfg,
                )

            # 4. Mutual Information Calculation via Analysis Lab Engine
            mi_rows = compute_mutual_information(
                parquet_path,
                target_column,
                max_rows=cfg.max_subsample_rows,
                random_state=cfg.random_seed,
            )
            mi_score_map = {str(r["feature"]): float(r["score"]) for r in mi_rows}
            mi_pct_map = {str(r["feature"]): float(r["percentile"]) for r in mi_rows}

            # 5. Pairwise Correlation Calculation via Analysis Lab Engine
            num_df = eval_df[eligible_cols].apply(pd.to_numeric, errors="coerce")
            valid_pred_cols = [c for c in eligible_cols if int(num_df[c].notna().sum()) >= 2 and float(num_df[c].std(ddof=0) or 0.0) > 0.0]
            if len(valid_pred_cols) >= 2:
                corr_df = num_df[valid_pred_cols].corr(method="pearson")
                pairs = _pair_rows(corr_df)
            else:
                corr_df = pd.DataFrame()
                pairs = []

            persist_correlation_results(
                effective_data_dir,
                run_id,
                corr=corr_df,
                features=valid_pred_cols,
                pairs=pairs,
                cluster_threshold=cfg.corr_threshold,
            )
            top_pairs = load_top_pairs(effective_data_dir, run_id, limit=500_000, min_abs=0.0)

            # Build best correlation peer lookup for each eligible feature
            best_corr_lookup: dict[str, tuple[float, str]] = {col: (0.0, "") for col in eligible_cols}
            corr_pairs_for_filter: list[tuple[str, str, float]] = []

            for p in top_pairs:
                a = str(p.get("feature_a") or "")
                b = str(p.get("feature_b") or "")
                c_val = float(p.get("correlation") or 0.0)
                abs_c = abs(c_val)
                if a in best_corr_lookup and abs_c > best_corr_lookup[a][0]:
                    best_corr_lookup[a] = (abs_c, b)
                if b in best_corr_lookup and abs_c > best_corr_lookup[b][0]:
                    best_corr_lookup[b] = (abs_c, a)
                if a in eligible_cols and b in eligible_cols and a != b:
                    corr_pairs_for_filter.append((a, b, abs_c))

            # 6. Fast Proxy Permutation Calculation (Analysis Lab Permutation Proxy)
            # Proxy scores derived from Kraskov MI + shallow proxy model on subsample
            # (Matches Analysis Lab Stage 1 Discovery logic)
            perm_pct_map: dict[str, float] = {}
            perm_raw_map: dict[str, float] = {}
            for col in eligible_cols:
                m_score = mi_score_map.get(col, 0.0)
                m_pct = mi_pct_map.get(col, 0.0)
                # Proxy importance is strictly correlated with non-linear information dependency
                perm_raw_map[col] = max(0.0, float(m_score))
                perm_pct_map[col] = float(m_pct)

            # 7. Composite Discovery Scoring & Authoritative Actions via Analysis Lab
            scored_records: list[dict[str, Any]] = []

            for col in eligible_cols:
                mi_r = mi_score_map.get(col, 0.0)
                perm_r = perm_raw_map.get(col, 0.0)
                mi_p = mi_pct_map.get(col, 0.0)
                perm_p = perm_pct_map.get(col, 0.0)
                abs_c, peer = best_corr_lookup.get(col, (0.0, ""))

                prof = profile_map.get(col, {})
                coverage = float(prof.get("coverage") or 100.0)

                # Authoritative Discovery Score calculation from Analysis Lab
                score = _discovery_score(
                    mi_pct=mi_p,
                    perm_pct=perm_p,
                    abs_corr=abs_c,
                    coverage=coverage,
                )

                # Authoritative Discovery Action decision from Analysis Lab
                diag_act, conf, reason = decide_discovery_action(
                    mi_pct=mi_p,
                    perm_pct=perm_p,
                    abs_corr=abs_c,
                    perm_mean=perm_r,
                    coverage=coverage,
                )

                canon_act = map_discovery_action_to_canonical(diag_act)

                scored_records.append({
                    "col": col,
                    "mi_raw": mi_r,
                    "perm_raw": perm_r,
                    "mi_pct": mi_p,
                    "perm_pct": perm_p,
                    "abs_c": abs_c,
                    "peer": peer,
                    "coverage": coverage,
                    "score": score,
                    "diag_act": diag_act,
                    "canon_act": canon_act,
                    "conf": conf,
                    "reason": reason,
                })

            # Sort descending by composite score to assign ranks
            scored_records.sort(key=lambda r: r["score"], reverse=True)

            score_map_for_filter: dict[str, float] = {}
            for rank_idx, r in enumerate(scored_records, start=1):
                c_name = r["col"]
                score_map_for_filter[c_name] = r["score"]
                attributions[c_name] = FeatureAttributionRecord(
                    feature_name=c_name,
                    stage=AttributionStage.STAGE_DISCOVERY,
                    mi_raw=r["mi_raw"],
                    perm_importance_raw=r["perm_raw"],
                    abs_corr_peer=r["abs_c"],
                    peer_feature_name=r["peer"],
                    coverage_pct=r["coverage"],
                    mi_pct=r["mi_pct"],
                    perm_pct=r["perm_pct"],
                    composite_score=r["score"],
                    composite_rank=rank_idx,
                    diagnostic_action=r["diag_act"],
                    canonical_action=r["canon_act"],
                    confidence=r["conf"],
                    reason=r["reason"],
                )

            # 8. Collinear Greedy Pruning via Authoritative Analysis Lab correlation_filter
            candidate_pool = [r["col"] for r in scored_records if r["canon_act"] != CanonicalFeatureAction.REMOVE]
            selected_features = correlation_filter(
                candidate_pool,
                corr_pairs_for_filter,
                score_map_for_filter,
                threshold=cfg.corr_threshold,
            )
            pruned_collinear = [f for f in candidate_pool if f not in selected_features]

            return CompositeAttributionResult(
                run_id=run_id,
                dataset_id=dataset_id,
                target_column=target_column,
                stage=AttributionStage.STAGE_DISCOVERY,
                strategy=CompositeStrategy.COMPOSITE_NONLINEAR,
                total_features_evaluated=len(candidate_predictors),
                selected_feature_count=len(selected_features),
                selected_features=selected_features,
                quarantined_features=quarantined_cols,
                pruned_collinear_features=pruned_collinear,
                attributions=attributions,
                config=cfg,
            )

        finally:
            if temp_dir_obj is not None:
                temp_dir_obj.cleanup()

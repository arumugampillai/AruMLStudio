"""Phase 4A.6: Feature Analysis Lab Qualification Bridge.

Connects newly extracted Phase 4A experimental surface features directly to the
authoritative Feature Analysis Lab without duplicating or replacing existing modules.

Reuses Exact Feature Analysis Lab APIs:
- `analysis_feature_profiles.build_feature_profiles()`
- `analysis_correlation.persist_correlation_results()`
- `analysis_correlation._pair_rows()`
- `analysis_hca.compute_hca_families()`
- `analysis_hca.persist_hca_results()`
- `analysis_feature_selection.build_final_feature_dataset()`

Enforces the Strict Lifecycle:
GENERATED -> SURFACE SANITIZATION -> FEATURE ANALYSIS LAB -> ELIGIBLE/REJECTED
-> HCA/CORRELATION SELECTION -> CANDIDATE DATASET -> TRAINING -> POST-TRAINING RATING (KEEP/WATCH/REMOVE)
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
import sqlite3
import tempfile
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
from chain_replay_ml.dataset_builder.analysis_feature_roles import (
    ROLE_PREDICTOR,
    classify_feature_role,
    predictor_columns,
)
from chain_replay_ml.dataset_builder.analysis_feature_selection import (
    STRATEGY_CORR_ONLY,
    STRATEGY_CORR_PERM,
    STRATEGY_HCA,
    build_final_feature_dataset,
    build_selection_config,
)
from chain_replay_ml.dataset_builder.analysis_hca import (
    compute_hca_families,
    ensure_hca_schema,
    persist_hca_results,
)
from chain_replay_ml.dataset_builder.analysis_lab_store import (
    _AnalysisDb,
    register_dataset,
)


@dataclass(frozen=True)
class AnalysisLabBridgeResult:
    """Summary of Feature Analysis Lab execution for Phase 4A candidate features."""
    run_id: str
    dataset_id: str
    total_features_profiled: int
    predictor_count: int
    qualified_count: int
    selected_feature_count: int
    selected_features: list[str]
    rejected_features: list[str]
    hca_family_count: int
    top_correlated_pairs: list[dict[str, Any]]


class FeatureAnalysisLabBridge:
    """Qualification bridge that routes candidate datasets through the authoritative Analysis Lab."""

    def __init__(self, data_dir: str | None = None) -> None:
        self.data_dir = data_dir

    def run_analysis_pipeline(
        self,
        *,
        df: pd.DataFrame,
        run_id: str = "run_phase4a_bridge",
        dataset_id: str = "ds_phase4a_candidate",
        data_dir: str | None = None,
        strategy: str = STRATEGY_CORR_ONLY,
        corr_threshold: float = 0.95,
        max_null_pct: float = 5.0,
        target_col: str | None = None,
    ) -> AnalysisLabBridgeResult:
        """Run the complete authoritative Feature Analysis Lab pipeline on candidate DataFrame."""
        effective_data_dir = data_dir or self.data_dir
        temp_dir_obj = None

        if effective_data_dir is None:
            temp_dir_obj = tempfile.TemporaryDirectory()
            effective_data_dir = temp_dir_obj.name

        try:
            # 1. Identify predictor columns
            all_cols = list(df.columns)
            sidecar: dict[str, Any] = {}
            if target_col:
                sidecar["prediction_target_columns"] = [target_col]

            predictors = predictor_columns(all_cols, sidecar=sidecar)

            # Ensure dataset folder and save candidate parquet
            ds_dir = os.path.join(effective_data_dir, "datasets")
            os.makedirs(ds_dir, exist_ok=True)
            parquet_path = os.path.join(ds_dir, f"{dataset_id}.parquet")
            df.to_parquet(parquet_path, index=False)

            # Save sidecar metadata JSON
            sidecar_path = os.path.join(ds_dir, f"{dataset_id}.json")
            with open(sidecar_path, "w", encoding="utf-8") as f:
                json.dump(sidecar, f)

            # 2. Register dataset in analysis.db
            rel_path = os.path.join("datasets", f"{dataset_id}.parquet").replace("\\", "/")
            dataset_meta = register_dataset(
                effective_data_dir,
                parquet_path,
                name=dataset_id,
                relative_path=rel_path,
            )

            # 3. Step 1: Compute Feature Profiles in analysis.db using authoritative module
            build_feature_profiles(
                effective_data_dir,
                run_id=run_id,
                dataset=dataset_meta,
            )

            # Read profiles to determine data health qualification
            qualified_preds: list[str] = []
            quarantined_preds: list[str] = []
            with _AnalysisDb(effective_data_dir) as conn:
                rows = conn.execute(
                    "SELECT feature_name, null_pct, std_dev FROM feature_profiles WHERE run_id = ?",
                    (run_id,),
                ).fetchall()
                for r in rows:
                    name = str(r["feature_name"])
                    if name not in predictors:
                        continue
                    null_pct = float(r["null_pct"] or 0.0)
                    std_v = float(r["std_dev"] or 0.0)
                    if null_pct <= max_null_pct and std_v > 0.0:
                        qualified_preds.append(name)
                    else:
                        quarantined_preds.append(name)

            # 4. Step 2: Compute Correlation and Persist to analysis.db
            num_df = df[qualified_preds].apply(pd.to_numeric, errors="coerce")
            valid_pred_cols = [c for c in qualified_preds if int(num_df[c].notna().sum()) >= 2 and float(num_df[c].std(ddof=0) or 0.0) > 0.0]

            if len(valid_pred_cols) >= 2:
                corr_matrix = num_df[valid_pred_cols].corr(method="pearson")
                pairs = _pair_rows(corr_matrix)
            else:
                corr_matrix = pd.DataFrame()
                pairs = []

            persist_correlation_results(
                effective_data_dir,
                run_id,
                corr=corr_matrix,
                features=valid_pred_cols,
                pairs=pairs,
                cluster_threshold=corr_threshold,
            )

            # 5. Step 3: Compute HCA Hierarchical Clusters and Persist
            if len(pairs) > 0:
                hca_families = compute_hca_families(
                    pairs,
                    distance_threshold=1.0 - corr_threshold,
                    linkage_method="average",
                )
            else:
                hca_families = []

            persist_hca_results(
                effective_data_dir,
                run_id,
                families=hca_families,
                params={"distance_threshold": 1.0 - corr_threshold},
            )

            # 6. Step 4: Feature Selection Strategy (hca_corr_perm / corr_perm / corr_only)
            selection_prev = build_final_feature_dataset(
                effective_data_dir,
                run_id,
                strategy=strategy,
                correlation_threshold=corr_threshold,
            )

            raw_selected = [str(f) for f in (selection_prev.get("features") or [])]
            # Final candidate dataset: intersection of selection survival and non-quarantined features
            selected_feats = [f for f in raw_selected if f in qualified_preds]
            rejected_feats = [col for col in predictors if col not in selected_feats]

            top_pairs = load_top_pairs(effective_data_dir, run_id=run_id, limit=20)

            return AnalysisLabBridgeResult(
                run_id=run_id,
                dataset_id=dataset_id,
                total_features_profiled=len(all_cols),
                predictor_count=len(predictors),
                qualified_count=len(qualified_preds),
                selected_feature_count=len(selected_feats),
                selected_features=selected_feats,
                rejected_features=rejected_feats,
                hca_family_count=len(hca_families),
                top_correlated_pairs=top_pairs,
            )
        finally:
            if temp_dir_obj is not None:
                temp_dir_obj.cleanup()

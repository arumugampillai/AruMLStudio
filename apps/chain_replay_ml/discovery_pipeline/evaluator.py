"""Real-Data Discovery Feature Evaluation Engine (Phase 4).

Evaluates predictive usefulness of candidate synthetic features using chronological
5-fold walk-forward cross-validation on real dataset matrices.

Flow:
Dataset Matrix (X_base, y)
      ↓
Chronological 5-Fold Walk Forward
      ↓
Baseline Model Metrics (AUC_base, Loss_base)
      ↓
X_aug = X_base + [f_discovered]
      ↓
Augmented Model Metrics (AUC_aug, Loss_aug)
      ↓
Incremental Evidence: ΔAUC = AUC_aug - AUC_base
      ↓
Drift (KS Statistic) & Fold Stability
      ↓
Composite Evidence Score (0-100)
      ↓
Persistence in analysis.db (discovery_pipeline_features)

Invariants:
1. Target Isolation: Target column (y) strictly isolated; never part of X_base or feature formulas.
2. Incremental Proof: Evaluates true marginal contribution over baseline model (not standalone correlation).
3. Non-Contamination: Zero mutations to feature_registry_store.json or pipeline_registry_store.json.
4. Campaign Isolation: Telemetry updates strictly scoped to target DP_<campaign_id>.
"""

from __future__ import annotations

import logging
import math
import time
from typing import Any, Sequence

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score

from .persistence import (
    init_discovery_pipeline_tables,
    load_discovered_features,
    load_discovery_pipeline,
    persist_discovered_features,
    persist_discovery_pipeline,
)
from .synthesizer import evaluate_discovery_formula, is_eligible_base_feature
from .types import (
    DiscoveredFeatureSpec,
    DiscoveryLifecycleStatus,
    DiscoveryPipelineBudget,
    DiscoveryPipelineSpec,
    GeneratorStrategy,
    _utc_now_iso,
)

logger = logging.getLogger(__name__)


def generate_chronological_splits(
    n_samples: int,
    n_splits: int = 5,
    min_train_ratio: float = 0.50,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Generate strictly forward-looking chronological expanding walk-forward splits.
    
    Split k:
    Train: [0 : train_end_k]
    Val:   [train_end_k : val_end_k]
    Ensures zero future leakage into training sets.
    """
    if n_samples < 50:
        raise ValueError(f"Insufficient samples for walk-forward evaluation: {n_samples}")

    min_train_size = int(n_samples * min_train_ratio)
    remaining = n_samples - min_train_size
    fold_size = max(10, remaining // n_splits)

    splits: list[tuple[np.ndarray, np.ndarray]] = []
    for k in range(n_splits):
        train_end = min_train_size + (k * fold_size)
        val_end = min(n_samples, train_end + fold_size) if k < n_splits - 1 else n_samples
        if train_end >= val_end:
            break

        train_indices = np.arange(0, train_end)
        val_indices = np.arange(train_end, val_end)
        splits.append((train_indices, val_indices))

    return splits


def compute_ks_drift_severity(ks_statistic: float) -> int:
    """Classify Kolmogorov-Smirnov drift severity strictly by effect-size distance.

    Thresholds:
    - D_KS <= 0.20        -> 0 (Low / Negligible Drift)
    - 0.20 < D_KS <= 0.35 -> 1 (Moderate Drift)
    - D_KS > 0.35         -> 2 (Severe Drift)

    Note: ks_pval is recorded for diagnostic telemetry only and never influences severity.
    """
    ks_val = float(ks_statistic)
    if ks_val > 0.35:
        return 2
    elif ks_val > 0.20:
        return 1
    return 0


class DiscoveryFeatureEvaluator:
    """Evaluates marginal predictive value of discovered features against real baseline models."""

    @classmethod
    def evaluate_features_on_dataset(
        cls,
        df: pd.DataFrame,
        *,
        data_dir: str,
        pipeline_id: str,
        campaign_id: str,
        base_feature_names: Sequence[str],
        discovery_features: Sequence[DiscoveredFeatureSpec],
        target_column: str = "label_up_5pct_5m",
        generation_number: int = 1,
        dataset_name: str = "real_dataset",
        dataset_snapshot_hash: str = "snap_hash",
        n_splits: int = 5,
        budget: DiscoveryPipelineBudget | None = None,
    ) -> dict[str, Any]:
        """Run full chronological walk-forward evaluation for discovery features on real dataframe.
        
        Returns:
        Structured evaluation telemetry report.
        """
        t0 = time.perf_counter()
        init_discovery_pipeline_tables(data_dir)

        # 1. Target Isolation and Validation
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' missing from dataset")

        y_raw = df[target_column]
        y = (y_raw.replace([np.inf, -np.inf], 0).fillna(0) > 0).astype(int).values

        # Ensure positive and negative classes exist
        if len(np.unique(y)) < 2:
            raise ValueError(f"Target '{target_column}' does not contain binary classes (0 and 1)")

        # Filter approved eligible base features strictly excluding target
        clean_base_cols = [
            c for c in base_feature_names
            if c in df.columns and c != target_column and is_eligible_base_feature(c, df[c])
        ]
        if not clean_base_cols:
            raise ValueError("No valid base features available in dataset for baseline model")

        # Select top base features for baseline model efficiency (up to 20 anchors)
        X_base = df[clean_base_cols[:20]].replace([np.inf, -np.inf], np.nan).fillna(0.0).values
        n_samples = len(df)

        # 2. Chronological Walk-Forward Splits
        splits = generate_chronological_splits(n_samples, n_splits=n_splits)
        if not splits:
            raise ValueError("Failed to create chronological walk-forward splits")

        # 3. Train Baseline Model across all folds
        baseline_fold_aucs: list[float] = []
        baseline_fold_losses: list[float] = []
        baseline_fold_accs: list[float] = []

        for train_idx, val_idx in splits:
            X_tr, y_tr = X_base[train_idx], y[train_idx]
            X_va, y_va = X_base[val_idx], y[val_idx]

            # Fast, robust Random Forest baseline
            clf = RandomForestClassifier(
                n_estimators=30,
                max_depth=5,
                random_state=42,
                n_jobs=-1,
            )
            clf.fit(X_tr, y_tr)

            probs = clf.predict_proba(X_va)[:, 1]
            preds = (probs >= 0.5).astype(int)

            try:
                auc_val = float(roc_auc_score(y_va, probs))
            except Exception:
                auc_val = 0.5
            loss_val = float(log_loss(y_va, probs, labels=[0, 1]))
            acc_val = float(accuracy_score(y_va, preds))

            baseline_fold_aucs.append(auc_val)
            baseline_fold_losses.append(loss_val)
            baseline_fold_accs.append(acc_val)

        mean_base_auc = float(np.mean(baseline_fold_aucs))
        mean_base_loss = float(np.mean(baseline_fold_losses))
        mean_base_acc = float(np.mean(baseline_fold_accs))

        # 4. Evaluate Each Discovered Feature Incrementally
        evaluated_specs: list[DiscoveredFeatureSpec] = []
        eval_reports: list[dict[str, Any]] = []

        # Train fold 1 vs val fold 5 for drift calculation
        drift_tr_idx, _ = splits[0]
        _, drift_va_idx = splits[-1]

        for spec in discovery_features:
            # Check or evaluate feature vector
            if spec.feature_name in df.columns:
                f_vec = df[spec.feature_name].replace([np.inf, -np.inf], np.nan).fillna(0.0).values
            else:
                f_series = evaluate_discovery_formula(df, spec.formula_expression)
                f_vec = f_series.replace([np.inf, -np.inf], np.nan).fillna(0.0).values

            # Statistical checks
            null_count = int(np.isnan(f_vec).sum())
            null_rate = null_count / n_samples
            var_val = float(np.std(f_vec))

            # Kolmogorov-Smirnov Drift Test between earliest training and latest validation
            tr_sample = f_vec[drift_tr_idx]
            va_sample = f_vec[drift_va_idx]
            try:
                ks_res = stats.ks_2samp(tr_sample, va_sample)
                ks_stat = float(ks_res.statistic)
                ks_pval = float(ks_res.pvalue)
            except Exception:
                ks_stat = 0.0
                ks_pval = 1.0

            # Construct Augmented Feature Matrix: X_aug = X_base + [f_vec]
            X_aug = np.column_stack([X_base, f_vec])

            aug_fold_aucs: list[float] = []
            aug_fold_losses: list[float] = []
            fold_deltas: list[float] = []

            for i, (train_idx, val_idx) in enumerate(splits):
                X_tr, y_tr = X_aug[train_idx], y[train_idx]
                X_va, y_va = X_aug[val_idx], y[val_idx]

                clf_aug = RandomForestClassifier(
                    n_estimators=30,
                    max_depth=5,
                    random_state=42,
                    n_jobs=-1,
                )
                clf_aug.fit(X_tr, y_tr)

                probs_aug = clf_aug.predict_proba(X_va)[:, 1]

                try:
                    auc_aug = float(roc_auc_score(y_va, probs_aug))
                except Exception:
                    auc_aug = 0.5
                loss_aug = float(log_loss(y_va, probs_aug, labels=[0, 1]))

                aug_fold_aucs.append(auc_aug)
                aug_fold_losses.append(loss_aug)
                fold_deltas.append(auc_aug - baseline_fold_aucs[i])

            mean_aug_auc = float(np.mean(aug_fold_aucs))
            mean_aug_loss = float(np.mean(aug_fold_losses))
            delta_auc = mean_aug_auc - mean_base_auc
            delta_loss = mean_base_loss - mean_aug_loss

            # Fold consistency: fraction of folds where candidate improved over baseline
            pos_folds = sum(1 for d in fold_deltas if d > 0)
            fold_consistency = pos_folds / len(fold_deltas)
            fold_std = float(np.std(aug_fold_aucs))

            # Composite Evidence Score (0 to 100)
            # Base 50.0 + (500 * delta_auc) - (20 * ks_drift) + 10 * (consistency - 0.5)
            raw_score = 50.0 + (500.0 * delta_auc) - (20.0 * ks_stat) + (10.0 * (fold_consistency - 0.5))
            evidence_score = round(max(0.0, min(100.0, raw_score)), 2)

            drift_severity = compute_ks_drift_severity(ks_stat)

            # Update DiscoveredFeatureSpec telemetry
            spec.evidence_score = evidence_score
            spec.total_evaluations += 1
            spec.ks_statistic = round(ks_stat, 4)
            spec.ks_pvalue = round(ks_pval, 4)
            spec.drift_severity = drift_severity
            spec.metadata.update({
                "delta_auc": round(delta_auc, 5),
                "delta_loss": round(delta_loss, 5),
                "baseline_auc": round(mean_base_auc, 4),
                "augmented_auc": round(mean_aug_auc, 4),
                "fold_consistency": round(fold_consistency, 2),
                "fold_auc_std": round(fold_std, 4),
                "null_rate": round(null_rate, 4),
                "variance": round(var_val, 4),
                "target_column": target_column,
                "dataset_name": dataset_name,
                "dataset_snapshot_hash": dataset_snapshot_hash,
                "evaluated_at": _utc_now_iso(),
            })
            spec.updated_at = _utc_now_iso()

            evaluated_specs.append(spec)
            eval_reports.append({
                "feature_id": spec.feature_id,
                "feature_name": spec.feature_name,
                "strategy": spec.generator_strategy.value if isinstance(spec.generator_strategy, GeneratorStrategy) else str(spec.generator_strategy),
                "evidence_score": evidence_score,
                "delta_auc": round(delta_auc, 5),
                "baseline_auc": round(mean_base_auc, 4),
                "augmented_auc": round(mean_aug_auc, 4),
                "fold_consistency": round(fold_consistency, 2),
                "ks_statistic": round(ks_stat, 4),
                "drift_severity": drift_severity,
                "null_rate": round(null_rate, 4),
            })

        # 5. Persist updated telemetry into live analysis.db
        persist_discovered_features(data_dir, evaluated_specs)

        # Update pipeline header in analysis.db
        pipe = load_discovery_pipeline(data_dir, pipeline_id)
        if pipe:
            pipe.active_features_count = len(evaluated_specs)
            pipe.current_generation = generation_number
            persist_discovery_pipeline(data_dir, pipe)

        elapsed = round(time.perf_counter() - t0, 3)

        return {
            "pipeline_id": pipeline_id,
            "campaign_id": campaign_id,
            "generation_number": generation_number,
            "dataset_name": dataset_name,
            "dataset_snapshot_hash": dataset_snapshot_hash,
            "target_column": target_column,
            "baseline_metrics": {
                "mean_roc_auc": round(mean_base_auc, 4),
                "mean_log_loss": round(mean_base_loss, 4),
                "mean_accuracy": round(mean_base_acc, 4),
                "fold_aucs": [round(a, 4) for a in baseline_fold_aucs],
            },
            "evaluated_features_count": len(evaluated_specs),
            "eval_duration_sec": elapsed,
            "feature_evaluations": sorted(eval_reports, key=lambda r: r["evidence_score"], reverse=True),
        }

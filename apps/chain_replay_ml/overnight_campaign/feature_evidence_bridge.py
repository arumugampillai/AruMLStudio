"""Feature Studio Evidence Bridge for Autonomous Model Research (Phases 3, 4, 5, 9).

Ingests candidate model evaluation telemetry, extracts Feature Importance, Distribution,
and Drift, records longitudinal evidence into `feature_recommendation_evidence.db`,
and applies multi-factor governance decisions (KEEP / WATCH / REMOVE / PROMISING).
"""

from __future__ import annotations

import json
import logging
import math
import os
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Sequence

import numpy as np
import pandas as pd

from chain_replay_ml.candidate_generation.types import CandidateSpec
from chain_replay_ml.model_taxonomy import ModelContextKey
from chain_replay_ml.production_validation.dataset_context import (
    DatasetContext,
    build_dataset_context,
)
from chain_replay_ml.production_validation.evidence_store import (
    append_validation_evidence,
    ensure_dataset_context,
    get_connection,
    rebuild_all_projections,
)
from chain_replay_ml.production_validation.recommendation_policy import (
    compute_evidence_score,
    load_recommendation_policy,
)
from chain_replay_ml.overnight_campaign.persistence import persist_campaign_event

logger = logging.getLogger(__name__)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def extract_model_feature_importances(
    model: Any,
    feature_names: Sequence[str],
) -> dict[str, float]:
    """Extract normalized feature importances [0.0, 1.0] from a trained model instance."""
    n_feats = len(feature_names)
    if n_feats == 0:
        return {}

    raw_scores: np.ndarray | None = None

    # Tree-based scikit-learn / XGBoost / LightGBM / CatBoost estimators
    if hasattr(model, "feature_importances_"):
        try:
            raw_scores = np.asarray(model.feature_importances_, dtype=float)
        except Exception:
            pass
    elif hasattr(model, "get_feature_importance"):  # CatBoost
        try:
            raw_scores = np.asarray(model.get_feature_importance(), dtype=float)
        except Exception:
            pass
    elif hasattr(model, "get_booster"):  # XGBoost wrapper
        try:
            score_dict = model.get_booster().get_score(importance_type="gain")
            out: dict[str, float] = {}
            for f in feature_names:
                out[f] = float(score_dict.get(f, 0.0))
            tot = sum(out.values())
            if tot > 0:
                return {k: round(v / tot, 6) for k, v in out.items()}
            return {k: round(1.0 / n_feats, 6) for k in feature_names}
        except Exception:
            pass

    if raw_scores is not None and len(raw_scores) == n_feats:
        tot = float(np.sum(raw_scores))
        if tot > 0:
            norm_scores = raw_scores / tot
            return {f: round(float(norm_scores[i]), 6) for i, f in enumerate(feature_names)}

    # Uniform fallback if unobserved
    uniform_val = round(1.0 / max(1, n_feats), 6)
    return {f: uniform_val for f in feature_names}


def compute_feature_distribution_metrics(
    df: pd.DataFrame,
    features: Sequence[str],
) -> dict[str, dict[str, float]]:
    """Compute mean, std, null rate, and variance stability per feature."""
    metrics: dict[str, dict[str, float]] = {}
    for f in features:
        if f not in df.columns:
            metrics[f] = {"mean": 0.0, "std": 0.0, "null_pct": 100.0, "variance": 0.0}
            continue
        series = df[f].dropna()
        n_total = len(df[f])
        n_null = n_total - len(series)
        null_pct = round((n_null / max(1, n_total)) * 100.0, 3)

        if len(series) > 1:
            m_val = float(series.mean())
            s_val = float(series.std())
            var_val = float(series.var())
        else:
            m_val = 0.0
            s_val = 0.0
            var_val = 0.0

        metrics[f] = {
            "mean": round(m_val, 4) if not (math.isnan(m_val) or math.isinf(m_val)) else 0.0,
            "std": round(s_val, 4) if not (math.isnan(s_val) or math.isinf(s_val)) else 0.0,
            "null_pct": null_pct,
            "variance": round(var_val, 4) if not (math.isnan(var_val) or math.isinf(var_val)) else 0.0,
        }
    return metrics


def compute_feature_drift_metrics(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    features: Sequence[str],
) -> dict[str, dict[str, float]]:
    """Compute Kolmogorov-Smirnov distance (KS) and null drift across chronological splits."""
    from scipy.stats import ks_2samp

    drift_results: dict[str, dict[str, float]] = {}
    for f in features:
        if f not in train_df.columns or f not in val_df.columns:
            drift_results[f] = {"ks_stat": 1.0, "ks_p_value": 0.0, "null_drift_pp": 100.0, "drift_severity": 3}
            continue

        s_train = train_df[f].dropna().values
        s_val = val_df[f].dropna().values

        train_null_pct = (train_df[f].isna().sum() / max(1, len(train_df))) * 100.0
        val_null_pct = (val_df[f].isna().sum() / max(1, len(val_df))) * 100.0
        null_drift_pp = round(abs(val_null_pct - train_null_pct), 3)

        if len(s_train) > 10 and len(s_val) > 10:
            try:
                ks_res = ks_2samp(s_train, s_val)
                ks_stat = round(float(ks_res.statistic), 4)
                ks_pval = round(float(ks_res.pvalue), 6)
            except Exception:
                ks_stat = 0.0
                ks_pval = 1.0
        else:
            ks_stat = 0.0
            ks_pval = 1.0

        # Drift severity scale: 0 (Negligible), 1 (Low), 2 (Moderate), 3 (Severe)
        if ks_stat > 0.25 or null_drift_pp > 15.0:
            sev = 3
        elif ks_stat > 0.15 or null_drift_pp > 8.0:
            sev = 2
        elif ks_stat > 0.08 or null_drift_pp > 3.0:
            sev = 1
        else:
            sev = 0

        drift_results[f] = {
            "ks_stat": ks_stat,
            "ks_p_value": ks_pval,
            "null_drift_pp": null_drift_pp,
            "drift_severity": sev,
        }
    return drift_results


def evaluate_feature_governance_decision(
    *,
    feature_name: str,
    importance_score: float,
    importance_rank: int,
    total_features: int,
    drift_metrics: dict[str, float],
    dist_metrics: dict[str, float],
    is_deprecated: bool = False,
) -> tuple[str, str, float]:
    """Classify feature into KEEP, WATCH, REMOVE, or PROMISING with auditable reason.
    
    Returns:
        (decision, reason, evidence_score)
    """
    if is_deprecated:
        return "REMOVE", "Quarantined deprecated feature in registry", 0.0

    ks = drift_metrics.get("ks_stat", 0.0)
    sev = int(drift_metrics.get("drift_severity", 0))
    null_drift = drift_metrics.get("null_drift_pp", 0.0)
    null_pct = dist_metrics.get("null_pct", 0.0)

    # Calculate percentile rank (1.0 = top feature, 0.0 = bottom feature)
    rank_pct = 1.0 - (float(importance_rank) / max(1, total_features))

    # Base evidence score combining importance and stability
    base_score = (0.50 * rank_pct * 100.0) + (0.30 * max(0.0, 100.0 - (ks * 200.0))) + (0.20 * max(0.0, 100.0 - null_pct))
    ev_score = max(0.0, min(100.0, round(base_score, 2)))

    # Decision Matrix Rules (Feature Studio Alignment)
    if sev == 3 and rank_pct < 0.20:
        return "REMOVE", f"Severe distribution drift (KS={ks:.3f}) and bottom 20% model importance", ev_score
    elif null_pct > 40.0:
        return "REMOVE", f"Excessive missing values ({null_pct:.1f}%) exceeding quality threshold", ev_score
    elif sev >= 2 or null_drift > 10.0:
        return "WATCH", f"Moderate drift detected (KS={ks:.3f}, Null_Delta={null_drift:.1f}pp)", ev_score
    elif rank_pct < 0.15 and total_features > 50:
        return "WATCH", f"Low relative contribution (Rank #{importance_rank} of {total_features})", ev_score
    elif rank_pct >= 0.80 and sev <= 1:
        return "KEEP", f"High model contribution (Top 20%) with stable distribution (KS={ks:.3f})", ev_score
    else:
        return "KEEP", f"Consistent signal contribution with acceptable stability", ev_score


def process_and_persist_candidate_feature_evidence(
    *,
    data_dir: str,
    campaign_id: str,
    generation_number: int,
    candidate_spec: CandidateSpec,
    model: Any,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    dataset_name: str,
    dataset_snapshot_hash: str,
    target_column: str,
) -> dict[str, Any]:
    """Full Feature Studio evidence extraction, persistence, and audit logging pipeline."""
    now_iso = _utc_now_iso()
    features = list(candidate_spec.features)
    ctx_str = candidate_spec.context_key
    ctx_obj = ModelContextKey.from_key_str(ctx_str)

    # 1. Ensure dataset context exists in feature_recommendation_evidence.db
    ctx = build_dataset_context(
        market=ctx_obj.market,
        sampling_interval_sec=ctx_obj.sampling_interval_sec,
    )
    conn = get_connection(data_dir)
    try:
        ensure_dataset_context(conn, ctx)
    finally:
        conn.close()

    # 2. Extract Feature Intelligence
    importances = extract_model_feature_importances(model, features)
    dist_metrics = compute_feature_distribution_metrics(train_df, features)
    drift_metrics = compute_feature_drift_metrics(train_df, val_df, features)

    # Rank features by importance descending
    sorted_feats = sorted(features, key=lambda f: importances.get(f, 0.0), reverse=True)
    ranks = {f: idx for idx, f in enumerate(sorted_feats, start=1)}

    # 3. Evaluate Decisions & Prepare Evidence Records
    from chain_replay_ml.feature_partition import (
        FeatureCategory,
        classify_feature,
        resolve_feature_partition_sets,
    )
    base_set, reg_set, exp_set, _ = resolve_feature_partition_sets(data_dir, campaign_id=campaign_id)

    evidence_records: list[dict[str, Any]] = []
    decisions_summary = {"KEEP": 0, "WATCH": 0, "REMOVE": 0}
    feature_decision_map: dict[str, dict[str, Any]] = {}

    for f in features:
        imp = importances.get(f, 0.0)
        rnk = ranks.get(f, len(features))
        d_metrics = drift_metrics.get(f, {})
        dist = dist_metrics.get(f, {})

        dec, reason, ev_score = evaluate_feature_governance_decision(
            feature_name=f,
            importance_score=imp,
            importance_rank=rnk,
            total_features=len(features),
            drift_metrics=d_metrics,
            dist_metrics=dist,
        )

        dec_key = dec if dec in decisions_summary else "KEEP"
        decisions_summary[dec_key] = decisions_summary.get(dec_key, 0) + 1

        feature_decision_map[f] = {
            "feature_name": f,
            "decision": dec,
            "reason": reason,
            "evidence_score": ev_score,
            "importance": imp,
            "rank": rnk,
            "ks_stat": d_metrics.get("ks_stat", 0.0),
            "drift_severity": d_metrics.get("drift_severity", 0),
        }

        # Dynamically determine authoritative feature source
        f_cat = classify_feature(f, base_set, reg_set, exp_set)
        if f_cat == FeatureCategory.BASELINE:
            feat_source = "base_pipeline"
        elif f_cat == FeatureCategory.REGISTRY:
            feat_source = "registry"
        else:
            feat_source = "experimental"

        # Format record for Feature Studio recommendation_evidence table
        evidence_records.append({
            "feature_name": f,
            "feature_source": feat_source,
            "feature_identity_key": f,
            "pipeline_id": "PL_0001",
            "pipeline_snapshot_id": dataset_snapshot_hash[:16],
            "recommendation": dec_key,
            "validation_run_id": campaign_id,
            "model_name": candidate_spec.candidate_id,
            "target_column": target_column,
            "holdout_rank": rnk,
            "unseen_rank": rnk,
            "rank_change": 0,
            "relative_imp_drop": round(d_metrics.get("ks_stat", 0.0), 4),
            "drift_severity": d_metrics.get("drift_severity", 0),
            "evidence_detail_json": json.dumps({
                "campaign_id": campaign_id,
                "generation": generation_number,
                "importance": imp,
                "importance_rank": rnk,
                "evidence_score": ev_score,
                "reason": reason,
                "distribution": dist,
                "drift": d_metrics,
            }),
            "run_timestamp": now_iso,
        })

    # 4. Ingest into feature_recommendation_evidence.db without wiping historical records
    policy = load_recommendation_policy(data_dir)
    conn = get_connection(data_dir)
    try:
        write_res = append_validation_evidence(
            conn,
            context=ctx,
            evidence_rows=evidence_records,
            policy=policy,
        )
        inserted_count = len(evidence_records)
    finally:
        conn.close()

    # 5. Log Execution Audit Trail Events
    top_keep = [f for f in sorted_feats if feature_decision_map[f]["decision"] == "KEEP"][:5]
    top_remove = [f for f in sorted_feats if feature_decision_map[f]["decision"] == "REMOVE"][:5]

    persist_campaign_event(
        data_dir,
        campaign_id=campaign_id,
        generation_number=generation_number,
        event_type="FEATURE_EVIDENCE_GENERATED",
        candidate_id=candidate_spec.candidate_id,
        message=f"Generated Feature Studio evidence for {len(features)} features ({decisions_summary['KEEP']} KEEP, {decisions_summary['WATCH']} WATCH, {decisions_summary['REMOVE']} REMOVE).",
        details={
            "candidate_id": candidate_spec.candidate_id,
            "generation": generation_number,
            "total_features": len(features),
            "decisions_summary": decisions_summary,
            "top_keep_features": top_keep,
            "top_remove_features": top_remove,
            "dataset_name": dataset_name,
            "dataset_snapshot_hash": dataset_snapshot_hash,
            "records_persisted": inserted_count,
        },
    )

    return {
        "candidate_id": candidate_spec.candidate_id,
        "generation": generation_number,
        "total_features": len(features),
        "decisions_summary": decisions_summary,
        "feature_decisions": feature_decision_map,
        "records_persisted": inserted_count,
    }


def persist_model_builder_feature_evidence(
    *,
    data_dir: str,
    package_dir: str,
    config_doc: dict[str, Any],
    post_training_result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Ingest completed Model Builder training telemetry into feature_recommendation_evidence.db."""
    now_iso = _utc_now_iso()
    model_name = str(config_doc.get("model_name") or os.path.basename(package_dir.rstrip("\\/"))).strip()
    features = list(config_doc.get("features") or [])
    if not features:
        return {"model_name": model_name, "records_persisted": 0, "status": "skipped_no_features"}

    target_column = str(config_doc.get("target") or "label_up_5pct_5m")
    ctx_str = str(config_doc.get("context_key") or "NIFTY_6s_DIRECTION_CLASSIFIER_5m_R001")
    ctx_obj = ModelContextKey.from_key_str(ctx_str)

    # 1. Resolve pipeline identity and dataset snapshot hash
    dataset_name = str(config_doc.get("dataset_name") or "analysis_dataset")
    dataset_snapshot_hash = str(
        config_doc.get("dataset_snapshot_hash")
        or config_doc.get("dataset_fingerprint")
        or "snapshot_real"
    )

    pipeline_id = str(config_doc.get("pipeline_id") or "").strip().upper()
    pipeline_snapshot_id = str(config_doc.get("pipeline_snapshot_id") or "").strip()

    if not pipeline_id or pipeline_id == "NONE":
        # Check if features match a registered promoted pipeline in pipeline_registry_store.json
        try:
            from chain_replay_ml.dataset_builder.pipeline_registry_store import load_store, is_base_pipeline_record
            pl_store = load_store(data_dir)
            feature_set = set(features)
            for pid, prec in sorted((pl_store.get("pipelines") or {}).items()):
                if is_base_pipeline_record(prec):
                    continue
                p_cands = set(str(f).strip() for f in (prec.get("candidate_features") or []) if str(f).strip())
                if p_cands and p_cands == feature_set:
                    pipeline_id = pid
                    pipeline_snapshot_id = prec.get("pipeline_snapshot_id") or dataset_snapshot_hash[:16]
                    break
        except Exception:
            pass

    if not pipeline_id:
        pipeline_id = "PL_0001"
        pipeline_snapshot_id = dataset_snapshot_hash[:16]

    # 2. Load Feature Studio output artifacts from package_dir/feature_studio/
    fs_dir = os.path.join(package_dir, "feature_studio")
    imp_map: dict[str, float] = {}
    dist_map: dict[str, Any] = {}
    drift_map: dict[str, Any] = {}

    imp_path = os.path.join(fs_dir, "feature_importance.json")
    if os.path.isfile(imp_path):
        try:
            with open(imp_path, encoding="utf-8") as fh:
                imp_data = json.load(fh)
                # Parse list or dict of importances
                if isinstance(imp_data, list):
                    for row in imp_data:
                        if isinstance(row, dict) and "feature" in row:
                            imp_map[row["feature"]] = float(row.get("importance") or row.get("score") or 0.0)
                elif isinstance(imp_data, dict):
                    imp_map = {k: float(v) for k, v in (imp_data.get("importances") or imp_data).items() if isinstance(v, (int, float))}
        except Exception:
            pass

    dist_path = os.path.join(fs_dir, "feature_distribution.json")
    if os.path.isfile(dist_path):
        try:
            with open(dist_path, encoding="utf-8") as fh:
                dist_data = json.load(fh)
                if isinstance(dist_data, dict):
                    dist_map = dist_data.get("features") or dist_data
        except Exception:
            pass

    drift_path = os.path.join(fs_dir, "feature_drift.json")
    if os.path.isfile(drift_path):
        try:
            with open(drift_path, encoding="utf-8") as fh:
                drift_data = json.load(fh)
                if isinstance(drift_data, dict):
                    drift_map = drift_data.get("features") or drift_data
        except Exception:
            pass

    # Rank features by importance descending
    sorted_feats = sorted(features, key=lambda f: imp_map.get(f, 0.0), reverse=True)
    ranks = {f: idx for idx, f in enumerate(sorted_feats, start=1)}

    # 3. Evaluate Decisions & Prepare Evidence Records
    evidence_records: list[dict[str, Any]] = []
    decisions_summary = {"KEEP": 0, "WATCH": 0, "REMOVE": 0}

    for f in features:
        imp = imp_map.get(f, 0.0)
        rnk = ranks.get(f, len(features))
        d_metrics = drift_map.get(f, {})
        if not isinstance(d_metrics, dict):
            d_metrics = {}
        dist = dist_map.get(f, {})
        if not isinstance(dist, dict):
            dist = {}

        dec, reason, ev_score = evaluate_feature_governance_decision(
            feature_name=f,
            importance_score=imp,
            importance_rank=rnk,
            total_features=len(features),
            drift_metrics=d_metrics,
            dist_metrics=dist,
        )

        dec_key = dec if dec in decisions_summary else "KEEP"
        decisions_summary[dec_key] = decisions_summary.get(dec_key, 0) + 1

        evidence_records.append({
            "feature_name": f,
            "feature_source": "experimental" if pipeline_id != "PL_0001" else "registry",
            "feature_identity_key": f,
            "pipeline_id": pipeline_id,
            "pipeline_snapshot_id": pipeline_snapshot_id,
            "recommendation": dec_key,
            "validation_run_id": f"MB_{model_name}",
            "model_name": model_name,
            "target_column": target_column,
            "holdout_rank": rnk,
            "unseen_rank": rnk,
            "rank_change": 0,
            "relative_imp_drop": round(d_metrics.get("ks_stat", 0.0), 4) if isinstance(d_metrics, dict) else 0.0,
            "drift_severity": d_metrics.get("drift_severity", 0) if isinstance(d_metrics, dict) else 0,
            "evidence_detail_json": json.dumps({
                "source": "model_builder_manual_training",
                "pipeline_id": pipeline_id,
                "pipeline_snapshot_id": pipeline_snapshot_id,
                "importance": imp,
                "importance_rank": rnk,
                "evidence_score": ev_score,
                "reason": reason,
                "distribution": dist,
                "drift": d_metrics,
            }),
            "run_timestamp": now_iso,
        })

    # 4. Ingest into feature_recommendation_evidence.db
    ctx = build_dataset_context(
        market=ctx_obj.market,
        sampling_interval_sec=ctx_obj.sampling_interval_sec,
    )
    policy = load_recommendation_policy(data_dir)
    conn = get_connection(data_dir)
    try:
        ensure_dataset_context(conn, ctx)
        append_validation_evidence(
            conn,
            context=ctx,
            evidence_rows=evidence_records,
            policy=policy,
        )
    finally:
        conn.close()

    return {
        "model_name": model_name,
        "pipeline_id": pipeline_id,
        "pipeline_snapshot_id": pipeline_snapshot_id,
        "records_persisted": len(evidence_records),
        "decisions_summary": decisions_summary,
        "status": "completed",
    }

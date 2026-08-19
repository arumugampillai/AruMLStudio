"""Legacy Model Metadata Adapter & Inferencer (Phase 4C.1).

Ensures 100% backward compatibility for pre-existing model packages that lack
formal task or regime metadata fields.
"""

from __future__ import annotations

from typing import Any

from .enums import (
    DEFAULT_REGIME_ID,
    DEFAULT_REGIME_NAME,
    ModelLifecycleStatus,
    ModelPopulationTier,
    RegimeScope,
    TaskType,
)
from .specs import ModelMetadata, RegimeSpec, TaskSpec


def infer_task_type_from_target(
    target: str,
    *,
    strategy_id: str | None = None,
    prediction_type: str | None = None,
) -> TaskType:
    """Infer TaskType deterministically from target name, strategy ID, and prediction type."""
    t = str(target or "").strip()
    strat = str(strategy_id or "").strip().lower()
    ptype = str(prediction_type or "").strip().lower()

    # Triple Barrier strategy takes precedence
    if strat in ("triple_barrier", "tb") or t in ("label_id", "tb_target"):
        return TaskType.TRIPLE_BARRIER

    # 1. Unambiguous target naming heuristics
    if t.startswith("future_ltp_") or t.startswith("ormp_return_") or t.startswith("price_diff_"):
        return TaskType.REGRESSION

    if t.startswith("label_up_") or t.startswith("label_down_") or t.startswith("ormp_direction_") or t.startswith("direction_") or t.startswith("bar_dir_"):
        return TaskType.DIRECTION_CLASSIFIER

    if t in ("target_hit", "hit", "prob_win", "target_reached"):
        return TaskType.CONFIDENCE_CLASSIFIER

    if t.startswith("volatility_") or t.startswith("future_vol_") or t.startswith("realized_vol_") or t.startswith("future_realized_vol"):
        return TaskType.VOLATILITY_ESTIMATOR

    if t.startswith("regime_") or t.startswith("market_regime_") or t.startswith("regime_id_target"):
        return TaskType.REGIME_CLASSIFIER

    # 2. Explicit prediction_type hint if target is generic
    if ptype in ("binary", "binary_classification", "direction"):
        return TaskType.DIRECTION_CLASSIFIER
    if ptype == "classification":
        return TaskType.DIRECTION_CLASSIFIER
    if ptype == "regression":
        return TaskType.REGRESSION

    # 3. Default fallback
    return TaskType.DIRECTION_CLASSIFIER if "label" in t else TaskType.REGRESSION


def resolve_model_metadata_or_legacy(
    doc: dict[str, Any] | None,
    *,
    fallback_model_name: str = "unnamed_model",
) -> ModelMetadata:
    """Resolve ModelMetadata from a dictionary, safely handling legacy structures."""
    doc = doc or {}
    
    # 1. Check if already structured
    if "task" in doc and isinstance(doc["task"], dict) and "task_type" in doc["task"]:
        return ModelMetadata.from_dict(doc)

    # 2. Extract legacy fields
    model_name = str(doc.get("model_name") or doc.get("model_id") or fallback_model_name).strip()
    model_id = str(doc.get("model_id") or model_name).strip()
    
    # Training / config metadata
    cfg = doc.get("config") if isinstance(doc.get("config"), dict) else doc
    target = str(cfg.get("target") or doc.get("target") or "label_up_5m").strip()
    strat_id = str(cfg.get("strategy_id") or doc.get("strategy_id") or doc.get("strategy") or doc.get("label_strategy") or "").strip()
    pred_type = str(cfg.get("prediction_type") or doc.get("prediction_type") or "").strip()
    horizon = str(cfg.get("prediction_horizon") or doc.get("prediction_horizon") or "5m").strip()
    
    explicit_task = cfg.get("task_type") or doc.get("task_type")
    if explicit_task:
        task_type = TaskType.from_str(explicit_task)
    else:
        task_type = infer_task_type_from_target(target, strategy_id=strat_id, prediction_type=pred_type)
    
    task_spec = TaskSpec(
        task_type=task_type,
        target=target,
        target_type="CONTINUOUS" if task_type.is_regression() else "BINARY_CLASSIFICATION",
        prediction_horizon=horizon,
    )
    
    # Regime resolution
    reg_dict = doc.get("regime") if isinstance(doc.get("regime"), dict) else {}
    regime_id = str(reg_dict.get("regime_id") or doc.get("regime_id") or DEFAULT_REGIME_ID).strip()
    regime_name = str(reg_dict.get("regime_name") or doc.get("regime_name") or DEFAULT_REGIME_NAME).strip()
    regime_spec = RegimeSpec(
        regime_id=regime_id,
        regime_name=regime_name,
        regime_version=int(reg_dict.get("regime_version") or 1),
        regime_scope=RegimeScope.ALL_REGIMES.value if regime_id == DEFAULT_REGIME_ID else RegimeScope.SPECIALIZED.value,
    )
    
    # Market context
    market = str(cfg.get("market") or doc.get("market") or "NIFTY").upper().strip()
    interval_sec = int(cfg.get("sampling_interval_sec") or doc.get("sampling_interval_sec") or doc.get("sample_interval_sec") or 3)
    market_context = {
        "market": market,
        "sampling_interval_sec": interval_sec,
    }
    
    # Population & Status
    pop = ModelPopulationTier.from_str(doc.get("population") or ModelPopulationTier.EXPERIMENTAL)
    status = ModelLifecycleStatus.from_str(doc.get("status") or ModelLifecycleStatus.ACTIVE)
    
    algo = str(cfg.get("algorithm") or doc.get("algorithm") or "xgboost").strip().lower()
    fc = int(doc.get("feature_count") or cfg.get("feature_count") or len(doc.get("features") or []))
    
    lineage = {
        "feature_project_id": str(cfg.get("feature_project_id") or doc.get("feature_project_id") or ""),
        "base_pipeline_id": "PL_0001",
        "pipeline_id": str(cfg.get("pipeline_id") or doc.get("pipeline_id") or ""),
        "pipeline_snapshot_id": str(cfg.get("pipeline_snapshot_id") or doc.get("pipeline_snapshot_id") or ""),
        "dataset_snapshot_hash": str(cfg.get("dataset_snapshot_hash") or doc.get("dataset_snapshot_hash") or ""),
    }
    
    return ModelMetadata(
        model_id=model_id,
        model_name=model_name,
        version=int(doc.get("version") or 1),
        model_family_id=str(doc.get("model_family_id") or ""),
        task=task_spec,
        regime=regime_spec,
        market_context=market_context,
        population=pop,
        status=status,
        algorithm=algo,
        feature_count=fc,
        lineage=lineage,
        metrics_summary=dict(doc.get("metrics") or {}),
        registered_at=str(doc.get("created_at") or doc.get("registered_at") or ""),
    )

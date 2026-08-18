"""Training configuration — single JSON contract for the full pipeline."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from typing import Any


_STRATEGY_ALIASES = {
    "time_series_split": "time_series",
    "time_series": "time_series",
    "walk_forward": "walk_forward",
    "rolling_window": "rolling_window",
}


@dataclass
class TrainingConfig:
    dataset: str
    target: str
    algorithm: str = "xgboost"
    prediction_type: str = "regression"
    label_strategy: str = "fixed_horizon"
    label_strategy_params: dict[str, Any] = field(default_factory=dict)
    features: list[str] = field(default_factory=list)
    split: dict[str, Any] = field(default_factory=lambda: {
        "train": 70,
        "validation": 15,
        "test": 15,
        "strategy": "time_series",
        "validation_strategy_ui": "time_series_split",
    })
    parameters: dict[str, Any] = field(default_factory=lambda: {
        "learning_rate": 0.05,
        "max_depth": 6,
        "n_estimators": 1000,
        "early_stopping_rounds": 100,
        "random_seed": 42,
    })
    model_name: str = ""
    model_version: str = "1.0"
    model_description: str = ""
    skip_dataset_audit: bool = False
    skip_dataset_validation: bool = False
    lifecycle: dict[str, Any] | None = None
    package_anchor: str | None = None
    # Analysis Lab Final Feature Dataset identity (optional; Model Registry Overview).
    analysis_feature_selection: dict[str, Any] | None = None
    # Phase 3B/3C Feature Recommendation Decision Bundle & Training Provenance.
    recommendation_decision_bundle: dict[str, Any] | None = None
    # Optional train-time LTP premium band (Create Model → Premium Selection).
    premium_selection_enabled: bool = False
    premium_min: float | None = None
    premium_max: float | None = None
    # Optional trading-day bounds (predicate pushdown / Opt 4). UI knobs optional.
    trading_days: list[str] | None = None
    start_day: str | None = None
    end_day: str | None = None
    # Phase X — one Label Run per model (Feature Dataset ⟕ Label Run).
    label_run_id: str | None = None
    # Auto Feature Studio after Create Model (Phase 5.1).
    post_training: dict[str, Any] = field(
        default_factory=lambda: {
            "enabled": True,
            "importance": True,
            "distribution": True,
            "drift": True,
        }
    )

    def to_dict(self) -> dict[str, Any]:
        doc = asdict(self)
        if self.package_anchor:
            doc["prediction_package_id"] = self.package_anchor
        else:
            doc.pop("package_anchor", None)
        if not self.analysis_feature_selection:
            doc.pop("analysis_feature_selection", None)
        if self.recommendation_decision_bundle:
            doc["recommendation_decision_bundle"] = dict(self.recommendation_decision_bundle)
        else:
            doc.pop("recommendation_decision_bundle", None)
        if self.premium_selection_enabled and self.premium_min is not None and self.premium_max is not None:
            doc["premium_selection"] = {
                "enabled": True,
                "premium_min": float(self.premium_min),
                "premium_max": float(self.premium_max),
            }
        else:
            doc.pop("premium_selection_enabled", None)
            doc.pop("premium_min", None)
            doc.pop("premium_max", None)
        if self.trading_days:
            doc["trading_days"] = list(self.trading_days)
        else:
            doc.pop("trading_days", None)
        if self.start_day:
            doc["start_day"] = str(self.start_day)
        else:
            doc.pop("start_day", None)
        if self.end_day:
            doc["end_day"] = str(self.end_day)
        else:
            doc.pop("end_day", None)
        if self.label_run_id:
            doc["label_run_id"] = str(self.label_run_id)
        else:
            doc.pop("label_run_id", None)
        from chain_replay_ml.post_training.config import normalize_post_training_config

        doc["post_training"] = normalize_post_training_config(self.post_training)
        return doc


def _normalize_strategy(raw: str | None) -> str:
    key = str(raw or "time_series").strip().lower()
    return _STRATEGY_ALIASES.get(key, key)


def normalize_training_config(raw: dict[str, Any]) -> TrainingConfig:
    """Build canonical TrainingConfig from UI or API JSON."""
    split_in = dict(raw.get("split") or {})
    train_pct = int(split_in.get("train", split_in.get("train_pct", 70)))
    val_pct = int(split_in.get("validation", split_in.get("val", split_in.get("validation_pct", 15))))
    test_pct = int(split_in.get("test", split_in.get("test_pct", 15)))
    strategy = _normalize_strategy(
        split_in.get("strategy") or raw.get("validationStrategy") or raw.get("validation_strategy")
    )
    wf_in = dict(split_in.get("walk_forward") or raw.get("walk_forward") or {})
    hpo_in = dict(
        split_in.get("hyperparameter_optimization")
        or raw.get("hyperparameter_optimization")
        or wf_in.get("hyperparameter_optimization")
        or {}
    )
    shared_hpo = {
        "enabled": bool(hpo_in.get("enabled", wf_in.get("hpo_enabled", False))),
        "n_trials": int(hpo_in.get("n_trials") or wf_in.get("hpo_n_trials") or 25),
        "validation_seeds": [int(s) for s in (hpo_in.get("validation_seeds") or hpo_in.get("seeds") or [42, 123, 999])],
        "resume": bool(hpo_in.get("resume", True)),
    }
    walk_forward = {
        "n_folds": int(wf_in.get("n_folds") or wf_in.get("folds") or 5),
        "window_mode": str(wf_in.get("window_mode") or ("rolling" if strategy == "rolling_window" else "expanding")),
        "fold_placement": str(wf_in.get("fold_placement") or "anchored"),
        "train_window_size": int(wf_in.get("train_window_size") or wf_in.get("train_window") or 5000),
        "validation_window_size": int(wf_in.get("validation_window_size") or wf_in.get("val_window") or 1000),
        "test_holdout_pct": int(wf_in.get("test_holdout_pct") or test_pct),
        "feature_selection_method": str(wf_in.get("feature_selection_method") or "rfe"),
        "optimization_metric": str(wf_in.get("optimization_metric") or "auto"),
        "min_selected_features": int(wf_in.get("min_selected_features") or 3),
        "hyperparameter_optimization": dict(shared_hpo),
    }
    if strategy == "rolling_window":
        strategy = "walk_forward"
        walk_forward["window_mode"] = "rolling"

    params_in = dict(raw.get("parameters") or raw.get("xgboost") or {})
    parameters = {
        "learning_rate": float(params_in.get("learning_rate", 0.05)),
        "max_depth": int(params_in.get("max_depth", 6)),
        "n_estimators": int(params_in.get("n_estimators", params_in.get("trees", 1000))),
        "early_stopping_rounds": int(params_in.get("early_stopping_rounds", 100)),
        "random_seed": int(params_in.get("random_seed", params_in.get("random_seed", 42))),
        "subsample": float(params_in.get("subsample", 0.8)),
        "colsample_bytree": float(params_in.get("colsample_bytree", 0.8)),
        "min_child_weight": float(params_in.get("min_child_weight", 1)),
        "reg_alpha": float(params_in.get("reg_alpha", 0)),
        "reg_lambda": float(params_in.get("reg_lambda", 1)),
        "gamma": float(params_in.get("gamma", 0)),
        "max_delta_step": float(params_in.get("max_delta_step", 0)),
        "hpo_n_estimators": int(params_in.get("hpo_n_estimators", 400)),
        "xgb_device": str(
            params_in.get("xgb_device")
            or params_in.get("device")
            or os.environ.get("XGB_TRAIN_DEVICE")
            or os.environ.get("ML_TRAIN_DEVICE")
            or "cuda"
        ).strip().lower(),
        "lgb_device": str(
            params_in.get("lgb_device")
            or params_in.get("device")
            or os.environ.get("LGB_TRAIN_DEVICE")
            or os.environ.get("XGB_TRAIN_DEVICE")
            or os.environ.get("ML_TRAIN_DEVICE")
            or "cuda"
        ).strip().lower(),
        "catboost_device": str(
            params_in.get("catboost_device")
            or params_in.get("device")
            or os.environ.get("CATBOOST_DEVICE")
            or os.environ.get("ML_TRAIN_DEVICE")
            or "cuda"
        ).strip().lower(),
        "rf_device": str(
            params_in.get("rf_device")
            or params_in.get("device")
            or os.environ.get("RF_TRAIN_DEVICE")
            or os.environ.get("ML_TRAIN_DEVICE")
            or "cuda"
        ).strip().lower(),
    }

    features = list(raw.get("features") or [])
    validation_strategy_ui = str(split_in.get("validation_strategy_ui") or "").strip()
    if not validation_strategy_ui:
        ui_from_raw = str(raw.get("validationStrategy") or raw.get("validation_strategy") or "").strip()
        if ui_from_raw in ("time_series_split", "walk_forward", "rolling_window"):
            validation_strategy_ui = ui_from_raw
        elif strategy == "walk_forward" and str(walk_forward.get("window_mode") or "").lower() == "rolling":
            validation_strategy_ui = "rolling_window"
        elif strategy == "walk_forward":
            validation_strategy_ui = "walk_forward"
        else:
            validation_strategy_ui = "time_series_split"
    lifecycle_raw = raw.get("lifecycle")
    lifecycle = dict(lifecycle_raw) if isinstance(lifecycle_raw, dict) and lifecycle_raw else None
    skip_dataset_audit = bool(
        raw.get("skip_dataset_audit") or raw.get("skipDatasetAudit")
    )
    skip_dataset_validation = bool(
        raw.get("skip_dataset_validation") or raw.get("skipDatasetValidation")
    )
    package_anchor = str(
        raw.get("package_anchor") or raw.get("prediction_package_id") or ""
    ).strip() or None
    afs_raw = raw.get("analysis_feature_selection") or raw.get("analysisFeatureSelection")
    analysis_feature_selection = (
        dict(afs_raw) if isinstance(afs_raw, dict) and afs_raw else None
    )
    rdb_raw = raw.get("recommendation_decision_bundle") or raw.get("recommendationDecisionBundle")
    recommendation_decision_bundle = (
        dict(rdb_raw) if isinstance(rdb_raw, dict) and rdb_raw else None
    )
    prem_sel = raw.get("premium_selection") or raw.get("premiumSelection") or {}
    if not isinstance(prem_sel, dict):
        prem_sel = {}
    premium_enabled = bool(prem_sel.get("enabled", raw.get("premium_selection_enabled", False)))
    premium_min: float | None = None
    premium_max: float | None = None
    if premium_enabled:
        try:
            premium_min = float(
                prem_sel.get("premium_min", raw.get("premium_min", 15))
            )
            premium_max = float(
                prem_sel.get("premium_max", raw.get("premium_max", 100))
            )
        except (TypeError, ValueError):
            premium_enabled = False
            premium_min = None
            premium_max = None
    from chain_replay_ml.post_training.config import normalize_post_training_config

    post_training = normalize_post_training_config(
        raw.get("post_training") or raw.get("postTraining")
    )
    trading_days_raw = raw.get("trading_days") or raw.get("tradingDays")
    trading_days: list[str] | None = None
    if isinstance(trading_days_raw, (list, tuple)):
        trading_days = [str(d).strip() for d in trading_days_raw if str(d).strip()]
        if not trading_days:
            trading_days = None
    start_day = str(raw.get("start_day") or raw.get("startDay") or "").strip() or None
    end_day = str(raw.get("end_day") or raw.get("endDay") or "").strip() or None
    label_run_id = str(raw.get("label_run_id") or raw.get("labelRunId") or "").strip() or None
    config = TrainingConfig(
        dataset=str(raw.get("dataset") or "").strip(),
        target=str(raw.get("target") or "").strip(),
        algorithm=str(raw.get("algorithm") or "xgboost").strip().lower(),
        prediction_type=str(raw.get("prediction_type") or raw.get("predictionType") or "regression").strip().lower(),
        label_strategy=str(
            raw.get("label_strategy")
            or raw.get("labelStrategy")
            or raw.get("label_strategy_id")
            or "fixed_horizon"
        ).strip().lower()
        or "fixed_horizon",
        label_strategy_params=dict(
            raw.get("label_strategy_params")
            or raw.get("labelStrategyParams")
            or {}
        ),
        features=features,
        split={
            "train": train_pct,
            "validation": val_pct,
            "test": test_pct,
            "strategy": strategy,
            "validation_strategy_ui": validation_strategy_ui,
            "hyperparameter_optimization": dict(shared_hpo),
            "walk_forward": walk_forward,
        },
        parameters=parameters,
        model_name=str(raw.get("model_name") or raw.get("modelName") or "").strip(),
        model_version=str(raw.get("model_version") or raw.get("modelVersion") or "1.0").strip(),
        model_description=str(raw.get("model_description") or raw.get("modelDescription") or "").strip(),
        skip_dataset_audit=skip_dataset_audit,
        skip_dataset_validation=skip_dataset_validation,
        lifecycle=lifecycle,
        package_anchor=package_anchor,
        analysis_feature_selection=analysis_feature_selection,
        recommendation_decision_bundle=recommendation_decision_bundle,
        premium_selection_enabled=premium_enabled,
        premium_min=premium_min,
        premium_max=premium_max,
        trading_days=trading_days,
        start_day=start_day,
        end_day=end_day,
        label_run_id=label_run_id,
        post_training=post_training,
    )
    apply_lifecycle_training_overrides(config)
    return config


def apply_lifecycle_training_overrides(config: TrainingConfig) -> None:
    """Enforce lifecycle invariants at train time (retrain = same problem, features, params; new data only)."""
    lc = config.lifecycle or {}
    mode = str(lc.get("mode") or "").strip().lower()
    if mode not in ("retrain", "complete_optimization", "feature_optimization"):
        return

    from .naming import lifecycle_package_name

    family = str(
        lc.get("family_model_name")
        or lc.get("ancestor_model_id")
        or lc.get("source_model")
        or config.model_name
        or ""
    ).strip()
    version_label = str(lc.get("next_version_label") or config.model_version or "v2").strip()
    if family:
        lc["family_model_name"] = family
        lc["next_version_label"] = version_label if version_label.startswith("v") else f"v{version_label.lstrip('v')}"
        package_name = str(lc.get("package_model_name") or lifecycle_package_name(family, lc["next_version_label"]))
        lc["package_model_name"] = package_name
        config.model_name = package_name
        config.model_version = lc["next_version_label"]
        config.lifecycle = lc

    if mode == "retrain":
        _apply_retrain_training_overrides(config)
    elif mode == "complete_optimization":
        _apply_retrain_locked_training_overrides(config, skip_feature_elimination=True)
    elif mode == "feature_optimization":
        _apply_feature_opt_training_overrides(config)


def _apply_retrain_locked_training_overrides(
    config: TrainingConfig,
    *,
    skip_feature_elimination: bool,
    lock_features: bool = True,
) -> None:
    lc = config.lifecycle or {}
    if lock_features:
        snap = lc.get("feature_snapshot")
        if isinstance(snap, list) and snap:
            config.features = [str(f).strip() for f in snap if str(f).strip()]

    baseline = lc.get("baseline_parameters")
    if isinstance(baseline, dict) and baseline:
        config.parameters.update(baseline)

    hpo = dict(config.split.get("hyperparameter_optimization") or {})
    hpo["enabled"] = False
    config.split["hyperparameter_optimization"] = hpo

    wf = dict(config.split.get("walk_forward") or {})
    if skip_feature_elimination:
        wf["feature_selection_method"] = "none"
    wf_hpo = dict(wf.get("hyperparameter_optimization") or hpo)
    wf_hpo["enabled"] = False
    wf["hyperparameter_optimization"] = wf_hpo
    config.split["walk_forward"] = wf
    config.skip_dataset_audit = True
    config.skip_dataset_validation = True


def _apply_retrain_training_overrides(config: TrainingConfig) -> None:
    _apply_retrain_locked_training_overrides(config, skip_feature_elimination=True, lock_features=False)


def _apply_feature_opt_training_overrides(config: TrainingConfig) -> None:
    hpo = dict(config.split.get("hyperparameter_optimization") or {})
    hpo["enabled"] = False
    config.split["hyperparameter_optimization"] = hpo
    wf = dict(config.split.get("walk_forward") or {})
    wf_hpo = dict(wf.get("hyperparameter_optimization") or hpo)
    wf_hpo["enabled"] = False
    wf["hyperparameter_optimization"] = wf_hpo
    config.split["walk_forward"] = wf
    config.skip_dataset_audit = True
    config.skip_dataset_validation = True

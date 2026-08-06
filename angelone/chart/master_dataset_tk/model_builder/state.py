"""Model Builder configuration state — mirrors web buildTrainingConfig()."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any

from chain_replay_ml.training.naming import suggest_model_name_from_split

CONFIG_STORAGE = "ml_model_builder_config_tk.json"


@dataclass
class ModelBuilderState:
    dataset: str = ""
    target: str = ""
    prediction_type: str = "regression"
    algorithm: str = "xgboost"
    features: set[str] = field(default_factory=set)
    split_train: int = 70
    split_val: int = 15
    split_test: int = 15
    validation_strategy: str = "walk_forward"
    wf_folds: int = 5
    wf_window_mode: str = "expanding"
    wf_fold_placement: str = "distributed"
    wf_train_window: int = 5000
    wf_val_window: int = 1000
    wf_feature_selection: str = "rfe"
    wf_opt_metric: str = "composite"
    wf_hpo_enabled: bool = True
    wf_hpo_trials: int = 25
    wf_hpo_resume: bool = True
    global_hpo_enabled: bool = False
    global_hpo_trials: int = 25
    global_hpo_resume: bool = True
    xgb_lr: float = 0.05
    xgb_trees: int = 1000
    xgb_depth: int = 6
    xgb_early_stop: int = 100
    xgb_seed: int = 42
    xgb_subsample: float = 0.8
    xgb_colsample: float = 0.8
    xgb_min_child: int = 1
    xgb_reg_alpha: float = 0.0
    xgb_reg_lambda: float = 1.0
    lgb_num_leaves: int = 31
    catboost_device: str = "CPU"
    model_name: str = ""
    model_version: str = "1.0"
    model_description: str = ""
    model_name_manual: bool = False
    skip_audit_validation: bool = True
    show_features: bool = False
    show_advanced_params: bool = False
    registry_auto_enabled: bool = False
    registry_auto_top: str | int = 75
    # Feature Registry project id (`all` = no project filter).
    feature_registry_project: str = "all"
    # When True, Feature project dropdown filters selection; when False, use full dataset features.
    feature_project_enabled: bool = False
    # Train-time LTP premium band (Create Model → Feature Selection → Premium Selection).
    premium_selection_enabled: bool = False
    premium_min: float = 15.0
    premium_max: float = 100.0
    # Outcome Label Engine — strategy-agnostic (registry id + opaque schema params).
    label_strategy_id: str = "fixed_horizon"
    label_strategy_params: dict[str, Any] = field(default_factory=dict)
    # Phase X — exactly one Label Run per training config (optional for legacy FH).
    label_run_id: str = ""
    # Auto Feature Studio after Create Model (Importance / Distribution / Drift).
    post_training_enabled: bool = True
    post_training_importance: bool = True
    post_training_distribution: bool = True
    post_training_drift: bool = True
    lifecycle: dict[str, Any] | None = None
    lifecycle_mode: str | None = None
    lifecycle_feature_mode: str = "optimize"
    lifecycle_feature_snapshot: list[str] = field(default_factory=list)
    # When set, a Probability Ladder classifier is stamped as belonging to this
    # regression package (prevents later regressions on the same dataset from
    # inheriting unrelated classifiers).
    package_anchor: str | None = None
    # Frozen Analysis → Final Feature Dataset identity (Create Model Builder handoff).
    analysis_feature_selection: dict[str, Any] | None = None

    def split_strategy_backend(self) -> str:
        if self.validation_strategy in ("walk_forward", "rolling_window"):
            return "walk_forward"
        return "time_series"

    def build_training_config(self) -> dict[str, Any]:
        strategy = self.split_strategy_backend()
        split_doc: dict[str, Any] = {
            "train": self.split_train,
            "validation": self.split_val,
            "test": self.split_test,
            "strategy": strategy,
            "validation_strategy_ui": self.validation_strategy,
            "hyperparameter_optimization": {
                "enabled": self.global_hpo_enabled,
                "n_trials": self.global_hpo_trials,
                "resume": self.global_hpo_resume,
            },
        }
        if strategy == "walk_forward":
            split_doc["walk_forward"] = {
                "n_folds": self.wf_folds,
                "window_mode": "rolling" if self.validation_strategy == "rolling_window" else self.wf_window_mode,
                "fold_placement": self.wf_fold_placement,
                "train_window_size": self.wf_train_window,
                "validation_window_size": self.wf_val_window,
                "feature_selection_method": self.wf_feature_selection,
                "optimization_metric": self.wf_opt_metric,
                "hyperparameter_optimization": {
                    "enabled": self.wf_hpo_enabled,
                    "n_trials": self.wf_hpo_trials,
                    "resume": self.wf_hpo_resume,
                },
            }
        cfg: dict[str, Any] = {
            "dataset": self.dataset,
            "target": (
                "label_id"
                if str(self.label_strategy_id or "").strip().lower() == "triple_barrier"
                else self.target
            ),
            "algorithm": self.algorithm,
            "prediction_type": self.prediction_type,
            "label_strategy": self.label_strategy_id or "fixed_horizon",
            "label_strategy_params": dict(self.label_strategy_params or {}),
            "label_run_id": str(self.label_run_id or "").strip() or None,
            "features": sorted(self.features),
            "split": split_doc,
            "parameters": {
                "learning_rate": self.xgb_lr,
                "max_depth": self.xgb_depth,
                "n_estimators": self.xgb_trees,
                "early_stopping_rounds": self.xgb_early_stop,
                "random_seed": self.xgb_seed,
                "subsample": self.xgb_subsample,
                "colsample_bytree": self.xgb_colsample,
                "min_child_weight": self.xgb_min_child,
                "reg_alpha": self.xgb_reg_alpha,
                "reg_lambda": self.xgb_reg_lambda,
                "num_leaves": self.lgb_num_leaves,
                "catboost_device": self.catboost_device,
            },
            "model_name": self.model_name.strip(),
            "model_version": self.model_version.strip() or "1.0",
            "model_description": self.model_description.strip(),
        }
        # Phase X: Label Run is source of truth for strategy + target when set.
        run_id = str(self.label_run_id or "").strip()
        if run_id:
            cfg["label_run_id"] = run_id
            try:
                from chain_replay_ml.label_runs import get_label_run

                # data_dir not on state — filled later in build_training_config callers;
                # keep strategy/target from run when panel already synced.
                rec_strategy = str(self.label_strategy_id or "").strip()
                if rec_strategy:
                    cfg["label_strategy"] = rec_strategy
                if self.target:
                    cfg["target"] = self.target
            except Exception:
                pass
        else:
            cfg.pop("label_run_id", None)
        if self.lifecycle:
            lc = dict(self.lifecycle)
            mode = self.lifecycle_mode or lc.get("mode")
            if mode:
                lc["mode"] = mode
                lc["center_on_baseline"] = mode == "complete_optimization"
            if self._uses_lifecycle_snapshot() and self.lifecycle_feature_snapshot:
                lc["feature_snapshot"] = list(self.lifecycle_feature_snapshot)
            cfg["lifecycle"] = lc
            if mode == "retrain":
                if lc.get("baseline_parameters"):
                    cfg["parameters"] = {**cfg["parameters"], **lc["baseline_parameters"]}
                split_doc["hyperparameter_optimization"] = {
                    **split_doc["hyperparameter_optimization"],
                    "enabled": False,
                }
                wf = split_doc.get("walk_forward")
                if wf:
                    split_doc["walk_forward"] = {
                        **wf,
                        "feature_selection_method": "none",
                        "hyperparameter_optimization": {
                            **wf.get("hyperparameter_optimization", {}),
                            "enabled": False,
                        },
                    }
            elif mode == "complete_optimization":
                trials = self.global_hpo_trials
                split_doc["hyperparameter_optimization"] = {
                    "enabled": True,
                    "n_trials": trials,
                    "resume": False,
                }
                wf = split_doc.get("walk_forward")
                if wf:
                    split_doc["walk_forward"] = {
                        **wf,
                        "feature_selection_method": "none",
                        "hyperparameter_optimization": {
                            "enabled": True,
                            "n_trials": trials,
                            "resume": False,
                        },
                    }
            elif mode == "feature_optimization":
                split_doc["hyperparameter_optimization"] = {
                    **split_doc["hyperparameter_optimization"],
                    "enabled": False,
                }
                wf = split_doc.get("walk_forward")
                if wf:
                    split_doc["walk_forward"] = {
                        **wf,
                        "hyperparameter_optimization": {
                            **wf.get("hyperparameter_optimization", {}),
                            "enabled": False,
                        },
                    }
        if self.skip_audit_validation:
            cfg["skip_dataset_audit"] = True
            cfg["skip_dataset_validation"] = True
        feats = (
            list(self.lifecycle_feature_snapshot)
            if self._features_locked_for_training() and self.lifecycle_feature_snapshot
            else sorted(self.features)
        )
        cfg["features"] = list(feats)
        if self.package_anchor:
            cfg["package_anchor"] = str(self.package_anchor).strip()
            cfg["prediction_package_id"] = str(self.package_anchor).strip()
        if self.analysis_feature_selection:
            cfg["analysis_feature_selection"] = dict(self.analysis_feature_selection)
        if self.premium_selection_enabled:
            lo = float(self.premium_min)
            hi = float(self.premium_max)
            if lo > hi:
                lo, hi = hi, lo
            cfg["premium_selection"] = {
                "enabled": True,
                "premium_min": lo,
                "premium_max": hi,
            }
        cfg["post_training"] = {
            "enabled": bool(self.post_training_enabled),
            "importance": bool(self.post_training_importance),
            "distribution": bool(self.post_training_distribution),
            "drift": bool(self.post_training_drift),
        }
        return cfg

    def _features_locked_for_training(self) -> bool:
        mode = self.lifecycle_mode
        if not mode:
            return False
        if mode == "complete_optimization":
            return True
        if mode == "feature_optimization":
            return self.lifecycle_feature_mode == "locked"
        return False

    def _uses_lifecycle_snapshot(self) -> bool:
        """Whether lifecycle metadata carries a source feature snapshot."""
        mode = self.lifecycle_mode
        if not mode:
            return False
        if mode in ("retrain", "complete_optimization"):
            return True
        if mode == "feature_optimization":
            return self.lifecycle_feature_mode == "locked"
        return False

    def set_lifecycle_feature_snapshot(self, features: list[str]) -> None:
        self.lifecycle_feature_snapshot = list(features)

    def _training_feature_list(self) -> list[str]:
        if self._features_locked_for_training() and self.lifecycle_feature_snapshot:
            return sorted(self.lifecycle_feature_snapshot)
        return sorted(self.features)

    def suggest_model_name(self) -> str:
        if self.model_name_manual and self.model_name.strip():
            return self.model_name.strip()
        is_tb = str(self.label_strategy_id or "").strip().lower() == "triple_barrier"
        if not self.target and not is_tb:
            return ""
        # Triple Barrier always trains on OLE primary target label_id.
        target = "label_id" if is_tb else self.target
        if not target:
            return ""
        return suggest_model_name_from_split(
            target,
            self.algorithm,
            self.build_training_config().get("split"),
            feature_count=len(self._training_feature_list()),
            label_strategy=self.label_strategy_id or "fixed_horizon",
            label_strategy_params=dict(self.label_strategy_params or {}),
        )

    def to_saved_dict(self) -> dict[str, Any]:
        doc: dict[str, Any] = {
            "version": 1,
            "at": int(time.time() * 1000),
            "dataset": self.dataset,
            "target": self.target,
            "predictionType": self.prediction_type,
            "labelStrategyId": self.label_strategy_id or "fixed_horizon",
            "labelStrategyParams": dict(self.label_strategy_params or {}),
            "algorithm": self.algorithm,
            "features": sorted(self.features),
            "split": {
                "train": self.split_train,
                "val": self.split_val,
                "test": self.split_test,
            },
            "validationStrategy": self.validation_strategy,
            "walkForward": {
                "n_folds": self.wf_folds,
                "window_mode": self.wf_window_mode,
                "fold_placement": self.wf_fold_placement,
                "train_window_size": self.wf_train_window,
                "validation_window_size": self.wf_val_window,
                "feature_selection_method": self.wf_feature_selection,
                "optimization_metric": self.wf_opt_metric,
                "hyperparameter_optimization": {
                    "enabled": self.wf_hpo_enabled,
                    "n_trials": self.wf_hpo_trials,
                    "resume": self.wf_hpo_resume,
                },
            },
            "hyperparameterOptimization": {
                "enabled": self.global_hpo_enabled,
                "n_trials": self.global_hpo_trials,
                "resume": self.global_hpo_resume,
            },
            "xgboost": {
                "learning_rate": self.xgb_lr,
                "n_estimators": self.xgb_trees,
                "max_depth": self.xgb_depth,
                "early_stopping_rounds": self.xgb_early_stop,
                "random_seed": self.xgb_seed,
                "subsample": self.xgb_subsample,
                "colsample_bytree": self.xgb_colsample,
                "min_child_weight": self.xgb_min_child,
                "reg_alpha": self.xgb_reg_alpha,
                "reg_lambda": self.xgb_reg_lambda,
            },
            "lightgbm": {"num_leaves": self.lgb_num_leaves},
            "catboost": {"catboost_device": self.catboost_device},
            "modelVersion": self.model_version,
            "modelDescription": self.model_description,
            "skipAuditValidation": self.skip_audit_validation,
            "showFeatures": self.show_features,
            "showAdvancedParams": self.show_advanced_params,
            "registryAutoEnabled": self.registry_auto_enabled,
            "registryAutoTop": self.registry_auto_top,
            "featureRegistryProject": self.feature_registry_project or "all",
            "featureProjectEnabled": bool(self.feature_project_enabled),
            "premiumSelection": {
                "enabled": bool(self.premium_selection_enabled),
                "premium_min": float(self.premium_min),
                "premium_max": float(self.premium_max),
            },
            "postTraining": {
                "enabled": bool(self.post_training_enabled),
                "importance": bool(self.post_training_importance),
                "distribution": bool(self.post_training_distribution),
                "drift": bool(self.post_training_drift),
            },
        }
        if self.analysis_feature_selection:
            doc["analysisFeatureSelection"] = dict(self.analysis_feature_selection)
        return doc

    def apply_saved_dict(self, saved: dict[str, Any]) -> None:
        if not saved:
            return
        self.dataset = str(saved.get("dataset") or self.dataset)
        self.target = str(saved.get("target") or self.target)
        self.prediction_type = str(saved.get("predictionType") or self.prediction_type)
        if saved.get("labelStrategyId"):
            self.label_strategy_id = str(saved.get("labelStrategyId"))
        lsp = saved.get("labelStrategyParams")
        if isinstance(lsp, dict):
            self.label_strategy_params = dict(lsp)
        self.algorithm = str(saved.get("algorithm") or self.algorithm)
        feats = saved.get("features")
        if isinstance(feats, list):
            self.features = {str(f) for f in feats if f}
        split = saved.get("split") or {}
        if split.get("train") is not None:
            self.split_train = int(split["train"])
        if split.get("val") is not None:
            self.split_val = int(split["val"])
        elif split.get("validation") is not None:
            self.split_val = int(split["validation"])
        if split.get("test") is not None:
            self.split_test = int(split["test"])
        vs = saved.get("validationStrategy")
        if vs:
            self.validation_strategy = str(vs)
        wf = saved.get("walkForward") or {}
        for key, attr in (
            ("n_folds", "wf_folds"),
            ("window_mode", "wf_window_mode"),
            ("fold_placement", "wf_fold_placement"),
            ("train_window_size", "wf_train_window"),
            ("validation_window_size", "wf_val_window"),
            ("feature_selection_method", "wf_feature_selection"),
            ("optimization_metric", "wf_opt_metric"),
        ):
            if wf.get(key) is not None:
                setattr(self, attr, wf[key])
        hpo = wf.get("hyperparameter_optimization") or {}
        if "enabled" in hpo:
            self.wf_hpo_enabled = bool(hpo["enabled"])
        if hpo.get("n_trials") is not None:
            self.wf_hpo_trials = int(hpo["n_trials"])
        if "resume" in hpo:
            self.wf_hpo_resume = bool(hpo["resume"])
        gh = saved.get("hyperparameterOptimization") or {}
        if "enabled" in gh:
            self.global_hpo_enabled = bool(gh["enabled"])
        if gh.get("n_trials") is not None:
            self.global_hpo_trials = int(gh["n_trials"])
        if "resume" in gh:
            self.global_hpo_resume = bool(gh["resume"])
        x = saved.get("xgboost") or {}
        for src, attr in (
            ("learning_rate", "xgb_lr"),
            ("n_estimators", "xgb_trees"),
            ("max_depth", "xgb_depth"),
            ("early_stopping_rounds", "xgb_early_stop"),
            ("random_seed", "xgb_seed"),
            ("subsample", "xgb_subsample"),
            ("colsample_bytree", "xgb_colsample"),
            ("min_child_weight", "xgb_min_child"),
            ("reg_alpha", "xgb_reg_alpha"),
            ("reg_lambda", "xgb_reg_lambda"),
        ):
            if x.get(src) is not None:
                setattr(self, attr, x[src])
        lgb = saved.get("lightgbm") or {}
        if lgb.get("num_leaves") is not None:
            self.lgb_num_leaves = int(lgb["num_leaves"])
        cat = saved.get("catboost") or {}
        if cat.get("catboost_device"):
            self.catboost_device = str(cat["catboost_device"])
        if saved.get("modelVersion"):
            self.model_version = str(saved["modelVersion"])
        if saved.get("modelDescription"):
            self.model_description = str(saved["modelDescription"])
        if "skipAuditValidation" in saved:
            self.skip_audit_validation = bool(saved["skipAuditValidation"])
        if "showFeatures" in saved:
            self.show_features = bool(saved["showFeatures"])
        if "showAdvancedParams" in saved:
            self.show_advanced_params = bool(saved["showAdvancedParams"])
        if "registryAutoEnabled" in saved:
            self.registry_auto_enabled = bool(saved["registryAutoEnabled"])
        if saved.get("registryAutoTop") is not None:
            self.registry_auto_top = saved["registryAutoTop"]
        if saved.get("featureRegistryProject") is not None:
            self.feature_registry_project = str(saved.get("featureRegistryProject") or "all").strip() or "all"
        if "featureProjectEnabled" in saved:
            self.feature_project_enabled = bool(saved["featureProjectEnabled"])
        prem = saved.get("premiumSelection") or saved.get("premium_selection")
        if isinstance(prem, dict):
            if "enabled" in prem:
                self.premium_selection_enabled = bool(prem["enabled"])
            if prem.get("premium_min") is not None:
                try:
                    self.premium_min = float(prem["premium_min"])
                except (TypeError, ValueError):
                    pass
            if prem.get("premium_max") is not None:
                try:
                    self.premium_max = float(prem["premium_max"])
                except (TypeError, ValueError):
                    pass
        elif saved.get("premium_min") is not None and saved.get("premium_max") is not None:
            try:
                self.premium_min = float(saved["premium_min"])
                self.premium_max = float(saved["premium_max"])
                self.premium_selection_enabled = True
            except (TypeError, ValueError):
                pass
        pt = saved.get("postTraining") or saved.get("post_training")
        if isinstance(pt, dict):
            if "enabled" in pt:
                self.post_training_enabled = bool(pt["enabled"])
            if "importance" in pt:
                self.post_training_importance = bool(pt["importance"])
            if "distribution" in pt:
                self.post_training_distribution = bool(pt["distribution"])
            if "drift" in pt:
                self.post_training_drift = bool(pt["drift"])
        afs = saved.get("analysisFeatureSelection") or saved.get("analysis_feature_selection")
        if isinstance(afs, dict) and afs:
            self.analysis_feature_selection = dict(afs)


def _strip_lifecycle_from_saved(saved: dict[str, Any] | None) -> dict[str, Any] | None:
    """Lifecycle runs are session-only (web parity); never restore from draft config."""
    if not saved:
        return saved
    out = dict(saved)
    out.pop("lifecycle", None)
    return out


def config_storage_path(chart_dir: str) -> str:
    return os.path.join(chart_dir, "data", CONFIG_STORAGE)


def load_persisted_state(chart_dir: str) -> dict[str, Any] | None:
    path = config_storage_path(chart_dir)
    if not os.path.isfile(path):
        return None
    try:
        with open(path, encoding="utf-8") as fh:
            doc = json.load(fh)
        return doc if isinstance(doc, dict) else None
    except (OSError, json.JSONDecodeError):
        return None


def save_persisted_state(chart_dir: str, state: ModelBuilderState) -> None:
    path = config_storage_path(chart_dir)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(state.to_saved_dict(), fh, indent=2)

"""Training orchestrator — config in, model package out."""

from __future__ import annotations

import json
import os
import subprocess
import time
from collections.abc import Callable
from typing import Any

import pandas as pd

from .artifacts import save_model_package
from .config import TrainingConfig, normalize_training_config


def _hpo_center_on_baseline(config: TrainingConfig) -> bool:
    lc = config.lifecycle or {}
    return bool(lc.get("center_on_baseline")) or str(lc.get("mode") or "") == "complete_optimization"


def _lifecycle_baseline_composite(config: TrainingConfig) -> float | None:
    lc = config.lifecycle or {}
    src = lc.get("source_metrics") if isinstance(lc.get("source_metrics"), dict) else {}
    val = src.get("composite_score")
    try:
        return float(val) if val is not None else None
    except (TypeError, ValueError):
        return None
from .config_validator import validate_training_config
from .dataset_loader import DatasetLoaderError, load_training_xy
from .evaluator import evaluate_predictions, evaluate_regression, resolve_ltp_baseline_from_frames
from .feature_matrix import FeatureMatrixError, drop_invalid_rows, sanitize_training_features, validate_feature_matrix
from .naming import suggest_model_name_from_split
from .paths import model_artifact_paths, model_package_dir, safe_model_name
from .progress import TRAIN_STEP_ORDER, TrainStageTracker
from .shap_importance import compute_shap_importance
from .split import WalkForwardSplitError, split_xy
from .training_monitor import TrainingMonitor
from .training_log import TrainingLog
from .hyperparameter_optimizer import optimize_xgboost_hyperparameters
from .objective_scoring import composite_score_from_metrics
from .walk_forward_runner import (
    enrich_aggregated_from_fold_results,
    evaluate_hyperparameters_on_walk_forward,
    run_walk_forward_validation,
    train_final_model_after_walk_forward,
)
from .boost_trainer import TrainingCancelled, feature_importance_df, select_feature_columns, train_regressor
from .model_runtime import native_model_basename, native_model_path


def _current_git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=1.0,
            check=False,
        )
        if out.returncode != 0:
            return None
        val = str(out.stdout or "").strip()
        return val or None
    except Exception:
        return None


def _attach_post_training(
    *,
    data_dir: str,
    saved: dict[str, Any],
    result: dict[str, Any],
    on_progress: Callable[[dict[str, Any]], None] | None,
    post_training_config: dict[str, Any] | None = None,
    X: Any = None,
    y: Any = None,
) -> dict[str, Any]:
    """Run Feature Studio pipeline after package write. Never fails Create Model."""
    package_dir = str(saved.get("package_dir") or "")
    model_name = str(saved.get("model_name") or "")
    cfg = post_training_config
    if cfg is None:
        cfg_doc = result.get("config") if isinstance(result.get("config"), dict) else {}
        cfg = (cfg_doc or {}).get("post_training")
    try:
        from chain_replay_ml.post_training import run_safe

        pt = run_safe(
            package_dir,
            data_dir,
            model_name=model_name,
            progress=on_progress,
            config=cfg if isinstance(cfg, dict) else None,
            X=X,
            y=y,
        )

        # Ingest feature telemetry into feature_recommendation_evidence.db
        try:
            from chain_replay_ml.overnight_campaign.feature_evidence_bridge import persist_model_builder_feature_evidence
            cfg_doc = result.get("config") if isinstance(result.get("config"), dict) else {}
            ev_res = persist_model_builder_feature_evidence(
                data_dir=data_dir,
                package_dir=package_dir,
                config_doc=cfg_doc,
                post_training_result=pt,
            )
            pt["evidence_db_ingest"] = ev_res
        except Exception as ev_exc:
            logger.warning("[PostTraining] Evidence DB ingest failed: %s", ev_exc)

    except Exception as exc:  # pragma: no cover — run_safe already swallows
        pt = {
            "status": "failed",
            "warnings": [str(exc)],
            "model_name": model_name,
            "package_dir": package_dir,
        }
    result["post_training"] = pt
    return result


def _monitor_dashboard_fields(sample: dict[str, Any]) -> dict[str, Any]:
    return {
        "cpu_percent": sample.get("cpu_percent"),
        "ram_used_gb": sample.get("ram_used_gb"),
        "ram_percent": sample.get("ram_percent"),
        "ram_total_gb": sample.get("ram_total_gb"),
        "ram_available_gb": sample.get("ram_available_gb"),
        "gpu_percent": sample.get("gpu_percent"),
        "gpu_memory_used_gb": sample.get("gpu_memory_used_gb"),
        "gpu_memory_total_gb": sample.get("gpu_memory_total_gb"),
        "gpu_temperature": sample.get("gpu_temperature"),
        "gpu_power": sample.get("gpu_power"),
        "training_phase": sample.get("phase"),
        "current_trial": sample.get("trial"),
        "total_trials": sample.get("trial_total"),
        "best_trial_score": sample.get("best_score"),
        "elapsed_sec": sample.get("elapsed_seconds"),
    }


def _publish_premium_selection_dashboard(
    tracker: Any,
    metadata: dict[str, Any],
) -> None:
    """Surface Create Model Premium Selection on the live training dashboard."""
    prem = metadata.get("premium_selection")
    if not isinstance(prem, dict) or not prem:
        return
    tracker.update_dashboard(premium_selection=dict(prem))


def _build_and_save_training_metadata(
    *,
    monitor: TrainingMonitor,
    package_dir: str,
    config: TrainingConfig,
    metadata: dict[str, Any],
    train_params: dict[str, Any],
    hpo_trials: int,
    best_trial: int | None,
    final_training_duration_sec: float,
    hpo_duration_sec: float,
    model_name: str,
    optimization_result: dict[str, Any] | None = None,
    champion: str | None = None,
    production_model: str | None = None,
) -> dict[str, Any]:
    summary = monitor.build_summary(
        dataset=config.dataset,
        feature_count=len(config.features),
        target=config.target,
        device=str(train_params.get("xgb_device") or "cuda"),
        tree_method="hist",
        n_trials=hpo_trials,
        best_trial=best_trial,
        training_duration_sec=final_training_duration_sec,
        hpo_duration_sec=hpo_duration_sec,
        git_commit=_current_git_commit(),
        model_name=model_name,
    )
    summary["training"]["dataset_version"] = metadata.get("dataset_version") or metadata.get("builder_version")
    if optimization_result is not None:
        summary["optimization_result"] = optimization_result
    if champion:
        summary["champion"] = champion
    if production_model:
        summary["production_model"] = production_model
    monitor.save_summary(os.path.join(package_dir, "training_metadata.json"), summary)
    return summary


def _single_split_candidate_metrics(
    *,
    model: Any,
    use_features: list[str],
    val_X: pd.DataFrame,
    val_y: pd.Series,
    test_X: pd.DataFrame,
    test_y: pd.Series,
    context_df: pd.DataFrame | None = None,
    val_slice: tuple[int, int] | None = None,
    test_slice: tuple[int, int] | None = None,
    prediction_type: str,
    target: str | None = None,
) -> dict[str, Any]:
    val_X_feat, _ = select_feature_columns(val_X, use_features)
    test_X_feat, _ = select_feature_columns(test_X, use_features)
    val_pred = model.predict(val_X_feat)
    test_pred = model.predict(test_X_feat)

    val_ctx = (
        context_df.iloc[val_slice[0]:val_slice[1]].reset_index(drop=True)
        if context_df is not None and val_slice is not None and len(context_df) >= val_slice[1]
        else None
    )
    test_ctx = (
        context_df.iloc[test_slice[0]:test_slice[1]].reset_index(drop=True)
        if context_df is not None and test_slice is not None and len(context_df) >= test_slice[1]
        else None
    )
    val_baseline = resolve_ltp_baseline_from_frames(val_X, val_ctx)
    test_baseline = resolve_ltp_baseline_from_frames(test_X, test_ctx)

    val_metrics = evaluate_predictions(
        val_y,
        val_pred,
        prediction_type=prediction_type,
        baseline=val_baseline,
        target=target,
    )
    test_metrics = evaluate_predictions(
        test_y,
        test_pred,
        prediction_type=prediction_type,
        baseline=test_baseline,
        target=target,
    )
    composite = composite_score_from_metrics(val_metrics, prediction_type=prediction_type)
    return {
        "validation": val_metrics,
        "test": test_metrics,
        "composite_score": round(float(composite), 6),
    }


def _optimization_decision(
    *,
    baseline: dict[str, Any],
    tuned: dict[str, Any] | None,
) -> dict[str, Any]:
    if tuned is None:
        return {
            "enabled": False,
            "winner": "baseline",
            "reason": "Hyperparameter tuning disabled.",
            "baseline": baseline,
            "tuned": None,
            "improvement_pct": None,
            "difference": None,
        }
    base_comp = float(baseline.get("composite_score") or 0.0)
    tuned_comp = float(tuned.get("composite_score") or 0.0)
    diff = tuned_comp - base_comp
    pct = (diff / abs(base_comp) * 100.0) if base_comp != 0 else None
    tuned_wins = tuned_comp > base_comp
    return {
        "enabled": True,
        "winner": "tuned" if tuned_wins else "baseline",
        "reason": (
            "Tuned candidate has higher composite score."
            if tuned_wins
            else "Best Optuna trial did not exceed baseline composite score."
        ),
        "baseline": baseline,
        "tuned": tuned,
        "difference": round(float(diff), 6),
        "improvement_pct": round(float(pct), 2) if pct is not None else None,
    }


def _hpo_block_composite(hpo_result: dict[str, Any] | None, block_key: str) -> float | None:
    if not isinstance(hpo_result, dict):
        return None
    block = hpo_result.get(block_key) or {}
    if not isinstance(block, dict):
        return None
    raw = block.get("composite_score")
    try:
        val = float(raw)
    except (TypeError, ValueError):
        return None
    return val if val == val else None


def _build_composite_scores_doc(
    *,
    hpo_result: dict[str, Any] | None,
    chosen: dict[str, Any],
    optimization_result: dict[str, Any],
) -> dict[str, Any]:
    best_validation_score = _hpo_block_composite(hpo_result, "best_evaluation")
    prod_metrics = dict(chosen.get("production_validation_metrics") or chosen.get("validation_metrics") or {})
    prod_score_raw = prod_metrics.get("composite_score")
    try:
        prod_score = float(prod_score_raw) if prod_score_raw is not None else None
    except (TypeError, ValueError):
        prod_score = None
    if prod_score is not None and prod_score != prod_score:
        prod_score = None

    diff_abs = None
    diff_pct = None
    if best_validation_score is not None and prod_score is not None:
        diff_abs = round(prod_score - best_validation_score, 6)
        if best_validation_score != 0:
            diff_pct = round((prod_score - best_validation_score) / abs(best_validation_score) * 100.0, 2)

    return {
        "best_validation_composite": {
            "score": best_validation_score,
            "source": "Optuna validation during HPO",
            "source_file": "walk_forward/best_parameters.json",
            "source_path": "$.best_evaluation.composite_score",
            "purpose": "Model selection — fast Optuna trial evaluation used to pick hyperparameters",
        },
        "production_composite": {
            "score": prod_score,
            "source": "Retrained production model evaluated on walk-forward aggregate",
            "source_file": "metrics.json",
            "source_path": "$.production_walk_forward",
            "purpose": "Deployed model performance — champion hyperparameters re-evaluated on all WF folds with full training budget after final retrain",
            "champion": optimization_result.get("winner"),
        },
        "difference_abs": diff_abs,
        "difference_pct": diff_pct,
        "values_differ": (
            best_validation_score is not None
            and prod_score is not None
            and abs(prod_score - best_validation_score) > 1e-6
        ),
    }


def train_model(
    *,
    data_dir: str,
    raw_config: dict[str, Any],
    on_progress: Callable[[dict[str, Any]], None] | None = None,
    cancel_check: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    def _cancelled() -> bool:
        return bool(cancel_check and cancel_check())

    validation = validate_training_config(data_dir, raw_config)
    if validation.get("blocked"):
        return {
            "ok": False,
            "blocked": True,
            "validation": validation,
        }

    config = normalize_training_config(validation["config"])
    if config.split.get("strategy") == "walk_forward":
        return _train_walk_forward(
            data_dir=data_dir,
            config=config,
            on_progress=on_progress,
            cancel_check=cancel_check,
        )
    return _train_single_split(
        data_dir=data_dir,
        config=config,
        on_progress=on_progress,
        cancel_check=cancel_check,
    )


def _train_single_split(
    *,
    data_dir: str,
    config: TrainingConfig,
    on_progress: Callable[[dict[str, Any]], None] | None,
    cancel_check: Callable[[], bool] | None,
) -> dict[str, Any]:
    def _cancelled() -> bool:
        return bool(cancel_check and cancel_check())

    model_name = safe_model_name(
        config.model_name or suggest_model_name_from_split(
            config.target, config.algorithm, config.split, feature_count=len(config.features)
        )
    )
    package_dir = model_package_dir(data_dir, model_name)
    tracker = TrainStageTracker(on_progress)
    monitor = TrainingMonitor(
        csv_path=os.path.join(package_dir, "training_monitor.csv"),
        interval_sec=5.0,
        on_sample=lambda sample: tracker.update_dashboard(**_monitor_dashboard_fields(sample)),
    )
    log = TrainingLog()
    monitor.start()
    monitor.set_phase("Preparing Dataset")

    try:
        from .model_device import emit_startup_diagnostics_once

        emit_startup_diagnostics_once(log_fn=log.log)
    except Exception:
        pass

    def _check_cancel() -> None:
        if _cancelled():
            raise TrainingCancelled("Training cancelled")

    try:
        log.log("Loading dataset")
        tracker.emit("preparing_dataset", "running")
        _check_cancel()
        try:
            X, y, features, metadata, _expected, context_df = load_training_xy(data_dir, config)
        except DatasetLoaderError as exc:
            tracker.emit("preparing_dataset", "fail", detail=str(exc))
            return {"ok": False, "error": str(exc), "step": "preparing_dataset"}
        except MemoryError:
            msg = (
                "Out of memory loading training data. "
                "Use fewer trading days, disable HPO, or restart the server before training."
            )
            tracker.emit("preparing_dataset", "fail", detail=msg)
            return {"ok": False, "error": msg, "step": "preparing_dataset"}
        _publish_premium_selection_dashboard(tracker, metadata)
        tracker.emit("preparing_dataset", "done", rows_total=len(X))

        monitor.set_phase("Preparing Matrix")
        log.log("Preparing matrix")
        tracker.emit("preparing_matrix", "running")
        _check_cancel()
        try:
            X, y, context_df = drop_invalid_rows(X, y, context_df)
            X = sanitize_training_features(X)
            matrix_report = validate_feature_matrix(X, y)
        except FeatureMatrixError as exc:
            tracker.emit("preparing_matrix", "fail", detail=str(exc))
            return {"ok": False, "error": str(exc), "step": "preparing_matrix"}
        tracker.emit(
            "preparing_matrix",
            "done",
            x_shape=matrix_report["x_shape"],
            y_shape=matrix_report["y_shape"],
        )

        parts = split_xy(X, y, config)
        train_X = parts["train"]["X"]
        train_y = parts["train"]["y"]
        val_X = parts["validation"]["X"]
        val_y = parts["validation"]["y"]
        test_X = parts["test"]["X"]
        test_y = parts["test"]["y"]

        hpo_cfg = dict(config.split.get("hyperparameter_optimization") or {})
        hpo_enabled = bool(hpo_cfg.get("enabled", False)) and config.algorithm == "xgboost"
        hpo_result: dict[str, Any] | None = None
        hpo_duration_sec = 0.0
        train_params_base = dict(config.parameters)
        train_params_base["prediction_type"] = config.prediction_type
        train_params_base["target"] = config.target
        train_params_tuned = dict(config.parameters)
        train_params_tuned["prediction_type"] = config.prediction_type
        train_params_tuned["target"] = config.target

        artifact_paths = model_artifact_paths(data_dir, model_name)
        os.makedirs(package_dir, exist_ok=True)

        monitor.set_phase("Baseline Training")
        log.log("Training baseline candidate")
        tracker.emit("training", "running", trees_total=int(train_params_base.get("n_estimators", 1000)), current_tree=0)
        _check_cancel()

        def _on_iter(payload: dict[str, Any]) -> None:
            tracker.update_dashboard(**payload)
            tracker.emit("training", "running", **payload)

        try:
            baseline_trained = train_regressor(
                algorithm=config.algorithm,
                train_X=train_X,
                train_y=train_y,
                val_X=val_X,
                val_y=val_y,
                features=features,
                parameters=train_params_base,
                cancel_check=cancel_check,
                on_iteration=_on_iter,
                prediction_type=config.prediction_type,
            )
        except TrainingCancelled:
            tracker.emit("training", "cancelled")
            return {"ok": False, "cancelled": True}
        baseline_model = baseline_trained["model"]
        baseline_features = list(baseline_trained.get("features") or features)
        from .algorithm_runtime import format_runtime_log_block

        for _line in format_runtime_log_block(
            (baseline_trained.get("training_meta") or {}).get("algorithm_runtime")
        ):
            log.log(_line)
        baseline_metrics = _single_split_candidate_metrics(
            model=baseline_model,
            use_features=baseline_features,
            val_X=val_X,
            val_y=val_y,
            test_X=test_X,
            test_y=test_y,
            context_df=context_df,
            val_slice=parts["validation"].get("slice"),
            test_slice=parts["test"].get("slice"),
            prediction_type=config.prediction_type,
            target=config.target,
        )
        baseline_candidate = {
            "metrics": baseline_metrics,
            "training_time_sec": float(baseline_trained.get("training_time_sec") or 0.0),
            "trees_trained": int(baseline_trained.get("trees_trained") or 0),
            "training_meta": baseline_trained.get("training_meta") or {},
        }
        try:
            baseline_model.save_model(native_model_path(package_dir, "baseline_model", config.algorithm))
        except Exception:
            pass

        if hpo_enabled:
            n_trials = int(hpo_cfg.get("n_trials") or 25)
            monitor.set_phase("Hyperparameter Optimization", trial=0, total_trials=n_trials)
            log.log(f"Hyperparameter optimization ({n_trials} trials)")
            _baseline = _lifecycle_baseline_composite(config)
            tracker.emit(
                "hyperparameter_optimization", "running", trial=0, n_trials=n_trials,
                baseline_display_score=_baseline,
                baseline_composite_score=_baseline,
            )
            _check_cancel()
            hpo_started = time.monotonic()

            X_hpo = pd.concat([train_X, val_X], axis=0, ignore_index=True)
            y_hpo = pd.concat([train_y, val_y], axis=0, ignore_index=True)
            n_train = int(len(train_X))
            n_val = int(len(val_X))
            fold_defs = [{
                "fold": 1,
                "train": {"start": 0, "stop": n_train},
                "validation": {"start": n_train, "stop": n_train + n_val},
            }]
            hpo_ctx = None
            if context_df is not None and parts["train"].get("slice") and parts["validation"].get("slice"):
                tr = parts["train"]["slice"]
                va = parts["validation"]["slice"]
                if len(context_df) >= max(tr[1], va[1]):
                    hpo_ctx = pd.concat(
                        [
                            context_df.iloc[tr[0]:tr[1]].reset_index(drop=True),
                            context_df.iloc[va[0]:va[1]].reset_index(drop=True),
                        ],
                        axis=0,
                        ignore_index=True,
                    )

            def _on_hpo_trial(payload: dict[str, Any]) -> None:
                tracker.emit("hyperparameter_optimization", "running", **payload)
                trial = payload.get("trial")
                best = payload.get("best_display_score")
                monitor.set_phase("Hyperparameter Optimization", trial=int(trial or 0), total_trials=n_trials, best_score=best)

            try:
                hpo_result = optimize_xgboost_hyperparameters(
                    X=X_hpo,
                    y=y_hpo,
                    features=features,
                    fold_defs=fold_defs,
                    base_parameters=dict(config.parameters),
                    optimization_metric="auto",
                    prediction_type=config.prediction_type,
                    n_trials=n_trials,
                    validation_seeds=[int(s) for s in (hpo_cfg.get("validation_seeds") or [42, 123, 999])],
                    resume=bool(hpo_cfg.get("resume", True)),
                    artifacts_dir=os.path.join(package_dir, "hyperparameter_optimization"),
                    cancel_check=cancel_check,
                    on_trial=_on_hpo_trial,
                    center_on_baseline=_hpo_center_on_baseline(config),
                    context_df=hpo_ctx,
                )
                train_params_tuned = dict(hpo_result.get("best_parameters") or config.parameters)
            except TrainingCancelled:
                tracker.emit("hyperparameter_optimization", "cancelled")
                return {"ok": False, "cancelled": True}
            except RuntimeError as exc:
                tracker.emit("hyperparameter_optimization", "fail", detail=str(exc))
                return {"ok": False, "error": str(exc), "step": "hyperparameter_optimization"}

            hpo_duration_sec = max(0.0, time.monotonic() - hpo_started)
            tracker.emit(
                "hyperparameter_optimization",
                "done",
                hyperparameter_optimization=hpo_result,
                best_parameters=hpo_result.get("best_search_params"),
                best_objective=hpo_result.get("best_objective"),
                best_display_score=hpo_result.get("best_display_score"),
                n_trials=hpo_result.get("n_trials"),
                n_trials_resumed_from=hpo_result.get("n_trials_resumed_from"),
                validation_seeds=hpo_result.get("validation_seeds"),
            )

        tuned_candidate: dict[str, Any] | None = None
        if hpo_enabled:
            monitor.set_phase("Tuned Candidate Training")
            log.log("Training tuned candidate")
            tracker.emit("training", "running", trees_total=int(train_params_tuned.get("n_estimators", 1000)), current_tree=0)
            _check_cancel()
            try:
                tuned_trained = train_regressor(
                    algorithm=config.algorithm,
                    train_X=train_X,
                    train_y=train_y,
                    val_X=val_X,
                    val_y=val_y,
                    features=features,
                    parameters=train_params_tuned,
                    cancel_check=cancel_check,
                    on_iteration=_on_iter,
                    prediction_type=config.prediction_type,
                )
            except TrainingCancelled:
                tracker.emit("training", "cancelled")
                return {"ok": False, "cancelled": True}
            tuned_model = tuned_trained["model"]
            tuned_features = list(tuned_trained.get("features") or features)
            tuned_metrics = _single_split_candidate_metrics(
                model=tuned_model,
                use_features=tuned_features,
                val_X=val_X,
                val_y=val_y,
                test_X=test_X,
                test_y=test_y,
                context_df=context_df,
                val_slice=parts["validation"].get("slice"),
                test_slice=parts["test"].get("slice"),
                prediction_type=config.prediction_type,
                target=config.target,
            )
            tuned_candidate = {
                "trained": tuned_trained,
                "metrics": tuned_metrics,
                "training_time_sec": float(tuned_trained.get("training_time_sec") or 0.0),
                "trees_trained": int(tuned_trained.get("trees_trained") or 0),
                "training_meta": tuned_trained.get("training_meta") or {},
            }
            try:
                tuned_model.save_model(native_model_path(package_dir, "tuned_model", config.algorithm))
            except Exception:
                pass

        optimization_result = _optimization_decision(
            baseline={
                **baseline_candidate["metrics"],
                "training_time_sec": baseline_candidate["training_time_sec"],
                "trees_trained": baseline_candidate["trees_trained"],
            },
            tuned=(
                {
                    **(tuned_candidate or {}).get("metrics", {}),
                    "training_time_sec": (tuned_candidate or {}).get("training_time_sec"),
                    "trees_trained": (tuned_candidate or {}).get("trees_trained"),
                }
                if tuned_candidate is not None
                else None
            ),
        )

        if optimization_result["winner"] == "tuned" and tuned_candidate is not None:
            trained = tuned_candidate["trained"]
            model = trained["model"]
            booster = trained.get("booster") or model
            use_features = list(trained.get("features") or features)
            best_iter = int(tuned_candidate["trees_trained"])
            training_meta = dict(tuned_candidate["training_meta"])
            validation_loss_curve = trained["validation_loss_curve"]
            training_time_sec = float(tuned_candidate["training_time_sec"])
            metrics = {
                "validation": tuned_candidate["metrics"]["validation"],
                "test": tuned_candidate["metrics"]["test"],
            }
            champion_model_file = native_model_basename("tuned_model", config.algorithm)
        else:
            trained = baseline_trained
            model = baseline_model
            booster = baseline_trained.get("booster") or baseline_model
            use_features = baseline_features
            best_iter = int(baseline_candidate["trees_trained"])
            training_meta = dict(baseline_candidate["training_meta"])
            validation_loss_curve = baseline_trained["validation_loss_curve"]
            training_time_sec = float(baseline_candidate["training_time_sec"])
            metrics = {
                "validation": baseline_candidate["metrics"]["validation"],
                "test": baseline_candidate["metrics"]["test"],
            }
            champion_model_file = native_model_basename("baseline_model", config.algorithm)
        training_meta["optimization_result"] = optimization_result

        try:
            model.save_model(native_model_path(package_dir, "model", config.algorithm))
        except Exception:
            pass

        from .algorithm_runtime import format_runtime_log_block

        for _line in format_runtime_log_block(training_meta.get("algorithm_runtime")):
            log.log(_line)

        tracker.emit(
            "training",
            "done",
            current_tree=best_iter,
            trees_total=int((train_params_tuned if optimization_result["winner"] == "tuned" else train_params_base).get("n_estimators", 1000)),
            train_progress_pct=100,
            early_stopped=trained["early_stopped"],
            training_meta=training_meta,
            validation_loss_curve=validation_loss_curve,
        )
        tracker.update_dashboard(
            validation_rmse=training_meta.get("best_validation_rmse"),
            train_rmse=training_meta.get("train_rmse"),
            best_iteration=training_meta.get("best_iteration"),
            early_stopping_rounds=training_meta.get("early_stopping_rounds"),
        )

        monitor.set_phase("Evaluation")
        log.log("Evaluation complete")
        tracker.emit("evaluation", "running")
        _check_cancel()

        if hpo_result:
            metrics["hyperparameter_optimization"] = {
                "best_objective": hpo_result.get("best_objective"),
                "optimization_metric": hpo_result.get("optimization_metric"),
                "best_parameters": hpo_result.get("best_search_params"),
                "n_trials": hpo_result.get("n_trials"),
            }
        metrics["optimization_result"] = optimization_result
        tracker.emit("evaluation", "done", metrics=metrics)
        tracker.update_dashboard(test_rmse=metrics["test"].get("rmse"))

        importance = feature_importance_df(model, use_features)
        val_X_feat, _ = select_feature_columns(val_X, use_features)
        shap_rows = compute_shap_importance(booster, val_X_feat, use_features)
        monitor.set_phase("Saving")
        log.log("Saving model")
        tracker.emit("saving", "running", feature_importance=importance.head(20).to_dict(orient="records"))
        _check_cancel()

        monitor.stop()
        training_metadata_doc = _build_and_save_training_metadata(
            monitor=monitor,
            package_dir=package_dir,
            config=config,
            metadata=metadata,
            train_params=(train_params_tuned if optimization_result["winner"] == "tuned" else train_params_base),
            hpo_trials=int((hpo_result or {}).get("n_trials") or 0),
            best_trial=(hpo_result or {}).get("best_trial"),
            final_training_duration_sec=training_time_sec,
            hpo_duration_sec=hpo_duration_sec,
            model_name=model_name,
            optimization_result=optimization_result,
            champion=str(optimization_result.get("winner") or "baseline"),
            production_model=champion_model_file,
        )

        saved = save_model_package(
            data_dir=data_dir,
            config=config,
            model=model,
            metrics=metrics,
            feature_importance=importance,
            metadata=metadata,
            matrix_report=matrix_report,
            split_info=parts,
            training_log_text=log.text(),
            trees_trained=best_iter,
            early_stopped=trained["early_stopped"],
            training_time_sec=training_time_sec,
            total_elapsed_sec=tracker.elapsed_total(),
            training_meta=training_meta,
            validation_loss_curve=validation_loss_curve,
            shap_importance=shap_rows,
            training_metadata=training_metadata_doc,
        )
        tracker.emit(
            "saving",
            "done",
            model_name=saved["model_name"],
            feature_importance=importance.to_dict(orient="records"),
            training_summary=saved.get("training_summary"),
            training_metadata=training_metadata_doc,
            report_url="",
        )

        result = {
            "ok": True,
            "model_name": saved["model_name"],
            "package_dir": saved["package_dir"],
            "metrics": metrics,
            "feature_importance": importance.to_dict(orient="records"),
            "feature_importance_top20": importance.head(20).to_dict(orient="records"),
            "training_summary": saved.get("training_summary"),
            "training_meta": training_meta,
            "training_metadata": training_metadata_doc,
            "validation_loss_curve": validation_loss_curve,
            "shap_importance": shap_rows,
            "hyperparameter_optimization": hpo_result,
            "optimization_result": optimization_result,
            "report_url": "",
            "matrix_report": matrix_report,
            "steps": TRAIN_STEP_ORDER + (["hyperparameter_optimization"] if hpo_result else []),
            "config": config.to_dict(),
        }
        return _attach_post_training(
            data_dir=data_dir,
            saved=saved,
            result=result,
            on_progress=on_progress,
            post_training_config=dict(config.post_training or {}),
            X=X,
            y=y,
        )
    finally:
        monitor.stop()


def _train_walk_forward(
    *,
    data_dir: str,
    config: TrainingConfig,
    on_progress: Callable[[dict[str, Any]], None] | None,
    cancel_check: Callable[[], bool] | None,
) -> dict[str, Any]:
    def _cancelled() -> bool:
        return bool(cancel_check and cancel_check())

    model_name = safe_model_name(
        config.model_name or suggest_model_name_from_split(
            config.target, config.algorithm, config.split, feature_count=len(config.features)
        )
    )
    package_dir = model_package_dir(data_dir, model_name)
    wf_started = time.monotonic()
    tracker = TrainStageTracker(on_progress)
    monitor = TrainingMonitor(
        csv_path=os.path.join(package_dir, "training_monitor.csv"),
        interval_sec=5.0,
        on_sample=lambda sample: tracker.update_dashboard(**_monitor_dashboard_fields(sample)),
    )
    log = TrainingLog()
    monitor.start()
    monitor.set_phase("Preparing Dataset")

    try:
        from .model_device import emit_startup_diagnostics_once

        emit_startup_diagnostics_once(log_fn=log.log)
    except Exception:
        pass

    def _check_cancel() -> None:
        if _cancelled():
            raise TrainingCancelled("Training cancelled")

    try:
        log.log("Loading dataset")
        tracker.emit("preparing_dataset", "running")
        _check_cancel()
        try:
            X, y, features, metadata, _expected, context_df = load_training_xy(data_dir, config)
        except DatasetLoaderError as exc:
            tracker.emit("preparing_dataset", "fail", detail=str(exc))
            return {"ok": False, "error": str(exc), "step": "preparing_dataset"}
        except MemoryError:
            msg = (
                "Out of memory loading training data. "
                "Use fewer trading days, disable HPO, or restart the server before training."
            )
            tracker.emit("preparing_dataset", "fail", detail=msg)
            return {"ok": False, "error": msg, "step": "preparing_dataset"}
        _publish_premium_selection_dashboard(tracker, metadata)
        tracker.emit("preparing_dataset", "done", rows_total=len(X))

        monitor.set_phase("Preparing Matrix")
        log.log("Preparing matrix")
        tracker.emit("preparing_matrix", "running")
        _check_cancel()
        try:
            X, y, context_df = drop_invalid_rows(X, y, context_df)
            X = sanitize_training_features(X)
            matrix_report = validate_feature_matrix(X, y)
        except FeatureMatrixError as exc:
            tracker.emit("preparing_matrix", "fail", detail=str(exc))
            return {"ok": False, "error": str(exc), "step": "preparing_matrix"}
        tracker.emit(
            "preparing_matrix",
            "done",
            x_shape=matrix_report["x_shape"],
            y_shape=matrix_report["y_shape"],
        )

        try:
            parts = split_xy(X, y, config)
        except WalkForwardSplitError as exc:
            tracker.emit("walk_forward", "fail", detail=str(exc))
            return {"ok": False, "error": str(exc), "step": "walk_forward"}

        wf_cfg = parts.get("walk_forward") or config.split.get("walk_forward") or {}
        wf_artifacts_dir = os.path.join(package_dir, "walk_forward")

        monitor.set_phase("Walk-Forward Validation")
        log.log("Walk-forward validation started")
        tracker.emit("walk_forward", "running", fold=0, n_folds=wf_cfg.get("n_folds"), feature_count=len(features), current_features=len(features))

        def _on_fold(payload: dict[str, Any]) -> None:
            tracker.emit("walk_forward", "running", **{k: v for k, v in payload.items() if k != "status"})

        try:
            wf_result = run_walk_forward_validation(
                X=X,
                y=y,
                features=features,
                parameters={**dict(config.parameters), "prediction_type": config.prediction_type, "target": config.target},
                algorithm=config.algorithm,
                walk_forward_cfg={**wf_cfg, "test_holdout_pct": config.split.get("test", 15)},
                split_cfg=config.split,
                prediction_type=config.prediction_type,
                artifacts_dir=wf_artifacts_dir,
                cancel_check=cancel_check,
                on_fold_progress=_on_fold,
                context_df=context_df,
            )
        except TrainingCancelled:
            tracker.emit("walk_forward", "cancelled")
            return {"ok": False, "cancelled": True}
        except WalkForwardSplitError as exc:
            tracker.emit("walk_forward", "fail", detail=str(exc))
            return {"ok": False, "error": str(exc), "step": "walk_forward"}

        aggregated = wf_result["aggregated"]
        selected_features = wf_result["selected_features"]
        test_sl = wf_result["test_slice"]
        test_X = X.iloc[test_sl][selected_features]
        test_y = y.iloc[test_sl]

        selection_meta = aggregated.get("feature_selection") or {}

        tracker.emit(
            "walk_forward",
            "done",
            walk_forward=aggregated,
            feature_selection=selection_meta,
            n_folds=aggregated.get("n_folds"),
            mean_rmse=aggregated.get("mean_rmse"),
            mean_mae=aggregated.get("mean_mae"),
            mean_directional_accuracy_pct=aggregated.get("mean_directional_accuracy_pct"),
            selected_features=selected_features,
            walk_forward_summary=wf_result["summary"],
        )

        hpo_cfg = dict(wf_cfg.get("hyperparameter_optimization") or {})
        hpo_enabled = bool(hpo_cfg.get("enabled", True)) and config.algorithm == "xgboost"
        n_trials = int(hpo_cfg.get("n_trials") or 25)
        train_params_base = dict(config.parameters)
        train_params_base["prediction_type"] = config.prediction_type
        train_params_base["target"] = config.target
        train_params_tuned = dict(config.parameters)
        train_params_tuned["prediction_type"] = config.prediction_type
        train_params_tuned["target"] = config.target
        hpo_result: dict[str, Any] | None = None
        hpo_started = 0.0
        hpo_duration_sec = 0.0
        artifact_paths = model_artifact_paths(data_dir, model_name)
        os.makedirs(package_dir, exist_ok=True)

        if hpo_enabled:
            monitor.set_phase("Hyperparameter Optimization", trial=0, total_trials=n_trials)
            log.log(f"Hyperparameter optimization ({n_trials} trials)")
            _baseline = _lifecycle_baseline_composite(config)
            tracker.emit(
                "hyperparameter_optimization", "running", trial=0, n_trials=n_trials,
                baseline_display_score=_baseline,
                baseline_composite_score=_baseline,
            )
            _check_cancel()
            hpo_started = time.monotonic()

            def _on_hpo_trial(payload: dict[str, Any]) -> None:
                tracker.emit("hyperparameter_optimization", "running", **payload)
                trial = payload.get("trial")
                best = payload.get("best_display_score")
                monitor.set_phase("Hyperparameter Optimization", trial=int(trial or 0), total_trials=n_trials, best_score=best)

            try:
                hpo_result = optimize_xgboost_hyperparameters(
                    X=X,
                    y=y,
                    features=selected_features,
                    fold_defs=wf_result["summary"]["folds"],
                    base_parameters=dict(config.parameters),
                    optimization_metric=str(wf_cfg.get("optimization_metric") or "auto"),
                    prediction_type=config.prediction_type,
                    n_trials=n_trials,
                    validation_seeds=[int(s) for s in (hpo_cfg.get("validation_seeds") or [42, 123, 999])],
                    resume=bool(hpo_cfg.get("resume", True)),
                    artifacts_dir=wf_artifacts_dir,
                    cancel_check=cancel_check,
                    on_trial=_on_hpo_trial,
                    center_on_baseline=_hpo_center_on_baseline(config),
                    context_df=context_df,
                )
                train_params_tuned = hpo_result["best_parameters"]
            except TrainingCancelled:
                tracker.emit("hyperparameter_optimization", "cancelled")
                return {"ok": False, "cancelled": True}
            except RuntimeError as exc:
                tracker.emit("hyperparameter_optimization", "fail", detail=str(exc))
                return {"ok": False, "error": str(exc), "step": "hyperparameter_optimization"}
            hpo_duration_sec = max(0.0, time.monotonic() - hpo_started)

            tracker.emit(
                "hyperparameter_optimization",
                "done",
                hyperparameter_optimization=hpo_result,
                best_parameters=hpo_result.get("best_search_params"),
                best_objective=hpo_result.get("best_objective"),
                best_display_score=hpo_result.get("best_display_score"),
                n_trials=hpo_result.get("n_trials"),
                n_trials_resumed_from=hpo_result.get("n_trials_resumed_from"),
                validation_seeds=hpo_result.get("validation_seeds"),
            )

        monitor.set_phase("Baseline Candidate Training")
        log.log(f"Baseline candidate training on {len(selected_features)} selected features")
        tracker.emit("training", "running", trees_total=int(train_params_base.get("n_estimators", 1000)), current_tree=0)
        _check_cancel()

        def _on_iter(payload: dict[str, Any]) -> None:
            tracker.update_dashboard(**payload)
            tracker.emit("training", "running", **payload)

        try:
            baseline_trained = train_final_model_after_walk_forward(
                X=X,
                y=y,
                features=features,
                selected_features=selected_features,
                parameters=train_params_base,
                algorithm=config.algorithm,
                test_slice=test_sl,
                val_window_size=int(wf_cfg.get("validation_window_size") or 1000),
                cancel_check=cancel_check,
                on_iteration=_on_iter,
            )
        except TrainingCancelled:
            tracker.emit("training", "cancelled")
            return {"ok": False, "cancelled": True}

        baseline_model = baseline_trained["model"]
        baseline_features = list(baseline_model.feature_names_in_)
        baseline_training_meta = baseline_trained["training_meta"]
        baseline_loss_curve = baseline_trained["validation_loss_curve"]
        baseline_test_pred = baseline_model.predict(test_X)
        test_ctx = (
            context_df.iloc[test_sl].reset_index(drop=True)
            if context_df is not None and len(context_df) >= test_sl.stop
            else None
        )
        baseline_test_baseline = resolve_ltp_baseline_from_frames(test_X, test_ctx)
        baseline_test_metrics = evaluate_predictions(
            test_y, baseline_test_pred,
            prediction_type=config.prediction_type,
            baseline=baseline_test_baseline,
            target=config.target,
        )
        wf_score_refs = dict(wf_result["summary"].get("reference_stats") or {})
        wf_fold_defs = list(wf_result["summary"].get("folds") or [])

        monitor.set_phase("Baseline WF Production Evaluation")
        log.log("Re-evaluating baseline hyperparameters on walk-forward folds (production composite)")
        tracker.emit("evaluation", "running", detail="baseline_production_wf")
        _check_cancel()
        baseline_prod = evaluate_hyperparameters_on_walk_forward(
            X=X,
            y=y,
            features=selected_features,
            parameters=train_params_base,
            fold_defs=wf_fold_defs,
            algorithm=config.algorithm,
            prediction_type=config.prediction_type,
            score_refs=wf_score_refs,
            cancel_check=cancel_check,
            context_df=context_df,
        )
        baseline_validation_metrics = dict(baseline_prod["validation_metrics"])
        baseline_prod_agg = dict(baseline_prod["aggregated"])
        baseline_prod_folds = list(baseline_prod.get("fold_results") or [])
        baseline_hpo_composite = _hpo_block_composite(hpo_result, "baseline_evaluation")
        baseline_selection_composite = float(
            baseline_hpo_composite
            if baseline_hpo_composite is not None
            else (baseline_validation_metrics.get("composite_score") or 0.0)
        )
        baseline_candidate = {
            "trained": baseline_trained,
            "model": baseline_model,
            "booster": baseline_trained.get("booster") or baseline_model,
            "features": baseline_features,
            "training_meta": baseline_training_meta,
            "validation_loss_curve": baseline_loss_curve,
            "validation_metrics": baseline_validation_metrics,
            "production_validation_metrics": baseline_validation_metrics,
            "production_walk_forward_aggregate": baseline_prod_agg,
            "production_fold_results": baseline_prod_folds,
            "test_metrics": baseline_test_metrics,
            "composite_score": baseline_selection_composite,
            "hpo_validation_composite": baseline_hpo_composite,
            "training_time_sec": float(baseline_trained.get("training_time_sec") or 0.0),
            "trees_trained": int(baseline_trained.get("trees_trained") or 0),
        }
        try:
            baseline_model.save_model(native_model_path(package_dir, "baseline_model", config.algorithm))
        except Exception:
            pass

        tuned_candidate: dict[str, Any] | None = None
        if hpo_enabled and hpo_result is not None:
            monitor.set_phase("Tuned Candidate Training")
            log.log("Training tuned candidate")
            tracker.emit("training", "running", trees_total=int(train_params_tuned.get("n_estimators", 1000)), current_tree=0)
            _check_cancel()
            try:
                tuned_trained = train_final_model_after_walk_forward(
                    X=X,
                    y=y,
                    features=features,
                    selected_features=selected_features,
                    parameters=train_params_tuned,
                    algorithm=config.algorithm,
                    test_slice=test_sl,
                    val_window_size=int(wf_cfg.get("validation_window_size") or 1000),
                    cancel_check=cancel_check,
                    on_iteration=_on_iter,
                )
            except TrainingCancelled:
                tracker.emit("training", "cancelled")
                return {"ok": False, "cancelled": True}
            tuned_model = tuned_trained["model"]
            tuned_features = list(tuned_model.feature_names_in_)
            tuned_training_meta = tuned_trained["training_meta"]
            tuned_loss_curve = tuned_trained["validation_loss_curve"]
            tuned_test_pred = tuned_model.predict(test_X)
            tuned_test_baseline = resolve_ltp_baseline_from_frames(test_X, test_ctx)
            tuned_test_metrics = evaluate_predictions(
                test_y, tuned_test_pred,
                prediction_type=config.prediction_type,
                baseline=tuned_test_baseline,
                target=config.target,
            )
            monitor.set_phase("Tuned WF Production Evaluation")
            log.log("Re-evaluating tuned hyperparameters on walk-forward folds (production composite)")
            tracker.emit("evaluation", "running", detail="tuned_production_wf")
            _check_cancel()
            tuned_prod = evaluate_hyperparameters_on_walk_forward(
                X=X,
                y=y,
                features=selected_features,
                parameters=train_params_tuned,
                fold_defs=wf_fold_defs,
                algorithm=config.algorithm,
                prediction_type=config.prediction_type,
                score_refs=wf_score_refs,
                cancel_check=cancel_check,
                context_df=context_df,
            )
            tuned_validation_metrics = dict(tuned_prod["validation_metrics"])
            tuned_prod_agg = dict(tuned_prod["aggregated"])
            tuned_prod_folds = list(tuned_prod.get("fold_results") or [])
            tuned_hpo_composite = _hpo_block_composite(hpo_result, "best_evaluation")
            tuned_selection_composite = float(
                tuned_hpo_composite
                if tuned_hpo_composite is not None
                else (tuned_validation_metrics.get("composite_score") or 0.0)
            )
            tuned_candidate = {
                "trained": tuned_trained,
                "model": tuned_model,
                "booster": tuned_trained.get("booster") or tuned_model,
                "features": tuned_features,
                "training_meta": tuned_training_meta,
                "validation_loss_curve": tuned_loss_curve,
                "validation_metrics": tuned_validation_metrics,
                "production_validation_metrics": tuned_validation_metrics,
                "production_walk_forward_aggregate": tuned_prod_agg,
                "production_fold_results": tuned_prod_folds,
                "test_metrics": tuned_test_metrics,
                "composite_score": tuned_selection_composite,
                "hpo_validation_composite": tuned_hpo_composite,
                "training_time_sec": float(tuned_trained.get("training_time_sec") or 0.0),
                "trees_trained": int(tuned_trained.get("trees_trained") or 0),
            }
            try:
                tuned_model.save_model(native_model_path(package_dir, "tuned_model", config.algorithm))
            except Exception:
                pass

        optimization_result = _optimization_decision(
            baseline={
                "validation": baseline_candidate["validation_metrics"],
                "test": baseline_candidate["test_metrics"],
                "composite_score": baseline_candidate["composite_score"],
                "training_time_sec": baseline_candidate["training_time_sec"],
                "trees_trained": baseline_candidate["trees_trained"],
            },
            tuned=(
                {
                    "validation": tuned_candidate["validation_metrics"],
                    "test": tuned_candidate["test_metrics"],
                    "composite_score": tuned_candidate["composite_score"],
                    "training_time_sec": tuned_candidate["training_time_sec"],
                    "trees_trained": tuned_candidate["trees_trained"],
                }
                if tuned_candidate is not None
                else None
            ),
        )

        chosen = tuned_candidate if (optimization_result["winner"] == "tuned" and tuned_candidate is not None) else baseline_candidate
        champion_model_file = (
            native_model_basename("tuned_model", config.algorithm)
            if chosen is tuned_candidate
            else native_model_basename("baseline_model", config.algorithm)
        )
        chosen_params = train_params_tuned if optimization_result["winner"] == "tuned" else train_params_base
        prediction_run_doc: dict[str, Any] | None = None
        try:
            from chain_replay_ml.prediction_runs.writer import record_champion_prediction_run

            champion_cfg = normalize_training_config({
                **config.to_dict(),
                "features": chosen["features"],
                "parameters": chosen_params,
                "model_name": model_name,
            })
            prediction_run_doc = record_champion_prediction_run(
                data_dir=data_dir,
                model_id=model_name,
                model_version=config.model_version,
                config_dict=champion_cfg.to_dict(),
                metadata=metadata,
                dataset_name=config.dataset,
                target=config.target,
                features=list(chosen["features"]),
                parameters=dict(chosen_params),
                wf_cfg=wf_cfg,
                fold_defs=wf_fold_defs,
                X=X,
                y=y,
                context_df=context_df,
                package_dir=package_dir,
                training_duration_sec=max(0.0, time.monotonic() - wf_started),
                algorithm=config.algorithm,
                prediction_type=config.prediction_type,
                score_refs=wf_score_refs,
                cancel_check=cancel_check,
            )
            log.log(f"Prediction run saved: {prediction_run_doc.get('run_id')}")
        except Exception as exc:
            log.log(f"Prediction run persistence skipped: {exc}")
        trained = chosen["trained"]
        model = chosen["model"]
        booster = chosen["booster"]
        use_features = chosen["features"]
        training_meta = dict(chosen["training_meta"])
        validation_loss_curve = chosen["validation_loss_curve"]
        training_meta["optimization_result"] = optimization_result
        from .algorithm_runtime import format_runtime_log_block

        for _line in format_runtime_log_block(training_meta.get("algorithm_runtime")):
            log.log(_line)
        composite_scores_doc = _build_composite_scores_doc(
            hpo_result=hpo_result,
            chosen=chosen,
            optimization_result=optimization_result,
        )

        tracker.emit(
            "training",
            "done",
            current_tree=chosen["trees_trained"],
            trees_total=int((train_params_tuned if optimization_result["winner"] == "tuned" else train_params_base).get("n_estimators", 1000)),
            train_progress_pct=100,
            early_stopped=trained["early_stopped"],
            training_meta=training_meta,
            validation_loss_curve=validation_loss_curve,
            selected_features=use_features,
        )
        monitor.set_phase("Evaluation")
        log.log("Evaluation on holdout test set")
        tracker.emit("evaluation", "running")
        _check_cancel()
        aggregated = enrich_aggregated_from_fold_results(
            aggregated,
            wf_result.get("fold_results") or (wf_result["summary"].get("fold_results") if isinstance(wf_result.get("summary"), dict) else None),
        )
        metrics = {
            "validation": chosen["production_validation_metrics"],
            "test": chosen["test_metrics"],
            "walk_forward": aggregated,
            "production_walk_forward": chosen["production_walk_forward_aggregate"],
            "production_fold_results": chosen.get("production_fold_results") or [],
            "composite_scores": composite_scores_doc,
            "training_meta": training_meta,
            "optimization_result": optimization_result,
        }
        if hpo_result:
            metrics["hyperparameter_optimization"] = {
                "best_objective": hpo_result.get("best_objective"),
                "optimization_metric": hpo_result.get("optimization_metric"),
                "best_parameters": hpo_result.get("best_search_params"),
                "n_trials": hpo_result.get("n_trials"),
                "best_validation_composite": composite_scores_doc.get("best_validation_composite"),
                "production_composite": composite_scores_doc.get("production_composite"),
            }
        tracker.emit("evaluation", "done", metrics=metrics, walk_forward=aggregated)
        tracker.update_dashboard(test_rmse=metrics["test"].get("rmse"))

        importance = feature_importance_df(model, use_features)
        holdout_val_X = X.iloc[max(0, test_sl.start - int(wf_cfg.get("validation_window_size") or 1000)):test_sl.start][use_features]
        shap_rows = compute_shap_importance(booster, holdout_val_X, use_features) if len(holdout_val_X) else []

        final_config = normalize_training_config({
            **config.to_dict(),
            "features": use_features,
            "parameters": (train_params_tuned if optimization_result["winner"] == "tuned" else train_params_base),
            "model_name": model_name,
        })

        monitor.set_phase("Saving")
        log.log("Saving model package")
        tracker.emit("saving", "running", feature_importance=importance.head(20).to_dict(orient="records"))
        _check_cancel()

        wf_dir = os.path.join(package_dir, "walk_forward")
        os.makedirs(wf_dir, exist_ok=True)
        champion_agg_doc = {
            "champion": optimization_result.get("winner"),
            "production_model_file": champion_model_file,
            "config": (wf_result.get("summary") or {}).get("config"),
            "meta": (wf_result.get("summary") or {}).get("meta"),
            "aggregated": chosen["production_walk_forward_aggregate"],
            "validation_metrics": chosen["production_validation_metrics"],
            "fold_results": [
                {k: v for k, v in r.items() if k != "feature_importance"}
                for r in (chosen.get("production_fold_results") or [])
            ],
            "composite_score": chosen["production_validation_metrics"].get("composite_score"),
            "evaluation_note": (
                "Full walk-forward re-evaluation with champion hyperparameters after final retrain "
                f"({champion_model_file})"
            ),
        }
        if prediction_run_doc:
            champion_agg_doc["prediction_run"] = {
                "run_id": prediction_run_doc.get("run_id"),
                "prediction_count": prediction_run_doc.get("prediction_count_stored"),
                "fold_count": prediction_run_doc.get("fold_count"),
                "status": prediction_run_doc.get("status"),
            }
        with open(os.path.join(wf_dir, "champion_aggregate.json"), "w", encoding="utf-8") as fh:
            json.dump(champion_agg_doc, fh, indent=2)

        try:
            model.save_model(native_model_path(package_dir, "model", config.algorithm))
        except Exception:
            pass

        monitor.stop()
        training_metadata_doc = _build_and_save_training_metadata(
            monitor=monitor,
            package_dir=package_dir,
            config=final_config,
            metadata=metadata,
            train_params=(train_params_tuned if optimization_result["winner"] == "tuned" else train_params_base),
            hpo_trials=int((hpo_result or {}).get("n_trials") or 0),
            best_trial=(hpo_result or {}).get("best_trial"),
            final_training_duration_sec=float(trained.get("training_time_sec") or 0.0),
            hpo_duration_sec=hpo_duration_sec,
            model_name=model_name,
            optimization_result=optimization_result,
            champion=str(optimization_result.get("winner") or "baseline"),
            production_model=champion_model_file,
        )

        saved = save_model_package(
            data_dir=data_dir,
            config=final_config,
            model=model,
            metrics=metrics,
            feature_importance=importance,
            metadata=metadata,
            matrix_report=matrix_report,
            split_info=parts,
            training_log_text=log.text(),
            trees_trained=trained["trees_trained"],
            early_stopped=trained["early_stopped"],
            training_time_sec=trained["training_time_sec"],
            total_elapsed_sec=tracker.elapsed_total(),
            training_meta=training_meta,
            validation_loss_curve=validation_loss_curve,
            walk_forward_summary=wf_result["summary"],
            shap_importance=shap_rows,
            training_metadata=training_metadata_doc,
        )
        tracker.emit(
            "saving",
            "done",
            model_name=saved["model_name"],
            feature_importance=importance.to_dict(orient="records"),
            training_summary=saved.get("training_summary"),
            training_metadata=training_metadata_doc,
            walk_forward=aggregated,
            feature_selection=selection_meta,
            selected_features=use_features,
            report_url="",
        )

        result = {
            "ok": True,
            "model_name": saved["model_name"],
            "package_dir": saved["package_dir"],
            "metrics": metrics,
            "feature_importance": importance.to_dict(orient="records"),
            "feature_importance_top20": importance.head(20).to_dict(orient="records"),
            "training_summary": saved.get("training_summary"),
            "training_meta": training_meta,
            "training_metadata": training_metadata_doc,
            "validation_loss_curve": validation_loss_curve,
            "walk_forward": aggregated,
            "feature_selection": selection_meta,
            "walk_forward_summary": wf_result["summary"],
            "selected_features": use_features,
            "shap_importance": shap_rows,
            "report_url": "",
            "matrix_report": matrix_report,
            "steps": TRAIN_STEP_ORDER + ["walk_forward"] + (["hyperparameter_optimization"] if hpo_result else []),
            "config": final_config.to_dict(),
            "hyperparameter_optimization": hpo_result,
            "optimization_result": optimization_result,
        }
        return _attach_post_training(
            data_dir=data_dir,
            saved=saved,
            result=result,
            on_progress=on_progress,
            post_training_config=dict(final_config.post_training or {}),
            X=X,
            y=y,
        )
    finally:
        monitor.stop()

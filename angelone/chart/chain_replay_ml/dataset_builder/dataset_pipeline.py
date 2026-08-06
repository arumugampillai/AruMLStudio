"""One-click dataset → audit → validation → train pipeline orchestrator."""

from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from chain_replay_ml.dataset_builder.audit_investigation_engine import (
    is_training_allowed,
    normalize_training_recommendation,
)
from chain_replay_ml.dataset_builder.audit_options import AuditOptions
from chain_replay_ml.dataset_builder.auditor import audit_dataset
from chain_replay_ml.dataset_builder.dataset_validator import (
    run_dataset_validation,
    save_audit_cache,
)
from chain_replay_ml.dataset_builder.orchestrator import DatasetBuildConfig, DatasetBuildOrchestrator
from chain_replay_ml.dataset_builder.writer import _safe_filename, datasets_dir
from chain_replay_ml.training.naming import suggest_model_name
from chain_replay_ml.training.orchestrator import train_model

ProgressCallback = Callable[[dict[str, Any]], None]

BUILD_STAGE_NAMES = [
    "Load Database",
    "Validate Sources",
    "Sampling Grid",
    "Strike Selection",
    "Prediction Targets",
    "Feature Generation",
    "Dataset Validation",
    "Write Parquet",
]

POST_BUILD_STAGES = [
    ("audit", "Dataset Audit"),
    ("validation", "Dataset Validation"),
    ("training", "Model Training"),
    ("evaluation", "Model Evaluation"),
    ("registry", "Model Registry"),
]


@dataclass
class PipelineOptions:
    run_audit: bool = True
    run_validation: bool = True
    train_model: bool = True
    model_prefix: str = ""
    validation_n_sample: int = 100
    validation_tolerance: float = 1e-6
    training_config: dict[str, Any] | None = None
    build_profile: str = "production"
    skip_feature_audit: bool = False
    skip_data_validation: bool = False
    skip_dataset_statistics: bool = False
    skip_distribution_report: bool = False
    skip_leakage_audit: bool = False
    skip_quality_report: bool = False
    allow_training_without_audit: bool = False

    def audit_options(self) -> AuditOptions:
        return AuditOptions(
            build_profile=self.build_profile,
            skip_feature_audit=self.skip_feature_audit,
            skip_data_validation=self.skip_data_validation,
            skip_dataset_statistics=self.skip_dataset_statistics,
            skip_distribution_report=self.skip_distribution_report,
            skip_leakage_audit=self.skip_leakage_audit,
            skip_quality_report=self.skip_quality_report,
            allow_training_without_audit=self.allow_training_without_audit,
        )

    @classmethod
    def from_mapping(cls, doc: dict[str, Any] | None) -> PipelineOptions:
        doc = doc or {}
        profile = str(doc.get("build_profile") or "production").lower()
        fast = profile == "fast_experiment"
        audit_opts = AuditOptions.from_mapping(doc)

        def _bool(key: str, default: bool) -> bool:
            if key in doc:
                return bool(doc.get(key))
            return default

        skip_validation = audit_opts.skip_data_validation
        run_audit = _bool("run_audit", not audit_opts.audit_heavy_work_skipped() if fast else True)
        run_validation = _bool("run_validation", not skip_validation if fast else True)

        return cls(
            run_audit=run_audit,
            run_validation=run_validation,
            train_model=_bool("train_model", True),
            model_prefix=str(doc.get("model_prefix") or ""),
            validation_n_sample=int(doc.get("validation_n_sample") or 100),
            validation_tolerance=float(doc.get("validation_tolerance") or 1e-6),
            training_config=doc.get("training_config"),
            build_profile=profile,
            skip_feature_audit=audit_opts.skip_feature_audit,
            skip_data_validation=audit_opts.skip_data_validation,
            skip_dataset_statistics=audit_opts.skip_dataset_statistics,
            skip_distribution_report=audit_opts.skip_distribution_report,
            skip_leakage_audit=audit_opts.skip_leakage_audit,
            skip_quality_report=audit_opts.skip_quality_report,
            allow_training_without_audit=_bool(
                "allow_training_without_audit",
                fast or audit_opts.allow_training_without_audit,
            ),
        )


@dataclass
class DatasetPipelineOrchestrator:
    data_dir: str
    build_config: DatasetBuildConfig
    options: PipelineOptions = field(default_factory=PipelineOptions)
    cancel_event: threading.Event = field(default_factory=threading.Event)

    def cancel(self) -> None:
        self.cancel_event.set()

    def _cancelled(self) -> bool:
        return self.cancel_event.is_set()

    def run(self, on_progress: ProgressCallback | None = None) -> dict[str, Any]:
        t0 = time.monotonic()
        opts = self.options
        dataset_name = self.build_config.dataset_name
        stages = self._seed_stages()
        result: dict[str, Any] = {
            "status": "running",
            "dataset_name": dataset_name,
            "pipeline": {"stages": stages, "substages": [], "total_elapsed_sec": 0},
        }

        def emit(**extra: Any) -> None:
            if not on_progress:
                return
            result["pipeline"]["total_elapsed_sec"] = round(time.monotonic() - t0, 2)
            on_progress({**result, **extra})

        def set_stage(stage_id: str, status: str, *, error: str | None = None, started: float | None = None) -> None:
            for st in stages:
                if st["id"] == stage_id:
                    st["status"] = status
                    if started is not None and status in ("done", "failed", "skipped"):
                        st["elapsed_sec"] = round(time.monotonic() - started, 2)
                        st["elapsed_label"] = _fmt_duration(st["elapsed_sec"])
                    if error:
                        st["error"] = error
                    break

        # --- Stage: build (8 internal substages via DatasetBuildOrchestrator) ---
        build_started = time.monotonic()
        set_stage("build", "running")
        emit(status="running", phase="build")

        build_orchestrator = DatasetBuildOrchestrator(
            config=self.build_config,
            cancel_event=self.cancel_event,
        )
        audit_opts = opts.audit_options()

        def on_build_progress(payload: dict[str, Any]) -> None:
            if self._cancelled():
                return
            pl = payload.get("pipeline") or {}
            if pl:
                for key in (
                    "total_elapsed_sec",
                    "total_elapsed_label",
                    "rows_per_sec",
                    "avg_rows_per_sec",
                    "eta_sec",
                    "eta_label",
                    "current_rows",
                    "estimated_total_rows",
                    "slowest_stage",
                    "active_stage",
                    "build_complete",
                ):
                    if key in pl:
                        result["pipeline"][key] = pl[key]
                build_sub = []
                for i, bst in enumerate(pl.get("stages") or []):
                    name = bst.get("name") or (BUILD_STAGE_NAMES[i] if i < len(BUILD_STAGE_NAMES) else f"Stage {i + 1}")
                    build_sub.append({
                        "id": bst.get("id") or i + 1,
                        "label": name,
                        "name": name,
                        "status": bst.get("status") or "waiting",
                        "elapsed_sec": bst.get("elapsed_sec"),
                        "elapsed_label": bst.get("elapsed_label"),
                        "progress_current": bst.get("progress_current"),
                        "progress_total": bst.get("progress_total"),
                        "progress_unit": bst.get("progress_unit"),
                        "parent_stage": "build",
                    })
                result["pipeline"]["substages"] = build_sub + (pl.get("substages") or [])
            result.update({k: v for k, v in payload.items() if k not in ("pipeline",)})
            emit(status="running", phase="build")

        try:
            build_result = build_orchestrator.run(on_progress=on_build_progress)
        except Exception as exc:
            set_stage("build", "failed", error=str(exc), started=build_started)
            result["status"] = "failed"
            result["error"] = str(exc)
            result["failed_stage"] = "build"
            emit(status="failed", error=str(exc))
            return result

        if self._cancelled() or build_result.status == "cancelled":
            set_stage("build", "failed", error="Cancelled", started=build_started)
            result["status"] = "cancelled"
            emit(status="cancelled")
            return result

        if build_result.status != "completed":
            err = build_result.error or "Dataset build failed"
            set_stage("build", "failed", error=err, started=build_started)
            result["status"] = "failed"
            result["error"] = err
            result["failed_stage"] = "build"
            emit(status="failed", error=err)
            return result

        set_stage("build", "done", started=build_started)
        result.update(build_result.to_dict())
        result["status"] = "running"
        emit(status="running", phase="build_done")

        master_only = bool(getattr(self.build_config, "also_write_master_db", False))
        if master_only:
            stats = build_result.dataset_stats or {}
            for stage_id, _ in POST_BUILD_STAGES:
                set_stage(stage_id, "skipped")
            result["status"] = "completed"
            result["master_dataset_only"] = True
            result["master_db_path"] = stats.get("master_db_path")
            result["message"] = (
                f"Master Dataset updated — {int(stats.get('rows') or 0):,} rows "
                f"({stats.get('master_db_path') or 'master DB'})"
            )
            result["completion"] = {
                "dataset": {
                    "dataset_name": dataset_name,
                    "rows": int(stats.get("rows") or stats.get("rows_added") or 0),
                    "features": stats.get("feature_count"),
                    "targets": stats.get("target_count"),
                    "master_db_path": stats.get("master_db_path"),
                    "storage": "master_sqlite",
                    "file_size_bytes": 0,
                    "audit_status": "Skipped",
                    "validation_status": "Done (build-stage)",
                },
            }
            emit(status="completed", phase="done")
            return result

        audit_report: dict[str, Any] | None = None
        validation_report: dict[str, Any] | None = None
        train_result: dict[str, Any] | None = None

        # --- Audit ---
        if opts.run_audit and not audit_opts.audit_heavy_work_skipped():
            audit_started = time.monotonic()
            set_stage("audit", "running")
            emit(status="running", phase="audit")

            def on_audit_progress(payload: dict[str, Any]) -> None:
                emit(status="running", phase="audit", audit_progress=payload)

            try:
                audit_report = audit_dataset(
                    data_dir=self.data_dir,
                    dataset_name=dataset_name,
                    on_progress=on_audit_progress,
                    audit_options=audit_opts,
                )
                if not self._cancelled():
                    try:
                        save_audit_cache(self.data_dir, dataset_name, audit_report)
                    except OSError:
                        pass
            except Exception as exc:
                set_stage("audit", "failed", error=str(exc), started=audit_started)
                result["status"] = "failed"
                result["error"] = str(exc)
                result["failed_stage"] = "audit"
                result["audit"] = {"error": str(exc)}
                emit(status="failed", error=str(exc), failed_stage="audit")
                return result

            ok, msg = _audit_passed(audit_report, allow_training_without_audit=opts.allow_training_without_audit)
            result["audit"] = _audit_summary(audit_report)
            if not ok:
                set_stage("audit", "failed", error=msg, started=audit_started)
                result["status"] = "failed"
                result["error"] = msg
                result["failed_stage"] = "audit"
                emit(status="failed", error=msg, failed_stage="audit")
                return result
            set_stage("audit", "done", started=audit_started)
        else:
            set_stage("audit", "skipped")
            if opts.run_audit and audit_opts.audit_heavy_work_skipped():
                result["audit"] = {"status": "skipped", "label": "Skipped (fast experiment)"}

        # --- Validation ---
        if opts.run_validation:
            val_started = time.monotonic()
            set_stage("validation", "running")
            emit(status="running", phase="validation")

            def on_val_progress(payload: dict[str, Any]) -> None:
                emit(status="running", phase="validation", validation_progress=payload)

            try:
                validation_report = run_dataset_validation(
                    self.data_dir,
                    dataset_name,
                    n_sample=opts.validation_n_sample,
                    tolerance=opts.validation_tolerance,
                    on_progress=on_val_progress,
                    save_cache=True,
                )
            except Exception as exc:
                set_stage("validation", "failed", error=str(exc), started=val_started)
                result["status"] = "failed"
                result["error"] = str(exc)
                result["failed_stage"] = "validation"
                result["validation"] = {"error": str(exc)}
                emit(status="failed", error=str(exc), failed_stage="validation")
                return result

            ok, msg = _validation_passed(validation_report)
            result["validation"] = _validation_summary(validation_report)
            if not ok:
                set_stage("validation", "failed", error=msg, started=val_started)
                result["status"] = "failed"
                result["error"] = msg
                result["failed_stage"] = "validation"
                emit(status="failed", error=msg, failed_stage="validation")
                return result
            set_stage("validation", "done", started=val_started)
        else:
            set_stage("validation", "skipped")

        # --- Training ---
        if opts.train_model:
            import gc

            gc.collect()
            train_started = time.monotonic()
            set_stage("training", "running")
            emit(status="running", phase="training")

            raw_train = build_default_training_config(
                self.data_dir,
                dataset_name,
                model_prefix=opts.model_prefix,
                training_override=opts.training_config,
            )

            def on_train_progress(payload: dict[str, Any]) -> None:
                step = payload.get("step")
                if step == "evaluation":
                    set_stage("training", "done", started=train_started)
                    set_stage("evaluation", "running")
                elif step == "saving":
                    set_stage("evaluation", "done", started=train_started)
                    set_stage("registry", "running")
                emit(status="running", phase="training", train_progress=payload)

            try:
                train_result = train_model(
                    data_dir=self.data_dir,
                    raw_config=raw_train,
                    on_progress=on_train_progress,
                    cancel_check=self._cancelled,
                )
            except Exception as exc:
                set_stage("training", "failed", error=str(exc), started=train_started)
                result["status"] = "failed"
                result["error"] = str(exc)
                result["failed_stage"] = "training"
                result["training"] = {"error": str(exc)}
                emit(status="failed", error=str(exc), failed_stage="training")
                return result

            if self._cancelled() or train_result.get("cancelled"):
                set_stage("training", "failed", error="Cancelled", started=train_started)
                result["status"] = "cancelled"
                emit(status="cancelled")
                return result

            if train_result.get("blocked") or not train_result.get("ok"):
                err = train_result.get("error") or "Training blocked or failed"
                set_stage("training", "failed", error=err, started=train_started)
                result["status"] = "failed"
                result["error"] = err
                result["failed_stage"] = "training"
                result["training"] = train_result
                emit(status="failed", error=err, failed_stage="training")
                return result

            set_stage("training", "done", started=train_started)
            set_stage("evaluation", "done", started=train_started)
            set_stage("registry", "done", started=train_started)
            result["training"] = _training_summary(train_result)
            result["model_name"] = train_result.get("model_name")
        else:
            set_stage("training", "skipped")
            set_stage("evaluation", "skipped")
            set_stage("registry", "skipped")

        result["status"] = "completed"
        result["completion"] = build_completion_summary(
            self.data_dir,
            dataset_name,
            audit_report=audit_report,
            validation_report=validation_report,
            train_result=train_result,
        )
        emit(status="completed")
        return result

    def _seed_stages(self) -> list[dict[str, Any]]:
        opts = self.options
        stages: list[dict[str, Any]] = [
            {
                "id": "build",
                "name": "Create Dataset",
                "status": "waiting",
                "elapsed_sec": None,
                "elapsed_label": None,
            },
        ]
        if opts.run_audit:
            stages.append({"id": "audit", "name": "Dataset Audit", "status": "waiting"})
        if opts.run_validation:
            stages.append({"id": "validation", "name": "Dataset Validation", "status": "waiting"})
        if opts.train_model:
            stages.extend([
                {"id": "training", "name": "Model Training", "status": "waiting"},
                {"id": "evaluation", "name": "Model Evaluation", "status": "waiting"},
                {"id": "registry", "name": "Model Registry", "status": "waiting"},
            ])
        return stages


def _fmt_duration(sec: float | None) -> str:
    if sec is None:
        return "—"
    s = float(sec)
    if s >= 3600:
        return f"{int(s // 3600)}h {int((s % 3600) // 60)}m"
    if s >= 60:
        return f"{int(s // 60)}m {s % 60:.1f}s"
    return f"{s:.2f} s"


def _audit_passed(report: dict[str, Any], *, allow_training_without_audit: bool = False) -> tuple[bool, str]:
    if allow_training_without_audit:
        label = (report.get("overall") or {}).get("label") or "Fast experiment — audit non-blocking"
        return True, label
    overall = str((report.get("overall") or {}).get("status") or "").lower()
    rec = normalize_training_recommendation(report)
    readiness = report.get("training_readiness") or {}
    critical = int(readiness.get("critical_count") or 0)
    if overall == "fail" or critical > 0:
        label = (report.get("overall") or {}).get("label") or "Audit failed"
        blocking = (readiness.get("blocking_issues") or report.get("blocking_issues") or [])
        detail = blocking[0] if blocking else label
        return False, f"Dataset audit failed: {detail}"
    if not is_training_allowed(rec) and overall == "fail":
        return False, "Dataset audit did not pass — not ready for training"
    return True, (report.get("overall") or {}).get("label") or "PASS"


def _validation_passed(report: dict[str, Any]) -> tuple[bool, str]:
    status = str(report.get("status") or "").lower()
    if status == "fail":
        failures = int(report.get("failures") or report.get("comparisons_failed") or 0)
        return False, f"Dataset validation failed ({failures} comparison failures)"
    return True, report.get("label") or "PASS"


def _audit_summary(report: dict[str, Any]) -> dict[str, Any]:
    overall = report.get("overall") or {}
    return {
        "status": overall.get("status"),
        "label": overall.get("label"),
        "passed": bool(report.get("passed")),
        "errors": int((report.get("result") or {}).get("errors") or 0),
        "warnings": int((report.get("result") or {}).get("warnings") or 0),
    }


def _validation_summary(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": report.get("status"),
        "label": report.get("label"),
        "comparisons": int(report.get("comparisons") or 0),
        "failures": int(report.get("failures") or report.get("comparisons_failed") or 0),
    }


def _training_summary(train_result: dict[str, Any]) -> dict[str, Any]:
    metrics = train_result.get("metrics") or {}
    val_m = metrics.get("validation") or {}
    test_m = metrics.get("test") or {}
    summary = train_result.get("training_summary") or {}
    return {
        "model_name": train_result.get("model_name"),
        "algorithm": (train_result.get("config") or {}).get("algorithm") or "xgboost",
        "target": (train_result.get("config") or {}).get("target"),
        "rmse": val_m.get("rmse"),
        "mae": val_m.get("mae"),
        "r2": val_m.get("r2"),
        "test_rmse": test_m.get("rmse"),
        "training_time_sec": summary.get("training_time_sec"),
        "registry_status": "registered",
        "report_url": train_result.get("report_url"),
    }


def build_default_training_config(
    data_dir: str,
    dataset_name: str,
    *,
    model_prefix: str = "",
    training_override: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Default training JSON — same defaults as Model Builder, bound to the new dataset."""
    safe = _safe_filename(dataset_name)
    meta_path = os.path.join(datasets_dir(data_dir), f"{safe}.json")
    metadata: dict[str, Any] = {}
    if os.path.isfile(meta_path):
        with open(meta_path, encoding="utf-8") as fh:
            metadata = json.load(fh)

    features = list(metadata.get("feature_columns") or [])
    targets = list(metadata.get("prediction_target_columns") or [])
    if "future_ltp_5m" in targets:
        target = "future_ltp_5m"
    elif targets:
        target = str(targets[-1])
    else:
        target = "future_ltp_5m"

    algorithm = "xgboost"
    base_name = suggest_model_name(target, algorithm)
    prefix = str(model_prefix or "").strip()
    model_name = f"{prefix}{base_name}" if prefix else base_name

    cfg: dict[str, Any] = {
        "dataset": dataset_name,
        "target": target,
        "algorithm": algorithm,
        "prediction_type": "regression",
        "features": features,
        "split": {
            "train": 70,
            "validation": 15,
            "test": 15,
            "strategy": "time_series",
        },
        "parameters": {
            "learning_rate": 0.05,
            "n_estimators": 1000,
            "max_depth": 6,
            "early_stopping_rounds": 100,
            "random_seed": 42,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 1,
            "reg_alpha": 0,
            "reg_lambda": 1,
        },
        "model_name": model_name,
        "model_version": "1.0",
        "model_description": _default_model_description(target, metadata),
    }

    if training_override:
        _deep_merge_training(cfg, training_override)
        cfg["dataset"] = dataset_name
        cfg["features"] = features
        cfg["target"] = target
        if prefix:
            base_from_override = str(training_override.get("model_name") or "").strip()
            if base_from_override and not base_from_override.startswith(prefix):
                cfg["model_name"] = f"{prefix}{base_from_override}"
            elif not str(cfg.get("model_name") or "").startswith(prefix):
                cfg["model_name"] = f"{prefix}{base_name}"

    row_count = int(metadata.get("row_count") or 0)
    build_profile = str(metadata.get("build_profile") or "production").lower()
    if build_profile == "fast_experiment" or row_count > 500_000:
        split = dict(cfg.get("split") or {})
        split["hyperparameter_optimization"] = {
            **dict(split.get("hyperparameter_optimization") or {}),
            "enabled": False,
        }
        wf = dict(split.get("walk_forward") or {})
        wf["hyperparameter_optimization"] = {
            **dict(wf.get("hyperparameter_optimization") or {}),
            "enabled": False,
        }
        split["walk_forward"] = wf
        cfg["split"] = split
        params = dict(cfg.get("parameters") or {})
        params["n_estimators"] = min(int(params.get("n_estimators") or 1000), 500)
        params["early_stopping_rounds"] = min(int(params.get("early_stopping_rounds") or 100), 50)
        cfg["parameters"] = params
    return cfg


def _default_model_description(target: str, metadata: dict[str, Any]) -> str:
    cfg = metadata.get("dataset_configuration") or {}
    label = cfg.get("prediction_horizon_5m_label")
    if target == "future_ltp_5m" and label:
        return f"Predicts option LTP after {label.lower()}."
    if str(target).startswith("future_ltp"):
        return f"Predicts option LTP ({target.replace('future_ltp_', '')} horizon)."
    return "Auto-trained from one-click dataset pipeline."


def _deep_merge_training(base: dict[str, Any], override: dict[str, Any]) -> None:
    for key, val in override.items():
        if key in ("split", "parameters", "xgboost") and isinstance(val, dict):
            block = base.setdefault("parameters" if key == "xgboost" else key, {})
            if key == "xgboost":
                block = base.setdefault("parameters", {})
            for k, v in val.items():
                block[k] = v
        elif val is not None and val != "":
            base[key] = val


def build_completion_summary(
    data_dir: str,
    dataset_name: str,
    *,
    audit_report: dict[str, Any] | None,
    validation_report: dict[str, Any] | None,
    train_result: dict[str, Any] | None,
) -> dict[str, Any]:
    safe = _safe_filename(dataset_name)
    parquet_path = os.path.join(datasets_dir(data_dir), f"{safe}.parquet")
    meta_path = os.path.join(datasets_dir(data_dir), f"{safe}.json")
    metadata: dict[str, Any] = {}
    if os.path.isfile(meta_path):
        with open(meta_path, encoding="utf-8") as fh:
            metadata = json.load(fh)

    parquet_bytes = os.path.getsize(parquet_path) if os.path.isfile(parquet_path) else 0
    dataset_block = {
        "dataset_name": dataset_name,
        "rows": int(metadata.get("row_count") or 0),
        "features": int(metadata.get("feature_count") or len(metadata.get("feature_columns") or [])),
        "targets": int(metadata.get("target_count") or 0),
        "file_size_bytes": parquet_bytes,
        "audit_status": _audit_summary(audit_report).get("label") if audit_report else "Skipped",
        "validation_status": _validation_summary(validation_report).get("label") if validation_report else "Skipped",
    }
    model_block = None
    if train_result and train_result.get("ok"):
        model_block = _training_summary(train_result)
    return {"dataset": dataset_block, "model": model_block}

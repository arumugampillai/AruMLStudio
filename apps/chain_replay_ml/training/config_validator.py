"""Pre-training configuration validation — gate before train job starts."""

from __future__ import annotations

import os
from typing import Any

from chain_replay_ml.dataset_builder.audit_options import audit_validation_required_for_dataset
from chain_replay_ml.dataset_builder.audit_investigation_engine import (
    is_training_allowed,
    normalize_training_recommendation,
)
from chain_replay_ml.dataset_builder.dataset_validator import (
    load_audit_cache,
    load_validation_cache,
)
from chain_replay_ml.dataset_builder.schema_registry import load_feature_registry
from chain_replay_ml.dataset_builder.writer import _safe_filename, datasets_dir

from .config import TrainingConfig, normalize_training_config
from .dataset_loader import load_dataset_frame, missing_parquet_columns, parquet_column_names
from .feature_matrix import check_duplicate_features
from .target_kinds import (
    is_allowed_training_target,
    target_prediction_type_compatible,
)


MIN_TRAINING_ROWS = 500


def _load_dataset_meta(meta_path: str) -> dict[str, Any]:
    try:
        import json

        with open(meta_path, encoding="utf-8") as fh:
            doc = json.load(fh)
        return doc if isinstance(doc, dict) else {}
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return {}


def _is_fast_experiment_dataset(meta: dict[str, Any]) -> bool:
    if str(meta.get("build_profile") or "").lower() == "fast_experiment":
        return True
    # Set only when build skips validation (fast experiment path)
    return bool(meta.get("validation_skipped"))


def _check_row(label: str, passed: bool, detail: str = "") -> dict[str, Any]:
    return {"id": label, "label": label, "passed": passed, "detail": detail}


def _format_missing_features_detail(missing: list[str], *, dataset: str = "") -> str:
    """Human-readable missing-feature list for the Train / Retrain gate UI."""
    names = [str(f).strip() for f in missing if str(f).strip()]
    if not names:
        return ""
    ds = str(dataset or "").strip()
    where = f" in dataset {ds}" if ds else " in selected dataset"
    n = len(names)
    if n <= 20:
        body = "\n".join(f"  • {name}" for name in names)
        return f"{n} feature(s) missing{where}:\n{body}"
    preview = ", ".join(names[:30])
    extra = n - 30
    suffix = f", … +{extra} more" if extra > 0 else ""
    return f"{n} feature(s) missing{where}: {preview}{suffix}"


_LIFECYCLE_SKIP_MODES = frozenset({
    "retrain",
    "complete_optimization",
    "feature_optimization",
})


def _lifecycle_skip_detail(config: TrainingConfig) -> str:
    lc = config.lifecycle or {}
    mode = str(lc.get("mode") or "").strip().lower()
    if mode == "retrain":
        return "Skipped (retrain lifecycle)"
    if mode == "complete_optimization":
        return "Skipped (complete optimization lifecycle)"
    if mode == "feature_optimization":
        return "Skipped (feature optimization lifecycle)"
    if mode in _LIFECYCLE_SKIP_MODES:
        return f"Skipped ({mode.replace('_', ' ')} lifecycle)"
    return "Skipped (user override)"


def validate_training_config(
    data_dir: str,
    raw_config: dict[str, Any],
) -> dict[str, Any]:
    config = normalize_training_config(raw_config)
    checks: list[dict[str, Any]] = []

    safe = _safe_filename(config.dataset)
    out_dir = datasets_dir(data_dir)
    parquet_path = os.path.join(out_dir, f"{safe}.parquet")
    meta_path = os.path.join(out_dir, f"{safe}.json")

    dataset_exists = os.path.isfile(parquet_path) and os.path.isfile(meta_path)
    checks.append(_check_row(
        "Dataset exists",
        dataset_exists,
        config.dataset if dataset_exists else f"Missing dataset: {config.dataset}",
    ))

    meta_doc = _load_dataset_meta(meta_path) if dataset_exists else {}
    gates_required = audit_validation_required_for_dataset(meta_doc)
    fast_experiment = _is_fast_experiment_dataset(meta_doc)
    skip_audit = config.skip_dataset_audit
    skip_validation = config.skip_dataset_validation

    audit_cache = load_audit_cache(data_dir, config.dataset) if dataset_exists else None
    training_rec = normalize_training_recommendation(audit_cache)
    audit_ok = is_training_allowed(training_rec)
    if skip_audit:
        audit_ok = True
        audit_detail = _lifecycle_skip_detail(config)
    elif not gates_required:
        audit_ok = True
        audit_detail = "Not required for this dataset"
    elif fast_experiment:
        audit_ok = True
        audit_detail = "Skipped (fast experiment build)"
    else:
        audit_detail = training_rec if audit_cache else "Run dataset audit first"
    checks.append(_check_row(
        "Dataset audit passed",
        audit_ok,
        audit_detail,
    ))

    validation_cache = load_validation_cache(data_dir, config.dataset) if dataset_exists else None
    val_status = str((validation_cache or {}).get("status") or "").lower()
    validation_ok = validation_cache is not None and val_status in ("pass", "warn")
    if skip_validation:
        validation_ok = True
        validation_detail = _lifecycle_skip_detail(config)
    elif not gates_required:
        validation_ok = True
        validation_detail = "Not required for this dataset"
    elif fast_experiment:
        validation_ok = True
        validation_detail = "Skipped (fast experiment build)"
    else:
        validation_detail = (validation_cache or {}).get("label") or "Run feature validation first"
    checks.append(_check_row(
        "Validation complete",
        validation_ok,
        validation_detail,
    ))

    target_ok = bool(config.target)
    registry = load_feature_registry()
    targets = set((registry.get("targets") or {}).keys())
    target_in_schema = is_allowed_training_target(
        config.target,
        registry_targets=targets,
        meta=meta_doc,
    )
    label_run_id = str(getattr(config, "label_run_id", None) or "").strip()
    label_run_ok = True
    label_run_detail = "not used"
    if label_run_id:
        try:
            from chain_replay_ml.label_runs import get_label_run

            rec = get_label_run(data_dir, label_run_id)
            label_run_ok = bool(rec.exists)
            label_run_detail = rec.run_id if rec.exists else f"missing: {label_run_id}"
            if rec.exists:
                # Target comes from Label Run — parquet need not contain it.
                primary = str(rec.primary_target or "").strip()
                if primary:
                    config_target = str(config.target or "").strip()
                    if config_target and config_target != primary:
                        # Prefer Label Run primary; treat as OK if both classification-like.
                        pass
                    target_ok = True
                    target_in_schema = True
                    if not config_target:
                        target_ok = bool(primary)
        except Exception as exc:
            label_run_ok = False
            label_run_detail = str(exc)
            target_ok = False
    elif dataset_exists and target_ok:
        missing_target = missing_parquet_columns(parquet_path, [config.target])
        if missing_target:
            target_ok = False
        elif not parquet_column_names(parquet_path):
            try:
                df, _, _ = load_dataset_frame(data_dir, config.dataset, columns=[config.target])
                target_ok = config.target in df.columns
            except Exception:
                target_ok = False
    checks.append(_check_row(
        "Target exists",
        target_ok and target_in_schema,
        (
            f"{config.target} via Label Run"
            if label_run_id and target_ok
            else (config.target or "No target selected")
        ),
    ))
    if label_run_id or str(getattr(config, "label_strategy", "") or "").lower() == "triple_barrier":
        checks.append(_check_row(
            "Label Run",
            label_run_ok if label_run_id else False,
            label_run_detail if label_run_id else "Select a Label Run (or Open Outcome Label Engine)",
        ))

    feature_count = len(config.features)
    features_ok = feature_count > 0
    missing_feats: list[str] = []
    if dataset_exists and features_ok:
        missing_feats = missing_parquet_columns(parquet_path, list(config.features))
        if missing_feats:
            features_ok = False
        elif not parquet_column_names(parquet_path):
            try:
                df, _, _ = load_dataset_frame(
                    data_dir,
                    config.dataset,
                    columns=list(dict.fromkeys([*config.features, config.target])),
                )
                missing_feats = [f for f in config.features if f not in df.columns]
                features_ok = not missing_feats
            except Exception:
                features_ok = False
    if not features_ok and feature_count <= 0:
        features_detail = "No features selected"
    elif missing_feats:
        features_detail = _format_missing_features_detail(
            missing_feats,
            dataset=config.dataset,
        )
    else:
        features_detail = f"{feature_count} selected"
    checks.append(_check_row(
        f"{feature_count} Features exist",
        features_ok,
        features_detail,
    ))

    dups = check_duplicate_features(config.features)
    checks.append(_check_row(
        "No duplicate features",
        not dups,
        ", ".join(dups) if dups else "OK",
    ))

    target_type_ok = target_prediction_type_compatible(config.prediction_type, config.target)
    checks.append(_check_row(
        "Target / prediction type",
        target_type_ok,
        (
            f"{config.prediction_type} → {config.target}"
            if target_type_ok
            else (
                "Need regression+future_ltp*/ormp_return_* "
                "or binary/classification+target_reached|hit|label_up_*|ormp_direction_* "
                f"(got {config.prediction_type}/{config.target})"
            )
        ),
    ))

    from .algorithm_capabilities import (
        algorithm_supports_prediction_type,
        get_algorithm_capabilities,
    )

    algo_pred_ok = algorithm_supports_prediction_type(config.algorithm, config.prediction_type)
    caps = get_algorithm_capabilities(config.algorithm)
    checks.append(_check_row(
        "Algorithm / prediction type",
        algo_pred_ok,
        (
            f"{caps.label} supports {config.prediction_type}"
            if algo_pred_ok
            else (
                f"{caps.label} does not support '{config.prediction_type}' "
                f"(regression={caps.supports_regression}, "
                f"binary={caps.supports_binary_classification}, "
                f"multiclass={caps.supports_multiclass})"
            )
        ),
    ))

    row_count = 0
    if dataset_exists:
        try:
            import json

            with open(meta_path, encoding="utf-8") as fh:
                meta = json.load(fh)
            row_count = int(meta.get("row_count") or 0)
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            row_count = 0
    enough_rows = row_count >= MIN_TRAINING_ROWS
    checks.append(_check_row(
        "Enough rows",
        enough_rows,
        f"{row_count:,} rows" if row_count else "Unknown row count",
    ))

    strategy = str(config.split.get("strategy") or "time_series")
    if strategy == "walk_forward" and row_count:
        from .split import normalize_walk_forward_config, walk_forward_fold_slices

        try:
            wf_cfg = normalize_walk_forward_config(config.split, row_count)
            walk_forward_fold_slices(row_count, wf_cfg)
            wf_ok = True
            wf_detail = (
                f"{wf_cfg['n_folds']} folds · train={wf_cfg['train_window_size']:,} · "
                f"val={wf_cfg['validation_window_size']:,} · test holdout={wf_cfg['test_holdout_rows']:,}"
            )
        except Exception as exc:
            wf_ok = False
            wf_detail = str(exc)
        checks.append(_check_row("Walk-forward layout", wf_ok, wf_detail))

    nan_target_ok = True
    nan_detail = "OK"
    if target_ok and config.target:
        try:
            label_run_id = str(getattr(config, "label_run_id", None) or "").strip()
            if label_run_id:
                # Phase X: target lives on the Label Run, not the feature parquet.
                from chain_replay_ml.label_runs import load_label_run_frame, load_label_run_meta

                meta = load_label_run_meta(data_dir, label_run_id)
                primary = str(meta.get("primary_target") or config.target).strip()
                # Prefer checking only rows that would be joined for training.
                lab = load_label_run_frame(data_dir, label_run_id)
                if "is_valid" in lab.columns:
                    lab = lab[lab["is_valid"].fillna(True).astype(bool)]
                if primary not in lab.columns:
                    raise KeyError(f"Label Run missing primary target {primary!r}")
                nan_count = int(lab[primary].isna().sum())
                nan_target_ok = nan_count == 0 and len(lab) > 0
                if len(lab) == 0:
                    nan_detail = "Label Run has 0 valid rows"
                else:
                    nan_detail = (
                        f"{nan_count:,} NaN in Label Run"
                        if nan_count
                        else f"OK (Label Run · {len(lab):,} valid)"
                    )
            elif dataset_exists:
                df, _, _ = load_dataset_frame(data_dir, config.dataset, columns=[config.target])
                nan_count = int(df[config.target].isna().sum())
                valid_count = len(df) - nan_count
                nan_target_ok = valid_count >= MIN_TRAINING_ROWS
                nan_detail = f"OK ({nan_count:,} NaN dropped · {valid_count:,} valid)" if nan_count else "OK"
        except Exception as exc:
            nan_target_ok = False
            nan_detail = str(exc)
    checks.append(_check_row("No NaN target", nan_target_ok, nan_detail))

    all_passed = all(c["passed"] for c in checks)
    checks.append(_check_row("Ready", all_passed, "All checks passed" if all_passed else "Training blocked"))

    return {
        "ready": all_passed,
        "blocked": not all_passed,
        "training_blocked": not all_passed,
        "checks": checks,
        "missing_features": list(missing_feats),
        "config": config.to_dict(),
        "training_recommendation": training_rec,
        "build_profile": meta_doc.get("build_profile") or "production",
        "fast_experiment": fast_experiment,
        "audit_validation_required": gates_required,
        "skip_dataset_audit": skip_audit,
        "skip_dataset_validation": skip_validation,
    }

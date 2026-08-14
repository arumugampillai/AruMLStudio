"""Auto Candidate Generation — standalone config builder (not Manual tab)."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_HORIZONS_SEC: tuple[int, ...] = (6, 12, 30, 60, 120, 300)

TRANSFORM_LABELS: tuple[tuple[str, str], ...] = (
    ("lag", "Lag"),
    ("difference", "Difference"),
    ("return", "Return"),
    ("rolling", "Rolling Statistics"),
    ("exponential_rolling", "Exponential Rolling"),
    ("normalization", "Normalization"),
    ("regime", "Regime / Bucket"),
    ("math", "Math"),
    ("interaction", "Interaction"),
)

INTERACTION_OP_LABELS: tuple[tuple[str, str], ...] = (
    ("multiply", "Multiply"),
    ("divide", "Divide"),
    ("add", "Add"),
    ("subtract", "Subtract"),
    ("absolute_difference", "Absolute Difference"),
    ("ratio", "Ratio"),
)

SOURCE_OPTIONS: tuple[tuple[str, str], ...] = (
    ("pipeline", "Pipeline Features"),
    ("registry", "Feature Registry"),
    ("both", "Both"),
)

_DEFAULT_TRANSFORMS = {key: True for key, _ in TRANSFORM_LABELS}
_DEFAULT_HORIZONS = list(DEFAULT_HORIZONS_SEC)
_DEFAULT_OPS = {key: True for key, _ in INTERACTION_OP_LABELS}


def default_candidate_generation_prefs() -> dict[str, Any]:
    return {
        "source": "registry",
        "transformations": dict(_DEFAULT_TRANSFORMS),
        "horizons_sec": list(_DEFAULT_HORIZONS),
        "interaction_ops": dict(_DEFAULT_OPS),
    }


def normalize_candidate_generation_prefs(raw: dict[str, Any] | None) -> dict[str, Any]:
    src = raw if isinstance(raw, dict) else {}
    out = default_candidate_generation_prefs()
    source = str(src.get("source") or "registry").strip().lower()
    if source not in {k for k, _ in SOURCE_OPTIONS}:
        source = "registry"
    out["source"] = source

    transforms = src.get("transformations")
    if isinstance(transforms, dict):
        for key, _ in TRANSFORM_LABELS:
            if key in transforms:
                out["transformations"][key] = bool(transforms[key])

    horizons_raw = src.get("horizons_sec") or src.get("horizons")
    if isinstance(horizons_raw, list):
        horizons: list[int] = []
        for h in horizons_raw:
            try:
                horizons.append(int(h))
            except (TypeError, ValueError):
                continue
        if horizons:
            out["horizons_sec"] = sorted({h for h in horizons if h > 0})

    ops_raw = src.get("interaction_ops")
    if isinstance(ops_raw, dict):
        for key, _ in INTERACTION_OP_LABELS:
            if key in ops_raw:
                out["interaction_ops"][key] = bool(ops_raw[key])

    return out


def candidate_generation_prefs_snapshot(
    *,
    source: str,
    transformations: dict[str, bool],
    horizons_sec: list[int],
    interaction_ops: dict[str, bool],
) -> dict[str, Any]:
    return normalize_candidate_generation_prefs(
        {
            "source": source,
            "transformations": transformations,
            "horizons_sec": horizons_sec,
            "interaction_ops": interaction_ops,
        }
    )


def resolve_source_features(
    chart_dir: str,
    pipeline_id: str,
    source: str,
) -> list[str]:
    from chain_replay_ml.dataset_builder.feature_sources_catalog import (
        pipeline_feature_names,
        registry_feature_names,
    )
    from chain_replay_ml.dataset_builder.pipeline_registry_store import (
        resolve_registry_names,
    )

    from .build_service import chart_data_dir
    from .pipeline_registry_service import get_pipeline

    data_dir = chart_data_dir(chart_dir)
    row = get_pipeline(chart_dir, pipeline_id) or {}
    reg_ids = list(row.get("registry_feature_ids") or [])
    reg_names = resolve_registry_names(data_dir, reg_ids) if reg_ids else []
    if not reg_names:
        reg_names = registry_feature_names(data_dir=data_dir)

    pipe_catalog = pipeline_feature_names(data_dir=data_dir)
    pipe_candidates = list(row.get("candidate_features") or [])
    pipe_names = list(dict.fromkeys(pipe_catalog + pipe_candidates))

    mode = str(source or "registry").strip().lower()
    if mode == "pipeline":
        return pipe_names
    if mode == "registry":
        return reg_names
    return list(dict.fromkeys(reg_names + pipe_names))


def _interaction_op_keys(interaction_ops: dict[str, bool]) -> list[str]:
    keys: list[str] = []
    for key, label in INTERACTION_OP_LABELS:
        if not interaction_ops.get(key):
            continue
        if key == "ratio":
            keys.append("divide")
        else:
            keys.append(key)
    return list(dict.fromkeys(keys))


def build_auto_candidate_transformation_config(
    *,
    features: list[str],
    interval_sec: int,
    candidate_prefs: dict[str, Any],
) -> dict[str, Any]:
    """Build transformation config using Auto tab settings (reuses Manual merge helpers)."""
    prefs = normalize_candidate_generation_prefs(candidate_prefs)
    transforms = prefs["transformations"]
    horizons = [int(h) for h in prefs["horizons_sec"]]
    feats = [str(f).strip() for f in features if str(f).strip()]
    if not feats:
        return {"transformation_pipeline_version": 1, "transformations": []}

    partition = ["trading_day", "token"]
    interval = max(1, int(interval_sec))

    from chain_replay_ml.dataset_builder.transformations.lag_ui import (
        build_lag_transformation_config,
    )
    from chain_replay_ml.dataset_builder.transformations.rolling_ui import (
        merge_rolling_into_config,
    )
    from chain_replay_ml.dataset_builder.transformations.exponential_rolling_ui import (
        merge_exponential_rolling_into_config,
    )
    from chain_replay_ml.dataset_builder.transformations.interaction_ui import (
        bulk_interaction_pairs,
        merge_interaction_into_config,
    )
    from chain_replay_ml.dataset_builder.transformations.math_ui import (
        merge_math_into_config,
    )
    from chain_replay_ml.dataset_builder.transformations.normalization_ui import (
        merge_normalization_into_config,
    )
    from chain_replay_ml.dataset_builder.transformations.regime_ui import (
        merge_regime_into_config,
    )

    base = build_lag_transformation_config(
        enabled=bool(transforms.get("lag")),
        features=feats,
        lag_seconds=horizons,
        partition_by=partition,
        sample_interval_sec=interval,
        difference_enabled=bool(transforms.get("difference")),
        difference_features=feats,
        difference_lag_seconds=horizons,
        return_enabled=bool(transforms.get("return")),
        return_features=feats,
        return_lag_seconds=horizons,
    )

    roll_windows = sorted({h for h in horizons if h >= 30}) or [30, 60, 120, 300]
    with_rolling = merge_rolling_into_config(
        base,
        enabled=bool(transforms.get("rolling")),
        features=feats,
        windows=roll_windows,
        operations=["mean", "std", "min", "max"],
        partition_by=partition,
        sample_interval_sec=interval,
    )
    exp_periods = sorted({h for h in horizons if h >= 30}) or [30, 60, 120, 300]
    with_exp = merge_exponential_rolling_into_config(
        with_rolling,
        enabled=bool(transforms.get("exponential_rolling")),
        features=feats,
        periods=exp_periods,
        operations=["mean", "ema"],
        partition_by=partition,
        sample_interval_sec=interval,
    )
    with_math = merge_math_into_config(
        with_exp,
        enabled=bool(transforms.get("math")),
        features=feats,
        operations=["abs", "log", "sign"],
        partition_by=partition,
        sample_interval_sec=interval,
    )
    norm_windows = roll_windows
    with_norm = merge_normalization_into_config(
        with_math,
        enabled=bool(transforms.get("normalization")),
        features=feats,
        methods=["zscore"],
        windows=norm_windows,
        partition_by=partition,
        sample_interval_sec=interval,
    )
    with_regime = merge_regime_into_config(
        with_norm,
        enabled=bool(transforms.get("regime")),
        features=feats,
        methods=["bucket"],
        windows=norm_windows,
        n_bins=5,
        threshold=0.0,
        low=0.0,
        high=1.0,
        partition_by=partition,
        sample_interval_sec=interval,
    )

    pairs: list[dict[str, Any]] = []
    if transforms.get("interaction"):
        ops = _interaction_op_keys(prefs["interaction_ops"])
        # Cap pairwise explosion for very wide source sets.
        ix_feats = feats[:40]
        for op in ops:
            pairs.extend(
                bulk_interaction_pairs(ix_feats, ix_feats, op=op, skip_identical=True)
            )

    return merge_interaction_into_config(
        with_regime,
        enabled=bool(transforms.get("interaction")) and bool(pairs),
        pairs=pairs,
    )


@dataclass
class CandidateGenerationReport:
    target_pipeline_id: str = ""
    source_mode: str = ""
    source_feature_count: int = 0
    selected_transformations: dict[str, bool] = field(default_factory=dict)
    selected_horizons: list[int] = field(default_factory=list)
    selected_interaction_ops: dict[str, bool] = field(default_factory=dict)
    combinations_estimated: int = 0
    candidates_generated: int = 0
    candidates_rejected_policy: int = 0
    candidates_rejected_duplicates: int = 0
    candidates_added: int = 0
    errors: list[str] = field(default_factory=list)
    policy_rejected_names: list[str] = field(default_factory=list)
    duplicate_names: list[str] = field(default_factory=list)
    new_names: list[str] = field(default_factory=list)


def _policy_reject_names(
    names: list[str],
    *,
    source_features: set[str],
    registry_names: set[str],
) -> tuple[list[str], list[str]]:
    from chain_replay_ml.dataset_builder.transformations.lag_ui import META_SKIP_COLUMNS

    allowed: list[str] = []
    rejected: list[str] = []
    for name in names:
        n = str(name).strip()
        if not n:
            continue
        if n in META_SKIP_COLUMNS:
            rejected.append(n)
            continue
        if n in source_features:
            rejected.append(n)
            continue
        if n in registry_names:
            rejected.append(n)
            continue
        allowed.append(n)
    return allowed, rejected


def _estimate_combinations(
    *,
    source_count: int,
    prefs: dict[str, Any],
) -> int:
    if source_count <= 0:
        return 0
    transforms = prefs.get("transformations") if isinstance(prefs.get("transformations"), dict) else {}
    horizons = [int(h) for h in (prefs.get("horizons_sec") or []) if int(h) > 0]
    h = max(len(horizons), 1)
    n = source_count
    est = 0
    if transforms.get("lag"):
        est += n * h
    if transforms.get("difference"):
        est += n * h
    if transforms.get("return"):
        est += n * h
    if transforms.get("rolling"):
        roll_h = len([x for x in horizons if x >= 30]) or 4
        est += n * roll_h * 4
    if transforms.get("exponential_rolling"):
        exp_h = len([x for x in horizons if x >= 30]) or 4
        est += n * exp_h * 2
    if transforms.get("math"):
        est += n * 3
    if transforms.get("normalization"):
        roll_h = len([x for x in horizons if x >= 30]) or 4
        est += n * roll_h
    if transforms.get("regime"):
        roll_h = len([x for x in horizons if x >= 30]) or 4
        est += n * roll_h
    if transforms.get("interaction"):
        ops = prefs.get("interaction_ops") if isinstance(prefs.get("interaction_ops"), dict) else {}
        op_count = sum(1 for v in ops.values() if v)
        capped = min(n, 40)
        est += capped * capped * max(op_count, 1)
    return est


def _log_candidate_generation_report(report: CandidateGenerationReport) -> None:
    logger.info("Auto Candidate Generation")
    logger.info("  Target pipeline: %s", report.target_pipeline_id)
    logger.info("  Source feature count: %s", report.source_feature_count)
    logger.info("  Selected transformations: %s", report.selected_transformations)
    logger.info("  Selected horizons: %s", report.selected_horizons)
    logger.info("  Selected interaction operations: %s", report.selected_interaction_ops)
    logger.info("  Candidate combinations estimated: %s", report.combinations_estimated)
    logger.info("  Candidates generated: %s", report.candidates_generated)
    logger.info("  Candidates rejected by policy: %s", report.candidates_rejected_policy)
    logger.info("  Candidates rejected as duplicates: %s", report.candidates_rejected_duplicates)
    logger.info("  Candidates finally added: %s", report.candidates_added)
    if report.errors:
        logger.info("  Errors: %s", report.errors)
    if report.policy_rejected_names:
        logger.debug("  Policy rejected sample: %s", report.policy_rejected_names[:20])


def generate_pipeline_candidate_names(
    *,
    chart_dir: str,
    pipeline_id: str,
    interval_sec: int,
    candidate_prefs: dict[str, Any],
) -> CandidateGenerationReport:
    """Plan candidate feature names from Auto tab settings (not Manual tab)."""
    prefs = normalize_candidate_generation_prefs(candidate_prefs)
    pid = str(pipeline_id or "").strip().upper()
    report = CandidateGenerationReport(
        target_pipeline_id=pid,
        source_mode=str(prefs.get("source") or "registry"),
        selected_transformations=dict(prefs.get("transformations") or {}),
        selected_horizons=list(prefs.get("horizons_sec") or []),
        selected_interaction_ops=dict(prefs.get("interaction_ops") or {}),
    )

    features = resolve_source_features(chart_dir, pid, prefs.get("source", "registry"))
    report.source_feature_count = len(features)
    report.combinations_estimated = _estimate_combinations(
        source_count=len(features),
        prefs=prefs,
    )

    if not features:
        report.errors.append("No source features resolved for the selected source.")
        _log_candidate_generation_report(report)
        return report

    try:
        config = build_auto_candidate_transformation_config(
            features=features,
            interval_sec=interval_sec,
            candidate_prefs=prefs,
        )
    except Exception as exc:
        report.errors.append(str(exc))
        _log_candidate_generation_report(report)
        return report

    from chain_replay_ml.dataset_builder.pipeline_features_config import (
        expected_pipeline_outputs_from_config,
    )
    from chain_replay_ml.dataset_builder.feature_sources_catalog import registry_feature_names

    try:
        names = expected_pipeline_outputs_from_config(
            config,
            master_features=features,
        )
    except Exception as exc:
        report.errors.append(str(exc))
        _log_candidate_generation_report(report)
        return report

    report.candidates_generated = len(names)

    from .build_service import chart_data_dir

    data_dir = chart_data_dir(chart_dir)
    registry_names = set(registry_feature_names(data_dir=data_dir))
    source_set = set(features)
    allowed, policy_rejected = _policy_reject_names(
        names,
        source_features=source_set,
        registry_names=registry_names,
    )
    report.policy_rejected_names = policy_rejected
    report.candidates_rejected_policy = len(policy_rejected)

    from .pipeline_registry_service import get_pipeline

    row = get_pipeline(chart_dir, pid) or {}
    existing = set(row.get("candidate_features") or [])
    new_names: list[str] = []
    duplicates: list[str] = []
    seen: set[str] = set()
    for name in allowed:
        if name in seen:
            continue
        seen.add(name)
        if name in existing:
            duplicates.append(name)
            continue
        new_names.append(name)

    report.duplicate_names = duplicates
    report.candidates_rejected_duplicates = len(duplicates)
    report.new_names = new_names
    report.candidates_added = len(new_names)

    _log_candidate_generation_report(report)
    return report


def expected_candidate_feature_names(
    *,
    chart_dir: str,
    pipeline_id: str,
    interval_sec: int,
    candidate_prefs: dict[str, Any],
) -> tuple[list[str], list[str], list[str]]:
    """Return (new_names, skipped_existing, errors)."""
    report = generate_pipeline_candidate_names(
        chart_dir=chart_dir,
        pipeline_id=pipeline_id,
        interval_sec=interval_sec,
        candidate_prefs=candidate_prefs,
    )
    return list(report.new_names), list(report.duplicate_names), list(report.errors)

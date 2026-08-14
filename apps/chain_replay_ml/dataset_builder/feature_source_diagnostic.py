"""Diagnostic tracing for Analysis Dataset feature-source counts.

Traces Base Pipeline catalogue features and experimental pipeline candidates through
the same configuration steps used by ``analysis_dataset_export`` without modifying
build behaviour.  Intended for investigation only.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Literal

ReasonGroup = Literal[
    "IMPLEMENTATION_ERROR",
    "UNSUPPORTED_INTERVAL",
    "DEPENDENCY_MISSING",
    "POLICY_REJECTED",
    "WARMUP",
    "NULL_FILTER",
    "SOURCE_DATA_MISSING",
    "NOT_REQUESTED",
    "NAMING_CLASSIFICATION_ERROR",
    "DUPLICATE",
    "OTHER",
]

REASON_GROUPS: tuple[str, ...] = (
    "IMPLEMENTATION_ERROR",
    "UNSUPPORTED_INTERVAL",
    "DEPENDENCY_MISSING",
    "POLICY_REJECTED",
    "WARMUP",
    "NULL_FILTER",
    "SOURCE_DATA_MISSING",
    "NOT_REQUESTED",
    "NAMING_CLASSIFICATION_ERROR",
    "DUPLICATE",
    "OTHER",
)


@dataclass
class FeatureTrace:
    name: str
    feature_id: str = ""
    transformation: str = ""
    horizon_sec: str = ""
    interval_sec: float = 0.0
    pipeline_id: str = ""
    definition_generated: bool = False
    in_transform_plan: bool = False
    in_parquet: bool = False
    null_filter_dropped: bool = False
    classified_bucket: str = ""
    reason_code: str = ""
    reason_group: str = "OTHER"
    detail: str = ""


@dataclass
class IntervalSummary:
    interval_sec: float
    base_catalogue: int = 0
    base_in_plan: int = 0
    base_in_parquet: int = 0
    base_missing: int = 0
    other_catalogue: int = 0
    other_in_plan: int = 0
    other_in_parquet: int = 0
    other_missing: int = 0


@dataclass
class DiagnosticReport:
    data_dir: str
    dataset_name: str = ""
    pipeline_id: str = ""
    interval_sec: float = 0.0
    interval_summaries: list[IntervalSummary] = field(default_factory=list)
    base_missing: list[FeatureTrace] = field(default_factory=list)
    other_missing: list[FeatureTrace] = field(default_factory=list)
    base_reason_counts: dict[str, int] = field(default_factory=dict)
    other_reason_counts: dict[str, int] = field(default_factory=dict)
    config_notes: list[str] = field(default_factory=list)


def _schema_meta(name: str) -> dict[str, Any]:
    try:
        from .schema_registry import column_meta

        return column_meta(name) or {}
    except Exception:
        return {}


def _feature_id(name: str) -> str:
    meta = _schema_meta(name)
    for key in ("feature_id", "id", "registry_id", "formula_ref"):
        val = str(meta.get(key) or "").strip()
        if val:
            return val
    try:
        from .feature_migration import PIPELINE_OWNED_GENERATORS

        gen = PIPELINE_OWNED_GENERATORS.get(name)
        if gen:
            return f"pipeline:{gen}"
    except Exception:
        pass
    return ""


def _transformation_label(name: str) -> str:
    try:
        from .feature_migration import PIPELINE_OWNED_GENERATORS

        gen = PIPELINE_OWNED_GENERATORS.get(name)
        if gen:
            return str(gen)
    except Exception:
        pass
    meta = _schema_meta(name)
    return str(meta.get("group_id") or meta.get("category") or meta.get("group") or "")


def _horizon_label(name: str) -> str:
    import re

    m = re.search(r"_(\d+s|\d+m)(?:_|$)", str(name))
    if m:
        return m.group(1)
    m = re.search(r"(\d+)s$", str(name))
    if m:
        return f"{m.group(1)}s"
    return ""


def _expected_outputs(config: dict[str, Any] | None) -> set[str]:
    from .pipeline_features_config import expected_pipeline_outputs_from_config

    return {
        str(n).strip()
        for n in expected_pipeline_outputs_from_config(config)
        if str(n).strip()
    }


def _build_base_config(
    data_dir: str,
    interval_sec: float,
    *,
    apply_build_exclude: bool = True,
) -> dict[str, Any]:
    from .pipeline_features_config import build_pipeline_features_transformation_config
    from .pipeline_features_prefs import (
        load_pipeline_output_prune_features,
        load_transformation_forbidden_features,
    )

    skip = (
        load_pipeline_output_prune_features(data_dir)
        if apply_build_exclude
        else frozenset()
    )
    forbidden = load_transformation_forbidden_features(data_dir) if apply_build_exclude else frozenset()
    return build_pipeline_features_transformation_config(
        sample_interval_sec=float(interval_sec),
        exclude_features=skip if apply_build_exclude else None,
        interaction_operand_skip=forbidden if apply_build_exclude else None,
        source_forbidden=forbidden if apply_build_exclude else None,
    )


def _build_analysis_config(
    data_dir: str,
    interval_sec: float,
    pipeline_id: str,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None, list[str]]:
    """Mirror ``analysis_dataset_export`` transformation_config assembly."""
    from .feature_sources_catalog import base_pipeline_feature_names
    from .pipeline_registry_store import (
        ensure_default_existing_pipeline,
        get_pipeline,
        is_base_pipeline_record,
        resolve_pipeline_dataset_feature_names,
    )
    from .pipeline_features_config import build_pipeline_features_transformation_config
    from .pipeline_features_prefs import (
        load_pipeline_output_prune_features,
        load_transformation_forbidden_features,
    )
    from .transformations.config import (
        merge_transformation_configs,
        prune_transformation_config_for_interval,
    )
    from .pipeline_features_config import prune_pipeline_transformation_config

    doc = ensure_default_existing_pipeline(data_dir)
    rec = get_pipeline(doc, pipeline_id)
    if not rec or is_base_pipeline_record(rec):
        return None, None, []

    experimental_transform = (
        rec.get("transformation_config")
        if isinstance(rec.get("transformation_config"), dict)
        else None
    )
    base_names = sorted(base_pipeline_feature_names(data_dir))
    experimental_names = resolve_pipeline_dataset_feature_names(data_dir, doc, pipeline_id)
    pipe_names = list(dict.fromkeys(base_names + experimental_names))

    forbidden = load_transformation_forbidden_features(data_dir)
    exclude_for_pipeline = load_pipeline_output_prune_features(data_dir)
    base_cfg = build_pipeline_features_transformation_config(
        sample_interval_sec=float(interval_sec),
        exclude_features=exclude_for_pipeline,
        interaction_operand_skip=forbidden,
        source_forbidden=forbidden,
    )
    if experimental_transform is not None:
        cfg = merge_transformation_configs(base_cfg, experimental_transform)
    else:
        cfg = base_cfg

    cfg = prune_transformation_config_for_interval(cfg, float(interval_sec))
    cfg = prune_pipeline_transformation_config(
        cfg,
        exclude_for_pipeline,
        interaction_operand_skip=forbidden,
        source_exclude=forbidden,
    )
    prov = rec
    return cfg, prov, pipe_names


def _map_reason_group(reason_code: str) -> str:
    mapping = {
        "NOT_IN_TRANSFORM_PLAN": "IMPLEMENTATION_ERROR",
        "CONFIG_BUILD_POLICY_PRUNE": "POLICY_REJECTED",
        "CONFIG_ANALYSIS_POLICY_PRUNE": "POLICY_REJECTED",
        "CONFIG_INTERVAL_PRUNE": "UNSUPPORTED_INTERVAL",
        "INTERACTION_DEPENDENCY": "DEPENDENCY_MISSING",
        "NULL_FILTER_DROPPED": "NULL_FILTER",
        "NOT_IN_PARQUET_AFTER_PLAN": "IMPLEMENTATION_ERROR",
        "CLASSIFICATION_MISMATCH": "NAMING_CLASSIFICATION_ERROR",
        "PRESENT_OK": "OTHER",
    }
    return mapping.get(reason_code, "OTHER")


def _diagnose_base_feature(
    name: str,
    *,
    data_dir: str,
    interval_sec: float,
    parquet_cols: set[str],
    null_dropped: set[str],
    final_plan: set[str],
    raw_plan: set[str],
    after_build_exclude_plan: set[str],
    after_interval_plan: set[str],
    after_analysis_prune_plan: set[str],
    classified_bucket: str = "",
) -> FeatureTrace:
    trace = FeatureTrace(
        name=name,
        feature_id=_feature_id(name),
        transformation=_transformation_label(name),
        horizon_sec=_horizon_label(name),
        interval_sec=float(interval_sec),
        definition_generated=name in raw_plan,
        in_transform_plan=name in final_plan,
        in_parquet=name in parquet_cols,
        null_filter_dropped=name in null_dropped,
        classified_bucket=classified_bucket,
    )

    if name in parquet_cols and classified_bucket and classified_bucket != "base_pipeline":
        trace.reason_code = "CLASSIFICATION_MISMATCH"
        trace.detail = f"Column present but classified as {classified_bucket}"
    elif name in null_dropped:
        trace.reason_code = "NULL_FILTER_DROPPED"
        trace.detail = "Listed in metadata no_null_dropped_columns"
    elif name in final_plan and name not in parquet_cols:
        trace.reason_code = "NOT_IN_PARQUET_AFTER_PLAN"
        trace.detail = "In final transformation plan but column not in parquet"
    elif name not in raw_plan:
        trace.reason_code = "NOT_IN_TRANSFORM_PLAN"
        trace.detail = "Not in describe-plan even before build excludes"
    elif name in raw_plan and name not in after_build_exclude_plan:
        trace.reason_code = "CONFIG_BUILD_POLICY_PRUNE"
        trace.detail = "Removed by build_pipeline_features_transformation_config exclude_features prune"
    elif name in after_build_exclude_plan and name not in after_interval_plan:
        trace.reason_code = "CONFIG_INTERVAL_PRUNE"
        trace.detail = "Removed by prune_transformation_config_for_interval"
    elif name in after_interval_plan and name not in after_analysis_prune_plan:
        trace.reason_code = "CONFIG_ANALYSIS_POLICY_PRUNE"
        trace.detail = "Removed by analysis_dataset_export policy prune (operand/output skip)"
    elif name in final_plan and name in parquet_cols:
        trace.reason_code = "PRESENT_OK"
        trace.detail = "Present in dataset"
    else:
        trace.reason_code = "NOT_IN_TRANSFORM_PLAN"
        trace.detail = "Not emitted by transformation plan"

    trace.reason_group = _map_reason_group(trace.reason_code)
    return trace


def _diagnose_experimental_candidate(
    name: str,
    *,
    data_dir: str,
    interval_sec: float,
    pipeline_id: str,
    parquet_cols: set[str],
    null_dropped: set[str],
    experimental_plan: set[str],
    merged_plan: set[str],
    candidate_catalogue: set[str],
    classified_bucket: str = "",
) -> FeatureTrace:
    trace = FeatureTrace(
        name=name,
        feature_id=_feature_id(name),
        transformation=_transformation_label(name),
        horizon_sec=_horizon_label(name),
        interval_sec=float(interval_sec),
        pipeline_id=pipeline_id,
        definition_generated=name in experimental_plan or name in candidate_catalogue,
        in_transform_plan=name in merged_plan,
        in_parquet=name in parquet_cols,
        null_filter_dropped=name in null_dropped,
        classified_bucket=classified_bucket,
    )

    if name in parquet_cols:
        if classified_bucket == "other_pipeline":
            trace.reason_code = "PRESENT_OK"
            trace.detail = "Present in dataset"
        else:
            trace.reason_code = "CLASSIFICATION_MISMATCH"
            trace.detail = f"Present but classified as {classified_bucket}"
    elif name in null_dropped:
        trace.reason_code = "NULL_FILTER_DROPPED"
    elif name in merged_plan:
        trace.reason_code = "NOT_IN_PARQUET_AFTER_PLAN"
        trace.detail = "In merged transform plan but not in parquet"
    elif name in experimental_plan:
        trace.reason_code = "CONFIG_ANALYSIS_POLICY_PRUNE"
        trace.detail = "In experimental config plan but removed by merged/policy prune"
    elif name in candidate_catalogue:
        trace.reason_code = "NOT_IN_TRANSFORM_PLAN"
        trace.detail = "Listed as pipeline candidate but not in experimental transformation plan outputs"
    else:
        trace.reason_code = "NOT_REQUESTED"
        trace.detail = "Not in pipeline candidate catalogue snapshot"

    trace.reason_group = _map_reason_group(trace.reason_code)
    return trace


def _load_dataset_context(
    data_dir: str,
    dataset_name: str,
) -> tuple[dict[str, Any], set[str], str]:
    from .writer import datasets_dir

    json_path = os.path.join(datasets_dir(data_dir), f"{dataset_name}.json")
    meta: dict[str, Any] = {}
    if os.path.isfile(json_path):
        with open(json_path, encoding="utf-8") as fh:
            meta = json.load(fh)
    parquet_cols: set[str] = set()
    pq_rel = str(meta.get("output_parquet") or "").strip()
    pq_path = pq_rel if os.path.isabs(pq_rel) else os.path.join(data_dir, pq_rel)
    if pq_path and os.path.isfile(pq_path):
        try:
            import pyarrow.parquet as pq

            parquet_cols = set(pq.read_schema(pq_path).names)
        except Exception:
            pass
    pipeline_id = str(meta.get("pipeline_id") or "").strip().upper()
    if not pipeline_id:
        prov = meta.get("pipeline_provenance") or {}
        if isinstance(prov, dict):
            pipeline_id = str(prov.get("pipeline_id") or "").strip().upper()
    return meta, parquet_cols, pipeline_id


def _classify_columns(
    cols: set[str],
    *,
    data_dir: str,
    meta: dict[str, Any],
) -> dict[str, str]:
    from .feature_sources_catalog import (
        classify_dataset_feature_source,
        dataset_base_pipeline_export_feature_names,
        dataset_registry_export_feature_names,
    )

    registry = dataset_registry_export_feature_names(meta, data_dir=data_dir)
    base = dataset_base_pipeline_export_feature_names(meta, data_dir=data_dir)
    out: dict[str, str] = {}
    for col in cols:
        out[col] = classify_dataset_feature_source(
            col,
            data_dir=data_dir,
            registry_names=registry,
            base_pipeline_names=base,
        )
    return out


def diagnose_feature_sources(
    data_dir: str,
    *,
    dataset_name: str | None = None,
    pipeline_id: str | None = None,
    intervals: tuple[float, ...] = (3.0, 6.0, 9.0),
) -> DiagnosticReport:
    from .feature_sources_catalog import (
        base_pipeline_feature_names,
        other_pipeline_feature_names_from_metadata,
    )
    from .pipeline_features_config import prune_pipeline_transformation_config
    from .pipeline_features_prefs import (
        load_pipeline_output_prune_features,
        load_transformation_forbidden_features,
    )
    from .transformations.config import prune_transformation_config_for_interval

    report = DiagnosticReport(data_dir=data_dir, dataset_name=dataset_name or "")

    meta: dict[str, Any] = {}
    parquet_cols: set[str] = set()
    null_dropped: set[str] = set()
    interval_used = 0.0

    if dataset_name:
        meta, parquet_cols, meta_pid = _load_dataset_context(data_dir, dataset_name)
        report.dataset_name = dataset_name
        null_dropped = {
            str(c).strip()
            for c in (meta.get("no_null_dropped_columns") or [])
            if str(c).strip()
        }
        interval_used = float(meta.get("sample_interval_sec") or meta.get("sampling", {}).get("interval_sec") or 0)
        if not pipeline_id:
            pipeline_id = meta_pid

    if pipeline_id:
        report.pipeline_id = str(pipeline_id).strip().upper()

    classification = _classify_columns(parquet_cols, data_dir=data_dir, meta=meta) if parquet_cols else {}

    for interval_sec in intervals:
        catalogue = base_pipeline_feature_names(data_dir)
        raw_cfg = _build_base_config(data_dir, interval_sec, apply_build_exclude=False)
        build_cfg = _build_base_config(data_dir, interval_sec, apply_build_exclude=True)
        raw_plan = _expected_outputs(raw_cfg)
        after_build_plan = _expected_outputs(build_cfg)
        after_interval_cfg = prune_transformation_config_for_interval(build_cfg, interval_sec)
        after_interval_plan = _expected_outputs(after_interval_cfg)

        exclude = load_pipeline_output_prune_features(data_dir)
        forbidden = load_transformation_forbidden_features(data_dir)
        after_analysis_cfg = prune_pipeline_transformation_config(
            after_interval_cfg,
            exclude,
            interaction_operand_skip=forbidden,
            source_exclude=forbidden,
        )
        after_analysis_plan = _expected_outputs(after_analysis_cfg)

        merged_plan: set[str] = after_analysis_plan
        experimental_catalogue: set[str] = set()
        experimental_plan: set[str] = set()

        if report.pipeline_id:
            merged_cfg, _prov, _ = _build_analysis_config(data_dir, interval_sec, report.pipeline_id)
            if merged_cfg:
                merged_plan = _expected_outputs(merged_cfg)
            prov_meta = meta if meta else {"pipeline_provenance": _prov}
            experimental_catalogue = other_pipeline_feature_names_from_metadata(prov_meta)
            if isinstance(_prov, dict) and isinstance(_prov.get("transformation_config"), dict):
                experimental_plan = _expected_outputs(_prov.get("transformation_config"))

        base_in_parquet = {
            n for n in catalogue if n in parquet_cols
        } if parquet_cols and interval_sec == interval_used else set()

        summary = IntervalSummary(
            interval_sec=float(interval_sec),
            base_catalogue=len(catalogue),
            base_in_plan=len(catalogue & merged_plan),
            base_in_parquet=len(base_in_parquet),
            base_missing=len(catalogue) - (
                len(base_in_parquet) if parquet_cols and interval_sec == interval_used
                else len(catalogue & merged_plan)
            ),
            other_catalogue=len(experimental_catalogue),
            other_in_plan=len(experimental_catalogue & merged_plan),
            other_in_parquet=len(
                {n for n in experimental_catalogue if n in parquet_cols}
            ) if parquet_cols and interval_sec == interval_used else 0,
            other_missing=len(experimental_catalogue) - (
                len({n for n in experimental_catalogue if n in parquet_cols})
                if parquet_cols and interval_sec == interval_used
                else len(experimental_catalogue & merged_plan)
            ),
        )
        report.interval_summaries.append(summary)

        if dataset_name and interval_sec == interval_used:
            for name in sorted(catalogue):
                if name in parquet_cols:
                    continue
                trace = _diagnose_base_feature(
                    name,
                    data_dir=data_dir,
                    interval_sec=interval_sec,
                    parquet_cols=parquet_cols,
                    null_dropped=null_dropped,
                    final_plan=merged_plan,
                    raw_plan=raw_plan,
                    after_build_exclude_plan=after_build_plan,
                    after_interval_plan=after_interval_plan,
                    after_analysis_prune_plan=after_analysis_plan,
                    classified_bucket=classification.get(name, ""),
                )
                report.base_missing.append(trace)

            for name in sorted(experimental_catalogue):
                if name in parquet_cols:
                    continue
                trace = _diagnose_experimental_candidate(
                    name,
                    data_dir=data_dir,
                    interval_sec=interval_sec,
                    pipeline_id=report.pipeline_id,
                    parquet_cols=parquet_cols,
                    null_dropped=null_dropped,
                    experimental_plan=experimental_plan,
                    merged_plan=merged_plan,
                    candidate_catalogue=experimental_catalogue,
                    classified_bucket=classification.get(name, ""),
                )
                report.other_missing.append(trace)

    report.base_reason_counts = _count_groups(report.base_missing)
    report.other_reason_counts = _count_groups(report.other_missing)

    skip_n = len(load_pipeline_output_prune_features(data_dir))
    build_n = len(_expected_outputs(_build_base_config(data_dir, 6.0, apply_build_exclude=True)))
    raw_n = len(_expected_outputs(_build_base_config(data_dir, 6.0, apply_build_exclude=False)))
    report.config_notes.append(
        f"build_pipeline_features_transformation_config with exclude_features: "
        f"{build_n} planned outputs (raw without exclude: {raw_n}); skip set size={skip_n}"
    )
    report.config_notes.append(
        "build_pipeline_features_transformation_config uses interaction_operand_skip "
        "(retired features) for interaction operands; exclude_features prunes outputs only."
    )
    return report


def _count_groups(traces: list[FeatureTrace]) -> dict[str, int]:
    out: dict[str, int] = {g: 0 for g in REASON_GROUPS}
    for t in traces:
        out[t.reason_group] = out.get(t.reason_group, 0) + 1
    return {k: v for k, v in out.items() if v}


def format_diagnostic_report(report: DiagnosticReport) -> str:
    lines: list[str] = []
    lines.append("Feature Source Diagnostic Report")
    lines.append(f"data_dir: {report.data_dir}")
    if report.dataset_name:
        lines.append(f"dataset: {report.dataset_name}")
    if report.pipeline_id:
        lines.append(f"pipeline_id: {report.pipeline_id}")
    lines.append("")
    lines.append("Interval summary (catalogue vs transform plan vs parquet when dataset loaded):")
    lines.append(
        "Interval | Base Cat | Base Plan | Base Parquet | Base Missing | "
        "Other Cat | Other Plan | Other Parquet | Other Missing"
    )
    for s in report.interval_summaries:
        lines.append(
            f"{int(s.interval_sec)}s | {s.base_catalogue} | {s.base_in_plan} | "
            f"{s.base_in_parquet} | {s.base_missing} | "
            f"{s.other_catalogue} | {s.other_in_plan} | {s.other_in_parquet} | {s.other_missing}"
        )
    lines.append("")
    if report.config_notes:
        lines.append("Config notes:")
        for note in report.config_notes:
            lines.append(f"  - {note}")
        lines.append("")
    if report.base_reason_counts:
        lines.append("Base missing — reason groups:")
        for k, v in sorted(report.base_reason_counts.items()):
            lines.append(f"  {k}: {v}")
        lines.append("")
    if report.other_reason_counts:
        lines.append("Experimental missing — reason groups:")
        for k, v in sorted(report.other_reason_counts.items()):
            lines.append(f"  {k}: {v}")
        lines.append("")
    if report.base_missing:
        lines.append("Base missing detail (first 40):")
        lines.append(
            "feature | id | transform | horizon | reason_group | reason_code | detail"
        )
        for t in report.base_missing[:40]:
            lines.append(
                f"{t.name} | {t.feature_id} | {t.transformation} | {t.horizon_sec} | "
                f"{t.reason_group} | {t.reason_code} | {t.detail}"
            )
        if len(report.base_missing) > 40:
            lines.append(f"  ... {len(report.base_missing) - 40} more")
        lines.append("")
    if report.other_missing:
        lines.append("Experimental missing detail (first 40):")
        lines.append(
            "candidate | id | transform | horizon | reason_group | reason_code | detail"
        )
        for t in report.other_missing[:40]:
            lines.append(
                f"{t.name} | {t.feature_id} | {t.transformation} | {t.horizon_sec} | "
                f"{t.reason_group} | {t.reason_code} | {t.detail}"
            )
        if len(report.other_missing) > 40:
            lines.append(f"  ... {len(report.other_missing) - 40} more")
    return "\n".join(lines)


__all__ = [
    "DiagnosticReport",
    "FeatureTrace",
    "IntervalSummary",
    "diagnose_feature_sources",
    "format_diagnostic_report",
]

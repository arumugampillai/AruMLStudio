"""Load dataset-level build configuration for Day Metadata → Build Info tab."""

from __future__ import annotations

from typing import Any

from chain_replay_ml.dataset_builder.build_summary import build_summary_labels
from chain_replay_ml.dataset_builder.expected_spec import format_sampling_interval_label
from chain_replay_ml.dataset_builder.gap_policy import GAP_POLICY_VERSION


def _as_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _horizons_sec(targets_raw: dict[str, Any]) -> list[int]:
    raw = targets_raw.get("horizons_sec") or targets_raw.get("horizonsSec") or []
    out: list[int] = []
    for h in raw:
        try:
            out.append(int(h))
        except (TypeError, ValueError):
            continue
    return out


def load_master_build_info(store: Any) -> dict[str, Any]:
    """Read persisted master DB metadata into display-friendly build info."""
    build_summary = _as_dict(store.get_meta("build_summary"))
    master_config = _as_dict(store.get_meta("master_config"))
    build_schema = _as_dict(store.get_meta("build_schema"))
    dataset_configuration = _as_dict(store.get_meta("dataset_configuration"))
    feature_policy = _as_dict(store.get_meta("feature_policy"))
    meta_row = store.read_master_meta_dict() if hasattr(store, "read_master_meta_dict") else {}

    interval = float(
        build_summary.get("sampling_interval_sec")
        or master_config.get("sampling_interval_sec")
        or meta_row.get("sampling_interval_sec")
        or 0
    )
    stride = float(
        build_summary.get("sliding_stride_sec")
        or master_config.get("sliding_stride_sec")
        or interval
    )
    strike_raw = _as_dict(
        master_config.get("strike_selection") or build_summary.get("strike_selection")
    )
    gap_raw = _as_dict(master_config.get("gap_policy") or build_summary.get("gap_policy"))
    targets_raw = _as_dict(
        master_config.get("prediction_targets") or build_summary.get("prediction_targets")
    )
    horizons = _horizons_sec(targets_raw)

    labels = build_summary_labels(
        sampling_interval_sec=interval,
        strike_selection=strike_raw,
        gap_policy=gap_raw,
        prediction_targets={"horizonsSec": horizons},
    )
    sampling_label = (
        str(build_summary.get("sampling_label") or "")
        or labels["sampling"]
        or format_sampling_interval_label(interval)
        or (f"{interval:g}s" if interval else "—")
    )
    stride_label = str(build_summary.get("sliding_stride_label") or f"{stride:g}s")
    feature_window = float(
        build_summary.get("feature_window_sec")
        or dataset_configuration.get("feature_window_sec")
        or interval
    )

    feature_names = list(
        build_schema.get("feature_columns") or build_summary.get("feature_names") or []
    )
    target_columns = list(build_schema.get("target_columns") or [])

    lb = _as_dict(dataset_configuration.get("lookback_policy"))
    lb_method = str(lb.get("method") or "—").replace("_", " ").title()

    groups_raw = feature_policy.get("feature_groups") or dataset_configuration.get("feature_groups")
    if isinstance(groups_raw, list):
        groups_text = ", ".join(str(g) for g in groups_raw if str(g).strip()) or "—"
    else:
        groups_text = "—"

    profile = (
        feature_policy.get("feature_profile")
        or dataset_configuration.get("feature_profile")
        or "—"
    )

    kv_fields = {
        "market": str(master_config.get("market") or meta_row.get("market") or "—"),
        "sampling": sampling_label,
        "sliding_stride": stride_label,
        "feature_window": f"{feature_window:g} sec",
        "strike_selection": labels["strike"],
        "gap_policy": labels["gap"],
        "target_labels": labels["targets"],
        "feature_count": f"{len(feature_names):,}",
        "target_count": f"{len(target_columns):,}",
        "atm_band": str(master_config.get("atm_band") or "—"),
        "lookback_policy": lb_method,
        "feature_profile": str(profile),
        "feature_groups": groups_text,
        "dataset_version": str(
            meta_row.get("builder_version") or meta_row.get("dataset_version") or "—"
        ),
        "registry_version": str(
            feature_policy.get("registry_version")
            or meta_row.get("feature_registry_version")
            or "—"
        ),
        "feature_engine_version": str(feature_policy.get("engine_version") or "—"),
        "gap_policy_version": str(
            gap_raw.get("version") or GAP_POLICY_VERSION or "—"
        ),
        "builder_version": str(meta_row.get("builder_version") or "—"),
        "schema_hash": str(meta_row.get("schema_hash") or "—")[:16],
        "created_at": str(meta_row.get("created_at") or meta_row.get("updated_at") or "—")[:19],
        "updated_at": str(meta_row.get("updated_at") or "—")[:19],
    }

    return {
        "kv_fields": kv_fields,
        "feature_names": feature_names,
        "target_columns": target_columns,
    }


def format_feature_names_text(info: dict[str, Any]) -> str:
    names = list(info.get("feature_names") or [])
    targets = list(info.get("target_columns") or [])
    lines = [f"Features ({len(names)})", "—" * 32]
    if names:
        lines.extend(names)
    else:
        lines.append("No feature list in build metadata.")
    if targets:
        lines.extend(["", f"Target columns ({len(targets)})", "—" * 32])
        lines.extend(targets)
    return "\n".join(lines)

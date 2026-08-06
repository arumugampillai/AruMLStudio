"""Lookback policy — shared by dataset builder, validator, and audit."""

from __future__ import annotations

from typing import Any

from .feature_plugins import horizon_label

# Canonical policy method ids (persisted in metadata.lookback_policy)
POLICY_NEAREST_SNAPSHOT = "nearest_snapshot"
POLICY_EXACT_TIMESTAMP = "exact_timestamp"

# Legacy aliases stored in older datasets / configs
_LEGACY_NEAREST = frozenset({"nearest_snapshot", "nearest_greek_snapshot"})
_LEGACY_EXACT = frozenset({"exact_timestamp", "exact"})

DEFAULT_LOOKBACK_POLICY: dict[str, Any] = {
    "method": POLICY_NEAREST_SNAPSHOT,
    "label": "Nearest Snapshot",
    "tolerance_min_pct": 50,
    "tolerance_max_pct": 150,
}

DEFAULT_EXACT_TOLERANCE_SEC = 15.0


def lookback_policy_identity_material() -> dict[str, Any]:
    """Stable material for lookback_policy_hash."""
    return {
        "default": DEFAULT_LOOKBACK_POLICY,
        "exact_tolerance_sec": DEFAULT_EXACT_TOLERANCE_SEC,
        "methods": [POLICY_NEAREST_SNAPSHOT, POLICY_EXACT_TIMESTAMP],
    }


def lookback_policy_hash() -> str:
    import hashlib
    import json

    blob = json.dumps(lookback_policy_identity_material(), sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:8].upper()


def normalize_policy_method(method: str | None) -> str:
    """Map legacy / variant ids to canonical policy method."""
    m = str(method or "").strip().lower()
    if m in _LEGACY_NEAREST:
        return POLICY_NEAREST_SNAPSHOT
    if m in _LEGACY_EXACT:
        return POLICY_EXACT_TIMESTAMP
    if m == POLICY_NEAREST_SNAPSHOT:
        return POLICY_NEAREST_SNAPSHOT
    if m == POLICY_EXACT_TIMESTAMP:
        return POLICY_EXACT_TIMESTAMP
    return POLICY_NEAREST_SNAPSHOT


def normalize_policy_doc(policy: dict[str, Any] | None) -> dict[str, Any]:
    """Return policy dict with canonical method and defaults filled."""
    base = dict(DEFAULT_LOOKBACK_POLICY)
    if policy:
        base.update(policy)
    base["method"] = normalize_policy_method(base.get("method"))
    if base["method"] == POLICY_NEAREST_SNAPSHOT:
        base.setdefault("label", "Nearest Snapshot")
        base.setdefault("tolerance_min_pct", 50)
        base.setdefault("tolerance_max_pct", 150)
    else:
        base.setdefault("label", "Exact Timestamp")
    return base


def policy_label(policy: dict[str, Any]) -> str:
    """Human-readable policy label for UI."""
    pol = normalize_policy_doc(policy)
    if pol["method"] == POLICY_NEAREST_SNAPSHOT:
        lo = int(pol.get("tolerance_min_pct") or 50)
        hi = int(pol.get("tolerance_max_pct") or 150)
        return f"{pol.get('label') or 'Nearest Snapshot'} ({lo}–{hi}%)"
    return str(pol.get("label") or "Exact Timestamp")


def build_dataset_configuration(
    *,
    sampling: dict[str, Any],
    prediction_targets: dict[str, Any] | list[int] | None = None,
    horizons_sec: list[int] | None = None,
    lookback_policy_method: str | None = None,
    gap_max_sec: float | None = None,
) -> dict[str, Any]:
    """Canonical dataset configuration persisted in metadata and expected spec."""
    if horizons_sec is None:
        if isinstance(prediction_targets, dict):
            horizons_sec = [int(h) for h in (prediction_targets.get("horizonsSec") or [])]
        else:
            horizons_sec = []
    interval = int(
        sampling.get("trainingIntervalSec")
        or sampling.get("interval_sec")
        or 10
    )
    method = normalize_policy_method(lookback_policy_method or POLICY_NEAREST_SNAPSHOT)
    policy = normalize_policy_doc({**DEFAULT_LOOKBACK_POLICY, "method": method})
    out = {
        "sampling_interval_sec": interval,
        "feature_grid_step_sec": interval,
        "lookback_policy": policy,
        "future_targets_sec": horizons_sec,
        "future_targets_labels": [horizon_label(h) for h in horizons_sec],
        "prediction_horizon_5m_sec": 300,
        "prediction_horizon_5m_label": "5 minutes",
    }
    try:
        from chain_replay_ml.feature_policy import (
            build_dataset_policy_manifest,
            load_feature_policy_registry,
        )
        from chain_replay_ml.feature_policy.types import DEFAULT_GAP_MAX_SEC

        reg = load_feature_policy_registry()
        resolved_gap = float(gap_max_sec if gap_max_sec is not None else DEFAULT_GAP_MAX_SEC)
        out["feature_policy"] = build_dataset_policy_manifest(
            reg,
            sampling_interval_sec=float(interval),
            gap_max_sec=resolved_gap,
        )
        out["gap_policy"] = {"gapMaxSec": resolved_gap}
        from chain_replay_ml.feature_policy.build_readiness import build_feature_readiness_manifest

        out["feature_readiness"] = build_feature_readiness_manifest(gap_max_sec=resolved_gap)
    except Exception:
        pass
    return out


def read_dataset_configuration(
    meta_doc: dict[str, Any] | None,
    expected_doc: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Load configuration from metadata / expected spec, with legacy inference."""
    meta_doc = meta_doc or {}
    expected_doc = expected_doc or {}

    config: dict[str, Any] = {}
    if meta_doc.get("dataset_configuration"):
        config = dict(meta_doc["dataset_configuration"])
    elif expected_doc.get("dataset_configuration"):
        config = dict(expected_doc["dataset_configuration"])

    if not config:
        sampling = meta_doc.get("sampling") or expected_doc.get("sampling") or {}
        horizons_sec: list[int] = []
        for col in meta_doc.get("prediction_target_columns") or expected_doc.get("prediction_target_columns") or []:
            from .audit_diagnostics import _horizon_sec_from_column
            h = _horizon_sec_from_column(str(col))
            if h > 0:
                horizons_sec.append(h)
        if not horizons_sec:
            for lbl in meta_doc.get("prediction_targets") or expected_doc.get("prediction_targets") or []:
                from .audit_diagnostics import _horizon_sec_from_column
                h = _horizon_sec_from_column(str(lbl))
                if h > 0:
                    horizons_sec.append(h)
        horizons_sec = sorted(set(horizons_sec))
        top_method = meta_doc.get("lookback_policy") or expected_doc.get("lookback_policy")
        config = build_dataset_configuration(
            sampling=sampling,
            horizons_sec=horizons_sec,
            lookback_policy_method=top_method,
        )
    else:
        # Merge top-level lookback_policy string if present
        top_method = meta_doc.get("lookback_policy") or expected_doc.get("lookback_policy")
        if top_method:
            pol = normalize_policy_doc(config.get("lookback_policy"))
            pol["method"] = normalize_policy_method(top_method)
            config["lookback_policy"] = pol

    pol = normalize_policy_doc(config.get("lookback_policy"))
    config["lookback_policy"] = pol
    return config


def lookback_policy(config: dict[str, Any]) -> dict[str, Any]:
    return normalize_policy_doc((config or {}).get("lookback_policy"))


def read_builder_policy(
    meta_doc: dict[str, Any] | None,
    expected_doc: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Policy recorded when the dataset was built (source of truth)."""
    config = read_dataset_configuration(meta_doc, expected_doc)
    return lookback_policy(config)


def read_validator_policy(
    meta_doc: dict[str, Any] | None,
    expected_doc: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Policy the validator must use — always the dataset build policy."""
    return read_builder_policy(meta_doc, expected_doc)


def policies_match(
    builder_policy: dict[str, Any],
    validator_policy: dict[str, Any],
) -> bool:
    return normalize_policy_method(builder_policy.get("method")) == normalize_policy_method(
        validator_policy.get("method")
    )


def detect_configuration_mismatch(
    meta_doc: dict[str, Any] | None,
    expected_doc: dict[str, Any] | None = None,
    *,
    validator_policy: dict[str, Any] | None = None,
) -> tuple[bool, str | None]:
    """
    Return (mismatch, reason).
    Mismatch when metadata sources disagree or validator policy != builder policy.
    """
    meta_doc = meta_doc or {}
    expected_doc = expected_doc or {}
    builder = read_builder_policy(meta_doc, expected_doc)
    validator = normalize_policy_doc(validator_policy) if validator_policy else read_validator_policy(
        meta_doc, expected_doc
    )

    if not policies_match(builder, validator):
        return True, (
            f"Validator policy ({policy_label(validator)}) differs from "
            f"builder policy ({policy_label(builder)})."
        )

    top = meta_doc.get("lookback_policy") or expected_doc.get("lookback_policy")
    nested_method = (meta_doc.get("dataset_configuration") or expected_doc.get("dataset_configuration") or {}).get(
        "lookback_policy", {}
    )
    if isinstance(nested_method, dict):
        nested_method = nested_method.get("method")
    if top and nested_method and normalize_policy_method(top) != normalize_policy_method(nested_method):
        return True, (
            f"Metadata lookback_policy ({top}) disagrees with "
            f"dataset_configuration ({nested_method})."
        )

    return False, None


def greek_snapshot_at(
    snapshots: list[tuple[float, dict[str, float]]],
    ts: float,
    lookback_sec: float,
    key: str,
) -> tuple[float | None, float | None]:
    """Nearest snapshot within 50–150% of lookback window (builder default)."""
    best_ts: float | None = None
    best_val: float | None = None
    best_dt = lookback_sec + 1.0
    for t, g in snapshots:
        dt = ts - t
        if lookback_sec * 0.5 <= dt <= lookback_sec * 1.5 and dt < best_dt:
            best_dt = dt
            best_ts = t
            val = g.get(key)
            best_val = float(val) if val is not None else None
    return best_ts, best_val


def greek_at_exact_timestamp(
    snapshots: list[tuple[float, dict[str, float]]],
    ts: float,
    lookback_sec: float,
    key: str,
    *,
    tolerance_sec: float = DEFAULT_EXACT_TOLERANCE_SEC,
) -> tuple[float | None, float | None]:
    """Greek at the snapshot closest to ts − lookback (within tolerance)."""
    target = ts - lookback_sec
    best_ts: float | None = None
    best_val: float | None = None
    best_gap = tolerance_sec + 1.0
    for t, g in snapshots:
        gap = abs(t - target)
        if gap <= tolerance_sec and gap < best_gap:
            best_gap = gap
            best_ts = t
            val = g.get(key)
            best_val = float(val) if val is not None else None
    return best_ts, best_val


def greek_at_lookback(
    snapshots: list[tuple[float, dict[str, float]]],
    ts: float,
    lookback_sec: float,
    key: str,
    policy: dict[str, Any] | None,
) -> float | None:
    """Resolve past greek using the configured lookback policy."""
    pol = normalize_policy_doc(policy)
    if pol["method"] == POLICY_EXACT_TIMESTAMP:
        _snap_ts, val = greek_at_exact_timestamp(snapshots, ts, lookback_sec, key)
        return val
    _snap_ts, val = greek_snapshot_at(snapshots, ts, lookback_sec, key)
    return val


def greek_at_lookback_with_ts(
    snapshots: list[tuple[float, dict[str, float]]],
    ts: float,
    lookback_sec: float,
    key: str,
    policy: dict[str, Any] | None,
) -> tuple[float | None, float | None]:
    """Return (snapshot_ts, greek_value) for diagnostics."""
    pol = normalize_policy_doc(policy)
    if pol["method"] == POLICY_EXACT_TIMESTAMP:
        return greek_at_exact_timestamp(snapshots, ts, lookback_sec, key)
    return greek_snapshot_at(snapshots, ts, lookback_sec, key)


def policy_alignment_view(
    meta_doc: dict[str, Any] | None,
    expected_doc: dict[str, Any] | None = None,
    *,
    validator_policy: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Builder vs validator policy block for audit / investigation UI."""
    builder = read_builder_policy(meta_doc, expected_doc)
    validator = normalize_policy_doc(validator_policy) if validator_policy else read_validator_policy(
        meta_doc, expected_doc
    )
    mismatch, reason = detect_configuration_mismatch(
        meta_doc, expected_doc, validator_policy=validator,
    )
    suggested_fix = None
    if mismatch:
        suggested_fix = (
            "Align validator with the dataset build policy: "
            f"use {policy_label(builder)} as recorded in metadata."
        )
    return {
        "builder_policy": policy_label(builder),
        "builder_policy_id": builder.get("method"),
        "validator_policy": policy_label(validator),
        "validator_policy_id": validator.get("method"),
        "policies_match": not mismatch,
        "configuration_mismatch": mismatch,
        "mismatch_reason": reason,
        "suggested_fix": suggested_fix,
    }


def dataset_specification_view(
    config: dict[str, Any],
    *,
    feature: str | None = None,
) -> dict[str, Any]:
    """Human-readable specification block for RCA UI."""
    policy = lookback_policy(config)
    horizons = config.get("future_targets_labels") or config.get("future_targets_sec") or []
    tol_lo = int(policy.get("tolerance_min_pct") or 50)
    tol_hi = int(policy.get("tolerance_max_pct") or 150)
    return {
        "prediction_horizon": config.get("prediction_horizon_5m_label") or "5 minutes",
        "prediction_horizon_sec": int(config.get("prediction_horizon_5m_sec") or 300),
        "configured_method": policy_label(policy),
        "configured_method_id": policy.get("method"),
        "tolerance_label": f"{tol_lo}–{tol_hi}%" if policy["method"] == POLICY_NEAREST_SNAPSHOT else "—",
        "sampling_interval_sec": int(config.get("sampling_interval_sec") or 10),
        "feature_grid_step_sec": int(
            config.get("feature_grid_step_sec") or config.get("sampling_interval_sec") or 10
        ),
        "future_targets": list(horizons),
        "lookback_policy": policy,
    }

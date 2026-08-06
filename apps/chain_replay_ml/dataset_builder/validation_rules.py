"""Validation rules loader — audit thresholds independent of schema registry."""

from __future__ import annotations

import hashlib
import json
import os
from typing import Any

from path_config import CHART_DATA_ROOT as _CHART_DIR
_RULES_PATH = os.path.join(_CHART_DIR, "static", "ml_validation_rules.json")

_RULES_CACHE: dict[str, Any] | None = None

# Embedded fallback (matches ml_validation_rules.json v1.4.3)
_FALLBACK: dict[str, Any] = {
    "version": "1.4.3",
    "confidence_weights": {
        "integrity": {"label": "Dataset Integrity", "weight_pct": 25},
        "formula_validation": {"label": "Formula Validation", "weight_pct": 25},
        "replay_validation": {"label": "Replay Validation", "weight_pct": 20},
        "feature_coverage": {"label": "Feature Coverage", "weight_pct": 15},
        "audit": {"label": "Audit", "weight_pct": 10},
        "spec_match": {"label": "Specification Match", "weight_pct": 5},
    },
    "independent_checks": [
        {"feature": "future_ltp_3s", "horizon_sec": 3, "kind": "future_ltp"},
        {"feature": "future_ltp_5s", "horizon_sec": 5, "kind": "future_ltp"},
        {"feature": "future_ltp_10s", "horizon_sec": 10, "kind": "future_ltp"},
        {"feature": "future_ltp_30s", "horizon_sec": 30, "kind": "future_ltp"},
        {"feature": "ltp_change_5m", "lookback_sec": 300, "kind": "ltp_change"},
        {"feature": "oi_change_5m", "lookback_sec": 300, "kind": "oi_change"},
        {"feature": "delta_change_5m", "lookback_sec": 300, "kind": "delta_change"},
    ],
    "columns": {
        "delta": {"distribution": {"enabled": True, "expected_min": -1.05, "expected_max": 1.05}},
        "gamma": {"distribution": {"enabled": True, "expected_min": 0.0, "expected_max": 0.05}},
        "theta": {"distribution": {"enabled": True, "expected_min": -50.0, "expected_max": 0.5}},
        "current_iv": {"distribution": {"enabled": True, "expected_min": 0.0, "expected_max": 300.0}},
        "oi": {"distribution": {"enabled": True, "expected_min": 0.0, "expected_max": None}},
        "chain_pcr": {
            "distribution": {
                "enabled": True, "expected_min": 0.0, "expected_max": 50.0, "alt_column": "chain_pcr_vol",
            },
        },
        "atm_straddle": {"distribution": {"enabled": True, "expected_min": 0.0, "expected_max": None}},
    },
}


def validation_rules_path() -> str:
    return _RULES_PATH


def load_validation_rules(path: str | None = None, *, use_cache: bool = True) -> dict[str, Any]:
    global _RULES_CACHE
    rules_path = path or _RULES_PATH
    if use_cache and path is None and _RULES_CACHE is not None:
        return _RULES_CACHE
    try:
        with open(rules_path, encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError):
        data = dict(_FALLBACK)
    if path is None and use_cache:
        _RULES_CACHE = data
    return data


def validation_rules_identity_material(rules: dict[str, Any] | None = None) -> dict[str, Any]:
    reg = rules or load_validation_rules()
    return {
        "version": reg.get("version"),
        "confidence_weights": reg.get("confidence_weights"),
        "replay_validation": reg.get("replay_validation"),
        "correlation_tests": reg.get("correlation_tests"),
        "independent_checks": reg.get("independent_checks"),
        "columns": reg.get("columns"),
    }


def validation_rules_hash(rules: dict[str, Any] | None = None) -> str:
    blob = json.dumps(validation_rules_identity_material(rules), sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:8].upper()


def confidence_weights(rules: dict[str, Any] | None = None) -> dict[str, tuple[str, int]]:
    reg = rules or load_validation_rules()
    raw = reg.get("confidence_weights") or _FALLBACK["confidence_weights"]
    out: dict[str, tuple[str, int]] = {}
    for key, block in raw.items():
        if isinstance(block, dict):
            out[key] = (str(block.get("label") or key), int(block.get("weight_pct") or 0))
    return out


def distribution_features(rules: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """Distribution audit specs — compatible with audit_extended.audit_feature_distributions."""
    from .schema_registry import column_display_name

    reg = rules or load_validation_rules()
    cols = reg.get("columns") or {}
    out: list[dict[str, Any]] = []
    for column, block in cols.items():
        dist = (block or {}).get("distribution") or {}
        if not dist.get("enabled"):
            continue
        alt = dist.get("alt_column")
        out.append({
            "column": column,
            "label": column_display_name(column),
            "min": dist.get("expected_min"),
            "max": dist.get("expected_max"),
            **({"alt": alt} if alt else {}),
        })
    return out


def independent_checks(rules: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    reg = rules or load_validation_rules()
    return list(reg.get("independent_checks") or _FALLBACK["independent_checks"])


def correlation_test_specs(rules: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    reg = rules or load_validation_rules()
    return list(reg.get("correlation_tests") or [])


def validation_identity_material(rules: dict[str, Any] | None = None) -> dict[str, Any]:
    """Stable material hashed into validation pipeline stage."""
    return {
        "independent_checks": independent_checks(rules),
        "distribution_features": distribution_features(rules),
        "correlation_tests": correlation_test_specs(rules),
    }

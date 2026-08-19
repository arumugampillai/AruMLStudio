"""Authoritative Regime Registry Store (Phase 4C.3).

This module manages the persistent JSON catalog `regime_registry_store.json`, storing
regime definitions, detection specifications, required features, parent/child hierarchy,
and immutable version history.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from datetime import datetime, timezone
from typing import Any

from .enums import (
    BASELINE_REGIME_CATALOG,
    DEFAULT_REGIME_ID,
    DEFAULT_REGIME_NAME,
    RegimeScope,
)
from .specs import RegimeSpec

_REGIME_ID_RE = re.compile(r"^R\d{3}(?:_[A-Z0-9_]+)?$", re.IGNORECASE)
_VALID_SCOPES = frozenset({RegimeScope.ALL_REGIMES.value, RegimeScope.SPECIALIZED.value, RegimeScope.DISCOVERED.value})
_VALID_STATUSES = frozenset({"ACTIVE", "RETIRED"})
_VALID_DETECTION_TYPES = frozenset({"UNIVERSAL", "RULE_BASED", "ML_CLASSIFIER", "MANUAL"})


def validate_regime_id_format(regime_id: str) -> bool:
    """Validate that regime_id follows standard format (e.g. 'R000', 'R001', 'R008_VOL_EXP')."""
    return bool(_REGIME_ID_RE.match(str(regime_id or "").strip()))


def regime_registry_path(data_dir: str) -> str:
    """Return the absolute path to `regime_registry_store.json`."""
    return os.path.join(data_dir, "regime_registry_store.json")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def compute_regime_definition_hash(
    *,
    detection_type: str,
    detection_spec: dict[str, Any] | None,
    required_features: list[str] | None,
    parent_regime_id: str | None = None,
) -> str:
    """Compute deterministic canonical SHA-256 hash of a regime definition."""
    payload = {
        "detection_type": str(detection_type or "RULE_BASED").upper().strip(),
        "detection_spec": detection_spec or {},
        "required_features": sorted(list(set(required_features or []))),
        "parent_regime_id": str(parent_regime_id).upper().strip() if parent_regime_id else None,
    }
    canonical_bytes = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(canonical_bytes).hexdigest()


def _build_baseline_regime_record(
    regime_id: str,
    *,
    name: str,
    family: str,
    description: str,
    scope: str,
    parent_regime_id: str | None,
    detection_type: str,
    detection_spec: dict[str, Any],
    required_features: list[str],
    created_at: str = "2026-01-01T00:00:00Z",
) -> dict[str, Any]:
    def_hash = compute_regime_definition_hash(
        detection_type=detection_type,
        detection_spec=detection_spec,
        required_features=required_features,
        parent_regime_id=parent_regime_id,
    )
    return {
        "regime_id": regime_id,
        "regime_name": name,
        "display_name": name.replace("_", " ").title(),
        "family": family,
        "description": description,
        "scope": scope,
        "status": "ACTIVE",
        "parent_regime_id": parent_regime_id,
        "current_version": 1,
        "definition_hash": def_hash,
        "detection_type": detection_type,
        "detection_spec": detection_spec,
        "required_features": required_features,
        "created_at": created_at,
        "updated_at": created_at,
        "version_history": [
            {
                "version": 1,
                "definition_hash": def_hash,
                "detection_type": detection_type,
                "detection_spec": detection_spec,
                "required_features": required_features,
                "parent_regime_id": parent_regime_id,
                "created_at": created_at,
            }
        ],
    }


def _default_baseline_store() -> dict[str, Any]:
    now = _utc_now_iso()
    regimes: dict[str, Any] = {}

    # R000: Universal Root
    regimes["R000"] = _build_baseline_regime_record(
        "R000",
        name="ALL_REGIMES",
        family="UNIVERSAL",
        description="Universal unconditioned market scope covering all market history.",
        scope=RegimeScope.ALL_REGIMES.value,
        parent_regime_id=None,
        detection_type="UNIVERSAL",
        detection_spec={},
        required_features=[],
    )

    # R001: Trend
    regimes["R001"] = _build_baseline_regime_record(
        "R001",
        name="TREND",
        family="DIRECTIONAL_MOMENTUM",
        description="Strong directional momentum with sustained order flow imbalance.",
        scope=RegimeScope.SPECIALIZED.value,
        parent_regime_id="R000",
        detection_type="RULE_BASED",
        detection_spec={"primary_indicator": "adx_14", "condition": "gte", "threshold": 25.0},
        required_features=["spot", "adx_14"],
    )

    # R002: Sideways
    regimes["R002"] = _build_baseline_regime_record(
        "R002",
        name="SIDEWAYS",
        family="MEAN_REVERSION",
        description="Range-bound, mean-reverting oscillator with compressed ADX.",
        scope=RegimeScope.SPECIALIZED.value,
        parent_regime_id="R000",
        detection_type="RULE_BASED",
        detection_spec={"primary_indicator": "adx_14", "condition": "lt", "threshold": 20.0},
        required_features=["spot", "adx_14"],
    )

    # R003: High Volatility
    regimes["R003"] = _build_baseline_regime_record(
        "R003",
        name="HIGH_VOLATILITY",
        family="VOLATILITY_EXPANSION",
        description="Realized IV > 85th percentile, wide spreads, and elevated strike variance.",
        scope=RegimeScope.SPECIALIZED.value,
        parent_regime_id="R000",
        detection_type="RULE_BASED",
        detection_spec={"primary_indicator": "atm_iv_pctile", "condition": "gte", "threshold": 80.0},
        required_features=["spot", "atm_iv_pctile"],
    )

    # R004: Low Volatility
    regimes["R004"] = _build_baseline_regime_record(
        "R004",
        name="LOW_VOLATILITY",
        family="VOLATILITY_COMPRESSION",
        description="Realized IV < 25th percentile with compressed straddle premiums.",
        scope=RegimeScope.SPECIALIZED.value,
        parent_regime_id="R000",
        detection_type="RULE_BASED",
        detection_spec={"primary_indicator": "atm_iv_pctile", "condition": "lte", "threshold": 25.0},
        required_features=["spot", "atm_iv_pctile"],
    )

    # R005: Breakout
    regimes["R005"] = _build_baseline_regime_record(
        "R005",
        name="BREAKOUT",
        family="LIQUIDITY_EXPANSION",
        description="Volatility compression breakout with volume and order velocity spike.",
        scope=RegimeScope.SPECIALIZED.value,
        parent_regime_id="R000",
        detection_type="RULE_BASED",
        detection_spec={"primary_indicator": "volume_zscore", "condition": "gte", "threshold": 2.5},
        required_features=["spot", "volume_zscore"],
    )

    # R006: Reversal
    regimes["R006"] = _build_baseline_regime_record(
        "R006",
        name="REVERSAL",
        family="MICROSTRUCTURE_REVERSAL",
        description="Exhaustion divergence at key structural liquidity support/resistance.",
        scope=RegimeScope.SPECIALIZED.value,
        parent_regime_id="R000",
        detection_type="RULE_BASED",
        detection_spec={"primary_indicator": "rsi_divergence", "condition": "eq", "threshold": 1.0},
        required_features=["spot", "rsi_divergence"],
    )

    # R007: Expiry Pinning
    regimes["R007"] = _build_baseline_regime_record(
        "R007",
        name="EXPIRY_PINNING",
        family="OPTION_GAMMA",
        description="Gamma pinning and decay compression near max-pain strike on expiry.",
        scope=RegimeScope.SPECIALIZED.value,
        parent_regime_id="R000",
        detection_type="RULE_BASED",
        detection_spec={"primary_indicator": "dte_minutes", "condition": "lte", "threshold": 120.0},
        required_features=["spot", "dte_minutes", "max_pain_strike"],
    )

    return {
        "schema_version": "1.0",
        "updated_at": now,
        "default_regime_id": DEFAULT_REGIME_ID,
        "regimes": regimes,
    }


def load_regime_registry(data_dir: str) -> dict[str, Any]:
    """Load the regime registry store, initializing with baseline R000-R007 if missing."""
    path = regime_registry_path(data_dir)
    if not os.path.isfile(path):
        store = _default_baseline_store()
        save_regime_registry(data_dir, store)
        return store

    try:
        with open(path, "r", encoding="utf-8") as fh:
            doc = json.load(fh)
        if not isinstance(doc, dict) or "regimes" not in doc:
            store = _default_baseline_store()
            save_regime_registry(data_dir, store)
            return store
        return doc
    except (OSError, json.JSONDecodeError):
        store = _default_baseline_store()
        save_regime_registry(data_dir, store)
        return store


def save_regime_registry(data_dir: str, store: dict[str, Any]) -> None:
    """Atomically write the regime registry store to disk."""
    path = regime_registry_path(data_dir)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    store["updated_at"] = _utc_now_iso()

    tmp_dir = os.path.dirname(os.path.abspath(path))
    with tempfile.NamedTemporaryFile("w", dir=tmp_dir, delete=False, encoding="utf-8") as tf:
        json.dump(store, tf, indent=2)
        tmp_name = tf.name

    os.replace(tmp_name, path)


def get_regime_record(data_dir: str, regime_id: str, *, version: int | None = None) -> dict[str, Any] | None:
    """Look up a regime record by ID and optional historical version number."""
    store = load_regime_registry(data_dir)
    regimes = store.get("regimes", {})
    rid = str(regime_id or "").strip().upper()
    rec = regimes.get(rid)
    if not rec:
        return None

    if version is None or version == rec.get("current_version"):
        return dict(rec)

    for h in rec.get("version_history", []):
        if h.get("version") == version:
            out = dict(rec)
            out.update(h)
            return out

    return None


def list_regimes(
    data_dir: str,
    *,
    include_retired: bool = False,
    parent_id: str | None = None,
    scope: str | None = None,
) -> list[dict[str, Any]]:
    """List all registered regimes with optional filtering."""
    store = load_regime_registry(data_dir)
    regimes = store.get("regimes", {})
    out: list[dict[str, Any]] = []

    for rid, rec in sorted(regimes.items()):
        if not include_retired and rec.get("status") == "RETIRED":
            continue
        if parent_id is not None and str(rec.get("parent_regime_id") or "").upper() != str(parent_id).upper():
            continue
        if scope is not None and str(rec.get("scope") or "").upper() != str(scope).upper():
            continue
        out.append(dict(rec))

    return out


def register_regime(
    data_dir: str,
    *,
    regime_id: str,
    regime_name: str,
    display_name: str | None = None,
    family: str = "CUSTOM",
    description: str = "",
    scope: str = RegimeScope.SPECIALIZED.value,
    parent_regime_id: str | None = DEFAULT_REGIME_ID,
    detection_type: str = "RULE_BASED",
    detection_spec: dict[str, Any] | None = None,
    required_features: list[str] | None = None,
) -> dict[str, Any]:
    """Register a new regime into the authoritative store."""
    rid = str(regime_id or "").strip().upper()
    if not validate_regime_id_format(rid):
        raise ValueError(f"Invalid regime_id format: '{regime_id}'. Must match 'Rxxx' pattern.")

    store = load_regime_registry(data_dir)
    if rid in store.get("regimes", {}):
        raise ValueError(f"Regime ID '{rid}' is already registered. Use update_regime_definition to create a new version.")

    p_id = str(parent_regime_id).upper().strip() if parent_regime_id else None
    if p_id and p_id not in store.get("regimes", {}):
        raise ValueError(f"Parent regime '{p_id}' does not exist.")

    rname = str(regime_name or "").strip().upper()
    dname = display_name or rname.replace("_", " ").title()
    spec = detection_spec or {}
    req_f = list(required_features or [])
    now = _utc_now_iso()

    rec = _build_baseline_regime_record(
        rid,
        name=rname,
        family=family,
        description=description,
        scope=scope if scope in _VALID_SCOPES else RegimeScope.SPECIALIZED.value,
        parent_regime_id=p_id,
        detection_type=detection_type if detection_type in _VALID_DETECTION_TYPES else "RULE_BASED",
        detection_spec=spec,
        required_features=req_f,
        created_at=now,
    )

    store.setdefault("regimes", {})[rid] = rec
    save_regime_registry(data_dir, store)
    return rec


def update_regime_definition(
    data_dir: str,
    regime_id: str,
    *,
    detection_spec: dict[str, Any] | None = None,
    required_features: list[str] | None = None,
    detection_type: str | None = None,
    parent_regime_id: str | None = None,
    description: str | None = None,
    display_name: str | None = None,
) -> dict[str, Any]:
    """Update a regime definition.
    
    If mathematical definition (spec/features/type/parent) changes, increments version
    and preserves previous version in version_history.
    """
    rid = str(regime_id or "").strip().upper()
    store = load_regime_registry(data_dir)
    regimes = store.get("regimes", {})
    if rid not in regimes:
        raise KeyError(f"Regime '{rid}' not found.")

    rec = regimes[rid]
    new_type = str(detection_type or rec["detection_type"]).upper().strip()
    new_spec = detection_spec if detection_spec is not None else rec["detection_spec"]
    new_features = list(required_features) if required_features is not None else rec["required_features"]
    new_parent = parent_regime_id if parent_regime_id is not None else rec.get("parent_regime_id")
    if new_parent:
        new_parent = str(new_parent).upper().strip()

    new_hash = compute_regime_definition_hash(
        detection_type=new_type,
        detection_spec=new_spec,
        required_features=new_features,
        parent_regime_id=new_parent,
    )

    now = _utc_now_iso()
    if description is not None:
        rec["description"] = str(description)
    if display_name is not None:
        rec["display_name"] = str(display_name)

    if new_hash != rec.get("definition_hash"):
        # Definition changed -> Increment version and archive
        new_ver = int(rec.get("current_version") or 1) + 1
        rec["current_version"] = new_ver
        rec["definition_hash"] = new_hash
        rec["detection_type"] = new_type
        rec["detection_spec"] = new_spec
        rec["required_features"] = new_features
        rec["parent_regime_id"] = new_parent
        rec["updated_at"] = now

        history_entry = {
            "version": new_ver,
            "definition_hash": new_hash,
            "detection_type": new_type,
            "detection_spec": new_spec,
            "required_features": new_features,
            "parent_regime_id": new_parent,
            "created_at": now,
        }
        rec.setdefault("version_history", []).append(history_entry)

    save_regime_registry(data_dir, store)
    return rec


def retire_regime(data_dir: str, regime_id: str) -> dict[str, Any]:
    """Retire a regime without deleting its historical records."""
    rid = str(regime_id or "").strip().upper()
    if rid == DEFAULT_REGIME_ID:
        raise ValueError("Cannot retire universal root regime R000.")

    store = load_regime_registry(data_dir)
    regimes = store.get("regimes", {})
    if rid not in regimes:
        raise KeyError(f"Regime '{rid}' not found.")

    rec = regimes[rid]
    rec["status"] = "RETIRED"
    rec["updated_at"] = _utc_now_iso()
    save_regime_registry(data_dir, store)
    return rec


def reactivate_regime(data_dir: str, regime_id: str) -> dict[str, Any]:
    """Reactivate a previously retired regime."""
    rid = str(regime_id or "").strip().upper()
    store = load_regime_registry(data_dir)
    regimes = store.get("regimes", {})
    if rid not in regimes:
        raise KeyError(f"Regime '{rid}' not found.")

    rec = regimes[rid]
    rec["status"] = "ACTIVE"
    rec["updated_at"] = _utc_now_iso()
    save_regime_registry(data_dir, store)
    return rec

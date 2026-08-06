"""Strategy config schema — entry/exit/risk/execution rules."""

from __future__ import annotations

import copy
import re
from typing import Any

LIFECYCLE_LABELS = {
    "new_strategy": "New Strategy",
    "clone": "Clone",
    "edit": "Edit",
    "calibration": "Calibration",
}

_DEFAULT_CONFIG: dict[str, Any] = {
    "entry": {
        "direction": "long",
        "min_confidence": 0.0,
        "premium_min": 15.0,
        "premium_max": 30.0,
        "atm_band": 15,
        "expiry": "current",
        "entry_cadence_sec": 3,
        # 0 = V1 behavior (direction only). >0 also requires predicted move size.
        "minimum_predicted_move_pct": 0.0,
        # True = require predicted_ltp direction (+ optional min move). False = skip
        # regression gates; entry relies on premium/ATM/option/classifier filters.
        "use_regression": True,
        "option_types": ["CE", "PE"],
    },
    "exit": {
        "mode": "target_stop_hold",
    },
    "stop": {
        "stop_loss_pct": 5.0,
    },
    "target": {
        "target_profit_pct": 8.0,
        # false = % target from entry; true = entry-row predicted_ltp as target price.
        "use_predicted_ltp": False,
    },
    "hold_time": {
        "max_hold_sec": 30,
    },
    "confidence": {
        "min_signal_strength": 0.0,
        "use_model_confidence": False,
    },
    "position_size": {
        "lots": 1,
        "qty_per_lot": 65,
    },
    "execution": {
        "fees_mode": "rupee_charges",
        "slippage_ticks": 0,
        "allow_averaging": False,
    },
}


def default_strategy_template(*, name: str = "OTM Premium Buyer") -> dict[str, Any]:
    doc = copy.deepcopy(_DEFAULT_CONFIG)
    doc["name"] = name
    doc["description"] = (
        "Current expiry, OTM ±15, premium ₹15–30, 3s entry cadence, "
        "30s max hold, 8% target / 5% stop."
    )
    return normalize_strategy_config(doc)


def safe_strategy_slug(name: str) -> str:
    slug = re.sub(r"[^\w\-]+", "_", str(name or "").strip().lower())
    slug = re.sub(r"_+", "_", slug).strip("_")
    return slug[:48] or "strategy"


def normalize_strategy_config(doc: dict[str, Any] | None) -> dict[str, Any]:
    src = dict(doc or {})
    out = copy.deepcopy(_DEFAULT_CONFIG)
    for section in ("entry", "exit", "stop", "target", "hold_time", "confidence", "position_size", "execution"):
        if section in src and isinstance(src[section], dict):
            out[section].update(src[section])
    if src.get("name"):
        out["name"] = str(src["name"]).strip()
    if src.get("description") is not None:
        out["description"] = str(src["description"])
    entry = out["entry"]
    entry["premium_min"] = float(entry.get("premium_min") or 0)
    entry["premium_max"] = float(entry.get("premium_max") or 0)
    entry["atm_band"] = int(entry.get("atm_band") or 0)
    entry["entry_cadence_sec"] = int(entry.get("entry_cadence_sec") or 0)
    try:
        entry["minimum_predicted_move_pct"] = float(
            entry.get("minimum_predicted_move_pct")
            if entry.get("minimum_predicted_move_pct") is not None
            else 0.0
        )
    except (TypeError, ValueError):
        entry["minimum_predicted_move_pct"] = 0.0
    entry["use_regression"] = bool(entry.get("use_regression", True))
    out["stop"]["stop_loss_pct"] = float(out["stop"].get("stop_loss_pct") or 0)
    out["target"]["target_profit_pct"] = float(out["target"].get("target_profit_pct") or 0)
    out["target"]["use_predicted_ltp"] = bool(out["target"].get("use_predicted_ltp"))
    out["hold_time"]["max_hold_sec"] = int(out["hold_time"].get("max_hold_sec") or 0)
    out["confidence"]["min_signal_strength"] = float(out["confidence"].get("min_signal_strength") or 0)
    out["position_size"]["lots"] = int(out["position_size"].get("lots") or 1)
    out["position_size"]["qty_per_lot"] = int(out["position_size"].get("qty_per_lot") or 65)
    out["execution"]["slippage_ticks"] = int(out["execution"].get("slippage_ticks") or 0)
    return out


def validate_strategy_config(doc: dict[str, Any]) -> list[str]:
    cfg = normalize_strategy_config(doc)
    errors: list[str] = []
    if not str(cfg.get("name") or "").strip():
        errors.append("name is required")
    entry = cfg["entry"]
    if entry["premium_min"] > entry["premium_max"]:
        errors.append("entry.premium_min must be <= entry.premium_max")
    if entry["atm_band"] < 0:
        errors.append("entry.atm_band must be >= 0")
    if entry["entry_cadence_sec"] < 1:
        errors.append("entry.entry_cadence_sec must be >= 1")
    if float(entry.get("minimum_predicted_move_pct") or 0) < 0:
        errors.append("entry.minimum_predicted_move_pct must be >= 0")
    if cfg["hold_time"]["max_hold_sec"] < 1:
        errors.append("hold_time.max_hold_sec must be >= 1")
    if cfg["stop"]["stop_loss_pct"] <= 0:
        errors.append("stop.stop_loss_pct must be > 0")
    if cfg["target"]["target_profit_pct"] <= 0:
        errors.append("target.target_profit_pct must be > 0")
    if cfg["position_size"]["lots"] < 1:
        errors.append("position_size.lots must be >= 1")
    return errors

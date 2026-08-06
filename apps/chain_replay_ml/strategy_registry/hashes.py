"""Stable hashes for immutable strategy versions."""

from __future__ import annotations

import hashlib
import json
from typing import Any


def _hash_obj(obj: Any) -> str:
    raw = json.dumps(obj, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def strategy_config_hash(config: dict[str, Any]) -> str:
    """Hash normalized trading rules only (excludes display metadata)."""
    payload = {
        "entry": config.get("entry"),
        "exit": config.get("exit"),
        "stop": config.get("stop"),
        "target": config.get("target"),
        "hold_time": config.get("hold_time"),
        "confidence": config.get("confidence"),
        "position_size": config.get("position_size"),
        "execution": config.get("execution"),
    }
    return _hash_obj(payload)

"""Post-training Feature Studio configuration (Milestone 3)."""

from __future__ import annotations

import os
from typing import Any

STAGE_KEYS: tuple[str, ...] = ("importance", "distribution", "drift")

_DEFAULTS: dict[str, Any] = {
    "enabled": True,
    "importance": True,
    "distribution": True,
    "drift": True,
}


def _as_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    raw = str(value).strip().lower()
    if raw in ("1", "true", "yes", "on", "enabled"):
        return True
    if raw in ("0", "false", "no", "off", "disabled"):
        return False
    return default


def env_master_enabled() -> bool:
    """Hard kill switch. Default on. ``ARUMLSTUDIO_POST_TRAINING=off`` (or ``ARUNEO_POST_TRAINING=off``) skips all stages."""
    raw = str(
        os.getenv("ARUMLSTUDIO_POST_TRAINING")
        or os.getenv("ARUNEO_POST_TRAINING")
        or "on"
    ).strip().lower()
    return raw not in ("off", "0", "false", "no", "disabled")


def normalize_post_training_config(raw: Any = None) -> dict[str, bool]:
    """Normalize UI/API JSON into a flat boolean config.

    Accepted shapes::

        {"enabled": true, "importance": true, ...}
        {"enabled": true, "stages": {"importance": true, ...}}
        true / false  → master enabled only (stages default on)
    """
    out = dict(_DEFAULTS)
    if raw is None:
        return out
    if isinstance(raw, bool):
        out["enabled"] = raw
        return out
    if not isinstance(raw, dict):
        return out

    if "enabled" in raw:
        out["enabled"] = _as_bool(raw.get("enabled"), True)
    stages = raw.get("stages")
    stage_src = stages if isinstance(stages, dict) else raw
    for key in STAGE_KEYS:
        if key in stage_src:
            out[key] = _as_bool(stage_src.get(key), True)
        # camelCase aliases from Tk persisted drafts
        camel = {"importance": "Importance", "distribution": "Distribution", "drift": "Drift"}[key]
        alt = f"run{camel}"
        if alt in raw:
            out[key] = _as_bool(raw.get(alt), True)
    return out


def resolve_post_training_config(raw: Any = None) -> dict[str, Any]:
    """Apply env kill switch on top of normalized config.

    Returns dict with boolean flags plus ``env_disabled`` / ``active_stages``.
    """
    cfg = normalize_post_training_config(raw)
    env_ok = env_master_enabled()
    enabled = bool(cfg["enabled"]) and env_ok
    active = [s for s in STAGE_KEYS if enabled and cfg.get(s, True)]
    return {
        "enabled": enabled,
        "importance": bool(cfg["importance"]),
        "distribution": bool(cfg["distribution"]),
        "drift": bool(cfg["drift"]),
        "env_disabled": not env_ok,
        "active_stages": active,
    }

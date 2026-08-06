"""Gap policy — rolling-feature reset threshold for dataset builds."""

from __future__ import annotations

from typing import Any

from chain_replay_ml.feature_policy.types import DEFAULT_GAP_MAX_SEC

GAP_POLICY_VERSION = 1
# Selectable thresholds in Create Dataset → Build configuration → Gap policy.
GAP_PRESET_SEC: tuple[int, ...] = (20, 60, 90, 120, 180)
# Older saved prefs may still use these; normalize accepts them.
_LEGACY_PRESET_SEC: tuple[int, ...] = (5, 10, 30)
_ALL_PRESET_SEC: tuple[int, ...] = tuple(
    sorted({*GAP_PRESET_SEC, *_LEGACY_PRESET_SEC})
)


def default_gap_policy() -> dict[str, Any]:
    return {
        "configVersion": GAP_POLICY_VERSION,
        "enabled": True,
        "preset": "20",
        "gapMaxSec": float(DEFAULT_GAP_MAX_SEC),
        "applied": True,
    }


def normalize_gap_policy(raw: dict[str, Any] | None) -> dict[str, Any]:
    cfg = default_gap_policy()
    if not isinstance(raw, dict):
        return cfg

    if "enabled" in raw:
        cfg["enabled"] = bool(raw.get("enabled"))
    elif "enableGapReset" in raw:
        cfg["enabled"] = bool(raw.get("enableGapReset"))

    preset = str(raw.get("preset") or "").strip().lower()
    if preset == "custom":
        try:
            custom = float(raw.get("gapMaxSec") or DEFAULT_GAP_MAX_SEC)
        except (TypeError, ValueError):
            custom = float(DEFAULT_GAP_MAX_SEC)
        cfg["preset"] = "custom"
        cfg["gapMaxSec"] = max(1.0, min(custom, 600.0))
    elif preset.isdigit() and int(preset) in _ALL_PRESET_SEC:
        sec = int(preset)
        cfg["preset"] = str(sec)
        cfg["gapMaxSec"] = float(sec)
    else:
        try:
            sec = float(raw.get("gapMaxSec") or DEFAULT_GAP_MAX_SEC)
        except (TypeError, ValueError):
            sec = float(DEFAULT_GAP_MAX_SEC)
        nearest = min(_ALL_PRESET_SEC, key=lambda p: abs(p - sec))
        if abs(nearest - sec) < 0.001:
            cfg["preset"] = str(nearest)
            cfg["gapMaxSec"] = float(nearest)
        else:
            cfg["preset"] = "custom"
            cfg["gapMaxSec"] = max(1.0, min(sec, 600.0))
    cfg["applied"] = True
    return cfg


def gap_max_sec_from_policy(policy: dict[str, Any] | None) -> float:
    """Effective gap threshold for the build. 0 disables gap resets."""
    doc = normalize_gap_policy(policy)
    if not bool(doc.get("enabled", True)):
        return 0.0
    return float(doc.get("gapMaxSec") or DEFAULT_GAP_MAX_SEC)


def gap_summary_label(policy: dict[str, Any] | None) -> str:
    doc = normalize_gap_policy(policy)
    if not bool(doc.get("enabled", True)):
        return "off (no gap reset)"
    sec = float(doc.get("gapMaxSec") or DEFAULT_GAP_MAX_SEC)
    if str(doc.get("preset")) == "custom":
        return f"{sec:g}s (custom)"
    return f"{sec:g}s"

"""Strike selection — parity with web Create Dataset UI."""

from __future__ import annotations

from typing import Any

from chain_replay_ml.dataset_builder.master_defaults import default_master_strike_selection

STRIKE_CONFIG_VERSION = 3
MASTER_DATASET_ATM_BAND = 15
CUSTOM_OFFSETS = list(range(-10, 11))

ATM_BAND_OPTIONS: list[tuple[str, str]] = [
    ("0", "ATM Only"),
    ("1", "±1"),
    ("2", "±2"),
    ("3", "±3"),
    ("5", "±5"),
    ("7", "±7"),
    ("10", "±10"),
    ("15", "±15"),
    ("all", "All Strikes"),
]


def default_strike_config() -> dict[str, Any]:
    cfg = dict(default_master_strike_selection())
    cfg.update({
        "configVersion": STRIKE_CONFIG_VERSION,
        "mode": "atm_band",
        "atmBand": MASTER_DATASET_ATM_BAND,
        "premiumMin": 15,
        "premiumMax": 30,
        "premiumIgnoreOutside": False,
        "deltaType": "absolute",
        "deltaMin": 0.15,
        "deltaMax": 0.50,
        "customOffsets": [-3, -2, -1, 0, 1, 2, 3],
        "applied": True,
    })
    return cfg


def mode_label(mode: str) -> str:
    labels = {
        "premium_band": "Premium Band",
        "delta_range": "Delta Range",
        "custom": "Custom Strikes",
    }
    return labels.get(str(mode), "ATM Band")


def atm_band_count(band: Any) -> int | None:
    if band == "all":
        return None
    try:
        n = int(band)
    except (TypeError, ValueError):
        return 1
    if n <= 0:
        return 1
    return 2 * n + 1


def atm_band_summary_label(band: Any) -> str:
    if band == "all":
        return "All Strikes"
    try:
        n = int(band)
    except (TypeError, ValueError):
        return "—"
    if n == 0:
        return "ATM Only (1 Strike)"
    count = atm_band_count(n)
    return f"±{n} ({count} Strikes)" if count else f"±{n}"


def atm_band_hint_text(band: Any) -> str:
    if band == "all":
        return "All strikes in the chain are monitored · No band limit"
    try:
        n = int(band)
    except (TypeError, ValueError):
        return ""
    if n == 0:
        return "ATM only · Total monitored strikes: 1"
    total = 2 * n + 1
    return f"ATM ±{n} = {n} CE + ATM + {n} PE · Total monitored strikes: {total}"


def delta_summary_label(cfg: dict[str, Any]) -> str:
    dt = str(cfg.get("deltaType") or "absolute")
    try:
        lo = float(cfg.get("deltaMin") or 0)
        hi = float(cfg.get("deltaMax") or 0)
    except (TypeError, ValueError):
        return "—"
    type_lbl = "CE" if dt == "ce" else ("PE" if dt == "pe" else "|Δ|")
    return f"{type_lbl} {min(lo, hi):.2f}–{max(lo, hi):.2f}"


def delta_preview_rule(delta_type: str, d_min: float, d_max: float) -> str:
    lo = min(d_min, d_max)
    hi = max(d_min, d_max)
    if delta_type == "ce":
        return f"{lo:.2f} ≤ CE Delta ≤ {hi:.2f}"
    if delta_type == "pe":
        return f"{(-hi):.2f} ≤ PE Delta ≤ {(-lo):.2f}"
    return f"{lo:.2f} ≤ |Delta| ≤ {hi:.2f}"


def premium_filter_label(cfg: dict[str, Any]) -> str:
    if str(cfg.get("mode")) != "premium_band":
        return "Disabled"
    try:
        lo = float(cfg.get("premiumMin") or 0)
        hi = float(cfg.get("premiumMax") or 0)
        return f"₹{lo:g} – ₹{hi:g}"
    except (TypeError, ValueError):
        return "Enabled"


def strike_summary_label(cfg: dict[str, Any]) -> str:
    mode = str(cfg.get("mode") or "atm_band")
    if mode == "atm_band":
        return atm_band_summary_label(cfg.get("atmBand"))
    if mode == "delta_range":
        return delta_summary_label(cfg)
    if mode == "custom":
        offs = cfg.get("customOffsets") or []
        text = ", ".join(f"+{o}" if int(o) > 0 else str(o) for o in offs)
        return text or "—"
    if mode == "premium_band":
        return premium_filter_label(cfg)
    return mode_label(mode)


def normalize_strike_config(raw: dict[str, Any] | None) -> dict[str, Any]:
    cfg = default_strike_config()
    if not isinstance(raw, dict):
        return cfg
    if int(raw.get("configVersion") or 0) != STRIKE_CONFIG_VERSION:
        cfg.update({k: v for k, v in raw.items() if k in cfg})
        cfg["configVersion"] = STRIKE_CONFIG_VERSION
        return cfg
    cfg.update(raw)
    band = cfg.get("atmBand")
    if band is not None and band != "all":
        try:
            cfg["atmBand"] = int(band)
        except (TypeError, ValueError):
            cfg["atmBand"] = MASTER_DATASET_ATM_BAND
    offsets = cfg.get("customOffsets")
    if not isinstance(offsets, list):
        cfg["customOffsets"] = default_strike_config()["customOffsets"]
    else:
        cfg["customOffsets"] = sorted({int(o) for o in offsets})
    if str(cfg.get("mode")) != "premium_band":
        cfg["premiumIgnoreOutside"] = False
    else:
        cfg["premiumIgnoreOutside"] = True
    cfg["applied"] = True
    return cfg


def strike_selection_for_master(applied: dict[str, Any] | None = None) -> dict[str, Any]:
    """Normalize strike selection for master SQLite insert (respects selected ATM band)."""
    cfg = normalize_strike_config(applied)
    cfg["applied"] = True
    return cfg


def atm_band_from_strike_config(cfg: dict[str, Any] | None) -> int:
    """Resolve ATM band integer used for master row estimates and builds."""
    doc = normalize_strike_config(cfg)
    band = doc.get("atmBand")
    if band == "all":
        return MASTER_DATASET_ATM_BAND
    try:
        return max(0, int(band))
    except (TypeError, ValueError):
        return MASTER_DATASET_ATM_BAND

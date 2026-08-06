"""Auto-generated model names — Future_LTP_5m_TSS_100f_XGB_HHMM_D (IST)."""

from __future__ import annotations

from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

_IST = ZoneInfo("Asia/Kolkata")

_ALGO_SHORT = {
    "xgboost": "XGB",
    "lightgbm": "LGB",
    "catboost": "CAT",
    "random_forest": "RF",
    "extra_trees": "ET",
    "linear": "LIN",
    "neural": "NN",
}

_VAL_SHORT = {
    "time_series_split": "TSS",
    "time_series": "TSS",
    "walk_forward": "WF",
    "rolling_window": "RW",
    "tss": "TSS",
    "wf": "WF",
    "rw": "RW",
}


def training_timestamp_slug(*, when: datetime | None = None) -> str:
    """Build time suffix HHMM (IST, 24-hour)."""
    dt = when or datetime.now(_IST)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=_IST)
    else:
        dt = dt.astimezone(_IST)
    return dt.strftime("%H%M")


def training_day_suffix(*, when: datetime | None = None) -> str:
    """Day of month in IST without zero padding (e.g. 7 for the 7th)."""
    dt = when or datetime.now(_IST)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=_IST)
    else:
        dt = dt.astimezone(_IST)
    return str(dt.day)


def training_name_stamp(*, when: datetime | None = None) -> str:
    """HHMM + day suffix, e.g. 0351_7."""
    return f"{training_timestamp_slug(when=when)}_{training_day_suffix(when=when)}"


def lifecycle_package_name(family_name: str, version_label: str) -> str:
    """Unique on-disk package folder for a lifecycle version within a model family."""
    from .paths import safe_model_name

    family = safe_model_name(str(family_name or "").strip())
    ver = str(version_label or "v2").strip().lower()
    if not ver.startswith("v"):
        ver = f"v{ver.lstrip('v')}"
    return safe_model_name(f"{family}__{ver}")


def target_horizon_short(target: str) -> str:
    t = str(target or "").strip()
    if t.startswith("future_ltp_"):
        return t[len("future_ltp_") :]
    if t.startswith("ormp_return_"):
        rest = t[len("ormp_return_") :]
        for suffix in ("_points", "_percent"):
            if rest.endswith(suffix):
                return rest[: -len(suffix)]
        return rest or "5m"
    if t.startswith("ormp_direction_"):
        return t[len("ormp_direction_") :] or "5m"
    return "5m"


def validation_strategy_short(strategy: str = "time_series", *, window_mode: str | None = None) -> str:
    """Map UI/backend validation strategy to TSS, WF, or RW."""
    key = str(strategy or "time_series").strip().lower()
    if key in _VAL_SHORT:
        return _VAL_SHORT[key]
    if key == "walk_forward" and str(window_mode or "").strip().lower() == "rolling":
        return "RW"
    return "TSS"


def feature_count_slug(feature_count: int | None) -> str:
    n = max(0, int(feature_count or 0))
    return f"{n}f"


def barrier_params_slug(params: dict[str, Any] | None) -> str:
    """Compact TP/SL segment for model names, e.g. ``tp_20_sl_10``."""
    raw = params if isinstance(params, dict) else {}

    def _fmt(value: Any, default: float) -> str:
        try:
            num = float(value if value is not None else default)
        except (TypeError, ValueError):
            num = float(default)
        if abs(num - round(num)) < 1e-9:
            return str(int(round(num)))
        text = f"{num:.4g}".replace(".", "p")
        return text

    tp = _fmt(raw.get("tp_value", raw.get("tp_points")), 20.0)
    sl = _fmt(raw.get("sl_value", raw.get("sl_points")), 10.0)
    return f"tp_{tp}_sl_{sl}"


def suggest_model_name(
    target: str,
    algorithm: str = "xgboost",
    *,
    validation_strategy: str = "time_series",
    window_mode: str | None = None,
    feature_count: int | None = None,
    when: datetime | None = None,
    label_strategy: str | None = None,
    label_strategy_params: dict[str, Any] | None = None,
) -> str:
    """Return e.g. Future_LTP_5m_TSS_100f_XGB_2150_7 or TB_tp_20_sl_10_…."""
    algo = str(algorithm or "xgboost").strip().lower()
    suffix = _ALGO_SHORT.get(algo) or algo.upper()[:4]
    val = validation_strategy_short(validation_strategy, window_mode=window_mode)
    feat = feature_count_slug(feature_count)
    hor = target_horizon_short(target)
    stamp = training_name_stamp(when=when)
    strat = str(label_strategy or "fixed_horizon").strip().lower()
    if strat == "triple_barrier":
        barriers = barrier_params_slug(label_strategy_params)
        return f"TB_{barriers}_{val}_{feat}_{suffix}_{stamp}"
    if str(target).startswith("future_ltp"):
        return f"Future_LTP_{hor}_{val}_{feat}_{suffix}_{stamp}"
    label = str(target or "Model").replace(" ", "_").replace("(", "").replace(")", "")
    return f"{label}_{val}_{feat}_{suffix}_{stamp}"


def suggest_model_name_from_split(
    target: str,
    algorithm: str = "xgboost",
    split: dict[str, Any] | None = None,
    *,
    feature_count: int | None = None,
    when: datetime | None = None,
    label_strategy: str | None = None,
    label_strategy_params: dict[str, Any] | None = None,
) -> str:
    """Suggest a model name using split.strategy and walk_forward.window_mode."""
    split_doc = split or {}
    wf = dict(split_doc.get("walk_forward") or {})
    ui_strategy = split_doc.get("validation_strategy_ui")
    strategy = ui_strategy or split_doc.get("strategy") or "time_series"
    return suggest_model_name(
        target,
        algorithm,
        validation_strategy=str(strategy),
        window_mode=wf.get("window_mode"),
        feature_count=feature_count,
        when=when,
        label_strategy=label_strategy,
        label_strategy_params=label_strategy_params,
    )

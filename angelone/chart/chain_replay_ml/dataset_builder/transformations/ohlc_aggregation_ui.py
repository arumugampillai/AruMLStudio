"""OHLC Aggregation UI helpers — timeframes, outputs, config, preview, validation."""

from __future__ import annotations

from typing import Any

from .ohlc_aggregation import (
    DEFAULT_OHLC_FIELDS,
    OHLC_FIELDS,
    normalize_ohlc_field,
    ohlc_aggregation_column_name,
)
from .ohlc_history_profiles import (
    available_ohlc_timeframes,
    get_ohlc_interval_profile,
    resolve_timeframe_spec,
    timeframe_specs_metadata,
)

# Re-export for panel imports.
__all__ = [
    "DEFAULT_OHLC_FIELDS",
    "DEFAULT_OHLC_TIMEFRAMES",
    "OHLC_FIELDS",
    "TF_DISPLAY_LABELS",
    "FIELD_DISPLAY_LABELS",
    "available_ohlc_timeframes",
    "build_ohlc_aggregation_config",
    "merge_ohlc_aggregation_into_config",
    "ohlc_aggregation_column_count",
    "preview_ohlc_aggregation_columns",
    "planned_ohlc_aggregation_columns",
    "validate_ohlc_aggregation_settings",
    "format_ohlc_aggregation_preview_text",
    "timeframe_display_label",
    "default_selected_ohlc_features",
]

# Default UI checklist when interval is 3s (legacy import name).
DEFAULT_OHLC_TIMEFRAMES: tuple[str, ...] = available_ohlc_timeframes(3)

TF_DISPLAY_LABELS: dict[str, str] = {
    "3m": "3m",
    "5m": "5m",
    "15m": "15m",
    "30m": "30m",
}

FIELD_DISPLAY_LABELS: dict[str, str] = {
    "open": "Open",
    "high": "High",
    "low": "Low",
    "close": "Close",
}

# Prefer price-path levels when seeding OHLC Input Features.
_DEFAULT_OHLC_FEATURE_PRIORITY: tuple[str, ...] = (
    "spot",
    "ltp",
    "underlying",
    "spot_ltp",
)


def default_selected_ohlc_features(available: list[str]) -> list[str]:
    """Default OHLC inputs: spot/ltp-like levels only (not greeks/volume)."""
    avail = [str(f).strip() for f in available if str(f).strip()]
    avail_set = set(avail)
    chosen: list[str] = []
    for name in _DEFAULT_OHLC_FEATURE_PRIORITY:
        if name in avail_set and name not in chosen:
            chosen.append(name)
    if chosen:
        return chosen
    # Fallback: first available master column if nothing matches.
    return avail[:1]


def timeframe_display_label(
    tf: str,
    *,
    sample_interval_sec: float | int | None = None,
) -> str:
    key = str(tf or "").strip().lower()
    base = TF_DISPLAY_LABELS.get(key, key)
    if sample_interval_sec is None:
        return base
    try:
        spec = resolve_timeframe_spec(sample_interval_sec, key)
        if spec.is_approximate:
            samples = spec.sample_count(sample_interval_sec)
            return (
                f"{base} (~{spec.actual_duration_sec}s×{samples}, "
                f"hist {spec.history})"
            )
        return f"{base} (hist {spec.history})"
    except Exception:
        return base


def planned_ohlc_aggregation_columns(
    *,
    features: list[str],
    timeframes: list[str],
    outputs: list[str],
    sample_interval_sec: float | int = 3,
) -> list[str]:
    cols: list[str] = []
    interval = float(sample_interval_sec)
    for feat in features:
        f = str(feat).strip()
        if not f:
            continue
        for tf_raw in timeframes:
            try:
                spec = resolve_timeframe_spec(interval, tf_raw)
            except Exception:
                continue
            for h in range(1, spec.history + 1):
                for out_raw in outputs:
                    try:
                        fld = normalize_ohlc_field(out_raw)
                    except Exception:
                        continue
                    cols.append(
                        ohlc_aggregation_column_name(
                            f,
                            spec.key,
                            h,
                            fld,
                            sample_interval_sec=interval,
                            history_len=spec.history,
                        )
                    )
    return cols


def ohlc_aggregation_column_count(
    *,
    enabled: bool,
    features: list[str],
    timeframes: list[str],
    outputs: list[str],
    sample_interval_sec: float | int = 3,
) -> int:
    if not enabled:
        return 0
    return len(
        planned_ohlc_aggregation_columns(
            features=features,
            timeframes=timeframes,
            outputs=outputs,
            sample_interval_sec=sample_interval_sec,
        )
    )


def build_ohlc_aggregation_config(
    *,
    enabled: bool,
    features: list[str],
    timeframes: list[str] | None = None,
    outputs: list[str] | None = None,
    partition_by: list[str] | None = None,
    sample_interval_sec: float | int | None = None,
    order: int = 44,
) -> dict[str, Any]:
    feats = [str(f).strip() for f in features if str(f).strip()]
    interval = float(sample_interval_sec) if sample_interval_sec is not None else 3.0
    default_tfs = list(available_ohlc_timeframes(interval))
    tfs: list[str] = []
    for tf in timeframes if timeframes is not None else default_tfs:
        try:
            key = resolve_timeframe_spec(interval, tf).key
        except Exception:
            continue
        if key not in tfs:
            tfs.append(key)
    fields: list[str] = []
    for fld in outputs if outputs is not None else list(OHLC_FIELDS):
        try:
            key = normalize_ohlc_field(fld)
        except Exception:
            continue
        if key not in fields:
            fields.append(key)
    if not fields:
        fields = list(DEFAULT_OHLC_FIELDS)
    entry: dict[str, Any] = {
        "id": "ohlc_aggregation",
        "enabled": bool(enabled) and bool(feats) and bool(tfs) and bool(fields),
        "order": int(order),
        "name": "OHLC Aggregation",
        "params": {
            "features": feats,
            "timeframes": tfs,
            "outputs": fields,
            "partition_by": list(partition_by or ["trading_day", "token"]),
        },
    }
    if sample_interval_sec is not None:
        try:
            entry["params"]["sample_interval_sec"] = float(sample_interval_sec)
        except (TypeError, ValueError):
            pass
    if tfs and sample_interval_sec is not None:
        try:
            entry["params"]["timeframe_specs"] = timeframe_specs_metadata(
                sample_interval_sec, tfs
            )
        except Exception:
            pass
    return {
        "transformation_pipeline_version": 1,
        "transformations": [entry] if entry["enabled"] else [],
    }


def merge_ohlc_aggregation_into_config(
    base: dict[str, Any] | None,
    *,
    enabled: bool,
    features: list[str],
    timeframes: list[str] | None = None,
    outputs: list[str] | None = None,
    partition_by: list[str] | None = None,
    sample_interval_sec: float | int | None = None,
) -> dict[str, Any]:
    from .config import normalize_transformation_config

    cfg = normalize_transformation_config(base)
    transforms = [
        t
        for t in (cfg.get("transformations") or [])
        if isinstance(t, dict) and str(t.get("id") or "") != "ohlc_aggregation"
    ]
    ohlc_cfg = build_ohlc_aggregation_config(
        enabled=enabled,
        features=features,
        timeframes=timeframes,
        outputs=outputs,
        partition_by=partition_by,
        sample_interval_sec=sample_interval_sec,
    )
    for entry in ohlc_cfg.get("transformations") or []:
        transforms.append(entry)
    cfg["transformations"] = transforms
    return cfg


def preview_ohlc_aggregation_columns(
    *,
    features: list[str],
    timeframes: list[str],
    outputs: list[str],
    sample_interval_sec: float | int = 3,
    limit: int = 12,
) -> list[str]:
    cols = planned_ohlc_aggregation_columns(
        features=features,
        timeframes=timeframes,
        outputs=outputs,
        sample_interval_sec=sample_interval_sec,
    )
    return cols[: max(0, int(limit))]


def validate_ohlc_aggregation_settings(
    *,
    enabled: bool,
    features: list[str],
    timeframes: list[str] | None = None,
    outputs: list[str] | None = None,
    available_features: list[str] | None = None,
    sample_interval_sec: float | int | None = None,
) -> str | None:
    if not enabled:
        return None
    feats = [str(f).strip() for f in features if str(f).strip()]
    if not feats:
        return "OHLC Aggregation is enabled but no features are selected."
    if sample_interval_sec is None:
        return "OHLC Aggregation requires sample_interval_sec from the dataset."
    try:
        interval = float(sample_interval_sec)
        profile = get_ohlc_interval_profile(interval)
    except Exception as exc:
        msg = str(exc).split("\n")[-1] if str(exc) else str(exc)
        return msg or "OHLC Aggregation sample interval has no history profile."
    tfs_in = (
        timeframes
        if timeframes is not None
        else list(profile.timeframe_keys())
    )
    tfs: list[str] = []
    for tf in tfs_in:
        try:
            tfs.append(resolve_timeframe_spec(interval, tf).key)
        except Exception as exc:
            return str(exc).split("\n")[-1] if str(exc) else f"Invalid timeframe {tf!r}."
    if not tfs:
        return "OHLC Aggregation is enabled but no timeframes are selected."
    outs_in = outputs if outputs is not None else list(OHLC_FIELDS)
    fields: list[str] = []
    for fld in outs_in:
        try:
            fields.append(normalize_ohlc_field(fld))
        except Exception as exc:
            return str(exc).split("\n")[-1] if str(exc) else f"Invalid output {fld!r}."
    if not fields:
        return "OHLC Aggregation is enabled but no outputs are selected."
    seen: set[str] = set()
    for name in planned_ohlc_aggregation_columns(
        features=feats,
        timeframes=tfs,
        outputs=fields,
        sample_interval_sec=interval,
    ):
        if name in seen:
            return f"Duplicate OHLC Aggregation output name: {name}"
        seen.add(name)
    if available_features is not None:
        avail = {str(a) for a in available_features}
        missing = [f for f in feats if f not in avail]
        if missing:
            return "OHLC Aggregation features not in Master:\n" + "\n".join(missing[:8])
    return None


def format_ohlc_aggregation_preview_text(
    *,
    enabled: bool,
    feature_count: int,
    timeframe_count: int,
    output_count: int,
    columns_to_add: int,
    sample_names: list[str] | None = None,
    sample_interval_sec: float | int | None = None,
) -> str:
    if not enabled:
        return "OHLC Aggregation disabled — no extra columns."
    lines = [
        f"Features    : {feature_count}",
        f"Timeframes  : {timeframe_count}",
        f"Outputs     : {output_count}",
        f"New Columns : {columns_to_add}",
    ]
    if sample_interval_sec is not None:
        try:
            profile = get_ohlc_interval_profile(sample_interval_sec)
            depths = ", ".join(
                f"{k}×{v.history}" for k, v in profile.timeframes.items()
            )
            lines.append(f"Profile @{int(sample_interval_sec)}s: {depths}")
        except Exception:
            lines.append(f"Interval   : {sample_interval_sec}s")
    if sample_names:
        lines.append("Examples    : " + ", ".join(sample_names[:4]))
    return "\n".join(lines)

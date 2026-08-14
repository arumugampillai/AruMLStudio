"""Lag transform UI helpers — categories, search, presets, preview, validation."""

from __future__ import annotations

from typing import Any

from .horizon_policy import (
    default_horizons_for_interval,
    warmup_seconds_for_interval,
)

# Backward-compatible default list = 3s policy horizons (not a hardcoded product list).
try:
    DEFAULT_LAG_SECONDS: tuple[int, ...] = default_horizons_for_interval(3)
except Exception:  # pragma: no cover - policy file missing during partial checkout
    DEFAULT_LAG_SECONDS = (30, 60, 90, 120, 180, 240, 300)

# Fallback only when interval policy / master meta unavailable.
DEFAULT_WARMUP_SECONDS: float = 900.0
BYTES_PER_CELL: int = 8  # float64 estimate for memory preview

SUGGESTED_LAG_FEATURES: tuple[str, ...] = (
    "current_ltp",
    "ltp",
    "spot",
    "current_iv",
    "oi",
    "volume",
)
META_SKIP_COLUMNS: frozenset[str] = frozenset({
    "trading_day",
    "timestamp",
    "token",
    "market",
    "expiry",
    "symbol",
    "source_id",
    "row_id",
})

# Display domains shown in Feature Transformations / Lag UI (business taxonomy).
# Source of truth: feature_domains.DOMAIN_ORDER / DOMAIN_LABELS.
from chain_replay_ml.dataset_builder.feature_domains import (
    DOMAIN_LABELS,
    DOMAIN_ORDER,
    group_features_by_domain as _group_by_domain,
    primary_domain_label,
)

CATEGORY_ORDER: tuple[str, ...] = tuple(DOMAIN_LABELS[d] for d in DOMAIN_ORDER) + ("Other",)

# Deprecated: plugin group_id → old UI category. Kept for tests that assert migration
# away from Engineered/ATM; classification now uses primary_domain_label().
_GROUP_TO_CATEGORY: dict[str, str] = {
    "price": "Price & Premium",
    "moneyness": "Price & Premium",
    "ltp_to_spot": "Price & Premium",
    "market_microstructure": "Volume & Liquidity",
    "historic_spot_ema": "Historical Context",
    "greeks": "Greeks",
    "iv": "Implied Volatility",
    "iv_zscore": "Implied Volatility",
    "iv_ema_ratio": "Implied Volatility",
    "oi": "Open Interest",
    "volume": "Volume & Liquidity",
    "atm_straddle": "Chain Analytics",
    "atm6_ltp": "Price & Premium",
    "chain": "Chain Analytics",
    "time": "Time & Session",
    "momentum": "Market Structure",
    "sharp_momentum": "Historical Context",
    "spot_hl": "Spot & Futures",
    "ratio": "Price & Premium",
    "dgt_reiv": "Price & Premium",
    "ltp_to_others": "Price & Premium",
    "spot_and_other_ratio": "Spot & Futures",
    "advanced": "Metadata",
    "chain_flow": "Chain Analytics",
    "historical": "Historical Context",
}

# Preset id → display domain label (or special "dynamic").
PRESET_SELECT_DYNAMIC = "dynamic"
PRESET_SELECT_PRICE = "Price & Premium"
PRESET_SELECT_GREEKS = "Greeks"
PRESET_SELECT_OI = "Open Interest"

_FEATURE_TO_CATEGORY_CACHE: dict[str, str] | None = None


def _feature_to_category_map() -> dict[str, str]:
    global _FEATURE_TO_CATEGORY_CACHE
    if _FEATURE_TO_CATEGORY_CACHE is not None:
        return _FEATURE_TO_CATEGORY_CACHE
    out: dict[str, str] = {}
    try:
        from chain_replay_ml.dataset_builder.feature_plugins import _REGISTRY_FEATURES
        from chain_replay_ml.dataset_builder.feature_domains import primary_domain_label as _pdl

        for _gid, feats in _REGISTRY_FEATURES.items():
            for feat in feats:
                name = str(feat)
                out.setdefault(name, _pdl(name))
    except Exception:
        pass
    _FEATURE_TO_CATEGORY_CACHE = out
    return out


def clear_feature_category_cache() -> None:
    """Reset cached feature→domain map (tests / after registry edits)."""
    global _FEATURE_TO_CATEGORY_CACHE
    _FEATURE_TO_CATEGORY_CACHE = None


def classify_feature(name: str) -> str:
    """Map a feature column name to its primary business domain label."""
    n = str(name or "").strip()
    if not n:
        return "Other"
    mapped = _feature_to_category_map().get(n)
    if mapped:
        return mapped
    try:
        return primary_domain_label(n)
    except Exception:
        return "Other"


def group_features_by_category(features: list[str]) -> dict[str, list[str]]:
    """Return ordered domain → sorted feature list (empty domains omitted)."""
    ordered = _group_by_domain([str(f) for f in features])
    return {label: sorted(names) for label, names in ordered}


def filter_features_by_search(features: list[str], query: str) -> list[str]:
    q = str(query or "").strip().lower()
    if not q:
        return list(features)
    return [f for f in features if q in str(f).lower()]


def features_for_preset(preset: str, available: list[str]) -> list[str]:
    """Return features to select for a preset button."""
    avail = [str(f) for f in available]
    avail_set = set(avail)
    key = str(preset or "").strip()
    if key == PRESET_SELECT_DYNAMIC:
        return [f for f in SUGGESTED_LAG_FEATURES if f in avail_set]
    if key in (PRESET_SELECT_PRICE, PRESET_SELECT_GREEKS, PRESET_SELECT_OI):
        return [f for f in avail if classify_feature(f) == key]
    return []


def lag_seconds_label(lag_seconds: int, sample_interval_sec: float | int) -> str:
    """Human label: '30 sec (10 rows)'."""
    sec = int(lag_seconds)
    try:
        interval = float(sample_interval_sec)
    except (TypeError, ValueError):
        return f"{sec} sec"
    if interval <= 0:
        return f"{sec} sec"
    rows = sec / interval
    rows_i = int(round(rows))
    if abs(rows - rows_i) > 1e-9:
        return f"{sec} sec ({rows:g} rows)"
    return f"{sec} sec ({rows_i} rows)"


def lag_seconds_is_valid_multiple(lag_seconds: int, sample_interval_sec: float | int) -> bool:
    try:
        interval = float(sample_interval_sec)
    except (TypeError, ValueError):
        return False
    if interval <= 0:
        return False
    rows = lag_seconds / interval
    rows_i = int(round(rows))
    return abs(rows - rows_i) <= 1e-9 and rows_i >= 1


def default_lag_seconds_for_interval(sample_interval_sec: float | int) -> tuple[int, ...]:
    """Interval-specific dynamic horizons (shared policy for Lag/Diff/Return/…)."""
    return default_horizons_for_interval(sample_interval_sec)


def resolve_warmup_seconds(
    detail: dict[str, Any] | None = None,
    *,
    sample_interval_sec: float | int | None = None,
) -> float:
    """Best-effort warm-up budget (seconds) for UI warnings.

    Preference: master/detail metadata → interval horizon policy → fallback.
    """
    d = detail if isinstance(detail, dict) else {}
    for key in ("warmup_seconds", "warmup_sec", "estimated_warmup_time_sec"):
        raw = d.get(key)
        if raw is not None and str(raw).strip() != "":
            try:
                return float(raw)
            except (TypeError, ValueError):
                pass
    for nest_key in ("dataset_meta", "build_summary", "build_summary_metadata"):
        nest = d.get(nest_key)
        if not isinstance(nest, dict):
            continue
        for key in ("warmup_seconds", "warmup_sec", "estimated_warmup_time_sec"):
            raw = nest.get(key)
            if raw is not None and str(raw).strip() != "":
                try:
                    return float(raw)
                except (TypeError, ValueError):
                    pass
    if sample_interval_sec is not None:
        try:
            return float(warmup_seconds_for_interval(sample_interval_sec))
        except Exception:
            pass
    return float(DEFAULT_WARMUP_SECONDS)


def lag_warmup_warning(
    *,
    enabled: bool,
    lag_seconds: list[int],
    warmup_seconds: float | int | None,
) -> str | None:
    """Warn when largest selected lag exceeds warm-up budget."""
    if not enabled:
        return None
    lags = [int(s) for s in lag_seconds if int(s) > 0]
    if not lags:
        return None
    try:
        warm = float(warmup_seconds) if warmup_seconds is not None else float(DEFAULT_WARMUP_SECONDS)
    except (TypeError, ValueError):
        warm = float(DEFAULT_WARMUP_SECONDS)
    if warm <= 0:
        return None
    largest = max(lags)
    if largest > warm:
        return (
            f"Largest lag ({largest:g}s) exceeds warm-up period ({warm:g}s). "
            "Early session rows will be null for that lag."
        )
    return None


def estimate_memory_mb(
    *,
    final_columns: int,
    estimated_rows: int | None,
) -> float | None:
    if estimated_rows is None or int(estimated_rows) <= 0 or int(final_columns) <= 0:
        return None
    return (int(final_columns) * int(estimated_rows) * BYTES_PER_CELL) / (1024.0 * 1024.0)


def format_memory_mb(mb: float | None) -> str:
    if mb is None:
        return "—"
    if mb < 1:
        return f"{mb * 1024:.0f} KB"
    if mb < 1024:
        return f"{mb:.1f} MB"
    return f"{mb / 1024:.2f} GB"


def build_lag_transformation_config(
    *,
    enabled: bool,
    features: list[str],
    lag_seconds: list[int],
    partition_by: list[str] | None = None,
    sample_interval_sec: float | int | None = None,
    difference_enabled: bool = False,
    return_enabled: bool = False,
    difference_features: list[str] | None = None,
    difference_lag_seconds: list[int] | None = None,
    return_features: list[str] | None = None,
    return_lag_seconds: list[int] | None = None,
) -> dict[str, Any]:
    """Build versioned pipeline config for Lag / Difference / Return.

    Each stage may supply its own ``features`` / horizons. When stage-specific
    lists are omitted, they reuse the Lag ``features`` / ``lag_seconds``.
    """
    feats = [str(f).strip() for f in features if str(f).strip()]
    lags = sorted({int(s) for s in lag_seconds if int(s) > 0})
    parts = [str(p).strip() for p in (partition_by or ["trading_day", "token"]) if str(p).strip()]
    any_enabled = bool(enabled or difference_enabled or return_enabled)
    if not any_enabled:
        return {
            "transformation_pipeline_version": 1,
            "transformations": [],
        }

    def _params(feat_list: list[str], sec_list: list[int]) -> dict[str, Any]:
        params: dict[str, Any] = {
            "features": list(feat_list),
            "lag_seconds": list(sec_list),
            "partition_by": parts,
        }
        if sample_interval_sec is not None:
            try:
                interval = float(sample_interval_sec)
            except (TypeError, ValueError):
                interval = None
            if interval is not None and interval > 0:
                params["sample_interval_sec"] = (
                    int(interval) if float(interval).is_integer() else interval
                )
        return params

    diff_feats = (
        [str(f).strip() for f in difference_features if str(f).strip()]
        if difference_features is not None
        else list(feats)
    )
    diff_secs = (
        sorted({int(s) for s in difference_lag_seconds if int(s) > 0})
        if difference_lag_seconds is not None
        else list(lags)
    )
    ret_feats = (
        [str(f).strip() for f in return_features if str(f).strip()]
        if return_features is not None
        else list(feats)
    )
    ret_secs = (
        sorted({int(s) for s in return_lag_seconds if int(s) > 0})
        if return_lag_seconds is not None
        else list(lags)
    )

    transforms: list[dict[str, Any]] = []
    if enabled:
        transforms.append({"id": "lag", "enabled": True, "params": _params(feats, lags)})
    if difference_enabled:
        transforms.append({
            "id": "difference",
            "enabled": True,
            "params": _params(diff_feats, diff_secs),
        })
    if return_enabled:
        transforms.append({
            "id": "return",
            "enabled": True,
            "params": _params(ret_feats, ret_secs),
        })
    return {
        "transformation_pipeline_version": 1,
        "transformations": transforms,
    }


def validate_time_shift_settings(
    *,
    lag_enabled: bool,
    difference_enabled: bool,
    return_enabled: bool,
    features: list[str],
    lag_seconds: list[int],
    sample_interval_sec: float | int,
    available_features: list[str] | None = None,
    difference_features: list[str] | None = None,
    difference_lag_seconds: list[int] | None = None,
    return_features: list[str] | None = None,
    return_lag_seconds: list[int] | None = None,
) -> str | None:
    """Validate each enabled Lag / Difference / Return stage independently."""
    checks: list[tuple[str, bool, list[str], list[int]]] = [
        ("Lag", lag_enabled, list(features), list(lag_seconds)),
        (
            "Difference",
            difference_enabled,
            list(difference_features if difference_features is not None else features),
            list(difference_lag_seconds if difference_lag_seconds is not None else lag_seconds),
        ),
        (
            "Return",
            return_enabled,
            list(return_features if return_features is not None else features),
            list(return_lag_seconds if return_lag_seconds is not None else lag_seconds),
        ),
    ]
    for label, on, feats, secs in checks:
        if not on:
            continue
        err = validate_lag_for_export(
            enabled=True,
            features=feats,
            lag_seconds=secs,
            sample_interval_sec=sample_interval_sec,
            available_features=available_features,
        )
        if err:
            return err.replace("Lag is enabled", f"{label} is enabled").replace(
                "Lag features", f"{label} features"
            ).replace("Lag seconds", f"{label} seconds")
    return None


def lag_preview_counts(
    *,
    enabled: bool,
    selected_features: list[str],
    lag_seconds: list[int],
    current_columns: int,
    estimated_rows: int | None = None,
    transform_count: int = 1,
) -> dict[str, Any]:
    n_feat = len([f for f in selected_features if str(f).strip()]) if enabled else 0
    n_lags = len([int(s) for s in lag_seconds if int(s) > 0]) if enabled else 0
    n_xf = max(1, int(transform_count)) if enabled else 0
    lag_cols = n_feat * n_lags * n_xf
    current = max(0, int(current_columns or 0))
    final = current + lag_cols
    mem = estimate_memory_mb(final_columns=final, estimated_rows=estimated_rows) if enabled else None
    return {
        "selected_features": n_feat,
        "lag_intervals": n_lags,
        "transform_count": n_xf,
        "columns_to_add": lag_cols,
        "current_columns": current,
        "lag_columns": lag_cols,
        "final_columns": final,
        "estimated_rows": int(estimated_rows) if estimated_rows is not None else None,
        "estimated_memory_mb": mem,
        "estimated_memory_label": format_memory_mb(mem),
    }


def format_lag_preview_text(
    counts: dict[str, Any],
    *,
    enabled: bool,
    warmup_warning: str | None = None,
) -> str:
    if not enabled:
        return "Time-shift transforms disabled — no extra columns."
    lines = [
        f"Selected Features : {counts.get('selected_features', 0)}",
        f"Horizons          : {counts.get('lag_intervals', 0)}",
        f"Transforms        : {counts.get('transform_count', 1)}",
        f"Current Columns   : {counts.get('current_columns', 0)}",
        f"New Columns       : {counts.get('lag_columns', 0)}",
        f"Final Columns     : {counts.get('final_columns', 0)}",
        f"Estimated Memory  : {counts.get('estimated_memory_label', '—')}",
    ]
    if warmup_warning:
        lines.append("")
        lines.append(f"⚠ {warmup_warning}")
    return "\n".join(lines)


def validate_lag_for_export(
    *,
    enabled: bool,
    features: list[str],
    lag_seconds: list[int],
    sample_interval_sec: float | int,
    available_features: list[str] | None = None,
    warmup_seconds: float | int | None = None,
) -> str | None:
    """Return an error message, or None if OK."""
    if not enabled:
        return None
    feats = [str(f).strip() for f in features if str(f).strip()]
    lags = [int(s) for s in lag_seconds if int(s) > 0]
    if not feats:
        return "Lag is enabled but no features are selected."
    if not lags:
        return "Lag is enabled but no lag seconds are selected."
    try:
        interval = float(sample_interval_sec)
    except (TypeError, ValueError):
        return "Sample interval is missing — cannot convert lag seconds to rows."
    if interval <= 0:
        return f"Invalid sample interval: {sample_interval_sec}"
    bad: list[str] = []
    for sec in lags:
        if not lag_seconds_is_valid_multiple(sec, interval):
            rows = sec / interval
            bad.append(f"{sec}s (÷ {interval:g}s = {rows:g} rows)")
    if bad:
        return (
            "Lag seconds must be exact multiples of the sample interval "
            f"({interval:g}s).\nInvalid:\n" + "\n".join(bad)
        )
    if available_features is not None:
        avail = {str(f) for f in available_features}
        missing = [f for f in feats if f not in avail]
        if missing:
            return "Lag features not found in master schema:\n" + "\n".join(missing)
    # Warm-up mismatch is a warning in the UI, not a hard export block.
    _ = warmup_seconds
    return None


def default_selected_lag_features(available: list[str]) -> list[str]:
    avail = {str(f) for f in available}
    return [f for f in SUGGESTED_LAG_FEATURES if f in avail]


def canonical_registry_feature_names() -> set[str]:
    """Feature Registry source of truth (``_REGISTRY_FEATURES``)."""
    from chain_replay_ml.dataset_builder.schema_registry import canonical_plugin_feature_names

    return canonical_plugin_feature_names()


def filter_to_registry_features(
    columns: list[str],
    *,
    exclude_names: set[str] | frozenset[str] | None = None,
) -> list[str]:
    """Keep only columns that exist in the canonical Feature Registry.

    Stale Master DB columns (pipeline-owned / retired / migrated) are dropped so
    Feature Transformation and Create Dataset stay aligned with the registry.
    """
    registry = canonical_registry_feature_names()
    if exclude_names:
        registry -= {str(n) for n in exclude_names}
    if not registry:
        return [str(c) for c in columns if str(c).strip()]
    return [str(c) for c in columns if str(c).strip() and str(c) in registry]


def registry_feature_count_from_master(
    feature_columns: list[str] | None,
    *,
    fallback_feature_count: int = 0,
    exclude_names: set[str] | frozenset[str] | None = None,
) -> int:
    """Count of master feature columns that are still active in the Feature Registry."""
    blocked = {str(n) for n in (exclude_names or ())}
    cols = [str(c) for c in (feature_columns or []) if str(c).strip()]
    if cols:
        kept = filter_to_registry_features(cols, exclude_names=blocked)
        return len(kept)
    registry = canonical_registry_feature_names()
    if blocked:
        registry -= blocked
    if registry:
        return len(registry)
    return max(0, int(fallback_feature_count or 0))


def filter_laggable_features(
    columns: list[str],
    *,
    registry_only: bool = True,
    exclude_names: set[str] | frozenset[str] | None = None,
    data_dir: str | None = None,
) -> list[str]:
    """Columns offered in Feature Transformation feature selection.

    When ``registry_only`` is True (default), only canonical registry features are
    listed — not obsolete Master columns that still sit on disk.

    When ``data_dir`` is supplied, the authoritative active registry list from
    ``get_active_feature_names`` is used instead of the static canonical catalogue.
    """
    names = [
        str(c) for c in columns
        if str(c).strip() and str(c) not in META_SKIP_COLUMNS
    ]
    if registry_only:
        if data_dir:
            from chain_replay_ml.dataset_builder.feature_sources_catalog import (
                get_active_feature_names,
            )

            active = set(get_active_feature_names(data_dir))
            blocked = {str(n) for n in (exclude_names or ())}
            names = [n for n in names if n in active and n not in blocked]
        else:
            names = filter_to_registry_features(names, exclude_names=exclude_names)
    return sorted(names)

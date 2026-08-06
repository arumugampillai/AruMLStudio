"""Feature expectation / availability — registry-aware defaults for Day Metadata.

Each feature can declare (now or later in Feature Registry)::

    required: bool
    expected_source: str
    can_be_empty: bool

Until the registry stores these explicitly, we infer from name + group.
"""

from __future__ import annotations

from typing import Any

# UI / catalog labels
SOURCE_SPOT = "Spot Feed"
SOURCE_OPTION = "Option Feed"
SOURCE_FUTURES = "Futures Feed"
SOURCE_IV = "IV Engine"
SOURCE_DERIVED = "Derived"
SOURCE_TARGET = "Target"
SOURCE_META = "Meta"
SOURCE_UNKNOWN = "Unknown"

AVAIL_AVAILABLE = "Available"
AVAIL_UNAVAILABLE = "Unavailable"
AVAIL_OPTIONAL = "Optional"
AVAIL_DISABLED = "Disabled"
AVAIL_DEPRECATED = "Deprecated"

# Hard-required core columns for a trading option dataset.
_REQUIRED_NAMES: frozenset[str] = frozenset(
    {
        "spot",
        "ltp",
        "timestamp",
        "trading_day",
        "token",
        "strike",
        "option_type",
    }
)

# Optional / often absent collectors.
_OPTIONAL_PREFIXES: tuple[str, ...] = (
    "futures_",
)

_OPTIONAL_NAMES: frozenset[str] = frozenset(
    {
        "futures_ltp",
        "futures_bid",
        "futures_ask",
        "futures_vwap",
        "futures_oi",
        "futures_spread",
        "futures_day_volume",
    }
)

_SOURCE_BY_PREFIX: tuple[tuple[str, str], ...] = (
    ("futures_", SOURCE_FUTURES),
    ("future_", SOURCE_TARGET),  # prediction targets e.g. future_ltp_5m
    ("atm_iv", SOURCE_IV),
    ("iv_", SOURCE_IV),
)

_GROUP_SOURCE: dict[str, str] = {
    "price": SOURCE_OPTION,
    "greeks": SOURCE_DERIVED,
    "iv": SOURCE_IV,
    "iv_zscore": SOURCE_IV,
    "iv_ema_ratio": SOURCE_IV,
    "market_microstructure": SOURCE_OPTION,
    "dgt_reiv": SOURCE_DERIVED,
    "time": SOURCE_META,
}


def _as_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    raw = str(value).strip().lower()
    if raw in ("1", "true", "yes", "on"):
        return True
    if raw in ("0", "false", "no", "off"):
        return False
    return default


def infer_source(name: str, *, group_id: str | None = None) -> str:
    n = str(name or "").strip()
    low = n.lower()
    if low in ("spot",):
        return SOURCE_SPOT
    if low in ("trading_day", "market", "expiry", "timestamp", "token", "symbol", "master_row_id"):
        return SOURCE_META
    if low.startswith("future_") and not low.startswith("futures_"):
        return SOURCE_TARGET
    for prefix, src in _SOURCE_BY_PREFIX:
        if low.startswith(prefix):
            return src
    if low in ("current_iv",) or "iv" in low and not low.startswith("bid"):
        if any(k in low for k in ("iv", "vol")):
            return SOURCE_IV
    if group_id and group_id in _GROUP_SOURCE:
        if low in ("spot",):
            return SOURCE_SPOT
        if low in ("ltp", "option_vwap", "option_bid", "option_ask", "bid_ask_spread"):
            return SOURCE_OPTION
        return _GROUP_SOURCE[group_id]
    if low in ("ltp", "bid", "ask", "option_vwap", "option_bid", "option_ask", "bid_ask_spread"):
        return SOURCE_OPTION
    if any(k in low for k in ("delta", "gamma", "theta", "vega", "charm", "vanna", "vomma", "rho")):
        return SOURCE_DERIVED
    if any(k in low for k in ("ema", "sma", "slope", "zscore", "vwap", "pred", "reiv")):
        return SOURCE_DERIVED
    return SOURCE_UNKNOWN


def infer_can_be_empty(name: str, *, required: bool | None = None) -> bool:
    n = str(name or "").strip()
    low = n.lower()
    if required is True:
        return False
    if n in _REQUIRED_NAMES or low in _REQUIRED_NAMES:
        return False
    if n in _OPTIONAL_NAMES or low in _OPTIONAL_NAMES:
        return True
    if any(low.startswith(p) for p in _OPTIONAL_PREFIXES):
        return True
    if low.startswith("future_") and not low.startswith("futures_"):
        # Targets may be trimmed at session edges — allow empty-ish but usually partial.
        return True
    return False


def infer_required(name: str) -> bool:
    n = str(name or "").strip()
    return n in _REQUIRED_NAMES or n.lower() in _REQUIRED_NAMES


def expectation_from_registry_entry(
    name: str,
    entry: dict[str, Any] | None = None,
    *,
    group_id: str | None = None,
) -> dict[str, Any]:
    """Normalize registry / inferred expectation for one feature."""
    doc = dict(entry or {})
    required = _as_bool(doc.get("required"), infer_required(name))
    can_be_empty = _as_bool(
        doc.get("can_be_empty", doc.get("canBeEmpty")),
        infer_can_be_empty(name, required=required),
    )
    if required:
        can_be_empty = False
    source = str(
        doc.get("expected_source")
        or doc.get("expectedSource")
        or doc.get("source")
        or infer_source(name, group_id=group_id)
    ).strip() or SOURCE_UNKNOWN
    # Map short source keys to labels.
    source_map = {
        "spot": SOURCE_SPOT,
        "option": SOURCE_OPTION,
        "futures": SOURCE_FUTURES,
        "iv": SOURCE_IV,
        "derived": SOURCE_DERIVED,
        "target": SOURCE_TARGET,
        "meta": SOURCE_META,
    }
    source = source_map.get(source.lower(), source)
    availability_hint = str(doc.get("availability") or "").strip()
    deprecated = _as_bool(doc.get("deprecated"), False)
    disabled = _as_bool(doc.get("disabled"), False)
    return {
        "name": str(name),
        "required": required,
        "can_be_empty": can_be_empty,
        "expected_source": source,
        "source": source,
        "deprecated": deprecated,
        "disabled": disabled,
        "availability_hint": availability_hint,
        "group_id": group_id,
    }


def build_expectation_index(registry: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    """feature_name → expectation dict from Feature Registry (+ inference)."""
    out: dict[str, dict[str, Any]] = {}
    if not isinstance(registry, dict):
        return out
    groups = registry.get("groups") or {}
    if not isinstance(groups, dict):
        return out
    for gid, gdoc in groups.items():
        feats = []
        if isinstance(gdoc, dict):
            feats = gdoc.get("features") or []
        elif isinstance(gdoc, list):
            feats = gdoc
        for f in feats:
            if isinstance(f, dict):
                name = str(f.get("name") or "").strip()
                if not name:
                    continue
                out[name] = expectation_from_registry_entry(name, f, group_id=str(gid))
            else:
                name = str(f).strip()
                if not name:
                    continue
                out[name] = expectation_from_registry_entry(name, None, group_id=str(gid))
    return out


def classify_population(
    *,
    name: str,
    coverage_pct: float,
    non_null: int,
    is_warmup: bool,
    is_constant: bool,
    is_meta: bool,
    registry_missing: bool,
    expectation: dict[str, Any] | None,
    futures_feed_empty: bool = False,
) -> dict[str, Any]:
    """Return status, reason, availability, expected_empty for one column."""
    exp = expectation or expectation_from_registry_entry(name)
    required = bool(exp.get("required"))
    can_be_empty = bool(exp.get("can_be_empty"))
    source = str(exp.get("source") or SOURCE_UNKNOWN)
    disabled = bool(exp.get("disabled"))
    deprecated = bool(exp.get("deprecated"))

    if disabled:
        return {
            "status": "Empty",
            "reason": "Disabled by configuration",
            "availability": AVAIL_DISABLED,
            "expected_empty": True,
            "source": source,
            "required": required,
            "can_be_empty": True,
        }
    if deprecated and non_null == 0:
        return {
            "status": "Expected Empty",
            "reason": "Deprecated feature",
            "availability": AVAIL_DEPRECATED,
            "expected_empty": True,
            "source": source,
            "required": required,
            "can_be_empty": True,
        }

    if registry_missing:
        if can_be_empty or not required:
            return {
                "status": "Registry Missing",
                "reason": "Optional / not materialized",
                "availability": AVAIL_OPTIONAL,
                "expected_empty": True,
                "source": source,
                "required": required,
                "can_be_empty": True,
            }
        return {
            "status": "Registry Missing",
            "reason": "Required column absent",
            "availability": AVAIL_UNAVAILABLE,
            "expected_empty": False,
            "source": source,
            "required": True,
            "can_be_empty": False,
        }

    if non_null == 0:
        if can_be_empty:
            reason = "Feature not collected"
            if source == SOURCE_FUTURES or str(name).startswith("futures_"):
                reason = (
                    "Futures feed not collected"
                    if futures_feed_empty
                    else "Futures source empty"
                )
            elif source == SOURCE_IV:
                reason = "IV engine produced no values"
            return {
                "status": "Expected Empty",
                "reason": reason,
                "availability": AVAIL_OPTIONAL,
                "expected_empty": True,
                "source": source,
                "required": required,
                "can_be_empty": True,
            }
        return {
            "status": "Unexpected Empty",
            "reason": (
                "Unexpected; required source should be present"
                if required
                else "Unexpected empty; feature should be populated"
            ),
            "availability": AVAIL_UNAVAILABLE,
            "expected_empty": False,
            "source": source,
            "required": required,
            "can_be_empty": False,
        }

    if is_warmup and coverage_pct >= 50.0:
        return {
            "status": "Warm-up",
            "reason": "Rolling window initialization",
            "availability": AVAIL_AVAILABLE,
            "expected_empty": False,
            "source": source,
            "required": required,
            "can_be_empty": can_be_empty,
        }
    if coverage_pct < 50.0:
        return {
            "status": "Sparse",
            "reason": "Signal occurs conditionally",
            "availability": AVAIL_AVAILABLE,
            "expected_empty": False,
            "source": source,
            "required": required,
            "can_be_empty": can_be_empty,
        }
    if is_constant and not is_meta:
        return {
            "status": "Constant",
            "reason": "Expected constant" if can_be_empty or name.startswith("is_") else "Single distinct value",
            "availability": AVAIL_AVAILABLE,
            "expected_empty": False,
            "source": source,
            "required": required,
            "can_be_empty": can_be_empty,
        }
    if coverage_pct < 100.0:
        return {
            "status": "Partial",
            "reason": f"Coverage {coverage_pct:.2f}%",
            "availability": AVAIL_AVAILABLE,
            "expected_empty": False,
            "source": source,
            "required": required,
            "can_be_empty": can_be_empty,
        }
    return {
        "status": "Healthy",
        "reason": "Fully populated",
        "availability": AVAIL_AVAILABLE,
        "expected_empty": False,
        "source": source,
        "required": required,
        "can_be_empty": can_be_empty,
    }

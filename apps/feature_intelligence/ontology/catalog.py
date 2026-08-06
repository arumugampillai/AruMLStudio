"""Frozen vocabulary + ontology seed catalog (Sprint 6 pack 1.0.0)."""

from __future__ import annotations

import hashlib
import json
from typing import NamedTuple

from feature_intelligence.ontology.identity import derive_ontology_uuid
from feature_intelligence.ontology.models import (
    ONTOLOGY_VERSION,
    VOCAB_CATALOG_VERSION,
    VOCAB_PACK_VERSION,
    normalize_id_list,
)

# Re-export pack versions
__all__ = [
    "ONTOLOGY_VERSION",
    "VOCAB_PACK_VERSION",
    "VOCAB_CATALOG_VERSION",
    "SEED_VOCABULARIES",
    "SEED_ONTOLOGY_ROWS",
    "EXPECTED_VOCAB_SEED_HASH",
    "EXPECTED_ONTOLOGY_SEED_HASH",
    "compute_vocab_seed_hash",
    "compute_ontology_seed_hash",
]


class SeedVocabulary(NamedTuple):
    vocabulary_id: str
    vocabulary_type: str
    canonical_name: str
    display_name: str
    description: str
    ontology_version: str
    active: int
    catalog_version: str
    sort_order: int | None


class SeedOntology(NamedTuple):
    object_type: str
    object_id: str
    domain: str
    signal_type: tuple[str, ...]
    mathematical_family: tuple[str, ...]
    horizon: str
    output_type: str
    frequency: str
    stability: str
    input_dependencies: tuple[str, ...]
    meaning: str | None
    classification_source: str


def _v(
    vid: str,
    vtype: str,
    canonical: str,
    display: str,
    desc: str,
    sort: int | None = None,
) -> SeedVocabulary:
    return SeedVocabulary(
        vocabulary_id=vid,
        vocabulary_type=vtype,
        canonical_name=canonical,
        display_name=display,
        description=desc,
        ontology_version=ONTOLOGY_VERSION,
        active=1,
        catalog_version=VOCAB_CATALOG_VERSION,
        sort_order=sort,
    )


def _ont(
    object_type: str,
    object_id: str,
    domain: str,
    signal_type: tuple[str, ...],
    mathematical_family: tuple[str, ...],
    horizon: str,
    output_type: str,
    frequency: str,
    stability: str = "STAB_STABLE",
    deps: tuple[str, ...] = (),
    meaning: str | None = None,
) -> SeedOntology:
    return SeedOntology(
        object_type=object_type,
        object_id=object_id,
        domain=domain,
        signal_type=tuple(normalize_id_list(list(signal_type))),
        mathematical_family=tuple(normalize_id_list(list(mathematical_family))),
        horizon=horizon,
        output_type=output_type,
        frequency=frequency,
        stability=stability,
        input_dependencies=tuple(normalize_id_list(list(deps))),
        meaning=meaning,
        classification_source="SEED",
    )


# ---------------------------------------------------------------------------
# Vocabulary seed (exactly the freeze §8 sets)
# ---------------------------------------------------------------------------

SEED_VOCABULARIES: tuple[SeedVocabulary, ...] = (
    # Domain
    _v("DOM_PRICE", "DOMAIN", "price", "Price", "Price / level of the underlier or quote", 1),
    _v("DOM_VOLUME", "DOMAIN", "volume", "Volume", "Traded volume", 2),
    _v("DOM_OPEN_INTEREST", "DOMAIN", "open_interest", "Open Interest", "Open interest", 3),
    _v("DOM_VOLATILITY", "DOMAIN", "volatility", "Volatility", "Realized or implied volatility", 4),
    _v("DOM_TIME", "DOMAIN", "time", "Time", "Clock / session / event time", 5),
    _v("DOM_CALENDAR", "DOMAIN", "calendar", "Calendar", "Calendar / expiry schedule concepts", 6),
    _v("DOM_ORDER_FLOW", "DOMAIN", "order_flow", "Order Flow", "Bid/ask / microstructure flow", 7),
    _v("DOM_GREEK", "DOMAIN", "greek", "Greek", "Option Greek risk dimensions", 8),
    _v("DOM_CONTRACT", "DOMAIN", "contract", "Contract", "Strike, expiry, contract specification", 9),
    _v("DOM_QUOTE", "DOMAIN", "quote", "Quote", "Bid/ask quote prices", 10),
    _v("DOM_DERIVED", "DOMAIN", "derived", "Derived", "Composite / derived from multiple domains", 11),
    # Signal type
    _v("SIG_RAW", "SIGNAL_TYPE", "raw", "Raw", "Untransformed market atom / identity signal", 1),
    _v("SIG_LEVEL", "SIGNAL_TYPE", "level", "Level", "Absolute level (price, strike, …)", 2),
    _v("SIG_TREND", "SIGNAL_TYPE", "trend", "Trend", "Directional trend", 3),
    _v("SIG_MOMENTUM", "SIGNAL_TYPE", "momentum", "Momentum", "Rate-of-change / momentum", 4),
    _v("SIG_MEAN_REVERSION", "SIGNAL_TYPE", "mean_reversion", "Mean Reversion", "Pull-to-mean behaviour", 5),
    _v("SIG_VOLATILITY", "SIGNAL_TYPE", "volatility", "Volatility", "Volatility / dispersion signal", 6),
    _v("SIG_LIQUIDITY", "SIGNAL_TYPE", "liquidity", "Liquidity", "Liquidity / depth related", 7),
    _v("SIG_PARTICIPATION", "SIGNAL_TYPE", "participation", "Participation", "Volume/OI participation", 8),
    _v("SIG_SPREAD", "SIGNAL_TYPE", "spread", "Spread", "Spread between series", 9),
    _v("SIG_RATIO", "SIGNAL_TYPE", "ratio", "Ratio", "Ratio signal", 10),
    _v("SIG_DIFFERENCE", "SIGNAL_TYPE", "difference", "Difference", "Difference / delta signal", 11),
    _v("SIG_STATISTICAL", "SIGNAL_TYPE", "statistical", "Statistical", "Statistical summary signal", 12),
    _v("SIG_STRUCTURE", "SIGNAL_TYPE", "structure", "Structure", "Contract/structure attribute", 13),
    _v("SIG_INTERACTION", "SIGNAL_TYPE", "interaction", "Interaction", "Cross-feature interaction", 14),
    _v("SIG_TRANSFORM", "SIGNAL_TYPE", "transform", "Transform", "Pure numeric transform (log, abs, …)", 15),
    # Mathematical family
    _v("MATH_IDENTITY", "MATH_FAMILY", "identity", "Identity", "Pass-through / primitive identity", 1),
    _v("MATH_MOVING_AVERAGE", "MATH_FAMILY", "moving_average", "Moving Average", "EMA / SMA / WMA family", 2),
    _v("MATH_ROLLING_WINDOW", "MATH_FAMILY", "rolling_window", "Rolling Window", "Rolling min/max/mean family", 3),
    _v("MATH_NORMALIZATION", "MATH_FAMILY", "normalization", "Normalization", "Z-score, normalize, clip", 4),
    _v("MATH_ARITHMETIC", "MATH_FAMILY", "arithmetic", "Arithmetic", "+ − × ÷ / ratio / product", 5),
    _v("MATH_STATISTICAL", "MATH_FAMILY", "statistical", "Statistical", "Mean, median, std, var, percentile", 6),
    _v("MATH_COMPARISON", "MATH_FAMILY", "comparison", "Comparison", "Min / max style comparisons", 7),
    _v("MATH_RANKING", "MATH_FAMILY", "ranking", "Ranking", "Rank / percentile-rank style", 8),
    _v("MATH_TRANSFORMATION", "MATH_FAMILY", "transformation", "Transformation", "Abs, log, exp, sqrt, …", 9),
    _v("MATH_AGGREGATION", "MATH_FAMILY", "aggregation", "Aggregation", "Sum / multi-input aggregates", 10),
    _v("MATH_TIME_SHIFT", "MATH_FAMILY", "time_shift", "Time Shift", "Lag / lead / delta / ROC", 11),
    _v("MATH_SLOPE", "MATH_FAMILY", "slope", "Slope", "Slope / linear trend fit", 12),
    _v("MATH_INTERACTION", "MATH_FAMILY", "interaction", "Interaction", "Interaction products", 13),
    # Horizon
    _v("HOR_TICK", "HORIZON", "tick", "Tick", "Tick / event scale", 1),
    _v("HOR_INTRADAY", "HORIZON", "intraday", "Intraday", "Intraday bars / session", 2),
    _v("HOR_SHORT", "HORIZON", "short", "Short", "Short horizon (minutes–hours class)", 3),
    _v("HOR_MEDIUM", "HORIZON", "medium", "Medium", "Medium horizon", 4),
    _v("HOR_LONG", "HORIZON", "long", "Long", "Long / multi-day class", 5),
    _v("HOR_MULTI_SCALE", "HORIZON", "multi_scale", "Multi-scale", "Explicitly multi-horizon", 6),
    _v("HOR_CONTRACT", "HORIZON", "contract", "Contract", "Contract-lifetime / expiry-tied", 7),
    _v("HOR_STATIC", "HORIZON", "static", "Static", "Non-temporal attribute (e.g. strike)", 8),
    # Output type
    _v("OUT_NUMERIC", "OUTPUT_TYPE", "numeric", "Numeric", "Continuous / numeric series", 1),
    _v("OUT_BOOLEAN", "OUTPUT_TYPE", "boolean", "Boolean", "Boolean flag", 2),
    _v("OUT_CATEGORY", "OUTPUT_TYPE", "category", "Category", "Discrete category", 3),
    _v("OUT_RANKING", "OUTPUT_TYPE", "ranking", "Ranking", "Rank-valued output", 4),
    _v("OUT_PROBABILITY", "OUTPUT_TYPE", "probability", "Probability", "Probability-valued output", 5),
    # Frequency
    _v("FREQ_TICK", "FREQUENCY", "tick", "Tick", "Tick / event frequency", 1),
    _v("FREQ_1S", "FREQUENCY", "1s", "1s", "1-second bars", 2),
    _v("FREQ_3S", "FREQUENCY", "3s", "3s", "3-second bars", 3),
    _v("FREQ_1M", "FREQUENCY", "1m", "1m", "1-minute bars", 4),
    _v("FREQ_5M", "FREQUENCY", "5m", "5m", "5-minute bars", 5),
    _v("FREQ_15M", "FREQUENCY", "15m", "15m", "15-minute bars", 6),
    _v("FREQ_DAILY", "FREQUENCY", "daily", "Daily", "Daily bars", 7),
    _v("FREQ_ANY", "FREQUENCY", "any", "Any", "Frequency-agnostic / not bound", 8),
    _v("FREQ_EVENT", "FREQUENCY", "event", "Event", "Event-driven (not fixed bar)", 9),
    # Stability
    _v("STAB_STABLE", "STABILITY", "stable", "Stable", "Classification marked stable", 1),
    _v("STAB_EXPERIMENTAL", "STABILITY", "experimental", "Experimental", "Classification experimental", 2),
    _v("STAB_DEPRECATED", "STABILITY", "deprecated", "Deprecated", "Classification deprecated", 3),
)

SEED_VOCAB_BY_ID: dict[str, SeedVocabulary] = {
    v.vocabulary_id: v for v in SEED_VOCABULARIES
}


# ---------------------------------------------------------------------------
# Required ontology seed — 14 primitives + 31 operators
# ---------------------------------------------------------------------------

_PR = "PRIMITIVE"
_OP = "OPERATOR"
_HI = "HOR_INTRADAY"
_ON = "OUT_NUMERIC"
_FA = "FREQ_ANY"
_MI = ("MATH_IDENTITY",)

SEED_ONTOLOGY_ROWS: tuple[SeedOntology, ...] = (
    # Primitives (freeze §16.1)
    _ont(_PR, "PR_SPOT", "DOM_PRICE", ("SIG_RAW", "SIG_LEVEL"), _MI, _HI, _ON, _FA),
    _ont(_PR, "PR_VOLUME", "DOM_VOLUME", ("SIG_RAW", "SIG_PARTICIPATION"), _MI, _HI, _ON, _FA),
    _ont(_PR, "PR_OI", "DOM_OPEN_INTEREST", ("SIG_RAW", "SIG_PARTICIPATION"), _MI, _HI, _ON, _FA),
    _ont(_PR, "PR_IV", "DOM_VOLATILITY", ("SIG_RAW", "SIG_VOLATILITY"), _MI, _HI, _ON, _FA),
    _ont(_PR, "PR_DELTA", "DOM_GREEK", ("SIG_RAW",), _MI, _HI, _ON, _FA),
    _ont(_PR, "PR_GAMMA", "DOM_GREEK", ("SIG_RAW",), _MI, _HI, _ON, _FA),
    _ont(_PR, "PR_THETA", "DOM_GREEK", ("SIG_RAW",), _MI, _HI, _ON, _FA),
    _ont(_PR, "PR_VEGA", "DOM_GREEK", ("SIG_RAW", "SIG_VOLATILITY"), _MI, _HI, _ON, _FA),
    _ont(_PR, "PR_RHO", "DOM_GREEK", ("SIG_RAW",), _MI, _HI, _ON, _FA),
    _ont(_PR, "PR_STRIKE", "DOM_CONTRACT", ("SIG_STRUCTURE", "SIG_LEVEL"), _MI, "HOR_STATIC", _ON, _FA),
    _ont(_PR, "PR_EXPIRY", "DOM_CALENDAR", ("SIG_STRUCTURE",), _MI, "HOR_CONTRACT", "OUT_CATEGORY", "FREQ_EVENT"),
    _ont(_PR, "PR_TIME", "DOM_TIME", ("SIG_RAW",), _MI, "HOR_TICK", _ON, "FREQ_TICK"),
    _ont(_PR, "PR_BID", "DOM_QUOTE", ("SIG_RAW", "SIG_LEVEL", "SIG_LIQUIDITY"), _MI, "HOR_TICK", _ON, "FREQ_TICK"),
    _ont(_PR, "PR_ASK", "DOM_QUOTE", ("SIG_RAW", "SIG_LEVEL", "SIG_LIQUIDITY"), _MI, "HOR_TICK", _ON, "FREQ_TICK"),
    # Operators (freeze §16.2 guidance)
    _ont(_OP, "OP_EMA", "DOM_DERIVED", ("SIG_TREND",), ("MATH_MOVING_AVERAGE",), _HI, _ON, _FA),
    _ont(_OP, "OP_SMA", "DOM_DERIVED", ("SIG_TREND",), ("MATH_MOVING_AVERAGE",), _HI, _ON, _FA),
    _ont(_OP, "OP_WMA", "DOM_DERIVED", ("SIG_TREND",), ("MATH_MOVING_AVERAGE",), _HI, _ON, _FA),
    _ont(_OP, "OP_SLOPE", "DOM_DERIVED", ("SIG_TREND", "SIG_MOMENTUM"), ("MATH_SLOPE",), _HI, _ON, _FA),
    _ont(_OP, "OP_RATIO", "DOM_DERIVED", ("SIG_RATIO",), ("MATH_ARITHMETIC",), _HI, _ON, _FA),
    _ont(_OP, "OP_DIFFERENCE", "DOM_DERIVED", ("SIG_DIFFERENCE",), ("MATH_ARITHMETIC",), _HI, _ON, _FA),
    _ont(_OP, "OP_SUM", "DOM_DERIVED", ("SIG_STATISTICAL",), ("MATH_AGGREGATION",), _HI, _ON, _FA),
    _ont(_OP, "OP_PRODUCT", "DOM_DERIVED", ("SIG_STATISTICAL",), ("MATH_ARITHMETIC",), _HI, _ON, _FA),
    _ont(_OP, "OP_DIVIDE", "DOM_DERIVED", ("SIG_RATIO",), ("MATH_ARITHMETIC",), _HI, _ON, _FA),
    _ont(_OP, "OP_LAG", "DOM_TIME", ("SIG_DIFFERENCE",), ("MATH_TIME_SHIFT",), _HI, _ON, _FA),
    _ont(_OP, "OP_LEAD", "DOM_TIME", ("SIG_DIFFERENCE",), ("MATH_TIME_SHIFT",), _HI, _ON, _FA),
    _ont(_OP, "OP_DELTA", "DOM_DERIVED", ("SIG_DIFFERENCE",), ("MATH_TIME_SHIFT",), _HI, _ON, _FA),
    _ont(_OP, "OP_ROC", "DOM_DERIVED", ("SIG_MOMENTUM",), ("MATH_TIME_SHIFT",), _HI, _ON, _FA),
    _ont(_OP, "OP_MIN", "DOM_DERIVED", ("SIG_STATISTICAL",), ("MATH_COMPARISON",), _HI, _ON, _FA),
    _ont(_OP, "OP_MAX", "DOM_DERIVED", ("SIG_STATISTICAL",), ("MATH_COMPARISON",), _HI, _ON, _FA),
    _ont(_OP, "OP_MEAN", "DOM_DERIVED", ("SIG_STATISTICAL",), ("MATH_STATISTICAL",), _HI, _ON, _FA),
    _ont(_OP, "OP_MEDIAN", "DOM_DERIVED", ("SIG_STATISTICAL",), ("MATH_STATISTICAL",), _HI, _ON, _FA),
    _ont(_OP, "OP_STDDEV", "DOM_DERIVED", ("SIG_STATISTICAL",), ("MATH_STATISTICAL",), _HI, _ON, _FA),
    _ont(_OP, "OP_VARIANCE", "DOM_DERIVED", ("SIG_STATISTICAL",), ("MATH_STATISTICAL",), _HI, _ON, _FA),
    _ont(_OP, "OP_PERCENTILE", "DOM_DERIVED", ("SIG_STATISTICAL",), ("MATH_STATISTICAL",), _HI, _ON, _FA),
    _ont(_OP, "OP_ZSCORE", "DOM_DERIVED", ("SIG_STATISTICAL",), ("MATH_NORMALIZATION",), _HI, _ON, _FA),
    _ont(_OP, "OP_CLIP", "DOM_DERIVED", ("SIG_TRANSFORM",), ("MATH_NORMALIZATION",), _HI, _ON, _FA),
    _ont(_OP, "OP_NORMALIZE", "DOM_DERIVED", ("SIG_STATISTICAL",), ("MATH_NORMALIZATION",), _HI, _ON, _FA),
    _ont(_OP, "OP_ABS", "DOM_DERIVED", ("SIG_TRANSFORM",), ("MATH_TRANSFORMATION",), _HI, _ON, _FA),
    _ont(_OP, "OP_LOG", "DOM_DERIVED", ("SIG_TRANSFORM",), ("MATH_TRANSFORMATION",), _HI, _ON, _FA),
    _ont(_OP, "OP_EXP", "DOM_DERIVED", ("SIG_TRANSFORM",), ("MATH_TRANSFORMATION",), _HI, _ON, _FA),
    _ont(_OP, "OP_SQRT", "DOM_DERIVED", ("SIG_TRANSFORM",), ("MATH_TRANSFORMATION",), _HI, _ON, _FA),
    _ont(_OP, "OP_ROLLING_MIN", "DOM_DERIVED", ("SIG_STATISTICAL",), ("MATH_ROLLING_WINDOW",), _HI, _ON, _FA),
    _ont(_OP, "OP_ROLLING_MAX", "DOM_DERIVED", ("SIG_STATISTICAL",), ("MATH_ROLLING_WINDOW",), _HI, _ON, _FA),
    _ont(_OP, "OP_ROLLING_MEAN", "DOM_DERIVED", ("SIG_STATISTICAL",), ("MATH_ROLLING_WINDOW",), _HI, _ON, _FA),
    _ont(_OP, "OP_INTERACTION", "DOM_DERIVED", ("SIG_INTERACTION",), ("MATH_INTERACTION",), _HI, _ON, _FA),
)


def canonical_vocab_seed_document(
    seeds: tuple[SeedVocabulary, ...] | None = None,
) -> str:
    rows = sorted(seeds or SEED_VOCABULARIES, key=lambda v: v.vocabulary_id)
    lines: list[str] = []
    for v in rows:
        desc = v.description if v.description is not None else ""
        sort = "" if v.sort_order is None else str(v.sort_order)
        lines.append(
            f"{v.vocabulary_id}|{v.vocabulary_type}|{v.canonical_name}|"
            f"{v.display_name}|{desc}|{v.ontology_version}|{v.active}|"
            f"{v.catalog_version}|{sort}"
        )
    return "\n".join(lines)


def compute_vocab_seed_hash(
    seeds: tuple[SeedVocabulary, ...] | None = None,
) -> str:
    return hashlib.sha256(
        canonical_vocab_seed_document(seeds).encode("utf-8")
    ).hexdigest()


def _ont_line(row: SeedOntology) -> str:
    sig = json.dumps(list(row.signal_type), separators=(",", ":"))
    math = json.dumps(list(row.mathematical_family), separators=(",", ":"))
    deps = json.dumps(list(row.input_dependencies), separators=(",", ":"))
    return (
        f"{row.object_type}|{row.object_id}|{row.domain}|{sig}|{math}|"
        f"{row.horizon}|{row.output_type}|{row.frequency}|{row.stability}|{deps}"
    )


def canonical_ontology_seed_document(
    seeds: tuple[SeedOntology, ...] | None = None,
) -> str:
    rows = sorted(
        seeds or SEED_ONTOLOGY_ROWS,
        key=lambda r: (r.object_type, r.object_id),
    )
    return "\n".join(_ont_line(r) for r in rows)


def compute_ontology_seed_hash(
    seeds: tuple[SeedOntology, ...] | None = None,
) -> str:
    return hashlib.sha256(
        canonical_ontology_seed_document(seeds).encode("utf-8")
    ).hexdigest()


def ontology_uuid_for_seed(row: SeedOntology) -> str:
    return derive_ontology_uuid(row.object_type, row.object_id)


# Locked for pack 1.0.0 — bump only with intentional freeze change.
EXPECTED_VOCAB_SEED_HASH = (
    "e31e6a7eb941b9ced9c538a8e2663d3441bf2451a7e3b351e02ecbfeb1c6923a"
)
EXPECTED_ONTOLOGY_SEED_HASH = (
    "8afb597111988f7468c5d0e2f007ab477ddc89973a1e9c9fd3da09c78ad1d429"
)

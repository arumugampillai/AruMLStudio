"""Frozen Primitive Catalog seed set (Sprint 1 / catalog version 1.0)."""

from __future__ import annotations

import hashlib
from typing import NamedTuple

CATALOG_VERSION = "1.0"

PRIMITIVE_ID_PATTERN = r"^PR_[A-Z][A-Z0-9_]*$"

PRIMITIVE_TYPES = frozenset(
    {"PRICE", "SIZE", "VOLATILITY", "GREEK", "TIME", "CONTRACT", "QUOTE", "OTHER"}
)
DATA_SOURCES = frozenset(
    {
        "MARKET_FEED",
        "OPTION_CHAIN",
        "COMPUTED_GREEK",
        "CLOCK",
        "CONTRACT_SPEC",
        "OTHER",
    }
)
UNITS = frozenset(
    {
        "CURRENCY",
        "COUNT",
        "RATIO",
        "PERCENT",
        "YEAR",
        "SECOND",
        "DATE",
        "STRIKE",
        "GREEK",
        "NONE",
        "OTHER",
    }
)


class SeedPrimitive(NamedTuple):
    primitive_id: str
    name: str
    primitive_type: str
    description: str
    data_source: str
    units: str
    catalog_version: str


SEED_PRIMITIVES: tuple[SeedPrimitive, ...] = (
    SeedPrimitive("PR_SPOT", "Spot", "PRICE", "Underlying / spot price atom", "MARKET_FEED", "CURRENCY", "1.0"),
    SeedPrimitive("PR_VOLUME", "Volume", "SIZE", "Traded volume atom", "MARKET_FEED", "COUNT", "1.0"),
    SeedPrimitive("PR_OI", "OI", "SIZE", "Open interest atom", "OPTION_CHAIN", "COUNT", "1.0"),
    SeedPrimitive("PR_IV", "IV", "VOLATILITY", "Implied volatility atom", "OPTION_CHAIN", "RATIO", "1.0"),
    SeedPrimitive("PR_DELTA", "Delta", "GREEK", "Option delta", "COMPUTED_GREEK", "GREEK", "1.0"),
    SeedPrimitive("PR_GAMMA", "Gamma", "GREEK", "Option gamma", "COMPUTED_GREEK", "GREEK", "1.0"),
    SeedPrimitive("PR_THETA", "Theta", "GREEK", "Option theta", "COMPUTED_GREEK", "GREEK", "1.0"),
    SeedPrimitive("PR_VEGA", "Vega", "GREEK", "Option vega", "COMPUTED_GREEK", "GREEK", "1.0"),
    SeedPrimitive("PR_RHO", "Rho", "GREEK", "Option rho", "COMPUTED_GREEK", "GREEK", "1.0"),
    SeedPrimitive("PR_STRIKE", "Strike", "CONTRACT", "Option strike", "CONTRACT_SPEC", "STRIKE", "1.0"),
    SeedPrimitive("PR_EXPIRY", "Expiry", "CONTRACT", "Option expiry", "CONTRACT_SPEC", "DATE", "1.0"),
    SeedPrimitive("PR_TIME", "Time", "TIME", "Market/event time atom", "CLOCK", "SECOND", "1.0"),
    SeedPrimitive("PR_BID", "Bid", "QUOTE", "Bid price atom", "MARKET_FEED", "CURRENCY", "1.0"),
    SeedPrimitive("PR_ASK", "Ask", "QUOTE", "Ask price atom", "MARKET_FEED", "CURRENCY", "1.0"),
)

SEED_BY_ID: dict[str, SeedPrimitive] = {p.primitive_id: p for p in SEED_PRIMITIVES}


def canonical_seed_document(seeds: tuple[SeedPrimitive, ...] | None = None) -> str:
    """Build the canonical UTF-8 seed document used for SHA-256 hashing."""
    rows = sorted(seeds or SEED_PRIMITIVES, key=lambda p: p.primitive_id)
    lines: list[str] = []
    for p in rows:
        desc = p.description if p.description is not None else ""
        lines.append(
            f"{p.primitive_id}|{p.name}|{p.primitive_type}|{desc}|"
            f"{p.data_source}|{p.units}|{p.catalog_version}"
        )
    return "\n".join(lines) + "\n"


def compute_seed_catalog_hash(seeds: tuple[SeedPrimitive, ...] | None = None) -> str:
    """SHA-256 (lowercase hex) of the canonical seed catalog document."""
    payload = canonical_seed_document(seeds).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


# Locked for catalog 1.0 — update only with an intentional freeze bump.
# Recomputed via compute_seed_catalog_hash(); must match SEED_PRIMITIVES exactly.
EXPECTED_SEED_CATALOG_HASH = (
    "b03c0ae4fa0a3b9906d2054fb2b7157a5b47a0c912bad587636d2bf29626f857"
)

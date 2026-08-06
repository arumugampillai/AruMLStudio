"""Structured query language parser (Sprint 9) — field:value tokens only."""

from __future__ import annotations

import re

from feature_intelligence.query import error_codes as ec
from feature_intelligence.query.models import QUERY_FIELDS, QuerySpec, QueryToken

# User-facing aliases → canonical QUERY_FIELDS names
FIELD_ALIASES: dict[str, str] = {
    "feat": "feature",
}

_FIELD_SET = frozenset(QUERY_FIELDS) | frozenset(FIELD_ALIASES)
_TOKEN_RE = re.compile(r"^([a-z_]+):(\S+)$")
_FUZZY_MARKERS = ("*", "~", "%", "?", "[", "]")
_NL_HINTS = re.compile(
    r"\b(show|find|give|list|all|me|important|features|with|that|which)\b",
    re.IGNORECASE,
)


class QueryParseError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


def tokenize(query: str) -> list[str]:
    """Split on whitespace; empty / whitespace-only → []."""
    if query is None:
        return []
    return [t for t in str(query).split() if t]


def _reject_fuzzy(raw: str) -> None:
    for m in _FUZZY_MARKERS:
        if m in raw:
            raise QueryParseError(
                ec.QUERY_FUZZY_FORBIDDEN,
                f"Fuzzy / wildcard tokens are forbidden: {raw!r}",
            )


def _looks_like_nl(raw: str, tokens: list[str]) -> bool:
    if ":" not in raw and tokens:
        return True
    if any(":" not in t for t in tokens):
        # bare words without field:value
        joined = " ".join(tokens)
        if _NL_HINTS.search(joined) or any(":" not in t for t in tokens):
            return True
    return False


def parse_query(query: str | None, *, match_all: bool = False) -> QuerySpec:
    """
    Parse structured query into QuerySpec.

    Empty query is invalid unless match_all=True (API list-all mode).
    """
    text = "" if query is None else str(query).strip()
    if match_all:
        if text:
            raise QueryParseError(
                ec.QUERY_MATCH_ALL_CONFLICT,
                "match_all=True requires empty query",
            )
        return QuerySpec(tokens=(), match_all=True)

    if not text:
        raise QueryParseError(
            ec.QUERY_EMPTY,
            "Empty query is invalid for search; supply ≥1 field:value token",
        )

    tokens_raw = tokenize(text)
    if _looks_like_nl(text, tokens_raw):
        raise QueryParseError(
            ec.QUERY_NL_FORBIDDEN,
            "Natural-language queries are forbidden; use field:value tokens",
        )

    parsed: list[QueryToken] = []
    for raw in tokens_raw:
        _reject_fuzzy(raw)
        m = _TOKEN_RE.match(raw)
        if m is None:
            raise QueryParseError(
                ec.QUERY_MALFORMED,
                f"Malformed token (expected field:value): {raw!r}",
            )
        field, value = m.group(1), m.group(2)
        if field not in _FIELD_SET:
            raise QueryParseError(
                ec.QUERY_UNKNOWN_FIELD,
                f"Unknown query field: {field!r}",
            )
        field = FIELD_ALIASES.get(field, field)
        if not value:
            raise QueryParseError(
                ec.QUERY_EMPTY_VALUE,
                f"Empty value for field: {field!r}",
            )
        parsed.append(QueryToken(field=field, value=value, raw=raw))

    if not parsed:
        raise QueryParseError(ec.QUERY_EMPTY, "No tokens after parse")

    return QuerySpec(tokens=tuple(parsed), match_all=False)

"""Query language validation (Sprint 9) — no DB mutations."""

from __future__ import annotations

from pathlib import Path

from feature_intelligence.query import error_codes as ec
from feature_intelligence.query.filters import (
    FilterResolveError,
    build_filter_context,
    compile_predicates,
    resolve_feature_token_existence,
)
from feature_intelligence.query.language import QueryParseError, parse_query
from feature_intelligence.query.models import QueryValidationReport


def validate_query(
    query: str | None,
    *,
    db_path: Path | None = None,
    match_all: bool = False,
    check_feature_existence: bool = False,
) -> QueryValidationReport:
    """
    Validate structured query language (+ optional vocab / alias resolve).

    Does not mutate DB. Does not create FRRs.
    """
    errors: list[dict[str, str]] = []
    warnings: list[str] = []
    try:
        spec = parse_query(query, match_all=match_all)
    except QueryParseError as exc:
        return QueryValidationReport(
            ok=False,
            errors=[{"code": exc.code, "message": exc.message}],
        )

    if db_path is not None and not spec.match_all:
        ctx = build_filter_context(Path(db_path))
        try:
            compile_predicates(ctx, spec)
        except FilterResolveError as exc:
            errors.append({"code": exc.code, "message": exc.message})
        if check_feature_existence:
            for t in spec.tokens:
                if t.field == "feature" and not resolve_feature_token_existence(
                    ctx, t.value
                ):
                    errors.append(
                        {
                            "code": ec.QUERY_FEATURE_MISSING,
                            "message": f"Feature not found: {t.value}",
                        }
                    )

    return QueryValidationReport(
        ok=len(errors) == 0,
        errors=errors,
        warnings=warnings,
        spec=spec if not errors else spec,
    )

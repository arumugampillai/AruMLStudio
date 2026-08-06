"""Semantic Query engine orchestration (Sprint 9) — read-only."""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path
from typing import Any

from feature_intelligence.core.logging import get_logger
from feature_intelligence.lineage.service import LineageService
from feature_intelligence.ontology.models import OBJECT_TYPE_FEATURE
from feature_intelligence.ontology.store import OntologyStore
from feature_intelligence.query import error_codes as ec
from feature_intelligence.query.diagnostics import (
    MATCH_ALL_SQL,
    explorer_empty_hint,
    registry_and_frr_counts,
)
from feature_intelligence.query.filters import (
    FilterResolveError,
    apply_filters,
    build_filter_context,
    compile_predicates,
    sort_frrs,
)
from feature_intelligence.query.inspect import build_inspect_model
from feature_intelligence.query.language import QueryParseError, parse_query
from feature_intelligence.query.models import (
    QUERY_ENGINE_VERSION,
    QUERY_LANGUAGE_VERSION,
    SCHEMA_VERSION,
    ApiEnvelope,
    capabilities_payload,
)
from feature_intelligence.query.resolve import ResolveError, resolve_to_frr
from feature_intelligence.query.summary import (
    build_platform_summary,
    empty_references_payload,
    enrich_search_hits,
)
from feature_intelligence.query.validation import validate_query
from feature_intelligence.registry.feature_service import FeatureRegistryService
from feature_intelligence.research.store import ResearchStore

_log = get_logger("query.engine")


def _err(code: str, message: str, *, execution_ms: float | None = None) -> ApiEnvelope:
    return ApiEnvelope(
        ok=False,
        data=None,
        error={"code": code, "message": message},
        execution_ms=execution_ms,
    )


def _ok(data: Any, *, execution_ms: float | None = None) -> ApiEnvelope:
    return ApiEnvelope(ok=True, data=data, error=None, execution_ms=execution_ms)


def _ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000.0, 3)


class QueryEngine:
    """FRR-centric read-only query engine."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)

    def get_capabilities(self) -> ApiEnvelope:
        return _ok(capabilities_payload())

    def get_platform_summary(self) -> ApiEnvelope:
        start = time.perf_counter()
        abs_db = self.db_path.resolve()
        try:
            data = build_platform_summary(self.db_path)
        except sqlite3.OperationalError as exc:
            feat_c, frr_c = registry_and_frr_counts(abs_db)
            hint = explorer_empty_hint(feature_count=feat_c, frr_count=frr_c)
            msg = f"Database not ready at {abs_db}: {exc}"
            if hint:
                msg = f"{msg}. {hint}"
            _log.warning("[FIC DEBUG] get_platform_summary failed: %s", msg)
            print(f"[FIC DEBUG] get_platform_summary failed: {msg}")
            return _err(ec.QUERY_DB_NOT_READY, msg, execution_ms=_ms(start))
        return _ok(data, execution_ms=_ms(start))

    def get_references(
        self,
        *,
        feature_uuid: str | None = None,
        research_uuid: str | None = None,
    ) -> ApiEnvelope:
        """
        Phase 1 stub: returns empty linkage lists.

        Resolves FRR when identifiers are provided so callers get subject ids;
        never invents model/dataset links.
        """
        start = time.perf_counter()
        if not feature_uuid and not research_uuid:
            return _ok(
                empty_references_payload(),
                execution_ms=_ms(start),
            )
        try:
            subject = resolve_to_frr(
                self.db_path,
                feature_uuid=feature_uuid,
                research_uuid=research_uuid,
            )
        except ResolveError as exc:
            return _err(exc.code, exc.message, execution_ms=_ms(start))
        return _ok(
            empty_references_payload(
                feature_uuid=subject.research.feature_uuid,
                research_uuid=subject.research.research_uuid,
            ),
            execution_ms=_ms(start),
        )

    def search_features(
        self,
        *,
        query: str | None = None,
        match_all: bool = False,
    ) -> ApiEnvelope:
        start = time.perf_counter()
        abs_db = self.db_path.resolve()
        feat_c, frr_c = registry_and_frr_counts(abs_db)
        sql_path = (
            MATCH_ALL_SQL
            if match_all
            else "SELECT * FROM feature_research_record ... + in-memory predicates"
        )
        debug_pre = (
            f"[FIC DEBUG] search_features db={abs_db} "
            f"feature_registry={feat_c} feature_research_record={frr_c} "
            f"query={query!r} match_all={match_all} sql={sql_path}"
        )
        _log.info(debug_pre)
        print(debug_pre)

        try:
            spec = parse_query(query, match_all=match_all)
        except QueryParseError as exc:
            return _err(exc.code, exc.message, execution_ms=_ms(start))

        try:
            ctx = build_filter_context(self.db_path)
            predicates = compile_predicates(ctx, spec)
            rows = sort_frrs(apply_filters(ctx.research.list_records(), predicates))
            items = enrich_search_hits(ctx, rows)
        except FilterResolveError as exc:
            return _err(exc.code, exc.message, execution_ms=_ms(start))
        except sqlite3.OperationalError as exc:
            hint = explorer_empty_hint(feature_count=feat_c, frr_count=frr_c)
            msg = f"Database not ready at {abs_db}: {exc}"
            if hint:
                msg = f"{msg}. {hint}"
            debug_err = f"[FIC DEBUG] search_features FAILED hits=0 — {msg}"
            _log.warning(debug_err)
            print(debug_err)
            return _err(ec.QUERY_DB_NOT_READY, msg, execution_ms=_ms(start))

        debug_post = (
            f"[FIC DEBUG] search_features hits={len(items)} "
            f"match_all={spec.match_all} frr_scanned={frr_c}"
        )
        if not items:
            hint = explorer_empty_hint(feature_count=feat_c, frr_count=frr_c)
            if hint:
                debug_post = f"{debug_post} — {hint}"
        _log.info(debug_post)
        print(debug_post)

        return _ok(
            {
                "count": len(items),
                "items": items,
                "query": None if spec.match_all else query,
                "match_all": spec.match_all,
                "debug": {
                    "db_path": str(abs_db),
                    "feature_registry_count": feat_c,
                    "feature_research_record_count": frr_c,
                    "sql": sql_path,
                    "empty_hint": explorer_empty_hint(
                        feature_count=feat_c, frr_count=frr_c
                    )
                    if not items
                    else None,
                },
            },
            execution_ms=_ms(start),
        )

    def get_feature(
        self,
        *,
        feature_uuid: str | None = None,
        research_uuid: str | None = None,
        canonical_name: str | None = None,
    ) -> ApiEnvelope:
        start = time.perf_counter()
        try:
            subject = resolve_to_frr(
                self.db_path,
                feature_uuid=feature_uuid,
                research_uuid=research_uuid,
                canonical_name=canonical_name,
            )
        except ResolveError as exc:
            return _err(exc.code, exc.message, execution_ms=_ms(start))

        frr = subject.research
        identity: dict[str, Any] | None = None
        try:
            feat = FeatureRegistryService(self.db_path).get_by_uuid(frr.feature_uuid)
            identity = {
                "feature_uuid": feat.feature_uuid,
                "canonical_name": feat.canonical_name,
                "display_name": feat.display_name,
                "definition_version": feat.definition_version,
                "implementation_version": feat.implementation_version,
                "definition_hash": feat.definition_hash,
                "research_state": feat.research_state,
                "primitive_ids": list(feat.primitive_ids),
                "transformation_uuid": feat.transformation_uuid,
            }
        except Exception:
            identity = {"feature_uuid": frr.feature_uuid}

        return _ok(
            {
                "identity": identity,
                "research_uuid": frr.research_uuid,
                "feature_uuid": frr.feature_uuid,
                "canonical_name": subject.canonical_name,
                "frr_pointers": {
                    "ontology_uuid": frr.ontology_uuid,
                    "transformation_uuid": frr.transformation_uuid,
                    "lineage_version": frr.lineage_version,
                    "compiler_version": frr.compiler_version,
                    "grammar_version": frr.grammar_version,
                    "research_status": frr.research_status,
                    "validation_status": frr.validation_status,
                },
            },
            execution_ms=_ms(start),
        )

    def get_research(
        self,
        *,
        feature_uuid: str | None = None,
        research_uuid: str | None = None,
    ) -> ApiEnvelope:
        start = time.perf_counter()
        try:
            subject = resolve_to_frr(
                self.db_path,
                feature_uuid=feature_uuid,
                research_uuid=research_uuid,
                canonical_name=None,
            )
        except ResolveError as exc:
            return _err(exc.code, exc.message, execution_ms=_ms(start))
        return _ok(subject.research.to_dict(), execution_ms=_ms(start))

    def get_lineage(
        self,
        *,
        feature_uuid: str | None = None,
        research_uuid: str | None = None,
        direction: str = "both",
    ) -> ApiEnvelope:
        start = time.perf_counter()
        try:
            subject = resolve_to_frr(
                self.db_path,
                feature_uuid=feature_uuid,
                research_uuid=research_uuid,
            )
        except ResolveError as exc:
            return _err(exc.code, exc.message, execution_ms=_ms(start))

        fid = subject.research.feature_uuid
        svc = LineageService(self.db_path)
        d = (direction or "both").lower()
        allowed = {"parents", "children", "ancestors", "descendants", "both"}
        if d not in allowed:
            return _err(
                ec.QUERY_MALFORMED,
                f"Invalid direction {direction!r}",
                execution_ms=_ms(start),
            )

        data: dict[str, Any] = {
            "research_uuid": subject.research.research_uuid,
            "feature_uuid": fid,
            "direction": d,
        }
        if d in ("parents", "both"):
            data["parents"] = svc.parents(fid)
        if d in ("children", "both"):
            data["children"] = svc.children(fid)
        if d in ("ancestors", "both"):
            data["ancestors"] = svc.ancestors(fid)
        if d in ("descendants", "both"):
            data["descendants"] = svc.descendants(fid)
        return _ok(data, execution_ms=_ms(start))

    def get_ontology(
        self,
        *,
        feature_uuid: str | None = None,
        research_uuid: str | None = None,
        ontology_uuid: str | None = None,
    ) -> ApiEnvelope:
        start = time.perf_counter()
        store = OntologyStore(self.db_path)

        if ontology_uuid and not feature_uuid and not research_uuid:
            # Resolve ontology row, then require FRR for the feature object
            rec = None
            for ot in ("FEATURE", "PRIMITIVE", "OPERATOR", "TRANSFORMATION"):
                for row in store.list_ontology(ot):
                    if row.ontology_uuid == ontology_uuid:
                        rec = row
                        break
                if rec is not None:
                    break
            if rec is None:
                return _err(
                    ec.QUERY_NOT_FOUND,
                    f"Ontology not found: {ontology_uuid}",
                    execution_ms=_ms(start),
                )
            if rec.object_type != OBJECT_TYPE_FEATURE:
                return _ok(
                    {
                        "ontology": rec.to_dict(),
                        "research_uuid": None,
                        "note": "Non-feature ontology; FRR chain applies to FEATURE only",
                    },
                    execution_ms=_ms(start),
                )
            try:
                subject = resolve_to_frr(
                    self.db_path, feature_uuid=rec.object_id
                )
            except ResolveError as exc:
                return _err(exc.code, exc.message, execution_ms=_ms(start))
            return _ok(
                {
                    "research_uuid": subject.research.research_uuid,
                    "feature_uuid": subject.research.feature_uuid,
                    "ontology": rec.to_dict(),
                },
                execution_ms=_ms(start),
            )

        try:
            subject = resolve_to_frr(
                self.db_path,
                feature_uuid=feature_uuid,
                research_uuid=research_uuid,
            )
        except ResolveError as exc:
            return _err(exc.code, exc.message, execution_ms=_ms(start))

        frr = subject.research
        rec = None
        if frr.ontology_uuid:
            for row in store.list_ontology(OBJECT_TYPE_FEATURE):
                if row.ontology_uuid == frr.ontology_uuid:
                    rec = row
                    break
        if rec is None:
            rec = store.get_ontology(OBJECT_TYPE_FEATURE, frr.feature_uuid)

        return _ok(
            {
                "research_uuid": frr.research_uuid,
                "feature_uuid": frr.feature_uuid,
                "ontology": None if rec is None else rec.to_dict(),
            },
            execution_ms=_ms(start),
        )

    def inspect_feature(
        self,
        *,
        feature_uuid: str | None = None,
        research_uuid: str | None = None,
        canonical_name: str | None = None,
    ) -> ApiEnvelope:
        start = time.perf_counter()
        try:
            subject = resolve_to_frr(
                self.db_path,
                feature_uuid=feature_uuid,
                research_uuid=research_uuid,
                canonical_name=canonical_name,
            )
        except ResolveError as exc:
            return _err(exc.code, exc.message, execution_ms=_ms(start))

        data = build_inspect_model(self.db_path, subject)
        return _ok(data, execution_ms=_ms(start))

    def validate(
        self,
        query: str | None,
        *,
        match_all: bool = False,
        check_feature_existence: bool = False,
    ) -> ApiEnvelope:
        report = validate_query(
            query,
            db_path=self.db_path,
            match_all=match_all,
            check_feature_existence=check_feature_existence,
        )
        return ApiEnvelope(
            ok=report.ok,
            data=report.to_dict(),
            error=None
            if report.ok
            else (
                report.errors[0]
                if report.errors
                else {"code": ec.QUERY_MALFORMED, "message": "validation failed"}
            ),
            execution_ms=None,
        )


def coverage_gate(db_path: Path) -> dict[str, Any]:
    """
    Phase 1 query coverage: every FRR reachable via ≥1 semantic dimension.
    Read-only check — status / feature identity always counts as a dimension.
    """
    store = ResearchStore(Path(db_path))
    rows = store.list_records()
    reachable = 0
    for r in rows:
        # Every FRR is reachable by status: and feature: (identity dimension)
        if r.research_status and r.feature_uuid:
            reachable += 1
    total = len(rows)
    return {
        "total_frr": total,
        "reachable": reachable,
        "coverage_pct": 100.0 if total == 0 else 100.0 * reachable / total,
        "query_engine_version": QUERY_ENGINE_VERSION,
        "query_language_version": QUERY_LANGUAGE_VERSION,
        "schema_version": SCHEMA_VERSION,
        "passed": reachable == total,
    }

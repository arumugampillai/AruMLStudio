"""Unit tests for Semantic Query / Public API (Sprint 9)."""

from __future__ import annotations

import inspect
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from feature_intelligence import __version__
from feature_intelligence.api import (
    PUBLIC_CALLABLES,
    get_capabilities,
    get_feature,
    get_platform_summary,
    get_references,
    inspect_feature,
    search_features,
)
from feature_intelligence.api import public as public_api
from feature_intelligence.migrations.runner import MigrationRunner
from feature_intelligence.ontology.identity import derive_ontology_uuid
from feature_intelligence.ontology.models import (
    OBJECT_TYPE_FEATURE,
    ONTOLOGY_VERSION,
    OntologyRecord,
)
from feature_intelligence.ontology.store import OntologyStore
from feature_intelligence.query.engine import QueryEngine, coverage_gate
from feature_intelligence.query.language import QueryParseError, parse_query
from feature_intelligence.query.models import (
    QUERY_ENGINE_VERSION,
    QUERY_LANGUAGE_VERSION,
    SCHEMA_VERSION,
    QUERY_FIELDS,
)
from feature_intelligence.query.validation import validate_query
from feature_intelligence.registry.feature_service import FeatureRegistryService
from feature_intelligence.research.service import ResearchService


def _register_feature(db: Path, name: str = "spot_ema20") -> str:
    svc = FeatureRegistryService(db)
    rec = svc.register_feature(
        canonical_name=name,
        display_name=name,
        primitive_ids=["PR_SPOT"],
        created_by="test",
        controller_owner="test",
        warmup_periods=20,
        gap_policy="RESET_ON_GAP",
        memory_model="SLIDING_WINDOW",
    )
    return rec.feature_uuid


def _attach_feature_ontology(db: Path, feature_uuid: str) -> str:
    store = OntologyStore(db)
    ont_uuid = derive_ontology_uuid(OBJECT_TYPE_FEATURE, feature_uuid)
    rec = OntologyRecord(
        ontology_uuid=ont_uuid,
        object_type=OBJECT_TYPE_FEATURE,
        object_id=feature_uuid,
        ontology_version=ONTOLOGY_VERSION,
        domain="DOM_PRICE",
        signal_type=["SIG_MOMENTUM", "SIG_LEVEL"],
        mathematical_family=["MATH_MOVING_AVERAGE"],
        horizon="HOR_INTRADAY",
        output_type="OUT_NUMERIC",
        frequency="FREQ_ANY",
        stability="STAB_STABLE",
        classification_source="IMPORT",
    )
    store.upsert_ontology(rec)
    return ont_uuid


class TestQueryLanguage(unittest.TestCase):
    def test_parse_and_tokens(self) -> None:
        spec = parse_query("domain:DOM_PRICE status:ACTIVE")
        self.assertEqual(len(spec.tokens), 2)
        self.assertEqual(spec.tokens[0].field, "domain")
        self.assertFalse(spec.match_all)

    def test_empty_invalid(self) -> None:
        with self.assertRaises(QueryParseError) as ctx:
            parse_query("")
        self.assertEqual(ctx.exception.code, "QUERY_EMPTY")

    def test_match_all(self) -> None:
        spec = parse_query(None, match_all=True)
        self.assertTrue(spec.match_all)

    def test_nl_rejected(self) -> None:
        with self.assertRaises(QueryParseError) as ctx:
            parse_query("show me important EMA features")
        self.assertEqual(ctx.exception.code, "QUERY_NL_FORBIDDEN")

    def test_fuzzy_rejected(self) -> None:
        with self.assertRaises(QueryParseError) as ctx:
            parse_query("domain:*PRICE*")
        self.assertEqual(ctx.exception.code, "QUERY_FUZZY_FORBIDDEN")

    def test_unknown_field(self) -> None:
        with self.assertRaises(QueryParseError) as ctx:
            parse_query("importance:high")
        self.assertEqual(ctx.exception.code, "QUERY_UNKNOWN_FIELD")


class TestQueryEngine(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.db = Path(self._tmp.name) / "feature_intelligence.db"
        MigrationRunner(self.db).upgrade()
        self.feat_a = _register_feature(self.db, "alpha_feat")
        self.feat_b = _register_feature(self.db, "beta_feat")
        self.research = ResearchService(self.db)
        self.research.sync_research()
        ont = _attach_feature_ontology(self.db, self.feat_a)
        # link FRR ontology pointer (read path); use store upsert — sync may have left null
        frr = self.research.get_research_by_feature(self.feat_a)
        from feature_intelligence.research.models import FeatureResearchRecord

        self.research.store.upsert_record(
            FeatureResearchRecord(
                research_uuid=frr.research_uuid,
                feature_uuid=frr.feature_uuid,
                ontology_uuid=ont,
                research_status="ACTIVE",
                validation_status="pending",
                created_at=frr.created_at,
            )
        )
        self.engine = QueryEngine(self.db)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_capabilities(self) -> None:
        env = self.engine.get_capabilities()
        self.assertTrue(env.ok)
        data = env.data
        self.assertEqual(data["query_engine_version"], QUERY_ENGINE_VERSION)
        self.assertEqual(data["query_language_version"], QUERY_LANGUAGE_VERSION)
        self.assertEqual(data["schema_version"], SCHEMA_VERSION)
        self.assertEqual(data["supported_filters"], list(QUERY_FIELDS))
        self.assertTrue(data["read_only"])
        self.assertTrue(data["frr_mandatory"])
        self.assertNotIn("saved_queries", data)

    def test_search_frr_centric_deterministic(self) -> None:
        env = self.engine.search_features(match_all=True)
        self.assertTrue(env.ok)
        self.assertIsNotNone(env.execution_ms)
        self.assertGreaterEqual(env.execution_ms, 0)
        items = env.data["items"]
        self.assertEqual(env.data["count"], 2)
        uuids = [i["research_uuid"] for i in items]
        self.assertEqual(uuids, sorted(uuids))
        for item in items:
            self.assertTrue(item["research_uuid"].startswith("FRR_"))
            # Enriched grid fields always present as keys
            for key in (
                "domain",
                "primary_operator",
                "primary_primitive",
                "compiler_version",
                "ontology_version",
                "grammar_version",
            ):
                self.assertIn(key, item)

    def test_search_enriched_domain_and_primitive(self) -> None:
        env = self.engine.search_features(query="domain:DOM_PRICE")
        self.assertTrue(env.ok)
        self.assertEqual(env.data["count"], 1)
        hit = env.data["items"][0]
        self.assertEqual(hit["feature_uuid"], self.feat_a)
        self.assertIsNotNone(hit["domain"])
        self.assertEqual(hit["primary_primitive"], "PR_SPOT")
        self.assertIsNotNone(hit["ontology_version"])

    def test_feat_alias(self) -> None:
        env = self.engine.search_features(query=f"feat:{self.feat_a}")
        self.assertTrue(env.ok, env.error)
        self.assertEqual(env.data["count"], 1)

    def test_platform_summary(self) -> None:
        env = get_platform_summary(db_path=self.db)
        self.assertTrue(env.ok, env.error)
        data = env.data
        self.assertTrue(data["read_only"])
        counts = data["counts"]
        self.assertGreaterEqual(counts["features"], 2)
        self.assertGreaterEqual(counts["research_records"], 2)
        self.assertIn("primitives", counts)
        self.assertIn("operators", counts)
        self.assertIn("transformations", counts)
        self.assertIn("ontology_records", counts)
        vers = data["versions"]
        self.assertIn("compiler_version", vers)
        self.assertIn("grammar_pack_version", vers)
        self.assertIn("ontology_version", vers)

    def test_get_references_empty(self) -> None:
        env = get_references(feature_uuid=self.feat_a, db_path=self.db)
        self.assertTrue(env.ok, env.error)
        data = env.data
        self.assertEqual(data["models"], [])
        self.assertEqual(data["datasets"], [])
        self.assertEqual(data["research_programs"], [])
        self.assertEqual(data["experiments"], [])
        self.assertEqual(data["feature_uuid"], self.feat_a)

    def test_inspect_overview_summary_and_identity_owner(self) -> None:
        env = self.engine.inspect_feature(canonical_name="alpha_feat")
        self.assertTrue(env.ok, env.error)
        data = env.data
        self.assertIn("overview_summary", data)
        self.assertIn("alpha_feat", data["overview_summary"])
        self.assertEqual(data["identity"]["controller_owner"], "test")
        self.assertEqual(data["references"]["models"], [])
        self.assertEqual(data["references"]["datasets"], [])
        self.assertEqual(data["references"]["research_programs"], [])

    def test_search_domain_vocab_and_alias(self) -> None:
        by_id = self.engine.search_features(query="domain:DOM_PRICE")
        self.assertTrue(by_id.ok, by_id.error)
        self.assertEqual(by_id.data["count"], 1)
        by_alias = self.engine.search_features(query="domain:price")
        self.assertTrue(by_alias.ok, by_alias.error)
        self.assertEqual(by_alias.data["count"], 1)
        self.assertEqual(
            by_id.data["items"][0]["research_uuid"],
            by_alias.data["items"][0]["research_uuid"],
        )

    def test_search_status_and(self) -> None:
        env = self.engine.search_features(query="status:ACTIVE domain:DOM_PRICE")
        self.assertTrue(env.ok)
        self.assertEqual(env.data["count"], 1)

    def test_search_missing_feature_empty(self) -> None:
        env = self.engine.search_features(query="feature:no_such_feature")
        self.assertTrue(env.ok)
        self.assertEqual(env.data["count"], 0)

    def test_inspect_sections_present_and_execution_ms(self) -> None:
        env = self.engine.inspect_feature(canonical_name="alpha_feat")
        self.assertTrue(env.ok, env.error)
        self.assertIsNotNone(env.execution_ms)
        data = env.data
        sp = data["sections_present"]
        self.assertTrue(sp["research"])
        self.assertTrue(sp["references"])
        self.assertTrue(sp["identity"])
        self.assertTrue(sp["ontology"])
        self.assertIn("compiler", sp)
        self.assertIn("ast", sp)
        self.assertIn("lineage", sp)

    def test_missing_frr_error_no_create(self) -> None:
        # new feature without sync
        orphan = _register_feature(self.db, "orphan_feat")
        before = self.research.store.count_records()
        env = self.engine.inspect_feature(feature_uuid=orphan)
        self.assertFalse(env.ok)
        self.assertEqual(env.error["code"], "QUERY_FRR_MISSING")
        self.assertEqual(self.research.store.count_records(), before)

    def test_validate_query(self) -> None:
        bad = validate_query("domain:NotARealDomain", db_path=self.db)
        self.assertFalse(bad.ok)
        good = validate_query("status:active", db_path=self.db)
        self.assertTrue(good.ok, good.errors)

    def test_api_envelopes_and_public_surface(self) -> None:
        caps = get_capabilities(db_path=self.db)
        self.assertEqual(caps.schema_version, SCHEMA_VERSION)
        self.assertTrue(caps.ok)
        hits = search_features(query="status:EMPTY", db_path=self.db)
        self.assertIn("execution_ms", hits.to_dict())
        feat = get_feature(canonical_name="beta_feat", db_path=self.db)
        self.assertTrue(feat.ok)
        self.assertTrue(feat.data["research_uuid"].startswith("FRR_"))

        # public module exposes only read callables (no write helpers)
        for name in PUBLIC_CALLABLES:
            self.assertTrue(callable(getattr(public_api, name)))
        src = inspect.getsource(public_api)
        for forbidden in ("sync_research", "upsert", "INSERT", "create_frr"):
            self.assertNotIn(forbidden, src)

    def test_read_only_no_new_rows(self) -> None:
        before_frr = self.research.store.count_records()
        conn = sqlite3.connect(str(self.db))
        try:
            before_feat = conn.execute(
                "SELECT COUNT(*) FROM feature_registry"
            ).fetchone()[0]
            before_ont = conn.execute(
                "SELECT COUNT(*) FROM feature_ontology"
            ).fetchone()[0]
        finally:
            conn.close()

        self.engine.search_features(match_all=True)
        self.engine.inspect_feature(canonical_name="alpha_feat")
        get_capabilities(db_path=self.db)
        get_platform_summary(db_path=self.db)
        get_references(feature_uuid=self.feat_a, db_path=self.db)

        self.assertEqual(self.research.store.count_records(), before_frr)
        conn = sqlite3.connect(str(self.db))
        try:
            after_feat = conn.execute(
                "SELECT COUNT(*) FROM feature_registry"
            ).fetchone()[0]
            after_ont = conn.execute(
                "SELECT COUNT(*) FROM feature_ontology"
            ).fetchone()[0]
        finally:
            conn.close()
        self.assertEqual(after_feat, before_feat)
        self.assertEqual(after_ont, before_ont)

    def test_coverage_gate(self) -> None:
        report = coverage_gate(self.db)
        self.assertTrue(report["passed"])
        self.assertEqual(report["total_frr"], report["reachable"])

    def test_version_bump(self) -> None:
        self.assertEqual(__version__, "0.9.0")

    def test_no_migration_0010(self) -> None:
        versions_dir = (
            Path(__file__).resolve().parents[2] / "migrations" / "versions"
        )
        self.assertFalse((versions_dir / "0010_semantic_query.py").exists())
        names = [p.name for p in versions_dir.glob("0010*")]
        self.assertEqual(names, [])


class TestStudioSmokeImport(unittest.TestCase):
    def test_import_studio_panel_without_gui(self) -> None:
        # Import module graph only — do not instantiate Tk root
        import master_dataset_tk.feature_intelligence_studio_panel as fi_host
        import master_dataset_tk.feature_studio_panel as studio

        self.assertTrue(hasattr(studio, "FeatureStudioPanel"))
        self.assertTrue(hasattr(fi_host, "FeatureIntelligenceStudioPanel"))
        # Source must still list legacy analysis tab labels
        src = Path(studio.__file__).read_text(encoding="utf-8")
        for label in (
            "Importance",
            "Distribution",
            "Drift",
            "Studio Compare",
            "Diagnostics",
            "Experiment Planner",
            "Feature Intelligence",
        ):
            self.assertIn(label, src)


if __name__ == "__main__":
    unittest.main()

"""Unit tests for Feature Lineage (Sprint 7)."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from feature_intelligence.lineage import error_codes as ec
from feature_intelligence.lineage.derive import collect_derive_triples, derive_lineage
from feature_intelligence.lineage.graph import (
    ancestors_of,
    children_of,
    descendants_of,
    has_cycle,
    parents_of,
    would_introduce_cycle,
)
from feature_intelligence.lineage.identity import derive_lineage_uuid
from feature_intelligence.lineage.import_export import (
    edge_to_envelope,
    export_lineage,
    import_lineage,
)
from feature_intelligence.lineage.models import (
    GRAPH_EXPORT_VERSION,
    GRAPH_SCHEMA_VERSION,
    LINEAGE_VERSION,
    REL_DERIVED_FROM,
    REL_GENERATED_BY,
    REL_INPUT_TO,
    REL_USES,
    LineageEdge,
)
from feature_intelligence.lineage.relationships import (
    EXPECTED_RELATIONSHIP_SEED_HASH,
    SEED_RELATIONSHIPS,
    compute_relationship_seed_hash,
)
from feature_intelligence.lineage.service import LineageService
from feature_intelligence.lineage.store import (
    LineageStore,
    compute_graph_checksum,
    empty_graph_checksum,
)
from feature_intelligence.migrations.runner import MigrationRunner
from feature_intelligence.registry.models import ValidationReport


class TestLineageSeedFreeze(unittest.TestCase):
    def test_relationship_seed_hash_matches_expected(self) -> None:
        self.assertEqual(
            compute_relationship_seed_hash(), EXPECTED_RELATIONSHIP_SEED_HASH
        )

    def test_exactly_five_relationships(self) -> None:
        self.assertEqual(len(SEED_RELATIONSHIPS), 5)
        ids = {r.relationship_id for r in SEED_RELATIONSHIPS}
        self.assertEqual(
            ids,
            {
                "REL_USES",
                "REL_GENERATED_BY",
                "REL_DEPENDS_ON",
                "REL_DERIVED_FROM",
                "REL_INPUT_TO",
            },
        )

    def test_error_codes_catalog(self) -> None:
        self.assertGreaterEqual(len(ec.ALL_ERROR_CODES), 16)
        self.assertIn(ec.CYCLE_DETECTED, ec.ALL_ERROR_CODES)

    def test_cache_module_is_stub_only(self) -> None:
        import feature_intelligence.lineage.cache as cache_mod
        import inspect

        src = inspect.getsource(cache_mod)
        self.assertIn("RESERVED", src)
        # No traversal cache helpers implemented
        self.assertFalse(hasattr(cache_mod, "get"))
        self.assertFalse(hasattr(cache_mod, "put"))
        self.assertFalse(hasattr(cache_mod, "TraversalCache"))


class TestLineageIdentity(unittest.TestCase):
    def test_lineage_uuid_determinism(self) -> None:
        a = derive_lineage_uuid("PR_SPOT", "TR_" + "A" * 32, REL_INPUT_TO)
        b = derive_lineage_uuid("PR_SPOT", "TR_" + "A" * 32, REL_INPUT_TO)
        self.assertEqual(a, b)
        self.assertRegex(a, r"^LINEAGE_[0-9A-F]{32}$")

    def test_lineage_uuid_formula(self) -> None:
        parent, child, rel = "PR_SPOT", "TR_" + "B" * 32, REL_INPUT_TO
        material = f"{parent}|{child}|{rel}".encode("utf-8")
        expected = "LINEAGE_" + hashlib.sha256(material).hexdigest()[:32].upper()
        self.assertEqual(derive_lineage_uuid(parent, child, rel), expected)

    def test_checksum_empty_and_sorted(self) -> None:
        self.assertEqual(empty_graph_checksum(), hashlib.sha256(b"").hexdigest())
        triples = [
            ("PR_B", "TR_X", REL_INPUT_TO),
            ("PR_A", "TR_X", REL_INPUT_TO),
        ]
        # Wrong order input still sorts
        c1 = compute_graph_checksum(triples)
        c2 = compute_graph_checksum(list(reversed(triples)))
        self.assertEqual(c1, c2)
        payload = (
            f"PR_A\tTR_X\t{REL_INPUT_TO}\n"
            f"PR_B\tTR_X\t{REL_INPUT_TO}\n"
        )
        self.assertEqual(c1, hashlib.sha256(payload.encode("utf-8")).hexdigest())


class TestLineageDagRules(unittest.TestCase):
    def test_multi_parent_allowed(self) -> None:
        edges = [
            ("PR_A", "FEAT_1"),
            ("PR_B", "FEAT_1"),
        ]
        self.assertEqual(parents_of("FEAT_1", edges), ["PR_A", "PR_B"])
        self.assertFalse(has_cycle(edges))

    def test_cycle_rejected(self) -> None:
        edges = [("A", "B"), ("B", "C"), ("C", "A")]
        self.assertTrue(has_cycle(edges))
        self.assertTrue(would_introduce_cycle([("A", "B"), ("B", "C")], "C", "A"))

    def test_navigation_ancestors_descendants(self) -> None:
        edges = [
            ("PR_SPOT", "TR_1"),
            ("OP_EMA", "TR_1"),
            ("TR_1", "FEAT_1"),
            ("PR_SPOT", "FEAT_1"),
        ]
        self.assertEqual(children_of("PR_SPOT", edges), ["FEAT_1", "TR_1"])
        self.assertEqual(
            ancestors_of("FEAT_1", edges),
            ["OP_EMA", "PR_SPOT", "TR_1"],
        )
        self.assertEqual(descendants_of("PR_SPOT", edges), ["FEAT_1", "TR_1"])
        self.assertNotIn("FEAT_1", ancestors_of("FEAT_1", edges))


class TestLineageMigrateAndStats(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.db = Path(self._tmp.name) / "feature_intelligence.db"
        MigrationRunner(self.db).upgrade()
        self.svc = LineageService(self.db)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_migration_seeds_relationships_no_edges(self) -> None:
        self.assertEqual(len(self.svc.store.list_relationships()), 5)
        self.assertEqual(self.svc.store.count_edges(), 0)
        pack = self.svc.store.get_pack()
        assert pack is not None
        self.assertEqual(pack["lineage_version"], LINEAGE_VERSION)
        self.assertEqual(pack["graph_schema_version"], GRAPH_SCHEMA_VERSION)
        self.assertEqual(pack["graph_checksum"], empty_graph_checksum())
        self.assertEqual(
            pack["relationship_seed_checksum"], EXPECTED_RELATIONSHIP_SEED_HASH
        )

    def test_validate_empty_passes_and_writes_stats(self) -> None:
        self.assertEqual(self.svc.store.count_statistics(), 0)
        report = self.svc.validate_lineage(mode="present")
        self.assertIsInstance(report, ValidationReport)
        self.assertTrue(report.passed, report.failed_rules)
        self.assertEqual(self.svc.store.count_statistics(), 1)
        self.svc.validate_lineage(mode="present")
        self.assertEqual(self.svc.store.count_statistics(), 2)

    def test_stats_read_then_miss_regen(self) -> None:
        self.assertEqual(self.svc.store.count_statistics(), 0)
        first = self.svc.lineage_stats()
        self.assertTrue(first.from_snapshot)
        self.assertEqual(self.svc.store.count_statistics(), 1)
        self.assertEqual(len(first.relationship_counts), 5)
        second = self.svc.lineage_stats()
        self.assertEqual(self.svc.store.count_statistics(), 1)
        self.assertEqual(second.snapshot_created_at, first.snapshot_created_at)

    def test_import_never_writes_stats(self) -> None:
        self.assertEqual(self.svc.store.count_statistics(), 0)
        tr = "TR_" + "C" * 32
        edge = LineageEdge(
            lineage_uuid=derive_lineage_uuid("PR_SPOT", tr, REL_INPUT_TO),
            parent_object="PR_SPOT",
            child_object=tr,
            relationship_id=REL_INPUT_TO,
            edge_source="IMPORT",
        )
        envelope = {
            "lineage_version": LINEAGE_VERSION,
            "graph_schema_version": GRAPH_SCHEMA_VERSION,
            "graph_export_version": GRAPH_EXPORT_VERSION,
            "edges": [edge_to_envelope(edge)],
        }
        path = Path(self._tmp.name) / "lin.json"
        path.write_text(json.dumps(envelope), encoding="utf-8")
        report = import_lineage(self.svc, path, fmt="json")
        self.assertTrue(report.passed, report.failed_rules)
        self.assertEqual(self.svc.store.count_statistics(), 0)
        self.assertEqual(self.svc.store.count_edges(), 1)

    def test_import_export_roundtrip(self) -> None:
        tr = "TR_" + "D" * 32
        feat = "FEAT_" + "E" * 32
        for parent, child, rel in (
            ("PR_SPOT", tr, REL_INPUT_TO),
            ("OP_EMA", tr, REL_USES),
            (tr, feat, REL_GENERATED_BY),
            ("PR_SPOT", feat, REL_DERIVED_FROM),
        ):
            self.svc.store.upsert_edge(
                LineageEdge(
                    lineage_uuid=derive_lineage_uuid(parent, child, rel),
                    parent_object=parent,
                    child_object=child,
                    relationship_id=rel,
                    edge_source="IMPORT",
                )
            )
        out1 = Path(self._tmp.name) / "a.json"
        out2 = Path(self._tmp.name) / "b.json"
        export_lineage(self.svc, out1, fmt="json")
        data1 = json.loads(out1.read_text(encoding="utf-8"))
        self.assertEqual(data1["graph_export_version"], GRAPH_EXPORT_VERSION)
        self.assertEqual(data1["graph_schema_version"], GRAPH_SCHEMA_VERSION)
        # wipe and re-import
        import sqlite3

        conn = sqlite3.connect(str(self.db))
        conn.execute("DELETE FROM lineage_edges")
        conn.commit()
        conn.close()
        report = import_lineage(self.svc, out1, fmt="json")
        self.assertTrue(report.passed, report.failed_rules)
        export_lineage(self.svc, out2, fmt="json")
        data2 = json.loads(out2.read_text(encoding="utf-8"))
        self.assertEqual(data1["edges"], data2["edges"])

    def test_self_edge_and_cycle_on_import(self) -> None:
        bad = {
            "lineage_version": LINEAGE_VERSION,
            "graph_schema_version": GRAPH_SCHEMA_VERSION,
            "graph_export_version": GRAPH_EXPORT_VERSION,
            "edges": [
                {
                    "parent_object": "PR_SPOT",
                    "child_object": "PR_SPOT",
                    "relationship_id": REL_INPUT_TO,
                }
            ],
        }
        path = Path(self._tmp.name) / "self.json"
        path.write_text(json.dumps(bad), encoding="utf-8")
        report = import_lineage(self.svc, path, fmt="json")
        self.assertFalse(report.passed)
        self.assertIn(ec.SELF_EDGE, report.failed_rules)

    def test_multi_parent_store_and_nav(self) -> None:
        feat = "FEAT_" + "F" * 32
        for pr in ("PR_SPOT", "PR_ASK"):
            self.svc.store.upsert_edge(
                LineageEdge(
                    lineage_uuid=derive_lineage_uuid(pr, feat, REL_DERIVED_FROM),
                    parent_object=pr,
                    child_object=feat,
                    relationship_id=REL_DERIVED_FROM,
                    edge_source="IMPORT",
                )
            )
        self.assertEqual(self.svc.parents(feat), ["PR_ASK", "PR_SPOT"])
        self.assertEqual(self.svc.ancestors(feat), ["PR_ASK", "PR_SPOT"])

    def test_checksum_refreshed_on_validate(self) -> None:
        tr = "TR_" + "G" * 32
        self.svc.store.upsert_edge(
            LineageEdge(
                lineage_uuid=derive_lineage_uuid("PR_SPOT", tr, REL_INPUT_TO),
                parent_object="PR_SPOT",
                child_object=tr,
                relationship_id=REL_INPUT_TO,
                edge_source="IMPORT",
            )
        )
        # Corrupt stored checksum
        self.svc.store.update_graph_checksum("deadbeef" * 8)
        report = self.svc.validate_lineage(mode="present")
        # CHECKSUM_MISMATCH reported but checksum refreshed
        pack = self.svc.store.get_pack()
        assert pack is not None
        expected = compute_graph_checksum(
            [
                (e.parent_object, e.child_object, e.relationship_id)
                for e in self.svc.list_edges()
            ]
        )
        self.assertEqual(pack["graph_checksum"], expected)
        self.assertIn(ec.CHECKSUM_MISMATCH, report.failed_rules)


class TestLineageDerive(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.db = Path(self._tmp.name) / "feature_intelligence.db"
        MigrationRunner(self.db).upgrade()
        self.store = LineageStore(self.db)
        self._seed_compiler_fixtures()

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _seed_compiler_fixtures(self) -> None:
        """Insert minimal Sprint 5 rows without storing AST into lineage."""
        import sqlite3

        tr = "TR_" + "1" * 32
        feat = "FEAT_" + "2" * 32
        ast_hash = "a" * 64
        expr_hash = "b" * 64
        self.tr = tr
        self.feat = feat
        conn = sqlite3.connect(str(self.db))
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute(
            """
            INSERT OR IGNORE INTO feature_registry(
                feature_uuid, canonical_name, display_name,
                definition_version, implementation_version, definition_hash,
                created_by, controller_owner, warmup_periods, gap_policy,
                memory_model
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                feat,
                "lineage_test_feat",
                "Lineage Test",
                "1.0.0",
                "1.0.0",
                "d" * 64,
                "test",
                "test",
                0,
                "FORWARD_FILL",
                "STATELESS",
            ),
        )
        conn.execute(
            """
            INSERT OR IGNORE INTO transformation_registry(
                transformation_uuid, expression_hash, canonical_text
            ) VALUES (?,?,?)
            """,
            (tr, expr_hash, "OP_EMA(source=PR_SPOT, period=20)"),
        )
        comp = "COMP_" + "3" * 32
        conn.execute(
            """
            INSERT OR IGNORE INTO compilation_registry(
                compilation_uuid, transformation_uuid, ast_schema_version,
                grammar_version, compiler_version, operator_pack_version,
                ast_hash, root_node_id, status
            ) VALUES (?,?,?,?,?,?,?,?,?)
            """,
            (
                comp,
                tr,
                "1.0",
                "1.0.0",
                "1.0.0",
                "1.0.0",
                ast_hash,
                "N0",
                "success",
            ),
        )
        for node_id, ntype, op, pr, fu, ordinal in (
            ("N0", "operator", "OP_EMA", None, None, 0),
            ("N1", "primitive", None, "PR_SPOT", None, 1),
            ("N2", "literal", None, None, None, 2),
        ):
            conn.execute(
                """
                INSERT OR IGNORE INTO ast_nodes(
                    transformation_uuid, ast_hash, node_id, node_type,
                    parent_node_id, param_name, operator_id, primitive_id,
                    feature_uuid, literal_json, child_node_ids_json,
                    subtree_hash, ordinal
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    tr,
                    ast_hash,
                    node_id,
                    ntype,
                    None,
                    None,
                    op,
                    pr,
                    fu,
                    None,
                    "[]",
                    "e" * 64,
                    ordinal,
                ),
            )
        conn.execute(
            """
            INSERT OR IGNORE INTO feature_ast(
                feature_uuid, transformation_uuid, compilation_uuid,
                ast_schema_version, ast_json, ast_fingerprint, subtree_hash
            ) VALUES (?,?,?,?,?,?,?)
            """,
            (
                feat,
                tr,
                comp,
                "1.0",
                json.dumps({"nodes": []}),
                ast_hash,
                "f" * 64,
            ),
        )
        conn.commit()
        conn.close()

    def test_derive_emits_edges_without_ast_copy(self) -> None:
        result = derive_lineage(self.store, include_closure=True)
        self.assertGreaterEqual(result.upserted, 3)
        edges = self.store.list_edges()
        triples = {(e.parent_object, e.child_object, e.relationship_id) for e in edges}
        self.assertIn(("PR_SPOT", self.tr, REL_INPUT_TO), triples)
        self.assertIn(("OP_EMA", self.tr, REL_USES), triples)
        self.assertIn((self.tr, self.feat, REL_GENERATED_BY), triples)
        self.assertIn(("PR_SPOT", self.feat, REL_DERIVED_FROM), triples)
        # No AST columns / tables in lineage
        import sqlite3

        conn = sqlite3.connect(str(self.db))
        cols = {
            r[1] for r in conn.execute("PRAGMA table_info(lineage_edges)").fetchall()
        }
        conn.close()
        self.assertNotIn("ast_json", cols)
        self.assertNotIn("ast_hash", cols)
        self.assertNotIn("manifest", cols)
        # Idempotent
        again = derive_lineage(self.store, include_closure=True)
        self.assertEqual(self.store.count_edges(), len(edges))
        self.assertEqual(again.upserted, len(edges))

    def test_derive_does_not_write_stats(self) -> None:
        svc = LineageService(self.db)
        before = svc.store.count_statistics()
        svc.derive_lineage()
        self.assertEqual(svc.store.count_statistics(), before)

    def test_collect_no_literal_edges(self) -> None:
        triples, _ = collect_derive_triples(self.store, include_closure=False)
        for p, c, r in triples:
            self.assertTrue(
                p.startswith(("PR_", "OP_", "TR_", "FEAT_")),
                msg=f"bad parent {p}",
            )


if __name__ == "__main__":
    unittest.main()

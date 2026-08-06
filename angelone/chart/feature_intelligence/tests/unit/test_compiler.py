"""Compiler / AST unit tests (Sprint 5)."""

from __future__ import annotations

import hashlib
import sqlite3
import tempfile
import unittest
import uuid
from pathlib import Path

from feature_intelligence.ast.hashing import compute_subtree_hash, hash_tree
from feature_intelligence.ast.nodes import AST_SCHEMA_VERSION, AstNode, LiteralPayload, assign_node_ids
from feature_intelligence.compiler import (
    COMPILER_VERSION,
    compile,
    decompile,
    derive_transformation_uuid,
    expression_hash,
    parse,
)
from feature_intelligence.compiler.cache import CompilerCache
from feature_intelligence.compiler.store import CompilerStore
from feature_intelligence.grammar.formatter import format_expression
from feature_intelligence.migrations.runner import MigrationRunner
from feature_intelligence.registry.feature_ids import (
    COMPILATION_UUID_PATTERN,
    TRANSFORM_UUID_PATTERN,
    generate_compilation_uuid,
    normalize_compilation_uuid,
    normalize_transformation_uuid,
)


EMA = "OP_EMA(period=20)"
RATIO = "OP_RATIO(left=PR_SPOT, right=PR_SPOT)"
NESTED = "OP_RATIO(left=OP_EMA(period=20), right=PR_SPOT)"


class TestTransformationIds(unittest.TestCase):
    def test_deterministic_tr(self) -> None:
        canonical = format_expression(EMA)
        a = derive_transformation_uuid(canonical)
        b = derive_transformation_uuid(canonical)
        self.assertEqual(a, b)
        self.assertRegex(a, TRANSFORM_UUID_PATTERN.pattern)
        digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        self.assertEqual(a, "TR_" + digest[:32].upper())
        self.assertEqual(expression_hash(canonical), digest)

    def test_tr_normalize_shape_only_no_v7(self) -> None:
        # Hash-derived body need not be UUIDv7
        raw = "tr_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
        self.assertEqual(
            normalize_transformation_uuid(raw),
            "TR_AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
        )

    def test_comp_uuid_v7(self) -> None:
        cid = generate_compilation_uuid()
        self.assertRegex(cid, COMPILATION_UUID_PATTERN.pattern)
        body = cid.removeprefix("COMP_")
        self.assertEqual(uuid.UUID(hex=body).version, 7)
        self.assertEqual(normalize_compilation_uuid(cid.lower()), cid)


class TestAstHashing(unittest.TestCase):
    def test_subtree_and_envelope(self) -> None:
        lit = AstNode(
            node_id="",
            node_type="literal",
            literal=LiteralPayload(kind="int", value=20),
            param_name="period",
        )
        root = AstNode(
            node_id="",
            node_type="operator",
            operator_id="OP_EMA",
            children=[lit],
        )
        assign_node_ids(root)
        self.assertEqual(root.node_id, "N0")
        self.assertEqual(root.children[0].node_id, "N1")
        compute_subtree_hash(root)
        self.assertEqual(len(root.subtree_hash), 64)
        self.assertEqual(len(root.children[0].subtree_hash), 64)
        ast_h = hash_tree(
            root,
            grammar_version="1.0",
            compiler_version="1.0.0",
            operator_pack_version="1.0.0",
        )
        expected = hashlib.sha256(
            f"1.0|1.0.0|1.0.0|{root.subtree_hash}".encode("utf-8")
        ).hexdigest()
        self.assertEqual(ast_h, expected)
        self.assertIsNone(root.stable_node_hash)


class TestCompilerPipeline(unittest.TestCase):
    def _db(self) -> Path:
        tmp = Path(tempfile.mkdtemp()) / "feature_intelligence.db"
        MigrationRunner(tmp).upgrade()
        return tmp

    def test_parse_and_compile_ema(self) -> None:
        db = self._db()
        cache = CompilerCache()
        parsed = parse(EMA, mode="bound", db=db)
        self.assertTrue(parsed.ok, parsed.report.failed_rules)
        assert parsed.transformation is not None
        self.assertEqual(parsed.transformation.ast_schema_version, AST_SCHEMA_VERSION)
        self.assertEqual(parsed.transformation.compiler_version, COMPILER_VERSION)
        self.assertIsNone(parsed.transformation.compilation_uuid)

        result = compile(
            EMA, mode="bound", db=db, persist=False, cache=cache
        )
        self.assertTrue(result.ok, result.report.failed_rules)
        assert result.transformation is not None
        obj = result.transformation
        self.assertRegex(obj.transformation_uuid, r"^TR_[0-9A-F]{32}$")
        self.assertRegex(obj.compilation_uuid or "", r"^COMP_[0-9A-F]{32}$")
        self.assertEqual(obj.ast_schema_version, "1.0")
        self.assertFalse(obj.cache_hit)

    def test_roundtrip(self) -> None:
        db = self._db()
        result = compile(NESTED, mode="bound", db=db, persist=False)
        self.assertTrue(result.ok, result.report.failed_rules)
        assert result.transformation is not None
        canonical = result.transformation.canonical_text
        again = format_expression(decompile(result.transformation.root), db_path=db)
        self.assertEqual(again, canonical)

    def test_determinism_1000(self) -> None:
        db = self._db()
        cache = CompilerCache()
        first = compile(
            RATIO, mode="bound", db=db, persist=False, cache=cache,
            record_cache_hit_event=False,
        )
        self.assertTrue(first.ok, first.report.failed_rules)
        assert first.transformation is not None
        tr = first.transformation.transformation_uuid
        eh = first.transformation.expression_hash
        ah = first.transformation.ast_hash
        for _ in range(999):
            r = compile(
                RATIO,
                mode="bound",
                db=db,
                persist=False,
                cache=cache,
                record_cache_hit_event=False,
            )
            self.assertTrue(r.ok)
            assert r.transformation is not None
            self.assertEqual(r.transformation.transformation_uuid, tr)
            self.assertEqual(r.transformation.expression_hash, eh)
            self.assertEqual(r.transformation.ast_hash, ah)

    def test_cache_hit(self) -> None:
        db = self._db()
        cache = CompilerCache()
        a = compile(EMA, mode="bound", db=db, persist=True, cache=cache)
        self.assertTrue(a.ok, a.report.failed_rules)
        assert a.transformation is not None
        self.assertFalse(a.transformation.cache_hit)
        b = compile(EMA, mode="bound", db=db, persist=True, cache=cache)
        self.assertTrue(b.ok, b.report.failed_rules)
        assert b.transformation is not None
        self.assertTrue(b.transformation.cache_hit)
        self.assertEqual(
            a.transformation.transformation_uuid,
            b.transformation.transformation_uuid,
        )
        self.assertNotEqual(
            a.transformation.compilation_uuid,
            b.transformation.compilation_uuid,
        )
        store = CompilerStore(db)
        comps = []
        conn = sqlite3.connect(str(db))
        try:
            comps = conn.execute(
                "SELECT cache_hit FROM compilation_registry ORDER BY compiled_at"
            ).fetchall()
            node_count = conn.execute("SELECT COUNT(*) FROM ast_nodes").fetchone()[0]
        finally:
            conn.close()
        self.assertGreaterEqual(len(comps), 2)
        self.assertEqual(comps[-1][0], 1)
        # Single AST store — nodes not duplicated on cache hit
        self.assertGreater(node_count, 0)
        self.assertFalse(store.compilation_has_compiled_json_column())

    def test_reject_source_param(self) -> None:
        db = self._db()
        result = compile(
            "OP_EMA(source=PR_SPOT, period=20)",
            mode="bound",
            db=db,
            persist=False,
        )
        self.assertFalse(result.ok)
        self.assertTrue(
            any(r.startswith("UNKNOWN_PARAM") for r in result.report.failed_rules),
            result.report.failed_rules,
        )
        # Must not persist TR_
        conn = sqlite3.connect(str(db))
        try:
            n = conn.execute("SELECT COUNT(*) FROM transformation_registry").fetchone()[0]
        finally:
            conn.close()
        self.assertEqual(n, 0)

    def test_single_ast_storage(self) -> None:
        db = self._db()
        result = compile(EMA, mode="bound", db=db, persist=True)
        self.assertTrue(result.ok, result.report.failed_rules)
        conn = sqlite3.connect(str(db))
        try:
            cols = {
                r[1]
                for r in conn.execute(
                    "PRAGMA table_info(compilation_registry)"
                ).fetchall()
            }
            self.assertNotIn("compiled_json", cols)
            nodes = conn.execute("SELECT COUNT(*) FROM ast_nodes").fetchone()[0]
            self.assertGreater(nodes, 0)
        finally:
            conn.close()

    def test_bound_failure_before_persist(self) -> None:
        db = self._db()
        result = compile("OP_NOTREAL(period=20)", mode="bound", db=db, persist=True)
        self.assertFalse(result.ok)
        conn = sqlite3.connect(str(db))
        try:
            n = conn.execute("SELECT COUNT(*) FROM transformation_registry").fetchone()[0]
        finally:
            conn.close()
        self.assertEqual(n, 0)


if __name__ == "__main__":
    unittest.main()

"""Unit tests for the migration runner."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from feature_intelligence.migrations.runner import MigrationRunner


class TestMigrations(unittest.TestCase):
    def test_upgrade_and_downgrade(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "feature_intelligence.db"
            runner = MigrationRunner(db)
            applied = runner.upgrade()
            self.assertEqual(
                applied,
                ["0001", "0002", "0003", "0004", "0005", "0006", "0007", "0008", "0009"],
            )
            self.assertEqual(runner.current_version(), "0009")

            # Idempotent
            self.assertEqual(runner.upgrade(), [])

            # Smoke + primitive + feature + operator + grammar + AST tables exist
            import sqlite3

            from feature_intelligence.grammar.pack import (
                EXPECTED_GRAMMAR_CHECKSUM,
                GRAMMAR_VERSION,
            )

            conn = sqlite3.connect(str(db))
            try:
                row = conn.execute(
                    "SELECT note FROM fic_infra_ping WHERE id = 1"
                ).fetchone()
                self.assertEqual(row[0], "sprint0-ok")
                count = conn.execute(
                    "SELECT COUNT(*) FROM primitive_registry"
                ).fetchone()[0]
                self.assertEqual(count, 14)
                feat_table = conn.execute(
                    "SELECT 1 FROM sqlite_master WHERE type='table' AND name='feature_registry'"
                ).fetchone()
                self.assertIsNotNone(feat_table)
                op_count = conn.execute(
                    "SELECT COUNT(*) FROM operator_registry"
                ).fetchone()[0]
                self.assertEqual(op_count, 31)
                gram = conn.execute(
                    "SELECT grammar_version, checksum FROM grammar_registry"
                ).fetchone()
                self.assertIsNotNone(gram)
                self.assertEqual(gram[0], GRAMMAR_VERSION)
                self.assertEqual(gram[1], EXPECTED_GRAMMAR_CHECKSUM)
                for table in (
                    "transformation_registry",
                    "compilation_registry",
                    "ast_nodes",
                    "feature_ast",
                    "compiler_statistics",
                    "vocabulary_registry",
                    "ontology_registry",
                    "primitive_ontology",
                    "operator_ontology",
                    "transformation_ontology",
                    "feature_ontology",
                    "ontology_statistics",
                    "lineage_registry",
                    "lineage_relationship_registry",
                    "lineage_edges",
                    "lineage_statistics",
                    "relationship_statistics",
                    "research_registry",
                    "feature_research_record",
                    "research_statistics",
                ):
                    present = conn.execute(
                        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
                        (table,),
                    ).fetchone()
                    self.assertIsNotNone(present, table)
                self.assertEqual(
                    conn.execute(
                        "SELECT COUNT(*) FROM lineage_relationship_registry"
                    ).fetchone()[0],
                    5,
                )
                self.assertEqual(
                    conn.execute("SELECT COUNT(*) FROM lineage_edges").fetchone()[0],
                    0,
                )
                self.assertEqual(
                    conn.execute(
                        "SELECT COUNT(*) FROM feature_research_record"
                    ).fetchone()[0],
                    0,
                )
                self.assertEqual(
                    conn.execute(
                        "SELECT research_version, schema_version FROM research_registry"
                    ).fetchone(),
                    ("1.0.0", "1.0"),
                )
                cols = {
                    r[1]
                    for r in conn.execute(
                        "PRAGMA table_info(compilation_registry)"
                    ).fetchall()
                }
                self.assertNotIn("compiled_json", cols)
                self.assertEqual(
                    conn.execute(
                        "SELECT COUNT(*) FROM vocabulary_registry"
                    ).fetchone()[0],
                    64,
                )
                self.assertEqual(
                    conn.execute(
                        "SELECT COUNT(*) FROM primitive_ontology"
                    ).fetchone()[0],
                    14,
                )
                self.assertEqual(
                    conn.execute(
                        "SELECT COUNT(*) FROM operator_ontology"
                    ).fetchone()[0],
                    31,
                )
            finally:
                conn.close()

            rolled = runner.downgrade(steps=9)
            self.assertEqual(
                rolled,
                [
                    "0009",
                    "0008",
                    "0007",
                    "0006",
                    "0005",
                    "0004",
                    "0003",
                    "0002",
                    "0001",
                ],
            )
            self.assertIsNone(runner.current_version())
            history = runner.history()
            actions = [h["action"] for h in history]
            self.assertIn("upgrade", actions)
            self.assertIn("downgrade", actions)


if __name__ == "__main__":
    unittest.main()

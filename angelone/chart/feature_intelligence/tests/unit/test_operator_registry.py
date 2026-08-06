"""Unit tests for Operator Registry (Sprint 3)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from feature_intelligence.core.config import (
    DatabaseConfig,
    FeatureIntelligenceConfig,
    FicConfig,
    LoggingConfig,
)
from feature_intelligence.core.database import init_database
from feature_intelligence.operators.catalog import (
    EXPECTED_OPERATOR_CATALOG_HASH,
    SEED_OPERATORS,
    compute_operator_catalog_hash,
    write_catalog_artifacts,
)
from feature_intelligence.operators.operator_import_export import (
    export_operators,
    import_operators,
)
from feature_intelligence.operators.operator_service import OperatorRegistryService


class TestOperatorRegistry(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        write_catalog_artifacts()

    def test_seed_hash_and_count(self) -> None:
        self.assertEqual(len(SEED_OPERATORS), 31)
        self.assertEqual(compute_operator_catalog_hash(), EXPECTED_OPERATOR_CATALOG_HASH)

    def test_migrate_validate_lookup(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "feature_intelligence.db"
            cfg = FicConfig(
                feature_intelligence=FeatureIntelligenceConfig(
                    schema_version="0.0.1",
                    environment="test",
                    data_dir=Path(tmp),
                ),
                database=DatabaseConfig(path=db),
                logging=LoggingConfig(level="WARNING"),
            )
            init_database(cfg, apply_migrations=True)
            svc = OperatorRegistryService(db)
            self.assertEqual(len(svc.list_operators()), 31)
            ema = svc.get_by_id("OP_EMA")
            self.assertEqual(ema.canonical_name, "ema")
            self.assertEqual(ema.warmup_policy, "WINDOW")
            self.assertTrue(ema.deterministic)
            self.assertIsNone(ema.depends_on_operator_ids)
            report = svc.validate_registry()
            self.assertTrue(report.passed, report.failed_rules)
            self.assertEqual(report.validated_objects, "31 operators")

    def test_export_import_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "a.db"
            cfg = FicConfig(
                feature_intelligence=FeatureIntelligenceConfig(
                    schema_version="0.0.1",
                    environment="test",
                    data_dir=Path(tmp),
                ),
                database=DatabaseConfig(path=db),
                logging=LoggingConfig(level="WARNING"),
            )
            init_database(cfg, apply_migrations=True)
            svc = OperatorRegistryService(db)
            out = Path(tmp) / "ops.json"
            export_operators(svc, out, fmt="json")

            db2 = Path(tmp) / "b.db"
            cfg2 = FicConfig(
                feature_intelligence=FeatureIntelligenceConfig(
                    schema_version="0.0.1",
                    environment="test",
                    data_dir=Path(tmp),
                ),
                database=DatabaseConfig(path=db2),
                logging=LoggingConfig(level="WARNING"),
            )
            init_database(cfg2, apply_migrations=True)
            # Fresh DB already seeded — import should upsert metadata idempotently
            svc2 = OperatorRegistryService(db2)
            ids = import_operators(svc2, out, fmt="json")
            self.assertEqual(len(ids), 31)
            self.assertEqual(svc2.get_by_name("ema").operator_id, "OP_EMA")


if __name__ == "__main__":
    unittest.main()

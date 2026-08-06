"""Integration: migrate + seed + validate primitives."""

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
from feature_intelligence.migrations.runner import MigrationRunner
from feature_intelligence.registry import (
    EXPECTED_SEED_CATALOG_HASH,
    PrimitiveCatalogService,
    validate_primitives,
)


class TestPrimitiveRegistryIntegration(unittest.TestCase):
    def test_init_seeds_and_validates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "feature_intelligence.db"
            cfg = FicConfig(
                feature_intelligence=FeatureIntelligenceConfig(
                    schema_version="0.0.1",
                    environment="test",
                    data_dir=Path(tmp),
                ),
                database=DatabaseConfig(path=db_path),
                logging=LoggingConfig(level="WARNING"),
            )
            init_database(cfg, apply_migrations=True)
            runner = MigrationRunner(db_path)
            self.assertEqual(runner.current_version(), "0009")

            report = validate_primitives(db_path)
            self.assertTrue(report.passed, report.failed_rules)
            self.assertEqual(report.seed_hash, EXPECTED_SEED_CATALOG_HASH)
            self.assertEqual(report.expected_seed_hash, EXPECTED_SEED_CATALOG_HASH)
            self.assertTrue(report.timestamp)
            self.assertEqual(report.failed_rules, [])

            svc = PrimitiveCatalogService(db_path)
            self.assertEqual(len(svc.list_primitives()), 14)


if __name__ == "__main__":
    unittest.main()

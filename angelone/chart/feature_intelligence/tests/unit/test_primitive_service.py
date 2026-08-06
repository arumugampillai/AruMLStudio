"""Unit tests for PrimitiveCatalogService (uses temp DB)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from feature_intelligence.core.database import init_database
from feature_intelligence.core.config import (
    DatabaseConfig,
    FeatureIntelligenceConfig,
    FicConfig,
    LoggingConfig,
)
from feature_intelligence.registry.service import (
    PrimitiveCatalogService,
    PrimitiveNotFoundError,
)


class TestPrimitiveService(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.db_path = Path(self._tmp.name) / "feature_intelligence.db"
        cfg = FicConfig(
            feature_intelligence=FeatureIntelligenceConfig(
                schema_version="0.0.1",
                environment="test",
                data_dir=Path(self._tmp.name),
            ),
            database=DatabaseConfig(path=self.db_path),
            logging=LoggingConfig(level="WARNING"),
        )
        init_database(cfg, apply_migrations=True)
        self.svc = PrimitiveCatalogService(self.db_path)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_list_and_get(self) -> None:
        rows = self.svc.list_primitives()
        self.assertEqual(len(rows), 14)
        self.assertEqual(rows[0].primitive_id, "PR_ASK")  # ASC order
        spot = self.svc.get_primitive("PR_SPOT")
        self.assertEqual(spot.name, "Spot")
        self.assertEqual(spot.catalog_version, "1.0")
        self.assertTrue(self.svc.primitive_exists("PR_OI"))
        self.assertFalse(self.svc.primitive_exists("PR_NOPE"))
        by_name = self.svc.get_primitive_by_name("Delta")
        self.assertEqual(by_name.primitive_id, "PR_DELTA")

    def test_missing_raises(self) -> None:
        with self.assertRaises(PrimitiveNotFoundError):
            self.svc.get_primitive("PR_MISSING")

    def test_update_description(self) -> None:
        updated = self.svc.update_primitive_metadata(
            "PR_SPOT",
            description="Spot price (updated doc)",
        )
        self.assertEqual(updated.description, "Spot price (updated doc)")
        report = self.svc.validate_primitives()
        self.assertTrue(report.passed, report.failed_rules)


if __name__ == "__main__":
    unittest.main()

"""Validation suite: seed completeness and ValidationReport shape."""

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
from feature_intelligence.registry.catalog import SEED_PRIMITIVES
from feature_intelligence.registry.validation import validate_primitives


class TestPrimitiveValidation(unittest.TestCase):
    def test_report_fields_and_seed_count(self) -> None:
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
            report = validate_primitives(db_path)
            payload = report.to_dict()
            self.assertIn("passed", payload)
            self.assertIn("failed_rules", payload)
            self.assertIn("warnings", payload)
            self.assertIn("seed_hash", payload)
            self.assertIn("expected_seed_hash", payload)
            self.assertIn("timestamp", payload)
            self.assertIn("validated_objects", payload)
            self.assertTrue(report.passed)
            self.assertEqual(report.validated_objects, "14 primitives")
            self.assertEqual(len(SEED_PRIMITIVES), 14)


if __name__ == "__main__":
    unittest.main()

"""Integration: database initialization end-to-end."""

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


class TestDbInit(unittest.TestCase):
    def test_init_database_creates_file_and_migrates(self) -> None:
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
            out = init_database(cfg, apply_migrations=True)
            self.assertEqual(out, db_path)
            self.assertTrue(db_path.is_file())
            runner = MigrationRunner(db_path)
            self.assertEqual(runner.current_version(), "0009")
            # Ensure no open handles before TemporaryDirectory cleanup (Windows).
            del runner


if __name__ == "__main__":
    unittest.main()

"""Unit tests for configuration loading."""

from __future__ import annotations

import unittest
from pathlib import Path

from feature_intelligence.core.config import load_config
from feature_intelligence.core.paths import CONFIG_DIR


class TestConfig(unittest.TestCase):
    def test_load_default_package_config(self) -> None:
        cfg = load_config(CONFIG_DIR)
        self.assertEqual(cfg.feature_intelligence.schema_version, "0.0.1")
        self.assertEqual(cfg.logging.level, "INFO")
        self.assertGreater(cfg.database.timeout_seconds, 0)
        self.assertTrue(str(cfg.database.path).endswith("feature_intelligence.db"))

    def test_load_override_dir(self) -> None:
        import tempfile

        from feature_intelligence.core import _yaml_lite

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "feature_intelligence.yaml").write_text(
                "schema_version: '9.9.9'\nenvironment: test\n",
                encoding="utf-8",
            )
            (root / "database.yaml").write_text(
                f"path: {root.as_posix()}/custom.db\ntimeout_seconds: 5\n",
                encoding="utf-8",
            )
            (root / "logging.yaml").write_text("level: DEBUG\n", encoding="utf-8")
            # sanity: lite parser works
            self.assertEqual(_yaml_lite.loads("level: DEBUG\n")["level"], "DEBUG")
            cfg = load_config(root)
            self.assertEqual(cfg.feature_intelligence.schema_version, "9.9.9")
            self.assertEqual(cfg.logging.level, "DEBUG")
            self.assertEqual(cfg.database.path, root / "custom.db")


if __name__ == "__main__":
    unittest.main()

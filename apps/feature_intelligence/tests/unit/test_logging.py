"""Unit tests for logging setup."""

from __future__ import annotations

import logging
import unittest

from feature_intelligence.core.config import LoggingConfig
from feature_intelligence.core.logging import get_logger, setup_logging


class TestLogging(unittest.TestCase):
    def test_setup_and_levels(self) -> None:
        logger = setup_logging(LoggingConfig(level="DEBUG"))
        self.assertEqual(logger.level, logging.DEBUG)
        child = get_logger("unit")
        self.assertTrue(child.name.endswith("unit"))
        # Ensure standard levels are recognized
        for level_name in ("INFO", "WARNING", "ERROR", "DEBUG"):
            setup_logging(LoggingConfig(level=level_name))
            root = get_logger()
            self.assertEqual(root.level, getattr(logging, level_name))


if __name__ == "__main__":
    unittest.main()

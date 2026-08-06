"""Validation harness smoke — proves test discovery for validation suite."""

from __future__ import annotations

import unittest

from feature_intelligence import __version__


class TestValidationHarnessSmoke(unittest.TestCase):
    def test_package_version_present(self) -> None:
        self.assertTrue(__version__)


if __name__ == "__main__":
    unittest.main()

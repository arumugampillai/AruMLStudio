"""Tests for source file navigation from feature detail."""

from __future__ import annotations

import unittest
from pathlib import Path

from master_dataset_tk.source_navigation import (
    chart_package_root,
    find_source_line,
    resolve_module_file,
    resolve_source_location,
)


class SourceNavigationTests(unittest.TestCase):
    def test_resolve_chain_maps(self) -> None:
        path = resolve_module_file("chain_replay_ml/dataset_builder/chain_maps.py")
        self.assertIsNotNone(path)
        assert path is not None
        self.assertTrue(path.is_file())
        self.assertEqual(path.parent.name, "dataset_builder")

    def test_find_feature_line(self) -> None:
        path = resolve_module_file("chain_replay_ml/dataset_builder/chain_maps.py")
        self.assertIsNotNone(path)
        assert path is not None
        line = find_source_line(
            path,
            feature_name="ce_pe_atm6_ltp_ratio",
            function_ref="chain_features_at()",
        )
        self.assertGreaterEqual(line, 250)
        text = path.read_text(encoding="utf-8").splitlines()[line - 1]
        self.assertIn("ce_pe_atm6_ltp_ratio", text)

    def test_resolve_source_location(self) -> None:
        loc = resolve_source_location(
            feature_name="ce_pe_atm6_ltp_ratio",
            module_path="chain_replay_ml/dataset_builder/chain_maps.py",
            function_ref="chain_features_at()",
        )
        self.assertTrue(loc.get("ok"))
        self.assertIn("chain_maps.py", str(loc.get("path")))
        self.assertGreater(int(loc.get("line") or 0), 0)

    def test_chart_package_root(self) -> None:
        root = chart_package_root()
        self.assertTrue((root / "chain_replay_ml").is_dir())


if __name__ == "__main__":
    unittest.main()

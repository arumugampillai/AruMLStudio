"""Automated enforcement of AruMLStudio architecture, dependency, and storage boundaries."""

from __future__ import annotations

import ast
import os
import sys
import tempfile
import unittest
from pathlib import Path

_apps_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _apps_dir not in sys.path:
    sys.path.insert(0, _apps_dir)
from path_config import ensure_ml_studio_paths

ensure_ml_studio_paths()

import __version__


class TestArchitectureBoundaries(unittest.TestCase):
    def setUp(self) -> None:
        self.apps_path = Path(_apps_dir)

    def test_no_aruneo_or_angelone_imports_in_production_code(self) -> None:
        """AST audit to guarantee zero forbidden imports in apps/."""
        forbidden_roots = {"angelone", "aruneo", "AruNeo"}
        violations = []

        for py_path in self.apps_path.rglob("*.py"):
            # Exclude tests folder from AST import check if testing legacy compatibility
            rel_parts = py_path.relative_to(self.apps_path).parts
            if "tests" in rel_parts:
                continue

            try:
                tree = ast.parse(py_path.read_text(encoding="utf-8"), filename=str(py_path))
            except Exception:
                continue

            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        top = alias.name.split(".")[0]
                        if top in forbidden_roots:
                            violations.append(f"{py_path.name}:{node.lineno} -> import {alias.name}")
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        top = node.module.split(".")[0]
                        if top in forbidden_roots:
                            # Storage broker adapter has optional lazy import behind try-except
                            if "storage" in rel_parts and py_path.name == "angel_historic_fetch.py":
                                continue
                            violations.append(f"{py_path.name}:{node.lineno} -> from {node.module} import ...")

        self.assertEqual(violations, [], f"Forbidden legacy imports discovered:\n" + "\n".join(violations))

    def test_no_developer_specific_absolute_paths_in_production_code(self) -> None:
        """Scan production files for forbidden developer-specific hardcoded paths."""
        forbidden_patterns = [
            r"C:\Users\admin",
            r"C:/Users/admin",
            r"PycharmProjects\v1\AruNeo",
            r"PycharmProjects/v1/AruNeo",
        ]
        violations = []

        for py_path in self.apps_path.rglob("*.py"):
            rel_parts = py_path.relative_to(self.apps_path).parts
            if "tests" in rel_parts:
                continue

            content = py_path.read_text(encoding="utf-8", errors="ignore")
            for pat in forbidden_patterns:
                if pat.lower() in content.lower():
                    violations.append(f"{py_path.name} contains forbidden path: {pat}")

        self.assertEqual(violations, [], "Forbidden hardcoded developer paths found:\n" + "\n".join(violations))

    def test_sys_path_safety_under_data_dir_pollution_simulation(self) -> None:
        """Simulate an external chart data folder and verify apps/ remains sys.path[0]."""
        with tempfile.TemporaryDirectory() as fake_data_dir:
            # Create a fake master_dataset_tk module inside the data dir
            fake_pkg = os.path.join(fake_data_dir, "master_dataset_tk")
            os.makedirs(fake_pkg, exist_ok=True)
            with open(os.path.join(fake_pkg, "__init__.py"), "w") as fh:
                fh.write("__fake__ = True\n")

            # Try to hijack sys.path by prepending fake_data_dir
            sys.path.insert(0, fake_data_dir)
            
            # Enforce AruMLStudio path ordering
            ensure_ml_studio_paths()

            # Verify apps/ is strictly index 0
            self.assertEqual(sys.path[0], _apps_dir)

            # Verify import resolves the authentic AruMLStudio module, not fake one
            import master_dataset_tk
            self.assertFalse(hasattr(master_dataset_tk, "__fake__"))

            # Cleanup
            while fake_data_dir in sys.path:
                sys.path.remove(fake_data_dir)

    def test_appdata_isolation_and_no_unwanted_writes_to_aruneo(self) -> None:
        """Verify normal runtime operations write only to %APPDATA%/AruMLStudio."""
        with tempfile.TemporaryDirectory() as tmp_appdata:
            old_appdata = os.environ.get("APPDATA")
            os.environ["APPDATA"] = tmp_appdata
            try:
                from master_dataset_tk.project_config import save_project_config, load_project_config
                from master_dataset_tk.ui_state import UIStateManager

                # Perform writes
                save_project_config("D:/TestData/chart")
                ui_mgr = UIStateManager()
                ui_mgr.set("window_geometry", "1400x900")
                ui_mgr.flush()

                # Verify AruMLStudio state directory exists
                self.assertTrue(os.path.isdir(os.path.join(tmp_appdata, "AruMLStudio")))
                # Verify legacy AruNeo directory was NEVER created or written to
                self.assertFalse(os.path.exists(os.path.join(tmp_appdata, "AruNeo")))
            finally:
                if old_appdata is None:
                    os.environ.pop("APPDATA", None)
                else:
                    os.environ["APPDATA"] = old_appdata

    def test_canonical_version_unification(self) -> None:
        """Verify canonical version is defined once in __version__.py and exposed across packages."""
        v = __version__.__version__
        self.assertTrue(bool(v))

        import master_dataset_tk
        import chain_replay_ml

        self.assertEqual(master_dataset_tk.__version__, v)
        self.assertEqual(chain_replay_ml.__version__, v)

    def test_layered_dependency_rules_no_ui_imports_in_engine(self) -> None:
        """Verify core dataset builder & training modules never import master_dataset_tk UI classes."""
        forbidden_in_engine = ["master_dataset_tk.app", "master_dataset_tk.create_dataset_panel"]
        violations = []

        engine_dir = self.apps_path / "chain_replay_ml" / "dataset_builder"
        for py_path in engine_dir.rglob("*.py"):
            try:
                tree = ast.parse(py_path.read_text(encoding="utf-8"), filename=str(py_path))
            except Exception:
                continue

            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module:
                    for fb in forbidden_in_engine:
                        if fb in node.module:
                            violations.append(f"{py_path.name} imports UI module: {node.module}")

        self.assertEqual(violations, [], "Engine modules illegally imported UI layer:\n" + "\n".join(violations))


if __name__ == "__main__":
    unittest.main()

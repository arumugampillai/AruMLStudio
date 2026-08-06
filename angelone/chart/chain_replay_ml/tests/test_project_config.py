"""Tests for ML Research Studio project folder resolution."""

from __future__ import annotations

import os
import tempfile
import unittest

from master_dataset_tk.project_config import (
    ensure_project_data_dir,
    resolve_chart_dir_from_selection,
    save_project_config,
    load_project_config,
)


class ProjectConfigTests(unittest.TestCase):
    def test_resolve_chart_dir_direct(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            os.makedirs(os.path.join(tmp, "data"))
            self.assertEqual(resolve_chart_dir_from_selection(tmp), os.path.abspath(tmp))

    def test_resolve_chart_dir_repo_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            chart = os.path.join(tmp, "angelone", "chart")
            os.makedirs(os.path.join(chart, "data"))
            self.assertEqual(resolve_chart_dir_from_selection(tmp), os.path.abspath(chart))

    def test_ensure_data_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data = ensure_project_data_dir(tmp)
            self.assertTrue(os.path.isdir(data))

    def test_save_and_load_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            chart = os.path.join(tmp, "angelone", "chart")
            os.makedirs(os.path.join(chart, "data"))
            old = os.environ.get("APPDATA")
            os.environ["APPDATA"] = tmp
            try:
                save_project_config(chart)
                loaded = load_project_config()
                self.assertEqual(loaded.get("chart_dir"), os.path.abspath(chart))
            finally:
                if old is None:
                    os.environ.pop("APPDATA", None)
                else:
                    os.environ["APPDATA"] = old


if __name__ == "__main__":
    unittest.main()

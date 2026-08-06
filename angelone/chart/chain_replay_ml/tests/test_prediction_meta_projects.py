"""Tests for prediction project registry."""

from __future__ import annotations

import os
import tempfile
import unittest

from chain_replay_ml.prediction_meta.projects import (
    create_project,
    db_filename_from_display_name,
    delete_project,
    get_project,
    list_projects,
    slugify_project_name,
)


class TestProjectNaming(unittest.TestCase):
    def test_slugify(self) -> None:
        self.assertEqual(slugify_project_name("Prediction_11Models_v1"), "prediction_11models_v1")
        self.assertEqual(db_filename_from_display_name("Test_NewModel"), "test_newmodel.db")


class TestProjectRegistry(unittest.TestCase):
    def test_create_list_delete(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            create_project(
                tmp,
                display_name="Prediction_Top5",
                source_master_db="master_dataset_nifty_3s.db",
                market="NIFTY",
                sampling_interval_sec=3,
                selected_models=["Model_A", "Model_B"],
            )
            rows = list_projects(tmp)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["display_name"], "Prediction_Top5")
            self.assertEqual(rows[0]["models_count"], 2)
            pid = rows[0]["project_id"]
            proj = get_project(tmp, pid)
            assert proj is not None
            self.assertTrue(os.path.isfile(proj.db_path(tmp)))
            delete_project(tmp, pid)
            self.assertEqual(len(list_projects(tmp)), 0)


if __name__ == "__main__":
    unittest.main()

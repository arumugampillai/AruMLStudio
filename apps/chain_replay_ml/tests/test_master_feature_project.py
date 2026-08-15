"""Tests for master dataset ↔ feature project binding."""

from __future__ import annotations

import os
import tempfile
import unittest

from chain_replay_ml.dataset_builder.master_feature_project import (
    MasterFeatureProjectError,
    ensure_master_feature_project_id,
    validate_feature_project_id,
)
from chain_replay_ml.dataset_builder.master_store import MasterStore


class TestMasterFeatureProject(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmp, "master_test.db")

    def test_ensure_defaults_existing_db_to_all(self) -> None:
        store = MasterStore(self.db_path)
        store.open()
        try:
            pid = ensure_master_feature_project_id(store, self.tmp)
            self.assertEqual(pid, "all")
            self.assertEqual(store.read_master_meta_dict().get("feature_project_id"), "all")
        finally:
            store.close()

    def test_validate_rejects_unknown_project(self) -> None:
        with self.assertRaises(MasterFeatureProjectError):
            validate_feature_project_id(self.tmp, "not_a_real_project_xyz")


if __name__ == "__main__":
    unittest.main()

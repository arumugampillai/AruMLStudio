"""Tests for master Delete All fresh-start wipe."""

from __future__ import annotations

import os
import tempfile
import unittest

from chain_replay_ml.dataset_builder.master_status import (
    delete_master_database_files,
    related_master_artifact_paths,
)


def _write(path: str, data: bytes) -> None:
    with open(path, "wb") as f:
        f.write(data)


class DeleteMasterDatabaseFilesTests(unittest.TestCase):
    def test_related_paths_include_bak_and_wal(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = os.path.join(tmp, "master_dataset_nifty_3s.db")
            _write(db, b"main")
            _write(db + "-wal", b"wal")
            _write(db + "-shm", b"shm")
            bak = db + ".pre_rebuild_20260727_220115.bak"
            _write(bak, b"bak")
            other = os.path.join(tmp, "master_dataset_nifty_10s.db")
            _write(other, b"other")

            related = related_master_artifact_paths(db)
            bases = {os.path.basename(p) for p in related}
            self.assertIn("master_dataset_nifty_3s.db", bases)
            self.assertIn("master_dataset_nifty_3s.db-wal", bases)
            self.assertIn("master_dataset_nifty_3s.db-shm", bases)
            self.assertIn(os.path.basename(bak), bases)
            self.assertNotIn("master_dataset_nifty_10s.db", bases)

    def test_delete_all_removes_db_meta_sidecars_and_bak(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = os.path.join(tmp, "master_dataset_nifty_3s.db")
            _write(db, b"x" * 100)
            _write(db + "-wal", b"w" * 50)
            bak = db + ".pre_rebuild_20260727_220115.bak"
            _write(bak, b"b" * 200)
            other = os.path.join(tmp, "master_dataset_nifty_10s.db")
            _write(other, b"keep")

            result = delete_master_database_files(db)
            self.assertFalse(result.still_exists, msg=str(result.errors))
            self.assertFalse(os.path.isfile(db))
            self.assertFalse(os.path.isfile(db + "-wal"))
            self.assertFalse(os.path.isfile(bak))
            self.assertTrue(os.path.isfile(other))
            self.assertGreaterEqual(len(result.removed), 2)
            self.assertEqual(os.path.getsize(other), 4)


if __name__ == "__main__":
    unittest.main()

"""Tests for Create Model dataset catalog filtering."""

from __future__ import annotations

import json
import os
import tempfile
import unittest

from master_dataset_tk.model_builder.service import list_builder_datasets


class ListBuilderDatasetsTests(unittest.TestCase):
    def test_includes_metadata_only_registry_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = os.path.join(tmp, "datasets")
            os.makedirs(out_dir)
            name = "MS_239f_3s_0406"
            meta = {
                "row_count": 540488,
                "export_source": "master_filter_export",
                "master_db_path": "datasets/master_dataset_nifty_3s.db",
            }
            with open(os.path.join(out_dir, f"{name}.json"), "w", encoding="utf-8") as fh:
                json.dump(meta, fh)
            rows = list_builder_datasets(tmp)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["dataset_name"], name)
            self.assertTrue(rows[0]["needs_parquet"])

    def test_includes_parquet_ready_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = os.path.join(tmp, "datasets")
            os.makedirs(out_dir)
            name = "ready_ds"
            meta = {"row_count": 100}
            with open(os.path.join(out_dir, f"{name}.json"), "w", encoding="utf-8") as fh:
                json.dump(meta, fh)
            with open(os.path.join(out_dir, f"{name}.parquet"), "wb") as fh:
                fh.write(b"PAR1")
            rows = list_builder_datasets(tmp)
            self.assertEqual(len(rows), 1)
            self.assertFalse(rows[0]["needs_parquet"])


if __name__ == "__main__":
    unittest.main()

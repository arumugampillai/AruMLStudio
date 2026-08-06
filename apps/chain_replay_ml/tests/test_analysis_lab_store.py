"""Tests for Phase 2 analysis.db store."""

from __future__ import annotations

import os
import tempfile
import unittest

import pandas as pd

from chain_replay_ml.dataset_builder.analysis_lab_store import (
    STATUS_COMPLETED,
    STATUS_FAILED,
    STATUS_STALE,
    dependency_blockers,
    ensure_analysis_run,
    fingerprint_parquet,
    list_datasets,
    module_statuses,
    register_dataset,
    set_module_status,
)


class AnalysisLabStoreTests(unittest.TestCase):
    def _write_pq(self, folder: str, name: str, rows: int = 10, cols: int = 3) -> str:
        path = os.path.join(folder, f"{name}.parquet")
        data = {f"f{i}": list(range(rows)) for i in range(cols)}
        pd.DataFrame(data).to_parquet(path, index=False)
        return path

    def test_register_and_module_lifecycle(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            pq = self._write_pq(tmp, "Future_LTP_5m_v12", rows=20, cols=4)
            ds = register_dataset(tmp, pq, name="Future_LTP_5m_v12")
            self.assertEqual(ds["dataset_id"], "Future_LTP_5m_v12")
            self.assertEqual(int(ds["rows"]), 20)
            self.assertEqual(int(ds["features"]), 4)
            self.assertTrue(ds["dataset_hash"])

            run = ensure_analysis_run(tmp, "Future_LTP_5m_v12")
            run_id = run["run_id"]
            statuses = {m["module_id"]: m["status"] for m in module_statuses(tmp, run_id)}
            self.assertEqual(statuses["correlation"], "not_run")

            # HCA blocked until correlation completes
            self.assertEqual(
                dependency_blockers(tmp, run_id, "hca"),
                ["correlation"],
            )
            set_module_status(tmp, run_id, "correlation", STATUS_COMPLETED, message="ok")
            self.assertEqual(dependency_blockers(tmp, run_id, "hca"), [])

            listed = list_datasets(tmp)
            self.assertEqual(len(listed), 1)

            from chain_replay_ml.dataset_builder.analysis_lab_store import (
                get_selected_analysis_dataset,
                set_selected_analysis_dataset,
            )

            self.assertEqual(get_selected_analysis_dataset(tmp), "")
            set_selected_analysis_dataset(tmp, "Future_LTP_5m_v12")
            self.assertEqual(
                get_selected_analysis_dataset(tmp), "Future_LTP_5m_v12"
            )
            set_selected_analysis_dataset(tmp, "other_ds")
            self.assertEqual(get_selected_analysis_dataset(tmp), "other_ds")

    def test_fingerprint_change_marks_stale(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            pq = self._write_pq(tmp, "ds1", rows=5, cols=2)
            register_dataset(tmp, pq, name="ds1")
            run = ensure_analysis_run(tmp, "ds1")
            set_module_status(
                tmp, run["run_id"], "correlation", STATUS_COMPLETED, message="v1"
            )

            # Change dataset contents → new hash → prior completed modules stale
            pd.DataFrame({"f0": [1, 2, 3], "f1": [4, 5, 6], "f2": [7, 8, 9]}).to_parquet(
                pq, index=False
            )
            register_dataset(tmp, pq, name="ds1")
            statuses = {
                m["module_id"]: m["status"]
                for m in module_statuses(tmp, run["run_id"])
            }
            self.assertEqual(statuses["correlation"], STATUS_STALE)

            # New run for new hash
            run2 = ensure_analysis_run(tmp, "ds1")
            self.assertNotEqual(run["run_id"], run2["run_id"])
            self.assertNotEqual(run["dataset_hash"], run2["dataset_hash"])

    def test_fingerprint_stable_for_unchanged_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            pq = self._write_pq(tmp, "stable", rows=8, cols=3)
            a = fingerprint_parquet(pq)
            b = fingerprint_parquet(pq)
            self.assertEqual(a.dataset_hash, b.dataset_hash)
            self.assertEqual(a.columns_hash, b.columns_hash)

    def test_catalog_skips_refingerprint_when_registered(self) -> None:
        from chain_replay_ml.dataset_builder.analysis_lab_store import (
            load_analysis_dataset_catalog,
        )

        with tempfile.TemporaryDirectory() as tmp:
            ds_dir = os.path.join(tmp, "datasets")
            os.makedirs(ds_dir)
            pq = self._write_pq(ds_dir, "Future_LTP_5m_v12", rows=12, cols=3)
            register_dataset(
                tmp,
                pq,
                name="Future_LTP_5m_v12",
                relative_path="datasets/Future_LTP_5m_v12.parquet",
            )
            first = list_datasets(tmp)[0]
            catalog = load_analysis_dataset_catalog(tmp, force_rescan=False)
            self.assertEqual(len(catalog), 1)
            self.assertEqual(catalog[0]["dataset_hash"], first["dataset_hash"])
            # New file only — registers without force
            self._write_pq(ds_dir, "New_DS", rows=5, cols=2)
            catalog2 = load_analysis_dataset_catalog(tmp, force_rescan=False)
            ids = {str(d["dataset_id"]) for d in catalog2}
            self.assertEqual(ids, {"Future_LTP_5m_v12", "New_DS"})


if __name__ == "__main__":
    unittest.main()

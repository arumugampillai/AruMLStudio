"""Tests for Dataset Registry CSV export artifacts."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from unittest import mock

import pandas as pd

from chain_replay_ml.dataset_builder.dataset_csv_export import (
    CsvExportAlreadyExistsError,
    CsvExportError,
    build_csv_export_metadata,
    delete_dataset_csv_export,
    generate_dataset_csv_export,
)


def _write_dataset(
    data_dir: str,
    name: str,
    df: pd.DataFrame,
    *,
    dataset_version: str = "v1",
) -> tuple[str, str]:
    out_dir = os.path.join(data_dir, "datasets")
    os.makedirs(out_dir, exist_ok=True)
    parquet_path = os.path.join(out_dir, f"{name}.parquet")
    meta_path = os.path.join(out_dir, f"{name}.json")
    df.to_parquet(parquet_path, index=False)
    meta = {
        "row_count": len(df),
        "column_count": len(df.columns),
        "dataset_id": f"id_{name}",
        "dataset_version": dataset_version,
    }
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(meta, fh)
    return parquet_path, meta_path


class DatasetCsvExportTests(unittest.TestCase):
    def test_successful_generation_matches_source_shape(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
            parquet_path, meta_path = _write_dataset(tmp, "ds_csv", df)
            parquet_before = open(parquet_path, "rb").read()
            meta_before = open(meta_path, "rb").read()

            result = generate_dataset_csv_export(tmp, "ds_csv")
            self.assertEqual(result["status"], "Generated")
            self.assertEqual(result["row_count"], 3)
            self.assertEqual(result["column_count"], 2)
            self.assertTrue(os.path.isfile(result["csv_path"]))

            loaded = pd.read_csv(result["csv_path"])
            self.assertEqual(list(loaded.columns), ["a", "b"])
            self.assertEqual(len(loaded), 3)

            self.assertEqual(open(parquet_path, "rb").read(), parquet_before)
            self.assertEqual(open(meta_path, "rb").read(), meta_before)

            view = build_csv_export_metadata(tmp, "ds_csv")
            self.assertEqual(view["status"], "Generated")
            self.assertEqual(view["source_dataset"]["dataset_version"], "v1")

    def test_already_exists_without_replace(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            df = pd.DataFrame({"x": [1]})
            _write_dataset(tmp, "dup", df)
            generate_dataset_csv_export(tmp, "dup")
            with self.assertRaises(CsvExportAlreadyExistsError):
                generate_dataset_csv_export(tmp, "dup", replace=False)

    def test_regeneration_replaces_csv_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            df = pd.DataFrame({"x": [1, 2]})
            parquet_path, _ = _write_dataset(tmp, "regen", df)
            first = generate_dataset_csv_export(tmp, "regen")
            first_path = first["csv_path"]
            first_mtime = os.path.getmtime(first_path)
            parquet_bytes = open(parquet_path, "rb").read()

            second = generate_dataset_csv_export(tmp, "regen", replace=True)
            self.assertEqual(second["status"], "Generated")
            self.assertEqual(second["row_count"], 2)
            self.assertGreaterEqual(os.path.getmtime(first_path), first_mtime)
            self.assertEqual(open(parquet_path, "rb").read(), parquet_bytes)

    def test_delete_csv_leaves_parquet_and_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            df = pd.DataFrame({"n": [5]})
            parquet_path, meta_path = _write_dataset(tmp, "delcsv", df)
            generate_dataset_csv_export(tmp, "delcsv")
            out = delete_dataset_csv_export(tmp, "delcsv")
            self.assertEqual(out["csv_export"]["status"], "Not Generated")
            self.assertTrue(os.path.isfile(parquet_path))
            self.assertTrue(os.path.isfile(meta_path))
            csv_path = os.path.join(tmp, "datasets", "delcsv.csv")
            self.assertFalse(os.path.isfile(csv_path))

    def test_failed_first_export_leaves_no_valid_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            df = pd.DataFrame({"z": [1]})
            parquet_path, _ = _write_dataset(tmp, "fail", df)
            parquet_before = open(parquet_path, "rb").read()
            csv_path = os.path.join(tmp, "datasets", "fail.csv")

            with mock.patch.object(
                pd.DataFrame,
                "to_csv",
                side_effect=OSError("write failed"),
            ):
                with self.assertRaises(CsvExportError):
                    generate_dataset_csv_export(tmp, "fail")

            view = build_csv_export_metadata(tmp, "fail")
            self.assertEqual(view["status"], "Not Generated")
            self.assertFalse(os.path.isfile(csv_path))
            self.assertEqual(open(parquet_path, "rb").read(), parquet_before)

    def test_failed_regenerate_restores_previous_csv(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            df = pd.DataFrame({"z": [1]})
            _write_dataset(tmp, "fail2", df)
            generate_dataset_csv_export(tmp, "fail2")
            csv_path = os.path.join(tmp, "datasets", "fail2.csv")
            with open(csv_path, encoding="utf-8") as fh:
                original_csv = fh.read()

            real_replace = os.replace

            def replace_side_effect(src: str, dst: str) -> None:
                if str(dst).endswith(".csv-export.json") and str(src).endswith(".part"):
                    raise OSError("sidecar replace failed")
                return real_replace(src, dst)

            with mock.patch(
                "chain_replay_ml.dataset_builder.dataset_csv_export.os.replace",
                side_effect=replace_side_effect,
            ):
                with self.assertRaises(CsvExportError):
                    generate_dataset_csv_export(tmp, "fail2", replace=True)

            with open(csv_path, encoding="utf-8") as fh:
                self.assertEqual(fh.read(), original_csv)
            view = build_csv_export_metadata(tmp, "fail2")
            self.assertEqual(view["status"], "Generated")

    def test_metadata_refresh_after_generate_and_delete(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            df = pd.DataFrame({"c": [9, 8]})
            _write_dataset(tmp, "meta", df)
            before = build_csv_export_metadata(tmp, "meta")
            self.assertEqual(before["status"], "Not Generated")

            generate_dataset_csv_export(tmp, "meta")
            mid = build_csv_export_metadata(tmp, "meta")
            self.assertEqual(mid["status"], "Generated")
            self.assertEqual(mid["row_count"], 2)

            delete_dataset_csv_export(tmp, "meta")
            after = build_csv_export_metadata(tmp, "meta")
            self.assertEqual(after["status"], "Not Generated")


if __name__ == "__main__":
    unittest.main()

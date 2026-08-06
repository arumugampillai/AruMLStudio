"""prediction_metadata.json sidecar for Trading Days UI."""

from __future__ import annotations

import json
import os
import tempfile
import unittest

from chain_replay_ml.model_lab.prediction_metadata import (
    merge_master_and_metadata,
    normalize_stale_running_days,
    prediction_metadata_path,
    read_prediction_metadata,
    rebuild_prediction_metadata_from_db,
    upsert_prediction_day_metadata,
    write_prediction_metadata,
)
from chain_replay_ml.model_lab.store import ModelLabStore


def _seed_lab(path: str, lab_uuid: str = "u1") -> None:
    with ModelLabStore(path) as store:
        store.write_info(
            lab_uuid=lab_uuid,
            lab_id="lab1",
            lab_name="Test Lab",
            parent_model_id="m1",
            parent_model_name="Model",
            model_checksum=None,
            description=None,
            purpose=None,
            created_by="test",
            status="READY",
            version=1,
            original_feature_count=1,
            selected_feature_count=1,
            training_rows=10,
            target="y",
            algorithm="xgb",
            dataset_snapshot={},
            model_snapshot={},
            training_config_snapshot={},
            wf_snapshot={},
            metrics_snapshot={},
            selected_features_snapshot=["f1"],
            feature_ranking_snapshot={},
            artifact_pointers={},
        )


class TestPredictionMetadata(unittest.TestCase):
    def test_path_next_to_lab_db(self) -> None:
        path = prediction_metadata_path(r"D:\data\model_research\model_lab_X_v1.db")
        self.assertTrue(path.endswith("model_lab_X_v1.prediction_metadata.json"))

    def test_atomic_write_and_read(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            lab = os.path.join(tmp, "lab.db")
            write_prediction_metadata(
                lab,
                {
                    "schema_version": 1,
                    "days": {
                        "2026-07-01": {
                            "status": "completed",
                            "dataset_rows": 10,
                            "prediction_rows": 10,
                            "build_time_sec": 1.5,
                            "completed_at": "2026-07-18T01:00:00",
                            "note": "complete",
                        }
                    },
                },
            )
            path = prediction_metadata_path(lab)
            self.assertTrue(os.path.isfile(path))
            doc = read_prediction_metadata(lab)
            self.assertEqual(doc["days"]["2026-07-01"]["prediction_rows"], 10)
            self.assertIsNotNone(doc.get("updated_at"))

    def test_upsert_and_merge(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            lab = os.path.join(tmp, "lab.db")
            upsert_prediction_day_metadata(
                lab,
                "2026-07-01",
                status="completed",
                prediction_rows=5,
                dataset_rows=5,
                note="complete",
            )
            days = merge_master_and_metadata(
                {"2026-07-01": 5, "2026-07-02": 9},
                read_prediction_metadata(lab),
            )
            by = {d["trading_day"]: d for d in days}
            self.assertEqual(by["2026-07-01"]["status"], "completed")
            self.assertEqual(by["2026-07-01"]["row_count"], 5)
            self.assertTrue(by["2026-07-01"]["ui_meta_ready"])
            self.assertEqual(by["2026-07-02"]["status"], "waiting")
            self.assertEqual(by["2026-07-02"]["rows_expected"], 9)

    def test_stale_running_normalized(self) -> None:
        doc = normalize_stale_running_days(
            {
                "days": {
                    "2026-07-01": {"status": "running", "prediction_rows": 0, "note": ""}
                }
            }
        )
        self.assertEqual(doc["days"]["2026-07-01"]["status"], "cancelled")

    def test_rebuild_from_db(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            lab = os.path.join(tmp, "lab.db")
            _seed_lab(lab)
            with ModelLabStore(lab) as store:
                store.set_build_day_status(
                    "u1",
                    "2026-07-01",
                    status="completed",
                    row_count=3,
                    rows_expected=3,
                    finished=True,
                    sync_ui_meta=False,
                )
            path = prediction_metadata_path(lab)
            if os.path.isfile(path):
                os.remove(path)
            doc = rebuild_prediction_metadata_from_db(lab)
            self.assertEqual(doc["days"]["2026-07-01"]["status"], "completed")
            self.assertEqual(doc["days"]["2026-07-01"]["prediction_rows"], 3)
            with open(path, encoding="utf-8") as fh:
                disk = json.load(fh)
            self.assertEqual(disk["days"]["2026-07-01"]["prediction_rows"], 3)

    def test_set_build_day_status_updates_sidecar(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            lab = os.path.join(tmp, "lab.db")
            _seed_lab(lab)
            with ModelLabStore(lab) as store:
                store.set_build_day_status(
                    "u1",
                    "2026-07-01",
                    status="running",
                    started=True,
                )
                store.set_build_day_status(
                    "u1",
                    "2026-07-01",
                    status="completed",
                    row_count=7,
                    rows_expected=7,
                    finished=True,
                )
            doc = read_prediction_metadata(lab)
            self.assertEqual(doc["days"]["2026-07-01"]["status"], "completed")
            self.assertEqual(doc["days"]["2026-07-01"]["prediction_rows"], 7)


if __name__ == "__main__":
    unittest.main()

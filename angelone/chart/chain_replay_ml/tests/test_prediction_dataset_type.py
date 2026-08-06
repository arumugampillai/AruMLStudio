"""Tests for Prediction Dataset Seen/Unseen (per-model training metadata)."""

from __future__ import annotations

import os
import tempfile
import unittest
from types import SimpleNamespace

from chain_replay_ml.model_lab.prediction_dataset_type import (
    build_day_dataset_types,
    dataset_type_for_day,
    resolve_model_seen_trading_days,
)
from chain_replay_ml.model_lab.prediction_schema import (
    DATASET_TYPE_SEEN,
    DATASET_TYPE_UNSEEN,
    normalize_dataset_type,
)
from chain_replay_ml.model_lab.store import ModelLabStore


class NormalizeDatasetTypeTests(unittest.TestCase):
    def test_missing_defaults_to_seen(self) -> None:
        self.assertEqual(normalize_dataset_type(None), DATASET_TYPE_SEEN)
        self.assertEqual(normalize_dataset_type(""), DATASET_TYPE_SEEN)
        self.assertEqual(normalize_dataset_type("  "), DATASET_TYPE_SEEN)

    def test_aliases(self) -> None:
        self.assertEqual(normalize_dataset_type("seen"), DATASET_TYPE_SEEN)
        self.assertEqual(normalize_dataset_type("Unseen"), DATASET_TYPE_UNSEEN)
        self.assertEqual(normalize_dataset_type("holdout"), DATASET_TYPE_UNSEEN)


class ResolveSeenDaysTests(unittest.TestCase):
    def test_reads_trading_day_labels_from_dataset_snapshot(self) -> None:
        lab = SimpleNamespace(
            dataset_snapshot={
                "dataset_build_snapshot": {
                    "trading_day_labels": "2026-06-22, 2026-06-25, 2026-07-01",
                }
            },
            training_config_snapshot={},
            wf_snapshot={},
        )
        seen = resolve_model_seen_trading_days(lab)
        self.assertEqual(
            seen,
            {"2026-06-22", "2026-06-25", "2026-07-01"},
        )

    def test_all_days_label_falls_back_to_parent_export(self) -> None:
        """Snapshot says 'All days' with no ISO list — parent export is Seen."""
        lab = SimpleNamespace(
            dataset_snapshot={
                "dataset_build_snapshot": {
                    "trading_day_labels": "All days",
                    "trading_days": 13,
                }
            },
            training_config_snapshot={},
            wf_snapshot={},
        )
        parent = [
            "2026-07-01",
            "2026-07-02",
            "2026-07-03",
            "2026-07-07",
            "2026-07-08",
            "2026-07-09",
            "2026-07-10",
            "2026-07-11",
            "2026-07-13",
        ]
        seen = resolve_model_seen_trading_days(lab, parent_trading_days=parent)
        self.assertEqual(seen, set(parent))
        types = build_day_dataset_types(
            parent + ["2026-07-14", "2026-07-15"],
            seen,
        )
        self.assertEqual(types["2026-07-13"], DATASET_TYPE_SEEN)
        self.assertEqual(types["2026-07-14"], DATASET_TYPE_UNSEEN)
        self.assertEqual(types["2026-07-15"], DATASET_TYPE_UNSEEN)

    def test_ignores_prefilter_selected_dates(self) -> None:
        """Expiry checked but excluded from export must not become Seen."""
        lab = SimpleNamespace(
            dataset_snapshot={
                "dataset_build_snapshot": {
                    "trading_day_filter": {
                        "mode": "exclude_expiry",
                        "selected_dates": ["2026-07-13", "2026-07-14"],
                        "exported_dates": ["2026-07-13"],
                        "excluded_dates": ["2026-07-14"],
                    }
                }
            },
            training_config_snapshot={},
            wf_snapshot={},
        )
        seen = resolve_model_seen_trading_days(lab)
        self.assertEqual(seen, {"2026-07-13"})
        self.assertNotIn("2026-07-14", seen)

    def test_per_model_classification_matrix(self) -> None:
        parent = [
            "2026-06-22",
            "2026-06-25",
            "2026-07-01",
            "2026-07-08",
            "2026-07-13",
        ]
        seen_a = {"2026-06-22", "2026-06-25", "2026-07-01"}
        seen_b = {"2026-06-22", "2026-07-08"}
        types_a = build_day_dataset_types(parent, seen_a)
        types_b = build_day_dataset_types(parent, seen_b)
        self.assertEqual(types_a["2026-06-22"], DATASET_TYPE_SEEN)
        self.assertEqual(types_a["2026-06-25"], DATASET_TYPE_SEEN)
        self.assertEqual(types_a["2026-07-01"], DATASET_TYPE_SEEN)
        self.assertEqual(types_a["2026-07-08"], DATASET_TYPE_UNSEEN)
        self.assertEqual(types_a["2026-07-13"], DATASET_TYPE_UNSEEN)
        self.assertEqual(types_b["2026-06-22"], DATASET_TYPE_SEEN)
        self.assertEqual(types_b["2026-06-25"], DATASET_TYPE_UNSEEN)
        self.assertEqual(types_b["2026-07-01"], DATASET_TYPE_UNSEEN)
        self.assertEqual(types_b["2026-07-08"], DATASET_TYPE_SEEN)
        self.assertEqual(types_b["2026-07-13"], DATASET_TYPE_UNSEEN)

    def test_empty_seen_defaults_to_seen(self) -> None:
        self.assertEqual(dataset_type_for_day("2026-07-13", set()), DATASET_TYPE_SEEN)
        self.assertEqual(dataset_type_for_day("2026-07-13", None), DATASET_TYPE_SEEN)


class DatasetTypePersistenceTests(unittest.TestCase):
    def test_ensure_build_days_writes_per_day_types(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "lab.db")
            with ModelLabStore(path) as store:
                lab_uuid = "lab-model-a"
                store.ensure_prediction_schema()
                store.ensure_build_days(
                    lab_uuid,
                    ["2026-06-22", "2026-06-25", "2026-07-13"],
                    day_dataset_types={
                        "2026-06-22": DATASET_TYPE_SEEN,
                        "2026-06-25": DATASET_TYPE_SEEN,
                        "2026-07-13": DATASET_TYPE_UNSEEN,
                    },
                )
                by_day = {
                    d["trading_day"]: d["dataset_type"]
                    for d in store.list_build_days(lab_uuid)
                }
                self.assertEqual(by_day["2026-06-22"], DATASET_TYPE_SEEN)
                self.assertEqual(by_day["2026-06-25"], DATASET_TYPE_SEEN)
                self.assertEqual(by_day["2026-07-13"], DATASET_TYPE_UNSEEN)

                # Same catalog day can be Unseen for another lab/model.
                lab_b = "lab-model-b"
                store.ensure_build_days(
                    lab_b,
                    ["2026-06-22", "2026-06-25", "2026-07-13"],
                    day_dataset_types={
                        "2026-06-22": DATASET_TYPE_SEEN,
                        "2026-06-25": DATASET_TYPE_UNSEEN,
                        "2026-07-13": DATASET_TYPE_UNSEEN,
                    },
                )
                by_b = {
                    d["trading_day"]: d["dataset_type"]
                    for d in store.list_build_days(lab_b)
                }
                self.assertEqual(by_b["2026-06-25"], DATASET_TYPE_UNSEEN)
                # Model A catalog unchanged.
                by_a = {
                    d["trading_day"]: d["dataset_type"]
                    for d in store.list_build_days(lab_uuid)
                }
                self.assertEqual(by_a["2026-06-25"], DATASET_TYPE_SEEN)

    def test_legacy_summary_without_type_reads_as_seen(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "legacy.db")
            with ModelLabStore(path) as store:
                lab_uuid = "legacy-uuid"
                store.ensure_prediction_schema()
                store.write_prediction_summary(
                    lab_uuid=lab_uuid,
                    status="not_generated",
                    row_count=0,
                    trading_days=0,
                )
                row = store.read_prediction_summary()
                assert row is not None
                self.assertEqual(row["dataset_type"], DATASET_TYPE_SEEN)

    def test_new_days_default_seen_without_type_map(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "lab.db")
            with ModelLabStore(path) as store:
                lab_uuid = "lab-test-uuid"
                store.ensure_prediction_schema()
                store.ensure_build_days(lab_uuid, ["2026-06-01", "2026-06-02"])
                days = store.list_build_days(lab_uuid)
                self.assertEqual(len(days), 2)
                for d in days:
                    self.assertEqual(d["dataset_type"], DATASET_TYPE_SEEN)


if __name__ == "__main__":
    unittest.main()

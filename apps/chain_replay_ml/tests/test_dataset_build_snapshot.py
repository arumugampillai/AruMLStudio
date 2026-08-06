"""Tests for frozen dataset build snapshots on model packages."""

from __future__ import annotations

import unittest

from chain_replay_ml.dataset_builder.dataset_summary import (
    _trading_day_labels,
    build_dataset_build_snapshot,
)
from chain_replay_ml.training.dataset_build_snapshot import (
    backfill_model_dataset_snapshot,
    dataset_meta_from_snapshot,
    resolve_dataset_build_snapshot,
    snapshot_is_complete,
)


class TestTradingDayLabelsPrefersConcreteDates(unittest.TestCase):
    """Auto tab "All days" builds must show real dates, not a vague label."""

    def test_all_days_with_trading_day_filter_shows_dates(self) -> None:
        meta = {
            "master_filter": {"all_days": True},
            "trading_day_filter": {
                "mode": "all",
                "selected_dates": ["2026-06-30", "2026-07-07", "2026-07-21"],
                "exported_dates": ["2026-06-30", "2026-07-07", "2026-07-21"],
            },
        }
        self.assertEqual(
            _trading_day_labels(meta),
            "2026-06-30, 2026-07-07, 2026-07-21",
        )

    def test_all_days_falls_back_to_exported_days_list(self) -> None:
        meta = {
            "master_filter": {"all_days": True},
            "days": [
                {"trading_day": "2026-07-14", "market": "NIFTY"},
                {"trading_day": "2026-07-21", "market": "NIFTY"},
            ],
        }
        self.assertEqual(_trading_day_labels(meta), "2026-07-14, 2026-07-21")

    def test_all_days_without_any_resolved_dates_uses_label(self) -> None:
        meta = {"master_filter": {"all_days": True}}
        self.assertEqual(_trading_day_labels(meta), "All days")

    def test_selected_days_label_still_works_without_dates(self) -> None:
        meta = {
            "master_filter": {
                "all_days": False,
                "selected_days": ["2026-01-02", "2026-01-06"],
            },
        }
        self.assertEqual(_trading_day_labels(meta), "2026-01-02, 2026-01-06")


class TestBuildDatasetBuildSnapshot(unittest.TestCase):
    def test_captures_filters_and_days(self) -> None:
        meta = {
            "dataset_name": "MS_42f_10s_1200",
            "created_at": "2026-01-02T10:00:00+00:00",
            "export_source": "master_filter_export",
            "market": "NIFTY",
            "row_count": 1200,
            "feature_count": 42,
            "target_count": 3,
            "trading_days": 3,
            "sampling": {"interval_sec": 10, "method": "fixed_interval"},
            "strike_selection": {"mode": "atm_band", "band": 10},
            "master_filter": {
                "all_days": False,
                "selected_days": ["2026-01-02", "2026-01-03", "2026-01-06"],
                "atm_band_filter": 10,
                "premium_enabled": True,
                "premium_min": 15.0,
                "premium_max": 40.0,
                "no_null_data": True,
            },
            "selection_method": {
                "summary": "2026-01-02, 2026-01-03, 2026-01-06 · ATM ±10 · LTP 15–40 · No null data",
                "criteria": {
                    "premium_enabled": True,
                    "premium_min": 15.0,
                    "premium_max": 40.0,
                    "no_null_data": True,
                },
            },
            "no_null_dropped_columns": ["col_a"],
            "builder_version": "v1.2.3",
        }
        snap = build_dataset_build_snapshot(meta, dataset_name="MS_42f_10s_1200", snapshotted_at="2026-01-10T00:00:00+00:00")
        self.assertEqual(snap["dataset_name"], "MS_42f_10s_1200")
        self.assertEqual(snap["trading_days"], 3)
        self.assertIn("2026-01-02", snap["trading_day_labels"])
        labels = {row["label"]: row["value"] for row in snap["filter_summary"]}
        self.assertEqual(labels.get("LTP / Premium"), "15–40")
        self.assertEqual(labels.get("No null data"), "On")
        self.assertEqual(labels.get("ATM band"), "±10")

    def test_resolve_from_model_detail(self) -> None:
        snap = {"dataset_name": "ds1", "filter_summary": [{"label": "No null data", "value": "On"}]}
        doc = {"dataset_build_snapshot": snap}
        self.assertEqual(resolve_dataset_build_snapshot(doc)["dataset_name"], "ds1")

    def test_dataset_meta_from_snapshot_for_retrain(self) -> None:
        snap = build_dataset_build_snapshot(
            {
                "market": "NIFTY",
                "master_filter": {
                    "premium_enabled": True,
                    "premium_min": 10.0,
                    "premium_max": 30.0,
                    "no_null_data": True,
                },
                "selection_method": {"summary": "test"},
                "sampling": {"interval_sec": 10},
            },
            dataset_name="ds1",
        )
        meta = dataset_meta_from_snapshot(snap)
        self.assertEqual(meta["market"], "NIFTY")
        self.assertTrue(meta["master_filter"]["no_null_data"])
        self.assertEqual(meta["sampling"]["interval_sec"], 10)

    def test_snapshot_is_complete(self) -> None:
        self.assertFalse(snapshot_is_complete({}))
        self.assertTrue(snapshot_is_complete({"filter_summary": [{"label": "No null data", "value": "On"}]}))


if __name__ == "__main__":
    unittest.main()

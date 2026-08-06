"""Fast Trading Days skeleton vs full catalog sync."""

from __future__ import annotations

import os
import tempfile
import unittest
from unittest import mock

from chain_replay_ml.model_lab.prediction_builder import _day_ui_meta_ready
from chain_replay_ml.model_lab.store import ModelLabStore


class TestPredictionDaysUiSkeleton(unittest.TestCase):
    def test_day_ui_meta_ready_heuristic(self) -> None:
        self.assertFalse(_day_ui_meta_ready({"ui_meta_ready": False, "status": "waiting"}))
        self.assertTrue(_day_ui_meta_ready({"ui_meta_ready": True}))
        self.assertTrue(_day_ui_meta_ready({"rows_expected": 10, "status": "waiting"}))
        self.assertTrue(_day_ui_meta_ready({"row_count": 5, "status": "waiting"}))
        self.assertTrue(_day_ui_meta_ready({"status": "completed"}))
        self.assertFalse(_day_ui_meta_ready({"status": "waiting", "row_count": 0}))

    def test_ensure_build_days_can_skip_pred_scan(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "lab.db")
            with ModelLabStore(path) as store:
                store.ensure_prediction_schema()
                lab_uuid = "lab-1"
                with mock.patch.object(
                    store,
                    "prediction_row_counts_by_day",
                    return_value={"2026-07-01": 99},
                ) as scanned:
                    store.ensure_build_days(
                        lab_uuid,
                        ["2026-07-01", "2026-07-02"],
                        sync_pred_counts=False,
                    )
                    scanned.assert_not_called()
                    rows = {d["trading_day"]: d for d in store.list_build_days(lab_uuid)}
                    self.assertEqual(rows["2026-07-01"]["row_count"], 0)
                    self.assertEqual(rows["2026-07-02"]["status"], "waiting")

                    store.ensure_build_days(
                        lab_uuid,
                        ["2026-07-01"],
                        sync_pred_counts=True,
                    )
                    scanned.assert_called()
                    rows2 = {d["trading_day"]: d for d in store.list_build_days(lab_uuid)}
                    self.assertEqual(rows2["2026-07-01"]["row_count"], 99)
                    self.assertEqual(rows2["2026-07-01"]["status"], "completed")


if __name__ == "__main__":
    unittest.main()

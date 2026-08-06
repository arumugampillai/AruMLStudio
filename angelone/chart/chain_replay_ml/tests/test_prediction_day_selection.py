"""Regression tests for prediction-build day selection.

Building a single day writes the caller's ``selected_days`` into the shared
``prediction_day_metadata.selected`` column (``ensure_build_days`` /
``set_days_selected``) and, as a side effect, *deselects every other day* in
the catalog. If that shared column is stale or gets rewritten by another
build in between "mark selected" and "compute days to run" (e.g. a
concurrent/otherwise-overlapping build for a different day), a day that was
just explicitly requested via ``selected_days`` can be silently dropped and
the caller sees "No days selected to build" even though the day clearly has
0 prediction rows and needs building.

The fix: when ``selected_days`` is provided, day-selection must be resolved
directly against that explicit list (via ``pending_build_days(...,
selected_only=False)``) instead of gating on the shared ``selected`` column.
"""

from __future__ import annotations

import os
import tempfile
import unittest
from unittest import mock

import pandas as pd

from chain_replay_ml.model_lab.prediction_builder import build_prediction_dataset
from chain_replay_ml.model_lab.prediction_schema import DAY_COMPLETED, DAY_WAITING
from chain_replay_ml.model_lab.service import create_model_lab
from chain_replay_ml.model_lab.store import ModelLabStore
from chain_replay_ml.tests.test_model_lab_phase1 import _sample_doc
from chain_replay_ml.training.inference_runtime import InferenceRuntimeInfo


class _FakeModel:
    feature_names_in_ = ["gamma", "ltp", "delta"]

    def predict(self, X):
        if hasattr(X, "columns"):
            return (X["ltp"].astype(float) + 1.0).to_numpy()
        import numpy as np

        return np.asarray(X, dtype=float)[:, 1] + 1.0


_FAKE_INFO = InferenceRuntimeInfo(
    algorithm="xgboost",
    device_label="CPU",
    device_param="cpu",
    gpu_requested=False,
    gpu_active=False,
)


class DaySelectedFlagStoreTests(unittest.TestCase):
    """Store-level: ``selected`` is not a reliable gate on its own."""

    def test_pending_build_days_selected_only_misses_deselected_day(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "lab.db")
            with ModelLabStore(path) as store:
                lab = "lab-1"
                store.ensure_prediction_schema()
                store.ensure_build_days(lab, ["2026-07-01", "2026-07-02"])

                # Day 1 gets built + selected; the destructive selection update
                # deselects day 2 as a side effect (mirrors ensure_build_days /
                # set_days_selected semantics exercised by a single-day build).
                store.set_build_day_status(
                    lab, "2026-07-01", status=DAY_COMPLETED, row_count=5, finished=True,
                    progress_pct=100.0,
                )
                store.set_days_selected(lab, ["2026-07-01"])

                days = {d["trading_day"]: d for d in store.list_build_days(lab)}
                self.assertEqual(days["2026-07-02"]["status"], DAY_WAITING)
                self.assertEqual(days["2026-07-02"]["row_count"], 0)
                self.assertFalse(days["2026-07-02"]["selected"])

                # Old behaviour: gating on the shared `selected` column hides a
                # day that is genuinely pending (0 rows, Waiting) simply because
                # an unrelated day's build flipped its selection off.
                self.assertEqual(
                    store.pending_build_days(lab, selected_only=True), []
                )

                # Fixed behaviour: resolve against the caller's explicit day
                # list instead of the shared column.
                pending_all = store.pending_build_days(lab, selected_only=False)
                wanted = {"2026-07-02"}
                self.assertEqual(
                    [d for d in pending_all if d in wanted], ["2026-07-02"]
                )


class BuildPredictionDatasetDaySelectionRaceTests(unittest.TestCase):
    """End-to-end: explicit selected_days must not be dropped by stale state."""

    def _make_dataset(self, data_dir: str) -> None:
        datasets = os.path.join(data_dir, "datasets")
        models = os.path.join(data_dir, "models", "Future_LTP_5m_WF_239f_XGB_0223_13")
        os.makedirs(datasets, exist_ok=True)
        os.makedirs(models, exist_ok=True)
        with open(os.path.join(models, "model.ubj"), "wb") as fh:
            fh.write(b"fake-model-bytes")

        df = pd.DataFrame(
            {
                "trading_day": ["2026-07-01", "2026-07-01", "2026-07-02"],
                "timestamp": [1.0, 2.0, 3.0],
                "token": ["t1", "t2", "t3"],
                "strike": [24000.0, 24100.0, 24200.0],
                "option_type": ["CE", "PE", "CE"],
                "expiry": ["2026-07-02", "2026-07-02", "2026-07-02"],
                "market": ["NIFTY", "NIFTY", "NIFTY"],
                "spot": [24050.0, 24050.0, 24100.0],
                "ltp": [100.0, 80.0, 90.0],
                "minutes_to_expiry": [120.0, 100.0, 80.0],
                "delta": [0.5, -0.4, 0.3],
                "gamma": [0.01, 0.02, 0.015],
                "future_ltp_5m": [101.0, 78.0, 95.0],
            }
        )
        df.to_parquet(os.path.join(datasets, "Master_NIFTY_239f.parquet"), index=False)

    def test_second_day_build_survives_stale_selected_flag(self) -> None:
        """Day B (0 rows, explicitly requested) must build even if some other
        write already reset its `selected` flag to 0 before this call, and
        even if this call's own re-selection step is lost (simulated race)."""
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = os.path.join(tmp, "data")
            research = os.path.join(tmp, "model_research")
            self._make_dataset(data_dir)
            info = create_model_lab(data_dir, _sample_doc(), research_dir=research)

            patches = (
                mock.patch(
                    "chain_replay_ml.model_lab.prediction_parallel.load_prediction_model_for_inference",
                    return_value=(_FakeModel(), _FAKE_INFO),
                ),
                mock.patch(
                    "chain_replay_ml.model_lab.prediction_builder.load_prediction_model_cached",
                    return_value=(_FakeModel(), 1.0, True),
                ),
                mock.patch(
                    "chain_replay_ml.model_lab.prediction_parallel._load_day_timelines",
                    return_value={},
                ),
            )

            # Build day 1 only — this is the action that (as a documented side
            # effect of ensure_build_days/set_days_selected) deselects day 2.
            with patches[0], patches[1], patches[2]:
                first = build_prediction_dataset(
                    data_dir,
                    info.db_path,
                    overwrite=False,
                    selected_days=["2026-07-01"],
                    enrich_path_outcomes=False,
                    workers=1,
                    print_timing=False,
                )
            self.assertTrue(first.get("ok"), first)
            self.assertEqual(first.get("days_processed"), ["2026-07-01"])

            with ModelLabStore(info.db_path) as store:
                lab_uuid = store.read_info().lab_uuid
                days = {d["trading_day"]: d for d in store.list_build_days(lab_uuid)}
            self.assertFalse(days["2026-07-02"]["selected"])
            self.assertEqual(days["2026-07-02"]["row_count"], 0)

            # Now build day 2, simulating a race where this call's own
            # "mark day 2 selected" write is lost (e.g. clobbered by a
            # concurrent/overlapping build for another day). Prior to the
            # fix, days_to_run was computed by gating on the shared
            # `selected` column, so this would fail with
            # "No days selected to build" despite day 2 explicitly requested
            # and having 0 prediction rows.
            with patches[0], patches[1], patches[2], mock.patch.object(
                ModelLabStore, "set_days_selected", lambda self, lab, days: None
            ), mock.patch.object(
                ModelLabStore,
                "ensure_build_days",
                lambda self, lab, days, **kw: None,
            ):
                second = build_prediction_dataset(
                    data_dir,
                    info.db_path,
                    overwrite=False,
                    resume=True,
                    selected_days=["2026-07-02"],
                    enrich_path_outcomes=False,
                    workers=1,
                    print_timing=False,
                )

            self.assertTrue(second.get("ok"), second)
            self.assertFalse(second.get("already_complete"), second)
            self.assertEqual(second.get("days_processed"), ["2026-07-02"])
            self.assertEqual(second.get("row_count"), 3)


if __name__ == "__main__":
    unittest.main()

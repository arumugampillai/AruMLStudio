"""Regression tests for Prediction Dataset Build integrity.

Covers two bugs:

1. Triple Barrier is an inference side-scorer (like Regression / Probability
   Ladder) and must run for every built prediction row, Seen *and* Unseen.
   Resume must not silently skip a day that already has prediction rows but
   is missing/stale on the requested TB model.
2. A Trading Day's status must show ``Complete`` only when every dataset row
   for that day has a corresponding prediction row; otherwise it must show
   ``Partial``.
"""

from __future__ import annotations

import os
import tempfile
import unittest
from unittest import mock

import pandas as pd

from chain_replay_ml.model_lab.prediction_builder import build_prediction_dataset
from chain_replay_ml.model_lab.prediction_schema import (
    DAY_COMPLETED,
    DAY_PARTIAL,
    resolve_day_completion_status,
)
from chain_replay_ml.model_lab.service import create_model_lab
from chain_replay_ml.model_lab.store import ModelLabStore
from chain_replay_ml.tests.test_model_lab_phase1 import _sample_doc
from chain_replay_ml.tests.test_triple_barrier_lab_build import _write_xgb_tb_package
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


def _write_primary_model(data_dir: str) -> None:
    models = os.path.join(data_dir, "models", "Future_LTP_5m_WF_239f_XGB_0223_13")
    os.makedirs(models, exist_ok=True)
    with open(os.path.join(models, "model.ubj"), "wb") as fh:
        fh.write(b"fake-model-bytes")


def _write_master_parquet(data_dir: str, df: pd.DataFrame) -> None:
    datasets = os.path.join(data_dir, "datasets")
    os.makedirs(datasets, exist_ok=True)
    df.to_parquet(os.path.join(datasets, "Master_NIFTY_239f.parquet"), index=False)


class DayCompletionStatusTests(unittest.TestCase):
    """Pure unit tests for ``resolve_day_completion_status``."""

    def test_partial_when_rows_written_less_than_expected(self) -> None:
        self.assertEqual(resolve_day_completion_status(24933, 130967), DAY_PARTIAL)

    def test_complete_when_rows_match_expected(self) -> None:
        self.assertEqual(resolve_day_completion_status(130967, 130967), DAY_COMPLETED)

    def test_complete_when_rows_exceed_expected(self) -> None:
        self.assertEqual(resolve_day_completion_status(5, 3), DAY_COMPLETED)

    def test_complete_fallback_when_expected_unknown(self) -> None:
        # No denominator to compare against -> legacy "finished processing" meaning.
        self.assertEqual(resolve_day_completion_status(5, None), DAY_COMPLETED)
        self.assertEqual(resolve_day_completion_status(5, 0), DAY_COMPLETED)


class DayStatusBuildIntegrationTests(unittest.TestCase):
    """End-to-end: build_prediction_dataset must never mark Complete on partial coverage."""

    def _make_dataset(self, data_dir: str) -> None:
        _write_primary_model(data_dir)
        df = pd.DataFrame(
            {
                "trading_day": ["2026-07-01", "2026-07-01", "2026-07-01"],
                "timestamp": [1.0, 2.0, 3.0],
                "token": ["t1", "t2", "t3"],
                "strike": [24000.0, 24100.0, 24200.0],
                "option_type": ["CE", "PE", "CE"],
                "expiry": ["2026-07-02"] * 3,
                "market": ["NIFTY"] * 3,
                "spot": [24050.0] * 3,
                "ltp": [100.0, 80.0, 90.0],
                "minutes_to_expiry": [120.0, 100.0, 80.0],
                "delta": [0.5, -0.4, 0.3],
                "gamma": [0.01, 0.02, 0.015],
                # t3 has no target -> dropped by the build (dropna on target),
                # so this day can only ever write 2 of its 3 dataset rows.
                "future_ltp_5m": [101.0, 78.0, None],
            }
        )
        _write_master_parquet(data_dir, df)

    def _run_build(self, data_dir: str, db_path: str) -> dict:
        with mock.patch(
            "chain_replay_ml.model_lab.prediction_parallel.load_prediction_model_for_inference",
            return_value=(_FakeModel(), _FAKE_INFO),
        ), mock.patch(
            "chain_replay_ml.model_lab.prediction_builder.load_prediction_model_cached",
            return_value=(_FakeModel(), 1.0, True),
        ), mock.patch(
            "chain_replay_ml.model_lab.prediction_parallel._load_day_timelines",
            return_value={},
        ):
            return build_prediction_dataset(
                data_dir,
                db_path,
                overwrite=False,
                enrich_path_outcomes=False,
                workers=1,
                print_timing=False,
            )

    def test_partial_row_coverage_never_shows_complete(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = os.path.join(tmp, "data")
            research = os.path.join(tmp, "model_research")
            self._make_dataset(data_dir)
            info = create_model_lab(data_dir, _sample_doc(), research_dir=research)

            with ModelLabStore(info.db_path) as store:
                lab_uuid = store.read_info().lab_uuid
                store.ensure_build_days(lab_uuid, ["2026-07-01"])
                # True dataset row count for the day, e.g. from Master / registry
                # counts — deliberately larger than what the build can ever write.
                store.set_day_rows_expected(lab_uuid, "2026-07-01", 3)

            result = self._run_build(data_dir, info.db_path)
            self.assertTrue(result.get("ok"), result)
            self.assertEqual(result.get("row_count"), 2)

            with ModelLabStore(info.db_path) as store:
                lab_uuid = store.read_info().lab_uuid
                days = {d["trading_day"]: d for d in store.list_build_days(lab_uuid)}
            self.assertEqual(days["2026-07-01"]["status"], DAY_PARTIAL)
            self.assertNotEqual(days["2026-07-01"]["status"], DAY_COMPLETED)
            self.assertEqual(days["2026-07-01"]["row_count"], 2)
            self.assertEqual(days["2026-07-01"]["rows_expected"], 3)

    def test_full_row_coverage_shows_complete(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = os.path.join(tmp, "data")
            research = os.path.join(tmp, "model_research")
            self._make_dataset(data_dir)
            info = create_model_lab(data_dir, _sample_doc(), research_dir=research)

            with ModelLabStore(info.db_path) as store:
                lab_uuid = store.read_info().lab_uuid
                store.ensure_build_days(lab_uuid, ["2026-07-01"])
                # Only 2 rows are ever buildable for this day (one target is
                # NaN) — expected matches, mirroring a Master/registry count
                # that already excludes the unusable row.
                store.set_day_rows_expected(lab_uuid, "2026-07-01", 2)

            result = self._run_build(data_dir, info.db_path)
            self.assertTrue(result.get("ok"), result)
            self.assertEqual(result.get("row_count"), 2)

            with ModelLabStore(info.db_path) as store:
                lab_uuid = store.read_info().lab_uuid
                days = {d["trading_day"]: d for d in store.list_build_days(lab_uuid)}
            self.assertEqual(days["2026-07-01"]["status"], DAY_COMPLETED)


class TripleBarrierResumeAndUnseenTests(unittest.TestCase):
    """Triple Barrier must score every built row, Seen and Unseen, and must
    not be silently skipped by resume just because a day already has rows."""

    def test_days_needing_tb_rescore_includes_unseen_day(self) -> None:
        """Store-level: TB-stale detection must not filter by dataset_type.

        A day previously classified Unseen (never used in training) but
        already built without TB (or with a different TB model) must still
        surface as needing a TB rescore — TB is not restricted to Seen days.
        """
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "lab.db")
            with ModelLabStore(path) as store:
                lab_uuid = "lab-1"
                store.ensure_prediction_schema()
                store.ensure_build_days(lab_uuid, ["2026-07-01"])
                store.apply_day_dataset_types(lab_uuid, {"2026-07-01": "unseen"})

                cols = store.list_prediction_columns()
                self.assertIn("tb_model_name", cols)
                store.conn.execute(
                    """
                    INSERT INTO prediction_dataset
                        (prediction_id, lab_uuid, trading_day, tb_model_name)
                    VALUES (?, ?, ?, NULL)
                    """,
                    ("pred-1", lab_uuid, "2026-07-01"),
                )
                store.conn.commit()

                days = {d["trading_day"]: d for d in store.list_build_days(lab_uuid)}
                self.assertEqual(days["2026-07-01"].get("dataset_type"), "Unseen")

                stale = store.days_needing_tb_rescore(
                    lab_uuid, ["2026-07-01"], "TB_test_model"
                )
                self.assertIn("2026-07-01", stale)

    def test_days_needing_tb_rescore_detects_all_null_probability_same_model(self) -> None:
        """Store-level: a day already stamped with the requested TB model but
        whose ``tb_pred_probability`` is NULL for every row (the "Triple
        Barrier enabled but scored 0 of N rows" failure — model resolve
        failure, missing features, predict exception) must still surface as
        needing a rescore. Matching ``tb_model_name`` alone is not enough
        evidence that TB actually scored the day; otherwise a silently
        failed TB run looks identical to a fully-scored one and is never
        retried on resume.
        """
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "lab.db")
            with ModelLabStore(path) as store:
                lab_uuid = "lab-1"
                store.ensure_prediction_schema()
                store.ensure_build_days(lab_uuid, ["2026-07-01"])
                store.conn.execute(
                    """
                    INSERT INTO prediction_dataset
                        (prediction_id, lab_uuid, trading_day, tb_model_name, tb_pred_probability)
                    VALUES (?, ?, ?, ?, NULL)
                    """,
                    ("pred-1", lab_uuid, "2026-07-01", "TB_test_model"),
                )
                store.conn.commit()

                stale = store.days_needing_tb_rescore(
                    lab_uuid, ["2026-07-01"], "TB_test_model"
                )
                self.assertIn(
                    "2026-07-01",
                    stale,
                    "day with matching tb_model_name but NULL probability must be stale",
                )

    def test_days_needing_tb_rescore_leaves_fully_scored_day_alone(self) -> None:
        """A day whose rows already carry the requested model name *and* a
        real probability must not be flagged — avoid needless day rebuilds
        (tick-timeline reloads) once TB has actually scored successfully."""
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "lab.db")
            with ModelLabStore(path) as store:
                lab_uuid = "lab-1"
                store.ensure_prediction_schema()
                store.ensure_build_days(lab_uuid, ["2026-07-01"])
                store.conn.execute(
                    """
                    INSERT INTO prediction_dataset
                        (prediction_id, lab_uuid, trading_day, tb_model_name, tb_pred_probability)
                    VALUES (?, ?, ?, ?, 0.42)
                    """,
                    ("pred-1", lab_uuid, "2026-07-01", "TB_test_model"),
                )
                store.conn.commit()

                stale = store.days_needing_tb_rescore(
                    lab_uuid, ["2026-07-01"], "TB_test_model"
                )
                self.assertNotIn("2026-07-01", stale)

    def test_build_with_tb_enabled_scores_unseen_and_seen_days(self) -> None:
        """Acceptance: after Build with TB enabled, all four tb_* columns are
        populated for every prediction row written that day — for a day
        explicitly marked Unseen just as much as one marked Seen."""
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = os.path.join(tmp, "data")
            research = os.path.join(tmp, "model_research")
            _write_primary_model(data_dir)
            _write_xgb_tb_package(data_dir, "TB_test_model", features=["delta", "gamma"])

            df = pd.DataFrame(
                {
                    "trading_day": ["2026-07-01", "2026-07-01", "2026-07-02"],
                    "timestamp": [1.0, 2.0, 3.0],
                    "token": ["t1", "t2", "t3"],
                    "strike": [24000.0, 24100.0, 24200.0],
                    "option_type": ["CE", "PE", "CE"],
                    "expiry": ["2026-07-02"] * 3,
                    "market": ["NIFTY"] * 3,
                    "spot": [24050.0, 24050.0, 24100.0],
                    "ltp": [100.0, 80.0, 90.0],
                    "minutes_to_expiry": [120.0, 100.0, 80.0],
                    "delta": [0.5, -0.4, 0.3],
                    "gamma": [0.01, 0.02, 0.015],
                    "future_ltp_5m": [101.0, 78.0, 95.0],
                }
            )
            _write_master_parquet(data_dir, df)
            info = create_model_lab(data_dir, _sample_doc(), research_dir=research)

            # Seed catalog so day 2 can be explicitly flagged Unseen before Build
            # (mirrors the real UI's catalog-sync step running ahead of Build).
            with ModelLabStore(info.db_path) as store:
                lab_uuid = store.read_info().lab_uuid
                store.ensure_build_days(lab_uuid, ["2026-07-01", "2026-07-02"])
                store.apply_day_dataset_types(lab_uuid, {"2026-07-02": "unseen"})
                days = {d["trading_day"]: d for d in store.list_build_days(lab_uuid)}
                self.assertEqual(days["2026-07-01"].get("dataset_type"), "Seen")
                self.assertEqual(days["2026-07-02"].get("dataset_type"), "Unseen")

            with mock.patch(
                "chain_replay_ml.model_lab.prediction_parallel.load_prediction_model_for_inference",
                return_value=(_FakeModel(), _FAKE_INFO),
            ), mock.patch(
                "chain_replay_ml.model_lab.prediction_builder.load_prediction_model_cached",
                return_value=(_FakeModel(), 1.0, True),
            ), mock.patch(
                "chain_replay_ml.model_lab.prediction_parallel._load_day_timelines",
                return_value={},
            ):
                result = build_prediction_dataset(
                    data_dir,
                    info.db_path,
                    overwrite=False,
                    enrich_path_outcomes=False,
                    workers=1,
                    print_timing=False,
                    tb_model_name="TB_test_model",
                )
            self.assertTrue(result.get("ok"), result)
            self.assertEqual(result.get("row_count"), 3)
            self.assertEqual(sorted(result.get("days_processed") or []), [
                "2026-07-01",
                "2026-07-02",
            ])

            with ModelLabStore(info.db_path) as store:
                rows = store.conn.execute(
                    """
                    SELECT trading_day, tb_model_name, tb_label_run,
                           tb_pred_probability, tb_pred_class
                    FROM prediction_dataset
                    ORDER BY trading_day
                    """
                ).fetchall()

            self.assertEqual(len(rows), 3)
            by_day: dict[str, list] = {}
            for day, name, label_run, prob, cls in rows:
                by_day.setdefault(day, []).append((name, label_run, prob, cls))
                self.assertEqual(name, "TB_test_model")
                self.assertEqual(label_run, "run_tb_test")
                self.assertIsNotNone(prob)
                self.assertIsNotNone(cls)

            # Both the Seen day (2026-07-01) and the Unseen day (2026-07-02)
            # got scored — TB is not gated on Seen/Unseen.
            self.assertEqual(len(by_day.get("2026-07-01", [])), 2)
            self.assertEqual(len(by_day.get("2026-07-02", [])), 1)

    def test_resume_rescoures_previously_completed_day_missing_tb(self) -> None:
        """A day already Complete from a build that ran without TB must be
        reprocessed once TB is enabled on a resumed build, so tb_* columns
        stop being NULL forever for rows that were already written."""
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = os.path.join(tmp, "data")
            research = os.path.join(tmp, "model_research")
            _write_primary_model(data_dir)
            _write_xgb_tb_package(data_dir, "TB_test_model", features=["delta", "gamma"])

            df = pd.DataFrame(
                {
                    "trading_day": ["2026-07-01", "2026-07-01"],
                    "timestamp": [1.0, 2.0],
                    "token": ["t1", "t2"],
                    "strike": [24000.0, 24100.0],
                    "option_type": ["CE", "PE"],
                    "expiry": ["2026-07-02", "2026-07-02"],
                    "market": ["NIFTY", "NIFTY"],
                    "spot": [24050.0, 24050.0],
                    "ltp": [100.0, 80.0],
                    "minutes_to_expiry": [120.0, 100.0],
                    "delta": [0.5, -0.4],
                    "gamma": [0.01, 0.02],
                    "future_ltp_5m": [101.0, 78.0],
                }
            )
            _write_master_parquet(data_dir, df)
            info = create_model_lab(data_dir, _sample_doc(), research_dir=research)

            common_patches = (
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

            # First build: no TB — the historical "TB was Enabled=None" case.
            with common_patches[0], common_patches[1], common_patches[2]:
                first = build_prediction_dataset(
                    data_dir,
                    info.db_path,
                    overwrite=False,
                    enrich_path_outcomes=False,
                    workers=1,
                    print_timing=False,
                )
            self.assertTrue(first.get("ok"), first)
            self.assertEqual(first.get("row_count"), 2)

            with ModelLabStore(info.db_path) as store:
                tb_names = [
                    r[0]
                    for r in store.conn.execute(
                        "SELECT tb_model_name FROM prediction_dataset"
                    ).fetchall()
                ]
            self.assertEqual(tb_names, [None, None])

            # Second call: resume with TB enabled — must rescore the already
            # "completed" day rather than skip it because rows already exist.
            with common_patches[0], common_patches[1], common_patches[2]:
                second = build_prediction_dataset(
                    data_dir,
                    info.db_path,
                    overwrite=False,
                    resume=True,
                    enrich_path_outcomes=False,
                    workers=1,
                    print_timing=False,
                    tb_model_name="TB_test_model",
                )
            self.assertTrue(second.get("ok"), second)
            self.assertFalse(second.get("already_complete"), second)
            self.assertEqual(second.get("days_processed"), ["2026-07-01"])

            with ModelLabStore(info.db_path) as store:
                tb_rows = store.conn.execute(
                    "SELECT tb_model_name, tb_pred_probability, tb_pred_class "
                    "FROM prediction_dataset"
                ).fetchall()
            self.assertEqual(len(tb_rows), 2)
            for name, prob, cls in tb_rows:
                self.assertEqual(name, "TB_test_model")
                self.assertIsNotNone(prob)
                self.assertIsNotNone(cls)

    def test_resume_rescores_partial_day_missing_tb(self) -> None:
        """Reproduces the 2026-05-26 report: a day built *without* Triple
        Barrier that only ever writes fewer rows than the Master/registry
        expects (status ``Partial``, not ``Complete``) must still be picked
        up and rescored the next time TB is enabled on Build — Partial days
        get exactly the same TB-rescore treatment as Complete days."""
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = os.path.join(tmp, "data")
            research = os.path.join(tmp, "model_research")
            _write_primary_model(data_dir)
            _write_xgb_tb_package(data_dir, "TB_test_model", features=["delta", "gamma"])

            df = pd.DataFrame(
                {
                    "trading_day": ["2026-05-26", "2026-05-26", "2026-05-26"],
                    "timestamp": [1.0, 2.0, 3.0],
                    "token": ["t1", "t2", "t3"],
                    "strike": [24000.0, 24100.0, 24200.0],
                    "option_type": ["CE", "PE", "CE"],
                    "expiry": ["2026-05-27"] * 3,
                    "market": ["NIFTY"] * 3,
                    "spot": [24050.0] * 3,
                    "ltp": [100.0, 80.0, 90.0],
                    "minutes_to_expiry": [120.0, 100.0, 80.0],
                    "delta": [0.5, -0.4, 0.3],
                    "gamma": [0.01, 0.02, 0.015],
                    # t3 has no target -> dropped by the build (dropna on target),
                    # so only 2 of 3 dataset rows can ever be written this day —
                    # mirrors the 81,715-of-276,280 Partial coverage reported.
                    "future_ltp_5m": [101.0, 78.0, None],
                }
            )
            _write_master_parquet(data_dir, df)
            info = create_model_lab(data_dir, _sample_doc(), research_dir=research)

            with ModelLabStore(info.db_path) as store:
                lab_uuid = store.read_info().lab_uuid
                store.ensure_build_days(lab_uuid, ["2026-05-26"])
                # True dataset row count for the day (Master/registry) —
                # deliberately larger than what the build can ever write.
                store.set_day_rows_expected(lab_uuid, "2026-05-26", 3)

            common_patches = (
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

            # First build: TB checkbox left off (the historical/real-world
            # click that produced the 2026-05-26 report) — Partial coverage.
            with common_patches[0], common_patches[1], common_patches[2]:
                first = build_prediction_dataset(
                    data_dir,
                    info.db_path,
                    overwrite=False,
                    enrich_path_outcomes=False,
                    workers=1,
                    print_timing=False,
                )
            self.assertTrue(first.get("ok"), first)
            self.assertEqual(first.get("row_count"), 2)

            with ModelLabStore(info.db_path) as store:
                lab_uuid = store.read_info().lab_uuid
                days = {d["trading_day"]: d for d in store.list_build_days(lab_uuid)}
                tb_names = [
                    r[0]
                    for r in store.conn.execute(
                        "SELECT tb_model_name FROM prediction_dataset"
                    ).fetchall()
                ]
            self.assertEqual(days["2026-05-26"]["status"], DAY_PARTIAL)
            self.assertEqual(days["2026-05-26"]["row_count"], 2)
            self.assertEqual(tb_names, [None, None])

            # Second build: resume with TB enabled — the Partial day must be
            # reprocessed (it already qualifies as "pending" by status alone,
            # same as any other incomplete day) and its rows scored.
            with common_patches[0], common_patches[1], common_patches[2]:
                second = build_prediction_dataset(
                    data_dir,
                    info.db_path,
                    overwrite=False,
                    resume=True,
                    enrich_path_outcomes=False,
                    workers=1,
                    print_timing=False,
                    tb_model_name="TB_test_model",
                )
            self.assertTrue(second.get("ok"), second)
            self.assertEqual(second.get("days_processed"), ["2026-05-26"])

            with ModelLabStore(info.db_path) as store:
                lab_uuid = store.read_info().lab_uuid
                days = {d["trading_day"]: d for d in store.list_build_days(lab_uuid)}
                tb_rows = store.conn.execute(
                    "SELECT tb_model_name, tb_pred_probability, tb_pred_class "
                    "FROM prediction_dataset"
                ).fetchall()

            # Row coverage is still Partial (the underlying data shortfall is
            # unrelated to TB), but every written row is now TB-scored.
            self.assertEqual(days["2026-05-26"]["status"], DAY_PARTIAL)
            self.assertEqual(days["2026-05-26"]["row_count"], 2)
            self.assertEqual(len(tb_rows), 2)
            for name, prob, cls in tb_rows:
                self.assertEqual(name, "TB_test_model")
                self.assertIsNotNone(prob)
                self.assertIsNotNone(cls)


if __name__ == "__main__":
    unittest.main()

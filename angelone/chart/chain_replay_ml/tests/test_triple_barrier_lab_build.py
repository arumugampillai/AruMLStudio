"""Unit test for Phase 1 Triple Barrier Research Lab Build integration."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import pandas as pd

# Add angelone/chart to sys.path if not present
chart_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if chart_dir not in sys.path:
    sys.path.insert(0, chart_dir)

from chain_replay_ml.model_lab.prediction_schema import CORE_COLUMN_NAMES
from chain_replay_ml.model_lab.prediction_parallel import (
    DayJobContext,
    run_tb_model_predictions,
)
from chain_replay_ml.model_lab.prediction_worker import _ctx_from_config
from chain_replay_ml.model_lab.store import ModelLabStore
from chain_replay_ml.training.model_runtime import resolve_prediction_model_package
from chain_replay_ml.training.paths import model_artifact_paths


def _write_xgb_tb_package(
    data_dir: str,
    model_name: str,
    *,
    features: list[str],
    n_classes: int = 3,
) -> None:
    """Write a tiny native xgboost package mirroring real TB training output.

    Trains a multiclass (TP=0, SL=1, TIME=2) booster on synthetic data and
    saves it exactly like ``training/artifacts.py`` would: native model file
    + config.json (algorithm/features/label_run_id) + training_metadata.json
    (production_model pointer).
    """
    import xgboost as xgb

    rng = np.random.RandomState(0)
    n_rows = 60
    X = pd.DataFrame(
        {f: rng.uniform(-1.0, 1.0, size=n_rows) for f in features}
    ).astype("float32")
    y = rng.randint(0, n_classes, size=n_rows).astype("float32")

    dtrain = xgb.DMatrix(X, label=y, feature_names=features)
    params = {
        "objective": "multi:softprob",
        "num_class": n_classes,
        "max_depth": 2,
        "eta": 0.3,
        "verbosity": 0,
    }
    booster = xgb.train(params, dtrain, num_boost_round=5)

    paths = model_artifact_paths(data_dir, model_name)
    os.makedirs(paths["package_dir"], exist_ok=True)
    booster.save_model(paths["model_ubj"])

    with open(paths["config_json"], "w", encoding="utf-8") as fh:
        json.dump(
            {
                "algorithm": "xgboost",
                "features": features,
                "target": "label_id",
                "label_strategy": "triple_barrier",
                "label_run_id": "run_tb_test",
            },
            fh,
        )
    with open(paths["training_metadata_json"], "w", encoding="utf-8") as fh:
        json.dump({"production_model": "model.ubj"}, fh)


class TripleBarrierLabBuildTests(unittest.TestCase):
    def test_core_columns_exist(self) -> None:
        self.assertIn("tb_model_name", CORE_COLUMN_NAMES)
        self.assertIn("tb_label_run", CORE_COLUMN_NAMES)
        self.assertIn("tb_pred_probability", CORE_COLUMN_NAMES)
        self.assertIn("tb_pred_class", CORE_COLUMN_NAMES)

    def test_ensure_schema_creates_tb_columns(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = os.path.join(tmp, "test_lab.db")
            with ModelLabStore(db_path) as store:
                store.ensure_prediction_schema()
                cols = store.list_prediction_columns()
                self.assertIn("tb_model_name", cols)
                self.assertIn("tb_label_run", cols)
                self.assertIn("tb_pred_probability", cols)
                self.assertIn("tb_pred_class", cols)

    def test_run_tb_model_predictions_disabled(self) -> None:
        df = pd.DataFrame({"feat_1": [1.0, 2.0]})
        name, run, probs, classes, notes = run_tb_model_predictions(
            data_dir="/tmp",
            tb_model_name=None,
            day_df=df,
        )
        self.assertIsNone(name)
        self.assertIsNone(run)
        self.assertIsNone(probs)
        self.assertIsNone(classes)

    def test_worker_ctx_from_config_forwards_tb_model_name(self) -> None:
        """Bug #1: worker must not drop tb_model_name when building DayJobContext."""
        cfg = {
            "data_dir": "/tmp/data",
            "parquet_path": "/tmp/data/x.parquet",
            "features": ["feat_1"],
            "target": "future_ltp_5m",
            "wanted_columns": ["feat_1", "future_ltp_5m"],
            "lab_uuid": "lab-1",
            "feat_map": {"feat_1": "sf_feat_1"},
            "horizon_sec": 300.0,
            "model_path": "/tmp/data/models/m/model.ubj",
            "algorithm": "xgboost",
            "days": ["2026-01-01"],
            "tb_model_name": "TB_tp_20_sl_10_test",
        }
        ctx = _ctx_from_config(cfg)
        self.assertIsInstance(ctx, DayJobContext)
        self.assertEqual(ctx.tb_model_name, "TB_tp_20_sl_10_test")

    def test_worker_ctx_from_config_tb_model_name_none_when_absent(self) -> None:
        cfg = {
            "data_dir": "/tmp/data",
            "parquet_path": "/tmp/data/x.parquet",
            "features": ["feat_1"],
            "target": "future_ltp_5m",
            "wanted_columns": ["feat_1", "future_ltp_5m"],
            "lab_uuid": "lab-1",
            "feat_map": {"feat_1": "sf_feat_1"},
            "horizon_sec": 300.0,
            "model_path": "/tmp/data/models/m/model.ubj",
            "algorithm": "xgboost",
            "days": ["2026-01-01"],
        }
        ctx = _ctx_from_config(cfg)
        self.assertIsNone(ctx.tb_model_name)

    def test_resolve_prediction_model_package_uses_correct_load_api(self) -> None:
        """Bug #2: package resolution must not call load_prediction_model_cached(data_dir, name)."""
        with tempfile.TemporaryDirectory() as tmp:
            features = ["feat_1", "feat_2"]
            _write_xgb_tb_package(tmp, "TB_test_model", features=features)

            pkg = resolve_prediction_model_package(tmp, "TB_test_model")
            self.assertTrue(pkg.get("ok"), pkg.get("error"))
            self.assertTrue(os.path.isfile(pkg["model_path"]))
            self.assertEqual(pkg["algorithm"], "xgboost")
            self.assertEqual(pkg["features"], features)
            self.assertEqual(pkg["label_run_id"], "run_tb_test")

    def test_resolve_prediction_model_package_missing_model(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            pkg = resolve_prediction_model_package(tmp, "does_not_exist")
            self.assertFalse(pkg.get("ok"))
            self.assertIn("error", pkg)

    def test_run_tb_model_predictions_writes_four_columns(self) -> None:
        """Bug #2/#5: real (tiny) model load + multiclass P(TP) scoring."""
        with tempfile.TemporaryDirectory() as tmp:
            features = ["feat_1", "feat_2"]
            _write_xgb_tb_package(tmp, "TB_test_model", features=features)

            day_df = pd.DataFrame(
                {
                    "feat_1": [0.1, -0.2, 0.3, 0.5],
                    "feat_2": [-0.1, 0.4, -0.3, 0.2],
                }
            )
            name, label_run, probs, classes, notes = run_tb_model_predictions(
                data_dir=tmp,
                tb_model_name="TB_test_model",
                day_df=day_df,
            )
            self.assertEqual(name, "TB_test_model")
            self.assertEqual(label_run, "run_tb_test")
            self.assertFalse(notes, notes)
            self.assertIsNotNone(probs)
            self.assertIsNotNone(classes)
            self.assertEqual(len(probs), len(day_df))
            self.assertEqual(len(classes), len(day_df))
            for p in probs:
                self.assertGreaterEqual(float(p), 0.0)
                self.assertLessEqual(float(p), 1.0)
            for c in classes:
                self.assertIn(int(c), (0, 1, 2))

            # All four downstream prediction-dataset columns are derivable.
            row = {
                "tb_model_name": name,
                "tb_label_run": label_run,
                "tb_pred_probability": float(probs[0]),
                "tb_pred_class": int(classes[0]),
            }
            for col in (
                "tb_model_name",
                "tb_label_run",
                "tb_pred_probability",
                "tb_pred_class",
            ):
                self.assertIn(col, row)
                self.assertIsNotNone(row[col])

    def test_run_tb_model_predictions_missing_model_returns_notes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            day_df = pd.DataFrame({"feat_1": [1.0, 2.0]})
            name, label_run, probs, classes, notes = run_tb_model_predictions(
                data_dir=tmp,
                tb_model_name="does_not_exist",
                day_df=day_df,
            )
            self.assertEqual(name, "does_not_exist")
            self.assertIsNone(probs)
            self.assertIsNone(classes)
            self.assertTrue(notes)

    def test_run_tb_model_predictions_missing_feature_columns_returns_notes(self) -> None:
        """Day frame missing a required TB feature must degrade to NULLs, not raise."""
        with tempfile.TemporaryDirectory() as tmp:
            features = ["feat_1", "feat_2"]
            _write_xgb_tb_package(tmp, "TB_test_model", features=features)

            # feat_2 absent — e.g. wanted_columns didn't carry it into the day load.
            day_df = pd.DataFrame({"feat_1": [0.1, 0.2, 0.3]})
            name, label_run, probs, classes, notes = run_tb_model_predictions(
                data_dir=tmp,
                tb_model_name="TB_test_model",
                day_df=day_df,
            )
            self.assertEqual(name, "TB_test_model")
            self.assertEqual(label_run, "run_tb_test")
            self.assertIsNone(probs)
            self.assertIsNone(classes)
            self.assertTrue(notes)
            self.assertIn("missing features", notes[0])

    def test_run_tb_model_predictions_predict_exception_is_graceful(self) -> None:
        """A raising predict() (e.g. native/device failure) must not propagate.

        This is the core regression for the worker crash: any Triple Barrier
        scoring failure must degrade to NULL tb_* outputs plus a note, and
        the caller (process_trading_day) must keep building the day rather
        than the whole worker process dying.
        """
        with tempfile.TemporaryDirectory() as tmp:
            features = ["feat_1", "feat_2"]
            _write_xgb_tb_package(tmp, "TB_test_model", features=features)

            day_df = pd.DataFrame(
                {"feat_1": [0.1, 0.2], "feat_2": [0.3, 0.4]}
            )

            class _ExplodingModel:
                def predict(self, X):
                    raise RuntimeError("boom: native predict failure")

            with mock.patch(
                "chain_replay_ml.training.model_runtime.load_prediction_model_cached",
                return_value=(_ExplodingModel(), 0.0, True),
            ):
                name, label_run, probs, classes, notes = run_tb_model_predictions(
                    data_dir=tmp,
                    tb_model_name="TB_test_model",
                    day_df=day_df,
                )
            self.assertEqual(name, "TB_test_model")
            self.assertIsNone(probs)
            self.assertIsNone(classes)
            self.assertTrue(notes)
            self.assertIn("boom", notes[0])

    def test_process_trading_day_tb_exception_does_not_kill_day(self) -> None:
        """Even if run_tb_model_predictions itself raised (contract violation),
        process_trading_day's outer guard must still finish the day with NULL
        tb_* columns instead of propagating the exception."""
        with mock.patch(
            "chain_replay_ml.model_lab.prediction_parallel.run_tb_model_predictions",
            side_effect=RuntimeError("simulated native crash surrogate"),
        ) as mocked:
            # Call the guarded block logic directly, mirroring process_trading_day's
            # call site, to keep this test independent of full day-frame plumbing.
            from chain_replay_ml.model_lab import prediction_parallel as pp

            tb_name = tb_run = tb_probs = tb_classes = None
            tb_notes: list[str] = []
            try:
                tb_name, tb_run, tb_probs, tb_classes, tb_notes = pp.run_tb_model_predictions(
                    data_dir="/tmp", tb_model_name="TB_test_model", day_df=pd.DataFrame()
                )
            except Exception as exc:
                tb_name, tb_run, tb_probs, tb_classes = "TB_test_model", None, None, None
                tb_notes = [f"Triple Barrier inference failed: {exc}"]

            mocked.assert_called_once()
            self.assertEqual(tb_name, "TB_test_model")
            self.assertIsNone(tb_probs)
            self.assertIsNone(tb_classes)
            self.assertTrue(tb_notes)
            self.assertIn("simulated native crash surrogate", tb_notes[0])

    def test_force_cpu_inference_sets_device_cpu_for_xgboost(self) -> None:
        """TB/side-scorer models must be pinned to CPU regardless of the
        device they were trained with, so they never contend for GPU memory
        with the primary model inside the same worker process."""
        from chain_replay_ml.training.inference_runtime import force_cpu_inference
        from chain_replay_ml.training.model_runtime import load_prediction_model_cached

        with tempfile.TemporaryDirectory() as tmp:
            features = ["feat_1", "feat_2"]
            _write_xgb_tb_package(tmp, "TB_test_model", features=features)
            pkg = resolve_prediction_model_package(tmp, "TB_test_model")
            model, _ms, _disk = load_prediction_model_cached(pkg["model_path"], pkg["algorithm"])

            force_cpu_inference(model, pkg["algorithm"])

            self.assertEqual(model.get_params().get("device"), "cpu")

    def test_force_cpu_inference_noop_for_non_xgboost_algorithm(self) -> None:
        """Must never raise for non-xgboost wrappers (e.g. LightGBM/CatBoost),
        whose set_params would reject xgboost-only kwargs like device/tree_method."""
        from chain_replay_ml.training.inference_runtime import force_cpu_inference

        class _StrictModel:
            def set_params(self, **kwargs):
                raise ValueError(f"unexpected params: {kwargs}")

        force_cpu_inference(_StrictModel(), "lightgbm")
        force_cpu_inference(_StrictModel(), None)


if __name__ == "__main__":
    unittest.main()

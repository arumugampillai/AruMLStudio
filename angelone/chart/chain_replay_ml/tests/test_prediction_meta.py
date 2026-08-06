"""Unit tests for prediction meta dataset builder helpers."""

from __future__ import annotations

import os
import tempfile
import unittest

from chain_replay_ml.prediction_meta.meta_features import (
    build_prediction_id,
    confidence_features,
    extend_ensemble_meta,
    model_deltas_and_ranks,
    resolve_minutes_to_expiry,
)
from chain_replay_ml.prediction_meta.outcomes import (
    compute_prediction_quality,
    map_point_outcomes,
)
from chain_replay_ml.prediction_meta.model_registry import (
    resolve_or_register_prediction_version,
    read_model_registry,
)
from chain_replay_ml.prediction_meta.status import read_prediction_meta_status
from chain_replay_ml.prediction_meta.store import PredictionMetaStore


class TestEnsembleMeta(unittest.TestCase):
    def test_empty_preds(self) -> None:
        meta = extend_ensemble_meta([], models_ok=0, models_failed=3, entry_ltp=100.0)
        self.assertIsNone(meta["ensemble_mean"])
        self.assertEqual(meta["models_failed"], 3)
        self.assertIsNone(meta["prediction_min"])

    def test_three_models(self) -> None:
        meta = extend_ensemble_meta([100.0, 102.0, 98.0], models_ok=3, models_failed=0, entry_ltp=100.0)
        self.assertAlmostEqual(meta["ensemble_mean"], 100.0)
        self.assertAlmostEqual(meta["ensemble_spread"], 4.0)
        self.assertEqual(meta["prediction_min"], 98.0)
        self.assertEqual(meta["mean_minus_current_ltp"], 0.0)


class TestRanking(unittest.TestCase):
    def test_deltas_and_ranks(self) -> None:
        deltas, ranks = model_deltas_and_ranks([98.0, 100.0, 102.0], ensemble_mean=100.0)
        self.assertEqual(deltas["model_1_delta_from_mean"], -2.0)
        self.assertEqual(deltas["model_3_delta_from_mean"], 2.0)
        self.assertEqual(ranks["model_1_rank"], 1.0)
        self.assertEqual(ranks["model_3_rank"], 3.0)


class TestPredictionId(unittest.TestCase):
    def test_human_readable_id(self) -> None:
        pid = build_prediction_id(
            trading_day="2026-07-03",
            timestamp=1720000000.0,
            strike=24350,
            option_type="CE",
            token="12345",
            prediction_version=1,
        )
        self.assertTrue(pid.startswith("2026-07-03|"))
        self.assertIn("|24350|CE|12345|v1", pid)


class TestOutcomes(unittest.TestCase):
    def test_map_point_outcomes(self) -> None:
        row = {
            "future_ltp_30s": 101.5,
            "future_ltp_1m": 102.0,
            "future_ltp_180s": 103.0,
            "future_ltp_5m": 105.0,
        }
        out = map_point_outcomes(row, 100.0)
        self.assertEqual(out["actual_30s_ltp"], 101.5)
        self.assertEqual(out["actual_3m_ltp"], 103.0)

    def test_prediction_quality(self) -> None:
        q = compute_prediction_quality(ensemble_mean=105.0, entry_ltp=100.0, actual_ltp=110.0)
        self.assertEqual(q["direction_correct"], 1.0)
        self.assertAlmostEqual(q["prediction_error"], -5.0)

    def test_minutes_to_expiry_from_row(self) -> None:
        mte = resolve_minutes_to_expiry({"minutes_to_expiry": 120.5}, timestamp=100.0)
        self.assertEqual(mte, 120.5)


class TestConfidence(unittest.TestCase):
    def test_range_pct(self) -> None:
        conf = confidence_features([98.0, 102.0], mean=100.0, median=100.0, entry_ltp=100.0)
        self.assertAlmostEqual(conf["prediction_range_pct"], 4.0)


class TestPredictionMetaStore(unittest.TestCase):
    def test_append_only_and_prediction_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "prediction_meta_nifty_3s.db")
            with PredictionMetaStore(path) as store:
                store.start_job(rows_total=100, resume=False)
                store.prepare_insert([
                    "prediction_id", "trading_day", "timestamp", "token", "ensemble_mean",
                ])
                row = {
                    "prediction_id": "2026-07-02|09:15:03|24350|CE|12345|v1",
                    "trading_day": "2026-07-02",
                    "timestamp": 100.0,
                    "token": "12345",
                    "ensemble_mean": 50.0,
                }
                store.insert_rows([row])
                store.insert_rows([row])  # duplicate ignored
                store.update_checkpoint(
                    rows_done=1,
                    trading_day="2026-07-02",
                    timestamp=100.0,
                    token="12345",
                )
                store.mark_complete(1)

            status = read_prediction_meta_status(tmp, db_path=path)
            self.assertTrue(status["exists"])
            self.assertEqual(status["status"], "complete")
            self.assertEqual(status["row_count"], 1)


class TestModelRegistry(unittest.TestCase):
    def test_register_and_reuse_version(self) -> None:
        specs = [
            {"model_name": "Model_A", "target": "future_ltp_5m", "mae": 1.2, "rmse": 1.5},
            {"model_name": "Model_B", "target": "future_ltp_5m", "mae": 1.1, "rmse": 1.4},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "prediction_meta_nifty_3s.db")
            with PredictionMetaStore(path) as store:
                v1 = resolve_or_register_prediction_version(
                    store.conn, tmp, specs, model_registry_version="sig-v1",
                )
                v1b = resolve_or_register_prediction_version(
                    store.conn, tmp, specs, model_registry_version="sig-v1",
                )
                v2 = resolve_or_register_prediction_version(
                    store.conn, tmp, specs, model_registry_version="sig-v2",
                )
                slots_v1 = read_model_registry(store.conn, v1)

            self.assertEqual(v1, 1)
            self.assertEqual(v1, v1b)
            self.assertEqual(v2, 2)
            self.assertEqual(len(slots_v1), 2)
            self.assertEqual(slots_v1[0]["slot"], "model_1")
            self.assertEqual(slots_v1[0]["model_name"], "Model_A")
            self.assertEqual(slots_v1[1]["slot"], "model_2")
            self.assertEqual(slots_v1[1]["model_name"], "Model_B")


if __name__ == "__main__":
    unittest.main()

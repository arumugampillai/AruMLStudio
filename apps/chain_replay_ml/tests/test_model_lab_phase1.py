"""Phase 1 Model Lab — create workspace + immutable snapshots."""

from __future__ import annotations

import os
import sqlite3
import tempfile
import unittest

from chain_replay_ml.model_lab.paths import lab_db_filename, list_lab_db_paths, next_lab_version
from chain_replay_ml.model_lab.service import create_model_lab, find_latest_lab, load_lab
from chain_replay_ml.model_lab.snapshots import (
    extract_feature_ranking,
    extract_original_feature_count,
    extract_selected_features,
)
from chain_replay_ml.model_lab.store import LAB_PHASE, LAB_SCHEMA_VERSION, STATUS_READY


def _sample_doc() -> dict:
    return {
        "model_name": "Future_LTP_5m_WF_239f_XGB_0223_13",
        "is_walk_forward": True,
        "selected_features": ["ltp", "delta", "gamma"],
        "metadata": {
            "model_name": "Future_LTP_5m_WF_239f_XGB_0223_13",
            "model_version": "1",
            "algorithm": "xgboost",
            "target": "future_ltp_5m",
            "dataset": "Master_NIFTY_239f",
            "feature_count": 3,
            "row_count": 1000,
            "trained_at": "2026-07-01T00:00:00+00:00",
        },
        "config": {
            "algorithm": "xgboost",
            "target": "future_ltp_5m",
            "dataset": "Master_NIFTY_239f",
            "parameters": {"max_depth": 6},
            "split": {"strategy": "walk_forward", "walk_forward": {"n_folds": 5}},
        },
        "metrics": {
            "validation": {"mae": 1.2, "rmse": 2.0},
            "test": {"mae": 1.5, "rmse": 2.3},
            "production_walk_forward": {"mae": 1.4},
        },
        "training_summary": {"trees_trained": 100, "training_time_sec": 12},
        "walk_forward": {
            "available": True,
            "selected_features": {
                "available": True,
                "rows": [
                    {
                        "feature": "ltp",
                        "final_rank": 1,
                        "Selected in Folds": "5/5",
                        "Selected": "Yes",
                        "Gain Importance %": 40.0,
                    },
                    {
                        "feature": "delta",
                        "final_rank": 2,
                        "Selected in Folds": "4/5",
                        "Selected": "Yes",
                        "Gain Importance %": 30.0,
                    },
                    {
                        "feature": "gamma",
                        "final_rank": 3,
                        "Selected in Folds": "3/5",
                        "Selected": "Yes",
                        "Gain Importance %": 20.0,
                    },
                ],
            },
            "display": {"n_folds": 5, "selected_feature_count": 3},
            "summary": {
                "available": True,
                "data": {
                    "aggregated": {"mean_mae": 1.1},
                    "feature_selection": {"started_features": 239},
                },
            },
        },
    }


class ModelLabPhase1Tests(unittest.TestCase):
    def test_ranking_prefers_wf_csv_rows(self) -> None:
        ranking = extract_feature_ranking(_sample_doc())
        self.assertTrue(ranking["available"])
        self.assertEqual(ranking["source"], "walk_forward/selected_features.csv")
        self.assertEqual(len(ranking["rows"]), 3)

    def test_selected_features(self) -> None:
        feats = extract_selected_features(_sample_doc())
        self.assertEqual(feats, ["ltp", "delta", "gamma"])

    def test_original_feature_count(self) -> None:
        self.assertEqual(extract_original_feature_count(_sample_doc(), 3), 239)

    def test_create_lab_metadata_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = os.path.join(tmp, "data")
            research = os.path.join(tmp, "model_research")
            os.makedirs(data_dir, exist_ok=True)
            # Provide a tiny binary for checksum
            models = os.path.join(data_dir, "models", "Future_LTP_5m_WF_239f_XGB_0223_13")
            os.makedirs(models, exist_ok=True)
            with open(os.path.join(models, "model.ubj"), "wb") as fh:
                fh.write(b"fake-model-bytes")

            doc = _sample_doc()
            info = create_model_lab(
                data_dir,
                doc,
                research_dir=research,
                lab_name="Gamma Research Lab",
                description="Expiry + gamma investigation",
                purpose="Gamma Research",
            )
            self.assertEqual(info.version, 1)
            self.assertEqual(info.status, STATUS_READY)
            self.assertEqual(info.phase, LAB_PHASE)
            self.assertEqual(info.lab_schema_version, LAB_SCHEMA_VERSION)
            self.assertTrue(info.lab_uuid)
            self.assertEqual(len(info.lab_uuid), 36)
            self.assertEqual(info.lab_name, "Gamma Research Lab")
            self.assertEqual(info.purpose, "Gamma Research")
            self.assertEqual(info.description, "Expiry + gamma investigation")
            self.assertTrue(info.model_checksum)
            self.assertEqual(len(info.model_checksum or ""), 64)
            self.assertEqual(info.original_feature_count, 239)
            self.assertEqual(info.selected_feature_count, 3)
            self.assertTrue(os.path.isfile(info.db_path))
            self.assertEqual(os.path.basename(info.db_path), lab_db_filename(doc["model_name"], 1))

            conn = sqlite3.connect(info.db_path)
            try:
                tables = {
                    r[0]
                    for r in conn.execute(
                        "SELECT name FROM sqlite_master WHERE type='table'"
                    ).fetchall()
                }
                for required in (
                    "model_lab_info",
                    "prediction_dataset",
                    "prediction_explanation",
                    "research_history",
                    "knowledge",
                    "settings",
                ):
                    self.assertIn(required, tables)
                pred_n = conn.execute("SELECT COUNT(*) FROM prediction_dataset").fetchone()[0]
                self.assertEqual(pred_n, 0)
            finally:
                conn.close()

            latest = find_latest_lab(doc["model_name"], research_dir=research)
            self.assertIsNotNone(latest)
            assert latest is not None
            self.assertEqual(latest.lab_uuid, info.lab_uuid)

            self.assertEqual(next_lab_version(doc["model_name"], research_dir=research), 2)
            info2 = create_model_lab(data_dir, doc, research_dir=research)
            self.assertEqual(info2.version, 2)
            self.assertNotEqual(info2.lab_uuid, info.lab_uuid)
            self.assertEqual(len(list_lab_db_paths(doc["model_name"], research_dir=research)), 2)

    def test_ranking_unavailable_message(self) -> None:
        ranking = extract_feature_ranking({"model_name": "x"})
        self.assertFalse(ranking["available"])
        self.assertIn("unavailable", ranking.get("message", "").lower())

    def test_load_missing(self) -> None:
        self.assertIsNone(load_lab(os.path.join(tempfile.gettempdir(), "no_such_lab.db")))


if __name__ == "__main__":
    unittest.main()

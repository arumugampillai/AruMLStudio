"""Tests for model lifecycle registry / history store."""

from __future__ import annotations

import json
import os
import sqlite3
import tempfile
import unittest
from typing import Any
from unittest.mock import patch

from chain_replay_ml.training.config import normalize_training_config
from chain_replay_ml.training.lifecycle_store import (
    DEPRECATED_LIFECYCLE_METRIC_KEYS,
    _enrich_history_row_from_disk,
    _enrich_lifecycle_history_rows,
    build_improvement_summary,
    get_history_by_model_name,
    get_model_lifecycle_view,
    lifecycle_db_path,
    list_history_for_model,
    list_model_champions,
    record_training_history,
    rebuild_lifecycle_index,
)


def _cfg(**kwargs) -> Any:
    base = {
        "dataset": "MS_185f_3s_test",
        "target": "future_ltp_5m",
        "algorithm": "xgboost",
        "features": ["f1", "f2", "f3"],
        "parameters": {"learning_rate": 0.05, "max_depth": 6},
        "model_version": "1.0",
    }
    base.update(kwargs)
    return normalize_training_config(base)


def _write_package(data_dir: str, model_name: str, metrics: dict[str, Any], *, config: Any | None = None) -> str:
    pkg = os.path.join(data_dir, "models", model_name)
    os.makedirs(pkg, exist_ok=True)
    with open(os.path.join(pkg, "metrics.json"), "w", encoding="utf-8") as fh:
        json.dump(metrics, fh)
    cfg = config or _cfg()
    with open(os.path.join(pkg, "config.json"), "w", encoding="utf-8") as fh:
        json.dump(cfg.to_dict() if hasattr(cfg, "to_dict") else dict(cfg), fh)
    return pkg


class LifecycleStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp()
        os.makedirs(os.path.join(self.tmp, "models"), exist_ok=True)

    def test_new_model_creates_registry_and_history_without_persisting_metrics(self) -> None:
        metrics = {
            "validation": {"mae": 3.82, "rmse": 5.21, "directional_accuracy_pct": 72.1},
            "composite_scores": {"production_composite": 0.41},
        }
        result = record_training_history(
            data_dir=self.tmp,
            model_name="Future_LTP_5m_XGB_v1",
            trained_at="2026-07-01T10:00:00+00:00",
            config=_cfg(),
            metrics=metrics,
            metadata={"row_count": 450_973, "trading_days": 1},
            matrix_report={"x_shape": [450_973, 3]},
            lineage=None,
        )
        self.assertEqual(result["version_label"], "v1")
        self.assertEqual(result["lifecycle"], "new_model")
        champions = list_model_champions(self.tmp)
        self.assertEqual(len(champions), 1)
        self.assertEqual(champions[0]["current_model_name"], "Future_LTP_5m_XGB_v1")
        self.assertEqual(champions[0]["current_metrics"], {})

        hist = list_history_for_model(self.tmp, model_id=result["model_id"])
        self.assertEqual(len(hist), 1)
        # Deprecated DB columns must not be used for display (stripped on read).
        self.assertIsNone(hist[0]["mae"])
        self.assertIsNone(hist[0]["rmse"])

        # Raw SQLite also stores NULL / empty for metric columns.
        con = sqlite3.connect(lifecycle_db_path(self.tmp))
        row = con.execute(
            "SELECT mae, rmse, metrics_json FROM model_history WHERE model_name=?",
            ("Future_LTP_5m_XGB_v1",),
        ).fetchone()
        champ = con.execute("SELECT current_metrics_json FROM model_registry").fetchone()
        con.close()
        self.assertIsNone(row[0])
        self.assertIsNone(row[1])
        self.assertEqual(row[2], "{}")
        self.assertEqual(champ[0], "{}")

    def test_retrain_increments_version_and_records_changes(self) -> None:
        metrics_v1 = {
            "validation": {"mae": 3.82, "rmse": 5.21, "directional_accuracy_pct": 72.1},
            "composite_scores": {"production_composite": 0.41},
        }
        record_training_history(
            data_dir=self.tmp,
            model_name="FamilyRoot_v1",
            trained_at="2026-07-01T10:00:00+00:00",
            config=_cfg(model_version="1.0"),
            metrics=metrics_v1,
            metadata={"row_count": 450_973, "trading_days": 1},
            matrix_report={"x_shape": [450_973, 3]},
        )
        _write_package(self.tmp, "FamilyRoot_v1", metrics_v1, config=_cfg(model_version="1.0"))

        metrics_v2 = {
            "validation": {"mae": 2.41, "rmse": 4.11, "directional_accuracy_pct": 81.2},
            "composite_scores": {"production_composite": 0.53},
        }
        result = record_training_history(
            data_dir=self.tmp,
            model_name="FamilyRoot_v2",
            trained_at="2026-07-07T10:00:00+00:00",
            config=_cfg(model_version="2.0"),
            metrics=metrics_v2,
            metadata={"row_count": 6_890_000, "trading_days": 15},
            matrix_report={"x_shape": [6_890_000, 3]},
            lineage={
                "parent_model_id": "FamilyRoot_v1",
                "ancestor_model_id": "FamilyRoot_v1",
                "generation": 2,
                "lifecycle_mode": "retrain",
            },
        )
        _write_package(self.tmp, "FamilyRoot_v2", metrics_v2, config=_cfg(model_version="2.0"))

        self.assertEqual(result["version_label"], "v2")
        hist = list_history_for_model(self.tmp, model_id="FamilyRoot_v1")
        self.assertEqual(len(hist), 2)
        hist = [_enrich_history_row_from_disk(self.tmp, h) for h in hist]
        imp = build_improvement_summary(hist)
        self.assertEqual(imp["current_version"], "v2")
        self.assertIn("2.41", imp["improvement_since_initial"]["mae"])
        champions = list_model_champions(self.tmp)
        self.assertEqual(champions[0]["current_model_name"], "FamilyRoot_v2")

    def test_get_model_lifecycle_view_resolves_metrics_from_package(self) -> None:
        metrics = {
            "validation": {"mae": 1.5, "rmse": 2.5, "directional_accuracy_pct": 70.0},
            "composite_scores": {"production_composite": 0.5},
        }
        record_training_history(
            data_dir=self.tmp,
            model_name="ViewModel_v1",
            trained_at="2026-07-01T10:00:00+00:00",
            config=_cfg(),
            metrics=metrics,
            metadata={"row_count": 1000},
            matrix_report={"x_shape": [1000, 3]},
        )
        _write_package(self.tmp, "ViewModel_v1", metrics)

        view = get_model_lifecycle_view(self.tmp, model_name="ViewModel_v1")
        self.assertEqual(len(view["history"]), 1)
        self.assertIn("improvement", view)
        self.assertEqual(view["history"][0]["mae"], 1.5)
        self.assertEqual(view["history"][0]["metrics_source"], "package")
        self.assertTrue(str(view["history"][0].get("package_path") or "").endswith("ViewModel_v1"))
        self.assertEqual(view["improvement"]["current_metrics"]["mae"], 1.5)
        self.assertEqual(view["champion"]["current_metrics"]["mae"], 1.5)

    def test_stale_db_metrics_are_ignored_in_favor_of_package(self) -> None:
        """Legacy DB metric columns must never win over package metrics.json."""
        metrics = {
            "validation": {"mae": 1.0, "rmse": 2.0, "directional_accuracy_pct": 90.0},
        }
        record_training_history(
            data_dir=self.tmp,
            model_name="StaleIgnore_v1",
            trained_at="2026-07-01T10:00:00+00:00",
            config=_cfg(),
            metrics=metrics,
            metadata={"row_count": 100},
            matrix_report={"x_shape": [100, 3]},
        )
        _write_package(self.tmp, "StaleIgnore_v1", metrics)

        # Simulate legacy stale snapshot in SQLite.
        con = sqlite3.connect(lifecycle_db_path(self.tmp))
        con.execute(
            "UPDATE model_history SET mae=99.0, rmse=88.0, directional_accuracy_pct=11.0 WHERE model_name=?",
            ("StaleIgnore_v1",),
        )
        con.execute(
            "UPDATE model_registry SET current_metrics_json=? WHERE current_model_name=?",
            (json.dumps({"mae": 99.0, "rmse": 88.0, "directional_accuracy_pct": 11.0}), "StaleIgnore_v1"),
        )
        con.commit()
        con.close()

        view = get_model_lifecycle_view(self.tmp, model_name="StaleIgnore_v1")
        self.assertEqual(view["history"][0]["mae"], 1.0)
        self.assertEqual(view["history"][0]["rmse"], 2.0)
        self.assertEqual(view["history"][0]["directional_accuracy_pct"], 90.0)
        self.assertNotEqual(view["champion"]["current_metrics"].get("mae"), 99.0)

    @patch("chain_replay_ml.training.lifecycle_store._list_model_packages_for_index")
    def test_rebuild_skips_existing(self, mock_list) -> None:
        record_training_history(
            data_dir=self.tmp,
            model_name="ExistingModel",
            trained_at="2026-07-01T10:00:00+00:00",
            config=_cfg(),
            metrics={"validation": {"mae": 1.0}},
            metadata={"row_count": 500},
            matrix_report={"x_shape": [500, 3]},
        )
        mock_list.return_value = [{"model_name": "ExistingModel", "trained_at": "2026-07-01T10:00:00+00:00"}]
        stats = rebuild_lifecycle_index(self.tmp)
        self.assertEqual(stats["skipped"], 1)
        self.assertEqual(stats["indexed"], 0)
        self.assertIsNotNone(get_history_by_model_name(self.tmp, "ExistingModel"))

    @patch("chain_replay_ml.training.lifecycle_store._version_package_context")
    def test_enrich_lifecycle_history_feature_deltas(self, mock_ctx) -> None:
        mock_ctx.side_effect = [
            {"selected_features": ["a", "b", "c"], "sampling_interval_sec": 3},
            {"selected_features": ["a", "b"], "sampling_interval_sec": 3},
        ]
        history = [
            {
                "version_label": "v1",
                "model_name": "M_v1",
                "row_count": 457622,
                "trading_days": 203,
                "feature_count": 3,
                "dataset": "DS_test",
                "target": "future_ltp_5m",
            },
            {
                "version_label": "v2",
                "model_name": "M_v2",
                "row_count": 457622,
                "trading_days": 203,
                "feature_count": 2,
                "dataset": "DS_test",
                "target": "future_ltp_5m",
            },
        ]
        enriched = _enrich_lifecycle_history_rows(self.tmp, history)
        self.assertEqual(enriched[0]["features_removed"], 0)
        self.assertEqual(enriched[0]["features_added"], 0)
        self.assertEqual(enriched[0]["sampling_interval_sec"], 3)
        self.assertEqual(enriched[1]["features_removed"], 1)
        self.assertEqual(enriched[1]["features_added"], 0)
        self.assertEqual(enriched[1]["selected_feature_count"], 2)
        self.assertEqual(enriched[1]["total_feature_count"], 3)

    @patch("chain_replay_ml.training.registry.load_model_detail")
    def test_enrich_history_row_uses_authoritative_package_metrics(self, mock_load_detail) -> None:
        metrics = {
            "validation": {"mae": 2.5, "rmse": 4.1, "directional_accuracy_pct": 80.0},
            "composite_scores": {"production_composite": 0.42},
        }
        record_training_history(
            data_dir=self.tmp,
            model_name="LightMetrics_v1",
            trained_at="2026-07-01T10:00:00+00:00",
            config=_cfg(),
            metrics=metrics,
            metadata={"row_count": 1000, "trading_days": 5},
            matrix_report={"x_shape": [1000, 3]},
        )
        _write_package(self.tmp, "LightMetrics_v1", metrics)
        hist = list_history_for_model(self.tmp, model_name="LightMetrics_v1")[0]
        # Even if something re-injects stale DB values, package wins.
        hist = dict(hist)
        hist["mae"] = 99.0
        hist["rmse"] = 99.0
        enriched = _enrich_history_row_from_disk(self.tmp, hist)
        self.assertEqual(enriched["mae"], 2.5)
        self.assertEqual(enriched["rmse"], 4.1)
        self.assertEqual(enriched["metrics_source"], "package")
        mock_load_detail.assert_not_called()

    def test_deprecated_metric_keys_documented(self) -> None:
        self.assertIn("mae", DEPRECATED_LIFECYCLE_METRIC_KEYS)
        self.assertIn("directional_accuracy_pct", DEPRECATED_LIFECYCLE_METRIC_KEYS)
        self.assertIn("composite_score", DEPRECATED_LIFECYCLE_METRIC_KEYS)


if __name__ == "__main__":
    unittest.main()

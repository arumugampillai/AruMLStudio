"""Unit tests for Create Model package comparison helper."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from chain_replay_ml.dataset_engine.compare_create_model_runs import (
    compare_create_model_packages,
)


def _write_package(
    root: Path,
    *,
    feature_count: int = 10,
    row_count: int = 1000,
    mae: float = 1.5,
    mean_mae: float = 1.6,
    features: list[str] | None = None,
    load: dict | None = None,
) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "schema_hash.txt").write_text("abc123", encoding="utf-8")
    (root / "validation_hash.txt").write_text("val456", encoding="utf-8")
    meta = {
        "feature_count": feature_count,
        "row_count": row_count,
        "features": features or [f"f{i}" for i in range(feature_count)],
    }
    if load:
        meta["dataset_load"] = load
    (root / "metadata.json").write_text(json.dumps(meta), encoding="utf-8")
    metrics = {
        "test": {"rmse": mae + 0.2, "mae": mae, "r2": 0.99, "directional_accuracy_pct": 80.0, "mape": 2.0, "medae": 1.0},
        "production_walk_forward": {
            "mean_rmse": mean_mae + 0.2,
            "mean_mae": mean_mae,
            "mean_r2": 0.98,
            "mean_directional_accuracy_pct": 79.0,
            "mean_mape": 2.1,
        },
    }
    (root / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    wf = root / "walk_forward"
    wf.mkdir(exist_ok=True)
    feats = features or [f"f{i}" for i in range(feature_count)]
    lines = ["feature,selected"] + [f"{f},Yes" for f in feats]
    (wf / "selected_features.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return root


class CompareCreateModelRunsTests(unittest.TestCase):
    def test_identical_packages_pass(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            load = {
                "rows_returned": 1000,
                "columns_returned": 12,
                "partitions_scanned": 2,
                "load_time_sec": 1.5,
                "peak_rss_mb": 800,
            }
            off = _write_package(base / "off", load=load)
            on = _write_package(
                base / "on",
                load={**load, "load_time_sec": 1.2, "peak_rss_mb": 400},
            )
            report = compare_create_model_packages(off, on)
            self.assertTrue(report["ok"], report)
            self.assertTrue(report["transparent_replacement"])

    def test_metric_divergence_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            off = _write_package(base / "off", mae=1.5)
            on = _write_package(base / "on", mae=2.0)
            report = compare_create_model_packages(off, on)
            self.assertFalse(report["ok"])
            hold = next(c for c in report["checks"] if c["category"].startswith("Holdout"))
            self.assertFalse(hold["ok"])


if __name__ == "__main__":
    unittest.main()

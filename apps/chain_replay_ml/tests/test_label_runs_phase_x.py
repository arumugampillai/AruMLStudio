"""Phase X — Label Runs registry + join tests."""

from __future__ import annotations

import os
import tempfile
import unittest

import pandas as pd

from chain_replay_ml.label_runs import (
    create_triple_barrier_label_run,
    join_feature_frame_with_label_run,
    list_label_runs,
    promote_feature_column_to_label_run,
    write_label_run,
)
from chain_replay_ml.training.config import normalize_training_config


class LabelRunsPhaseXTests(unittest.TestCase):
    def test_write_list_join(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            # Feature dataset parquet
            from chain_replay_ml.dataset_builder.writer import datasets_dir, _safe_filename

            out = datasets_dir(tmp)
            os.makedirs(out, exist_ok=True)
            name = "feat_ds"
            safe = _safe_filename(name)
            feat = pd.DataFrame(
                {
                    "master_row_id": ["a", "b", "c", "d"],
                    "trading_day": ["2026-07-01"] * 4,
                    "timestamp": [1.0, 2.0, 3.0, 4.0],
                    "token": ["T1", "T1", "T1", "T1"],
                    "feat_a": [1.0, 2.0, 3.0, 4.0],
                    "future_ltp_5m": [10.0, 20.0, 30.0, 40.0],
                }
            )
            feat.to_parquet(os.path.join(out, f"{safe}.parquet"), index=False)

            result = promote_feature_column_to_label_run(
                tmp, name, "future_ltp_5m", strategy="fixed_horizon"
            )
            self.assertTrue(result["ok"])
            run_id = result["run_id"]
            rows = list_label_runs(tmp, dataset_id=name)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0].run_id, run_id)

            merged, info = join_feature_frame_with_label_run(
                feat.drop(columns=["future_ltp_5m"]), tmp, run_id
            )
            self.assertEqual(info["join_keys"], ["master_row_id"])
            self.assertEqual(len(merged), 4)
            self.assertIn("future_ltp_5m", merged.columns)
            self.assertEqual(float(merged["future_ltp_5m"].iloc[0]), 10.0)

            # Feature parquet unchanged (still has original columns only — we didn't rewrite it).
            feat2 = pd.read_parquet(os.path.join(out, f"{safe}.parquet"))
            self.assertIn("feat_a", feat2.columns)

    def test_triple_barrier_label_run_join_master_row_id(self) -> None:
        """Synthetic sample-grid paths → TB Label Run → join on master_row_id."""
        with tempfile.TemporaryDirectory() as tmp:
            from chain_replay_ml.dataset_builder.writer import datasets_dir, _safe_filename

            out = datasets_dir(tmp)
            os.makedirs(out, exist_ok=True)
            name = "tb_feat"
            safe = _safe_filename(name)
            # Dense path for one token: entry at t=1000, then marks hit +4% TP.
            entry_ts = 1_000_000.0
            rows = []
            for i in range(8):
                ts = entry_ts + float(i) * 10.0
                # After entry, climb to TP then stay.
                ltp = 100.0 + (0.0 if i == 0 else (5.0 if i >= 2 else 1.0))
                rows.append(
                    {
                        "master_row_id": f"m{i}",
                        "trading_day": "2026-07-01",
                        "timestamp": ts,
                        "token": "T1",
                        "ltp": ltp,
                        "feat_a": float(i),
                    }
                )
            feat = pd.DataFrame(rows)
            pq_path = os.path.join(out, f"{safe}.parquet")
            feat.to_parquet(pq_path, index=False)

            result = create_triple_barrier_label_run(
                tmp,
                name,
                barrier_type="percentage",
                holding_seconds=300,
                tp_value=4.0,
                sl_value=2.0,
                truncate_at_close=False,
            )
            self.assertTrue(result["ok"])
            run_id = result["run_id"]
            meta = result["meta"]
            self.assertEqual(meta["strategy"], "triple_barrier")
            self.assertEqual(meta["primary_target"], "label_id")
            self.assertEqual(meta["join_keys"], ["master_row_id"])
            self.assertEqual(meta["dataset_id"], name)

            labels = pd.read_parquet(result["parquet_path"])
            self.assertIn("master_row_id", labels.columns)
            self.assertIn("label_id", labels.columns)
            self.assertNotIn("feat_a", labels.columns)
            # First row should be able to hit TP on forward path.
            first = labels[labels["master_row_id"] == "m0"].iloc[0]
            self.assertTrue(bool(first["is_valid"]))
            self.assertEqual(str(first["label_name"]), "TP")

            merged, info = join_feature_frame_with_label_run(
                feat.drop(columns=["ltp"]), tmp, run_id
            )
            self.assertEqual(info["join_keys"], ["master_row_id"])
            # Last sample has empty forward path → dropped by join (drop_invalid=True).
            self.assertEqual(int(info["dropped_invalid_labels"]), 1)
            self.assertEqual(len(merged), len(feat) - 1)
            self.assertIn("label_id", merged.columns)
            self.assertIn("feat_a", merged.columns)

            # Feature parquet untouched.
            feat2 = pd.read_parquet(pq_path)
            self.assertListEqual(list(feat2.columns), list(feat.columns))

    def test_refuse_feature_smuggle(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            bad = pd.DataFrame(
                {
                    "master_row_id": ["a"],
                    "label_id": [0],
                    "feat_xyz": [1.0],
                }
            )
            with self.assertRaises(ValueError):
                write_label_run(
                    tmp,
                    frame=bad,
                    strategy="triple_barrier",
                    strategy_version="1.0",
                    engine_version="6.1",
                    dataset_id="ds",
                    dataset_hash=None,
                    parameters={},
                    primary_target="label_id",
                    join_keys=["master_row_id"],
                )

    def test_config_label_run_id(self) -> None:
        cfg = normalize_training_config(
            {
                "dataset": "ds",
                "target": "label_id",
                "features": ["a"],
                "label_run_id": "tb_run_1",
                "prediction_type": "binary",
            }
        )
        self.assertEqual(cfg.label_run_id, "tb_run_1")
        self.assertEqual(cfg.to_dict().get("label_run_id"), "tb_run_1")

    def test_validate_nan_target_uses_label_run(self) -> None:
        """No NaN target must read Label Run parquet, not feature columns."""
        with tempfile.TemporaryDirectory() as tmp:
            from chain_replay_ml.dataset_builder.writer import datasets_dir, _safe_filename
            from chain_replay_ml.training.config_validator import validate_training_config

            out = datasets_dir(tmp)
            os.makedirs(out, exist_ok=True)
            name = "feat_nan"
            safe = _safe_filename(name)
            feat = pd.DataFrame(
                {
                    "master_row_id": ["a", "b"],
                    "trading_day": ["2026-07-01", "2026-07-01"],
                    "timestamp": [1.0, 2.0],
                    "token": ["T1", "T1"],
                    "feat_a": [1.0, 2.0],
                    "future_ltp_5m": [10.0, 20.0],
                }
            )
            feat.to_parquet(os.path.join(out, f"{safe}.parquet"), index=False)
            # Minimal meta so dataset "exists" for validation.
            import json

            with open(os.path.join(out, f"{safe}_meta.json"), "w", encoding="utf-8") as fh:
                json.dump(
                    {
                        "dataset": name,
                        "row_count": 2,
                        "build_profile": "experiment",
                        "columns": list(feat.columns),
                    },
                    fh,
                )

            result = promote_feature_column_to_label_run(
                tmp, name, "future_ltp_5m", strategy="fixed_horizon"
            )
            self.assertTrue(result["ok"])
            run_id = result["run_id"]

            res = validate_training_config(
                tmp,
                {
                    "dataset": name,
                    "target": "future_ltp_5m",
                    "features": ["feat_a"],
                    "prediction_type": "regression",
                    "label_strategy": "fixed_horizon",
                    "label_run_id": run_id,
                    "algorithm": "xgboost",
                },
            )
            by_label = {c["label"]: c for c in res["checks"]}
            self.assertTrue(by_label["Target exists"]["passed"])
            self.assertTrue(by_label["Label Run"]["passed"])
            self.assertTrue(
                by_label["No NaN target"]["passed"],
                by_label["No NaN target"].get("detail"),
            )
            self.assertIn("Label Run", str(by_label["No NaN target"].get("detail") or ""))


class OlePanelModeStrategySyncTests(unittest.TestCase):
    """Sanity: mode radio forces strategy var (no Tk root required for helper)."""

    def test_sync_strategy_to_mode_helper(self) -> None:
        # Lightweight stand-in mirroring panel logic.
        class _Stub:
            def __init__(self) -> None:
                self._mode = "promote"
                self._strategy = "fixed_horizon"

            def _sync(self) -> None:
                if self._mode == "triple_barrier":
                    self._strategy = "triple_barrier"
                else:
                    self._strategy = "fixed_horizon"

        s = _Stub()
        s._mode = "triple_barrier"
        s._strategy = "fixed_horizon"  # stale prefs
        s._sync()
        self.assertEqual(s._strategy, "triple_barrier")
        s._mode = "promote"
        s._sync()
        self.assertEqual(s._strategy, "fixed_horizon")


if __name__ == "__main__":
    unittest.main()

"""Training load opts — direct Arrow bridge + skip mergesort when ordered."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from chain_replay_ml.frame_backend import BRIDGE_ARROW_PANDAS, arrow_table_to_pandas
from chain_replay_ml.training.dataset_loader import (
    _metadata_claims_sorted,
    _stabilize_row_order,
    _train_frame_bridge_via_polars,
)


class TrainLoadOptTests(unittest.TestCase):
    def test_direct_arrow_bridge_default(self) -> None:
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("ARUNEO_TRAIN_FRAME_BRIDGE", None)
            self.assertFalse(_train_frame_bridge_via_polars())
        with patch.dict(os.environ, {"ARUNEO_TRAIN_FRAME_BRIDGE": "polars"}):
            self.assertTrue(_train_frame_bridge_via_polars())

    def test_arrow_to_pandas_direct(self) -> None:
        import pyarrow as pa

        table = pa.table({"a": [1, 2, 3], "b": [0.1, 0.2, 0.3]})
        df, bridge = arrow_table_to_pandas(table, via_polars=False)
        self.assertEqual(bridge, BRIDGE_ARROW_PANDAS)
        self.assertEqual(list(df["a"]), [1, 2, 3])

    def test_skip_sort_when_metadata_sorted(self) -> None:
        df = pd.DataFrame(
            {
                "trading_day": ["2026-07-01", "2026-07-01", "2026-07-02"],
                "timestamp": [1.0, 2.0, 3.0],
                "token": ["A", "B", "C"],
                "x": [1, 2, 3],
            }
        )
        out, info = _stabilize_row_order(df, metadata={"is_sorted": True})
        self.assertFalse(info["python_sort"])
        self.assertEqual(info["skip_reason"], "metadata_is_sorted")
        self.assertEqual(list(out["x"]), [1, 2, 3])

    def test_skip_sort_when_already_ordered(self) -> None:
        df = pd.DataFrame(
            {
                "trading_day": ["2026-07-02", "2026-07-01"],
                "timestamp": [2.0, 1.0],
                "token": ["B", "A"],
            }
        )
        out, info = _stabilize_row_order(df, already_ordered=True)
        self.assertFalse(info["python_sort"])
        self.assertEqual(info["skip_reason"], "already_ordered")
        # Must not re-sort when Engine already ORDER BY'd.
        self.assertEqual(list(out["trading_day"]), ["2026-07-02", "2026-07-01"])

    def test_sort_when_unsorted_no_metadata(self) -> None:
        df = pd.DataFrame(
            {
                "trading_day": ["2026-07-02", "2026-07-01"],
                "timestamp": [2.0, 1.0],
                "token": ["B", "A"],
            }
        )
        out, info = _stabilize_row_order(df)
        self.assertTrue(info["python_sort"])
        self.assertEqual(list(out["trading_day"]), ["2026-07-01", "2026-07-02"])

    def test_metadata_claims_sorted_row_order(self) -> None:
        self.assertTrue(
            _metadata_claims_sorted(
                {"row_order": ["trading_day", "timestamp", "token"]}
            )
        )
        self.assertFalse(_metadata_claims_sorted({"row_count": 10}))

    def test_holdout_passthrough_skips_disk(self) -> None:
        from chain_replay_ml.feature_importance_studio.compute import _load_holdout_xy

        X = pd.DataFrame({"feat_a": [1.0, 2.0, 3.0, 4.0], "feat_b": [0.1, 0.2, 0.3, 0.4]})
        y = pd.Series([10.0, 20.0, 30.0, 40.0])
        with tempfile.TemporaryDirectory() as tmp:
            pkg = os.path.join(tmp, "model")
            os.makedirs(pkg, exist_ok=True)
            with open(os.path.join(pkg, "config.json"), "w", encoding="utf-8") as fh:
                json.dump(
                    {
                        "dataset": "unused",
                        "target": "future_ltp_60",
                        "features": ["feat_a"],
                        "prediction_type": "regression",
                        "split": {
                            "train": 50,
                            "validation": 25,
                            "test": 25,
                            "strategy": "time_series",
                        },
                    },
                    fh,
                )
            with patch(
                "chain_replay_ml.feature_importance_studio.compute.load_training_xy"
            ) as mock_load:
                with patch(
                    "chain_replay_ml.feature_importance_studio.compute._selected_feature_names",
                    return_value=["feat_a"],
                ):
                    X_ho, y_ho, feats, meta = _load_holdout_xy(
                        data_dir=tmp,
                        package_dir=pkg,
                        model_name="model",
                        doc={},
                        holdout_max_rows=None,
                        X=X,
                        y=y,
                    )
                mock_load.assert_not_called()
            self.assertEqual(meta.get("matrix_source"), "in_memory")
            self.assertEqual(feats, ["feat_a"])
            self.assertGreater(len(X_ho), 0)


if __name__ == "__main__":
    unittest.main()

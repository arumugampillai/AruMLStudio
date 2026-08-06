"""Tests for Interaction transform ops and Interaction Builder helpers."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from chain_replay_ml.dataset_builder.transformations import run_transformation_pipeline
from chain_replay_ml.dataset_builder.transformations.base import TransformContext
from chain_replay_ml.dataset_builder.transformations.interaction import (
    interaction_column_name,
    normalize_interaction_op,
    validate_interaction_pairs,
)
from chain_replay_ml.dataset_builder.transformations.interaction_ui import (
    available_interaction_features,
    bulk_interaction_pairs,
    format_lineage_tree,
    format_pipeline_ledger_text,
    group_features_by_source,
    merge_interaction_into_config,
    pipeline_feature_ledger,
)
from chain_replay_ml.dataset_builder.transformations.time_shift import LagConfigError


def _run_ix(df: pd.DataFrame, pairs: list[dict], **params):
    cfg = {
        "transformation_pipeline_version": 1,
        "transformations": [{
            "id": "interaction",
            "enabled": True,
            "params": {"pairs": pairs, "sample_interval_sec": 3, **params},
        }],
    }
    ctx = TransformContext(config=cfg, sample_interval_sec=3.0)
    return run_transformation_pipeline(df, cfg, context=ctx)


class InteractionOpsTests(unittest.TestCase):
    def test_op_aliases_and_naming(self) -> None:
        self.assertEqual(normalize_interaction_op("mul"), "multiply")
        self.assertEqual(normalize_interaction_op("x"), "multiply")
        self.assertEqual(
            interaction_column_name("a", "b", "multiply"),
            "a_x_b",
        )
        self.assertEqual(
            interaction_column_name("a", "b", "absolute_difference"),
            "a_absdiff_b",
        )

    def test_min_max_absdiff(self) -> None:
        df = pd.DataFrame({"a": [1.0, 5.0], "b": [3.0, 2.0]})
        result = _run_ix(df, [
            {"left": "a", "right": "b", "op": "min"},
            {"left": "a", "right": "b", "op": "max"},
            {"left": "a", "right": "b", "op": "absolute_difference"},
        ])
        self.assertEqual(result.frame.loc[0, "a_min_b"], 1.0)
        self.assertEqual(result.frame.loc[0, "a_max_b"], 3.0)
        self.assertEqual(result.frame.loc[0, "a_absdiff_b"], 2.0)
        self.assertEqual(result.frame.loc[1, "a_absdiff_b"], 3.0)

    def test_chaining_within_step(self) -> None:
        df = pd.DataFrame({"a": [2.0], "b": [3.0], "c": [4.0]})
        result = _run_ix(df, [
            {"left": "a", "right": "b", "op": "multiply"},
            {"left": "a_x_b", "right": "c", "op": "multiply"},
        ])
        self.assertEqual(result.frame.loc[0, "a_x_b"], 6.0)
        self.assertEqual(result.frame.loc[0, "a_x_b_x_c"], 24.0)

    def test_duplicate_output_fails(self) -> None:
        with self.assertRaises(LagConfigError):
            validate_interaction_pairs(
                [
                    {"left": "a", "right": "b", "op": "mul", "output": "x"},
                    {"left": "a", "right": "b", "op": "add", "output": "x"},
                ],
                existing_columns={"a", "b"},
            )

    def test_div_zero_fail(self) -> None:
        df = pd.DataFrame({"a": [1.0], "b": [0.0]})
        with self.assertRaises(LagConfigError):
            _run_ix(
                df,
                [{"left": "a", "right": "b", "op": "divide"}],
                div_zero="fail",
            )

    def test_return_then_interaction(self) -> None:
        df = pd.DataFrame({
            "token": ["A", "A", "A"],
            "ltp": [100.0, 110.0, 121.0],
            "moneyness": [1.0, 1.1, 1.2],
        })
        cfg = {
            "transformation_pipeline_version": 1,
            "transformations": [
                {
                    "id": "return",
                    "enabled": True,
                    "params": {
                        "features": ["ltp"],
                        "lag_seconds": [3],
                        "partition_by": ["token"],
                        "sample_interval_sec": 3,
                    },
                },
                {
                    "id": "interaction",
                    "enabled": True,
                    "params": {
                        "pairs": [{
                            "left": "ltp_return_3s",
                            "right": "moneyness",
                            "op": "multiply",
                        }],
                        "sample_interval_sec": 3,
                    },
                },
            ],
        }
        ctx = TransformContext(config=cfg, sample_interval_sec=3.0)
        result = run_transformation_pipeline(df, cfg, context=ctx)
        self.assertIn("ltp_return_3s_x_moneyness", result.frame.columns)


class InteractionUiHelperTests(unittest.TestCase):
    def test_bulk_pairs(self) -> None:
        pairs = bulk_interaction_pairs(
            ["iv", "spot"],
            ["moneyness", "delta"],
            op="multiply",
        )
        outs = {p["output"] for p in pairs}
        self.assertEqual(len(pairs), 4)
        self.assertIn("iv_x_moneyness", outs)
        self.assertIn("spot_x_delta", outs)

    def test_group_by_source(self) -> None:
        grouped = group_features_by_source(
            ["ltp", "ltp_lag_30s", "ltp_return_30s", "a_x_b"],
            master_features={"ltp"},
            interaction_outputs={"a_x_b"},
        )
        self.assertIn("ltp", grouped.get("Base", []))
        self.assertIn("ltp_lag_30s", grouped.get("Lag", []))
        self.assertIn("ltp_return_30s", grouped.get("Return", []))
        self.assertIn("a_x_b", grouped.get("Interaction", []))

    def test_available_includes_planned_return(self) -> None:
        avail = available_interaction_features(
            master_features=["ltp", "moneyness"],
            lag_features=["ltp"],
            lag_seconds=[30],
            return_enabled=True,
            interaction_pairs=[],
        )
        self.assertIn("ltp_return_30s", avail)
        self.assertIn("moneyness", avail)

    def test_lineage_and_ledger(self) -> None:
        parent_map = {
            "a_x_b": {"left": "a", "right": "b", "op": "multiply", "output": "a_x_b"},
        }
        tree = format_lineage_tree("a_x_b", parent_map=parent_map)
        self.assertIn("Interaction (multiply)", tree)
        self.assertIn("a", tree)
        ledger = pipeline_feature_ledger(
            master_count=133,
            lag_enabled=True,
            difference_enabled=True,
            return_enabled=False,
            interaction_count=8,
            selected_features=["ltp", "spot"],
            lag_seconds=[30, 60],
        )
        self.assertEqual(ledger["lag"], 4)
        self.assertEqual(ledger["difference"], 4)
        self.assertEqual(ledger["interaction"], 8)
        self.assertEqual(ledger["final"], 133 + 4 + 4 + 8)
        text = format_pipeline_ledger_text(ledger)
        self.assertIn("Final Dataset", text)

    def test_merge_into_config(self) -> None:
        base = {
            "transformation_pipeline_version": 1,
            "transformations": [
                {"id": "lag", "enabled": True, "params": {"features": ["ltp"]}},
            ],
        }
        cfg = merge_interaction_into_config(
            base,
            enabled=True,
            pairs=[{"left": "ltp", "right": "spot", "op": "mul"}],
        )
        ids = [t["id"] for t in cfg["transformations"]]
        self.assertIn("lag", ids)
        self.assertIn("interaction", ids)
        ix = next(t for t in cfg["transformations"] if t["id"] == "interaction")
        self.assertEqual(ix["params"]["pairs"][0]["op"], "multiply")


if __name__ == "__main__":
    unittest.main()

"""Phase 4: Model Builder ↔ OLE bridge — strategy-agnostic UI contract."""

from __future__ import annotations

import unittest
from dataclasses import dataclass
from typing import Any

from chain_replay_ml.outcome_label_engine import (
    StrategyCapabilities,
    StrategyMetadata,
    TargetDefinitions,
    clear_registry,
    ensure_builtin_strategies,
    register_strategy,
    unregister_strategy,
)
from chain_replay_ml.outcome_label_engine.model_builder_bridge import (
    config_schema_fields,
    default_strategy_id_for_prediction_type,
    preferred_target_column,
    resolve_training_target,
    strategy_selector_rows,
    strategies_for_prediction_type,
)
from chain_replay_ml.training.target_kinds import (
    is_classification_target,
    is_ole_class_target,
    prediction_type_for_target,
)
from master_dataset_tk.model_builder.state import ModelBuilderState


@dataclass
class _MockAdaptiveStrategy:
    """Future strategy — registering it must require zero Model Builder UI edits."""

    metadata: StrategyMetadata
    capabilities: StrategyCapabilities
    _schema: dict[str, Any]
    _targets: TargetDefinitions

    def get_config_schema(self) -> dict[str, Any]:
        return dict(self._schema)

    def get_target_definitions(self) -> TargetDefinitions:
        return self._targets

    def build_labels(self, source, samples, config):
        raise NotImplementedError("Phase 4 bridge tests do not compute labels")


class ModelBuilderOleBridgeTests(unittest.TestCase):
    def setUp(self) -> None:
        clear_registry()
        ensure_builtin_strategies()

    def tearDown(self) -> None:
        clear_registry()
        ensure_builtin_strategies()

    def test_selector_rows_use_registry_metadata(self) -> None:
        rows = strategy_selector_rows(None)
        by_id = {r["strategy_id"]: r for r in rows}
        self.assertIn("fixed_horizon", by_id)
        self.assertIn("triple_barrier", by_id)
        self.assertEqual(by_id["fixed_horizon"]["display_name"], "Fixed Horizon")
        self.assertEqual(by_id["triple_barrier"]["description"], "First TP / SL / Timeout")
        # No hard-coded Triple Barrier knowledge beyond registry.
        self.assertEqual(by_id["triple_barrier"]["category"], "Classification")

    def test_filter_by_prediction_type(self) -> None:
        reg = [s.metadata.strategy_id for s in strategies_for_prediction_type("regression")]
        self.assertIn("fixed_horizon", reg)
        clf = [s.metadata.strategy_id for s in strategies_for_prediction_type("classification")]
        self.assertIn("triple_barrier", clf)
        self.assertEqual(default_strategy_id_for_prediction_type("regression"), "fixed_horizon")

    def test_schema_fields_from_strategy_not_ui_branches(self) -> None:
        fh_fields = dict(config_schema_fields("fixed_horizon"))
        self.assertIn("horizons_sec", fh_fields)
        self.assertIn("max_stale_sec", fh_fields)
        tb_fields = dict(config_schema_fields("triple_barrier"))
        self.assertIn("holding_seconds", tb_fields)
        self.assertIn("barrier_type", tb_fields)
        self.assertEqual(tb_fields["barrier_type"]["type"], "enum")
        self.assertIn("tp_value", tb_fields)
        self.assertIn("sl_value", tb_fields)
        self.assertIn("truncate_at_close", tb_fields)
        self.assertTrue(tb_fields["truncate_at_close"]["default"])
        self.assertEqual(tb_fields["barrier_type"]["default"], "percentage")
        choices = {c["value"]: c for c in tb_fields["barrier_type"]["choices"]}
        self.assertFalse(choices["atr"].get("enabled", True))
        self.assertTrue(choices["percentage"].get("enabled", True))

    def test_preferred_target_primary(self) -> None:
        self.assertEqual(
            preferred_target_column("fixed_horizon", ["future_ltp_1m", "future_ltp_5m"]),
            "future_ltp_5m",
        )
        self.assertEqual(
            preferred_target_column("triple_barrier", ["label_id", "label_name", "future_ltp_5m"]),
            "label_id",
        )
        self.assertEqual(
            resolve_training_target(
                strategy_id="triple_barrier",
                available_columns=["label_id", "label_name"],
                current_target="future_ltp_5m",
            ),
            "label_id",
        )

    def test_switch_strategies_updates_training_target(self) -> None:
        cols = ["future_ltp_5m", "label_id", "label_name"]
        fh_target = resolve_training_target(
            strategy_id="fixed_horizon",
            available_columns=cols,
        )
        tb_target = resolve_training_target(
            strategy_id="triple_barrier",
            available_columns=cols,
        )
        self.assertEqual(fh_target, "future_ltp_5m")
        self.assertEqual(tb_target, "label_id")
        self.assertNotEqual(fh_target, tb_target)

    def test_mock_strategy_appears_without_ui_code_changes(self) -> None:
        mock = _MockAdaptiveStrategy(
            metadata=StrategyMetadata(
                strategy_id="adaptive_barrier",
                version="1.0",
                display_name="Adaptive Barrier",
                description="ATR-adjusted exits",
                category="Classification",
            ),
            capabilities=StrategyCapabilities(
                strategy_id="adaptive_barrier",
                supported_sources=frozenset({"prediction"}),
                supported_problem_types=frozenset({"multiclass"}),
            ),
            _schema={
                "atr_mult": {"type": "float", "default": 1.5},
                "holding_seconds": {"type": "int", "default": 240},
            },
            _targets=TargetDefinitions(
                primary_target="label_id",
                display_target="label_name",
                label_encoding={"TP": 0, "SL": 1, "TIME": 2},
            ),
        )
        register_strategy(mock, replace=True)
        try:
            rows = strategy_selector_rows("classification")
            ids = [r["strategy_id"] for r in rows]
            self.assertIn("adaptive_barrier", ids)
            adaptive = next(r for r in rows if r["strategy_id"] == "adaptive_barrier")
            self.assertEqual(adaptive["display_name"], "Adaptive Barrier")
            fields = dict(config_schema_fields("adaptive_barrier"))
            self.assertIn("atr_mult", fields)
            self.assertEqual(fields["atr_mult"]["default"], 1.5)
            self.assertEqual(
                preferred_target_column("adaptive_barrier", ["label_id"]),
                "label_id",
            )
        finally:
            unregister_strategy("adaptive_barrier")

    def test_training_config_carries_label_strategy(self) -> None:
        state = ModelBuilderState(
            dataset="ds1",
            target="label_id",
            prediction_type="classification",
            label_strategy_id="triple_barrier",
            label_strategy_params={
                "barrier_type": "points",
                "holding_seconds": 240,
                "tp_value": 10.0,
                "sl_value": 5.0,
            },
            features={"f1"},
        )
        cfg = state.build_training_config()
        self.assertEqual(cfg["target"], "label_id")
        self.assertEqual(cfg["label_strategy"], "triple_barrier")
        self.assertEqual(cfg["label_strategy_params"]["holding_seconds"], 240)
        self.assertEqual(cfg["label_strategy_params"]["barrier_type"], "points")
        saved = state.to_saved_dict()
        restored = ModelBuilderState()
        restored.apply_saved_dict(saved)
        self.assertEqual(restored.label_strategy_id, "triple_barrier")
        self.assertEqual(restored.label_strategy_params["tp_value"], 10.0)

    def test_target_kinds_recognize_ole_primary(self) -> None:
        self.assertTrue(is_ole_class_target("label_id"))
        self.assertTrue(is_classification_target("label_id"))
        self.assertEqual(prediction_type_for_target("label_id"), "classification")


if __name__ == "__main__":
    unittest.main()

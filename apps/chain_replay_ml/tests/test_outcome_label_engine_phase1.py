"""Phase 1 Outcome Label Engine foundation tests (interfaces only; no FH/TB math)."""

from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from chain_replay_ml.outcome_label_engine import (
    ENGINE_VERSION,
    DayChunkRunner,
    ImmutableArtifactWriter,
    LabelBatchResult,
    LabelRunMeta,
    LabelSourceContext,
    LabelStrategyConfig,
    StrategyAlreadyRegisteredError,
    StrategyCapabilities,
    StrategyMetadata,
    StrategyNotFoundError,
    TargetDefinitions,
    clear_registry,
    defaults_from_config_schema,
    discover_for_ui,
    filter_by_problem_type,
    filter_by_source,
    get_strategy,
    list_metadata,
    list_strategy_ids,
    mint_artifact_id,
    register_strategy,
    validate_config_against_schema,
)


@dataclass
class _MockStrategy:
    """Minimal strategy for registry / schema / runner tests — no real labeling."""

    metadata: StrategyMetadata
    capabilities: StrategyCapabilities
    _schema: dict[str, Any]
    _targets: TargetDefinitions

    def get_config_schema(self) -> dict[str, Any]:
        return dict(self._schema)

    def get_target_definitions(self) -> TargetDefinitions:
        return self._targets

    def build_labels(
        self,
        source: LabelSourceContext,
        samples: Any,
        config: LabelStrategyConfig,
    ) -> LabelBatchResult:
        rows = []
        for i, sample in enumerate(samples or []):
            rows.append(
                {
                    "sample_id": sample.get("id", i),
                    "is_valid": True,
                    "invalid_reason": None,
                    self._targets.primary_target: 0,
                }
            )
        return LabelBatchResult(
            rows=rows,
            target_columns=[self._targets.primary_target],
            target_definitions=self._targets,
            metadata={"day": source.day},
        )


@dataclass
class _MockDaySource:
    source_kind: str
    days: dict[str, list[dict[str, Any]]]

    def iter_days(self):
        return sorted(self.days.keys())

    def load_day(self, day: str):
        return list(self.days[day])


def _mock_regression() -> _MockStrategy:
    return _MockStrategy(
        metadata=StrategyMetadata(
            strategy_id="mock_fixed_horizon",
            version="1.0",
            display_name="Mock Fixed Horizon",
            description="Future premium at fixed horizon (mock)",
            category="Regression",
        ),
        capabilities=StrategyCapabilities(
            strategy_id="mock_fixed_horizon",
            supported_sources=frozenset({"master"}),
            supported_problem_types=frozenset({"regression", "binary_classification"}),
        ),
        _schema={
            "horizon_sec": {"type": "int", "default": 300},
        },
        _targets=TargetDefinitions(primary_target="future_ltp_5m"),
    )


def _mock_classification() -> _MockStrategy:
    return _MockStrategy(
        metadata=StrategyMetadata(
            strategy_id="mock_triple_barrier",
            version="1.0",
            display_name="Mock Triple Barrier",
            description="First TP / SL / Timeout (mock)",
            category="Classification",
        ),
        capabilities=StrategyCapabilities(
            strategy_id="mock_triple_barrier",
            supported_sources=frozenset({"prediction"}),
            supported_problem_types=frozenset(
                {"binary_classification", "multiclass"}
            ),
        ),
        _schema={
            "holding_seconds": {"type": "int", "default": 300},
            "tp_points": {"type": "float", "default": 10.0},
            "sl_points": {"type": "float", "default": 5.0},
            "truncate_at_close": {"type": "bool", "default": True},
        },
        _targets=TargetDefinitions(
            primary_target="label_id",
            display_target="label_name",
            label_encoding={"TP": 0, "SL": 1, "TIME": 2},
        ),
    )


class OutcomeLabelEnginePhase1Tests(unittest.TestCase):
    def setUp(self) -> None:
        clear_registry()

    def tearDown(self) -> None:
        clear_registry()

    def test_registry_loads_and_lists(self) -> None:
        register_strategy(_mock_regression())
        register_strategy(_mock_classification())
        self.assertEqual(
            list_strategy_ids(),
            ["mock_fixed_horizon", "mock_triple_barrier"],
        )
        self.assertEqual(get_strategy("mock_fixed_horizon").metadata.display_name, "Mock Fixed Horizon")

    def test_registry_rejects_duplicate(self) -> None:
        register_strategy(_mock_regression())
        with self.assertRaises(StrategyAlreadyRegisteredError):
            register_strategy(_mock_regression())

    def test_registry_missing_raises(self) -> None:
        with self.assertRaises(StrategyNotFoundError):
            get_strategy("does_not_exist")

    def test_metadata_discovery(self) -> None:
        register_strategy(_mock_regression())
        register_strategy(_mock_classification())
        metas = list_metadata()
        self.assertEqual(len(metas), 2)
        by_id = {m.strategy_id: m for m in metas}
        self.assertEqual(by_id["mock_triple_barrier"].description, "First TP / SL / Timeout (mock)")
        self.assertEqual(by_id["mock_triple_barrier"].category, "Classification")
        self.assertEqual(by_id["mock_fixed_horizon"].category, "Regression")

    def test_capability_filtering(self) -> None:
        register_strategy(_mock_regression())
        register_strategy(_mock_classification())
        reg = filter_by_problem_type("regression")
        self.assertEqual([s.metadata.strategy_id for s in reg], ["mock_fixed_horizon"])
        multi = filter_by_problem_type("multiclass")
        self.assertEqual([s.metadata.strategy_id for s in multi], ["mock_triple_barrier"])
        pred = filter_by_source("prediction")
        self.assertEqual([s.metadata.strategy_id for s in pred], ["mock_triple_barrier"])
        ui = discover_for_ui(problem_type="binary_classification")
        self.assertEqual(
            {m.strategy_id for m in ui},
            {"mock_fixed_horizon", "mock_triple_barrier"},
        )

    def test_schema_generation(self) -> None:
        s = _mock_classification()
        register_strategy(s)
        schema = get_strategy("mock_triple_barrier").get_config_schema()
        self.assertEqual(schema["holding_seconds"]["type"], "int")
        self.assertEqual(schema["holding_seconds"]["default"], 300)
        self.assertTrue(schema["truncate_at_close"]["default"])
        defaults = defaults_from_config_schema(schema)
        self.assertEqual(defaults["tp_points"], 10.0)
        merged = validate_config_against_schema({"tp_points": 12.5}, schema)
        self.assertEqual(merged["tp_points"], 12.5)
        self.assertEqual(merged["holding_seconds"], 300)
        with self.assertRaises(ValueError):
            validate_config_against_schema({"unknown_param": 1}, schema)

    def test_target_definition_generation(self) -> None:
        register_strategy(_mock_classification())
        defs = get_strategy("mock_triple_barrier").get_target_definitions()
        self.assertEqual(defs.primary_target, "label_id")
        self.assertEqual(defs.display_target, "label_name")
        self.assertEqual(defs.label_encoding, {"TP": 0, "SL": 1, "TIME": 2})
        # No sentinel classes in encoding.
        self.assertNotIn(-1, (defs.label_encoding or {}).values())
        roundtrip = TargetDefinitions.from_dict(defs.to_dict())
        self.assertEqual(roundtrip, defs)

    def test_day_chunk_runner_and_immutable_writer(self) -> None:
        strategy = _mock_classification()
        register_strategy(strategy)
        source = _MockDaySource(
            source_kind="prediction",
            days={
                "2024-01-02": [{"id": 1}, {"id": 2}],
                "2024-01-03": [{"id": 3}],
            },
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifact_id = mint_artifact_id(strategy.metadata.strategy_id, suffix="test")
            writer = ImmutableArtifactWriter(root, artifact_id)
            result = DayChunkRunner().run(
                strategy,
                source,
                LabelStrategyConfig(
                    strategy_id=strategy.metadata.strategy_id,
                    version=strategy.metadata.version,
                    params={"holding_seconds": 240},
                ),
                writer,
            )
            self.assertEqual(result.days_processed, ["2024-01-02", "2024-01-03"])
            self.assertEqual(result.run_meta.rows, 3)
            self.assertEqual(result.run_meta.engine_version, ENGINE_VERSION)
            self.assertEqual(result.run_meta.params["holding_seconds"], 240)
            self.assertTrue(result.run_meta.params["truncate_at_close"])
            meta_path = result.artifact_dir / "run_meta.json"
            self.assertTrue(meta_path.is_file())
            loaded = LabelRunMeta.from_dict(json.loads(meta_path.read_text(encoding="utf-8")))
            self.assertEqual(loaded.strategy, "mock_triple_barrier")
            self.assertEqual(loaded.target_definitions["primary_target"], "label_id")

            # Immutability: second open of same artifact id must fail.
            with self.assertRaises(FileExistsError):
                ImmutableArtifactWriter(root, artifact_id).open()

    def test_builtin_strategies_registered(self) -> None:
        """Phase 3: Fixed Horizon and Triple Barrier are both registered."""
        from chain_replay_ml.outcome_label_engine import ensure_builtin_strategies

        ensure_builtin_strategies()
        ids = list_strategy_ids()
        self.assertIn("fixed_horizon", ids)
        self.assertIn("triple_barrier", ids)
        meta = get_strategy("triple_barrier").metadata
        self.assertEqual(meta.display_name, "Triple Barrier")
        self.assertEqual(meta.category, "Classification")
if __name__ == "__main__":
    unittest.main()

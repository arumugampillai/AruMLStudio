"""Tests for Phase 2 strategy registry."""

from __future__ import annotations

import os
import tempfile
import unittest

from chain_replay_ml.strategy_registry import (
    clone_strategy_version,
    compare_strategy_versions,
    create_strategy,
    create_strategy_version,
    get_default_template,
    get_strategy_detail,
    list_strategies,
    set_champion_version,
)
from chain_replay_ml.strategy_registry.schema import validate_strategy_config


class StrategyRegistryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp()

    def test_create_strategy_and_champion_version(self) -> None:
        detail = create_strategy(
            self.tmp,
            display_name="OTM Premium Buyer",
            description="Test strategy",
            config=get_default_template(),
        )
        profile = detail["profile"]
        self.assertEqual(profile["display_name"], "OTM Premium Buyer")
        self.assertEqual(profile["current_version_label"], "v1")
        self.assertIsNotNone(profile["champion_config_hash"])
        self.assertEqual(len(detail["versions"]), 1)
        self.assertTrue(detail["versions"][0]["is_champion"])

        strategies = list_strategies(self.tmp)
        self.assertEqual(len(strategies), 1)

    def test_edit_creates_new_immutable_version(self) -> None:
        detail = create_strategy(self.tmp, display_name="Family A", config=get_default_template())
        sid = detail["profile"]["strategy_id"]
        v1_id = detail["champion_version"]["version_id"]

        cfg = dict(detail["champion_version"]["config"])
        cfg["target"]["target_profit_pct"] = 10.0
        v2 = create_strategy_version(self.tmp, strategy_id=sid, config=cfg, lifecycle="edit")
        self.assertEqual(v2["version_label"], "v2")
        self.assertNotEqual(v2["version_id"], v1_id)

        updated = get_strategy_detail(self.tmp, sid)
        assert updated is not None
        self.assertEqual(updated["profile"]["current_version_label"], "v2")
        self.assertEqual(len(updated["versions"]), 2)

    def test_same_config_reuses_version_hash(self) -> None:
        detail = create_strategy(self.tmp, display_name="Hash Test", config=get_default_template())
        sid = detail["profile"]["strategy_id"]
        cfg = detail["champion_version"]["config"]
        again = create_strategy_version(self.tmp, strategy_id=sid, config=cfg, lifecycle="edit")
        self.assertEqual(again["version_id"], detail["champion_version"]["version_id"])
        updated = get_strategy_detail(self.tmp, sid)
        assert updated is not None
        self.assertEqual(len(updated["versions"]), 1)

    def test_clone_strategy_version(self) -> None:
        detail = create_strategy(self.tmp, display_name="Source", config=get_default_template())
        v1 = detail["champion_version"]["version_id"]
        cloned = clone_strategy_version(
            self.tmp,
            source_version_id=v1,
            display_name="Cloned Family",
            config_overrides={"target": {"target_profit_pct": 12.0}},
        )
        self.assertNotEqual(cloned["strategy_id"], detail["profile"]["strategy_id"])
        self.assertEqual(cloned["config"]["target"]["target_profit_pct"], 12.0)

    def test_clone_same_family_creates_new_version(self) -> None:
        detail = create_strategy(self.tmp, display_name="Source", config=get_default_template())
        v1 = detail["champion_version"]["version_id"]
        cloned = clone_strategy_version(self.tmp, source_version_id=v1, display_name=None)
        self.assertEqual(cloned["strategy_id"], detail["profile"]["strategy_id"])
        self.assertNotEqual(cloned["version_id"], v1)
        updated = get_strategy_detail(self.tmp, detail["profile"]["strategy_id"])
        assert updated is not None
        self.assertEqual(len(updated["versions"]), 2)
        self.assertEqual(updated["profile"]["current_version_label"], "v2")

    def test_set_champion_version(self) -> None:
        detail = create_strategy(self.tmp, display_name="Champion Test", config=get_default_template())
        sid = detail["profile"]["strategy_id"]
        v1_id = detail["champion_version"]["version_id"]

        cfg = dict(detail["champion_version"]["config"])
        cfg["stop"]["stop_loss_pct"] = 4.0
        v2 = create_strategy_version(self.tmp, strategy_id=sid, config=cfg, lifecycle="calibration")
        set_champion_version(self.tmp, sid, v1_id)

        updated = get_strategy_detail(self.tmp, sid)
        assert updated is not None
        self.assertEqual(updated["profile"]["current_version_id"], v1_id)

    def test_compare_versions(self) -> None:
        detail = create_strategy(self.tmp, display_name="Compare", config=get_default_template())
        sid = detail["profile"]["strategy_id"]
        v1_id = detail["champion_version"]["version_id"]
        cfg = dict(detail["champion_version"]["config"])
        cfg["hold_time"]["max_hold_sec"] = 45
        v2 = create_strategy_version(self.tmp, strategy_id=sid, config=cfg)
        result = compare_strategy_versions(self.tmp, v1_id, v2["version_id"])
        self.assertTrue(result["ok"])
        self.assertFalse(result["same_hash"])
        paths = {c["path"] for c in result["changes"]}
        self.assertIn("hold_time.max_hold_sec", paths)

    def test_validate_strategy_config(self) -> None:
        cfg = get_default_template()
        cfg["entry"]["premium_min"] = 50
        cfg["entry"]["premium_max"] = 10
        errors = validate_strategy_config(cfg)
        self.assertTrue(any("premium_min" in e for e in errors))

    def test_minimum_predicted_move_pct_default_and_validate(self) -> None:
        cfg = get_default_template()
        self.assertEqual(cfg["entry"]["minimum_predicted_move_pct"], 0.0)
        cfg["entry"]["minimum_predicted_move_pct"] = -1
        errors = validate_strategy_config(cfg)
        self.assertTrue(any("minimum_predicted_move_pct" in e for e in errors))

    def test_use_predicted_ltp_default_false(self) -> None:
        cfg = get_default_template()
        self.assertFalse(cfg["target"]["use_predicted_ltp"])

    def test_use_regression_default_true(self) -> None:
        cfg = get_default_template()
        self.assertTrue(cfg["entry"]["use_regression"])
        cfg["entry"]["use_regression"] = False
        from chain_replay_ml.strategy_registry.schema import normalize_strategy_config

        norm = normalize_strategy_config(cfg)
        self.assertFalse(norm["entry"]["use_regression"])

    def test_strategy_package_written(self) -> None:
        detail = create_strategy(self.tmp, display_name="Package", config=get_default_template())
        sid = detail["profile"]["strategy_id"]
        pkg = os.path.join(self.tmp, "strategies", sid, "v1", "strategy.json")
        self.assertTrue(os.path.isfile(pkg))


if __name__ == "__main__":
    unittest.main()

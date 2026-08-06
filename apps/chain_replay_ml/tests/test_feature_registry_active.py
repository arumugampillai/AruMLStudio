"""Tests for registry-level feature active/disabled state."""

from __future__ import annotations

import os
import tempfile
import unittest

from chain_replay_ml.dataset_builder.feature_registry_catalog import build_feature_registry_catalog
from chain_replay_ml.dataset_builder.feature_registry_store import (
    DISABLED_GROUP_ID,
    disabled_registry_feature_names,
    load_store,
    set_feature_registry_active,
)
from chain_replay_ml.dataset_builder.feature_plugins import resolve_implemented_features_for_selection
from chain_replay_ml.dataset_builder.master_defaults import default_master_feature_selection
from chain_replay_ml.dataset_builder.schema_registry import load_feature_registry


class FeatureRegistryActiveTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.data_dir = self._tmp.name
        os.makedirs(self.data_dir, exist_ok=True)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_disable_moves_to_disabled_group_and_excludes_from_build(self) -> None:
        store = load_store(self.data_dir)
        set_feature_registry_active(
            self.data_dir,
            store,
            "spot",
            active=False,
            home_group_id="price",
        )
        catalog = build_feature_registry_catalog(self.data_dir)
        spot = next(f for f in catalog["features"] if f["name"] == "spot")
        self.assertFalse(spot["registry_active"])
        self.assertEqual(spot["group_id"], DISABLED_GROUP_ID)
        self.assertEqual(spot["home_group_id"], "price")

        registry = load_feature_registry()
        selection = default_master_feature_selection(registry)
        _, implemented, _, _ = resolve_implemented_features_for_selection(
            selection,
            registry,
            data_dir=self.data_dir,
        )
        self.assertNotIn("spot", implemented)

    def test_enable_restores_home_group(self) -> None:
        store = load_store(self.data_dir)
        set_feature_registry_active(
            self.data_dir,
            store,
            "spot",
            active=False,
            home_group_id="price",
        )
        store = load_store(self.data_dir)
        set_feature_registry_active(self.data_dir, store, "spot", active=True)
        self.assertEqual(disabled_registry_feature_names(load_store(self.data_dir)), set())

        catalog = build_feature_registry_catalog(self.data_dir)
        spot = next(f for f in catalog["features"] if f["name"] == "spot")
        self.assertTrue(spot["registry_active"])
        self.assertEqual(spot["group_id"], "price")


class FilterActiveRegistryNamesTests(unittest.TestCase):
    def test_filter_drops_disabled(self) -> None:
        import sys
        import tempfile
        from pathlib import Path

        root = Path(__file__).resolve().parents[2]
        chart = root / "angelone" / "chart"
        if str(chart) not in sys.path:
            sys.path.insert(0, str(chart))
        from master_dataset_tk import feature_registry_service as fr_svc
        from chain_replay_ml.dataset_builder.feature_registry_store import (
            load_store,
            set_feature_registry_active,
        )

        with tempfile.TemporaryDirectory() as tmp:
            data_dir = os.path.join(tmp, "data")
            os.makedirs(data_dir, exist_ok=True)
            chart_dir = tmp
            store = load_store(data_dir)
            set_feature_registry_active(
                data_dir, store, "spot", active=False, home_group_id="price",
            )
            out = fr_svc.filter_active_registry_names(
                chart_dir, ["spot", "ltp", "iv_zscore_30m"],
            )
            self.assertEqual(out, ["ltp", "iv_zscore_30m"])


if __name__ == "__main__":
    unittest.main()

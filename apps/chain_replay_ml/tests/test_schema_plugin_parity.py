"""Invariant: ml_schema_registry.json ↔ _REGISTRY_FEATURES are 1:1."""

from __future__ import annotations

import json
import os
import tempfile
import unittest

from chain_replay_ml.dataset_builder.feature_plugins import _REGISTRY_FEATURES
from chain_replay_ml.dataset_builder.feature_registry_catalog import (
    build_feature_registry_catalog,
)
from chain_replay_ml.dataset_builder.schema_registry import (
    SchemaRegistrySyncError,
    canonical_plugin_feature_names,
    load_schema_registry,
    rebuild_schema_registry_from_plugins,
    schema_feature_column_names,
    schema_registry_path,
    validate_schema_plugin_parity,
    _sync_builtin_plugin_features,
)


class SchemaPluginParityTests(unittest.TestCase):
    def test_on_disk_schema_matches_plugins_after_load(self) -> None:
        schema = load_schema_registry(use_cache=False)
        plugin = canonical_plugin_feature_names()
        schema_feats = schema_feature_column_names(schema)
        self.assertEqual(plugin, schema_feats)
        self.assertEqual(len(plugin), sum(len(v) for v in _REGISTRY_FEATURES.values()))
        validate_schema_plugin_parity(schema, raise_on_error=True)

    def test_rebuild_is_full_replace_not_additive(self) -> None:
        stale = {
            "version": 5,
            "columns": {
                "spot": {"name": "spot", "type": "feature", "group": "price"},
                "ghost_historical_feat": {
                    "name": "ghost_historical_feat",
                    "type": "feature",
                    "group": "historical",
                },
                "timestamp": {"name": "timestamp", "type": "metadata"},
            },
            "groups": {
                "price": {"label": "Price", "features": ["spot", "ghost_historical_feat"]},
                "historical": {"label": "Historical", "features": ["ghost_historical_feat"]},
            },
            "groupOrder": ["price", "historical"],
            "dependencies": {},
        }
        synced = _sync_builtin_plugin_features(dict(stale))
        feats = schema_feature_column_names(synced)
        self.assertNotIn("ghost_historical_feat", feats)
        self.assertEqual(feats, canonical_plugin_feature_names())
        validate_schema_plugin_parity(synced, raise_on_error=True)

    def test_no_interaction_features_in_plugins_or_schema(self) -> None:
        from chain_replay_ml.dataset_builder.feature_ownership import is_interaction_feature

        plugin = canonical_plugin_feature_names()
        self.assertFalse(any(is_interaction_feature(n) for n in plugin))
        schema = rebuild_schema_registry_from_plugins()
        feats = schema_feature_column_names(schema)
        self.assertFalse(any(is_interaction_feature(n) for n in feats))
        validate_schema_plugin_parity(schema, raise_on_error=True)

    def test_validate_rejects_interaction_smuggled_into_schema(self) -> None:
        schema = rebuild_schema_registry_from_plugins()
        schema["columns"]["spot_x_delta"] = {
            "name": "spot_x_delta",
            "type": "feature",
            "group": "advanced",
        }
        with self.assertRaises(SchemaRegistrySyncError):
            validate_schema_plugin_parity(schema, raise_on_error=True)

    def test_catalog_feature_count_matches_plugins(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cat = build_feature_registry_catalog(tmp)
            plugin_n = len(canonical_plugin_feature_names())
            # Catalog may add backlog/imported overlays; base schema names must match plugins.
            names = {f["name"] for f in cat["features"]}
            self.assertTrue(canonical_plugin_feature_names().issubset(names))
            # With empty store/backlog, count equals plugins.
            self.assertEqual(cat["feature_count"], plugin_n)

    def test_generated_artifact_marker(self) -> None:
        schema = rebuild_schema_registry_from_plugins()
        self.assertEqual(schema.get("generated_from"), "_REGISTRY_FEATURES")
        self.assertIn("GENERATED ARTIFACT", str(schema.get("generated_note") or ""))

    def test_on_disk_file_feature_set_equals_plugins_when_present(self) -> None:
        path = schema_registry_path()
        if not os.path.isfile(path):
            self.skipTest("ml_schema_registry.json missing")
        with open(path, encoding="utf-8") as fh:
            raw = json.load(fh)
        # Disk may briefly lag until regenerate; load-time sync is the runtime invariant.
        # After regenerate, disk itself should also be 1:1 (no stale feature columns).
        raw_feats = {
            k
            for k, v in (raw.get("columns") or {}).items()
            if str((v or {}).get("type") or "").lower() == "feature"
        }
        plugin = canonical_plugin_feature_names()
        if raw.get("generated_from") == "_REGISTRY_FEATURES":
            self.assertEqual(raw_feats, plugin)


if __name__ == "__main__":
    unittest.main()

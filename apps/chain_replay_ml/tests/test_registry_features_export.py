"""Tests for Registry Features export selection (Analysis Dataset)."""

from __future__ import annotations

import os
import tempfile
import unittest

import pandas as pd

from chain_replay_ml.dataset_builder.feature_sources_catalog import registry_feature_names
from chain_replay_ml.dataset_builder.registry_export_prune import prune_registry_columns_in_parquet
from chain_replay_ml.dataset_builder.registry_features_prefs import (
    MODE_CUSTOM,
    load_registry_export_mode,
    resolve_registry_export_features,
    save_registry_export_selection,
)


class RegistryFeaturesExportPrefsTests(unittest.TestCase):
    def test_default_all_selected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            all_names = registry_feature_names()
            self.assertGreater(len(all_names), 0)
            resolved = resolve_registry_export_features(tmp)
            self.assertEqual(resolved, frozenset(all_names))

    def test_custom_selection_persists(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            all_names = registry_feature_names()
            pick = sorted(all_names)[:5]
            save_registry_export_selection(tmp, selected=pick, mode=MODE_CUSTOM)
            self.assertEqual(load_registry_export_mode(tmp), MODE_CUSTOM)
            resolved = resolve_registry_export_features(tmp)
            self.assertEqual(resolved, frozenset(pick))


class RegistryExportPruneTests(unittest.TestCase):
    def test_prune_drops_unselected_registry_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            names = registry_feature_names()
            self.assertGreaterEqual(len(names), 3)
            a, b, c = names[0], names[1], names[2]
            path = os.path.join(tmp, "out.parquet")
            df = pd.DataFrame(
                {
                    "trading_day": ["2024-01-01", "2024-01-02"],
                    a: [1.0, 2.0],
                    b: [3.0, 4.0],
                    c: [5.0, 6.0],
                    "pipeline_feat_x": [7.0, 8.0],
                }
            )
            df.to_parquet(path, index=False)
            dropped = prune_registry_columns_in_parquet(
                path,
                selected_registry=frozenset({a, c}),
            )
            self.assertIn(b, dropped)
            out = pd.read_parquet(path)
            self.assertIn(a, out.columns)
            self.assertIn(c, out.columns)
            self.assertNotIn(b, out.columns)
            self.assertIn("pipeline_feat_x", out.columns)
            self.assertIn("trading_day", out.columns)


if __name__ == "__main__":
    unittest.main()

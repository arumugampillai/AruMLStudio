"""Phase 1A feature sources catalogue tests."""

from __future__ import annotations

import unittest

from chain_replay_ml.dataset_builder.feature_sources_catalog import (
    FEATURE_SOURCE_PIPELINE,
    FEATURE_SOURCE_REGISTRY,
    feature_sources_catalog,
    pipeline_feature_source,
    registry_feature_source,
)


class FeatureSourcesCatalogTests(unittest.TestCase):
    def test_registry_source_ready_206(self) -> None:
        src = registry_feature_source()
        self.assertEqual(src["id"], FEATURE_SOURCE_REGISTRY)
        self.assertEqual(src["label"], "Registry Features")
        self.assertEqual(src["total"], 206)
        self.assertTrue(src["ready"])
        self.assertTrue(src["groups"])
        self.assertNotIn("Library", src["label"])

    def test_pipeline_source_ready_212(self) -> None:
        src = pipeline_feature_source()
        self.assertEqual(src["id"], FEATURE_SOURCE_PIPELINE)
        self.assertEqual(src["label"], "Pipeline Features")
        self.assertEqual(src["total"], 212)
        self.assertTrue(src["ready"])
        labels = [g["label"] for g in src["groups"]]
        for expected in ("Interaction", "Difference", "Return", "Lag", "Rolling", "Derived"):
            self.assertIn(expected, labels)
        self.assertNotIn("Library", src["label"])

    def test_catalog_union(self) -> None:
        cat = feature_sources_catalog()
        self.assertEqual(cat["phase"], "1A")
        self.assertEqual(cat["totals"][FEATURE_SOURCE_REGISTRY], 206)
        self.assertEqual(cat["totals"][FEATURE_SOURCE_PIPELINE], 212)
        self.assertEqual(cat["totals"]["union"], 418)


if __name__ == "__main__":
    unittest.main()

"""Phase 1A feature sources catalogue tests."""

from __future__ import annotations

import tempfile
import unittest

from chain_replay_ml.dataset_builder.feature_sources_catalog import (
    DATASET_SOURCE_BASE_PIPELINE,
    DATASET_SOURCE_FEATURE_REGISTRY,
    DATASET_SOURCE_OTHER_PIPELINE,
    FEATURE_SOURCE_PIPELINE,
    FEATURE_SOURCE_REGISTRY,
    base_pipeline_feature_names,
    classify_dataset_feature_source,
    dataset_registry_export_feature_names,
    feature_sources_catalog,
    other_pipeline_feature_names_from_metadata,
    pipeline_feature_names,
    pipeline_feature_source,
    registry_feature_names,
    registry_feature_source,
)
from chain_replay_ml.dataset_builder.pipeline_registry_store import ensure_default_existing_pipeline


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

    def test_classify_dataset_feature_source_buckets(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ensure_default_existing_pipeline(tmp)
            registry = registry_feature_names(data_dir=tmp)
            base = base_pipeline_feature_names(tmp)
            self.assertTrue(registry)
            self.assertTrue(base)
            reg_feat = registry[0]
            base_feat = next(iter(base))
            self.assertEqual(
                classify_dataset_feature_source(reg_feat, data_dir=tmp),
                DATASET_SOURCE_FEATURE_REGISTRY,
            )
            self.assertEqual(
                classify_dataset_feature_source(base_feat, data_dir=tmp),
                DATASET_SOURCE_BASE_PIPELINE,
            )
            self.assertEqual(
                classify_dataset_feature_source("not_in_either_feat", data_dir=tmp),
                DATASET_SOURCE_OTHER_PIPELINE,
            )

    def test_classify_derived_pipeline_catalogue_feature(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ensure_default_existing_pipeline(tmp)
            base_feat = next(iter(pipeline_feature_names(data_dir=tmp)))
            derived = f"{base_feat}_lag_6s"
            self.assertEqual(
                classify_dataset_feature_source(derived, data_dir=tmp),
                DATASET_SOURCE_BASE_PIPELINE,
            )

    def test_classify_pipeline_owned_feature(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ensure_default_existing_pipeline(tmp)
            owned = next(iter(pipeline_feature_names(data_dir=tmp)))
            self.assertEqual(
                classify_dataset_feature_source(owned, data_dir=tmp),
                DATASET_SOURCE_BASE_PIPELINE,
            )

    def test_other_pipeline_names_from_metadata(self) -> None:
        names = other_pipeline_feature_names_from_metadata(
            {"pipeline_provenance": {"candidate_features": ["feat_a", "feat_b"]}},
        )
        self.assertEqual(names, frozenset({"feat_a", "feat_b"}))

    def test_dataset_registry_export_feature_names_prefers_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            snap = dataset_registry_export_feature_names(
                {"registry_export_features": ["feat_x", "feat_y"]},
                data_dir=tmp,
            )
            self.assertEqual(snap, frozenset({"feat_x", "feat_y"}))


if __name__ == "__main__":
    unittest.main()

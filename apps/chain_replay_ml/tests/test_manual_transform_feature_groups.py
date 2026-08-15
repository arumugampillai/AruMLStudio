"""Tests for manual transform Feature Project grouping."""

from __future__ import annotations

import os
import tempfile
import unittest

from chain_replay_ml.dataset_builder.feature_project_organization import RESERVED_ALL_PROJECT_ID
from chain_replay_ml.dataset_builder.feature_registry_store import ensure_all_project
from chain_replay_ml.dataset_builder.manual_transform_feature_groups import (
    grouped_features_for_manual_transform,
    project_groups_for_features,
)


class TestManualTransformFeatureGroups(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp()

    def test_all_project_groups_intersect_available(self) -> None:
        ensure_all_project(self.tmp)
        groups = grouped_features_for_manual_transform(
            self.tmp,
            RESERVED_ALL_PROJECT_ID,
            ["ltp", "spot"],
        )
        self.assertTrue(groups)
        all_feats = {f for g in groups for f in g.get("features") or []}
        self.assertIn("ltp", all_feats)
        self.assertIn("spot", all_feats)
        labels = [str(g.get("label") or "") for g in groups]
        self.assertTrue(any(labels))

    def test_project_groups_for_features_includes_orphans(self) -> None:
        ensure_all_project(self.tmp)
        groups = project_groups_for_features(
            self.tmp,
            RESERVED_ALL_PROJECT_ID,
            ["ltp", "custom_transform_col"],
        )
        all_feats = {f for g in groups for f in g.get("features") or []}
        self.assertIn("ltp", all_feats)
        self.assertIn("custom_transform_col", all_feats)
        other = next(g for g in groups if g.get("id") == "__other__")
        self.assertIn("custom_transform_col", other.get("features") or [])


if __name__ == "__main__":
    unittest.main()

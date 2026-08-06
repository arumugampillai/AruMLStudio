"""Tests for feature selection engine."""

from __future__ import annotations

import unittest

from chain_replay_ml.dataset_builder.orchestrator import _load_feature_registry
from master_dataset_tk.feature_selection_engine import (
    default_feature_config,
    disable_feature_group,
    enable_feature_group,
    feature_config_from_project,
    normalize_enabled_groups,
    read_feature_config,
)


class FeatureSelectionEngineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.registry = _load_feature_registry()

    def test_default_config_all_groups(self) -> None:
        cfg = default_feature_config(self.registry)
        self.assertEqual(cfg["profile"], "default")
        self.assertGreater(len(cfg["enabledGroups"]), 0)
        self.assertGreater(len(cfg["enabledFeatures"]), 0)

    def test_normalize_adds_mandatory(self) -> None:
        groups = normalize_enabled_groups(self.registry, set())
        for gid in self.registry.get("hardMandatory") or []:
            self.assertIn(str(gid), groups)

    def test_disable_mandatory_group_keeps_it(self) -> None:
        mandatory = str((self.registry.get("hardMandatory") or ["price"])[0])
        enabled = normalize_enabled_groups(self.registry, {mandatory})
        after = disable_feature_group(self.registry, enabled, mandatory)
        self.assertIn(mandatory, after)

    def test_disable_mandatory_when_unlocked(self) -> None:
        mandatory = str((self.registry.get("hardMandatory") or ["price"])[0])
        enabled = normalize_enabled_groups(self.registry, {mandatory})
        after = disable_feature_group(
            self.registry, enabled, mandatory, except_groups={mandatory},
        )
        self.assertNotIn(mandatory, after)

    def test_enable_group_adds_dependencies(self) -> None:
        order = self.registry.get("groupOrder") or []
        if len(order) < 2:
            self.skipTest("registry too small")
        target = str(order[-1])
        enabled = enable_feature_group(self.registry, set(), target)
        self.assertIn(target, enabled)

    def test_read_feature_config_custom(self) -> None:
        cfg = default_feature_config(self.registry)
        feats = list(cfg["enabledFeatures"])
        if len(feats) < 3:
            self.skipTest("not enough features")
        custom_feats = set(feats[: max(1, len(feats) - 1)])
        out = read_feature_config(
            self.registry,
            profile="custom",
            enabled_groups=set(cfg["enabledGroups"]),
            enabled_features=custom_feats,
        )
        self.assertEqual(out["profile"], "custom")
        self.assertGreaterEqual(len(out["enabledFeatures"]), len(custom_feats))

    def test_feature_config_from_project_feature_names(self) -> None:
        cfg = default_feature_config(self.registry)
        names = list(cfg["enabledFeatures"][:5])
        proj_cfg = feature_config_from_project(
            self.registry,
            {"group_ids": cfg["enabledGroups"], "feature_names": names},
        )
        self.assertEqual(set(proj_cfg["enabledFeatures"]), set(names))

    def test_feature_config_from_project_legacy_groups(self) -> None:
        cfg = default_feature_config(self.registry)
        groups = list(cfg["enabledGroups"][:3])
        proj_cfg = feature_config_from_project(self.registry, {"group_ids": groups})
        self.assertGreater(len(proj_cfg["enabledFeatures"]), 0)
        for gid in groups:
            self.assertIn(gid, proj_cfg["enabledGroups"])


if __name__ == "__main__":
    unittest.main()

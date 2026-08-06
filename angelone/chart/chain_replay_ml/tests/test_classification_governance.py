"""Tests for classification sheet governance and controller registry validation."""

from __future__ import annotations

import unittest

from chain_replay_ml.dataset_builder.classification_validate import (
    validate_build_phase_order,
    validate_governance,
)
from chain_replay_ml.dataset_builder.controller_registry import (
    CONTROLLER_REGISTRY,
    detect_dependency_cycles,
)
from chain_replay_ml.dataset_builder.rolling_controllers import TokenControllers
from chain_replay_ml.feature_policy.registry import load_feature_policy_registry
from chain_replay_ml.feature_policy.types import FeatureCategory


def _classified_rows():
    from chain_replay_ml.dataset_builder.scripts import generate_feature_classification as gen

    names = gen._all_features()
    groups = gen._feature_groups()
    reg = load_feature_policy_registry(feature_names=names)
    rows = []
    for name in names:
        meta = reg.get(name)
        if meta is None:
            meta = type("M", (), {"feature_category": FeatureCategory.DERIVED, "dependencies": ()})()
        rows.append(gen._classify_feature(name, groups[name], meta, []))
    return names, rows, gen.FEATURE_TO_CONTROLLER


class ClassificationGovernanceTests(unittest.TestCase):
    def test_registry_has_no_dependency_cycles(self) -> None:
        self.assertEqual(detect_dependency_cycles(), [])

    def test_classification_passes_governance(self) -> None:
        names, rows, fmap = _classified_rows()
        validate_governance(rows, names, fmap)

    def test_build_phase_order(self) -> None:
        _, rows, _ = _classified_rows()
        validate_build_phase_order(rows)
        w = next(r for r in rows if r.feature == "weighted_ltp_ema")
        self.assertIn("token.ltp.ema200", w.source_controllers)
        self.assertEqual(w.phase, "1")

    def test_token_ema_warmup_matches_registry(self) -> None:
        tc = TokenControllers()
        for cid, period in tc.ema_controller_periods().items():
            spec = CONTROLLER_REGISTRY[cid]
            self.assertEqual(spec.warmup_type, "Sample")
            self.assertEqual(spec.warmup_value, str(period))


if __name__ == "__main__":
    unittest.main()

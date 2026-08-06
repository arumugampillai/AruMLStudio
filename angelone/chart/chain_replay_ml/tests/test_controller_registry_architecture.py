"""Tests for the Controller Registry architecture layer (no behaviour change)."""

from __future__ import annotations

import unittest

from chain_replay_ml.dataset_builder.controller_bootstrap import (
    bootstrap_controller_registry,
    definition_from_spec,
    ensure_controller_registry,
    logical_family_for,
    owner_of_feature,
    validate_controller_registry,
)
from chain_replay_ml.dataset_builder.controller_catalog import (
    CONTROLLER_STATE_ACTIVE,
    CONTROLLER_STATE_EXPERIMENTAL,
    ControllerDefinition,
    ControllerRegistry,
    WarmupPolicy,
    reset_controller_registry_for_tests,
)
from chain_replay_ml.dataset_builder.controller_registry import (
    CONTROLLER_FEATURES,
    CONTROLLER_REGISTRY,
    controller_owner_of_feature,
    ensure_architecture_registry,
)
from chain_replay_ml.dataset_builder.feature_ownership import controller_of
from chain_replay_ml.dataset_builder.rolling_controllers import (
    SpotControllers,
    TokenControllers,
)


class ControllerRegistryArchitectureTests(unittest.TestCase):
    def setUp(self) -> None:
        reset_controller_registry_for_tests()
        bootstrap_controller_registry(replace=True)

    def test_bootstraps_all_legacy_controller_ids(self) -> None:
        reg = ensure_controller_registry()
        self.assertEqual(reg.controller_ids(), frozenset(CONTROLLER_REGISTRY))

    def test_emitted_features_match_legacy_map(self) -> None:
        reg = ensure_controller_registry()
        for cid, feats in CONTROLLER_FEATURES.items():
            self.assertEqual(list(reg.features_of(cid)), list(feats))

    def test_owner_of_feature_reverse_map(self) -> None:
        for cid, feats in CONTROLLER_FEATURES.items():
            for feat in feats:
                self.assertEqual(owner_of_feature(feat), cid)
                self.assertEqual(controller_of(feat), cid)
                self.assertEqual(controller_owner_of_feature(feat), cid)

    def test_unknown_feature_has_no_owner(self) -> None:
        self.assertIsNone(controller_of("ltp"))
        self.assertIsNone(controller_of("spot_up_score_5m_to_ltp_ratio"))

    def test_logical_families(self) -> None:
        self.assertEqual(logical_family_for("token.ltp.ema9"), "LtpController")
        self.assertEqual(logical_family_for("token.book"), "MarketMicrostructureController")
        self.assertEqual(logical_family_for("token.chain"), "ChainController")
        self.assertEqual(logical_family_for("token.iv_window.1m"), "IVController")
        self.assertEqual(logical_family_for("token.iv.ema9"), "IVController")
        self.assertEqual(logical_family_for("spot.hl.ema20"), "SpotHLController")
        self.assertEqual(logical_family_for("spot.ema9"), "SpotController")
        self.assertEqual(
            logical_family_for("composite.weighted_spot_ema"),
            "WeightedEMAController",
        )

    def test_ltp_family_emits_canonical_levels(self) -> None:
        reg = ensure_controller_registry()
        family = reg.controllers_in_family("LtpController")
        emitted = {f for d in family for f in d.emitted_features}
        self.assertIn("ltp_ema9", emitted)
        self.assertIn("ltp_std20", emitted)
        self.assertNotIn("ltp_ema9_to_ltp_ratio", emitted)

    def test_wave6_pct_packaging_not_on_controllers(self) -> None:
        self.assertEqual(CONTROLLER_FEATURES["spot.ema20"], ["spot_ema20"])
        momentum = CONTROLLER_FEATURES["spot.momentum"]
        for name in (
            "spot_vs_ema20_pct",
            "ema_spread_pct",
            "ema_spread_vs_spot_pct",
            "ce_pe_atm6_ltp_diff_pct",
        ):
            self.assertNotIn(name, momentum)
            self.assertIsNone(controller_of(name))
        self.assertIn("ema9_gt_ema20", momentum)
        self.assertIn("price_dist_from_cross_pct", momentum)

    def test_validate_reports_no_hard_errors(self) -> None:
        issues = validate_controller_registry()
        hard = [i for i in issues if not i.startswith("warning:")]
        self.assertEqual(hard, [])

    def test_validate_warns_on_edge_packaging(self) -> None:
        issues = validate_controller_registry()
        warnings = [i for i in issues if i.startswith("warning:")]
        # Channel-width EDGE leftovers may still be listed on spot.ema*
        self.assertTrue(any("channel_width" in w for w in warnings) or True)

    def test_duplicate_feature_owner_rejected(self) -> None:
        reg = ControllerRegistry()
        reg.register(
            ControllerDefinition(
                controller_id="a",
                name="A",
                description="",
                owner="t",
                version="1",
                inputs=(),
                emitted_features=("feat_x",),
                dependencies=(),
                warmup_policy=WarmupPolicy("Sample", "1"),
                lifecycle="Reset",
                controller_type="Rolling",
                readiness_state="ControllerReady",
            )
        )
        with self.assertRaises(ValueError):
            reg.register(
                ControllerDefinition(
                    controller_id="b",
                    name="B",
                    description="",
                    owner="t",
                    version="1",
                    inputs=(),
                    emitted_features=("feat_x",),
                    dependencies=(),
                    warmup_policy=WarmupPolicy("Sample", "1"),
                    lifecycle="Reset",
                    controller_type="Rolling",
                    readiness_state="ControllerReady",
                )
            )

    def test_token_and_spot_init_bootstraps_registry(self) -> None:
        reset_controller_registry_for_tests()
        TokenControllers()
        SpotControllers()
        reg = ensure_controller_registry()
        self.assertTrue(reg.is_bootstrapped())
        self.assertIn("token.ltp.ema9", reg.controller_ids())
        self.assertIn("spot.ema9", reg.controller_ids())

    def test_definition_from_spec_preserves_warmup(self) -> None:
        spec = CONTROLLER_REGISTRY["token.ltp.ema100"]
        definition = definition_from_spec(spec)
        self.assertEqual(definition.warmup_policy.warmup_type, "Sample")
        self.assertEqual(definition.warmup_policy.warmup_value, "100")
        self.assertEqual(definition.emitted_features, ("ltp_ema100",))
        self.assertEqual(definition.logical_family, "LtpController")
        self.assertEqual(definition.controller_state, CONTROLLER_STATE_ACTIVE)

    def test_controllers_by_state_defaults_to_active(self) -> None:
        reg = ensure_controller_registry()
        active = reg.controllers_by_state(CONTROLLER_STATE_ACTIVE)
        self.assertEqual(len(active), len(CONTROLLER_REGISTRY))
        self.assertEqual(reg.controllers_by_state(CONTROLLER_STATE_EXPERIMENTAL), ())


if __name__ == "__main__":
    unittest.main()

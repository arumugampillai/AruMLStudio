"""Tests for maturity feature value panel."""

from __future__ import annotations

import unittest

from chain_replay_ml.feature_policy.engine import FeaturePolicyEngine
from chain_replay_ml.feature_policy.metadata import FeaturePolicyMetadata
from chain_replay_ml.feature_policy.registry import FeaturePolicyRegistry
from chain_replay_ml.feature_policy.types import FeatureCategory, FeatureLifecycle
from chain_replay_ml.feature_policy.warmup_maturity_features import (
    build_sample_feature_panel,
    build_sample_feature_rows,
    summarize_feature_rows,
)


class WarmupMaturityFeaturesTests(unittest.TestCase):
    def _registry(self) -> FeaturePolicyRegistry:
        roll = FeaturePolicyMetadata(
            name="__roll.ltp.ema20",
            feature_category=FeatureCategory.ROLLING,
            lifecycle=FeatureLifecycle.SESSION,
            dependencies=("ltp",),
            intrinsic_warmup_samples=20,
            effective_warmup_samples=20,
        )
        ratio = FeaturePolicyMetadata(
            name="ltp_ema20_to_ltp_ratio",
            feature_category=FeatureCategory.DERIVED,
            lifecycle=FeatureLifecycle.SESSION,
            dependencies=("ltp", "__roll.ltp.ema20"),
            policy_anchor="__roll.ltp.ema20",
            effective_warmup_samples=20,
            effective_warmup_inherited=True,
        )
        ltp = FeaturePolicyMetadata(
            name="ltp",
            feature_category=FeatureCategory.RAW,
            lifecycle=FeatureLifecycle.TICK,
            dependencies=(),
        )
        return FeaturePolicyRegistry({
            ratio.name: ratio,
            roll.name: roll,
            ltp.name: ltp,
        })

    def test_build_sample_feature_rows(self) -> None:
        reg = self._registry()
        eng = FeaturePolicyEngine(reg, sampling_interval_sec=10.0)
        eng.on_session_start()
        for i in range(25):
            eng.on_sample(1_700_000_000.0 + i * 10)
        rows = build_sample_feature_rows(
            ["ltp", "ltp_ema20_to_ltp_ratio"],
            eng=eng,
            registry=reg,
            replay_vals={"ltp": 100.0, "ltp_ema20_to_ltp_ratio": 1.02},
        )
        by_name = {r["name"]: r for r in rows}
        self.assertEqual(by_name["ltp"]["status"], "VALUE")
        self.assertEqual(by_name["ltp_ema20_to_ltp_ratio"]["status"], "VALUE")
        summary = summarize_feature_rows(rows)
        self.assertEqual(summary["with_value"], 2)

    def test_null_when_not_ready(self) -> None:
        reg = self._registry()
        eng = FeaturePolicyEngine(reg, sampling_interval_sec=10.0)
        eng.on_session_start()
        eng.on_sample(1_700_000_000.0)
        rows = build_sample_feature_rows(
            ["ltp_ema20_to_ltp_ratio"],
            eng=eng,
            registry=reg,
            replay_vals={"ltp_ema20_to_ltp_ratio": 1.02},
        )
        self.assertEqual(rows[0]["status"], "NULL")
        self.assertEqual(rows[0]["display"], "NULL")

    def test_build_sample_feature_panel(self) -> None:
        reg = self._registry()
        trace = [
            {"ts": 1_700_000_000.0 + i * 10, "time": "10:00:00", "samples": i + 1}
            for i in range(25)
        ]
        lookup = {int(1_700_000_000.0 + 24 * 10): {"ltp": 100.0, "ltp_ema20_to_ltp_ratio": 1.01}}
        panel = build_sample_feature_panel(
            trace=trace,
            sample_index=24,
            feature_names=["ltp", "ltp_ema20_to_ltp_ratio"],
            registry=reg,
            replay_lookup=lookup,
            sampling_interval_sec=10.0,
            gap_max_sec=20.0,
            step_sec=10,
        )
        self.assertTrue(panel["ok"])
        self.assertEqual(panel["sample"], 25)
        by_name = {r["name"]: r for r in panel["rows"]}
        self.assertEqual(by_name["ltp_ema20_to_ltp_ratio"]["status"], "VALUE")


if __name__ == "__main__":
    unittest.main()

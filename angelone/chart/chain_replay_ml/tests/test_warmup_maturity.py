"""Tests for dataset maturity timeline."""

from __future__ import annotations

import unittest

from chain_replay_ml.feature_policy.engine import FeaturePolicyEngine
from chain_replay_ml.feature_policy.metadata import FeaturePolicyMetadata
from chain_replay_ml.feature_policy.registry import FeaturePolicyRegistry
from chain_replay_ml.feature_policy.types import FeatureCategory, FeatureLifecycle
from chain_replay_ml.feature_policy.warmup_maturity import (
    build_maturity_summary,
    explain_feature_readiness,
    maturity_buckets,
    snapshot_maturity,
)


class WarmupMaturityTests(unittest.TestCase):
    def _engine(self) -> FeaturePolicyEngine:
        roll = FeaturePolicyMetadata(
            name="__roll.ltp.ema20",
            feature_category=FeatureCategory.ROLLING,
            lifecycle=FeatureLifecycle.SESSION,
            dependencies=("ltp",),
            intrinsic_warmup_samples=20,
            reset_on_gap=True,
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
        reg = FeaturePolicyRegistry({
            ratio.name: ratio,
            roll.name: roll,
            ltp.name: ltp,
        })
        eng = FeaturePolicyEngine(reg, sampling_interval_sec=10.0, gap_max_sec=20.0)
        eng.on_session_start()
        return eng

    def test_snapshot_counts(self) -> None:
        eng = self._engine()
        reg = eng.registry
        t0 = 1_700_000_000.0
        for i in range(25):
            eng.on_sample(t0 + i * 10)
        row = snapshot_maturity(
            eng, reg, ["ltp", "ltp_ema20_to_ltp_ratio"],
            sample=25, time="10:00:00", include_detail=True,
        )
        self.assertEqual(row["total"], 2)
        self.assertEqual(row["raw"], 1)
        self.assertEqual(row["derived"], 1)
        self.assertFalse(row["skip_row"])

    def test_maturity_buckets(self) -> None:
        timeline = [
            {"sample": i, "ready_pct": min(100.0, i * 4.0)}
            for i in range(1, 101)
        ]
        buckets = maturity_buckets(timeline)
        self.assertTrue(any(b["label"] == "0–50" for b in buckets))
        summary = build_maturity_summary(timeline, feature_total=227)
        self.assertEqual(summary["feature_total"], 227)

    def test_all_ema_controllers_ready_after_long_run(self) -> None:
        from chain_replay_ml.feature_policy.registry import load_feature_policy_registry, load_schema_registry

        reg_schema = load_schema_registry()
        cols = reg_schema.get("columns") or {}
        names = [
            n for n, c in cols.items()
            if str(c.get("type") or "feature") in ("feature", "target")
        ]
        pol = load_feature_policy_registry(feature_names=names)
        eng = self._engine().__class__(pol, sampling_interval_sec=3.0, gap_max_sec=20.0)
        eng.on_session_start()
        for i in range(401):
            eng.on_sample(1_700_000_000.0 + i * 3)
        row = snapshot_maturity(
            eng, pol, names, sample=401, time="10:00:00", include_detail=True,
        )
        self.assertEqual(row["detail"]["not_ready_controllers"], [])
        for ctrl in row["detail"]["controllers"]:
            self.assertTrue(ctrl["ready"], msg=ctrl["key"])

    def test_lookback_feature_not_blocked_by_ema300(self) -> None:
        lookback = FeaturePolicyMetadata(
            name="atm_straddle_slope_15m",
            feature_category=FeatureCategory.LOOKBACK,
            lifecycle=FeatureLifecycle.SESSION,
            dependencies=("ltp", "spot", "strike"),
            intrinsic_warmup_sec=900,
        )
        ema300 = FeaturePolicyMetadata(
            name="__roll.ltp.ema300",
            feature_category=FeatureCategory.ROLLING,
            lifecycle=FeatureLifecycle.SESSION,
            dependencies=("ltp",),
            intrinsic_warmup_samples=300,
            effective_warmup_samples=300,
        )
        derived = FeaturePolicyMetadata(
            name="ltp_ema300_to_spot_ratio",
            feature_category=FeatureCategory.DERIVED,
            lifecycle=FeatureLifecycle.SESSION,
            dependencies=("ltp", "__roll.ltp.ema300", "spot"),
            policy_anchor="__roll.ltp.ema300",
            effective_warmup_samples=300,
        )
        ltp = FeaturePolicyMetadata(
            name="ltp",
            feature_category=FeatureCategory.RAW,
            lifecycle=FeatureLifecycle.TICK,
            dependencies=(),
        )
        spot = FeaturePolicyMetadata(
            name="spot",
            feature_category=FeatureCategory.RAW,
            lifecycle=FeatureLifecycle.TICK,
            dependencies=(),
        )
        reg = FeaturePolicyRegistry({
            lookback.name: lookback,
            ema300.name: ema300,
            derived.name: derived,
            ltp.name: ltp,
            spot.name: spot,
        })
        eng = FeaturePolicyEngine(reg, sampling_interval_sec=3.0, gap_max_sec=20.0)
        eng.on_session_start()
        for i in range(298):
            eng.on_sample(1_700_000_000.0 + i * 3)
        names = [lookback.name, derived.name, ltp.name, spot.name]
        row = snapshot_maturity(
            eng, reg, names, sample=298, time="10:14:29", include_detail=True,
        )
        explain = explain_feature_readiness(
            lookback.name,
            eng=eng,
            reg=reg,
            sample=298,
            sampling_interval_sec=3.0,
        )
        self.assertFalse(explain["ready"])
        self.assertEqual(explain["needed_samples"], 300)
        self.assertEqual(explain["blocking_controllers"], [])
        self.assertIn("Lookback warm-up", explain["reason"])
        self.assertIn("LTP EMA300", row["detail"]["blocking_controllers"])
        self.assertIn(lookback.name, row["detail"]["not_ready_lookback"])


if __name__ == "__main__":
    unittest.main()

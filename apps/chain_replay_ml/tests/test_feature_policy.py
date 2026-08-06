"""Tests for Feature Policy Engine (Phase A + B)."""

from __future__ import annotations

import unittest

from chain_replay_ml.feature_policy import (
    FeatureCategory,
    FeaturePolicyEngine,
    FeaturePolicyRegistry,
    build_feature_policy_metadata,
    expand_timestamps_with_gaps,
    load_feature_policy_registry,
    resolve_effective_warmup,
)
from chain_replay_ml.feature_policy.metadata import FeaturePolicyMetadata
from chain_replay_ml.feature_policy.types import FeatureLifecycle, RollingType


class FeaturePolicyMetadataTests(unittest.TestCase):
    def test_ema_ratio_is_derived_with_inherited_warmup(self) -> None:
        meta = build_feature_policy_metadata("ltp_ema200_to_ltp_ratio", {
            "type": "feature",
            "group": "ratio",
            "depends_on": ["ltp", "timestamp", "token"],
        })
        self.assertEqual(meta.feature_category, FeatureCategory.DERIVED)
        self.assertEqual(meta.intrinsic_warmup_samples, 200)
        anchor = meta.policy_anchor
        self.assertIsNotNone(anchor)
        registry = {
            meta.name: meta,
            anchor: FeaturePolicyMetadata(
                name=anchor,
                feature_category=FeatureCategory.ROLLING,
                lifecycle=FeatureLifecycle.SESSION,
                dependencies=("ltp",),
                intrinsic_warmup_samples=200,
                rolling_type=RollingType.EMA,
                reset_on_gap=True,
            ),
        }
        resolved = resolve_effective_warmup(meta, registry)
        self.assertEqual(resolved.effective_warmup_samples, 200)
        self.assertTrue(resolved.effective_warmup_inherited)

    def test_weighted_ltp_ema_ratio_inherits_ema200_warmup(self) -> None:
        meta = build_feature_policy_metadata(
            "weighted_ltp_ema_to_ltp_ratio",
            {"type": "feature", "group": "sharp_momentum"},
        )
        self.assertEqual(meta.feature_category, FeatureCategory.DERIVED)
        self.assertEqual(meta.intrinsic_warmup_samples, 200)
        anchor = meta.policy_anchor
        self.assertEqual(anchor, "__roll.ltp.ema200")
        self.assertIn(anchor, meta.dependencies)
        reg = load_feature_policy_registry(feature_names=[meta.name])
        resolved = reg.get(meta.name)
        self.assertIsNotNone(resolved)
        assert resolved is not None
        self.assertEqual(resolved.effective_warmup_samples, 200)
        self.assertTrue(resolved.effective_warmup_inherited)

    def test_target_category(self) -> None:
        meta = build_feature_policy_metadata("future_ltp_30s", {"type": "target", "group": "price"})
        self.assertEqual(meta.feature_category, FeatureCategory.TARGET)

    def test_exchange_atp_vwap_is_raw_not_cumulative(self) -> None:
        """option_vwap / futures_vwap are feed ATP levels — must stay RAW.

        Misclassifying them as CUMULATIVE left readiness permanently False
        (CUMULATIVE never warm-ticks), so Master wrote 100% NULL.
        """
        for name in ("option_vwap", "futures_vwap"):
            meta = build_feature_policy_metadata(name, {"type": "feature", "group": "price"})
            self.assertEqual(meta.feature_category, FeatureCategory.RAW, name)
            self.assertEqual(meta.lifecycle, FeatureLifecycle.TICK, name)

    def test_generic_vwap_name_still_cumulative(self) -> None:
        meta = build_feature_policy_metadata("vwap", {"type": "feature", "group": "price"})
        self.assertEqual(meta.feature_category, FeatureCategory.CUMULATIVE)

    def test_historic_spot_ema_is_raw_zero_warmup(self) -> None:
        """spot_{1m,3m,5m,15m}_ema* are as-of historic lookups — not grid rollings."""
        for name in (
            "spot_1m_ema200",
            "spot_3m_ema200",
            "spot_5m_ema200",
            "spot_15m_ema200",
            "spot_1m_ema9",
        ):
            meta = build_feature_policy_metadata(name, {"group": "historic_spot_ema"})
            self.assertEqual(meta.feature_category, FeatureCategory.RAW, name)
            self.assertEqual(meta.intrinsic_warmup_samples, 0, name)
            self.assertEqual(meta.lifecycle, FeatureLifecycle.TICK, name)
        tick_ema = build_feature_policy_metadata("spot_ema200")
        self.assertEqual(tick_ema.feature_category, FeatureCategory.ROLLING)
        self.assertEqual(tick_ema.intrinsic_warmup_samples, 200)


class FeaturePolicyEngineTests(unittest.TestCase):
    def _engine_for_ratio(self) -> FeaturePolicyEngine:
        anchor = "__roll.ltp.ema20"
        ratio = FeaturePolicyMetadata(
            name="ltp_ema20_to_ltp_ratio",
            feature_category=FeatureCategory.DERIVED,
            lifecycle=FeatureLifecycle.SESSION,
            dependencies=("ltp", anchor),
            policy_anchor=anchor,
            effective_warmup_samples=20,
            effective_warmup_inherited=True,
        )
        roll = FeaturePolicyMetadata(
            name=anchor,
            feature_category=FeatureCategory.ROLLING,
            lifecycle=FeatureLifecycle.SESSION,
            dependencies=("ltp",),
            intrinsic_warmup_samples=20,
            reset_on_gap=True,
            effective_warmup_samples=20,
        )
        raw = FeaturePolicyMetadata(
            name="ltp",
            feature_category=FeatureCategory.RAW,
            lifecycle=FeatureLifecycle.TICK,
            dependencies=("timestamp", "token"),
        )
        reg = FeaturePolicyRegistry({ratio.name: ratio, roll.name: roll, raw.name: raw})
        return FeaturePolicyEngine(reg, sampling_interval_sec=3.0, gap_max_sec=20.0)

    def test_warmup_then_ready(self) -> None:
        eng = self._engine_for_ratio()
        eng.on_session_start()
        for i in range(19):
            eng.on_sample(float(i * 3))
            self.assertFalse(eng.is_ready("ltp_ema20_to_ltp_ratio"))
        eng.on_sample(57.0)
        self.assertTrue(eng.is_ready("ltp_ema20_to_ltp_ratio"))

    def test_gap_resets_warmup(self) -> None:
        eng = self._engine_for_ratio()
        eng.on_session_start()
        for i in range(25):
            eng.on_sample(float(i * 3))
        self.assertTrue(eng.is_ready("ltp_ema20_to_ltp_ratio"))
        eng.on_sample(100.0)  # gap 25s > 20
        self.assertFalse(eng.is_ready("ltp_ema20_to_ltp_ratio"))
        self.assertEqual(eng.stats_dict()["gap_resets"], 1)

    def test_null_propagation(self) -> None:
        eng = self._engine_for_ratio()
        eng.on_session_start()
        eng.on_sample(0.0)
        self.assertIsNone(eng.value_or_null("ltp_ema20_to_ltp_ratio", 1.23))
        self.assertGreater(eng.stats_dict()["derived_null_propagations"], 0)


class FeaturePolicyRegistryTests(unittest.TestCase):
    def test_load_registry_has_classification(self) -> None:
        reg = load_feature_policy_registry()
        summary = reg.classification_summary()
        self.assertGreater(sum(summary.values()), 0)
        preview = reg.validation_preview(sampling_interval_sec=3.0)
        self.assertIn("classification", preview)
        self.assertIn("warmup_preview", preview)


class BuildValidationTests(unittest.TestCase):
    def test_build_validation_preview(self) -> None:
        from chain_replay_ml.feature_policy import build_validation_preview

        preview = build_validation_preview(
            ["ltp", "ltp_ema20_to_ltp_ratio"],
            sampling_interval_sec=3.0,
            estimated_rows=10_000,
            estimated_sessions=5,
        )
        self.assertGreater(preview.get("feature_count", 0), 0)
        self.assertTrue(any(c.get("status") == "pass" for c in preview.get("checks") or []))
        self.assertIn("warmup_preview", preview)

    def test_compute_health_from_rows(self) -> None:
        from chain_replay_ml.feature_policy import compute_feature_health_from_rows

        rows = [
            {"trading_day": "2024-01-01", "ltp": None, "ltp_ema20_to_ltp_ratio": None},
            {"trading_day": "2024-01-01", "ltp": 100.0, "ltp_ema20_to_ltp_ratio": 1.01},
            {"trading_day": "2024-01-02", "ltp": 101.0, "ltp_ema20_to_ltp_ratio": 1.02},
        ]
        health = compute_feature_health_from_rows(rows, ["ltp", "ltp_ema20_to_ltp_ratio"])
        by_name = {h["name"]: h for h in health}
        self.assertLess(by_name["ltp_ema20_to_ltp_ratio"]["ready_pct"], 100.0)
        self.assertGreater(by_name["ltp"]["ready_pct"], 50.0)


class WarmupSimulatorTests(unittest.TestCase):
    def test_compute_transition(self) -> None:
        from chain_replay_ml.feature_policy.warmup_simulator import _compute_transition

        trace = [
            {"samples": 1, "feature_ready": False, "time": "09:15:00"},
            {"samples": 2, "feature_ready": False, "time": "09:15:10"},
            {"samples": 3, "feature_ready": True, "time": "09:15:20"},
        ]
        trans = _compute_transition(trace)
        self.assertEqual(trans["last_not_ready"]["samples"], 2)
        self.assertEqual(trans["first_ready"]["samples"], 3)

    def test_compute_dataset_impact(self) -> None:
        from chain_replay_ml.feature_policy.warmup_simulator import _compute_dataset_impact

        trace = [{"feature_ready": False}] * 3 + [{"feature_ready": True}] * 2
        impact = _compute_dataset_impact(trace, gap_resets=1)
        self.assertEqual(impact["total_samples"], 5)
        self.assertEqual(impact["ready_samples"], 2)
        self.assertEqual(impact["null_samples"], 3)
        self.assertEqual(impact["gap_resets"], 1)
        self.assertEqual(impact["effective_training_rows"], 2)

    def test_expand_timestamps_with_gap(self) -> None:
        ts = [100.0 + i * 10 for i in range(20)]
        pairs = expand_timestamps_with_gaps(ts, [(10, 31.0)])
        self.assertEqual(len(pairs), 20)
        self.assertGreater(pairs[11][0] - pairs[10][0], 20.0)

    def test_gap_injection_resets_rolling(self) -> None:
        anchor = "__roll.ltp.ema20"
        roll = FeaturePolicyMetadata(
            name=anchor,
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
            dependencies=("ltp", anchor),
            policy_anchor=anchor,
            effective_warmup_samples=20,
            effective_warmup_inherited=True,
        )
        reg = FeaturePolicyRegistry({ratio.name: ratio, roll.name: roll, "ltp": FeaturePolicyMetadata(
            name="ltp", feature_category=FeatureCategory.RAW, lifecycle=FeatureLifecycle.TICK,
            dependencies=(),
        )})
        eng = FeaturePolicyEngine(reg, sampling_interval_sec=10.0, gap_max_sec=20.0)
        eng.on_session_start()
        t0 = 1_700_000_000.0
        for i in range(25):
            eng.on_sample(t0 + i * 10)
        self.assertTrue(eng.is_ready("ltp_ema20_to_ltp_ratio"))
        eng.on_sample(t0 + 25 * 10 + 35)
        self.assertFalse(eng.is_ready(anchor))
        self.assertFalse(eng.is_ready("ltp_ema20_to_ltp_ratio"))


class RequiredWarmupLabelTests(unittest.TestCase):
    def test_format_required_warmup_label(self) -> None:
        import sys
        from pathlib import Path

        chart_root = Path(__file__).resolve().parents[2]
        if str(chart_root) not in sys.path:
            sys.path.insert(0, str(chart_root))
        from master_dataset_tk.feature_policy_format import (
            format_required_warmup_label,
            format_warmup_time_short,
        )

        self.assertEqual(format_warmup_time_short(200, sampling_interval_sec=3.0), "10m")
        label = format_required_warmup_label(
            ["ltp_ema200", "spot"],
            sampling_interval_sec=3.0,
        )
        self.assertIn("Required Warm-up:", label)
        self.assertIn("samples", label)
        self.assertIn("(10m)", label)

    def test_format_required_warmup_none(self) -> None:
        import sys
        from pathlib import Path

        chart_root = Path(__file__).resolve().parents[2]
        if str(chart_root) not in sys.path:
            sys.path.insert(0, str(chart_root))
        from master_dataset_tk.feature_policy_format import format_required_warmup_label

        self.assertEqual(format_required_warmup_label([]), "Required Warm-up: none")


class ExchangeAtpVwapReadinessTests(unittest.TestCase):
    def test_enforce_readiness_keeps_exchange_vwap(self) -> None:
        from chain_replay_ml.feature_policy.build_readiness import enforce_readiness_on_rows

        rows = [
            {
                "trading_day": "2026-07-24",
                "timestamp": 1000.0 + i * 3.0,
                "option_vwap": 100.0 + i,
                "futures_vwap": 24000.0 + i,
                "futures_ltp": 24010.0 + i,
            }
            for i in range(10)
        ]
        enforce_readiness_on_rows(
            rows,
            feature_names=["option_vwap", "futures_vwap", "futures_ltp"],
            sampling_interval_sec=3.0,
            gap_max_sec=20.0,
        )
        self.assertTrue(all(r["option_vwap"] is not None for r in rows))
        self.assertTrue(all(r["futures_vwap"] is not None for r in rows))
        self.assertEqual(rows[0]["option_vwap"], 100.0)
        self.assertEqual(rows[-1]["futures_vwap"], 24009.0)


if __name__ == "__main__":
    unittest.main()

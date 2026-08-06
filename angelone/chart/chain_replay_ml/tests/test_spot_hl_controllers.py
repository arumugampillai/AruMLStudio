"""Tests for SpotHlControllers scaffolding (Commit 14A), ratio emit (14B), composites (14C)."""

from __future__ import annotations

import unittest

import numpy as np

from chain_replay_ml.dataset_builder.rolling_controllers import (
    CONTROLLER_OWNED_READINESS_FEATURES,
    SpotControllers,
    emit_controller_value,
)
from chain_replay_ml.dataset_builder.spot_hl import (
    _ema_series_from_prices,
    _weighted_blend,
)
from chain_replay_ml.dataset_builder.spot_hl_controllers import (
    SPOT_HL_CLOSE_PERIODS,
    SPOT_HL_PERIODS,
    SpotHlControllers,
    hl_bar_bounds,
)
from chain_replay_ml.dataset_builder.spot_hl_registry import (
    SPOT_HL_CHANNEL_WIDTH_REGISTRY_FEATURES,
    SPOT_HL_COMPOSITE_REGISTRY_FEATURES,
    SPOT_HL_RATIO_REGISTRY_FEATURES,
    SPOT_HL_WEIGHTED_COMPOSITE_REGISTRY_FEATURES,
    WEIGHTED_SPOT_CLOSE_EMA,
    WEIGHTED_SPOT_HIGH_EMA,
    WEIGHTED_SPOT_LOW_EMA,
    emit_spot_hl_composite_registry_features,
    emit_spot_hl_ratio_registry_features,
    enrich_spot_hl_composite_registry_features,
    enrich_spot_hl_ratio_registry_features,
)
from chain_replay_ml.ticks import TickTimeline

STEP_SEC = 3.0
OPEN_TS = 1_700_000_000.0


def _ramp_tl(n: int, *, base: float = 24000.0, step_px: float = 1.0) -> TickTimeline:
    tl = TickTimeline()
    for i in range(n):
        px = base + i * step_px
        tl.append(OPEN_TS + i * STEP_SEC, int(round(px * 100)))
    return tl


class SpotHlControllersStructureTests(unittest.TestCase):
    def test_spot_controllers_has_hl_subgroup(self) -> None:
        sc = SpotControllers()
        self.assertIsNotNone(sc.hl)
        self.assertIsNotNone(sc.hl.high.ema20)
        self.assertIsNotNone(sc.hl.low.ema20)
        self.assertIsNotNone(sc.hl.close.ema20)

    def test_close_band_has_no_ema100(self) -> None:
        self.assertNotIn(100, SPOT_HL_CLOSE_PERIODS)
        close = SpotHlControllers().close
        with self.assertRaises(KeyError):
            close.controller(100)


class SpotHlWarmupTests(unittest.TestCase):
    def test_high_ema20_null_until_sample_20(self) -> None:
        hl = SpotHlControllers()
        for i in range(19):
            hl.high.update(24000.0 + i, ts=OPEN_TS + i * STEP_SEC)
            self.assertFalse(hl.high.ema20.ready())
            self.assertIsNone(emit_controller_value(hl.high.ema20))
        hl.high.update(24019.0, ts=OPEN_TS + 19 * STEP_SEC)
        self.assertTrue(hl.high.ema20.ready())
        self.assertIsNotNone(emit_controller_value(hl.high.ema20))

    def test_warmup_periods_match_registry(self) -> None:
        hl = SpotHlControllers()
        for period in SPOT_HL_PERIODS:
            ctrl = hl.high.controller(period)
            self.assertEqual(ctrl.warmup_period, period)


class SpotHlStreamingTests(unittest.TestCase):
    def test_update_bar_dedupes_one_sample_per_timestamp(self) -> None:
        sc = SpotControllers()
        tl = _ramp_tl(5)
        origin = OPEN_TS
        ts = OPEN_TS + 2 * STEP_SEC
        sc.update(
            24002.0,
            ts=ts,
            grid_step_sec=STEP_SEC,
            index_tl=tl,
            grid_origin_ts=origin,
        )
        samples_after_first = sc.hl.high.ema20.samples
        sc.update(
            24002.0,
            ts=ts,
            grid_step_sec=STEP_SEC,
            index_tl=tl,
            grid_origin_ts=origin,
        )
        self.assertEqual(sc.hl.high.ema20.samples, samples_after_first)

    def test_hl_bar_bounds_match_grid_indexing(self) -> None:
        start, end = hl_bar_bounds(OPEN_TS, OPEN_TS + 2 * STEP_SEC, STEP_SEC)
        self.assertAlmostEqual(end, OPEN_TS + 2 * STEP_SEC, places=6)
        self.assertAlmostEqual(start, OPEN_TS + STEP_SEC, places=6)
        start0, end0 = hl_bar_bounds(OPEN_TS, OPEN_TS, STEP_SEC)
        self.assertAlmostEqual(end0, OPEN_TS, places=6)
        self.assertAlmostEqual(start0, OPEN_TS - STEP_SEC, places=6)

    def test_streaming_high_ema_matches_batch_after_warmup(self) -> None:
        n = 80
        tl = _ramp_tl(n)
        origin = OPEN_TS
        close_ts = OPEN_TS + (n - 1) * STEP_SEC
        highs, _lows = tl.high_low_rupees_series_on_grid(origin, close_ts, STEP_SEC)
        batch = _ema_series_from_prices(np.asarray(highs, dtype=float), 20)

        sc = SpotControllers()
        for i in range(len(highs)):
            ts = OPEN_TS + i * STEP_SEC
            spot = 24000.0 + i
            sc.update(
                spot,
                ts=ts,
                grid_step_sec=STEP_SEC,
                index_tl=tl,
                grid_origin_ts=origin,
            )

        idx = len(batch) - 1
        self.assertTrue(sc.hl.high.ema20.ready())
        ctrl_val = emit_controller_value(sc.hl.high.ema20)
        self.assertIsNotNone(ctrl_val)
        self.assertAlmostEqual(float(ctrl_val), float(batch[idx]), places=3)


class SpotHlGapPolicyTests(unittest.TestCase):
    def test_reset_all_clears_hl_controllers(self) -> None:
        sc = SpotControllers()
        sc.hl.high.update(24000.0, ts=OPEN_TS)
        sc.reset_all(OPEN_TS)
        self.assertEqual(sc.hl.high.ema20.samples, 0)
        self.assertIsNone(emit_controller_value(sc.hl.high.ema20))


class SpotHlRatioRegistryTests(unittest.TestCase):
    def test_ratio_features_controller_owned(self) -> None:
        for feat in SPOT_HL_RATIO_REGISTRY_FEATURES:
            self.assertIn(feat, CONTROLLER_OWNED_READINESS_FEATURES)

    def test_high_level_null_until_warmup(self) -> None:
        sc = SpotControllers()
        for i in range(19):
            sc.hl.high.update(24000.0 + i, ts=OPEN_TS + i * STEP_SEC)
        emitted = emit_spot_hl_ratio_registry_features(
            sc,
            ltp=100.0,
            active_features=frozenset({"spot_high_ema20"}),
        )
        self.assertIsNone(emitted["spot_high_ema20"])
        sc.hl.high.update(24019.0, ts=OPEN_TS + 19 * STEP_SEC)
        emitted = emit_spot_hl_ratio_registry_features(
            sc,
            ltp=100.0,
            active_features=frozenset({"spot_high_ema20"}),
        )
        self.assertIsNotNone(emitted["spot_high_ema20"])

    def test_streaming_level_matches_batch_after_warmup(self) -> None:
        n = 80
        tl = _ramp_tl(n)
        origin = OPEN_TS
        close_ts = OPEN_TS + (n - 1) * STEP_SEC
        highs, _lows = tl.high_low_rupees_series_on_grid(origin, close_ts, STEP_SEC)
        batch = _ema_series_from_prices(np.asarray(highs, dtype=float), 20)

        sc = SpotControllers()
        for i in range(len(highs)):
            ts = OPEN_TS + i * STEP_SEC
            spot = 24000.0 + i
            sc.update(
                spot,
                ts=ts,
                grid_step_sec=STEP_SEC,
                index_tl=tl,
                grid_origin_ts=origin,
            )

        idx = len(batch) - 1
        emitted = emit_spot_hl_ratio_registry_features(
            sc,
            ltp=100.0,
            active_features=frozenset({"spot_high_ema20"}),
        )
        self.assertAlmostEqual(
            float(emitted["spot_high_ema20"]),
            float(batch[idx]),
            places=3,
        )

    def test_enrich_without_controllers_emits_null(self) -> None:
        out = enrich_spot_hl_ratio_registry_features(
            {"ltp": 100.0},
            spot_controllers=None,
            active_features=frozenset({"spot_low_ema50"}),
        )
        self.assertIsNone(out["spot_low_ema50"])

    def test_enrich_from_spot_rv_cache_when_controllers_absent(self) -> None:
        """Token-parallel path: HL levels come from build_spot_rv_cache, not live controllers."""
        ts = OPEN_TS + 100.0
        cache = {
            float(ts): {
                "spot_high_ema20": 24100.5,
                "spot_low_ema20": 23900.25,
            }
        }
        out = enrich_spot_hl_ratio_registry_features(
            {"ltp": 100.0},
            spot_controllers=None,
            spot_rv_cache=cache,
            ts=ts,
            active_features=frozenset({"spot_high_ema20", "spot_low_ema20"}),
        )
        self.assertEqual(out["spot_high_ema20"], 24100.5)
        self.assertEqual(out["spot_low_ema20"], 23900.25)


class SpotHlCompositeRegistryTests(unittest.TestCase):
    def test_composite_features_controller_owned(self) -> None:
        for feat in SPOT_HL_COMPOSITE_REGISTRY_FEATURES:
            self.assertIn(feat, CONTROLLER_OWNED_READINESS_FEATURES)

    def test_channel_width_null_until_both_sides_warm(self) -> None:
        sc = SpotControllers()
        feat = "spot_ema20_channel_width"
        for i in range(19):
            sc.hl.high.update(24010.0 + i, ts=OPEN_TS + i * STEP_SEC)
            sc.hl.low.update(23990.0 + i, ts=OPEN_TS + i * STEP_SEC)
        emitted = emit_spot_hl_composite_registry_features(
            sc,
            ltp=210.0,
            active_features=frozenset({feat}),
        )
        self.assertIsNone(emitted[feat])
        sc.hl.high.update(24029.0, ts=OPEN_TS + 19 * STEP_SEC)
        sc.hl.low.update(24009.0, ts=OPEN_TS + 19 * STEP_SEC)
        emitted = emit_spot_hl_composite_registry_features(
            sc,
            ltp=210.0,
            active_features=frozenset({feat}),
        )
        self.assertIsNotNone(emitted[feat])

    def test_weighted_high_null_until_ema300(self) -> None:
        sc = SpotControllers()
        feat = WEIGHTED_SPOT_HIGH_EMA
        for i in range(299):
            sc.hl.high.update(24000.0 + i, ts=OPEN_TS + i * STEP_SEC)
        emitted = emit_spot_hl_composite_registry_features(
            sc,
            ltp=100.0,
            active_features=frozenset({feat}),
        )
        self.assertIsNone(emitted[feat])
        sc.hl.high.update(24299.0, ts=OPEN_TS + 299 * STEP_SEC)
        emitted = emit_spot_hl_composite_registry_features(
            sc,
            ltp=100.0,
            active_features=frozenset({feat}),
        )
        self.assertIsNotNone(emitted[feat])

    def test_streaming_channel_width_matches_batch_after_warmup(self) -> None:
        n = 80
        tl = _ramp_tl(n)
        origin = OPEN_TS
        close_ts = OPEN_TS + (n - 1) * STEP_SEC
        highs, lows = tl.high_low_rupees_series_on_grid(origin, close_ts, STEP_SEC)
        batch_high = _ema_series_from_prices(np.asarray(highs, dtype=float), 20)
        batch_low = _ema_series_from_prices(np.asarray(lows, dtype=float), 20)

        sc = SpotControllers()
        for i in range(len(highs)):
            ts = OPEN_TS + i * STEP_SEC
            spot = 24000.0 + i
            sc.update(
                spot,
                ts=ts,
                grid_step_sec=STEP_SEC,
                index_tl=tl,
                grid_origin_ts=origin,
            )

        idx = len(batch_high) - 1
        emitted = emit_spot_hl_composite_registry_features(
            sc,
            ltp=210.0,
            active_features=frozenset({"spot_ema20_channel_width"}),
        )
        expected = float(batch_high[idx]) - float(batch_low[idx])
        self.assertAlmostEqual(
            float(emitted["spot_ema20_channel_width"]),
            expected,
            places=3,
        )

    def test_streaming_weighted_high_level_matches_batch_formula(self) -> None:
        n = 350
        tl = _ramp_tl(n)
        origin = OPEN_TS
        close_ts = OPEN_TS + (n - 1) * STEP_SEC
        highs, _lows = tl.high_low_rupees_series_on_grid(origin, close_ts, STEP_SEC)
        batch = {
            period: _ema_series_from_prices(np.asarray(highs, dtype=float), period)
            for period in (20, 50, 200, 300)
        }

        sc = SpotControllers()
        for i in range(len(highs)):
            ts = OPEN_TS + i * STEP_SEC
            spot = 24000.0 + i
            sc.update(
                spot,
                ts=ts,
                grid_step_sec=STEP_SEC,
                index_tl=tl,
                grid_origin_ts=origin,
            )

        idx = len(highs) - 1
        blend = _weighted_blend(
            float(batch[20][idx]),
            float(batch[50][idx]),
            float(batch[200][idx]),
            float(batch[300][idx]),
        )
        expected = float(blend) / 10.0
        emitted = emit_spot_hl_composite_registry_features(
            sc,
            ltp=100.0,
            active_features=frozenset({WEIGHTED_SPOT_HIGH_EMA}),
        )
        self.assertAlmostEqual(
            float(emitted[WEIGHTED_SPOT_HIGH_EMA]),
            expected,
            places=3,
        )

    def test_streaming_weighted_high_low_close_levels_match_batch(self) -> None:
        n = 350
        tl = _ramp_tl(n)
        origin = OPEN_TS
        close_ts = OPEN_TS + (n - 1) * STEP_SEC
        highs, lows = tl.high_low_rupees_series_on_grid(origin, close_ts, STEP_SEC)
        # Close series ≈ spot mid for ramp timeline (high/low from ticks).
        closes = [float(px) for px in highs]  # ramp_tl uses single px per bar
        batch_high = {
            period: _ema_series_from_prices(np.asarray(highs, dtype=float), period)
            for period in (20, 50, 200, 300)
        }
        batch_low = {
            period: _ema_series_from_prices(np.asarray(lows, dtype=float), period)
            for period in (20, 50, 200, 300)
        }
        batch_close = {
            period: _ema_series_from_prices(np.asarray(closes, dtype=float), period)
            for period in (20, 50, 200, 300)
        }

        sc = SpotControllers()
        for i in range(len(highs)):
            ts = OPEN_TS + i * STEP_SEC
            spot = 24000.0 + i
            sc.update(
                spot,
                ts=ts,
                grid_step_sec=STEP_SEC,
                index_tl=tl,
                grid_origin_ts=origin,
            )

        idx = len(highs) - 1
        expected_high = float(_weighted_blend(
            float(batch_high[20][idx]),
            float(batch_high[50][idx]),
            float(batch_high[200][idx]),
            float(batch_high[300][idx]),
        )) / 10.0
        expected_low = float(_weighted_blend(
            float(batch_low[20][idx]),
            float(batch_low[50][idx]),
            float(batch_low[200][idx]),
            float(batch_low[300][idx]),
        )) / 10.0
        expected_close = float(_weighted_blend(
            float(batch_close[20][idx]),
            float(batch_close[50][idx]),
            float(batch_close[200][idx]),
            float(batch_close[300][idx]),
        )) / 10.0
        emitted = emit_spot_hl_composite_registry_features(
            sc,
            ltp=100.0,
            active_features=frozenset({
                WEIGHTED_SPOT_HIGH_EMA,
                WEIGHTED_SPOT_LOW_EMA,
                WEIGHTED_SPOT_CLOSE_EMA,
            }),
        )
        self.assertAlmostEqual(float(emitted[WEIGHTED_SPOT_HIGH_EMA]), expected_high, places=3)
        self.assertAlmostEqual(float(emitted[WEIGHTED_SPOT_LOW_EMA]), expected_low, places=3)
        self.assertAlmostEqual(float(emitted[WEIGHTED_SPOT_CLOSE_EMA]), expected_close, places=3)

    def test_enrich_without_controllers_emits_null(self) -> None:
        out = enrich_spot_hl_composite_registry_features(
            {"ltp": 100.0},
            spot_controllers=None,
            active_features=frozenset({"spot_ema50_channel_width"}),
        )
        self.assertIsNone(out["spot_ema50_channel_width"])

    def test_registry_feature_sets_disjoint(self) -> None:
        self.assertEqual(len(SPOT_HL_CHANNEL_WIDTH_REGISTRY_FEATURES), 5)
        self.assertEqual(
            SPOT_HL_CHANNEL_WIDTH_REGISTRY_FEATURES,
            frozenset(f"spot_ema{p}_channel_width" for p in SPOT_HL_PERIODS),
        )
        self.assertEqual(len(SPOT_HL_WEIGHTED_COMPOSITE_REGISTRY_FEATURES), 3)
        self.assertEqual(
            SPOT_HL_WEIGHTED_COMPOSITE_REGISTRY_FEATURES,
            frozenset({WEIGHTED_SPOT_HIGH_EMA, WEIGHTED_SPOT_LOW_EMA, WEIGHTED_SPOT_CLOSE_EMA}),
        )
        overlap = SPOT_HL_RATIO_REGISTRY_FEATURES & SPOT_HL_COMPOSITE_REGISTRY_FEATURES
        self.assertEqual(len(overlap), 0)


if __name__ == "__main__":
    unittest.main()

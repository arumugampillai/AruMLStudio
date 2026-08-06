"""Tests for Sharp Momentum features."""

from __future__ import annotations

import unittest

from chain_replay_ml.dataset_builder.day_context import DayContext, SourceSpec
from chain_replay_ml.dataset_builder.rolling_controllers import (
    SpotControllers,
    weighted_spot_ema_level,
    weighted_spot_ema_ratio_from_values,
)
from chain_replay_ml.dataset_builder.sharp_momentum import (
    DECAY_AT_3S,
    SpotMomentumSnapshot,
    _apply_decay,
    _apply_spot_change,
    decay_factor,
    ensure_spot_momentum_cache,
    features_from_snapshot,
    spot_momentum_snapshot_at,
)
from chain_replay_ml.ticks import TickTimeline


def _spot_tl(prices: list[tuple[float, float]]) -> TickTimeline:
    tl = TickTimeline()
    for ts, rupees in prices:
        tl.append(ts, int(round(rupees * 100)))
    return tl


class TestSharpMomentumDecay(unittest.TestCase):
    def test_decay_factor_time_normalized(self) -> None:
        d3 = decay_factor(0.977, 3.0)
        d10 = decay_factor(0.977, 10.0)
        self.assertAlmostEqual(d3, 0.977, places=6)
        self.assertAlmostEqual(d10, 0.977 ** (10.0 / 3.0), places=6)

    def test_first_sample_zero(self) -> None:
        ctx = DayContext(
            source=SourceSpec("t", "2026-01-02", "NIFTY", "2026-01-08"),
            db_path="",
            expiry_norm="2026-01-08",
            open_ts=1000.0,
            close_ts=20000.0,
            expiry_ts=30000.0,
            index_tl=_spot_tl([(1000.0, 24000.0), (1003.0, 24002.0)]),
            strike_mapping={},
            feature_grid_step_sec=3,
        )
        from chain_replay_ml.dataset_builder import sharp_momentum as sm

        def _grid(_ctx, *, through_ts, step_sec, max_horizon_sec):
            return [1000.0]

        orig = sm._grid_timestamps_through
        sm._grid_timestamps_through = _grid
        try:
            ensure_spot_momentum_cache(ctx, through_ts=1000.0, step_sec=3, max_horizon_sec=0)
        finally:
            sm._grid_timestamps_through = orig

        snap = ctx.spot_momentum_by_ts[1000.0]
        self.assertEqual(snap.up_score["1m"], 0.0)
        self.assertEqual(snap.down_score["1m"], 0.0)

    def test_up_move_accumulates(self) -> None:
        ctx = DayContext(
            source=SourceSpec("t", "2026-01-02", "NIFTY", "2026-01-08"),
            db_path="",
            expiry_norm="2026-01-08",
            open_ts=1000.0,
            close_ts=20000.0,
            expiry_ts=30000.0,
            index_tl=_spot_tl([
                (1000.0, 24000.0),
                (1003.0, 24005.0),
            ]),
            strike_mapping={},
            feature_grid_step_sec=3,
        )
        # Monkey-patch grid to two points only.
        from chain_replay_ml.dataset_builder import sharp_momentum as sm

        def _grid(_ctx, *, through_ts, step_sec, max_horizon_sec):
            return [1000.0, 1003.0]

        orig = sm._grid_timestamps_through
        sm._grid_timestamps_through = _grid
        try:
            ensure_spot_momentum_cache(ctx, through_ts=1003.0, step_sec=3, max_horizon_sec=0)
        finally:
            sm._grid_timestamps_through = orig

        snap = ctx.spot_momentum_by_ts[1003.0]
        self.assertAlmostEqual(snap.up_score["1m"], 5.0, places=6)
        self.assertAlmostEqual(snap.up_count["1m"], 1.0, places=6)
        self.assertAlmostEqual(snap.down_score["1m"], 0.0, places=6)

    def test_features_from_snapshot(self) -> None:
        snap = SpotMomentumSnapshot()
        snap.up_score["5m"] = 10.0
        snap.up_count["5m"] = 2.0
        # Wave 4: Master emits canonical levels (packaging is Pipeline Owned).
        feats = features_from_snapshot(snap, 100.0)
        self.assertAlmostEqual(feats["spot_up_score_5m"], 10.0, places=6)
        self.assertAlmostEqual(feats["spot_up_sample_count_5m"], 2.0, places=6)
        self.assertNotIn("spot_up_score_5m_to_ltp_ratio", feats)
        self.assertNotIn("ltp_to_5m_spot_up_sample_count_ratio", feats)

    def test_weighted_ema_ratio(self) -> None:
        val = weighted_spot_ema_ratio_from_values(90.0, 80.0, 70.0, 60.0, 50.0)
        self.assertAlmostEqual(val, (90 * 4 + 80 * 3 + 70 * 2 + 60) / (10 * 50), places=6)

    def test_enrich_weighted_spot_ema_from_controllers(self) -> None:
        from chain_replay_ml.dataset_builder.sharp_momentum import enrich_sharp_momentum_features

        spot_ctrl = SpotControllers()
        for i in range(200):
            spot_ctrl.update(24000.0 + i, ts=1000.0 + i * 3.0)
        raw = enrich_sharp_momentum_features(
            {"ltp": 100.0},
            ts=1000.0 + 199 * 3.0,
            ctx=DayContext(
                source=SourceSpec("t", "2026-01-02", "NIFTY", "2026-01-08"),
                db_path="",
                expiry_norm="2026-01-08",
                open_ts=1000.0,
                close_ts=20000.0,
                expiry_ts=30000.0,
                index_tl=_spot_tl([(1000.0, 24000.0)]),
                strike_mapping={},
                feature_grid_step_sec=3,
            ),
            opt_state=None,
            option_timeline=None,
            open_ts=1000.0,
            close_ts=20000.0,
            active_features=frozenset(["weighted_spot_ema"]),
            spot_controllers=spot_ctrl,
        )
        val = raw.get("weighted_spot_ema")
        self.assertIsNotNone(val)
        self.assertAlmostEqual(val, weighted_spot_ema_level(spot_ctrl), places=9)

    def test_enrich_score_and_count_features(self) -> None:
        from chain_replay_ml.dataset_builder.day_context import DayContext, SourceSpec
        from chain_replay_ml.dataset_builder.sharp_momentum import (
            SpotMomentumSnapshot,
            enrich_sharp_momentum_features,
            ensure_spot_momentum_cache,
        )

        ctx = DayContext(
            source=SourceSpec("t", "2026-01-02", "NIFTY", "2026-01-08"),
            db_path="",
            expiry_norm="2026-01-08",
            open_ts=1000.0,
            close_ts=20000.0,
            expiry_ts=30000.0,
            index_tl=_spot_tl([(1000.0, 24000.0)]),
            strike_mapping={},
            feature_grid_step_sec=3,
        )
        from chain_replay_ml.dataset_builder import sharp_momentum as sm

        def _grid(_ctx, *, through_ts, step_sec, max_horizon_sec):
            return [1000.0]

        orig = sm._grid_timestamps_through
        sm._grid_timestamps_through = _grid
        try:
            ensure_spot_momentum_cache(ctx, through_ts=1000.0, step_sec=3, max_horizon_sec=0)
        finally:
            sm._grid_timestamps_through = orig

        raw = enrich_sharp_momentum_features(
            {"ltp": 100.0},
            ts=1000.0,
            ctx=ctx,
            opt_state=None,
            option_timeline=None,
            open_ts=1000.0,
            close_ts=20000.0,
            active_features=frozenset(["spot_up_score_1m", "spot_up_sample_count_1m"]),
        )
        self.assertIn("spot_up_score_1m", raw)
        self.assertIn("spot_up_sample_count_1m", raw)
        self.assertNotIn("spot_up_score_1m_to_ltp_ratio", raw)
        self.assertNotIn("ltp_to_1m_spot_up_sample_count_ratio", raw)
        snap = SpotMomentumSnapshot()
        snap.up_score["1m"] = 10.0
        _apply_decay(snap, 3.0)
        _apply_spot_change(snap, 0.0)
        self.assertAlmostEqual(snap.up_score["1m"], 10.0 * DECAY_AT_3S["1m"], places=6)


class TestSpotMomentumSnapshotLookup(unittest.TestCase):
    def test_snapshot_lookup_uses_prior_grid_point(self) -> None:
        ctx = DayContext(
            source=SourceSpec("t", "2026-01-02", "NIFTY", "2026-01-08"),
            db_path="",
            expiry_norm="2026-01-08",
            open_ts=1000.0,
            close_ts=2000.0,
            expiry_ts=30000.0,
            index_tl=_spot_tl([(1000.0, 24000.0)]),
            strike_mapping={},
            feature_grid_step_sec=3,
        )
        snap = SpotMomentumSnapshot()
        snap.up_score["1m"] = 42.0
        ctx.spot_momentum_by_ts = {1000.0: snap, 1003.0: snap}
        ctx._spot_momentum_grid_ts = [1000.0, 1003.0]
        got = spot_momentum_snapshot_at(ctx, 1002.5)
        self.assertAlmostEqual(got.up_score["1m"], 42.0)


if __name__ == "__main__":
    unittest.main()

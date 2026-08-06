"""Tests for token LTP rolling controllers (EMA, STD20)."""

from __future__ import annotations

import math
import unittest

import numpy as np

from chain_replay_ml.dataset_builder.chain_maps import ChainMaps
from chain_replay_ml.dataset_builder.controller_warmup_regression import CONTROLLER_WARMUP_SPEC
from chain_replay_ml.dataset_builder.extended_features import (
    OptionFeatureState,
    enrich_with_chain_maps,
    reset_option_rolling_state,
)
from chain_replay_ml.dataset_builder.gap_policy_instrumentation import row_gap_exceeds
from chain_replay_ml.dataset_builder.rolling_controllers import (
    CONTROLLER_OWNED_READINESS_FEATURES,
    ControllerSample,
    EmaController,
    IvHistoryController,
    IvSessionRankController,
    IvZscoreWindowController,
    RollController,
    RvController,
    SpotControllers,
    StdController,
    TokenControllers,
    DGT_OWNED_FEATURES,
    DgtController,
    assert_iv_history_reset_complete,
    assert_monotonic_controller_ts,
    assert_roll_controller_reset_complete,
    assert_rolling_controller_reset_complete,
    emit_controller_derived_quotient,
    emit_controller_ratio,
    emit_controller_value,
    emit_dgt_features,
    emit_roll_features,
    guard_controller_derived_rv_features,
    opt_rv_ratio,
    update_token_dgt_controller,
    update_token_iv_controllers,
    update_token_ltp_controllers,
    update_token_roll_controller,
    update_token_rv_controllers,
    weighted_ltp_ema_level,
    weighted_ltp_ema_ratio,
    weighted_spot_ema_level,
    weighted_spot_ema_level_from_values,
    weighted_spot_ema_ratio,
    weighted_spot_ema_ratio_from_values,
    resolve_weighted_spot_ema_to_ltp_ratio,
    build_spot_rv_cache,
)
from chain_replay_ml.constants import (
    DEFAULT_IV_THRESHOLD_PCT,
    DEFAULT_MAX_ROLL_AGE_MIN,
    DEFAULT_SPOT_THRESHOLD_PCT,
    RISK_FREE_RATE,
)
from chain_replay_ml.reanchor import iv_drift_from_roll_pct
from chain_replay_ml.ticks import TickTimeline

# MANDATORY regression — proves no session-open grid history leaks into token EMA.
REGRESSION_LTPS = [
    532.55, 529.50, 530.30, 534.35, 532.60, 530.05, 528.80, 528.75, 524.40,
]
REGRESSION_EMA9 = 529.452170
REGRESSION_STD20_LTPS = [
    532.55, 529.50, 530.30, 534.35, 532.60, 530.05, 528.80, 528.75, 524.40,
    520.00, 522.10, 525.30, 527.00, 529.00, 531.00, 533.00, 535.00, 537.00, 539.00, 541.00,
]
GAP_MAX_SEC = 10.0
STEP_SEC = 3.0


def _sim_row_ts(
    i: int,
    *,
    open_ts: float,
    last_ts: float | None,
    gap_after_row: int | None = None,
) -> float:
    """Monotonic row timestamps; gap row jumps forward, later rows continue from gap."""
    if i == 0:
        return open_ts
    assert last_ts is not None
    if gap_after_row is not None and i == gap_after_row:
        return last_ts + GAP_MAX_SEC + 1.0
    return last_ts + STEP_SEC


def _manual_ema9(prices: list[float]) -> float:
    alpha = 2.0 / 10.0
    ema = prices[0]
    for price in prices[1:]:
        ema = price * alpha + ema * (1.0 - alpha)
    return ema


def _manual_ema20(prices: list[float]) -> float:
    alpha = 2.0 / 21.0
    ema = prices[0]
    for price in prices[1:]:
        ema = price * alpha + ema * (1.0 - alpha)
    return ema


def _manual_ema50(prices: list[float]) -> float:
    alpha = 2.0 / 51.0
    ema = prices[0]
    for price in prices[1:]:
        ema = price * alpha + ema * (1.0 - alpha)
    return ema


def _manual_ema100(prices: list[float]) -> float:
    alpha = 2.0 / 101.0
    ema = prices[0]
    for price in prices[1:]:
        ema = price * alpha + ema * (1.0 - alpha)
    return ema


def _manual_ema200(prices: list[float]) -> float:
    alpha = 2.0 / 201.0
    ema = prices[0]
    for price in prices[1:]:
        ema = price * alpha + ema * (1.0 - alpha)
    return ema


def _emit_ltp_ema9_levels(
    prices: list[float],
    *,
    gap_after_row: int | None = None,
    spot: float = 25000.0,
) -> list[dict[str, float | None]]:
    """Simulate per-row controller updates + ratio emit (no session grid history)."""
    opt_state = OptionFeatureState()
    open_ts = 1_700_000_000.0
    out_rows: list[dict[str, float | None]] = []
    last_ts: float | None = None

    for i, ltp in enumerate(prices):
        ts = open_ts + i * STEP_SEC
        if gap_after_row is not None and i == gap_after_row:
            ts = last_ts + GAP_MAX_SEC + 1.0 if last_ts is not None else ts
        if last_ts is not None and row_gap_exceeds(ts, last_ts, GAP_MAX_SEC):
            opt_state.controllers.reset_all(ts=ts)
        update_token_ltp_controllers(opt_state.controllers, ltp, ts=ts)
        raw = {"ltp": ltp, "spot": spot}
        enriched = enrich_with_chain_maps(
            raw,
            ts=ts,
            chain_maps=ChainMaps(),
            strike_mapping={},
            index_tl=TickTimeline(),
            atm_strike=25000,
            expiry_ts=open_ts + 86400.0,
            opt_state=opt_state,
            option_timeline=TickTimeline(),
            open_ts=open_ts,
            close_ts=open_ts + 3600.0 * 6,
            active_features=frozenset({
                "ltp_ema9",
                "weighted_ltp_ema",
            }),
            feature_grid_step_sec=STEP_SEC,
        )
        out_rows.append({
            "ltp_ema9": enriched.get("ltp_ema9"),
        })
        last_ts = ts
    return out_rows


def _emit_ltp_std20_levels(
    prices: list[float],
    *,
    gap_after_row: int | None = None,
    spot: float = 25000.0,
) -> list[dict[str, float | None]]:
    """Simulate per-row controller updates + std20 ratio emit (no session grid history)."""
    opt_state = OptionFeatureState()
    open_ts = 1_700_000_000.0
    out_rows: list[dict[str, float | None]] = []
    last_ts: float | None = None

    for i, ltp in enumerate(prices):
        ts = open_ts + i * STEP_SEC
        if gap_after_row is not None and i == gap_after_row:
            ts = last_ts + GAP_MAX_SEC + 1.0 if last_ts is not None else ts
        if last_ts is not None and row_gap_exceeds(ts, last_ts, GAP_MAX_SEC):
            opt_state.controllers.reset_all(ts=ts)
        update_token_ltp_controllers(opt_state.controllers, ltp, ts=ts)
        raw = {"ltp": ltp, "spot": spot}
        enriched = enrich_with_chain_maps(
            raw,
            ts=ts,
            chain_maps=ChainMaps(),
            strike_mapping={},
            index_tl=TickTimeline(),
            atm_strike=25000,
            expiry_ts=open_ts + 86400.0,
            opt_state=opt_state,
            option_timeline=TickTimeline(),
            open_ts=open_ts,
            close_ts=open_ts + 3600.0 * 6,
            active_features=frozenset({
                "ltp_std20",
            }),
            feature_grid_step_sec=STEP_SEC,
        )
        out_rows.append({
            "ltp_std20": enriched.get("ltp_std20"),
        })
        last_ts = ts
    return out_rows


class EmaControllerTests(unittest.TestCase):
    def test_rows_1_to_8_null_for_ema9_level(self) -> None:
        rows = _emit_ltp_ema9_levels(REGRESSION_LTPS)
        for row in rows[:8]:
            self.assertIsNone(row["ltp_ema9"])

    def test_row_9_first_valid_ema9(self) -> None:
        rows = _emit_ltp_ema9_levels(REGRESSION_LTPS)
        row9 = rows[8]
        expected_ema = _manual_ema9(REGRESSION_LTPS)
        self.assertIsNotNone(row9["ltp_ema9"])
        self.assertAlmostEqual(row9["ltp_ema9"], expected_ema, places=9)

    def test_gap_resets_warmup(self) -> None:
        prices = [100.0] * 9 + [200.0]
        rows = _emit_ltp_ema9_levels(prices, gap_after_row=9)
        self.assertAlmostEqual(rows[8]["ltp_ema9"], 100.0, places=9)
        self.assertIsNone(rows[9]["ltp_ema9"])

    def test_permanent_regression_no_hidden_history(self) -> None:
        """Mandatory: N exported rows only — manual EMA must match builder exactly."""
        rows = _emit_ltp_ema9_levels(REGRESSION_LTPS)
        manual_ema = _manual_ema9(REGRESSION_LTPS)
        built = rows[8]["ltp_ema9"]
        self.assertAlmostEqual(built, manual_ema, places=9)
        self.assertAlmostEqual(built, REGRESSION_EMA9, places=6)
        ctrl = EmaController(9)
        for price in REGRESSION_LTPS:
            ctrl.update(price)
        self.assertAlmostEqual(ctrl.value(), manual_ema, places=9)
        self.assertAlmostEqual(manual_ema, REGRESSION_EMA9, places=6)

    def test_ema_controller_unit(self) -> None:
        ctrl = EmaController(9)
        for price in REGRESSION_LTPS:
            ctrl.update(price)
        self.assertTrue(ctrl.ready())
        self.assertAlmostEqual(ctrl.value(), _manual_ema9(REGRESSION_LTPS), places=9)

    def test_emit_controller_ratio_not_ready(self) -> None:
        ctrl = EmaController(9)
        ctrl.update(100.0)
        self.assertIsNone(emit_controller_ratio(ctrl, 100.0))

    def test_controller_debug_invariants(self) -> None:
        ctrl = EmaController(9)
        self.assertEqual(ctrl.samples, 0)
        self.assertFalse(ctrl.ready())
        for price in REGRESSION_LTPS[:3]:
            ctrl.update(price, ts=100.0)
        self.assertEqual(ctrl.samples, 3)
        self.assertFalse(ctrl.ready())
        self.assertGreaterEqual(ctrl.last_update_ts or 0, ctrl.last_reset_ts or 0)

    def test_weighted_ltp_ema_warmup_200(self) -> None:
        ctrl = TokenControllers()
        for i in range(199):
            update_token_ltp_controllers(ctrl, 100.0 + i * 0.1)
        self.assertIsNone(weighted_ltp_ema_level(ctrl))
        self.assertIsNone(weighted_ltp_ema_ratio(ctrl, 100.0))
        update_token_ltp_controllers(ctrl, 120.0)
        level = weighted_ltp_ema_level(ctrl)
        val = weighted_ltp_ema_ratio(ctrl, 120.0)
        self.assertIsNotNone(level)
        self.assertIsNotNone(val)
        self.assertGreater(level, 0.0)
        self.assertAlmostEqual(val, float(level) / 120.0, places=9)

    def test_weighted_spot_ema_formula_matches_legacy_blend(self) -> None:
        level = weighted_spot_ema_level_from_values(90.0, 80.0, 70.0, 60.0)
        self.assertAlmostEqual(level, (90.0 * 4 + 80.0 * 3 + 70.0 * 2 + 60.0) / 10.0)
        val = weighted_spot_ema_ratio_from_values(90.0, 80.0, 70.0, 60.0, 50.0)
        self.assertAlmostEqual(val, (90.0 * 4 + 80.0 * 3 + 70.0 * 2 + 60.0) / (10.0 * 50.0))

    def test_weighted_spot_ema_warmup_200(self) -> None:
        spot_ctrl = SpotControllers()
        spots = [24000.0 + i * 0.5 for i in range(199)]
        for i, spot in enumerate(spots):
            spot_ctrl.update(spot, ts=1000.0 + i * STEP_SEC)
        self.assertIsNone(weighted_spot_ema_level(spot_ctrl))
        self.assertIsNone(weighted_spot_ema_ratio(spot_ctrl, 100.0))
        spot_ctrl.update(24100.0, ts=1000.0 + 199 * STEP_SEC)
        level = weighted_spot_ema_level(spot_ctrl)
        val = weighted_spot_ema_ratio(spot_ctrl, 100.0)
        self.assertIsNotNone(level)
        self.assertIsNotNone(val)
        self.assertGreater(level, 0.0)
        self.assertAlmostEqual(val, float(level) / 100.0, places=9)

    def test_weighted_spot_ema_parallel_cache_matches_controller(self) -> None:
        open_ts = 1_700_000_000.0
        spots = [24000.0 + i for i in range(220)]
        timestamps = [open_ts + i * STEP_SEC for i in range(220)]
        tl = TickTimeline()
        for ts, spot in zip(timestamps, spots):
            tl.append(ts, int(round(spot * 100)))
        cache = build_spot_rv_cache(tl, timestamps)
        spot_ctrl = SpotControllers()
        for i, spot in enumerate(spots):
            spot_ctrl.update(spot, ts=timestamps[i])
            if i < 199:
                continue
            ltp = 100.0 + i
            ctrl_val = weighted_spot_ema_ratio(spot_ctrl, ltp)
            cache_val = resolve_weighted_spot_ema_to_ltp_ratio(
                {},
                ltp=ltp,
                spot_rv_cache=cache,
                ts=timestamps[i],
            )
            self.assertAlmostEqual(ctrl_val, cache_val, places=9)

    def test_controller_owned_readiness_includes_weighted_spot_ema(self) -> None:
        self.assertIn("weighted_spot_ema", CONTROLLER_OWNED_READINESS_FEATURES)
        self.assertIn("weighted_ltp_ema", CONTROLLER_OWNED_READINESS_FEATURES)
        self.assertIn("weighted_spot_high_ema", CONTROLLER_OWNED_READINESS_FEATURES)
        self.assertIn("weighted_spot_low_ema", CONTROLLER_OWNED_READINESS_FEATURES)
        self.assertIn("weighted_spot_close_ema", CONTROLLER_OWNED_READINESS_FEATURES)


class StdControllerTests(unittest.TestCase):
    def test_rows_1_to_19_null_for_std20_level(self) -> None:
        rows = _emit_ltp_std20_levels(REGRESSION_STD20_LTPS)
        for row in rows[:19]:
            self.assertIsNone(row["ltp_std20"])

    def test_row_20_first_valid_std20(self) -> None:
        rows = _emit_ltp_std20_levels(REGRESSION_STD20_LTPS)
        row20 = rows[19]
        expected_std = float(np.std(REGRESSION_STD20_LTPS, ddof=0))
        self.assertIsNotNone(row20["ltp_std20"])
        self.assertAlmostEqual(row20["ltp_std20"], expected_std, places=9)

    def test_gap_resets_warmup(self) -> None:
        prices = [100.0] * 20 + [200.0]
        rows = _emit_ltp_std20_levels(prices, gap_after_row=20)
        self.assertAlmostEqual(rows[19]["ltp_std20"], 0.0, places=9)
        self.assertIsNone(rows[20]["ltp_std20"])

    def test_permanent_regression_no_hidden_history(self) -> None:
        """Mandatory: N exported rows only — manual std must match builder exactly."""
        rows = _emit_ltp_std20_levels(REGRESSION_STD20_LTPS)
        expected_std = float(np.std(REGRESSION_STD20_LTPS, ddof=0))
        built = rows[19]["ltp_std20"]
        self.assertAlmostEqual(built, expected_std, places=9)
        ctrl = StdController()
        for price in REGRESSION_STD20_LTPS:
            ctrl.update(price)
        self.assertAlmostEqual(ctrl.value(), expected_std, places=9)

    def test_std_controller_unit(self) -> None:
        ctrl = StdController()
        for price in REGRESSION_STD20_LTPS:
            ctrl.update(price)
        self.assertTrue(ctrl.ready())
        self.assertAlmostEqual(
            ctrl.value(),
            float(np.std(REGRESSION_STD20_LTPS, ddof=0)),
            places=9,
        )
        self.assertEqual(ctrl.reset_feature_label(), "STD20")

    def test_emit_controller_ratio_not_ready(self) -> None:
        ctrl = StdController()
        ctrl.update(100.0)
        self.assertIsNone(emit_controller_ratio(ctrl, 100.0))


def _manual_rv(prices: list[float], period: int) -> float | None:
    if len(prices) < period + 1:
        return None
    returns = [
        (prices[i] - prices[i - 1]) / prices[i - 1] * 100.0
        for i in range(1, len(prices))
        if prices[i - 1] > 0
    ]
    if len(returns) < period:
        return None
    window = returns[-period:]
    return float(np.std(window, ddof=0))


def _emit_opt_rv(
    prices: list[float],
    *,
    gap_after_row: int | None = None,
    spot: float = 25000.0,
) -> list[dict[str, float | None]]:
    opt_state = OptionFeatureState()
    open_ts = 1_700_000_000.0
    out_rows: list[dict[str, float | None]] = []
    last_ts: float | None = None

    for i, ltp in enumerate(prices):
        ts = open_ts + i * STEP_SEC
        if gap_after_row is not None and i == gap_after_row:
            ts = last_ts + GAP_MAX_SEC + 1.0 if last_ts is not None else ts
        if last_ts is not None and row_gap_exceeds(ts, last_ts, GAP_MAX_SEC):
            opt_state.controllers.reset_all(ts=ts)
        update_token_rv_controllers(opt_state.controllers, ltp, ts=ts)
        raw = {"ltp": ltp, "spot": spot}
        enriched = enrich_with_chain_maps(
            raw,
            ts=ts,
            chain_maps=ChainMaps(),
            strike_mapping={},
            index_tl=TickTimeline(),
            atm_strike=25000,
            expiry_ts=open_ts + 86400.0,
            opt_state=opt_state,
            option_timeline=TickTimeline(),
            open_ts=open_ts,
            close_ts=open_ts + 3600.0 * 6,
            active_features=frozenset({
                "opt_rv_5m",
                "opt_rv_10m",
                "opt_rv_ratio",
            }),
            feature_grid_step_sec=STEP_SEC,
        )
        out_rows.append({
            "opt_rv_5m": enriched.get("opt_rv_5m"),
            "opt_rv_10m": enriched.get("opt_rv_10m"),
            "opt_rv_ratio": enriched.get("opt_rv_ratio"),
        })
        last_ts = ts
    return out_rows


def _emit_spot_rv(
    spot_prices: list[float],
    *,
    gap_after_row: int | None = None,
    reset_spot_on_gap: bool = False,
) -> list[dict[str, float | None]]:
    spot_ctrl = SpotControllers()
    open_ts = 1_700_000_000.0
    out_rows: list[dict[str, float | None]] = []
    last_ts: float | None = None

    for i, spot in enumerate(spot_prices):
        ts = open_ts + i * STEP_SEC
        if gap_after_row is not None and i == gap_after_row:
            ts = last_ts + GAP_MAX_SEC + 1.0 if last_ts is not None else ts
        if last_ts is not None and row_gap_exceeds(ts, last_ts, GAP_MAX_SEC) and reset_spot_on_gap:
            spot_ctrl.reset_all(ts=ts)
        spot_ctrl.update(spot, ts=ts)
        raw = {"ltp": 100.0, "spot": spot}
        enriched = enrich_with_chain_maps(
            raw,
            ts=ts,
            chain_maps=ChainMaps(),
            strike_mapping={},
            index_tl=TickTimeline(),
            atm_strike=25000,
            expiry_ts=open_ts + 86400.0,
            option_timeline=TickTimeline(),
            open_ts=open_ts,
            close_ts=open_ts + 3600.0 * 6,
            active_features=frozenset({
                "spot_rv_5m",
                "spot_rv_10m",
                "spot_rv_ratio",
            }),
            feature_grid_step_sec=STEP_SEC,
            spot_controllers=spot_ctrl,
        )
        out_rows.append({
            "spot_rv_5m": enriched.get("spot_rv_5m"),
            "spot_rv_10m": enriched.get("spot_rv_10m"),
            "spot_rv_ratio": enriched.get("spot_rv_ratio"),
        })
        last_ts = ts
    return out_rows


REGRESSION_RV5M_LTPS = [25000.0 + i * 2.5 + (i % 3) * 0.7 for i in range(35)]
REGRESSION_RV10M_LTPS = [25000.0 + i * 1.8 + (i % 5) * 0.5 for i in range(65)]


class RvControllerTests(unittest.TestCase):
    def test_rows_1_to_30_null_for_opt_rv_5m(self) -> None:
        rows = _emit_opt_rv(REGRESSION_RV5M_LTPS)
        for row in rows[:30]:
            self.assertIsNone(row["opt_rv_5m"])
            self.assertIsNone(row["opt_rv_ratio"])

    def test_row_31_first_valid_opt_rv_5m(self) -> None:
        rows = _emit_opt_rv(REGRESSION_RV5M_LTPS)
        row31 = rows[30]
        expected = _manual_rv(REGRESSION_RV5M_LTPS[:31], 30)
        self.assertIsNotNone(row31["opt_rv_5m"])
        self.assertAlmostEqual(row31["opt_rv_5m"], expected, places=9)

    def test_opt_rv_ratio_not_emitted_when_rv_ready(self) -> None:
        # Wave 2: opt_rv_ratio is Interaction / Pipeline Owned — not emitted by controllers.
        rows = _emit_opt_rv(REGRESSION_RV10M_LTPS)
        for row in rows:
            self.assertIsNone(row["opt_rv_ratio"])
        row61 = rows[60]
        self.assertIsNotNone(row61["opt_rv_5m"])
        self.assertIsNotNone(row61["opt_rv_10m"])

    def test_rv_family_null_together_during_partial_warmup(self) -> None:
        """Ratio must stay NULL whenever rv_10m is NULL (no rv5/(rv5+eps) orphan)."""
        opt_rows = _emit_opt_rv(REGRESSION_RV10M_LTPS)
        for row in opt_rows[30:60]:
            if row["opt_rv_10m"] is None:
                self.assertIsNone(row["opt_rv_ratio"])
                if row["opt_rv_5m"] is not None:
                    pseudo = row["opt_rv_5m"] / (row["opt_rv_5m"] + 1e-9)
                    self.assertNotAlmostEqual(row["opt_rv_ratio"] or -1.0, pseudo, places=6)

        spot_rows = _emit_spot_rv(REGRESSION_RV10M_LTPS)
        for row in spot_rows[30:60]:
            if row["spot_rv_10m"] is None:
                self.assertIsNone(row["spot_rv_ratio"])

    def test_guard_controller_derived_rv_features(self) -> None:
        out = {"opt_rv_5m": 0.35, "opt_rv_10m": None, "opt_rv_ratio": 0.999999997}
        guard_controller_derived_rv_features(out)
        self.assertIsNone(out["opt_rv_ratio"])

    def test_emit_controller_derived_quotient_requires_both_values(self) -> None:
        tc = TokenControllers()
        for price in REGRESSION_RV5M_LTPS[:31]:
            update_token_rv_controllers(tc, price)
        self.assertIsNone(emit_controller_derived_quotient(tc.rv5m, tc.rv10m))

    def test_token_gap_resets_rv_warmup(self) -> None:
        prices = [100.0 + i * 0.5 for i in range(31)] + [200.0]
        rows = _emit_opt_rv(prices, gap_after_row=31)
        self.assertIsNotNone(rows[30]["opt_rv_5m"])
        self.assertIsNone(rows[31]["opt_rv_5m"])

    def test_rv_controller_unit(self) -> None:
        ctrl = RvController(30)
        prices = REGRESSION_RV5M_LTPS
        for price in prices:
            ctrl.update(price)
        self.assertTrue(ctrl.ready())
        self.assertAlmostEqual(ctrl.value(), _manual_rv(prices, 30), places=9)
        self.assertEqual(ctrl.reset_feature_label(), "RV5M")

    def test_opt_rv_ratio_unit(self) -> None:
        tc = TokenControllers()
        for price in REGRESSION_RV10M_LTPS:
            update_token_rv_controllers(tc, price)
        ratio = opt_rv_ratio(tc)
        rv5 = tc.rv5m.value()
        rv10 = tc.rv10m.value()
        self.assertIsNotNone(ratio)
        assert rv5 is not None and rv10 is not None and ratio is not None
        self.assertAlmostEqual(ratio, rv5 / (rv10 + 1e-9), places=9)

    def test_spot_rv_warmup_periods_from_registry(self) -> None:
        spot_ctrl = SpotControllers()
        self.assertEqual(spot_ctrl.rv5m.warmup_period, 30)
        self.assertEqual(spot_ctrl.rv10m.warmup_period, 60)

    def test_spot_rv_dedupes_one_sample_per_timestamp(self) -> None:
        """Serial build calls update once per token row — same ts must not advance warmup."""
        spot_ctrl = SpotControllers()
        ts0 = 1_700_000_000.0
        for _ in range(100):
            spot_ctrl.update(25000.0, ts=ts0)
        self.assertFalse(spot_ctrl.rv5m.ready())
        self.assertEqual(spot_ctrl.rv5m.samples, 0)
        spot_ctrl.update(25001.0, ts=ts0 + STEP_SEC)
        self.assertEqual(spot_ctrl.rv5m.samples, 1)

    def test_rows_1_to_30_null_for_spot_rv_5m(self) -> None:
        rows = _emit_spot_rv(REGRESSION_RV5M_LTPS)
        for row in rows[:30]:
            self.assertIsNone(row["spot_rv_5m"])
            self.assertIsNone(row["spot_rv_ratio"])

    def test_row_31_first_valid_spot_rv_5m(self) -> None:
        rows = _emit_spot_rv(REGRESSION_RV5M_LTPS)
        row31 = rows[30]
        expected = _manual_rv(REGRESSION_RV5M_LTPS[:31], 30)
        self.assertIsNotNone(row31["spot_rv_5m"])
        self.assertAlmostEqual(row31["spot_rv_5m"], expected, places=9)

    def test_spot_rv_interleaved_tokens_do_not_reset(self) -> None:
        """Spot controllers continue across token switches (session-wide stream)."""
        spot_ctrl = SpotControllers()
        spots = [25000.0 + i * 1.0 for i in range(35)]
        open_ts = 1_700_000_000.0
        token_a_rows: list[float | None] = []
        token_b_rows: list[float | None] = []

        for i, spot in enumerate(spots):
            ts = open_ts + i * STEP_SEC
            spot_ctrl.update(spot, ts=ts)
            raw = {"ltp": 100.0 + i, "spot": spot}
            enriched = enrich_with_chain_maps(
                raw,
                ts=ts,
                chain_maps=ChainMaps(),
                strike_mapping={},
                index_tl=TickTimeline(),
                atm_strike=25000,
                expiry_ts=open_ts + 86400.0,
                option_timeline=TickTimeline(),
                open_ts=open_ts,
                close_ts=open_ts + 3600.0 * 6,
                active_features=frozenset({"spot_rv_5m"}),
                feature_grid_step_sec=STEP_SEC,
                spot_controllers=spot_ctrl,
            )
            if i % 2 == 0:
                token_a_rows.append(enriched.get("spot_rv_5m"))
            else:
                token_b_rows.append(enriched.get("spot_rv_5m"))

        # Recompute spot-only stream — values at even/odd indices must match monotonic session.
        expected_rows = _emit_spot_rv(spots)
        for j, val in enumerate(token_a_rows):
            idx = j * 2
            if idx < len(expected_rows):
                self.assertEqual(val, expected_rows[idx]["spot_rv_5m"])

    def test_token_gap_does_not_reset_spot_rv(self) -> None:
        spots = [25000.0 + i * 1.0 for i in range(31)] + [25100.0]
        no_reset = _emit_spot_rv(spots, gap_after_row=31, reset_spot_on_gap=False)
        with_reset = _emit_spot_rv(spots, gap_after_row=31, reset_spot_on_gap=True)
        self.assertIsNotNone(no_reset[30]["spot_rv_5m"])
        self.assertIsNotNone(no_reset[31]["spot_rv_5m"])
        self.assertIsNone(with_reset[31]["spot_rv_5m"])


REGRESSION_SPOTS_EMA9 = [
    25032.55, 25029.50, 25030.30, 25034.35, 25032.60, 25030.05, 25028.80, 25028.75, 25024.40,
    25020.00,
]


def _emit_spot_ema9(
    spots: list[float],
    *,
    ltp: float = 100.0,
    gap_after_row: int | None = None,
    reset_spot_on_gap: bool = False,
) -> list[dict[str, float | None]]:
    spot_ctrl = SpotControllers()
    open_ts = 1_700_000_000.0
    out_rows: list[dict[str, float | None]] = []
    last_ts: float | None = None

    for i, spot in enumerate(spots):
        ts = open_ts + i * STEP_SEC
        if gap_after_row is not None and i == gap_after_row:
            ts = last_ts + GAP_MAX_SEC + 1.0 if last_ts is not None else ts
        if last_ts is not None and row_gap_exceeds(ts, last_ts, GAP_MAX_SEC) and reset_spot_on_gap:
            spot_ctrl.reset_all(ts=ts)
        spot_ctrl.update(spot, ts=ts)
        raw = {"ltp": ltp, "spot": spot}
        enriched = enrich_with_chain_maps(
            raw,
            ts=ts,
            chain_maps=ChainMaps(),
            strike_mapping={},
            index_tl=TickTimeline(),
            atm_strike=25000,
            expiry_ts=open_ts + 86400.0,
            option_timeline=TickTimeline(),
            open_ts=open_ts,
            close_ts=open_ts + 3600.0 * 6,
            active_features=frozenset({"spot_ema9"}),
            feature_grid_step_sec=STEP_SEC,
            spot_controllers=spot_ctrl,
        )
        out_rows.append({
            "spot_ema9": enriched.get("spot_ema9"),
        })
        last_ts = ts
    return out_rows


class SpotEma9ControllerTests(unittest.TestCase):
    def test_rows_1_to_8_null_for_spot_ema9_level(self) -> None:
        rows = _emit_spot_ema9(REGRESSION_SPOTS_EMA9)
        for row in rows[:8]:
            self.assertIsNone(row["spot_ema9"])

    def test_row_9_first_valid_spot_ema9(self) -> None:
        rows = _emit_spot_ema9(REGRESSION_SPOTS_EMA9)
        row9 = rows[8]
        expected_ema = _manual_ema9(REGRESSION_SPOTS_EMA9[:9])
        self.assertIsNotNone(row9["spot_ema9"])
        self.assertAlmostEqual(row9["spot_ema9"], expected_ema, places=9)

    def test_spot_ema9_dedupes_one_sample_per_timestamp(self) -> None:
        spot_ctrl = SpotControllers()
        ts0 = 1_700_000_000.0
        for _ in range(100):
            spot_ctrl.update(25000.0, ts=ts0)
        self.assertFalse(spot_ctrl.ema9.ready())
        self.assertEqual(spot_ctrl.ema9.samples, 1)
        spot_ctrl.update(25001.0, ts=ts0 + STEP_SEC)
        self.assertEqual(spot_ctrl.ema9.samples, 2)

    def test_token_gap_does_not_reset_spot_ema9(self) -> None:
        spots = [25000.0 + i * 1.0 for i in range(10)]
        no_reset = _emit_spot_ema9(spots, gap_after_row=9, reset_spot_on_gap=False)
        with_reset = _emit_spot_ema9(spots, gap_after_row=9, reset_spot_on_gap=True)
        self.assertIsNotNone(no_reset[8]["spot_ema9"])
        self.assertIsNotNone(no_reset[9]["spot_ema9"])
        self.assertIsNone(with_reset[9]["spot_ema9"])

    def test_controller_owned_readiness_includes_spot_ema9(self) -> None:
        from chain_replay_ml.dataset_builder.rolling_controllers import (
            CONTROLLER_OWNED_READINESS_FEATURES,
        )

        self.assertIn("spot_ema9", CONTROLLER_OWNED_READINESS_FEATURES)


REGRESSION_SPOTS_EMA20 = REGRESSION_SPOTS_EMA9 + [
    25018.50, 25016.20, 25015.80, 25017.10, 25019.30,
    25021.00, 25022.50, 25023.10, 25024.00, 25025.50,
    25026.00,
]


def _emit_spot_ema20(
    spots: list[float],
    *,
    ltp: float = 100.0,
    gap_after_row: int | None = None,
    reset_spot_on_gap: bool = False,
) -> list[dict[str, float | None]]:
    spot_ctrl = SpotControllers()
    open_ts = 1_700_000_000.0
    out_rows: list[dict[str, float | None]] = []
    last_ts: float | None = None

    for i, spot in enumerate(spots):
        ts = open_ts + i * STEP_SEC
        if gap_after_row is not None and i == gap_after_row:
            ts = last_ts + GAP_MAX_SEC + 1.0 if last_ts is not None else ts
        if last_ts is not None and row_gap_exceeds(ts, last_ts, GAP_MAX_SEC) and reset_spot_on_gap:
            spot_ctrl.reset_all(ts=ts)
        spot_ctrl.update(spot, ts=ts)
        raw = {"ltp": ltp, "spot": spot}
        enriched = enrich_with_chain_maps(
            raw,
            ts=ts,
            chain_maps=ChainMaps(),
            strike_mapping={},
            index_tl=TickTimeline(),
            atm_strike=25000,
            expiry_ts=open_ts + 86400.0,
            option_timeline=TickTimeline(),
            open_ts=open_ts,
            close_ts=open_ts + 3600.0 * 6,
            active_features=frozenset({"spot_ema20"}),
            feature_grid_step_sec=STEP_SEC,
            spot_controllers=spot_ctrl,
        )
        out_rows.append({
            "spot_ema20": enriched.get("spot_ema20"),
        })
        last_ts = ts
    return out_rows


class SpotEma20ControllerTests(unittest.TestCase):
    def test_rows_1_to_19_null_for_spot_ema20_level(self) -> None:
        rows = _emit_spot_ema20(REGRESSION_SPOTS_EMA20)
        for row in rows[:19]:
            self.assertIsNone(row["spot_ema20"])

    def test_row_20_first_valid_spot_ema20(self) -> None:
        rows = _emit_spot_ema20(REGRESSION_SPOTS_EMA20)
        row20 = rows[19]
        expected_ema = _manual_ema20(REGRESSION_SPOTS_EMA20[:20])
        self.assertIsNotNone(row20["spot_ema20"])
        self.assertAlmostEqual(row20["spot_ema20"], expected_ema, places=9)

    def test_spot_ema20_dedupes_one_sample_per_timestamp(self) -> None:
        spot_ctrl = SpotControllers()
        ts0 = 1_700_000_000.0
        for _ in range(100):
            spot_ctrl.update(25000.0, ts=ts0)
        self.assertFalse(spot_ctrl.ema20.ready())
        self.assertEqual(spot_ctrl.ema20.samples, 1)
        spot_ctrl.update(25001.0, ts=ts0 + STEP_SEC)
        self.assertEqual(spot_ctrl.ema20.samples, 2)

    def test_token_gap_does_not_reset_spot_ema20(self) -> None:
        spots = [25000.0 + i * 1.0 for i in range(21)]
        no_reset = _emit_spot_ema20(spots, gap_after_row=20, reset_spot_on_gap=False)
        with_reset = _emit_spot_ema20(spots, gap_after_row=20, reset_spot_on_gap=True)
        self.assertIsNotNone(no_reset[19]["spot_ema20"])
        self.assertIsNotNone(no_reset[20]["spot_ema20"])
        self.assertIsNone(with_reset[20]["spot_ema20"])

    def test_spot_ema20_matches_standalone_ema_controller(self) -> None:
        standalone = EmaController(20)
        spot_ctrl = SpotControllers()
        open_ts = 1_700_000_000.0
        for i, spot in enumerate(REGRESSION_SPOTS_EMA20):
            ts = open_ts + i * STEP_SEC
            sample = ControllerSample.spot(spot, ts)
            standalone.update(sample)
            spot_ctrl.update(spot, ts=ts)
        self.assertAlmostEqual(
            spot_ctrl.ema20.value(),
            standalone.value(),
            places=9,
        )

    def test_spot_ema20_parallel_cache_matches_controller(self) -> None:
        from chain_replay_ml.dataset_builder.rolling_controllers import build_spot_rv_cache

        open_ts = 1_700_000_000.0
        tl = TickTimeline()
        timestamps: list[float] = []
        for i, spot in enumerate(REGRESSION_SPOTS_EMA20):
            ts = open_ts + i * STEP_SEC
            tl.append(ts, int(round(spot * 100)))
            timestamps.append(ts)
        cache = build_spot_rv_cache(tl, timestamps)
        spot_ctrl = SpotControllers()
        for i, spot in enumerate(REGRESSION_SPOTS_EMA20):
            spot_ctrl.update(spot, ts=timestamps[i])
            cached = cache[timestamps[i]].get("spot_ema20")
            ctrl_val = spot_ctrl.ema20.value()
            if ctrl_val is None:
                self.assertIsNone(cached)
            else:
                self.assertAlmostEqual(float(cached), ctrl_val, places=9)

    def test_controller_owned_readiness_includes_spot_ema20(self) -> None:
        from chain_replay_ml.dataset_builder.rolling_controllers import (
            CONTROLLER_OWNED_READINESS_FEATURES,
        )

        self.assertIn("spot_ema20", CONTROLLER_OWNED_READINESS_FEATURES)


REGRESSION_SPOTS_EMA50 = REGRESSION_SPOTS_EMA20 + [
    25027.30, 25028.10, 25029.50, 25030.00, 25031.20,
    25032.00, 25033.50, 25034.00, 25035.10, 25036.00,
    25037.20, 25038.00, 25039.10, 25040.00, 25041.50,
    25042.00, 25043.20, 25044.00, 25045.10, 25046.00,
    25047.30, 25048.00, 25049.10, 25050.00, 25051.20,
    25052.00, 25053.10, 25054.00, 25055.00,
]


def _emit_spot_ema50(
    spots: list[float],
    *,
    ltp: float = 100.0,
    gap_after_row: int | None = None,
    reset_spot_on_gap: bool = False,
) -> list[dict[str, float | None]]:
    spot_ctrl = SpotControllers()
    open_ts = 1_700_000_000.0
    out_rows: list[dict[str, float | None]] = []
    last_ts: float | None = None

    for i, spot in enumerate(spots):
        ts = open_ts + i * STEP_SEC
        if gap_after_row is not None and i == gap_after_row:
            ts = last_ts + GAP_MAX_SEC + 1.0 if last_ts is not None else ts
        if last_ts is not None and row_gap_exceeds(ts, last_ts, GAP_MAX_SEC) and reset_spot_on_gap:
            spot_ctrl.reset_all(ts=ts)
        spot_ctrl.update(spot, ts=ts)
        raw = {"ltp": ltp, "spot": spot}
        enriched = enrich_with_chain_maps(
            raw,
            ts=ts,
            chain_maps=ChainMaps(),
            strike_mapping={},
            index_tl=TickTimeline(),
            atm_strike=25000,
            expiry_ts=open_ts + 86400.0,
            option_timeline=TickTimeline(),
            open_ts=open_ts,
            close_ts=open_ts + 3600.0 * 6,
            active_features=frozenset({"spot_ema50"}),
            feature_grid_step_sec=STEP_SEC,
            spot_controllers=spot_ctrl,
        )
        out_rows.append({
            "spot_ema50": enriched.get("spot_ema50"),
        })
        last_ts = ts
    return out_rows


class SpotEma50ControllerTests(unittest.TestCase):
    def test_rows_1_to_49_null_for_spot_ema50_level(self) -> None:
        rows = _emit_spot_ema50(REGRESSION_SPOTS_EMA50)
        for row in rows[:49]:
            self.assertIsNone(row["spot_ema50"])

    def test_row_50_first_valid_spot_ema50(self) -> None:
        rows = _emit_spot_ema50(REGRESSION_SPOTS_EMA50)
        row50 = rows[49]
        expected_ema = _manual_ema50(REGRESSION_SPOTS_EMA50[:50])
        self.assertIsNotNone(row50["spot_ema50"])
        self.assertAlmostEqual(row50["spot_ema50"], expected_ema, places=9)

    def test_spot_ema50_dedupes_one_sample_per_timestamp(self) -> None:
        spot_ctrl = SpotControllers()
        ts0 = 1_700_000_000.0
        for _ in range(100):
            spot_ctrl.update(25000.0, ts=ts0)
        self.assertFalse(spot_ctrl.ema50.ready())
        self.assertEqual(spot_ctrl.ema50.samples, 1)
        spot_ctrl.update(25001.0, ts=ts0 + STEP_SEC)
        self.assertEqual(spot_ctrl.ema50.samples, 2)

    def test_token_gap_does_not_reset_spot_ema50(self) -> None:
        spots = [25000.0 + i * 1.0 for i in range(51)]
        no_reset = _emit_spot_ema50(spots, gap_after_row=50, reset_spot_on_gap=False)
        with_reset = _emit_spot_ema50(spots, gap_after_row=50, reset_spot_on_gap=True)
        self.assertIsNotNone(no_reset[49]["spot_ema50"])
        self.assertIsNotNone(no_reset[50]["spot_ema50"])
        self.assertIsNone(with_reset[50]["spot_ema50"])

    def test_spot_ema50_matches_standalone_ema_controller(self) -> None:
        standalone = EmaController(50)
        spot_ctrl = SpotControllers()
        open_ts = 1_700_000_000.0
        for i, spot in enumerate(REGRESSION_SPOTS_EMA50):
            ts = open_ts + i * STEP_SEC
            sample = ControllerSample.spot(spot, ts)
            standalone.update(sample)
            spot_ctrl.update(spot, ts=ts)
        self.assertAlmostEqual(
            spot_ctrl.ema50.value(),
            standalone.value(),
            places=9,
        )

    def test_spot_ema50_parallel_cache_matches_controller(self) -> None:
        from chain_replay_ml.dataset_builder.rolling_controllers import build_spot_rv_cache

        open_ts = 1_700_000_000.0
        tl = TickTimeline()
        timestamps: list[float] = []
        for i, spot in enumerate(REGRESSION_SPOTS_EMA50):
            ts = open_ts + i * STEP_SEC
            tl.append(ts, int(round(spot * 100)))
            timestamps.append(ts)
        cache = build_spot_rv_cache(tl, timestamps)
        spot_ctrl = SpotControllers()
        for i, spot in enumerate(REGRESSION_SPOTS_EMA50):
            spot_ctrl.update(spot, ts=timestamps[i])
            cached = cache[timestamps[i]].get("spot_ema50")
            ctrl_val = spot_ctrl.ema50.value()
            if ctrl_val is None:
                self.assertIsNone(cached)
            else:
                self.assertAlmostEqual(float(cached), ctrl_val, places=9)

    def test_controller_owned_readiness_includes_spot_ema50(self) -> None:
        from chain_replay_ml.dataset_builder.rolling_controllers import (
            CONTROLLER_OWNED_READINESS_FEATURES,
        )

        self.assertIn("spot_ema50", CONTROLLER_OWNED_READINESS_FEATURES)


REGRESSION_SPOTS_EMA100 = REGRESSION_SPOTS_EMA50 + [25056.0 + i for i in range(50)]
REGRESSION_SPOTS_EMA200 = REGRESSION_SPOTS_EMA100 + [25106.0 + i for i in range(100)]


def _emit_spot_ema100(
    spots: list[float],
    *,
    ltp: float = 100.0,
    gap_after_row: int | None = None,
    reset_spot_on_gap: bool = False,
) -> list[dict[str, float | None]]:
    spot_ctrl = SpotControllers()
    open_ts = 1_700_000_000.0
    out_rows: list[dict[str, float | None]] = []
    last_ts: float | None = None

    for i, spot in enumerate(spots):
        ts = open_ts + i * STEP_SEC
        if gap_after_row is not None and i == gap_after_row:
            ts = last_ts + GAP_MAX_SEC + 1.0 if last_ts is not None else ts
        if last_ts is not None and row_gap_exceeds(ts, last_ts, GAP_MAX_SEC) and reset_spot_on_gap:
            spot_ctrl.reset_all(ts=ts)
        spot_ctrl.update(spot, ts=ts)
        raw = {"ltp": ltp, "spot": spot}
        enriched = enrich_with_chain_maps(
            raw,
            ts=ts,
            chain_maps=ChainMaps(),
            strike_mapping={},
            index_tl=TickTimeline(),
            atm_strike=25000,
            expiry_ts=open_ts + 86400.0,
            option_timeline=TickTimeline(),
            open_ts=open_ts,
            close_ts=open_ts + 3600.0 * 6,
            active_features=frozenset({"spot_ema100"}),
            feature_grid_step_sec=STEP_SEC,
            spot_controllers=spot_ctrl,
        )
        out_rows.append({
            "spot_ema100": enriched.get("spot_ema100"),
        })
        last_ts = ts
    return out_rows


def _emit_spot_ema200(
    spots: list[float],
    *,
    ltp: float = 100.0,
    gap_after_row: int | None = None,
    reset_spot_on_gap: bool = False,
) -> list[dict[str, float | None]]:
    spot_ctrl = SpotControllers()
    open_ts = 1_700_000_000.0
    out_rows: list[dict[str, float | None]] = []
    last_ts: float | None = None

    for i, spot in enumerate(spots):
        ts = open_ts + i * STEP_SEC
        if gap_after_row is not None and i == gap_after_row:
            ts = last_ts + GAP_MAX_SEC + 1.0 if last_ts is not None else ts
        if last_ts is not None and row_gap_exceeds(ts, last_ts, GAP_MAX_SEC) and reset_spot_on_gap:
            spot_ctrl.reset_all(ts=ts)
        spot_ctrl.update(spot, ts=ts)
        raw = {"ltp": ltp, "spot": spot}
        enriched = enrich_with_chain_maps(
            raw,
            ts=ts,
            chain_maps=ChainMaps(),
            strike_mapping={},
            index_tl=TickTimeline(),
            atm_strike=25000,
            expiry_ts=open_ts + 86400.0,
            option_timeline=TickTimeline(),
            open_ts=open_ts,
            close_ts=open_ts + 3600.0 * 6,
            active_features=frozenset({"spot_ema200"}),
            feature_grid_step_sec=STEP_SEC,
            spot_controllers=spot_ctrl,
        )
        out_rows.append({
            "spot_ema200": enriched.get("spot_ema200"),
        })
        last_ts = ts
    return out_rows


class SpotEma100200FamilyTests(unittest.TestCase):
    def test_rows_1_to_99_null_for_spot_ema100_level(self) -> None:
        rows = _emit_spot_ema100(REGRESSION_SPOTS_EMA100)
        for row in rows[:99]:
            self.assertIsNone(row["spot_ema100"])

    def test_row_100_first_valid_spot_ema100(self) -> None:
        rows = _emit_spot_ema100(REGRESSION_SPOTS_EMA100)
        row100 = rows[99]
        expected_ema = _manual_ema100(REGRESSION_SPOTS_EMA100[:100])
        self.assertIsNotNone(row100["spot_ema100"])
        self.assertAlmostEqual(row100["spot_ema100"], expected_ema, places=9)

    def test_rows_1_to_199_null_for_spot_ema200_level(self) -> None:
        rows = _emit_spot_ema200(REGRESSION_SPOTS_EMA200)
        for row in rows[:199]:
            self.assertIsNone(row["spot_ema200"])

    def test_row_200_first_valid_spot_ema200(self) -> None:
        rows = _emit_spot_ema200(REGRESSION_SPOTS_EMA200)
        row200 = rows[199]
        expected_ema = _manual_ema200(REGRESSION_SPOTS_EMA200[:200])
        self.assertIsNotNone(row200["spot_ema200"])
        self.assertAlmostEqual(row200["spot_ema200"], expected_ema, places=9)

    def test_spot_ema100_dedupes_one_sample_per_timestamp(self) -> None:
        spot_ctrl = SpotControllers()
        ts0 = 1_700_000_000.0
        for _ in range(100):
            spot_ctrl.update(25000.0, ts=ts0)
        self.assertFalse(spot_ctrl.ema100.ready())
        self.assertEqual(spot_ctrl.ema100.samples, 1)
        spot_ctrl.update(25001.0, ts=ts0 + STEP_SEC)
        self.assertEqual(spot_ctrl.ema100.samples, 2)

    def test_spot_ema200_dedupes_one_sample_per_timestamp(self) -> None:
        spot_ctrl = SpotControllers()
        ts0 = 1_700_000_000.0
        for _ in range(200):
            spot_ctrl.update(25000.0, ts=ts0)
        self.assertFalse(spot_ctrl.ema200.ready())
        self.assertEqual(spot_ctrl.ema200.samples, 1)
        spot_ctrl.update(25001.0, ts=ts0 + STEP_SEC)
        self.assertEqual(spot_ctrl.ema200.samples, 2)

    def test_token_gap_does_not_reset_spot_ema100(self) -> None:
        spots = [25000.0 + i * 1.0 for i in range(101)]
        no_reset = _emit_spot_ema100(spots, gap_after_row=100, reset_spot_on_gap=False)
        with_reset = _emit_spot_ema100(spots, gap_after_row=100, reset_spot_on_gap=True)
        self.assertIsNotNone(no_reset[99]["spot_ema100"])
        self.assertIsNotNone(no_reset[100]["spot_ema100"])
        self.assertIsNone(with_reset[100]["spot_ema100"])

    def test_token_gap_does_not_reset_spot_ema200(self) -> None:
        spots = [25000.0 + i * 1.0 for i in range(201)]
        no_reset = _emit_spot_ema200(spots, gap_after_row=200, reset_spot_on_gap=False)
        with_reset = _emit_spot_ema200(spots, gap_after_row=200, reset_spot_on_gap=True)
        self.assertIsNotNone(no_reset[199]["spot_ema200"])
        self.assertIsNotNone(no_reset[200]["spot_ema200"])
        self.assertIsNone(with_reset[200]["spot_ema200"])

    def test_spot_ema100_matches_standalone_ema_controller(self) -> None:
        standalone = EmaController(100)
        spot_ctrl = SpotControllers()
        open_ts = 1_700_000_000.0
        for i, spot in enumerate(REGRESSION_SPOTS_EMA100):
            ts = open_ts + i * STEP_SEC
            sample = ControllerSample.spot(spot, ts)
            standalone.update(sample)
            spot_ctrl.update(spot, ts=ts)
        self.assertAlmostEqual(
            spot_ctrl.ema100.value(),
            standalone.value(),
            places=9,
        )

    def test_spot_ema200_matches_standalone_ema_controller(self) -> None:
        standalone = EmaController(200)
        spot_ctrl = SpotControllers()
        open_ts = 1_700_000_000.0
        for i, spot in enumerate(REGRESSION_SPOTS_EMA200):
            ts = open_ts + i * STEP_SEC
            sample = ControllerSample.spot(spot, ts)
            standalone.update(sample)
            spot_ctrl.update(spot, ts=ts)
        self.assertAlmostEqual(
            spot_ctrl.ema200.value(),
            standalone.value(),
            places=9,
        )

    def test_spot_ema100_parallel_cache_matches_controller(self) -> None:
        from chain_replay_ml.dataset_builder.rolling_controllers import build_spot_rv_cache

        open_ts = 1_700_000_000.0
        tl = TickTimeline()
        timestamps: list[float] = []
        for i, spot in enumerate(REGRESSION_SPOTS_EMA100):
            ts = open_ts + i * STEP_SEC
            tl.append(ts, int(round(spot * 100)))
            timestamps.append(ts)
        cache = build_spot_rv_cache(tl, timestamps)
        spot_ctrl = SpotControllers()
        for i, spot in enumerate(REGRESSION_SPOTS_EMA100):
            spot_ctrl.update(spot, ts=timestamps[i])
            cached = cache[timestamps[i]].get("spot_ema100")
            ctrl_val = spot_ctrl.ema100.value()
            if ctrl_val is None:
                self.assertIsNone(cached)
            else:
                self.assertAlmostEqual(float(cached), ctrl_val, places=9)

    def test_spot_ema200_parallel_cache_matches_controller(self) -> None:
        from chain_replay_ml.dataset_builder.rolling_controllers import build_spot_rv_cache

        open_ts = 1_700_000_000.0
        tl = TickTimeline()
        timestamps: list[float] = []
        for i, spot in enumerate(REGRESSION_SPOTS_EMA200):
            ts = open_ts + i * STEP_SEC
            tl.append(ts, int(round(spot * 100)))
            timestamps.append(ts)
        cache = build_spot_rv_cache(tl, timestamps)
        spot_ctrl = SpotControllers()
        for i, spot in enumerate(REGRESSION_SPOTS_EMA200):
            spot_ctrl.update(spot, ts=timestamps[i])
            cached = cache[timestamps[i]].get("spot_ema200")
            ctrl_val = spot_ctrl.ema200.value()
            if ctrl_val is None:
                self.assertIsNone(cached)
            else:
                self.assertAlmostEqual(float(cached), ctrl_val, places=9)

    def test_controller_owned_readiness_includes_spot_ema100_and_ema200(self) -> None:
        from chain_replay_ml.dataset_builder.rolling_controllers import (
            CONTROLLER_OWNED_READINESS_FEATURES,
        )

        self.assertIn("spot_ema100", CONTROLLER_OWNED_READINESS_FEATURES)
        self.assertIn("spot_ema200", CONTROLLER_OWNED_READINESS_FEATURES)


def _delayed_spot_tl(
    open_ts: float,
    delay_sec: float,
    n_samples: int,
    *,
    step_sec: float = STEP_SEC,
    base_price: float = 25000.0,
) -> TickTimeline:
    tl = TickTimeline()
    for i in range(n_samples):
        ts = open_ts + delay_sec + i * step_sec
        price = base_price + i * 1.0
        tl.append(ts, int(round(price * 100)))
    return tl


class EffectiveSessionStartTests(unittest.TestCase):
    """Delayed first tick — controllers must not assume unseen 09:15 history."""

    def test_delayed_start_spot_rv_5m_warmup_from_first_tick(self) -> None:
        open_ts = 1_700_000_000.0
        delay_sec = 44 * 60 + 33  # replay-style gap after exchange open
        effective = open_ts + delay_sec
        n = 35
        tl = _delayed_spot_tl(open_ts, delay_sec, n)
        timestamps = [effective + i * STEP_SEC for i in range(n)]

        from chain_replay_ml.dataset_builder.rolling_controllers import build_spot_rv_cache

        cache = build_spot_rv_cache(tl, timestamps)
        for i, ts in enumerate(timestamps[:30]):
            self.assertIsNone(cache[ts]["spot_rv_5m"], f"unexpected spot_rv_5m at sample {i + 1}")
            self.assertIsNone(cache[ts]["spot_rv_ratio"])
        self.assertIsNotNone(cache[timestamps[30]]["spot_rv_5m"])

    def test_delayed_start_feature_grid_origin(self) -> None:
        from chain_replay_ml.dataset_builder.day_context import DayContext, SourceSpec
        from chain_replay_ml.dataset_builder.tick_coverage import feature_grid_origin_ts
        from chain_replay_ml.ticks import uniform_grid

        open_ts = 1_700_000_000.0
        delay_sec = 44 * 60 + 33
        effective = open_ts + delay_sec
        n = 15
        tl = _delayed_spot_tl(open_ts, delay_sec, n, base_price=25000.0)
        close_ts = open_ts + 6 * 3600.0

        ctx = DayContext(
            source=SourceSpec("t", "2026-01-02", "NIFTY", "2026-01-08"),
            db_path="",
            expiry_norm="2026-01-08",
            open_ts=open_ts,
            effective_session_start_ts=effective,
            close_ts=close_ts,
            expiry_ts=close_ts + 86400.0,
            index_tl=tl,
            strike_mapping={},
            feature_grid_step_sec=STEP_SEC,
        )

        grid_origin = feature_grid_origin_ts(ctx)
        expected_first = float(uniform_grid(effective, close_ts, STEP_SEC)[0])
        self.assertAlmostEqual(grid_origin, expected_first, places=6)
        self.assertGreater(grid_origin, open_ts + delay_sec - STEP_SEC)

        wrong_first = float(uniform_grid(open_ts, close_ts, STEP_SEC)[0])
        self.assertNotAlmostEqual(grid_origin, wrong_first, places=4)


IV_ACTIVE_FEATURES = frozenset({
    "iv_zscore_1m",
    "iv_zscore_5m",
    "iv_zscore_15m",
    "iv_zscore_30m",
    "iv_rank_session",
    "iv_change_1m",
    "iv_change_5m",
    "iv_change_15m",
    "iv_pct_change_1m",
})

IV_VERIFICATION_ROWS: tuple[tuple[str, str, int, int], ...] = tuple(
    (spec.feature_name, spec.controller_id, spec.expected_first_valid, spec.expected_first_valid)
    for spec in CONTROLLER_WARMUP_SPEC
    if spec.feature_name.startswith("iv_")
)


def _manual_iv_zscore(
    history: list[tuple[float, float]],
    ts: float,
    iv: float,
    window_sec: float,
) -> float:
    cutoff = ts - window_sec
    priors = [v for t, v in history if t >= cutoff and t < ts]
    mean = sum(priors) / len(priors)
    variance = sum((v - mean) ** 2 for v in priors) / len(priors)
    std = float(np.std(priors, ddof=0)) if priors else 0.0
    return (iv - mean) / std if std > 1e-8 else 0.0


def _manual_session_rank(ivs: list[float], iv: float) -> float:
    min_iv = min(ivs)
    max_iv = max(ivs)
    span = max_iv - min_iv
    if span <= 1e-12:
        return 50.0
    return (iv - min_iv) / span * 100.0


def _emit_iv_features(
    ivs: list[float],
    *,
    gap_after_row: int | None = None,
) -> list[dict[str, float | None]]:
    opt_state = OptionFeatureState()
    open_ts = 1_700_000_000.0
    out_rows: list[dict[str, float | None]] = []
    last_ts: float | None = None
    history: list[tuple[float, float]] = []

    for i, iv in enumerate(ivs):
        ts = _sim_row_ts(i, open_ts=open_ts, last_ts=last_ts, gap_after_row=gap_after_row)
        if last_ts is not None and row_gap_exceeds(ts, last_ts, GAP_MAX_SEC):
            opt_state.controllers.reset_all(ts=ts)
            history.clear()
        update_token_iv_controllers(opt_state.controllers, iv, ts=ts)
        history.append((ts, iv))
        raw = {"ltp": 100.0, "spot": 25000.0}
        enriched = enrich_with_chain_maps(
            raw,
            ts=ts,
            chain_maps=ChainMaps(),
            strike_mapping={},
            index_tl=TickTimeline(),
            atm_strike=25000,
            expiry_ts=open_ts + 86400.0,
            opt_state=opt_state,
            option_timeline=TickTimeline(),
            open_ts=open_ts,
            close_ts=open_ts + 3600.0 * 6,
            active_features=IV_ACTIVE_FEATURES,
            feature_grid_step_sec=STEP_SEC,
        )
        out_rows.append({feat: enriched.get(feat) for feat in IV_ACTIVE_FEATURES})
        last_ts = ts
    return out_rows


class IvControllerTests(unittest.TestCase):
    def test_verification_table_warmup_and_first_valid(self) -> None:
        """Policy-aligned warmup / first-valid sample for all IV controller features."""
        max_n = max(first for _, _, _, first in IV_VERIFICATION_ROWS)
        ivs = [0.18 + (i % 7) * 0.002 for i in range(max_n)]
        rows = _emit_iv_features(ivs)
        results: list[tuple[str, str, int, int, str]] = []

        for feat, controller_id, warmup, first_valid in IV_VERIFICATION_ROWS:
            null_ok = all(rows[i][feat] is None for i in range(first_valid - 1))
            valid_ok = rows[first_valid - 1][feat] is not None
            status = "PASS" if null_ok and valid_ok else "FAIL"
            results.append((feat, controller_id, warmup, first_valid, status))
            self.assertTrue(null_ok, f"{feat}: expected NULL for samples 1..{first_valid - 1}")
            self.assertTrue(valid_ok, f"{feat}: expected valid at sample {first_valid}")

        for feat, controller_id, warmup, first_valid, status in results:
            self.assertEqual(status, "PASS", f"{feat} verification failed")

    def test_iv_rank_session_valid_at_sample_1_neutral_50(self) -> None:
        rows = _emit_iv_features([0.22])
        self.assertAlmostEqual(rows[0]["iv_rank_session"], 50.0, places=9)

    def test_iv_rank_session_tracks_session_min_max(self) -> None:
        ivs = [0.20, 0.22, 0.18, 0.24]
        rows = _emit_iv_features(ivs)
        self.assertAlmostEqual(
            rows[-1]["iv_rank_session"],
            _manual_session_rank(ivs, ivs[-1]),
            places=9,
        )

    def test_iv_zscore_1m_null_samples_1_to_19_valid_at_20(self) -> None:
        ivs = [0.18 + (i % 5) * 0.003 for i in range(25)]
        rows = _emit_iv_features(ivs)
        for row in rows[:19]:
            self.assertIsNone(row["iv_zscore_1m"])
        self.assertIsNotNone(rows[19]["iv_zscore_1m"])
        open_ts = 1_700_000_000.0
        ts20 = open_ts + 19 * STEP_SEC
        history = [(open_ts + j * STEP_SEC, ivs[j]) for j in range(19)]
        expected = _manual_iv_zscore(history, ts20, ivs[19], 60.0)
        self.assertAlmostEqual(rows[19]["iv_zscore_1m"], expected, places=9)

    def test_iv_zscore_std_zero_returns_zero(self) -> None:
        ctrl = IvZscoreWindowController(60.0, 20)
        open_ts = 1_700_000_000.0
        for i in range(19):
            ctrl.update(0.20, ts=open_ts + i * STEP_SEC)
        ctrl.update(0.25, ts=open_ts + 19 * STEP_SEC)
        self.assertTrue(ctrl.ready())
        val = ctrl.value()
        self.assertIsNotNone(val)
        self.assertAlmostEqual(val, 0.0, places=9)

    def test_gap_reset_clears_iv_controllers(self) -> None:
        ivs = [0.18 + i * 0.001 for i in range(25)] + [0.30]
        rows = _emit_iv_features(ivs, gap_after_row=25)
        self.assertIsNotNone(rows[24]["iv_zscore_1m"])
        self.assertIsNone(rows[25]["iv_zscore_1m"])
        self.assertAlmostEqual(rows[25]["iv_rank_session"], 50.0, places=9)

    def test_iv_zscore_controller_unit(self) -> None:
        ctrl = IvZscoreWindowController(60.0, 20)
        open_ts = 1_700_000_000.0
        ivs = [0.18 + (i % 5) * 0.003 for i in range(20)]
        history: list[tuple[float, float]] = []
        for i, iv in enumerate(ivs):
            ts = open_ts + i * STEP_SEC
            ctrl.update(iv, ts=ts)
            if i == 19:
                expected = _manual_iv_zscore(history, ts, iv, 60.0)
                self.assertAlmostEqual(ctrl.value(), expected, places=9)
            history.append((ts, iv))
        self.assertEqual(ctrl.reset_feature_label(), "IVZ1M")

    def test_iv_session_rank_controller_unit(self) -> None:
        ctrl = IvSessionRankController()
        ctrl.update(0.22, ts=1_700_000_000.0)
        self.assertTrue(ctrl.ready())
        self.assertAlmostEqual(ctrl.value(), 50.0, places=9)
        ctrl.update(0.18, ts=1_700_000_003.0)
        self.assertAlmostEqual(ctrl.value(), 0.0, places=9)
        self.assertEqual(ctrl.reset_feature_label(), "IVRANK")

    def test_token_controllers_iv_warmup_periods(self) -> None:
        tc = TokenControllers()
        self.assertEqual(tc.iv_zscore_1m.warmup_period, 20)
        self.assertEqual(tc.iv_zscore_5m.warmup_period, 100)
        self.assertEqual(tc.iv_zscore_15m.warmup_period, 300)
        self.assertEqual(tc.iv_zscore_30m.warmup_period, 600)
        self.assertEqual(tc.iv_session_rank.warmup_period, 1)

    def test_controller_owned_readiness_includes_iv_features(self) -> None:
        from chain_replay_ml.dataset_builder.rolling_controllers import (
            CONTROLLER_OWNED_READINESS_FEATURES,
        )

        for feat in IV_ACTIVE_FEATURES:
            self.assertIn(feat, CONTROLLER_OWNED_READINESS_FEATURES)


IV_HIST_ACTIVE_FEATURES = frozenset({
    "iv_change_1m",
    "iv_change_5m",
    "iv_change_15m",
    "iv_pct_change_1m",
})


def _manual_iv_change_1m_from_history(
    history: list[tuple[float, float]],
    ts: float,
    current_iv: float,
) -> float | None:
    target = float(ts) - 60.0
    lag_iv = None
    for t, v in reversed(history):
        if t <= target + 1e-6:
            lag_iv = float(v)
            break
    if lag_iv is None:
        return None
    return float(current_iv) * 100.0 - lag_iv * 100.0


def _emit_iv_history_features(
    ivs: list[float],
    *,
    gap_after_row: int | None = None,
) -> list[dict[str, float | None]]:
    opt_state = OptionFeatureState()
    open_ts = 1_700_000_000.0
    out_rows: list[dict[str, float | None]] = []
    last_ts: float | None = None

    for i, iv in enumerate(ivs):
        ts = _sim_row_ts(i, open_ts=open_ts, last_ts=last_ts, gap_after_row=gap_after_row)
        if last_ts is not None and row_gap_exceeds(ts, last_ts, GAP_MAX_SEC):
            opt_state.controllers.reset_all(ts=ts)
        update_token_iv_controllers(opt_state.controllers, iv, ts=ts)
        raw = {"ltp": 100.0, "spot": 25000.0}
        enriched = enrich_with_chain_maps(
            raw,
            ts=ts,
            chain_maps=ChainMaps(),
            strike_mapping={},
            index_tl=TickTimeline(),
            atm_strike=25000,
            expiry_ts=open_ts + 86400.0,
            opt_state=opt_state,
            option_timeline=TickTimeline(),
            open_ts=open_ts,
            close_ts=open_ts + 3600.0 * 6,
            active_features=IV_HIST_ACTIVE_FEATURES,
            feature_grid_step_sec=STEP_SEC,
        )
        out_rows.append({feat: enriched.get(feat) for feat in IV_HIST_ACTIVE_FEATURES})
        last_ts = ts
    return out_rows


class IvHistoryControllerTests(unittest.TestCase):
    def test_iv_change_1m_first_valid_after_calendar_lag(self) -> None:
        ivs = [0.18 + (i % 5) * 0.003 for i in range(25)]
        rows = _emit_iv_history_features(ivs)
        for row in rows[:20]:
            self.assertIsNone(row["iv_change_1m"])
        self.assertIsNotNone(rows[20]["iv_change_1m"])

    def test_permanent_regression_no_hidden_iv_history(self) -> None:
        """Mandatory: after gap, iv_change_1m = current_iv − lag_iv(post-gap only)."""
        pre_gap_n = 25
        ivs = [0.20] * pre_gap_n + [0.22 + i * 0.002 for i in range(50)]
        rows = _emit_iv_history_features(ivs, gap_after_row=pre_gap_n)
        open_ts = 1_700_000_000.0
        post_hist: list[tuple[float, float]] = []
        last_ts: float | None = None
        for i, iv in enumerate(ivs):
            ts = _sim_row_ts(
                i, open_ts=open_ts, last_ts=last_ts, gap_after_row=pre_gap_n,
            )
            if i >= pre_gap_n:
                post_hist.append((ts, iv))
            change = rows[i].get("iv_change_1m")
            if i >= pre_gap_n and change is not None:
                manual = _manual_iv_change_1m_from_history(post_hist, ts, iv)
                self.assertIsNotNone(manual)
                self.assertAlmostEqual(change, manual, places=9)
                wrong = float(iv) * 100.0 - 0.20 * 100.0
                if abs(wrong - manual) > 1e-6:
                    self.assertNotAlmostEqual(change, wrong, places=6)
            last_ts = ts

    def test_gap_resets_iv_history(self) -> None:
        ivs = [0.18 + i * 0.001 for i in range(30)] + [0.30]
        rows = _emit_iv_history_features(ivs, gap_after_row=30)
        self.assertIsNotNone(rows[29]["iv_change_1m"])
        self.assertIsNone(rows[30]["iv_change_1m"])

    def test_debug_introspection_in_debug_mode(self) -> None:
        ctrl = IvHistoryController()
        if not __debug__:
            self.skipTest("Debug introspection requires __debug__")
        open_ts = 1_700_000_000.0
        ctrl.update(0.20, ts=open_ts)
        ctrl.update(0.21, ts=open_ts + STEP_SEC)
        self.assertEqual(ctrl.sample_count, 2)
        self.assertEqual(ctrl.oldest_timestamp, open_ts)
        self.assertEqual(ctrl.newest_timestamp, open_ts + STEP_SEC)

    def test_iv_history_controller_unit(self) -> None:
        ctrl = IvHistoryController()
        open_ts = 1_700_000_000.0
        ivs = [0.18 + (i % 5) * 0.003 for i in range(22)]
        for i, iv in enumerate(ivs):
            ctrl.update(iv, ts=open_ts + i * STEP_SEC)
        self.assertTrue(ctrl.ready_for_lag(60.0))
        change = ctrl.iv_change_pct_points(60.0)
        self.assertIsNotNone(change)
        lag = ctrl.lag_iv(60.0)
        assert lag is not None
        expected = ivs[-1] * 100.0 - lag * 100.0
        self.assertAlmostEqual(change, expected, places=9)


class MonotonicTimestampInvariantTests(unittest.TestCase):
    """Permanent: new_timestamp >= previous_timestamp on every controller update."""

    def test_permanent_monotonic_timestamp_rolling_controller(self) -> None:
        if not __debug__:
            self.skipTest("Monotonic timestamp invariant requires __debug__")
        ctrl = EmaController(9)
        open_ts = 1_700_000_000.0
        for i in range(5):
            ctrl.update(100.0 + i, ts=open_ts + i * STEP_SEC)
        with self.assertRaises(AssertionError):
            ctrl.update(105.0, ts=open_ts + 3 * STEP_SEC)

    def test_permanent_monotonic_timestamp_iv_history(self) -> None:
        if not __debug__:
            self.skipTest("Monotonic timestamp invariant requires __debug__")
        ctrl = IvHistoryController()
        open_ts = 1_700_000_000.0
        ctrl.update(0.20, ts=open_ts)
        ctrl.update(0.21, ts=open_ts + STEP_SEC)
        with self.assertRaises(AssertionError):
            ctrl.update(0.22, ts=open_ts)

    def test_equal_timestamp_allowed(self) -> None:
        if not __debug__:
            self.skipTest("Monotonic timestamp invariant requires __debug__")
        ctrl = EmaController(9)
        ts = 1_700_000_000.0
        ctrl.update(100.0, ts=ts)
        ctrl.update(101.0, ts=ts)
        self.assertEqual(ctrl.samples, 2)

    def test_assert_monotonic_helper_documents_replay_bug(self) -> None:
        if not __debug__:
            self.skipTest("Monotonic timestamp invariant requires __debug__")
        with self.assertRaises(AssertionError):
            assert_monotonic_controller_ts(1_700_000_083.0, 1_700_000_078.0)

    def test_permanent_non_monotonic_gap_replay_fails(self) -> None:
        """Broken post-gap timestamps (open_ts + i*step) must trip the invariant."""
        if not __debug__:
            self.skipTest("Monotonic timestamp invariant requires __debug__")
        opt_state = OptionFeatureState()
        open_ts = 1_700_000_000.0
        pre_gap_n = 25
        ivs = [0.20] * pre_gap_n + [0.22 + i * 0.002 for i in range(10)]
        last_ts: float | None = None
        with self.assertRaises(AssertionError):
            for i, iv in enumerate(ivs):
                ts = open_ts + i * STEP_SEC
                if i == pre_gap_n and last_ts is not None:
                    ts = last_ts + GAP_MAX_SEC + 1.0
                if last_ts is not None and row_gap_exceeds(ts, last_ts, GAP_MAX_SEC):
                    opt_state.controllers.reset_all(ts=ts)
                update_token_iv_controllers(opt_state.controllers, iv, ts=ts)
                last_ts = ts


class ControllerResetCompletenessTests(unittest.TestCase):
    """Permanent: reset() clears all cached state; first counted update yields samples == 1."""

    OPEN_TS = 1_700_000_000.0

    def _assert_rolling_reset_state(self, ctrl: EmaController | StdController | RvController | IvZscoreWindowController | IvSessionRankController, *, label: str) -> None:
        self.assertEqual(ctrl.samples, 0, label)
        self.assertFalse(ctrl.ready(), label)
        self.assertIsNone(ctrl.value(), label)
        self.assertIsNone(ctrl.last_update_ts, label)

    def test_permanent_reset_complete_all_rolling_controller_types(self) -> None:
        open_ts = self.OPEN_TS
        reset_ts = open_ts + 99_999.0
        cases: list[tuple[str, object, object]] = [
            ("ema9", EmaController(9), lambda c, i: c.update(100.0 + i, ts=open_ts + i * STEP_SEC)),
            ("std20", StdController(), lambda c, i: c.update(100.0 + i, ts=open_ts + i * STEP_SEC)),
            ("iv_zscore_1m", IvZscoreWindowController(60.0, 20), lambda c, i: c.update(ControllerSample.iv(0.20 + i * 0.001, open_ts + i * STEP_SEC))),
            ("iv_rank", IvSessionRankController(), lambda c, i: c.update(ControllerSample.iv(0.20 + i * 0.001, open_ts + i * STEP_SEC))),
        ]
        for label, ctrl, warm in cases:
            with self.subTest(controller=label):
                for i in range(25):
                    warm(ctrl, i)
                self.assertGreater(ctrl.samples, 0)
                ctrl.reset(reset_ts)
                self._assert_rolling_reset_state(ctrl, label=label)
                post_ts = reset_ts + STEP_SEC
                if label == "ema9" or label == "std20":
                    ctrl.update(100.0, ts=post_ts)
                else:
                    ctrl.update(ControllerSample.iv(0.22, post_ts))
                self.assertEqual(ctrl.samples, 1, label)

    def test_permanent_reset_complete_rv_first_return_on_second_update(self) -> None:
        """RV seeds price on first tick; samples increments on second (return available)."""
        ctrl = RvController(30)
        open_ts = self.OPEN_TS
        for i in range(35):
            ctrl.update(ControllerSample.ltp(100.0 + i * 0.1, open_ts + i * STEP_SEC))
        self.assertGreater(ctrl.samples, 0)
        ctrl.reset(open_ts + 99_999.0)
        self._assert_rolling_reset_state(ctrl, label="rv5m")
        ctrl.update(ControllerSample.ltp(200.0, open_ts + 100_000 * STEP_SEC))
        self.assertEqual(ctrl.samples, 0)
        ctrl.update(ControllerSample.ltp(201.0, open_ts + 100_001 * STEP_SEC))
        self.assertEqual(ctrl.samples, 1)

    def test_permanent_reset_complete_iv_history(self) -> None:
        if not __debug__:
            self.skipTest("IvHistory reset assertions require __debug__")
        ctrl = IvHistoryController()
        open_ts = self.OPEN_TS
        for i in range(25):
            ctrl.update(0.20 + i * 0.001, ts=open_ts + i * STEP_SEC)
        ctrl.reset(open_ts + 99_999.0)
        assert_iv_history_reset_complete(ctrl)
        self.assertIsNone(ctrl.last_update_ts)
        ctrl.update(0.22, ts=open_ts + 100_000 * STEP_SEC)
        self.assertEqual(ctrl.samples, 1)

    def test_permanent_reset_complete_token_controllers_reset_all(self) -> None:
        if not __debug__:
            self.skipTest("Reset completeness runtime asserts require __debug__")
        tc = TokenControllers()
        open_ts = self.OPEN_TS
        for i in range(320):
            ts = open_ts + i * STEP_SEC
            update_token_ltp_controllers(tc, 100.0 + i * 0.1, ts=ts)
            update_token_rv_controllers(tc, 100.0 + i * 0.1, ts=ts)
            update_token_iv_controllers(tc, 0.20 + (i % 10) * 0.001, ts=ts)
        reset_ts = open_ts + 99_999.0
        tc.reset_all(reset_ts)
        rolling: list[tuple[str, EmaController]] = [
            ("ema9", tc.ema9),
            ("ema20", tc.ema20),
            ("ema50", tc.ema50),
            ("ema100", tc.ema100),
            ("ema200", tc.ema200),
            ("std20", tc.std20),
            ("rv5m", tc.rv5m),
            ("rv10m", tc.rv10m),
            ("iv_zscore_1m", tc.iv_zscore_1m),
            ("iv_zscore_5m", tc.iv_zscore_5m),
            ("iv_zscore_15m", tc.iv_zscore_15m),
            ("iv_zscore_30m", tc.iv_zscore_30m),
            ("iv_rank_session", tc.iv_session_rank),
        ]
        for label, ctrl in rolling:
            with self.subTest(controller=label):
                assert_rolling_controller_reset_complete(ctrl)
        assert_iv_history_reset_complete(tc.iv_history)
        assert_roll_controller_reset_complete(tc.roll)

    def test_permanent_reset_complete_spot_controllers(self) -> None:
        spot = SpotControllers()
        open_ts = self.OPEN_TS
        for i in range(65):
            spot.update(25_000.0 + i * 0.5, ts=open_ts + i * STEP_SEC)
        spot.reset_all(open_ts + 99_999.0)
        assert_rolling_controller_reset_complete(spot.ema9)
        assert_rolling_controller_reset_complete(spot.ema20)
        assert_rolling_controller_reset_complete(spot.rv5m)
        assert_rolling_controller_reset_complete(spot.rv10m)
        self.assertIsNone(spot.momentum._latest_cross_ts)


REGRESSION_SPOTS_MOMENTUM = [24000.0 + i * 0.5 for i in range(80)]


def _legacy_momentum_formulas(
    *,
    spot: float,
    ts: float,
    ema9: float | None,
    ema20: float | None,
    ema9_1m: float | None,
    latest_cross_ts: float | None,
    latest_cross_price: float | None,
) -> dict[str, float | None]:
    """Controller-owned spot.momentum formulas (Wave 6: % packaging → Interaction)."""
    out: dict[str, float | None] = {
        "ema9_slope": None,
        "ema9_gt_ema20": 0.0,
        "time_since_cross_min": 60.0,
        "cross_age_decay": float(math.exp(-60.0 / 30.0)),
        "price_dist_from_cross_pct": 0.0,
    }
    if ema9 is None or ema20 is None:
        return out

    ema9_f = float(ema9)
    ema20_f = float(ema20)
    out["ema9_gt_ema20"] = 1.0 if ema9_f > ema20_f else 0.0
    if ema9_1m is not None and float(ema9_1m) > 0:
        out["ema9_slope"] = float(100.0 * (ema9_f - float(ema9_1m)) / float(ema9_1m))
    if latest_cross_ts is not None:
        mins = float(max(0.0, (float(ts) - float(latest_cross_ts)) / 60.0))
        out["time_since_cross_min"] = mins
        out["cross_age_decay"] = float(math.exp(-mins / 30.0))
        if latest_cross_price and float(latest_cross_price) > 0:
            cross_price = float(latest_cross_price)
            out["price_dist_from_cross_pct"] = float(
                100.0 * (spot - cross_price) / cross_price,
            )
    return out


SPOT_MOMENTUM_ACTIVE_FEATURES = frozenset({
    "ema9_slope",
    "ema9_gt_ema20",
    "time_since_cross_min",
    "cross_age_decay",
    "price_dist_from_cross_pct",
})


class SpotMomentumRegistryTests(unittest.TestCase):
    def test_defaults_before_ema20_ready(self) -> None:
        spot_ctrl = SpotControllers()
        spot_ctrl.update(24000.0, ts=1000.0)
        feats = spot_ctrl.momentum.emit(spot=24000.0, ts=1000.0)
        self.assertNotIn("spot_vs_ema20_pct", feats)
        self.assertEqual(feats["ema9_gt_ema20"], 0.0)
        self.assertEqual(feats["time_since_cross_min"], 60.0)

    def test_packaging_pct_not_emitted_after_ema20_ready(self) -> None:
        spot_ctrl = SpotControllers()
        open_ts = 1_700_000_000.0
        spots = REGRESSION_SPOTS_MOMENTUM
        for i, spot in enumerate(spots[:20]):
            spot_ctrl.update(spot, ts=open_ts + i * STEP_SEC, grid_step_sec=STEP_SEC)
        ts = open_ts + 19 * STEP_SEC
        spot = spots[19]
        feats = spot_ctrl.momentum.emit(spot=spot, ts=ts)
        self.assertIsNotNone(spot_ctrl.ema20.value())
        self.assertNotIn("spot_vs_ema20_pct", feats)
        self.assertNotIn("ema_spread_pct", feats)
        self.assertNotIn("ema_spread_vs_spot_pct", feats)
        self.assertIn("ema9_gt_ema20", feats)

    def test_parallel_cache_matches_controller(self) -> None:
        open_ts = 1_700_000_000.0
        spots = REGRESSION_SPOTS_MOMENTUM
        timestamps = [open_ts + i * STEP_SEC for i in range(len(spots))]
        tl = TickTimeline()
        for ts, spot in zip(timestamps, spots):
            tl.append(ts, int(round(spot * 100)))
        cache = build_spot_rv_cache(tl, timestamps, grid_step_sec=STEP_SEC)
        spot_ctrl = SpotControllers()
        for i, spot in enumerate(spots):
            spot_ctrl.update(spot, ts=timestamps[i], grid_step_sec=STEP_SEC)
            if i < 19:
                continue
            emitted = spot_ctrl.momentum.emit(spot=spot, ts=timestamps[i])
            cached = {k: cache[timestamps[i]].get(k) for k in emitted}
            for key, ctrl_val in emitted.items():
                self.assertAlmostEqual(ctrl_val, cached.get(key), places=9, msg=key)

    def test_replay_parity_vs_legacy_formulas(self) -> None:
        """Controller emit must match legacy formulas on controller-owned EMA + crossover state."""
        open_ts = 1_700_000_000.0
        spots = REGRESSION_SPOTS_MOMENTUM
        timestamps = [open_ts + i * STEP_SEC for i in range(len(spots))]

        spot_ctrl = SpotControllers()
        for i, spot in enumerate(spots):
            ts = timestamps[i]
            spot_ctrl.update(spot, ts=ts, grid_step_sec=STEP_SEC)
            if not spot_ctrl.ema20.ready():
                continue
            momentum = spot_ctrl.momentum
            ctrl = momentum.emit(spot=spot, ts=ts)
            legacy = _legacy_momentum_formulas(
                spot=spot,
                ts=ts,
                ema9=spot_ctrl.ema9.value(),
                ema20=spot_ctrl.ema20.value(),
                ema9_1m=momentum._ema9_1m_ago(),
                latest_cross_ts=momentum._latest_cross_ts,
                latest_cross_price=momentum._latest_cross_price,
            )
            for key in SPOT_MOMENTUM_ACTIVE_FEATURES:
                c_val = ctrl.get(key)
                l_val = legacy.get(key)
                if c_val is None and l_val is None:
                    continue
                self.assertIsNotNone(c_val, msg=f"{key} at i={i}")
                self.assertIsNotNone(l_val, msg=f"{key} at i={i}")
                self.assertAlmostEqual(float(c_val), float(l_val), places=9, msg=key)

    def test_controller_owned_readiness_includes_momentum_features(self) -> None:
        from chain_replay_ml.dataset_builder.rolling_controllers import (
            CONTROLLER_OWNED_READINESS_FEATURES,
        )

        for feat in SPOT_MOMENTUM_ACTIVE_FEATURES:
            self.assertIn(feat, CONTROLLER_OWNED_READINESS_FEATURES)


ROLL_ACTIVE_FEATURES = frozenset({
    "roll_iv",
    "roll_age_min",
    "rows_since_roll",
    "bs_reiv_pred",
    "dgt_reiv_pred",
    "iv_drift_from_roll",
})


def _manual_roll_preds(
    *,
    roll_iv: float,
    roll_ltp: float,
    roll_spot: float,
    roll_greeks: dict[str, float],
    spot: float,
    roll_fwd_min: float,
    option_type: str,
    strike_rupees: float,
    t_exp: float,
) -> tuple[float, float]:
    from chain_replay_ml import bs

    bs_reiv = max(
        0.0,
        bs.bs_price(option_type, spot, strike_rupees, RISK_FREE_RATE, t_exp, roll_iv),
    )
    dgt_reiv = bs.greek_predicted_ltp(
        roll_ltp,
        roll_greeks,
        spot - roll_spot,
        roll_fwd_min,
        0.0,
    )
    return bs_reiv, dgt_reiv


def _emit_roll_rows(
    samples: list[tuple[float, float, float, float]],
    *,
    gap_after_row: int | None = None,
    option_type: str = "CE",
    strike_rupees: float = 25000.0,
) -> list[dict[str, float | None]]:
    """Simulate roll features (spot, ltp, iv per row; timestamps are monotonic 3s grid)."""
    from chain_replay_ml import bs

    opt_state = OptionFeatureState()
    open_ts = 1_700_000_000.0
    expiry_ts = open_ts + 7 * 86400.0
    out_rows: list[dict[str, float | None]] = []
    last_ts: float | None = None

    for i, (_ts_off, spot, ltp, iv) in enumerate(samples):
        ts = _sim_row_ts(i, open_ts=open_ts, last_ts=last_ts, gap_after_row=gap_after_row)
        if last_ts is not None and row_gap_exceeds(ts, last_ts, GAP_MAX_SEC):
            reset_option_rolling_state(opt_state, ts=ts)
        update_token_roll_controller(
            opt_state.controllers,
            actual_iv=iv,
            spot=spot,
            ltp=ltp,
            ts=ts,
            option_type=option_type,
            strike_rupees=strike_rupees,
            expiry_ts=expiry_ts,
        )
        t_exp = bs.time_to_expiry_years(expiry_ts, ts)
        feats = emit_roll_features(
            opt_state.controllers.roll,
            actual_iv=iv,
            spot=spot,
            ltp=ltp,
            t_exp=t_exp,
            option_type=option_type,
            strike_rupees=strike_rupees,
            ts=ts,
        )
        out_rows.append(dict(feats))
        last_ts = ts
    return out_rows


class RollControllerTests(unittest.TestCase):
    OPEN_TS = 1_700_000_000.0
    EXPIRY_TS = OPEN_TS + 7 * 86400.0
    STRIKE = 25000.0

    def _update_row(
        self,
        ctrl: RollController,
        *,
        iv: float,
        spot: float,
        ltp: float,
        ts: float,
        option_type: str = "CE",
    ) -> dict[str, float | None]:
        from chain_replay_ml import bs

        t_exp = bs.time_to_expiry_years(self.EXPIRY_TS, ts)
        ctrl.update(
            actual_iv=iv,
            spot=spot,
            ltp=ltp,
            ts=ts,
            option_type=option_type,
            strike_rupees=self.STRIKE,
            expiry_ts=self.EXPIRY_TS,
        )
        return emit_roll_features(
            ctrl,
            actual_iv=iv,
            spot=spot,
            ltp=ltp,
            t_exp=t_exp,
            option_type=option_type,
            strike_rupees=self.STRIKE,
            ts=ts,
        )

    def test_first_row_initializes_roll_iv(self) -> None:
        ctrl = RollController()
        ts = self.OPEN_TS
        feats = self._update_row(ctrl, iv=0.22, spot=25000.0, ltp=180.0, ts=ts)
        self.assertTrue(ctrl.ready())
        self.assertAlmostEqual(feats["roll_iv"], 22.0, places=9)
        self.assertEqual(feats["rows_since_roll"], 0.0)
        self.assertIsNotNone(feats["bs_reiv_pred"])
        self.assertIsNotNone(feats["dgt_reiv_pred"])

    def test_rows_before_session_init_null(self) -> None:
        ctrl = RollController()
        feats = emit_roll_features(
            ctrl,
            actual_iv=0.22,
            spot=25000.0,
            ltp=180.0,
            t_exp=0.02,
            option_type="CE",
            strike_rupees=self.STRIKE,
            ts=self.OPEN_TS,
        )
        for feat in ROLL_ACTIVE_FEATURES:
            self.assertIsNone(feats[feat], feat)

    def test_iv_trigger_rolls_anchor(self) -> None:
        ctrl = RollController()
        ts0 = self.OPEN_TS
        self._update_row(ctrl, iv=0.20, spot=25000.0, ltp=180.0, ts=ts0)
        ts1 = ts0 + 3.0
        new_iv = 0.20 * (1.0 + (DEFAULT_IV_THRESHOLD_PCT + 0.1) / 100.0)
        feats = self._update_row(ctrl, iv=new_iv, spot=25000.0, ltp=181.0, ts=ts1)
        self.assertAlmostEqual(feats["roll_iv"], new_iv * 100.0, places=9)
        self.assertEqual(feats["rows_since_roll"], 0.0)

    def test_spot_trigger_rolls_anchor(self) -> None:
        ctrl = RollController()
        ts0 = self.OPEN_TS
        spot0 = 25000.0
        self._update_row(ctrl, iv=0.20, spot=spot0, ltp=180.0, ts=ts0)
        ts1 = ts0 + 3.0
        new_spot = spot0 * (1.0 + (DEFAULT_SPOT_THRESHOLD_PCT + 0.05) / 100.0)
        feats = self._update_row(ctrl, iv=0.20, spot=new_spot, ltp=181.0, ts=ts1)
        self.assertAlmostEqual(feats["roll_iv"], 20.0, places=9)
        self.assertEqual(feats["rows_since_roll"], 0.0)

    def test_time_trigger_rolls_anchor(self) -> None:
        ctrl = RollController()
        ts0 = self.OPEN_TS
        self._update_row(ctrl, iv=0.20, spot=25000.0, ltp=180.0, ts=ts0)
        ts1 = ts0 + DEFAULT_MAX_ROLL_AGE_MIN * 60.0 + 1.0
        new_iv = 0.205
        feats = self._update_row(ctrl, iv=new_iv, spot=25000.0, ltp=182.0, ts=ts1)
        self.assertAlmostEqual(feats["roll_iv"], new_iv * 100.0, places=9)
        self.assertEqual(feats["rows_since_roll"], 0.0)

    def test_rows_since_roll_increments_without_roll(self) -> None:
        ctrl = RollController()
        ts0 = self.OPEN_TS
        self._update_row(ctrl, iv=0.20, spot=25000.0, ltp=180.0, ts=ts0)
        feats1 = self._update_row(ctrl, iv=0.201, spot=25001.0, ltp=180.5, ts=ts0 + 3.0)
        feats2 = self._update_row(ctrl, iv=0.202, spot=25002.0, ltp=181.0, ts=ts0 + 6.0)
        self.assertEqual(feats1["rows_since_roll"], 1.0)
        self.assertEqual(feats2["rows_since_roll"], 2.0)

    def test_bs_dgt_iv_drift_match_manual_calculation(self) -> None:
        from chain_replay_ml import bs

        ctrl = RollController()
        iv0, spot0, ltp0 = 0.22, 25000.0, 185.0
        ts0 = self.OPEN_TS
        self._update_row(ctrl, iv=iv0, spot=spot0, ltp=ltp0, ts=ts0)
        ts1 = ts0 + 120.0
        iv1, spot1, ltp1 = 0.221, 25050.0, 188.0
        feats = self._update_row(ctrl, iv=iv1, spot=spot1, ltp=ltp1, ts=ts1)
        t_exp = bs.time_to_expiry_years(self.EXPIRY_TS, ts1)
        roll_fwd_min = (ts1 - ts0) / 60.0
        manual_bs, manual_dgt = _manual_roll_preds(
            roll_iv=iv0,
            roll_ltp=ltp0,
            roll_spot=spot0,
            roll_greeks=ctrl.roll_greeks,
            spot=spot1,
            roll_fwd_min=roll_fwd_min,
            option_type="CE",
            strike_rupees=self.STRIKE,
            t_exp=t_exp,
        )
        self.assertAlmostEqual(feats["bs_reiv_pred"], manual_bs, places=6)
        self.assertAlmostEqual(feats["dgt_reiv_pred"], manual_dgt, places=6)
        self.assertAlmostEqual(
            feats["iv_drift_from_roll"],
            iv_drift_from_roll_pct(iv1, iv0),
            places=9,
        )

    def test_gap_reset_clears_roll_state(self) -> None:
        from chain_replay_ml import bs

        opt_state = OptionFeatureState()
        ts0 = self.OPEN_TS
        update_token_roll_controller(
            opt_state.controllers,
            actual_iv=0.20,
            spot=25000.0,
            ltp=180.0,
            ts=ts0,
            option_type="CE",
            strike_rupees=self.STRIKE,
            expiry_ts=self.EXPIRY_TS,
        )
        self.assertTrue(opt_state.controllers.roll.ready())
        ts_gap = ts0 + GAP_MAX_SEC + 1.0
        reset_option_rolling_state(opt_state, ts=ts_gap)
        if __debug__:
            assert_roll_controller_reset_complete(opt_state.controllers.roll)
        t_exp = bs.time_to_expiry_years(self.EXPIRY_TS, ts_gap)
        feats_before = emit_roll_features(
            opt_state.controllers.roll,
            actual_iv=0.24,
            spot=25100.0,
            ltp=190.0,
            t_exp=t_exp,
            option_type="CE",
            strike_rupees=self.STRIKE,
            ts=ts_gap,
        )
        self.assertIsNone(feats_before["roll_iv"])
        self.assertIsNone(feats_before["rows_since_roll"])
        update_token_roll_controller(
            opt_state.controllers,
            actual_iv=0.24,
            spot=25100.0,
            ltp=190.0,
            ts=ts_gap,
            option_type="CE",
            strike_rupees=self.STRIKE,
            expiry_ts=self.EXPIRY_TS,
        )
        feats_after = emit_roll_features(
            opt_state.controllers.roll,
            actual_iv=0.24,
            spot=25100.0,
            ltp=190.0,
            t_exp=t_exp,
            option_type="CE",
            strike_rupees=self.STRIKE,
            ts=ts_gap,
        )
        self.assertAlmostEqual(feats_after["roll_iv"], 24.0, places=9)
        self.assertEqual(feats_after["rows_since_roll"], 0.0)

    def test_no_hidden_pre_gap_roll_iv_after_gap(self) -> None:
        pre_iv = 0.20
        post_iv = 0.26
        samples = [
            (0.0, 25000.0, 180.0, pre_iv),
            (0.0, 25001.0, 180.5, pre_iv),
        ] + [(0.0, 25010.0 + i, 181.0 + i * 0.1, post_iv) for i in range(5)]
        rows = _emit_roll_rows(samples, gap_after_row=2)
        self.assertAlmostEqual(rows[2]["roll_iv"], post_iv * 100.0, places=9)
        wrong = pre_iv * 100.0
        self.assertNotAlmostEqual(rows[2]["roll_iv"], wrong, places=6)

    def test_roll_controller_unit_reset(self) -> None:
        if not __debug__:
            self.skipTest("Roll reset assertions require __debug__")
        ctrl = RollController()
        self._update_row(ctrl, iv=0.22, spot=25000.0, ltp=180.0, ts=self.OPEN_TS)
        ctrl.reset(self.OPEN_TS + 99_999.0)
        assert_roll_controller_reset_complete(ctrl)
        self.assertEqual(ctrl.reset_feature_label(), "ROLL")

    def test_controller_owned_readiness_includes_roll_features(self) -> None:
        from chain_replay_ml.dataset_builder.rolling_controllers import (
            CONTROLLER_OWNED_READINESS_FEATURES,
        )

        for feat in ROLL_ACTIVE_FEATURES:
            self.assertIn(feat, CONTROLLER_OWNED_READINESS_FEATURES)

    def test_token_controllers_roll_reset_via_reset_all(self) -> None:
        tc = TokenControllers()
        update_token_roll_controller(
            tc,
            actual_iv=0.22,
            spot=25000.0,
            ltp=180.0,
            ts=self.OPEN_TS,
            option_type="CE",
            strike_rupees=self.STRIKE,
            expiry_ts=self.EXPIRY_TS,
        )
        self.assertTrue(tc.roll.ready())
        tc.reset_all(self.OPEN_TS + 99_999.0)
        if __debug__:
            assert_roll_controller_reset_complete(tc.roll)


def _dgt_row(
    tc: TokenControllers,
    *,
    ts: float,
    iv: float,
    spot: float,
    ltp: float,
    expiry_ts: float,
    strike: float = 25000.0,
) -> dict[str, float | None]:
    from chain_replay_ml import bs

    update_token_roll_controller(
        tc,
        actual_iv=iv,
        spot=spot,
        ltp=ltp,
        ts=ts,
        option_type="CE",
        strike_rupees=strike,
        expiry_ts=expiry_ts,
    )
    t_exp = bs.time_to_expiry_years(expiry_ts, ts)
    roll_feats = emit_roll_features(
        tc.roll,
        actual_iv=iv,
        spot=spot,
        ltp=ltp,
        t_exp=t_exp,
        option_type="CE",
        strike_rupees=strike,
        ts=ts,
    )
    dgt_reiv = roll_feats["dgt_reiv_pred"]
    dgt_feats = emit_dgt_features(tc.dgt, ts=ts, ltp=ltp, dgt_reiv=dgt_reiv, spot=spot)
    update_token_dgt_controller(tc, ts=ts, ltp=ltp, dgt_reiv=dgt_reiv)
    return dgt_feats


class DgtControllerTests(unittest.TestCase):
    OPEN_TS = 1_700_000_000.0
    EXPIRY_TS = OPEN_TS + 7 * 86400.0

    def test_prediction_error_on_first_row(self) -> None:
        tc = TokenControllers()
        feats = _dgt_row(
            tc, ts=self.OPEN_TS, iv=0.22, spot=25000.0, ltp=180.0, expiry_ts=self.EXPIRY_TS,
        )
        self.assertNotIn("dgt_reiv_pred_lag_10s", feats)
        roll_dgt = emit_roll_features(
            tc.roll,
            actual_iv=0.22,
            spot=25000.0,
            ltp=180.0,
            t_exp=0.02,
            option_type="CE",
            strike_rupees=25000.0,
            ts=self.OPEN_TS,
        )["dgt_reiv_pred"]
        self.assertIsNotNone(roll_dgt)
        self.assertAlmostEqual(feats["dgt_prediction_error"], 180.0 - roll_dgt, places=6)

    def test_current_state_features_after_history(self) -> None:
        tc = TokenControllers()
        ts0 = self.OPEN_TS
        _dgt_row(tc, ts=ts0, iv=0.20, spot=25000.0, ltp=180.0, expiry_ts=self.EXPIRY_TS)
        feats = _dgt_row(
            tc, ts=ts0 + 12.0, iv=0.20, spot=25000.0, ltp=181.0, expiry_ts=self.EXPIRY_TS,
        )
        self.assertIn("dgt_prediction_error", feats)
        self.assertIsNotNone(feats["dgt_prediction_error"])
        self.assertNotIn("dgt_reiv_pred_lag_10s", feats)
        self.assertNotIn("dgt_reiv_pred_lag_30s", feats)

    def test_gap_reset_clears_dgt_history(self) -> None:
        tc = TokenControllers()
        ts0 = self.OPEN_TS
        _dgt_row(tc, ts=ts0, iv=0.20, spot=25000.0, ltp=180.0, expiry_ts=self.EXPIRY_TS)
        _dgt_row(tc, ts=ts0 + 3.0, iv=0.20, spot=25001.0, ltp=180.5, expiry_ts=self.EXPIRY_TS)
        tc.reset_all(ts0 + GAP_MAX_SEC + 1.0)
        feats = emit_dgt_features(tc.dgt, ts=ts0 + GAP_MAX_SEC + 1.0, ltp=190.0, dgt_reiv=175.0, spot=25100.0)
        self.assertEqual(feats["dgt_prediction_error"], 15.0)
        # Wave 2: packaged dgt÷ltp / dgt÷spot ratios are Interaction only.
        self.assertNotIn("dgt_reiv_to_ltp_ratio", feats)
        self.assertNotIn("dgt_to_spot_ratio", feats)
        self.assertNotIn("dgt_reiv_pred_lag_10s", feats)
        self.assertNotIn("dgt_prediction_error_lag_10s", feats)

    def test_controller_owned_readiness_includes_dgt_features(self) -> None:
        from chain_replay_ml.dataset_builder.rolling_controllers import (
            CONTROLLER_OWNED_READINESS_FEATURES,
        )

        for feat in DGT_OWNED_FEATURES:
            self.assertIn(feat, CONTROLLER_OWNED_READINESS_FEATURES)


if __name__ == "__main__":
    unittest.main()

"""
Live chain-level context: straddle, PCR, OI walls (matches ``chain_replay_ml.pipeline`` maps).

Computed from subscribed band contracts + tick-ring timelines. Not merged into
ATM-band ``FEATURE_COLUMNS`` yet; exposed for monitoring and future enrichment.
"""
from __future__ import annotations

import bisect
import math
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from research.atm_band_ml.band_evaluator import BandContract, BandEvalContext
from research.atm_band_ml.band_evaluator import strike_step_for_index
from research.atm_band_ml.feature_builder import index_ltp_rupees_at

STRADDLE_ZSCORE_WINDOW = 30
OI_BUILD_LOOKBACK_SEC = 15 * 60
LOCAL_ATM_STRIKE_RADIUS = 5


@dataclass(frozen=True, slots=True)
class ChainContextAtTs:
    """Chain structure metrics at one timestamp."""

    ts: float
    spot: float | None = None
    straddle: float | None = None
    straddle_zscore: float | None = None
    chain_pcr: float | None = None
    atm_pcr: float | None = None
    max_call_oi_dist: float | None = None
    max_put_oi_dist: float | None = None
    max_call_oi_pct: float | None = None
    max_put_oi_pct: float | None = None
    oi_wall_bias: float | None = None
    dist_call_build: float | None = None
    dist_put_build: float | None = None
    pinning_pressure: float | None = None
    straddle_change_5m: float | None = None
    chain_pcr_change_5m: float | None = None
    zscore_change_5m: float | None = None

    @property
    def has_spot(self) -> bool:
        return self.spot is not None and float(self.spot) > 0

    def to_registry_features(self) -> dict[str, float]:
        """Registry model column names (``chain_maps.chain_features_at`` subset)."""
        out: dict[str, float] = {}
        if self.straddle is not None:
            out["atm_straddle"] = float(self.straddle)
        if self.straddle_zscore is not None:
            out["atm_straddle_zscore_30m"] = float(self.straddle_zscore)
        if self.straddle_change_5m is not None:
            out["atm_straddle_change_5m"] = float(self.straddle_change_5m)
        if self.zscore_change_5m is not None:
            out["atm_straddle_zscore_change_5m"] = float(self.zscore_change_5m)
        if self.chain_pcr is not None:
            out["chain_pcr"] = float(self.chain_pcr)
        if self.atm_pcr is not None:
            out["atm_pcr"] = float(self.atm_pcr)
        if self.chain_pcr_change_5m is not None:
            out["chain_pcr_change_5m"] = float(self.chain_pcr_change_5m)
        if self.max_call_oi_dist is not None:
            out["distance_to_max_call_oi_strikes"] = float(self.max_call_oi_dist)
        if self.max_put_oi_dist is not None:
            out["distance_to_max_put_oi_strikes"] = float(self.max_put_oi_dist)
        if self.max_call_oi_pct is not None:
            out["max_call_oi_pct"] = float(self.max_call_oi_pct)
        if self.max_put_oi_pct is not None:
            out["max_put_oi_pct"] = float(self.max_put_oi_pct)
        if self.dist_call_build is not None:
            out["distance_to_call_build_wall"] = float(self.dist_call_build)
        if self.dist_put_build is not None:
            out["distance_to_put_build_wall"] = float(self.dist_put_build)
        if self.oi_wall_bias is not None:
            out["oi_wall_bias"] = float(self.oi_wall_bias)
        if self.pinning_pressure is not None:
            out["pinning_pressure"] = float(self.pinning_pressure)
        return out

    def to_feature_overrides(self) -> dict[str, float | None]:
        """Optional merge into feature dicts (Phase 1 / build_option_rows column names)."""
        return {
            "straddle": self.straddle,
            "straddle_zscore": self.straddle_zscore,
            "distance_to_max_call_oi_strikes": self.max_call_oi_dist,
            "distance_to_max_put_oi_strikes": self.max_put_oi_dist,
            "max_call_oi_pct": self.max_call_oi_pct,
            "max_put_oi_pct": self.max_put_oi_pct,
            "chain_pcr": self.chain_pcr,
            "atm_pcr": self.atm_pcr,
            "oi_wall_bias": self.oi_wall_bias,
            "distance_to_call_build_strike": self.dist_call_build,
            "distance_to_put_build_strike": self.dist_put_build,
            "pinning_pressure": self.pinning_pressure,
            "straddle_change_5m": self.straddle_change_5m,
            "chain_pcr_change_5m": self.chain_pcr_change_5m,
            "zscore_change_5m": self.zscore_change_5m,
        }


@dataclass
class ChainContextMaps:
    """Per-timestamp maps (same keys as backtest ``pipeline.build_option_rows`` inputs)."""

    straddle_map: dict[float, float] = field(default_factory=dict)
    zscore_map: dict[float, float] = field(default_factory=dict)
    max_call_oi_dist_map: dict[float, float] = field(default_factory=dict)
    max_put_oi_dist_map: dict[float, float] = field(default_factory=dict)
    max_call_oi_pct_map: dict[float, float] = field(default_factory=dict)
    max_put_oi_pct_map: dict[float, float] = field(default_factory=dict)
    chain_pcr_map: dict[float, float] = field(default_factory=dict)
    atm_pcr_map: dict[float, float] = field(default_factory=dict)
    oi_wall_bias_map: dict[float, float] = field(default_factory=dict)
    dist_call_build_map: dict[float, float] = field(default_factory=dict)
    dist_put_build_map: dict[float, float] = field(default_factory=dict)
    pinning_pressure_map: dict[float, float] = field(default_factory=dict)

    def snapshot_at(self, ts: float) -> ChainContextAtTs:
        t = float(ts)
        straddle = self.straddle_map.get(t)
        zscore = self.zscore_map.get(t)
        pcr = self.chain_pcr_map.get(t)
        past_straddle = self.straddle_map.get(t - 300.0)
        past_pcr = self.chain_pcr_map.get(t - 300.0)
        past_z = self.zscore_map.get(t - 300.0)
        straddle_chg = (
            float(straddle - past_straddle)
            if straddle is not None and past_straddle is not None
            else None
        )
        pcr_chg = float(pcr - past_pcr) if pcr is not None and past_pcr is not None else None
        z_chg = float(zscore - past_z) if zscore is not None and past_z is not None else None
        return ChainContextAtTs(
            ts=t,
            straddle=straddle,
            straddle_zscore=zscore,
            chain_pcr=pcr,
            atm_pcr=self.atm_pcr_map.get(t),
            max_call_oi_dist=self.max_call_oi_dist_map.get(t),
            max_put_oi_dist=self.max_put_oi_dist_map.get(t),
            max_call_oi_pct=self.max_call_oi_pct_map.get(t),
            max_put_oi_pct=self.max_put_oi_pct_map.get(t),
            oi_wall_bias=self.oi_wall_bias_map.get(t),
            dist_call_build=self.dist_call_build_map.get(t),
            dist_put_build=self.dist_put_build_map.get(t),
            pinning_pressure=self.pinning_pressure_map.get(t),
            straddle_change_5m=straddle_chg,
            chain_pcr_change_5m=pcr_chg,
            zscore_change_5m=z_chg,
        )


def strike_timeline_map(
    ctx: BandEvalContext,
    contracts: Sequence[BandContract],
) -> dict[tuple[float, str], Any]:
    """``(strike, CE|PE)`` → replay ``TickTimeline`` from ring store."""
    out: dict[tuple[float, str], Any] = {}
    for contract in contracts:
        key = contract.key
        if key in out:
            continue
        out[key] = ctx.option_timeline(contract.token)
    return out


def unique_strikes(strike_to_timeline: Mapping[tuple[float, str], Any]) -> list[float]:
    strikes = sorted({float(k[0]) for k in strike_to_timeline})
    return strikes


def _option_ltp_rupees(
    timeline,
    token: str,
    ts: float,
    live_ltps: Mapping[str, float] | None,
) -> float | None:
    if timeline is not None:
        px = timeline.ltp_rupees_at(ts)
        if px is not None and px > 0:
            return float(px)
    if not live_ltps or not token:
        return None
    raw = live_ltps.get(str(token).strip())
    if raw is None:
        return None
    try:
        px = float(raw)
    except (TypeError, ValueError):
        return None
    return px if px > 0 else None


def _blended_straddle(
    spot: float,
    strike_list: Sequence[float],
    strike_to_timeline: Mapping[tuple[float, str], Any],
    ts: float,
    *,
    token_by_key: Mapping[tuple[float, str], str] | None = None,
    live_ltps: Mapping[str, float] | None = None,
) -> float | None:
    if not strike_list:
        return None
    strikes = list(strike_list)
    idx = bisect.bisect_right(strikes, spot)
    if idx == 0:
        k_lower = k_upper = strikes[0]
    elif idx >= len(strikes):
        k_lower = k_upper = strikes[-1]
    else:
        k_lower = strikes[idx - 1]
        k_upper = strikes[idx]

    if k_upper == k_lower:
        w_upper = 0.5
        w_lower = 0.5
    else:
        w_upper = (spot - k_lower) / (k_upper - k_lower)
        w_lower = 1.0 - w_upper

    def _pair_sum(strike: float) -> float | None:
        ce_tl = strike_to_timeline.get((strike, "CE"))
        pe_tl = strike_to_timeline.get((strike, "PE"))
        ce_tok = (token_by_key or {}).get((strike, "CE"), "")
        pe_tok = (token_by_key or {}).get((strike, "PE"), "")
        l_ce = _option_ltp_rupees(ce_tl, str(ce_tok), ts, live_ltps)
        l_pe = _option_ltp_rupees(pe_tl, str(pe_tok), ts, live_ltps)
        if l_ce is None or l_pe is None:
            return None
        return float(l_ce + l_pe)

    lower = _pair_sum(k_lower)
    upper = _pair_sum(k_upper)
    if lower is not None and upper is not None:
        return w_lower * lower + w_upper * upper
    if lower is not None:
        return lower
    if upper is not None:
        return upper
    return None


def _zscore_from_window(window: deque[float]) -> float:
    if len(window) < 2:
        return 0.0
    values = list(window)
    mean = sum(values) / len(values)
    variance = max(0.0, sum(v * v for v in values) / len(values) - mean * mean)
    std = math.sqrt(variance)
    latest = values[-1]
    return float((latest - mean) / std) if std > 1e-6 else 0.0


def compute_chain_context_at_ts(
    *,
    ts: float,
    index_timeline,
    strike_to_timeline: Mapping[tuple[float, str], Any],
    strike_step: int,
    straddle_window: deque[float] | None = None,
    token_by_key: Mapping[tuple[float, str], str] | None = None,
    live_ltps: Mapping[str, float] | None = None,
) -> ChainContextAtTs:
    """
    Compute chain PCR / OI walls / blended straddle at one timestamp.

    ``straddle_window`` — optional rolling deque (max 30) for z-score; updated in place
    when a new blended straddle is computed.
    """
    target_ts = float(ts)
    spot = index_ltp_rupees_at(index_timeline, target_ts)
    if spot is None or spot <= 0:
        return ChainContextAtTs(ts=target_ts, spot=spot)

    strikes = unique_strikes(strike_to_timeline)
    if not strikes:
        return ChainContextAtTs(ts=target_ts, spot=spot)

    blended = _blended_straddle(
        spot,
        strikes,
        strike_to_timeline,
        target_ts,
        token_by_key=token_by_key,
        live_ltps=live_ltps,
    )
    zscore: float | None = None
    if blended is not None and straddle_window is not None:
        if len(straddle_window) == STRADDLE_ZSCORE_WINDOW:
            straddle_window.popleft()
        straddle_window.append(blended)
        zscore = _zscore_from_window(straddle_window)

    total_call_oi = 0
    total_put_oi = 0
    max_call_oi = -1
    max_call_strike: float | None = None
    max_put_oi = -1
    max_put_strike: float | None = None
    call_builds: list[tuple[int, float]] = []
    put_builds: list[tuple[int, float]] = []

    nearest_atm = min(strikes, key=lambda s: abs(s - spot))
    atm_idx = strikes.index(nearest_atm)
    start_idx = max(0, atm_idx - LOCAL_ATM_STRIKE_RADIUS)
    end_idx = min(len(strikes) - 1, atm_idx + LOCAL_ATM_STRIKE_RADIUS)
    local_atm_strikes = set(strikes[start_idx : end_idx + 1])
    local_call_oi = 0
    local_put_oi = 0

    for strike_r in strikes:
        ce_tl = strike_to_timeline.get((strike_r, "CE"))
        pe_tl = strike_to_timeline.get((strike_r, "PE"))

        if ce_tl is not None:
            oi_ce = ce_tl.oi_at(target_ts)
            if oi_ce is not None and oi_ce > 0:
                total_call_oi += int(oi_ce)
                if oi_ce > max_call_oi:
                    max_call_oi = int(oi_ce)
                    max_call_strike = strike_r
                oi_ce_past = ce_tl.oi_at(target_ts - OI_BUILD_LOOKBACK_SEC)
                if oi_ce_past is not None:
                    build_ce = int(oi_ce) - int(oi_ce_past)
                    if build_ce > 0:
                        call_builds.append((build_ce, strike_r))
                if strike_r in local_atm_strikes:
                    local_call_oi += int(oi_ce)

        if pe_tl is not None:
            oi_pe = pe_tl.oi_at(target_ts)
            if oi_pe is not None and oi_pe > 0:
                total_put_oi += int(oi_pe)
                if oi_pe > max_put_oi:
                    max_put_oi = int(oi_pe)
                    max_put_strike = strike_r
                oi_pe_past = pe_tl.oi_at(target_ts - OI_BUILD_LOOKBACK_SEC)
                if oi_pe_past is not None:
                    build_pe = int(oi_pe) - int(oi_pe_past)
                    if build_pe > 0:
                        put_builds.append((build_pe, strike_r))
                if strike_r in local_atm_strikes:
                    local_put_oi += int(oi_pe)

    chain_pcr = float(total_put_oi / total_call_oi) if total_call_oi > 0 else None
    atm_pcr = float(local_put_oi / local_call_oi) if local_call_oi > 0 else None

    max_call_oi_dist: float | None = None
    max_put_oi_dist: float | None = None
    max_call_oi_pct: float | None = None
    max_put_oi_pct: float | None = None
    oi_wall_bias: float | None = None
    pinning_pressure: float | None = None
    dist_call_build: float | None = None
    dist_put_build: float | None = None

    step = max(1, int(strike_step))
    if max_call_strike is not None:
        max_call_oi_dist = float((spot - max_call_strike) / step)
        if total_call_oi > 0:
            max_call_oi_pct = float(max_call_oi / total_call_oi)
    if max_put_strike is not None:
        max_put_oi_dist = float((spot - max_put_strike) / step)
        if total_put_oi > 0:
            max_put_oi_pct = float(max_put_oi / total_put_oi)
    if max_call_strike is not None and max_put_strike is not None:
        oi_wall_bias = float(max_put_oi_dist - max_call_oi_dist)  # type: ignore[operator]
        pinning_pressure = float(abs(max_call_oi_dist) + abs(max_put_oi_dist))  # type: ignore[arg-type]

    if call_builds:
        best_call_strike = max(call_builds, key=lambda x: x[0])[1]
        dist_call_build = float((spot - best_call_strike) / step)
    elif max_call_oi_dist is not None:
        dist_call_build = max_call_oi_dist

    if put_builds:
        best_put_strike = max(put_builds, key=lambda x: x[0])[1]
        dist_put_build = float((spot - best_put_strike) / step)
    elif max_put_oi_dist is not None:
        dist_put_build = max_put_oi_dist

    return ChainContextAtTs(
        ts=target_ts,
        spot=spot,
        straddle=blended,
        straddle_zscore=zscore,
        chain_pcr=chain_pcr,
        atm_pcr=atm_pcr,
        max_call_oi_dist=max_call_oi_dist,
        max_put_oi_dist=max_put_oi_dist,
        max_call_oi_pct=max_call_oi_pct,
        max_put_oi_pct=max_put_oi_pct,
        oi_wall_bias=oi_wall_bias,
        dist_call_build=dist_call_build,
        dist_put_build=dist_put_build,
        pinning_pressure=pinning_pressure,
    )


@dataclass
class ChainContextBuilder:
    """
    Stateful builder across a 10s (or minute) grid — fills maps + 5m deltas.

    Straddle z-score uses a 30-sample rolling window (one sample per processed
    timestamp; match backtest by passing minute-aligned grid_times only).
    """

    strike_step: int = 50
    maps: ChainContextMaps = field(default_factory=ChainContextMaps)
    _straddle_window: deque[float] = field(default_factory=deque, repr=False)

    def update(
        self,
        ts: float,
        index_timeline,
        strike_to_timeline: Mapping[tuple[float, str], Any],
        *,
        token_by_key: Mapping[tuple[float, str], str] | None = None,
        live_ltps: Mapping[str, float] | None = None,
    ) -> ChainContextAtTs:
        snap = compute_chain_context_at_ts(
            ts=ts,
            index_timeline=index_timeline,
            strike_to_timeline=strike_to_timeline,
            strike_step=self.strike_step,
            straddle_window=self._straddle_window,
            token_by_key=token_by_key,
            live_ltps=live_ltps,
        )
        t = float(ts)
        if snap.straddle is not None:
            self.maps.straddle_map[t] = snap.straddle
        if snap.straddle_zscore is not None:
            self.maps.zscore_map[t] = snap.straddle_zscore
        if snap.chain_pcr is not None:
            self.maps.chain_pcr_map[t] = snap.chain_pcr
        if snap.atm_pcr is not None:
            self.maps.atm_pcr_map[t] = snap.atm_pcr
        if snap.max_call_oi_dist is not None:
            self.maps.max_call_oi_dist_map[t] = snap.max_call_oi_dist
        if snap.max_put_oi_dist is not None:
            self.maps.max_put_oi_dist_map[t] = snap.max_put_oi_dist
        if snap.max_call_oi_pct is not None:
            self.maps.max_call_oi_pct_map[t] = snap.max_call_oi_pct
        if snap.max_put_oi_pct is not None:
            self.maps.max_put_oi_pct_map[t] = snap.max_put_oi_pct
        if snap.oi_wall_bias is not None:
            self.maps.oi_wall_bias_map[t] = snap.oi_wall_bias
        if snap.dist_call_build is not None:
            self.maps.dist_call_build_map[t] = snap.dist_call_build
        if snap.dist_put_build is not None:
            self.maps.dist_put_build_map[t] = snap.dist_put_build
        if snap.pinning_pressure is not None:
            self.maps.pinning_pressure_map[t] = snap.pinning_pressure

        enriched = self.maps.snapshot_at(t)
        enriched = ChainContextAtTs(
            ts=t,
            spot=snap.spot,
            straddle=snap.straddle,
            straddle_zscore=snap.straddle_zscore,
            chain_pcr=snap.chain_pcr,
            atm_pcr=snap.atm_pcr,
            max_call_oi_dist=snap.max_call_oi_dist,
            max_put_oi_dist=snap.max_put_oi_dist,
            max_call_oi_pct=snap.max_call_oi_pct,
            max_put_oi_pct=snap.max_put_oi_pct,
            oi_wall_bias=snap.oi_wall_bias,
            dist_call_build=snap.dist_call_build,
            dist_put_build=snap.dist_put_build,
            pinning_pressure=snap.pinning_pressure,
            straddle_change_5m=enriched.straddle_change_5m,
            chain_pcr_change_5m=enriched.chain_pcr_change_5m,
            zscore_change_5m=enriched.zscore_change_5m,
        )
        return enriched


def build_chain_context_over_grid(
    ctx: BandEvalContext,
    contracts: Sequence[BandContract],
    grid_times: Sequence[float],
) -> ChainContextBuilder:
    """Evaluate chain context on each grid timestamp; returns filled builder."""
    strike_to_tl = strike_timeline_map(ctx, contracts)
    index_tl = ctx.index_timeline()
    step = ctx.strike_step or strike_step_for_index(ctx.index_key)
    builder = ChainContextBuilder(strike_step=step)
    for ts in grid_times:
        builder.update(float(ts), index_tl, strike_to_tl)
    return builder


def _token_map_for_contracts(contracts: Sequence[BandContract]) -> dict[tuple[float, str], str]:
    return {c.key: str(c.token) for c in contracts}


def chain_context_for_band(
    ctx: BandEvalContext,
    contracts: Sequence[BandContract],
    ts: float,
    *,
    builder: ChainContextBuilder | None = None,
    live_ltps: Mapping[str, float] | None = None,
) -> ChainContextAtTs:
    """Single-slot chain context using band contracts (optional shared builder)."""
    strike_to_tl = strike_timeline_map(ctx, contracts)
    index_tl = ctx.index_timeline()
    token_by_key = _token_map_for_contracts(contracts)
    if builder is not None:
        return builder.update(
            float(ts),
            index_tl,
            strike_to_tl,
            token_by_key=token_by_key,
            live_ltps=live_ltps,
        )
    return compute_chain_context_at_ts(
        ts=float(ts),
        index_timeline=index_tl,
        strike_to_timeline=strike_to_tl,
        strike_step=ctx.strike_step,
        token_by_key=token_by_key,
        live_ltps=live_ltps,
    )


def enrich_features_with_chain_context(
    features: dict[str, Any],
    chain: ChainContextAtTs,
) -> dict[str, Any]:
    """Non-destructive merge of chain metrics into a feature row."""
    out = dict(features)
    for key, val in chain.to_feature_overrides().items():
        if val is not None:
            out[key] = val
    for key, val in chain.to_registry_features().items():
        if val is not None:
            out[key] = val
    return out

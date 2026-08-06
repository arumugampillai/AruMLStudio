"""
Live registry features via the dataset builder pipeline (single source of truth).

Reuses extract_timeline_features → enrich_dataset_features → enrich_with_chain_maps
→ pick_features_from_row — same as ``dataset_builder.stages.build_day_rows`` Stage 6.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Mapping, Sequence
from zoneinfo import ZoneInfo

from research.atm_band_ml.band_evaluator import BandContract, BandEvalContext, index_ltp_rupees_at
from research.atm_band_ml.feature_builder import ema_inputs_at_ts

IST = ZoneInfo("Asia/Kolkata")

LOOKBACK_START_SEC = 60.0
GRID_STEP_SEC = 10.0


def default_enabled_groups() -> list[str]:
    from chain_replay_ml.dataset_builder.feature_plugins import GROUP_FEATURE_SOURCES

    return list(GROUP_FEATURE_SOURCES.keys())


def session_grid_timestamps(open_ts: float, through_ts: float, *, step_sec: float = GRID_STEP_SEC) -> list[float]:
    start = float(open_ts) + LOOKBACK_START_SEC
    end = float(through_ts)
    if end < start:
        return []
    out: list[float] = []
    t = start
    while t <= end + 0.001:
        out.append(t)
        t += float(step_sec)
    return out


def strike_mapping_from_contracts(
    band_ctx: BandEvalContext,
    contracts: Sequence[BandContract],
) -> dict[tuple[float, str], tuple[str, str, Any]]:
    mapping: dict[tuple[float, str], tuple[str, str, Any]] = {}
    for contract in contracts:
        key = (float(contract.strike), str(contract.option_type).upper())
        if key in mapping:
            continue
        opt_tl = band_ctx.option_timeline(contract.token)
        mapping[key] = (str(contract.token), str(contract.symbol or ""), opt_tl)
    return mapping


@dataclass
class LiveRegistrySession:
    """Stateful live session — OptionFeatureState per token + chain maps cache."""

    _opt_states: dict[str, Any] = field(default_factory=dict)
    _last_advanced: dict[str, float] = field(default_factory=dict)
    _chain_maps: Any = None
    _chain_maps_through: float = 0.0
    _mapping_key: tuple[tuple[float, str, str], ...] = ()

    def reset(self) -> None:
        self._opt_states.clear()
        self._last_advanced.clear()
        self._chain_maps = None
        self._chain_maps_through = 0.0
        self._mapping_key = ()

    def _mapping_cache_key(
        self, strike_mapping: Mapping[tuple[float, str], tuple[str, str, Any]]
    ) -> tuple[tuple[float, str, str], ...]:
        return tuple(sorted((k[0], k[1], v[0]) for k, v in strike_mapping.items()))

    def ensure_chain_maps(
        self,
        *,
        band_ctx: BandEvalContext,
        index_tl: Any,
        strike_mapping: Mapping[tuple[float, str], tuple[str, str, Any]],
        signal_ts: float,
    ) -> Any:
        from chain_replay_ml.dataset_builder.chain_maps import precompute_chain_maps

        key = self._mapping_cache_key(strike_mapping)
        if (
            self._chain_maps is None
            or key != self._mapping_key
            or float(signal_ts) > float(self._chain_maps_through)
        ):
            ts_list = session_grid_timestamps(band_ctx.open_ts, signal_ts)
            self._chain_maps = precompute_chain_maps(
                index_tl=index_tl,
                strike_mapping=dict(strike_mapping),
                timestamps=ts_list,
                strike_step=int(band_ctx.strike_step),
            )
            self._chain_maps_through = float(signal_ts)
            self._mapping_key = key
        return self._chain_maps

    def advance_opt_state(
        self,
        *,
        token: str,
        band_ctx: BandEvalContext,
        contract: BandContract,
        index_tl: Any,
        opt_tl: Any,
        signal_ts: float,
        trading_day: str,
        expiry_norm: str,
    ) -> Any:
        from chain_replay_ml.dataset_builder.extended_features import OptionFeatureState, enrich_dataset_features
        from chain_replay_ml.features_atm_band import extract_timeline_features, find_atm_strike

        tok = str(token).strip()
        opt_state = self._opt_states.get(tok)
        if opt_state is None:
            opt_state = OptionFeatureState()
            self._opt_states[tok] = opt_state

        last = float(self._last_advanced.get(tok, band_ctx.open_ts))
        if float(signal_ts) <= last + 0.001:
            return opt_state

        ema_ctx = band_ctx.ema_ctx or {}
        for t in session_grid_timestamps(band_ctx.open_ts, signal_ts):
            if t <= last + 0.001:
                continue
            spot = index_ltp_rupees_at(index_tl, t)
            if spot is None or spot <= 0:
                continue
            atm = find_atm_strike(float(spot), int(band_ctx.strike_step))
            ema_in = ema_inputs_at_ts(t, float(spot), ema_ctx, band_ctx.open_ts)
            raw = extract_timeline_features(
                ts=t,
                index_timeline=index_tl,
                option_timeline=opt_tl,
                option_type=str(contract.option_type).upper(),
                strike_rupees=float(contract.strike),
                atm_strike_price=atm,
                expiry_ts=float(band_ctx.expiry_ts),
                ema9_now=ema_in.get("ema9_now"),
                ema20_now=ema_in.get("ema20_now"),
                ema9_1m_ago=ema_in.get("ema9_1m_ago"),
                ema9_gt_ema20=ema_in.get("ema9_gt_ema20"),
                ema_spread_vs_spot_pct=ema_in.get("ema_spread_vs_spot_pct"),
                time_since_cross_min=ema_in.get("time_since_cross_min"),
                price_dist_from_cross=ema_in.get("price_dist_from_cross"),
                open_ts=float(band_ctx.open_ts),
                close_ts=float(band_ctx.close_ts),
            )
            raw["option_type"] = str(contract.option_type).upper()
            enrich_dataset_features(
                raw,
                ts=t,
                option_timeline=opt_tl,
                index_timeline=index_tl,
                option_type=str(contract.option_type).upper(),
                strike_rupees=float(contract.strike),
                atm_strike=atm,
                strike_step=int(band_ctx.strike_step),
                expiry_ts=float(band_ctx.expiry_ts),
                open_ts=float(band_ctx.open_ts),
                close_ts=float(band_ctx.close_ts),
                trading_day=str(trading_day),
                expiry_norm=str(expiry_norm),
                opt_state=opt_state,
            )
        self._last_advanced[tok] = float(signal_ts)
        return opt_state


def _session_labels(state: Any) -> tuple[str, str]:
    trading_day = datetime.now(tz=IST).strftime("%Y-%m-%d")
    expiry_norm = ""
    top = getattr(state, "top_menu", None)
    if top is not None:
        try:
            exp = top.expiry_var.get() if hasattr(top, "expiry_var") else ""
            expiry_norm = str(exp or "").strip()
        except Exception:
            expiry_norm = ""
    return trading_day, expiry_norm


def build_live_registry_features(
    *,
    session: LiveRegistrySession,
    band_ctx: BandEvalContext,
    contract: BandContract,
    signal_ts: float,
    contracts: Sequence[BandContract],
    enabled_groups: Sequence[str] | None = None,
    state: Any = None,
) -> dict[str, Any]:
    """Build full registry feature dict for one band contract at ``signal_ts``."""
    from chain_replay_ml.dataset_builder.registry_features import build_registry_features_at_ts
    from chain_replay_ml.features_atm_band import find_atm_strike

    groups = list(enabled_groups or default_enabled_groups())
    target_ts = float(signal_ts)
    index_tl = band_ctx.index_timeline(target_ts)
    spot = index_ltp_rupees_at(index_tl, target_ts)
    if spot is None or spot <= 0:
        return {}

    strike_mapping = strike_mapping_from_contracts(band_ctx, contracts)
    chain_maps = session.ensure_chain_maps(
        band_ctx=band_ctx,
        index_tl=index_tl,
        strike_mapping=strike_mapping,
        signal_ts=target_ts,
    )
    opt_tl = band_ctx.option_timeline(contract.token)
    trading_day, expiry_norm = _session_labels(state)

    opt_state = session.advance_opt_state(
        token=str(contract.token),
        band_ctx=band_ctx,
        contract=contract,
        index_tl=index_tl,
        opt_tl=opt_tl,
        signal_ts=target_ts,
        trading_day=trading_day,
        expiry_norm=expiry_norm,
    )

    atm = find_atm_strike(float(spot), int(band_ctx.strike_step))

    return build_registry_features_at_ts(
        ts=target_ts,
        strike=float(contract.strike),
        option_type=str(contract.option_type),
        opt_tl=opt_tl,
        index_tl=index_tl,
        strike_mapping=strike_mapping,
        chain_maps=chain_maps,
        opt_state=opt_state,
        strike_step=int(band_ctx.strike_step),
        expiry_ts=float(band_ctx.expiry_ts),
        open_ts=float(band_ctx.open_ts),
        close_ts=float(band_ctx.close_ts),
        enabled_groups=groups,
        trading_day=trading_day,
        expiry_norm=expiry_norm,
        atm_strike=atm,
    )


def build_legacy_live_features(row: Any) -> dict[str, Any]:
    """Previous live path: band row extractor keys only (for parity diff)."""
    return dict(getattr(getattr(row, "result", None), "features", None) or {})

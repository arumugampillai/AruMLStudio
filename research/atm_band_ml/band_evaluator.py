"""
ATM±N band evaluator on the live 10s ML signal grid.

For each grid timestamp: spot → ATM → 21 strikes × CE/PE → feature rows
(Phase 2 ``build_strike_features``). Read-only on ``AppState`` / chain trees.
"""
from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

from api.tick_ring import TickRingStore
from app.market_session import market_session_bounds_today, ml_signal_evaluation_times
from app.state import AppState
from research.atm_band_ml.chart_tick_client import ChartTickClient
from research.atm_band_ml.feature_builder import (
    FeatureBuildResult,
    build_strike_features,
    index_ltp_rupees_at,
    is_feature_window_complete,
    missing_5s_buckets,
    precompute_ema_context,
)
from research.atm_band_ml.feature_window import feature_window_diagnostics
from research.atm_band_ml.tick_timeline import (
    index_timeline_with_chart,
    option_timeline_with_chart_volume,
    ring_to_tick_timeline,
)

_CHART_DIR = Path(__file__).resolve().parents[2] / "apps"

DEFAULT_BAND_SIZE = 10
DEFAULT_BAND_SIZE_STRIKES = DEFAULT_BAND_SIZE * 2 + 1  # 21


def _ensure_chart_on_path() -> None:
    chart_dir = str(_CHART_DIR)
    if chart_dir not in sys.path:
        sys.path.insert(0, chart_dir)


def _band_helpers():
    _ensure_chart_on_path()
    from chain_replay_ml.features_atm_band import find_atm_strike, select_atm_band_strikes

    return find_atm_strike, select_atm_band_strikes


def strike_step_for_index(index: str) -> int:
    _ensure_chart_on_path()
    from chain_replay_ml.constants import STRIKE_STEP_BY_INDEX

    return int(STRIKE_STEP_BY_INDEX.get(str(index or "").upper(), 50))


def index_ring_key(index: str) -> str:
    key = str(index or "NIFTY").strip().upper()
    if key == "NIFTY50":
        return "NIFTY"
    return key


def band_grid_times(
    open_ts: float | None = None,
    close_ts: float | None = None,
) -> list[float]:
    """10s evaluation grid (same as live ML engine / backtest signal grid)."""
    return ml_signal_evaluation_times(open_ts, close_ts)


def atm_band_strikes(
    spot: float,
    *,
    strike_step: int,
    band_size: int = DEFAULT_BAND_SIZE,
) -> list[int]:
    find_atm_strike, select_atm_band_strikes = _band_helpers()
    atm = find_atm_strike(float(spot), int(strike_step))
    return select_atm_band_strikes(atm, int(strike_step), band_size=band_size)


@dataclass(frozen=True, slots=True)
class BandContract:
    strike: float
    option_type: str
    token: str
    symbol: str = ""

    @property
    def key(self) -> tuple[float, str]:
        return (float(self.strike), str(self.option_type).upper())


@dataclass
class BandEvalRow:
    ts: float
    contract: BandContract
    result: FeatureBuildResult
    atm_strike: int | None = None
    spot: float | None = None

    @property
    def model_complete(self) -> bool:
        return bool(self.result.model_complete)


@dataclass
class BandEvalSnapshot:
    ts: float
    spot: float | None
    atm_strike: int | None
    band_strikes: tuple[int, ...]
    rows: list[BandEvalRow] = field(default_factory=list)
    window_complete: bool = False
    skipped_missing_contracts: int = 0

    @property
    def row_count(self) -> int:
        return len(self.rows)

    @property
    def model_complete_count(self) -> int:
        return sum(1 for r in self.rows if r.model_complete)


def index_chart_token(index: str) -> str:
    """Angel chart-server token for NIFTY / SENSEX index spot."""
    from api.chain_quote import ANGEL_INDEX_SPOT

    key = index_ring_key(index)
    meta = ANGEL_INDEX_SPOT.get(key)
    if not meta:
        return ""
    return str(meta.get("token") or "").strip()


def index_ring_store_key(index: str) -> str:
    """Key used in ``tick_ring_store`` (Angel token); Neo feed may use index label."""
    tok = index_chart_token(index)
    if tok:
        return tok
    return index_ring_key(index)


@dataclass
class BandEvalContext:
    """Reusable caches for one evaluation batch (same session / ring snapshot)."""

    store: TickRingStore
    index_key: str
    open_ts: float
    close_ts: float
    expiry_ts: float
    strike_step: int = 50
    band_size: int = DEFAULT_BAND_SIZE
    ema_ctx: dict[str, Any] | None = None
    missing_buckets: set[float] | None = None
    chart_client: ChartTickClient | None = None
    use_chart_volume: bool = True
    _index_tl: Any = field(default=None, repr=False)
    _index_tl_source: str = field(default="ring", repr=False)
    _opt_tls: dict[str, Any] = field(default_factory=dict, repr=False)
    _chart_fetch_ts: dict[str, float] = field(default_factory=dict, repr=False)

    def index_timeline(self, as_of_ts: float | None = None):
        """Index timeline: chart-server session ticks + ring overlay (live)."""
        chart_tok = index_chart_token(self.index_key)
        chart_payload = None
        if self.use_chart_volume and chart_tok:
            chart_payload = self._chart_ticks_payload(chart_tok)
        ring_tl = ring_to_tick_timeline(self.store, index_ring_store_key(self.index_key))
        if chart_payload:
            index_tl = index_timeline_with_chart(
                self.store,
                self.index_key,
                chart_payload,
                open_ts=self.open_ts,
                close_ts=self.close_ts,
            )
            self._index_tl_source = (
                "chart+ring" if index_tl.timestamps and ring_tl.timestamps else "chart"
            )
        else:
            index_tl = ring_tl
            self._index_tl_source = "ring"
        self._index_tl = index_tl
        if self._index_tl.timestamps:
            self.ema_ctx = precompute_ema_context(
                self._index_tl, self.open_ts, self.close_ts
            )
        as_of = float(as_of_ts if as_of_ts is not None else self.close_ts)
        self.missing_buckets = missing_5s_buckets(
            self.open_ts,
            self.close_ts,
            self._index_tl,
            as_of_ts=as_of,
        )
        return self._index_tl

    def _chart_ticks_payload(self, token: str) -> dict[str, Any] | None:
        client = self.chart_client
        if client is None or not self.use_chart_volume:
            return None
        tok = str(token).strip()
        if not tok:
            return None
        return client.fetch_ticks(tok)

    def option_timeline(self, token: str):
        tok = str(token).strip()
        now = time.time()
        fetched_at = self._chart_fetch_ts.get(tok)
        client = self.chart_client
        ttl = client.cache_ttl_sec if client is not None else 30.0
        if fetched_at is not None and (now - fetched_at) >= ttl:
            self._opt_tls.pop(tok, None)
            self._chart_fetch_ts.pop(tok, None)

        tl = self._opt_tls.get(tok)
        if tl is not None:
            return tl

        chart_payload = self._chart_ticks_payload(tok)
        tl = option_timeline_with_chart_volume(
            self.store,
            tok,
            chart_payload,
            open_ts=self.open_ts,
            close_ts=self.close_ts,
        )
        self._opt_tls[tok] = tl
        self._chart_fetch_ts[tok] = now
        return tl


def band_contracts_from_trees(ce_tree, pe_tree) -> list[BandContract]:
    """Parse subscribed CE/PE chain trees into band contracts."""
    from ui.option_chain_columns import COL_STRIKE, COL_SYMBOL, COL_TOKEN, COL_TSYM, COL_TYPE

    out: list[BandContract] = []
    seen: set[tuple[float, str]] = set()
    for tree, default_type in ((ce_tree, "CE"), (pe_tree, "PE")):
        if tree is None:
            continue
        for iid in tree.get_children():
            values = tree.item(iid, "values") or ()
            if len(values) <= COL_TOKEN:
                continue
            opt_type = str(values[COL_TYPE] or default_type).strip().upper() or default_type
            token = str(values[COL_TOKEN]).strip()
            if not token:
                continue
            try:
                strike = float(values[COL_STRIKE])
            except (TypeError, ValueError):
                continue
            key = (strike, opt_type)
            if key in seen:
                continue
            seen.add(key)
            symbol = ""
            if len(values) > COL_TSYM and values[COL_TSYM]:
                symbol = str(values[COL_TSYM]).strip()
            if not symbol and len(values) > COL_SYMBOL:
                symbol = str(values[COL_SYMBOL]).strip()
            out.append(
                BandContract(
                    strike=strike,
                    option_type=opt_type,
                    token=token,
                    symbol=symbol,
                )
            )
    return out


def band_contracts_from_state(state: AppState) -> list[BandContract]:
    return band_contracts_from_trees(state.ce_tree, state.pe_tree)


def contracts_in_band(
    contracts: Sequence[BandContract],
    band_strikes: Sequence[int],
) -> list[BandContract]:
    strike_set = {int(s) for s in band_strikes}
    picked: list[BandContract] = []
    seen: set[tuple[float, str]] = set()
    for c in contracts:
        strike_i = int(round(float(c.strike)))
        if strike_i not in strike_set:
            continue
        if c.key in seen:
            continue
        seen.add(c.key)
        picked.append(c)
    return picked


def evaluate_band_at_ts(
    ctx: BandEvalContext,
    ts: float,
    contracts: Sequence[BandContract],
    *,
    probe: bool = False,
) -> BandEvalSnapshot:
    """Evaluate all in-band CE/PE contracts at one 10s grid timestamp."""
    find_atm_strike, _ = _band_helpers()
    target_ts = float(ts)
    index_tl = ctx.index_timeline(target_ts)
    window_ok = is_feature_window_complete(
        target_ts,
        index_tl,
        missing_buckets=ctx.missing_buckets,
        open_ts=ctx.open_ts,
        close_ts=ctx.close_ts,
        as_of_ts=target_ts,
        live=True,
    )
    spot = index_ltp_rupees_at(index_tl, target_ts)
    atm: int | None = None
    band: list[int] = []
    if spot is not None and spot > 0:
        atm = find_atm_strike(spot, ctx.strike_step)
        _, select_band = _band_helpers()
        band = select_band(atm, ctx.strike_step, band_size=ctx.band_size)

    snap = BandEvalSnapshot(
        ts=target_ts,
        spot=spot,
        atm_strike=atm,
        band_strikes=tuple(band),
        window_complete=window_ok,
    )
    if spot is None or spot <= 0 or not band:
        return snap
    if not window_ok and not probe:
        return snap

    in_band = contracts_in_band(contracts, band)
    present = {(int(round(c.strike)), c.option_type) for c in in_band}
    expected = len(band) * 2
    snap.skipped_missing_contracts = max(0, expected - len(present))

    for contract in in_band:
        opt_tl = ctx.option_timeline(contract.token)
        result = build_strike_features(
            ts=target_ts,
            index_timeline=index_tl,
            option_timeline=opt_tl,
            option_type=contract.option_type,
            strike_rupees=contract.strike,
            atm_strike_price=atm,
            expiry_ts=ctx.expiry_ts,
            open_ts=ctx.open_ts,
            close_ts=ctx.close_ts,
            ema_ctx=ctx.ema_ctx,
            missing_buckets=ctx.missing_buckets,
            strike_step=ctx.strike_step,
            live=True,
            allow_partial_window=probe,
        )
        snap.rows.append(
            BandEvalRow(
                ts=target_ts,
                contract=contract,
                result=result,
                atm_strike=atm,
                spot=spot,
            )
        )
    return snap


def evaluate_band_grid(
    ctx: BandEvalContext,
    contracts: Sequence[BandContract],
    *,
    grid_times: Sequence[float] | None = None,
    max_slots: int | None = None,
) -> list[BandEvalSnapshot]:
    """Evaluate ATM band across the session 10s grid (or a provided time list)."""
    times = list(grid_times if grid_times is not None else band_grid_times(ctx.open_ts, ctx.close_ts))
    if max_slots is not None:
        times = times[: int(max_slots)]
    return [evaluate_band_at_ts(ctx, t, contracts) for t in times]


def iter_model_complete_rows(
    snapshots: Sequence[BandEvalSnapshot],
) -> Iterator[BandEvalRow]:
    for snap in snapshots:
        for row in snap.rows:
            if row.model_complete:
                yield row


def band_eval_summary(snapshots: Sequence[BandEvalSnapshot]) -> dict[str, Any]:
    total_rows = sum(s.row_count for s in snapshots)
    model_complete = sum(s.model_complete_count for s in snapshots)
    grid_with_spot = sum(1 for s in snapshots if s.spot is not None and s.spot > 0)
    return {
        "grid_slots": len(snapshots),
        "grid_with_spot": grid_with_spot,
        "total_rows": total_rows,
        "model_complete_rows": model_complete,
        "model_complete_pct": round(100.0 * model_complete / total_rows, 2) if total_rows else 0.0,
    }


def make_band_eval_context(
    state: AppState,
    *,
    index: str | None = None,
    expiry_ts: float | None = None,
    open_ts: float | None = None,
    close_ts: float | None = None,
    band_size: int = DEFAULT_BAND_SIZE,
    chart_client: ChartTickClient | None = None,
    use_chart_volume: bool = True,
) -> BandEvalContext | None:
    """Build context from ``AppState`` (read-only). Returns None if ring store missing."""
    store = state.tick_ring_store
    if store is None:
        return None
    idx = index_ring_key(index or getattr(state, "selected_index", None) or "NIFTY")
    if open_ts is None or close_ts is None:
        open_ts, close_ts = market_session_bounds_today()
    if expiry_ts is None:
        expiry_ts = float(close_ts)
    client = chart_client
    if client is None and use_chart_volume:
        client = ChartTickClient()
    return BandEvalContext(
        store=store,
        index_key=idx,
        open_ts=float(open_ts),
        close_ts=float(close_ts),
        expiry_ts=float(expiry_ts),
        strike_step=strike_step_for_index(idx),
        band_size=int(band_size),
        chart_client=client,
        use_chart_volume=bool(use_chart_volume),
    )

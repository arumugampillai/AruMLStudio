"""Load and validate per-day replay database context."""

from __future__ import annotations

import os
import sqlite3
from dataclasses import dataclass, field
from typing import Any

from storage.chain_replay_export import (
    ChainReplayError,
    bootstrap_provider_for_underlying,
    ist_market_session_bounds,
    normalize_expiry_param,
    require_v1_ticks_schema,
    resolve_chain_tokens,
)
from chain_replay_ml.export_atm_pipeline import (
    INDEX_CONFIG,
    STRIKE_STEP,
    normalize_index_name,
    replay_db_path,
)
from chain_replay_ml.bs import expiry_close_ts, normalize_strike_rupees
from chain_replay_ml.ticks import EMA_BAR_INTERVAL_SEC, TickTimeline, load_tick_timelines


@dataclass
class SourceSpec:
    source_id: str
    trading_day: str
    market: str
    expiry: str
    date_label: str = ""


@dataclass
class DayContext:
    source: SourceSpec
    db_path: str
    expiry_norm: str
    open_ts: float
    close_ts: float
    expiry_ts: float
    index_tl: TickTimeline
    strike_mapping: dict[tuple[float, str], tuple[str, str, TickTimeline]]
    # Front-month FUTIDX timeline (Phase 1). None / empty when unavailable (soft-fail).
    futures_tl: TickTimeline | None = None
    futures_token: str | None = None
    futures_symbol: str | None = None
    futures_expiry: str | None = None
    futures_ticks: int = 0
    # Session OHLC from token_day_meta (exchange SNAP_QUOTE day levels), rupees.
    spot_open: float | None = None
    spot_high: float | None = None
    spot_low: float | None = None
    spot_prev_close: float | None = None
    option_session_ohlc: dict[str, dict[str, float | None]] = field(default_factory=dict)
    effective_session_start_ts: float = 0.0
    source_ticks: int = 0
    spot_ticks: int = 0
    chain_ticks: int = 0
    feature_grid_step_sec: int = EMA_BAR_INTERVAL_SEC
    feature_grid_gap_max_sec: float = 0.0
    validation_lines: list[str] = field(default_factory=list)
    # Sharp Momentum — shared spot impulse state (sequential, time-normalized decay).
    spot_momentum_by_ts: dict[float, Any] = field(default_factory=dict)
    spot_momentum_run_state: Any = None
    spot_momentum_prev_ts: float | None = None
    spot_momentum_prev_spot: float | None = None
    spot_momentum_ready_through: float | None = None
    spot_momentum_step_sec: int = 0
    # Historic NIFTY multi-TF EMA book (angel_historic_bars) — day-scoped as-of join.
    historic_spot_ema_book: Any = None

    @property
    def spot_tl(self) -> TickTimeline:
        """Uniform alias for the INDEX / spot timeline."""
        return self.index_tl

    @property
    def history_tl(self) -> Any:
        """Uniform alias for historic multi-TF context (EMA book today)."""
        return self.historic_spot_ema_book

def _count_ticks(conn: sqlite3.Connection, tokens: list[str]) -> int:
    if not tokens:
        return 0
    placeholders = ",".join("?" * len(tokens))
    row = conn.execute(
        f"SELECT COUNT(*) FROM ticks WHERE token IN ({placeholders})",
        tokens,
    ).fetchone()
    return int(row[0]) if row else 0


def load_day_context(
    chart_dir: str,
    source: SourceSpec,
    *,
    feature_grid_step_sec: int | None = None,
    max_tick_ts: float | None = None,
    tick_pad_before_sec: float = 0.0,
) -> DayContext:
    db_path = replay_db_path(chart_dir, source.trading_day)
    if not db_path or not os.path.isfile(db_path):
        raise ChainReplayError(f"No database for {source.trading_day}")

    underlying = normalize_index_name(source.market)
    bootstrap_provider_for_underlying(underlying, INDEX_CONFIG, normalize_index_name=normalize_index_name)
    expiry_norm = normalize_expiry_param(source.expiry)
    open_ts, close_ts = ist_market_session_bounds(source.trading_day)
    expiry_ts = expiry_close_ts(expiry_norm)
    tick_query_end = min(close_ts, float(max_tick_ts)) if max_tick_ts is not None else close_ts
    tick_query_start = open_ts - max(0.0, float(tick_pad_before_sec))

    conn = sqlite3.connect(db_path)
    try:
        require_v1_ticks_schema(conn)
        token_meta = resolve_chain_tokens(
            conn,
            underlying=underlying,
            expiry=expiry_norm,
            as_of_date=source.trading_day,
            index_config=INDEX_CONFIG,
            normalize_index_name=normalize_index_name,
        )
        index_token = None
        for tok, meta in token_meta.items():
            if meta.get("type") == "INDEX":
                index_token = tok
                break
        if not index_token:
            raise ChainReplayError("Spot/index token not found")

        index_tl = load_tick_timelines(conn, [index_token], tick_query_start, tick_query_end).get(index_token)
        if not index_tl or not index_tl.timestamps:
            raise ChainReplayError("No spot ticks in session")

        opt_tokens = [tok for tok, meta in token_meta.items() if meta.get("type") in ("CE", "PE")]
        if not opt_tokens:
            raise ChainReplayError(f"No option chain tokens for expiry {expiry_norm}")

        all_timelines = load_tick_timelines(conn, opt_tokens, tick_query_start, tick_query_end)
        strike_mapping: dict[tuple[float, str], tuple[str, str, TickTimeline]] = {}
        chain_ticks = 0
        for tok, meta in token_meta.items():
            if meta.get("type") not in ("CE", "PE"):
                continue
            strike_r = normalize_strike_rupees(meta.get("strike"))
            opt_type = str(meta.get("type") or "").upper()
            opt_tl = all_timelines.get(tok)
            if opt_tl and opt_tl.timestamps:
                strike_mapping[(strike_r, opt_type)] = (tok, str(meta.get("symbol") or ""), opt_tl)
                chain_ticks += len(opt_tl.timestamps)

        spot_ticks = len(index_tl.timestamps)
        source_ticks = spot_ticks + chain_ticks

        # Front-month futures — resolve once per day; soft-fail if missing.
        futures_tl: TickTimeline | None = None
        futures_token: str | None = None
        futures_symbol: str | None = None
        futures_expiry: str | None = None
        futures_ticks = 0
        futures_line = "○ Futures unavailable (soft-fail)"
        try:
            from .futures_context import resolve_front_month_futures

            fut_meta = resolve_front_month_futures(
                conn,
                underlying=underlying,
                trading_day=source.trading_day,
                normalize_index_name=normalize_index_name,
                open_ts=tick_query_start,
                close_ts=tick_query_end,
            )
            if fut_meta and fut_meta.get("token"):
                fut_tok = str(fut_meta["token"])
                loaded = load_tick_timelines(
                    conn, [fut_tok], tick_query_start, tick_query_end,
                ).get(fut_tok)
                if loaded is not None and loaded.timestamps:
                    futures_tl = loaded
                    futures_token = fut_tok
                    futures_symbol = str(fut_meta.get("symbol") or "") or None
                    futures_expiry = str(fut_meta.get("expiry") or "") or None
                    futures_ticks = len(loaded.timestamps)
                    source_ticks += futures_ticks
                    label = futures_symbol or fut_tok
                    futures_line = f"✓ Futures Loaded ({label})"
        except Exception:
            futures_line = "○ Futures unavailable (soft-fail)"

        # Session OHLC / prev_close from token_day_meta (feed-backed).
        spot_open = spot_high = spot_low = spot_prev_close = None
        option_session_ohlc: dict[str, dict[str, float | None]] = {}
        try:
            from .session_ohlc import load_session_ohlc_by_token

            opt_toks = [tok for tok, _, _ in strike_mapping.values()]
            loaded = load_session_ohlc_by_token(
                conn,
                [index_token, *opt_toks],
                as_of_date=source.trading_day,
            )
            spot_sess = loaded.get(index_token) or {}
            spot_open = spot_sess.get("open")
            spot_high = spot_sess.get("high")
            spot_low = spot_sess.get("low")
            spot_prev_close = spot_sess.get("prev_close")
            option_session_ohlc = {
                tok: loaded.get(tok) or {"open": None, "high": None, "low": None, "prev_close": None}
                for tok in opt_toks
            }
        except Exception:
            option_session_ohlc = {}

        from .tick_coverage import spot_tick_bounds

        bounds = spot_tick_bounds(index_tl)
        effective_session_start_ts = float(bounds[0]) if bounds else float(open_ts)

        grid_step = max(int(feature_grid_step_sec or EMA_BAR_INTERVAL_SEC), 1)

        lines = [
            f"✓ Database Opened",
            f"✓ Spot Loaded",
            futures_line,
            f"✓ Option Chain Loaded",
            f"✓ Expiry {source.expiry} Found",
        ]
        return DayContext(
            source=source,
            db_path=db_path,
            expiry_norm=expiry_norm,
            open_ts=open_ts,
            effective_session_start_ts=effective_session_start_ts,
            close_ts=close_ts,
            expiry_ts=expiry_ts,
            index_tl=index_tl,
            futures_tl=futures_tl,
            futures_token=futures_token,
            futures_symbol=futures_symbol,
            futures_expiry=futures_expiry,
            futures_ticks=futures_ticks,
            spot_open=spot_open,
            spot_high=spot_high,
            spot_low=spot_low,
            spot_prev_close=spot_prev_close,
            option_session_ohlc=option_session_ohlc,
            strike_mapping=strike_mapping,
            source_ticks=source_ticks,
            spot_ticks=spot_ticks,
            chain_ticks=chain_ticks,
            feature_grid_step_sec=grid_step,
            validation_lines=lines,
        )
    finally:
        conn.close()


def probe_first_spot_tick_ts(chart_dir: str, source: SourceSpec) -> float | None:
    """Earliest spot tick in session — used to size simulator tick loads."""
    db_path = replay_db_path(chart_dir, source.trading_day)
    if not db_path or not os.path.isfile(db_path):
        return None
    underlying = normalize_index_name(source.market)
    bootstrap_provider_for_underlying(underlying, INDEX_CONFIG, normalize_index_name=normalize_index_name)
    expiry_norm = normalize_expiry_param(source.expiry)
    open_ts, close_ts = ist_market_session_bounds(source.trading_day)
    conn = sqlite3.connect(db_path)
    try:
        require_v1_ticks_schema(conn)
        token_meta = resolve_chain_tokens(
            conn,
            underlying=underlying,
            expiry=expiry_norm,
            as_of_date=source.trading_day,
            index_config=INDEX_CONFIG,
            normalize_index_name=normalize_index_name,
        )
        index_token = None
        for tok, meta in token_meta.items():
            if meta.get("type") == "INDEX":
                index_token = tok
                break
        if not index_token:
            return None
        row = conn.execute(
            """
            SELECT MIN(ts) FROM ticks
            WHERE token = ? AND ts >= ? AND ts <= ?
              AND ltp IS NOT NULL AND ltp > 0
            """,
            (index_token, open_ts, close_ts),
        ).fetchone()
        if not row or row[0] is None:
            return None
        return float(row[0])
    except (ChainReplayError, sqlite3.Error, TypeError, ValueError):
        return None
    finally:
        conn.close()


def simulator_tick_load_bounds(
    trading_day: str,
    *,
    duration_minutes: int,
    first_spot_ts: float | None = None,
    pad_before_sec: float = 60.0,
    max_horizon_sec: float = 0.0,
) -> tuple[float, float]:
    """Return (tick_query_start, tick_query_end) for warm-up simulator duration."""
    open_ts, session_close = ist_market_session_bounds(trading_day)
    duration_sec = max(int(duration_minutes), 1) * 60.0
    partial_from_open = min(session_close, open_ts + duration_sec)
    if first_spot_ts is not None and float(first_spot_ts) > partial_from_open:
        tick_end = min(session_close, float(first_spot_ts) + duration_sec)
    else:
        tick_end = partial_from_open
    if max_horizon_sec > 0:
        tick_end = min(session_close, tick_end + float(max_horizon_sec))
    tick_start = open_ts - max(0.0, float(pad_before_sec))
    return tick_start, tick_end


def validate_day_context(ctx: DayContext) -> list[str]:
    issues: list[str] = []
    if ctx.spot_ticks <= 0:
        issues.append("Spot ticks missing")
    if ctx.chain_ticks <= 0:
        issues.append("Chain ticks missing")
    if not ctx.strike_mapping:
        issues.append("No strike timelines for selected expiry")
    if ctx.close_ts <= ctx.open_ts:
        issues.append("Invalid session bounds")
    return issues


def token_timelines_from_day_context(ctx: DayContext) -> dict[str, TickTimeline]:
    """Token → option timeline map already loaded during feature build."""
    out: dict[str, TickTimeline] = {}
    for tok, _symbol, opt_tl in ctx.strike_mapping.values():
        if opt_tl and opt_tl.timestamps:
            out[str(tok)] = opt_tl
    return out


def reset_ctx_build_caches(ctx: DayContext, *, step_sec: int | None = None) -> None:
    """Clear derived build caches on a reused DayContext (cold-build benchmark)."""
    from chain_replay_ml.dataset_builder.gap_policy_instrumentation import reset_day_context_feature_grid

    reset_day_context_feature_grid(ctx)
    step = max(int(step_sec or ctx.feature_grid_step_sec or EMA_BAR_INTERVAL_SEC), 1)
    from .sharp_momentum import _reset_spot_momentum_cache

    _reset_spot_momentum_cache(ctx, step)

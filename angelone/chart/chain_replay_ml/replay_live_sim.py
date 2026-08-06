"""Replay paper sim: ATM-band XGB on exported features, backtest target/SL."""
from __future__ import annotations

import math
import os
import sqlite3
import sys
from typing import Any

_CHART_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_ROOT_DIR = os.path.abspath(os.path.join(_CHART_DIR, "..", ".."))
for _p in (_CHART_DIR, _ROOT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from chain_replay_ml.execution_audit import check_scalp_outcome_seconds_config_b
from chain_replay_ml.export_atm_pipeline import replay_db_path
from chain_replay_ml.recompute_2_1_ratio import _build_scored_ml_frame, _strat_target_sl
from chain_replay_ml.ticks import load_tick_timelines
from chain_replay_ml.training.default_model import resolve_default_model_name
from shared.data.data_api_utils import calculate_charges
from storage.chain_replay_export import ist_market_session_bounds
from research.atm_band_ml.xgb_inference import DEFAULT_SCORE_THRESHOLD

# Import protocol via chart package path
from trade_decision import protocol as td_proto

ML_EXECUTION_WINDOW_SEC = 300.0
_DEFAULT_QTY = 65
DEFAULT_SYMBOL_COOLDOWN_SLOTS = 0
DEFAULT_DECISION_SLOT_LIMIT = 5


class ReplayDecisionSession:
    """Hub-aligned decision slots + cooldown for replay (Phase 7)."""

    def __init__(
        self,
        *,
        decision_slot_limit: int,
        symbol_cooldown_slots: int = DEFAULT_SYMBOL_COOLDOWN_SLOTS,
        trade_ids: td_proto.TradeIdSequencer | None = None,
    ) -> None:
        self._decision_slot_limit = max(1, int(decision_slot_limit))
        self._symbol_cooldown_slots = max(0, int(symbol_cooldown_slots))
        self._symbol_cooldown: dict[str, int] = {}
        self._trade_ids = trade_ids or td_proto.TradeIdSequencer()
        self._skip_reason_counts: dict[str, int] = {}

    def on_eval_tick(self) -> None:
        if not self._symbol_cooldown:
            return
        for sym in list(self._symbol_cooldown.keys()):
            remaining = int(self._symbol_cooldown.get(sym) or 0)
            if remaining <= 0:
                self._symbol_cooldown.pop(sym, None)
                continue
            self._symbol_cooldown[sym] = remaining - 1
            if self._symbol_cooldown[sym] <= 0:
                self._symbol_cooldown.pop(sym, None)

    def active_decision_slots(self) -> int:
        return sum(1 for n in self._symbol_cooldown.values() if int(n) > 0)

    def decision_slots_full(self) -> bool:
        return self.active_decision_slots() >= self._decision_slot_limit

    def is_symbol_in_cooldown(self, symbol: str) -> bool:
        sym = str(symbol or "").strip().upper()
        return int(self._symbol_cooldown.get(sym) or 0) > 0

    def check_enter_gates(self, symbol: str) -> tuple[bool, str]:
        if self._symbol_cooldown_slots <= 0:
            return True, ""
        sym = str(symbol or "").strip().upper()
        if sym and self.is_symbol_in_cooldown(sym):
            return False, "Symbol in cooldown"
        return True, ""

    def mint_trade_id(
        self,
        symbol: str,
        ts: float,
        *,
        strike: float | None = None,
        opt_type: str | None = None,
    ) -> str:
        return self._trade_ids.mint(symbol, ts, strike=strike, opt_type=opt_type)

    def mark_enter(self, symbol: str) -> None:
        sym = str(symbol or "").strip().upper()
        if sym:
            self._symbol_cooldown[sym] = self._symbol_cooldown_slots

    def record_skip(self, reason_codes: list[str]) -> None:
        for code in reason_codes:
            self._skip_reason_counts[code] = self._skip_reason_counts.get(code, 0) + 1


def parse_max_positions(value: str | int | None) -> int | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in ("unconstrained", "0"):
        return None
    try:
        return max(1, min(30, int(text)))
    except (TypeError, ValueError):
        return 1


def _fmt_ist_hms(ts: float) -> str:
    from datetime import datetime
    from zoneinfo import ZoneInfo

    try:
        return datetime.fromtimestamp(float(ts), tz=ZoneInfo("Asia/Kolkata")).strftime("%H:%M:%S")
    except (TypeError, ValueError, OSError, OverflowError):
        return "—"


def _trade_pnl_rs(entry_ltp: float, exit_ltp: float, qty: int = _DEFAULT_QTY) -> dict[str, float]:
    v_buy = entry_ltp * qty
    v_sell = exit_ltp * qty
    gross = v_sell - v_buy
    charges = float(calculate_charges(v_buy, v_sell))
    return {
        "gross_pnl": round(gross, 2),
        "charges": round(charges, 2),
        "net_pnl": round(gross - charges, 2),
    }


def _simulate_positions(
    candidates: list[dict[str, Any]],
    max_concurrent: int | None,
) -> list[dict[str, Any]]:
    """Apply position cap; ``None`` = unconstrained (Neo ML panel parity)."""
    limit = max_concurrent if max_concurrent is not None else 999_999
    sorted_candidates = sorted(candidates, key=lambda x: float(x["entry_ts"]))
    executed: list[dict[str, Any]] = []
    active: list[tuple[float, dict[str, Any]]] = []
    for t in sorted_candidates:
        entry_ts = float(t["entry_ts"])
        active = [(ex, tr) for ex, tr in active if ex > entry_ts]
        if len(active) < limit:
            active.append((float(t["exit_ts"]), t))
            executed.append(t)
    return executed


def _prune_open_positions(
    active: list[tuple[float, str]],
    entry_ts: float,
) -> list[tuple[float, str]]:
    """Drop positions closed at or before entry_ts (matches report simulate_positions)."""
    return [(ex, sym) for ex, sym in active if ex > entry_ts]


def _signal_row(
    signal_ts: float,
    *,
    action: str,
    reason: str,
    symbol: str = "",
    score: float | None = None,
    p_hit: float | None = None,
    delta_band: str = "",
    spot: float | None = None,
    trade_id: str = "",
    kill_reason: str = "",
    reason_codes: list[str] | None = None,
) -> dict[str, Any]:
    event = "ENTER" if action == "ENTER" else "SKIP"
    kill, codes = td_proto.map_skip_reason(reason) if event == "SKIP" else ("", [])
    if event == "SKIP" and kill_reason:
        kill = kill_reason
    if event == "SKIP" and reason_codes:
        codes = list(reason_codes)
    row = {
        "ts": signal_ts,
        "time": _fmt_ist_hms(signal_ts),
        "action": action,
        "event": event,
        "reason": reason,
        "mode": "xgboost_band",
        "spot": spot,
        "symbol": symbol,
        "score": score,
        "p_hit": p_hit,
        "delta_band": delta_band,
    }
    if trade_id:
        row["trade_id"] = trade_id
    if event == "SKIP":
        row["kill_reason"] = kill or reason
        row["reason_codes"] = codes
    elif event == "ENTER":
        row["reason_codes"] = list(td_proto.ENTER_REASON_CODES)
    return row


def run_replay_live_sim(
    *,
    date_str: str,
    expiry: str,
    underlying: str = "NIFTY",
    model_name: str | None = None,
    stamp: str | None = None,
    position_limit: int | None = 1,
    decision_slot_limit: int | None = None,
    symbol_cooldown_slots: int = DEFAULT_SYMBOL_COOLDOWN_SLOTS,
    min_score: float = DEFAULT_SCORE_THRESHOLD,
) -> dict[str, Any]:
    """
    Score registry model on dataset rows for the replay day,
    apply report-aligned concurrent position limit + backtest target/SL, return signals + trades.
    """
    use_model = model_name or stamp
    if not use_model:
        use_model = resolve_default_model_name(os.path.join(_CHART_DIR, "data"))
    slot_limit = decision_slot_limit if decision_slot_limit is not None else position_limit
    if slot_limit is None:
        slot_limit = 999_999

    df = _build_scored_ml_frame(date_str, use_model, expiry_hint=expiry)
    if df.empty:
        return _empty_result(use_model or "", slot_limit, min_score)

    open_ts, close_ts = ist_market_session_bounds(date_str)
    db_path = replay_db_path(_CHART_DIR, date_str)
    tokens = sorted({str(t) for t in df["token"].astype(str)})
    conn = sqlite3.connect(db_path)
    try:
        timelines = load_tick_timelines(conn, tokens, open_ts, close_ts)
    finally:
        conn.close()

    session = ReplayDecisionSession(
        decision_slot_limit=max(1, int(slot_limit)) if slot_limit < 999_999 else 1,
        symbol_cooldown_slots=int(symbol_cooldown_slots),
    )
    signals: list[dict[str, Any]] = []
    trades: list[dict[str, Any]] = []
    grid_slots = 0
    position_cap = slot_limit if slot_limit < 999_999 else 999_999
    active_open: list[tuple[float, str]] = []

    for ts, group in df.groupby("timestamp"):
        session.on_eval_tick()
        grid_slots += 1
        signal_ts = float(ts)
        top = group.sort_values(by="score", ascending=False).iloc[0]
        score = float(top["score"]) if top.get("score") is not None else 0.0
        symbol = str(top.get("symbol") or "")
        p_hit = float(top["P_hit"]) if top.get("P_hit") is not None else None
        band = str(top.get("delta_band") or "")
        spot = float(top["spot"]) if "spot" in top.index and top.get("spot") is not None else None
        strike = float(top.get("strike") or 0) or None
        opt_type = str(top.get("option_type") or "").upper() or None

        if score < min_score:
            kill, codes = td_proto.map_skip_reason(f"No score >= {min_score:g}")
            session.record_skip(codes)
            signals.append(
                _signal_row(
                    signal_ts,
                    action="SKIP",
                    reason=f"No score >= {min_score:g}",
                    symbol=symbol,
                    score=score,
                    p_hit=p_hit,
                    delta_band=band,
                    spot=spot,
                    kill_reason=kill,
                    reason_codes=codes,
                )
            )
            continue

        active_open = _prune_open_positions(active_open, signal_ts)
        if len(active_open) >= position_cap:
            gate_reason = (
                f"Position limit ({len(active_open)}/{position_cap})"
                if position_cap < 999_999
                else "Position limit"
            )
            kill, codes = td_proto.map_skip_reason(gate_reason)
            session.record_skip(codes)
            signals.append(
                _signal_row(
                    signal_ts,
                    action="SKIP",
                    reason=gate_reason,
                    symbol=symbol,
                    score=score,
                    p_hit=p_hit,
                    delta_band=band,
                    spot=spot,
                    kill_reason=kill,
                    reason_codes=codes,
                )
            )
            continue

        ok, gate_reason = session.check_enter_gates(symbol)
        if not ok:
            kill, codes = td_proto.map_skip_reason(gate_reason)
            session.record_skip(codes)
            signals.append(
                _signal_row(
                    signal_ts,
                    action="SKIP",
                    reason=gate_reason,
                    symbol=symbol,
                    score=score,
                    p_hit=p_hit,
                    delta_band=band,
                    spot=spot,
                    kill_reason=kill,
                    reason_codes=codes,
                )
            )
            continue

        entry_ltp = float(top.get("ltp") or 0.0)
        if entry_ltp <= 0:
            kill, codes = td_proto.map_skip_reason("No live LTP")
            session.record_skip(codes)
            signals.append(
                _signal_row(
                    signal_ts,
                    action="SKIP",
                    reason="No entry LTP",
                    symbol=symbol,
                    score=score,
                    kill_reason=kill,
                    reason_codes=codes,
                )
            )
            continue

        tok = str(top["token"])
        strat_tl = timelines.get(tok)
        if strat_tl is None:
            kill, codes = td_proto.map_skip_reason("No option timeline")
            session.record_skip(codes)
            signals.append(
                _signal_row(
                    signal_ts,
                    action="SKIP",
                    reason="No option timeline",
                    symbol=symbol,
                    score=score,
                    kill_reason=kill,
                    reason_codes=codes,
                )
            )
            continue

        trade_id = session.mint_trade_id(symbol, signal_ts, strike=strike, opt_type=opt_type)
        if symbol_cooldown_slots > 0:
            session.mark_enter(symbol)

        tgt_pct, sl_pct = _strat_target_sl(entry_ltp)
        outcome, elapsed, exit_p, exit_ts = check_scalp_outcome_seconds_config_b(
            strat_tl, signal_ts, ML_EXECUTION_WINDOW_SEC, tgt_pct, sl_pct,
        )
        outcome_type = "timeout"
        if outcome == 1:
            outcome_type = "target"
        elif outcome == -1:
            outcome_type = "sl"

        trade_row = {
            "id": trade_id,
            "trade_id": trade_id,
            "entry_ts": signal_ts,
            "exit_ts": float(exit_ts or signal_ts + ML_EXECUTION_WINDOW_SEC),
            "token": tok,
            "symbol": symbol,
            "opt_type": opt_type or "",
            "strike": strike or 0,
            "entry_ltp": round(entry_ltp, 2),
            "exit_ltp": round(float(exit_p or entry_ltp), 2),
            "target_pct": tgt_pct,
            "sl_pct": sl_pct,
            "score": score,
            "p_hit": p_hit or 0.0,
            "pred_max_return": float(top.get("pred_max_return") or 0),
            "pred_min_return": float(top.get("pred_min_return") or 0),
            "delta_band": band,
            "outcome_type": outcome_type,
            "elapsed_sec": float(elapsed or 0),
            "spot": spot,
            "source": "xgboost_band",
            "entered": True,
        }
        pnl = _trade_pnl_rs(trade_row["entry_ltp"], trade_row["exit_ltp"])
        trades.append({
            **trade_row,
            "status": "CLOSED",
            "time": _fmt_ist_hms(signal_ts),
            "exit_time": _fmt_ist_hms(trade_row["exit_ts"]),
            **pnl,
        })
        active_open.append((float(trade_row["exit_ts"]), symbol))
        signals.append(
            _signal_row(
                signal_ts,
                action="ENTER",
                reason="Score>=3, Delta OK, Premium OK",
                symbol=symbol,
                score=score,
                p_hit=p_hit,
                delta_band=band,
                spot=spot,
                trade_id=trade_id,
            )
        )

    entered_trades = list(trades)
    net_pnl = sum(float(t.get("net_pnl") or 0) for t in entered_trades)
    wins = sum(1 for t in entered_trades if t.get("outcome_type") == "target")
    losses = sum(1 for t in entered_trades if t.get("outcome_type") == "sl")
    timeouts = sum(1 for t in entered_trades if t.get("outcome_type") == "timeout")

    return {
        "model_name": use_model,
        "model_stamp": use_model,
        "signal_mode": "xgboost_band",
        "signal_grid": "10s",
        "min_score": min_score,
        "position_limit": slot_limit if slot_limit < 999_999 else None,
        "decision_slot_limit": slot_limit if slot_limit < 999_999 else None,
        "symbol_cooldown_slots": symbol_cooldown_slots,
        "position_limit_label": "Unconstrained" if slot_limit >= 999_999 else str(slot_limit),
        "grid_slots": grid_slots,
        "signal_count": len(signals),
        "candidate_count": len(trades),
        "entered_count": len(entered_trades),
        "skip_reason_counts": dict(session._skip_reason_counts),
        "signals": signals,
        "trades": trades,
        "entered_trades": entered_trades,
        "summary": {
            "trades": len(entered_trades),
            "wins": wins,
            "losses": losses,
            "timeouts": timeouts,
            "win_rate": round(100.0 * wins / len(entered_trades), 2) if entered_trades else 0.0,
            "net_pnl": round(net_pnl, 2),
        },
    }


def _empty_result(model_name: str, slot_limit: int, min_score: float) -> dict[str, Any]:
    label = "Unconstrained" if slot_limit >= 999_999 else str(slot_limit)
    return {
        "model_name": model_name,
        "model_stamp": model_name,
        "signal_mode": "xgboost_band",
        "signal_grid": "10s",
        "min_score": min_score,
        "position_limit": slot_limit if slot_limit < 999_999 else None,
        "decision_slot_limit": slot_limit if slot_limit < 999_999 else None,
        "symbol_cooldown_slots": DEFAULT_SYMBOL_COOLDOWN_SLOTS,
        "position_limit_label": label,
        "grid_slots": 0,
        "signal_count": 0,
        "candidate_count": 0,
        "entered_count": 0,
        "skip_reason_counts": {},
        "signals": [],
        "trades": [],
        "entered_trades": [],
        "summary": {
            "trades": 0,
            "wins": 0,
            "losses": 0,
            "timeouts": 0,
            "win_rate": 0.0,
            "net_pnl": 0.0,
        },
    }

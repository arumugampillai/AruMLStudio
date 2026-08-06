"""Live ML-style signals from tick ring + cross-symbol ranking; paper execution on rings."""
from __future__ import annotations

import bisect
import math
import re
import time
import tkinter as tk
from typing import Any, Sequence

from api.cross_symbol_ranking import RankedSymbol
from app.market_session import (
    ML_EXECUTION_WINDOW_SEC,
    ML_SIGNAL_TAIL_SEC,
    ML_SIGNAL_WARMUP_SEC,
    fmt_ist_hms,
    market_session_bounds_today,
    ml_signal_evaluation_times,
    now_ist_in_market_session,
)
from app.state import AppState
from research.atm_band_ml.band_evaluator import (
    band_contracts_from_state,
    evaluate_band_at_ts,
    index_ring_key,
    make_band_eval_context,
)
from research.atm_band_ml.chain_context import (
    ChainContextBuilder,
    chain_context_for_band,
)
from research.atm_band_ml.live_registry_builder import (
    LiveRegistrySession,
    build_legacy_live_features,
    build_live_registry_features,
)
from research.atm_band_ml.feature_parity import live_parity_report
from research.atm_band_ml.decision_debug import build_decision_debug_record, empty_decision_debug
from research.atm_band_ml.feature_coverage import (
    audit_registry_feature_coverage,
    empty_feature_coverage,
)
from research.atm_band_ml.pnl import aggregate_ml_pnl
from research.atm_band_ml.feature_builder import feature_completeness_ratio
from research.atm_band_ml.feature_window import feature_window_diagnostics
from research.atm_band_ml.xgb_inference import (
    DEFAULT_SCORE_THRESHOLD,
    AtmBandModelScorer,
)

_MIN_RANKING_SCORE = 35.0
_MIN_SIGNAL_QUALITY = 0.45
_MAX_SLOTS_PER_TICK = 3
FEATURE_PROBE_INTERVAL_SEC = 1.0
XGB_SIGNAL_INTERVAL_SEC = 1.0
_MAX_FEATURE_PROBE_LOG = 120
ML_MAX_POSITION_CHOICES = ("1", "2", "3", "4", "5", "Unconstrained")
ML_SIGNAL_MODE_CHOICES = ("ranking_proxy", "xgboost_band")
DEFAULT_ML_SIGNAL_MODE = "xgboost_band"


def parse_signal_mode(value: str | None) -> str:
    key = str(value or DEFAULT_ML_SIGNAL_MODE).strip().lower()
    if key in ("xgboost", "xgboost_band", "band", "xgb"):
        return "xgboost_band"
    return "ranking_proxy"


def format_signal_mode(mode: str | None) -> str:
    return "XGBoost band" if parse_signal_mode(mode) == "xgboost_band" else "Ranking proxy"


def parse_max_positions(value: str | int | None) -> int | None:
    """Return None for unconstrained; otherwise 1–5."""
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() == "unconstrained":
        return None
    try:
        return max(1, min(5, int(text)))
    except (TypeError, ValueError):
        return 1


def format_max_positions(limit: int | None) -> str:
    if limit is None:
        return "Unconstrained"
    return str(int(limit))


def position_limit_reached(open_count: int, max_positions: int | None) -> bool:
    if max_positions is None:
        return False
    return open_count >= int(max_positions)


def target_sl_pct_for_premium(ltp: float) -> tuple[float, float]:
    if ltp > 100.0:
        return 10.0, 5.0
    if ltp >= 50.0:
        return 3.0, 5.0
    if ltp >= 20.0:
        return 5.0, 5.0
    return 10.0, 5.0


def parse_option_symbol(symbol: str) -> tuple[float | None, str | None]:
    s = str(symbol or "").strip().upper()
    if not s:
        return None, None
    try:
        from angelone.symbol_bridge import compact_strike_option_symbol

        compact = compact_strike_option_symbol(s)
    except ImportError:
        compact = None
    if compact:
        m = re.match(r"(\d+)(CE|PE)$", compact, re.IGNORECASE)
        if m:
            try:
                return float(m.group(1)), m.group(2).upper()
            except ValueError:
                pass
    m = re.search(r"(\d+)(CE|PE)$", s)
    if not m:
        return None, None
    try:
        return float(m.group(1)), m.group(2).upper()
    except ValueError:
        return None, None


def _ring_samples(state: AppState, token: str) -> list[Any]:
    store = state.tick_ring_store
    if store is None:
        return []
    return store.snapshot(str(token))


def _ltp_at_or_before(samples: list[Any], target_ts: float) -> float | None:
    if not samples:
        return None
    ts_list = [float(s.ts) for s in samples]
    idx = bisect.bisect_right(ts_list, target_ts) - 1
    if idx < 0:
        return None
    px = float(samples[idx].ltp)
    return px if px > 0 else None


def _scan_outcome_on_ring(
    samples: list[Any],
    entry_ts: float,
    entry_ltp: float,
    target_pct: float,
    sl_pct: float,
) -> tuple[str, float, float, float]:
    """Return outcome_type, exit_ts, exit_ltp, elapsed_sec."""
    if not samples or entry_ltp <= 0:
        return "timeout", entry_ts + ML_EXECUTION_WINDOW_SEC, entry_ltp, ML_EXECUTION_WINDOW_SEC

    target_px = entry_ltp * (1.0 + target_pct / 100.0)
    stop_px = entry_ltp * (1.0 - sl_pct / 100.0)
    deadline = entry_ts + ML_EXECUTION_WINDOW_SEC

    ts_list = [float(s.ts) for s in samples]
    entry_idx = bisect.bisect_right(ts_list, entry_ts) - 1
    end_idx = bisect.bisect_right(ts_list, deadline)

    for idx in range(max(0, entry_idx + 1), end_idx):
        px = float(samples[idx].ltp)
        ts = float(samples[idx].ts)
        if px >= target_px:
            return "target", ts, px, max(0.0, ts - entry_ts)
        if px <= stop_px:
            return "sl", ts, px, max(0.0, ts - entry_ts)

    exit_px = _ltp_at_or_before(samples, deadline) or entry_ltp
    return "timeout", deadline, exit_px, ML_EXECUTION_WINDOW_SEC


def _pick_ranked_candidate(state: AppState) -> RankedSymbol | None:
    snap = state.cross_symbol_ranking_store.latest()
    if snap is None:
        return None
    pools = (
        list(snap.top_high_confidence_symbols or ()),
        list(snap.top_explosive_symbols or ()),
        list(snap.top_bullish_symbols or ()),
    )
    best: RankedSymbol | None = None
    for pool in pools:
        for row in pool:
            if row.ranking_score < _MIN_RANKING_SCORE:
                continue
            if row.signal_quality < _MIN_SIGNAL_QUALITY:
                continue
            if best is None or row.ranking_score > best.ranking_score:
                best = row
        if best is not None:
            break
    return best


def _resolve_index_name(state: AppState) -> str:
    top_menu = getattr(state, "top_menu", None)
    if top_menu is not None and hasattr(top_menu, "index_var"):
        try:
            return index_ring_key(str(top_menu.index_var.get() or "NIFTY"))
        except (tk.TclError, AttributeError):
            pass
    return "NIFTY"


def _resolve_expiry_ts(state: AppState) -> float | None:
    top_menu = getattr(state, "top_menu", None)
    if top_menu is None:
        return None
    try:
        display = str(top_menu.expiry_var.get() or "").strip()
        raw = top_menu.display_to_expiry(display)
        expiry_date = top_menu._coerce_expiry_date(raw)
        if expiry_date is None:
            expiry_date = top_menu._parse_expiry_date(raw)
        if expiry_date is None:
            return None
        import sys
        from pathlib import Path

        chart_dir = Path(__file__).resolve().parents[2] / "angelone" / "chart"
        chart_str = str(chart_dir)
        if chart_str not in sys.path:
            sys.path.insert(0, chart_str)
        from chain_replay_ml.bs import expiry_close_ts

        return float(expiry_close_ts(expiry_date.strftime("%Y-%m-%d")))
    except Exception:
        return None


class MlLiveEngine:
    """Live ML signals: 1s XGBoost grid or 10s ranking grid; paper trades on tick rings."""

    def __init__(self, state: AppState) -> None:
        self.state = state
        self.armed = False
        self.max_positions = None
        self.min_rank_score = _MIN_RANKING_SCORE
        self.signal_mode = DEFAULT_ML_SIGNAL_MODE
        self.min_score = float(DEFAULT_SCORE_THRESHOLD)
        self.model_stamp = None
        self.model_name = None
        self._signal_times: list[float] = []
        self._next_signal_idx = 0
        self._last_xgb_signal_ts = 0.0
        self._xgb_eval_count = 0
        self._last_grid_day: str | None = None
        self._recent_signals: list[dict[str, Any]] = []
        self._status = "Idle"
        self._last_tick_ts = 0.0
        self._band_scorer: AtmBandModelScorer | None = None
        self._band_eval_ctx = None
        self._chain_ctx_builder: ChainContextBuilder | None = None
        self._registry_session: LiveRegistrySession | None = None
        self._last_feature_coverage: dict[str, Any] = {}
        self._last_feature_parity: dict[str, Any] = {}
        self._last_decision_debug: dict[str, Any] = {}
        self._model_load_error: str | None = None
        self.feature_probe_enabled = True
        self._last_feature_probe_ts = 0.0
        self._feature_probe_log: list[dict[str, Any]] = []
        self._feature_probe_seq = 0
        self.trade_decision_hub = None
        self.server_brain_only = False
        self.allow_off_hours = False

    def clear_feature_probe_log(self) -> None:
        self._feature_probe_log.clear()
        self._feature_probe_seq = 0

    def reset_day_grid(self) -> None:
        open_ts, close_ts = market_session_bounds_today()
        self._signal_times = ml_signal_evaluation_times(open_ts, close_ts)
        self._next_signal_idx = 0
        self._last_xgb_signal_ts = 0.0
        self._xgb_eval_count = 0
        self._last_grid_day = time.strftime("%Y-%m-%d")
        self._band_eval_ctx = None
        self._chain_ctx_builder = None
        if self._registry_session is not None:
            self._registry_session.reset()
        self._registry_session = None

    def _xgb_grid_bounds(self) -> tuple[float, float]:
        open_ts, close_ts = market_session_bounds_today()
        grid_start = math.ceil(float(open_ts) + ML_SIGNAL_WARMUP_SEC)
        grid_end = float(close_ts) - ML_SIGNAL_TAIL_SEC
        return grid_start, grid_end

    def _eval_interval_sec(self) -> float:
        hub = self.trade_decision_hub
        if hub is not None:
            return max(1.0, float(getattr(hub, "_evaluation_interval_sec", XGB_SIGNAL_INTERVAL_SEC)))
        return XGB_SIGNAL_INTERVAL_SEC

    def _xgb_signal_due(self, ts: float) -> float | None:
        interval = self._eval_interval_sec()
        signal_ts = float(math.floor(ts))
        grid_start, grid_end = self._xgb_grid_bounds()
        if signal_ts < grid_start or signal_ts > grid_end + 0.001:
            return None
        if self._last_xgb_signal_ts > 0 and (signal_ts - self._last_xgb_signal_ts) < interval - 0.001:
            return None
        return signal_ts

    def _band_scorer_instance(self) -> AtmBandModelScorer:
        if self._band_scorer is None:
            self._band_scorer = AtmBandModelScorer(
                model_name=self.model_name or self.model_stamp,
            )
        return self._band_scorer

    def _ensure_band_scorer_loaded(self) -> bool:
        scorer = self._band_scorer_instance()
        if scorer.is_loaded():
            self._model_load_error = None
            return True
        try:
            scorer.load()
            self._model_load_error = None
            return True
        except FileNotFoundError as exc:
            self._model_load_error = str(exc)
            return False
        except ImportError as exc:
            self._model_load_error = str(exc)
            return False
        except Exception as exc:
            self._model_load_error = f"{type(exc).__name__}: {exc}"
            return False

    def _band_context(self):
        if self._band_eval_ctx is not None:
            return self._band_eval_ctx
        expiry_ts = _resolve_expiry_ts(self.state)
        open_ts, close_ts = market_session_bounds_today()
        if expiry_ts is None:
            expiry_ts = float(close_ts)
        self._band_eval_ctx = make_band_eval_context(
            self.state,
            index=_resolve_index_name(self.state),
            expiry_ts=expiry_ts,
            open_ts=open_ts,
            close_ts=close_ts,
        )
        return self._band_eval_ctx

    def tick(self, now: float | None = None) -> None:
        if getattr(self, "server_brain_only", False):
            self._status = "Server brain (local off)"
            return
        ts = float(now if now is not None else time.time())
        self._last_tick_ts = ts
        manager = self.state.target_manager
        open_entries = manager.get_open_ml_entries() if manager else []
        if not self.armed and not open_entries:
            self._status = "Idle"
            return

        day = time.strftime("%Y-%m-%d")
        if day != self._last_grid_day:
            self.reset_day_grid()

        if open_entries and self.trade_decision_hub is None:
            self._update_open_trades(ts)

        if not self.armed:
            self._status = "Disarmed"
            return
        if not getattr(self, "allow_off_hours", False) and not now_ist_in_market_session(ts):
            self._status = "Outside market hours"
            return
        if not self.state.tick_ring_store.tracked_keys():
            self._status = "Waiting for tick ring data"
            return

        if parse_signal_mode(self.signal_mode) == "xgboost_band":
            if not self._ensure_band_scorer_loaded():
                self._status = "Models unavailable"
            elif not band_contracts_from_state(self.state):
                self._status = "Waiting for option chain"
            else:
                self._status = "Live (XGB)"
        else:
            self._status = "Live"

        processed = 0
        if parse_signal_mode(self.signal_mode) == "xgboost_band":
            signal_ts = self._xgb_signal_due(ts)
            if signal_ts is not None:
                self._last_xgb_signal_ts = signal_ts
                self._xgb_eval_count += 1
                self._evaluate_signal_slot(signal_ts)
                processed = 1
        else:
            while (
                processed < _MAX_SLOTS_PER_TICK
                and self._next_signal_idx < len(self._signal_times)
                and self._signal_times[self._next_signal_idx] <= ts + 0.001
            ):
                signal_ts = self._signal_times[self._next_signal_idx]
                self._next_signal_idx += 1
                processed += 1
                self._evaluate_signal_slot(signal_ts)

        if parse_signal_mode(self.signal_mode) == "xgboost_band":
            grid_start, grid_end = self._xgb_grid_bounds()
            next_floor = self._last_xgb_signal_ts + self._eval_interval_sec()
            if self._last_xgb_signal_ts < grid_start:
                next_floor = grid_start
            if next_floor > grid_end + 0.001:
                self._refresh_live_status()
                return
            if next_floor <= ts + 0.001:
                self._status = "Catching up"
                return
        elif (
            self._next_signal_idx < len(self._signal_times)
            and self._signal_times[self._next_signal_idx] <= ts + 0.001
        ):
            self._status = "Catching up"
            return
        self._refresh_live_status()

    def _refresh_live_status(self) -> None:
        if parse_signal_mode(self.signal_mode) == "xgboost_band":
            if not self._ensure_band_scorer_loaded():
                self._status = "Models unavailable"
            elif not band_contracts_from_state(self.state):
                self._status = "Waiting for option chain"
            else:
                self._status = "Live (XGB)"
        else:
            self._status = "Live"

    def _features_for_token(self, snap, token: str) -> dict[str, Any]:
        tok = str(token or "").strip()
        for row in getattr(snap, "rows", ()) or ():
            if str(row.contract.token).strip() == tok:
                return dict(row.result.features or {})
        return {}

    def _best_scored(self, scored) -> Any | None:
        candidates = [
            s for s in scored if getattr(s, "scorable", False) and s.score is not None
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda s: float(s.score or 0.0))

    def _scored_skip_context(self, scored, log_top: Any | None = None) -> dict[str, Any]:
        """Best-effort symbol/score for suppressed XGB rows (decision stream display)."""
        row = log_top
        if row is None and scored:
            with_score = [
                s for s in scored if getattr(s, "score", None) is not None
            ]
            if with_score:
                row = max(with_score, key=lambda s: float(s.score or 0.0))
            else:
                row = scored[0]
        if row is None:
            return {}
        out: dict[str, Any] = {}
        sym = str(getattr(row, "symbol", "") or "").strip()
        tok = str(getattr(row, "token", "") or "").strip()
        strike = getattr(row, "strike", None)
        opt_type = str(getattr(row, "option_type", "") or "").strip().upper()
        if sym:
            out["symbol"] = sym
        if tok:
            out["token"] = tok
        if strike is not None:
            out["strike"] = float(strike)
        if opt_type in ("CE", "PE"):
            out["opt_type"] = opt_type
        sc = getattr(row, "score", None)
        if sc is not None:
            out["score"] = float(sc)
        else:
            note = str(getattr(row, "reason", "") or "").strip()
            if note:
                out["score_note"] = note.replace("_", " ")
        ph = getattr(row, "P_hit", None)
        if ph is not None:
            out["p_hit"] = float(ph)
        return out

    def _live_ltp_map(self) -> dict[str, float]:
        ltps: dict[str, float] = {}
        for src in (
            getattr(self.state, "latest_ltps_tracker", None),
            getattr(self.state, "latest_ltps", None),
        ):
            if not isinstance(src, dict):
                continue
            for tok, px in src.items():
                try:
                    val = float(px)
                except (TypeError, ValueError):
                    continue
                if val > 0:
                    ltps[str(tok).strip()] = val
        return ltps

    def _overlay_live_option_features(
        self,
        features: dict[str, Any],
        *,
        option_type: str,
        strike: float,
        spot: float,
        expiry_ts: float,
        ts: float,
        ltp: float,
    ) -> dict[str, Any]:
        out = dict(features or {})
        out["ltp"] = float(ltp)
        out["spot"] = float(spot)
        out["strike"] = float(strike)
        out["option_type"] = str(option_type).upper()
        if spot > 0:
            out["ltp_to_spot_ratio"] = float(ltp / spot)
            out["distance_from_spot_pct"] = float(100.0 * (strike - spot) / spot)
        try:
            from chain_replay_ml import bs
            from chain_replay_ml.constants import RISK_FREE_RATE

            t_exp = bs.time_to_expiry_years(float(expiry_ts), float(ts))
            out["minutes_to_expiry"] = max(0.0, (float(expiry_ts) - float(ts)) / 60.0)
            iv = bs.implied_volatility(str(option_type).upper(), ltp, spot, strike, RISK_FREE_RATE, t_exp)
            if iv is not None and iv > 0:
                out["current_iv"] = float(iv)
                greeks = bs.greeks(str(option_type).upper(), spot, strike, RISK_FREE_RATE, t_exp, iv)
                out["delta"] = float(greeks.get("delta") or 0.0)
                out["gamma"] = float(greeks.get("gamma") or 0.0)
                out["theta"] = float(greeks.get("theta") or 0.0)
                out["vega"] = float(greeks.get("vega") or 0.0)
                out["abs_delta"] = abs(float(out["delta"]))
                out["is_call"] = 1.0 if str(option_type).upper() == "CE" else 0.0
        except Exception:
            pass
        return out

    def feature_parity_snapshot(self) -> dict[str, Any]:
        if self._last_feature_parity:
            return dict(self._last_feature_parity)
        return {
            "dataset_builder_features": 124,
            "live_builder_features": 0,
            "parity_pct": 0.0,
            "implementation": {},
            "significant_differences": [],
            "reason": "No live evaluation yet",
        }

    def feature_coverage_snapshot(self) -> dict[str, Any]:
        """Latest strict registry feature coverage from the last live evaluation probe."""
        if self._last_feature_coverage:
            return dict(self._last_feature_coverage)
        stamp = str(self.model_name or self.model_stamp or "")
        scorer = self._band_scorer
        if scorer is not None and scorer.is_loaded():
            req = list(scorer._registry_features or [])
            base = empty_feature_coverage(
                model_name=stamp,
                reason="Armed evaluation has not run yet — load model and Start ML",
            )
            base["required_count"] = len(req)
            base["missing_count"] = len(req)
            base["validation"]["required_order"] = req
            return base
        return empty_feature_coverage(model_name=stamp, reason="Model not loaded")

    def decision_debug_snapshot(self) -> dict[str, Any]:
        if self._last_decision_debug:
            return dict(self._last_decision_debug)
        return empty_decision_debug()

    def _record_decision_debug(
        self,
        *,
        signal_ts: float,
        scorer,
        skip_reason: str,
        suppressed: bool,
        decision_top: Any | None,
        best_top: Any | None,
        scored: Sequence[Any],
    ) -> None:
        target = str(getattr(scorer, "_target", "") or "")
        self._last_decision_debug = build_decision_debug_record(
            signal_ts=signal_ts,
            model_name=str(scorer.model_name or scorer.stamp or ""),
            target=target,
            configured_threshold=float(self.min_score),
            skip_reason=skip_reason,
            suppressed=suppressed,
            decision_top=decision_top,
            best_top=best_top,
            scored=scored,
        )

    def _build_live_row_features(
        self,
        row,
        *,
        snap,
        signal_ts: float,
        ctx,
        contracts,
        live_ltps: dict[str, float],
        expiry_ts: float,
        spot: float,
    ) -> dict[str, Any]:
        if self._registry_session is None:
            self._registry_session = LiveRegistrySession()
        contract = row.contract
        feats = build_live_registry_features(
            session=self._registry_session,
            band_ctx=ctx,
            contract=contract,
            signal_ts=float(signal_ts),
            contracts=contracts,
            state=self.state,
        )
        tok = str(contract.token).strip()
        live_px = live_ltps.get(tok)
        if live_px is not None and spot > 0:
            feats = self._overlay_live_option_features(
                feats,
                option_type=contract.option_type,
                strike=float(contract.strike),
                spot=spot,
                expiry_ts=expiry_ts,
                ts=float(signal_ts),
                ltp=float(live_px),
            )
        return feats

    def _live_chain_snap(self, ctx, contracts, signal_ts: float, live_ltps: dict[str, float]):
        if self._chain_ctx_builder is None:
            self._chain_ctx_builder = ChainContextBuilder(strike_step=int(ctx.strike_step or 50))
        return chain_context_for_band(
            ctx,
            contracts,
            signal_ts,
            builder=self._chain_ctx_builder,
            live_ltps=live_ltps,
        )

    def _update_feature_coverage(
        self,
        scorer,
        snap,
        ctx,
        contracts,
        signal_ts: float,
    ) -> None:
        rows = list(getattr(snap, "rows", ()) or ())
        if not rows or scorer is None or not scorer.is_loaded():
            return
        live_ltps = self._live_ltp_map()
        chain_snap = self._live_chain_snap(ctx, contracts, signal_ts, live_ltps)
        expiry_ts = float(getattr(ctx, "expiry_ts", signal_ts) or signal_ts)
        spot = float(snap.spot or chain_snap.spot or 0.0)
        probe_row = None
        atm = getattr(snap, "atm_strike", None)
        if atm is not None:
            for row in rows:
                if (
                    int(round(float(row.contract.strike))) == int(atm)
                    and row.contract.option_type == "CE"
                ):
                    probe_row = row
                    break
        if probe_row is None:
            probe_row = max(
                rows,
                key=lambda r: len(r.result.features or {}),
            )
        feats = self._build_live_row_features(
            probe_row,
            snap=snap,
            signal_ts=signal_ts,
            ctx=ctx,
            contracts=contracts,
            live_ltps=live_ltps,
            expiry_ts=expiry_ts,
            spot=spot,
        )
        self._last_feature_coverage = audit_registry_feature_coverage(
            feats,
            scorer._registry_features,
            model_name=str(scorer.model_name or scorer.stamp or ""),
            probe_symbol=str(probe_row.contract.symbol or ""),
            probe_token=str(probe_row.contract.token or ""),
            signal_ts=float(signal_ts),
        )
        self._last_feature_parity = live_parity_report(
            registry_features=feats,
            legacy_features=build_legacy_live_features(probe_row),
            required=scorer._registry_features,
        )

    def _score_band_snapshot_live(self, scorer, snap, ctx, contracts, signal_ts: float):
        live_ltps = self._live_ltp_map()
        expiry_ts = float(getattr(ctx, "expiry_ts", signal_ts) or signal_ts)
        spot = float(snap.spot or 0.0)
        if spot <= 0:
            for row in getattr(snap, "rows", ()) or ():
                try:
                    spot = float(row.spot or 0)
                except (TypeError, ValueError):
                    spot = 0.0
                if spot > 0:
                    break
        scored = []
        for row in getattr(snap, "rows", ()) or ():
            contract = row.contract
            feats = self._build_live_row_features(
                row,
                snap=snap,
                signal_ts=signal_ts,
                ctx=ctx,
                contracts=contracts,
                live_ltps=live_ltps,
                expiry_ts=expiry_ts,
                spot=spot,
            )
            tok = str(contract.token).strip()
            scored.append(
                scorer.score_features(
                    feats,
                    ts=float(signal_ts),
                    token=tok,
                    symbol=str(contract.symbol or ""),
                    option_type=str(contract.option_type),
                    strike=float(contract.strike),
                    fill_missing=0.0,
                )
            )
        return scored

    def _feature_rows_payload(self, snap) -> list[dict[str, Any]]:
        if snap is None:
            return []
        rows: list[dict[str, Any]] = []
        for row in getattr(snap, "rows", ()) or ():
            feats = dict(row.result.features or {})
            cols = len(feats)
            rows.append(
                {
                    "symbol": row.contract.symbol,
                    "token": row.contract.token,
                    "strike": row.contract.strike,
                    "option_type": row.contract.option_type,
                    "window_complete": bool(row.result.window_complete),
                    "model_complete": bool(row.result.model_complete),
                    "feature_count": cols,
                    "missing_model_columns": list(row.result.missing_model_columns or ()),
                    "features": feats,
                }
            )
        return rows

    def _probe_features_from_snap(self, snap) -> tuple[dict[str, Any], str]:
        """Pick ATM CE row when possible; else the row with most populated features."""
        rows = getattr(snap, "rows", ()) or ()
        if not rows:
            return {}, ""
        atm = getattr(snap, "atm_strike", None)
        if atm is not None:
            for row in rows:
                if int(round(float(row.contract.strike))) == int(atm) and row.contract.option_type == "CE":
                    return dict(row.result.features or {}), str(row.contract.symbol or "")
        best = max(
            rows,
            key=lambda r: (
                len(r.result.features or {}),
                feature_completeness_ratio(r.result.features or {}),
            ),
        )
        return dict(best.result.features or {}), str(best.contract.symbol or "")

    def _record_xgb_feature_log(
        self,
        signal_ts: float,
        *,
        status: str,
        ctx=None,
        contracts: Sequence[Any] | None = None,
        snap=None,
        scored: Sequence[Any] | None = None,
        top=None,
    ) -> None:
        """Append feature log row from the 1s live XGB eval (no duplicate model run)."""
        if not self.feature_probe_enabled:
            return
        probe_ts = float(math.floor(signal_ts))
        window_diag = None
        snap_log = snap
        feature_rows: list[dict[str, Any]] = []
        probe_features: dict[str, Any] | None = None
        probe_symbol = ""
        if ctx is not None:
            index_tl = ctx.index_timeline(probe_ts)
            window_diag = feature_window_diagnostics(
                probe_ts,
                index_tl,
                open_ts=ctx.open_ts,
                close_ts=ctx.close_ts,
                missing_buckets=ctx.missing_buckets,
                live=True,
            )
            window_diag["index_source"] = getattr(ctx, "_index_tl_source", "ring")
            if contracts and (snap_log is None or not getattr(snap_log, "window_complete", False)):
                snap_log = evaluate_band_at_ts(ctx, probe_ts, contracts, probe=True)
        if snap_log is not None:
            feature_rows = self._feature_rows_payload(snap_log)
            feats, probe_symbol = self._probe_features_from_snap(snap_log)
            if feats:
                probe_features = feats
        leaders: list[dict[str, Any]] = []
        if scored:
            leaders = sorted(
                [s for s in scored if getattr(s, "scorable", False) and s.score is not None],
                key=lambda s: float(s.score or 0.0),
                reverse=True,
            )[:5]
            leaders = [
                {
                    "symbol": s.symbol,
                    "score": s.score,
                    "P_hit": s.P_hit,
                    "delta_band": s.delta_band,
                }
                for s in leaders
            ]
        log_top = top
        if log_top is not None and snap_log is not None:
            scored_feats = self._features_for_token(snap_log, log_top.token)
            if scored_feats:
                probe_features = scored_feats
                probe_symbol = str(log_top.symbol or probe_symbol)
        self._append_feature_probe_log(
            probe_ts,
            status=status,
            snap=snap_log,
            top=log_top,
            leaders=leaders,
            top_features=probe_features,
            window_diag=window_diag,
            feature_rows=feature_rows,
            probe_symbol=probe_symbol,
        )

    def _append_feature_probe_log(
        self,
        probe_ts: float,
        *,
        status: str,
        snap,
        top,
        leaders: Sequence[Any],
        top_features: dict[str, Any] | None,
        window_diag: dict[str, Any] | None = None,
        feature_rows: Sequence[dict[str, Any]] | None = None,
        probe_symbol: str = "",
    ) -> None:
        self._feature_probe_seq += 1
        top_payload: dict[str, Any] = {}
        if top is not None:
            top_payload = {
                "symbol": top.symbol,
                "token": top.token,
                "score": top.score,
                "P_hit": top.P_hit,
                "delta_band": top.delta_band,
                "delta": top.delta,
                "ltp": top.ltp,
                "pred_max_return": top.pred_max_return,
                "pred_min_return": top.pred_min_return,
            }
        row = {
            "id": f"fp_{self._feature_probe_seq}",
            "ts": probe_ts,
            "time": fmt_ist_hms(probe_ts),
            "spot": snap.spot if snap is not None else None,
            "atm_strike": snap.atm_strike if snap is not None else None,
            "window_complete": bool(snap.window_complete) if snap is not None else False,
            "row_count": snap.row_count if snap is not None else 0,
            "model_complete_count": snap.model_complete_count if snap is not None else 0,
            "status": status,
            "top": top_payload,
            "leaders": list(leaders),
            "top_features": top_features,
            "window_diag": window_diag,
            "feature_rows": list(feature_rows or ()),
            "probe_symbol": probe_symbol,
        }
        self._feature_probe_log.append(row)
        if len(self._feature_probe_log) > _MAX_FEATURE_PROBE_LOG:
            self._feature_probe_log = self._feature_probe_log[-_MAX_FEATURE_PROBE_LOG:]

    def _evaluate_signal_slot(self, signal_ts: float) -> None:
        mode_key = parse_signal_mode(self.signal_mode)
        try:
            if mode_key == "xgboost_band":
                self._evaluate_signal_slot_xgboost(signal_ts)
            else:
                self._evaluate_signal_slot_ranking(signal_ts)
        except Exception as exc:
            # Never swallow evaluation failures silently; emit a visible SKIP so
            # the decision stream explains why no ENTER/EXIT is appearing.
            err = f"Evaluation error: {type(exc).__name__}: {exc}"
            self._status = "Evaluation error"
            self._record_xgb_feature_log(signal_ts, status=err[:120])
            self._push_signal(
                signal_ts,
                None,
                suppressed=True,
                reason=err,
                mode=mode_key,
            )

    def _evaluate_signal_slot_ranking(self, signal_ts: float) -> None:
        manager = self.state.target_manager
        if manager is None:
            return
        open_count = len(manager.get_open_ml_entries())
        if position_limit_reached(open_count, self.max_positions):
            limit_txt = format_max_positions(self.max_positions)
            self._push_signal(
                signal_ts,
                None,
                suppressed=True,
                reason=f"Position limit ({limit_txt})",
            )
            return

        ranked = _pick_ranked_candidate(self.state)
        if ranked is None:
            self._push_signal(signal_ts, None, suppressed=True, reason="No ranked candidate")
            return

        token = str(ranked.token or "").strip()
        if not token:
            self._push_signal(signal_ts, None, suppressed=True, reason="Ranked row missing token")
            return

        ltp = self.state.latest_ltps_tracker.get(token)
        if ltp is None:
            ltp = self.state.latest_ltps.get(token)
        try:
            entry_ltp = float(ltp)
        except (TypeError, ValueError):
            entry_ltp = 0.0
        if entry_ltp <= 0:
            self._push_signal(signal_ts, ranked, suppressed=True, reason="No live LTP")
            return

        strike, opt_type = parse_option_symbol(ranked.symbol)
        tgt_pct, sl_pct = target_sl_pct_for_premium(entry_ltp)
        entry = {
            "token": token,
            "symbol": ranked.symbol,
            "strike": strike,
            "opt_type": opt_type,
            "target_pct": tgt_pct,
            "sl_pct": sl_pct,
            "entry_ltp": round(entry_ltp, 2),
            "ml_score": round(float(ranked.ranking_score), 2),
            "signal_score": round(float(ranked.signal_score), 2),
        }
        self._push_signal(
            signal_ts,
            ranked,
            suppressed=False,
            reason="",
            entry=entry,
            mode="ranking_proxy",
        )

    def _evaluate_signal_slot_xgboost(self, signal_ts: float) -> None:
        manager = self.state.target_manager
        if manager is None:
            return
        hub = self.trade_decision_hub
        open_count = len(manager.get_open_ml_entries())
        at_position_limit = (
            hub is None and position_limit_reached(open_count, self.max_positions)
        )
        limit_txt = format_max_positions(self.max_positions) if at_position_limit else ""

        if not self._ensure_band_scorer_loaded():
            self._record_xgb_feature_log(signal_ts, status="Models unavailable")
            self._push_signal(
                signal_ts,
                None,
                suppressed=True,
                reason="Models unavailable",
                mode="xgboost_band",
            )
            return

        contracts = band_contracts_from_state(self.state)
        if not contracts:
            self._record_xgb_feature_log(signal_ts, status="No chain contracts")
            self._push_signal(
                signal_ts,
                None,
                suppressed=True,
                reason="No chain contracts",
                mode="xgboost_band",
            )
            return

        ctx = self._band_context()
        if ctx is None:
            self._record_xgb_feature_log(signal_ts, status="Tick ring unavailable")
            self._push_signal(
                signal_ts,
                None,
                suppressed=True,
                reason="Tick ring unavailable",
                mode="xgboost_band",
            )
            return

        snap = evaluate_band_at_ts(ctx, signal_ts, contracts)
        if not snap.window_complete or snap.spot is None:
            reason = (
                "Feature window incomplete"
                if not snap.window_complete
                else "No index spot"
            )
            self._record_xgb_feature_log(
                signal_ts,
                status=reason,
                ctx=ctx,
                contracts=contracts,
                snap=snap,
            )
            self._push_signal(
                signal_ts,
                None,
                suppressed=True,
                reason=reason,
                mode="xgboost_band",
            )
            return

        scorer = self._band_scorer_instance()
        self._update_feature_coverage(scorer, snap, ctx, contracts, signal_ts)
        scored = self._score_band_snapshot_live(scorer, snap, ctx, contracts, signal_ts)
        top = scorer.pick_top_scored(scored, min_score=self.min_score)
        best_top = self._best_scored(scored)
        if best_top is None:
            log_status = "No scorable rows"
        elif top is None or not top.scorable:
            log_status = f"No score >= {self.min_score:g}"
        else:
            log_status = "Scored"
        if at_position_limit:
            log_status = (
                f"{log_status} · no entry (limit {limit_txt})"
                if log_status != "Scored"
                else f"Scored · no entry (limit {limit_txt})"
            )
        self._record_xgb_feature_log(
            signal_ts,
            status=log_status,
            ctx=ctx,
            contracts=contracts,
            snap=snap,
            scored=scored,
            top=best_top,
        )

        if at_position_limit:
            self._record_decision_debug(
                signal_ts=signal_ts,
                scorer=scorer,
                skip_reason=f"Position limit reached ({limit_txt})",
                suppressed=True,
                decision_top=top,
                best_top=best_top,
                scored=scored,
            )
            self._push_signal(
                signal_ts,
                None,
                suppressed=True,
                reason=f"Position limit reached ({limit_txt})",
                mode="xgboost_band",
                **self._scored_skip_context(scored, best_top),
            )
            return

        if top is None or not top.scorable:
            skip_reason = (
                "No scorable rows"
                if best_top is None
                else f"No score >= {self.min_score:g}"
            )
            self._record_decision_debug(
                signal_ts=signal_ts,
                scorer=scorer,
                skip_reason=skip_reason,
                suppressed=True,
                decision_top=top,
                best_top=best_top,
                scored=scored,
            )
            self._push_signal(
                signal_ts,
                None,
                suppressed=True,
                reason=skip_reason,
                mode="xgboost_band",
                **self._scored_skip_context(scored, best_top),
            )
            return

        token = str(top.token or "").strip()
        symbol = str(top.symbol or "").strip()
        for c in contracts:
            if str(c.token).strip() == token and c.symbol:
                symbol = str(c.symbol).strip()
                break
        if not token:
            self._push_signal(
                signal_ts,
                None,
                suppressed=True,
                reason="Top row missing token",
                mode="xgboost_band",
                score=top.score,
            )
            return

        ltp = self.state.latest_ltps_tracker.get(token)
        if ltp is None:
            ltp = self.state.latest_ltps.get(token)
        if ltp is None and top.ltp is not None:
            ltp = top.ltp
        try:
            entry_ltp = float(ltp)
        except (TypeError, ValueError):
            entry_ltp = 0.0
        if entry_ltp <= 0:
            self._push_signal(
                signal_ts,
                None,
                suppressed=True,
                reason="No live LTP",
                mode="xgboost_band",
                symbol=symbol,
                token=token,
                score=top.score,
            )
            return

        if hub is not None:
            ok, gate_reason = hub.check_enter_gates(symbol)
            if not ok:
                self._push_signal(
                    signal_ts,
                    None,
                    suppressed=True,
                    reason=gate_reason,
                    mode="xgboost_band",
                    symbol=symbol,
                    token=token,
                    score=top.score,
                    p_hit=top.P_hit,
                )
                return

        strike = float(top.strike) if top.strike else None
        opt_type = str(top.option_type or "").upper() or None
        if strike is None or not opt_type:
            strike, opt_type = parse_option_symbol(symbol)

        tgt_pct, sl_pct = target_sl_pct_for_premium(entry_ltp)
        entry = {
            "token": token,
            "symbol": symbol,
            "strike": strike,
            "opt_type": opt_type,
            "target_pct": tgt_pct,
            "sl_pct": sl_pct,
            "entry_ltp": round(entry_ltp, 2),
            "ml_score": round(float(top.score or 0.0), 3),
            "signal_score": round(float(top.P_hit or 0.0), 3),
        }
        self._record_decision_debug(
            signal_ts=signal_ts,
            scorer=scorer,
            skip_reason="",
            suppressed=False,
            decision_top=top,
            best_top=best_top,
            scored=scored,
        )
        self._push_signal(
            signal_ts,
            None,
            suppressed=False,
            reason="",
            entry=entry,
            mode="xgboost_band",
            symbol=symbol,
            token=token,
            score=top.score,
            p_hit=top.P_hit,
        )

    def _push_signal(
        self,
        signal_ts: float,
        ranked: RankedSymbol | None,
        *,
        suppressed: bool,
        reason: str,
        entry: dict[str, Any] | None = None,
        mode: str | None = None,
        symbol: str = "",
        token: str = "",
        score: float | None = None,
        p_hit: float | None = None,
        score_note: str = "",
        strike: float | None = None,
        opt_type: str = "",
    ) -> None:
        mode_key = parse_signal_mode(mode or self.signal_mode)
        display_score = score
        if ranked is not None:
            symbol = ranked.symbol
            token = ranked.token or ""
            display_score = ranked.ranking_score
        row = {
            "ts": signal_ts,
            "time": fmt_ist_hms(signal_ts),
            "suppressed": suppressed,
            "reason": reason,
            "symbol": symbol or "—",
            "token": token,
            "ranking_score": ranked.ranking_score if ranked else None,
            "signal_score": ranked.signal_score if ranked else None,
            "display_score": display_score,
            "mode": mode_key,
            "entered": entry is not None and not suppressed,
            "trade_id": entry.get("trade_id") if entry else "",
        }
        self._recent_signals.append(row)
        if len(self._recent_signals) > 80:
            self._recent_signals = self._recent_signals[-80:]

        hub = self.trade_decision_hub
        if hub is not None:
            hub.on_signal(
                self,
                signal_ts=signal_ts,
                suppressed=suppressed,
                reason=reason,
                entry=entry,
                symbol=symbol or row.get("symbol") or "",
                token=token or row.get("token") or "",
                score=display_score if isinstance(display_score, (int, float)) else score,
                p_hit=p_hit,
                score_note=score_note,
                strike=strike,
                opt_type=opt_type,
            )
            if entry is not None and entry.get("trade_id"):
                row["trade_id"] = str(entry.get("trade_id") or "")

        shadow = getattr(self, "trade_decision_shadow_client", None)
        if shadow is not None and not getattr(shadow, "server_brain", False):
            try:
                shadow.on_local_signal(
                    signal_ts=signal_ts,
                    suppressed=suppressed,
                    reason=reason,
                    symbol=symbol or row.get("symbol") or "",
                    score=float(display_score) if isinstance(display_score, (int, float)) else score,
                    entered=entry is not None and not suppressed,
                )
            except Exception:
                pass

    def _update_open_trades(self, now: float) -> None:
        manager = self.state.target_manager
        if manager is None:
            return
        for entry in manager.get_open_ml_entries():
            token = str(entry.get("token") or "")
            entry_ts = float(entry.get("entry_ts") or 0.0)
            entry_ltp = float(entry.get("entry_ltp") or 0.0)
            tgt = float(entry.get("target_pct") or 10.0)
            sl = float(entry.get("sl_pct") or 5.0)
            samples = _ring_samples(self.state, token)
            outcome, exit_ts, exit_ltp, elapsed = _scan_outcome_on_ring(
                samples, entry_ts, entry_ltp, tgt, sl,
            )
            if outcome == "timeout" and now < exit_ts - 0.5:
                continue
            if outcome in ("target", "sl") or (outcome == "timeout" and now >= exit_ts - 0.5):
                manager.close_ml_entry(
                    str(entry.get("ml_id")),
                    outcome_type=outcome,
                    exit_ts=exit_ts,
                    exit_ltp=round(exit_ltp, 2),
                    elapsed_sec=elapsed,
                )

    def summary(self) -> dict[str, Any]:
        manager = self.state.target_manager
        entries = manager.get_ml_entries() if manager else []
        open_n = len(manager.get_open_ml_entries()) if manager else 0
        closed = [e for e in entries if str(e.get("status")) == "CLOSED"]
        targets = sum(1 for e in closed if e.get("outcome_type") == "target")
        pnl = aggregate_ml_pnl(
            entries,
            self.state.latest_ltps_tracker or self.state.latest_ltps,
        )
        scorer = self._band_scorer
        model_stamp = (
            scorer.stamp
            if scorer is not None and scorer.is_loaded()
            else (self.model_name or self.model_stamp or "")
        )
        mode_key = parse_signal_mode(self.signal_mode)
        if mode_key == "xgboost_band":
            grid_start, grid_end = self._xgb_grid_bounds()
            slots_total = (
                int(grid_end - grid_start) + 1 if grid_end >= grid_start else 0
            )
            slots_done = self._xgb_eval_count
            if self._last_xgb_signal_ts < grid_start:
                next_signal_ts = grid_start
            elif self._last_xgb_signal_ts >= grid_end:
                next_signal_ts = None
            else:
                next_signal_ts = self._last_xgb_signal_ts + XGB_SIGNAL_INTERVAL_SEC
        else:
            slots_total = len(self._signal_times)
            slots_done = self._next_signal_idx
            next_signal_ts = (
                self._signal_times[self._next_signal_idx]
                if self._next_signal_idx < len(self._signal_times)
                else None
            )
        return {
            "armed": self.armed,
            "status": self._status,
            "signal_mode": parse_signal_mode(self.signal_mode),
            "signal_mode_label": format_signal_mode(self.signal_mode),
            "min_score": self.min_score,
            "model_stamp": model_stamp,
            "model_name": model_stamp,
            "model_load_error": self._model_load_error,
            "max_positions": self.max_positions,
            "max_positions_label": format_max_positions(self.max_positions),
            "open_trades": open_n,
            "closed_trades": len(closed),
            "target_hits": targets,
            "net_pnl_realized": pnl["net_pnl_realized"],
            "net_pnl_unrealized": pnl["net_pnl_unrealized"],
            "net_pnl_total": pnl["net_pnl_total"],
            "signal_slots_total": slots_total,
            "signal_slots_done": slots_done,
            "next_signal_ts": next_signal_ts,
            "ring_tokens": len(self.state.tick_ring_store.tracked_keys()),
            "recent_signals": list(self._recent_signals[-30:]),
            "feature_probe_enabled": self.feature_probe_enabled,
            "feature_probe_log": list(self._feature_probe_log[-80:]),
        }

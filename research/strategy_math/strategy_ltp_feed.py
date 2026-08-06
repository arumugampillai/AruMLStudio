"""In-app Angel WebSocket: strategy leg LTP + optional option chain + index spot."""
from __future__ import annotations

import threading
import time
from collections import defaultdict
from typing import Any, Callable, Iterable, Mapping

from api.chain_quote import ANGEL_INDEX_SPOT

_WS_CORRELATION_ID = "neo_angel_market_v1"
_WS_LANES = ("NIFTY", "SENSEX")
_EXCHANGE_TYPE_TO_LANE = {1: "NIFTY", 2: "NIFTY", 3: "SENSEX", 4: "SENSEX"}
_CORRELATION_BY_LANE = {
    "NIFTY": "neo_angel_market_nse",
    "SENSEX": "neo_angel_market_bse",
}
_PRIMARY_CHAIN_LANE = "NIFTY"
_SUBSCRIBE_MODE = 3  # SNAP_QUOTE
_SUBSCRIBE_CHUNK = 50
_ANGEL_EX_REST = {1: "NSE", 2: "NFO", 3: "BSE", 4: "BFO"}
_UI_DEBOUNCE_SEC = 0.35
_CHAIN_UI_DEBOUNCE_SEC = 0.12

_FEED: "StrategyLtpFeed | None" = None
_FEED_LOCK = threading.Lock()


def get_strategy_ltp_feed() -> "StrategyLtpFeed | None":
    return _FEED


def get_strategy_ltp_cache() -> dict[str, float]:
    feed = _FEED
    if feed is None:
        return {}
    return feed.ltp_by_token


def install_strategy_ltp_feed(feed: "StrategyLtpFeed") -> "StrategyLtpFeed":
    global _FEED
    with _FEED_LOCK:
        _FEED = feed
    return feed


def _parse_tick_ltp(message: Any) -> tuple[str | None, float | None]:
    if not isinstance(message, dict):
        return None, None
    token = (
        message.get("token")
        or message.get("symboltoken")
        or message.get("tk")
        or message.get("instrument_token")
    )
    if token is None:
        return None, None
    token_s = str(token).strip()
    raw = message.get("last_traded_price") or message.get("ltp") or message.get("lp")
    if raw in (None, ""):
        return token_s, None
    try:
        ltp = float(raw) / 100.0
    except (TypeError, ValueError):
        return token_s, None
    if ltp <= 0:
        return token_s, None
    return token_s, ltp


class StrategyLtpFeed:
    """Angel SmartWebSocketV2: strategy legs, option chain (pref), index spot."""

    def __init__(
        self,
        *,
        root: Any = None,
        on_ltps_updated: Callable[[], None] | None = None,
        on_fallback: Callable[[], None] | None = None,
    ) -> None:
        self._root = root
        self._on_ltps_updated = on_ltps_updated
        self._on_fallback = on_fallback
        self._state: Any = None
        self.ltp_by_token: dict[str, float] = {}
        self._index_spot: dict[str, float] = {}
        self._index_prev_close: dict[str, float] = {}
        self._index_last_message: dict[str, dict] = {}
        self._chain_angel_to_neo: dict[str, str] = {}
        self._chain_token_ex: dict[str, int] = {}
        self._chain_ltp_by_neo: dict[str, float] = {}
        self._strategy_token_ex: dict[str, int] = {}
        self._index_enabled = False
        self._index_data_lock = threading.Lock()
        self._subscribed_by_ex: dict[int, set[str]] = defaultdict(set)
        self._pending_by_ex: dict[int, set[str]] = defaultdict(set)
        self._sws_by_lane: dict[str, Any] = {}
        self._ws_threads: dict[str, threading.Thread | None] = {
            lane: None for lane in _WS_LANES
        }
        self._connected_by_lane: dict[str, bool] = {lane: False for lane in _WS_LANES}
        self._start_lock = threading.Lock()
        self._started = False
        self._stop = False
        self._last_ui_notify = 0.0
        self._last_chain_ui_notify = 0.0
        self._chain_refresh_after_id: str | None = None
        self._chain_rest_bootstrap_after_id: str | None = None
        self._chain_rest_bootstrap_rows: list[Mapping[str, Any]] | None = None
        self.neo_fallback = False

    def attach_state(self, state: Any) -> None:
        self._state = state

    @property
    def online(self) -> bool:
        return any(self._connected_by_lane.values())

    @staticmethod
    def _lane_for_exchange_type(exchange_type: int) -> str:
        return _EXCHANGE_TYPE_TO_LANE.get(int(exchange_type), _PRIMARY_CHAIN_LANE)

    @property
    def subscribed_tokens(self) -> set[str]:
        out: set[str] = set()
        for bucket in self._subscribed_by_ex.values():
            out |= bucket
        return out

    @property
    def pending_tokens(self) -> set[str]:
        out: set[str] = set()
        for bucket in self._pending_by_ex.values():
            out |= bucket
        return out

    @property
    def started(self) -> bool:
        return self._started

    def set_index_spot_enabled(self, enabled: bool) -> None:
        from ui.index_switch_perf import log, timed

        enabled = bool(enabled)
        turning_on = enabled and not self._index_enabled
        log("set_index_spot_enabled", enabled=int(enabled), turning_on=int(turning_on))
        if not enabled:
            self._index_enabled = False
            self._index_spot.clear()
            with timed("set_index_spot_enabled.sync_off"):
                self._sync_all_subscriptions()
            return
        self._index_enabled = True
        if turning_on and self.chain_uses_angel():
            with timed("bootstrap_index_spot_rest"):
                self._bootstrap_index_spot_rest()
        with timed("set_index_spot_enabled.sync"):
            self._sync_all_subscriptions()
        if turning_on and self.chain_uses_angel():
            with timed("force_resubscribe_index_spot"):
                self._force_resubscribe_index_spot()
            with timed("reapply_index_change_fields"):
                self._reapply_index_change_fields()
            with timed("replay_index_spot_headers"):
                self._replay_index_spot_headers()

    def _bootstrap_index_spot_rest(self) -> None:
        """Seed NIFTY + SENSEX change fields from Angel REST when WS only sends LTP."""
        st = self._state
        if st is None:
            return
        client = getattr(st, "angelone_client", None)
        if client is None:
            try:
                from angelone.smart_api_client import ensure_angel_ready, smartApi

                client = ensure_angel_ready(smartApi)
            except Exception:
                return
        if client is None:
            return
        for key in ("NIFTY", "SENSEX"):
            try:
                from shared.data.data_api_utils import (
                    _fetch_index_from_angel,
                    index_display_prev_close,
                )

                self._seed_index_prev_close_from_data(key)
                _fetch_index_from_angel(client, key, st.index_data, force=True)
                self._seed_index_prev_close_from_data(key)
                price = st.index_data.get(key, {}).get("price")
                if price:
                    try:
                        bucket = st.index_data.get(key)
                        prev = index_display_prev_close(
                            key,
                            float(price),
                            feed_prev=self._index_prev_close.get(key),
                            index_data_row=bucket if isinstance(bucket, dict) else None,
                        )
                        if prev is not None:
                            self._index_prev_close[key] = prev
                    except (TypeError, ValueError):
                        pass
            except Exception:
                pass
        self._reapply_index_change_fields()

    def _seed_index_prev_close_from_data(self, index_key: str) -> None:
        """Remember previous close from REST/bootstrap ``index_data`` row."""
        from shared.data.data_api_utils import _valid_prev_close

        st = self._state
        if st is None:
            return
        key = str(index_key or "").upper()
        bucket = st.index_data.get(key) if isinstance(st.index_data, dict) else None
        if not isinstance(bucket, dict):
            return
        try:
            price = float(bucket.get("price") or 0)
            cng = float(str(bucket.get("cng", "0")).replace(",", ""))
        except (TypeError, ValueError):
            return
        if price <= 0:
            return
        prev = _valid_prev_close(price - cng, price)
        if prev is not None:
            self._index_prev_close[key] = prev

    def _reapply_index_change_fields(self) -> None:
        """Repaint header change from cached prev close (survives broker switch)."""
        st = self._state
        if st is None:
            return
        from shared.data.data_api_utils import index_display_prev_close, normalize_index_quote_fields

        with self._index_data_lock:
            for key in ("NIFTY", "SENSEX"):
                bucket = st.index_data.get(key)
                if not isinstance(bucket, dict):
                    continue
                try:
                    ltp = float(bucket.get("price") or 0)
                except (TypeError, ValueError):
                    ltp = 0.0
                if ltp <= 0:
                    cached = self._index_spot.get(key)
                    if cached is not None and cached > 0:
                        ltp = cached
                        bucket["price"] = ltp
                if ltp <= 0:
                    continue
                prev = index_display_prev_close(
                    key,
                    ltp,
                    feed_prev=self._index_prev_close.get(key),
                    index_data_row=bucket,
                )
                if prev is None:
                    try:
                        cng_f = float(str(bucket.get("cng", "0")).replace(",", ""))
                        if abs(cng_f) > 0.005:
                            prev = ltp - cng_f
                    except (TypeError, ValueError):
                        prev = None
                if prev is None:
                    continue
                self._index_prev_close[key] = prev
                cng = ltp - prev
                _, c, n = normalize_index_quote_fields(ltp, cng, 0.0)
                bucket["cng"] = f"{c:.2f}"
                bucket["nc"] = f"{n:.2f}"
        top_menu = getattr(st, "top_menu", None)
        if top_menu is not None:
            try:
                top_menu.refresh_spot_price(st.index_data)
            except Exception:
                pass

    def _replay_index_spot_headers(self) -> None:
        """Re-apply last Angel index SNAP after resubscribe (broker switch)."""
        for key, ltp in list(self._index_spot.items()):
            if ltp <= 0:
                continue
            msg = self._index_last_message.get(str(key).upper())
            self._push_index_spot(key, ltp, message=msg)

    def _force_resubscribe_index_spot(self) -> None:
        """Unsub + resub index tokens so Angel sends fresh SNAP with prev close."""
        if not self._index_enabled or not self.online:
            return
        for _key, cfg in ANGEL_INDEX_SPOT.items():
            tok = str(cfg.get("token") or "").strip()
            if not tok:
                continue
            try:
                ex = int(cfg["exchange_type"])
            except (TypeError, ValueError):
                continue
            subscribed = self._subscribed_by_ex.setdefault(ex, set())
            if tok in subscribed:
                self._unsubscribe_chunks(ex, [tok])
                subscribed.discard(tok)
            self._subscribe_chunks(ex, [tok])

    def index_spot(self, index_key: str) -> float | None:
        return self._index_spot.get(str(index_key or "").upper())

    def chain_uses_angel(self) -> bool:
        from api.chain_quote import option_chain_via_angel_enabled

        if not option_chain_via_angel_enabled():
            return False
        if self.neo_fallback:
            return False
        return self._angel_session_ok()

    def ensure_started(self) -> None:
        if self._stop:
            return
        with self._start_lock:
            if self._stop:
                return
            if not self._angel_session_ok():
                self.neo_fallback = True
                return
            self.neo_fallback = False
            self._started = True
            for lane in _WS_LANES:
                t = self._ws_threads.get(lane)
                if t is not None and t.is_alive():
                    continue
                self._ws_threads[lane] = threading.Thread(
                    target=self._run_ws_lane,
                    args=(lane,),
                    daemon=True,
                    name=f"angel-market-ws-{lane.lower()}",
                )
                self._ws_threads[lane].start()

    def stop(self) -> None:
        self._stop = True
        for sws in list(self._sws_by_lane.values()):
            if sws is not None:
                try:
                    sws.close_connection()
                except Exception:
                    pass
        self._sws_by_lane.clear()
        for lane in _WS_LANES:
            self._connected_by_lane[lane] = False

    def sync_open_entries(self, entries: Iterable[Mapping[str, Any]]) -> None:
        from research.strategy_math.strategy_tracker import leg_angel_token

        ex_map: dict[str, int] = {}
        for entry in entries:
            if str(entry.get("status") or "").upper() != "OPEN":
                continue
            index = str(entry.get("index") or "NIFTY").upper()
            ex = 4 if index == "SENSEX" else 2
            for leg in entry.get("legs") or []:
                tok = leg_angel_token(leg)
                if tok:
                    ex_map[tok] = ex
        self._strategy_token_ex = ex_map
        self.ensure_started()
        self._sync_all_subscriptions()

    def sync_chain_tokens(self, rows: Iterable[Mapping[str, Any]]) -> None:
        from ui.index_switch_perf import log, timed

        with timed("sync_chain_tokens.parse"):
            row_list = list(rows)
            mapping: dict[str, str] = {}
            ex_map: dict[str, int] = {}
            for row in row_list:
                neo = str(row.get("neo_token") or "").strip()
                angel = str(row.get("angel_token") or "").strip()
                if not angel:
                    continue
                try:
                    ex = int(row.get("exchange_type") or 2)
                except (TypeError, ValueError):
                    ex = 2
                ex_map[angel] = ex
                if neo:
                    mapping[angel] = neo
        prev_ex = frozenset(self._chain_token_ex.values())
        prev_neo = frozenset(self._chain_angel_to_neo.values())
        new_ex = frozenset(ex_map.values())
        new_neo = frozenset(mapping.values())
        exchange_changed = bool(prev_ex and prev_ex != new_ex)
        index_switch = bool(prev_neo != new_neo or prev_ex != new_ex)
        log(
            "sync_chain_tokens.flags",
            rows=len(row_list),
            mapped=len(mapping),
            index_switch=int(index_switch),
            exchange_changed=int(exchange_changed),
            prev_ex=sorted(prev_ex),
            new_ex=sorted(new_ex),
        )
        self._chain_angel_to_neo = mapping
        self._chain_token_ex = ex_map
        if mapping:
            active_neo = frozenset(mapping.values())
            self._chain_ltp_by_neo = {
                k: v for k, v in self._chain_ltp_by_neo.items() if k in active_neo
            }
        if self.chain_uses_angel():
            with timed("sync_chain_tokens.ensure_started"):
                self.ensure_started()
        with timed("sync_chain_tokens.sync_all"):
            self._sync_all_subscriptions()
        if self.chain_uses_angel() and mapping:
            if prev_neo == new_neo and prev_ex == new_ex:
                log("sync_chain_tokens.path", path="unchanged_replay")
                with timed("replay_chain_ltps"):
                    self.replay_chain_ltps()
                with timed("bootstrap_chain_ltps_rest"):
                    self.bootstrap_chain_ltps_rest(row_list)
                self._schedule_chain_ltp_refresh()
            else:
                # Re-enable after Neo, index change, or window resize — need SNAP + cache replay.
                # (The old index_switch-only path skipped replay after clear_chain_ltp_cache.)
                log(
                    "sync_chain_tokens.path",
                    path="chain_tokens_refresh",
                    index_switch=int(index_switch),
                )
                with timed("force_resubscribe_chain_tokens"):
                    self._force_resubscribe_chain_tokens()
                with timed("replay_chain_ltps"):
                    self.replay_chain_ltps()
                with timed("bootstrap_chain_ltps_rest"):
                    self.bootstrap_chain_ltps_rest(row_list)
                self._schedule_chain_ltp_refresh()
                if index_switch:
                    self._schedule_deferred_chain_rest_bootstrap(row_list, delay_ms=350)

    def _schedule_deferred_chain_rest_bootstrap(
        self,
        rows: Iterable[Mapping[str, Any]],
        *,
        delay_ms: int = 350,
    ) -> None:
        """Fill missing chain LTPs via REST after WS subscribe (non-blocking index switch)."""
        from ui.index_switch_perf import log, timed

        self._chain_rest_bootstrap_rows = list(rows)
        root = self._root
        log("schedule_deferred_chain_rest_bootstrap", delay_ms=delay_ms, rows=len(self._chain_rest_bootstrap_rows))
        if self._chain_rest_bootstrap_after_id is not None and root is not None:
            try:
                root.after_cancel(self._chain_rest_bootstrap_after_id)
            except Exception:
                pass
            self._chain_rest_bootstrap_after_id = None

        def _do() -> None:
            self._chain_rest_bootstrap_after_id = None
            pending = self._chain_rest_bootstrap_rows or []
            self._chain_rest_bootstrap_rows = None
            if not pending or not self.chain_uses_angel():
                log("deferred_chain_rest_bootstrap.skip", pending=len(pending))
                return
            with timed("deferred_bootstrap_chain_ltps_rest", rows=len(pending)):
                self.bootstrap_chain_ltps_rest(pending)
            self._schedule_chain_ltp_refresh()

        if root is not None:
            try:
                self._chain_rest_bootstrap_after_id = root.after(delay_ms, _do)
                return
            except Exception:
                pass
        _do()

    def _angel_session_ok(self) -> bool:
        try:
            from angelone.smart_api_client import angel_has_session, smartApi

            return bool(angel_has_session(smartApi))
        except Exception:
            return False

    def _desired_by_exchange(self) -> dict[int, set[str]]:
        desired: dict[int, set[str]] = defaultdict(set)
        for tok, ex in self._strategy_token_ex.items():
            if tok:
                desired[int(ex)].add(tok)
        if self.chain_uses_angel():
            for tok, ex in self._chain_token_ex.items():
                if tok:
                    desired[int(ex)].add(tok)
        if self._index_enabled and self.chain_uses_angel():
            for cfg in ANGEL_INDEX_SPOT.values():
                tok = str(cfg.get("token") or "").strip()
                if tok:
                    desired[int(cfg["exchange_type"])].add(tok)
        return desired

    def _sync_all_subscriptions(self) -> None:
        from ui.index_switch_perf import log, timed

        with timed("_sync_all_subscriptions"):
            desired = self._desired_by_exchange()
            if not self.online:
                self._pending_by_ex = desired
                log("_sync_all_subscriptions", online=0, pending_exchanges=len(desired))
                return
            unsub_total = 0
            sub_total = 0
            for ex_type, token_set in list(self._subscribed_by_ex.items()):
                remove = token_set - desired.get(ex_type, set())
                if remove:
                    unsub_total += len(remove)
                    with timed("_unsubscribe_chunks", ex=ex_type, count=len(remove)):
                        self._unsubscribe_chunks(ex_type, sorted(remove))
                    self._subscribed_by_ex[ex_type] -= remove
            for ex_type, token_set in desired.items():
                add = token_set - self._subscribed_by_ex.get(ex_type, set())
                if add:
                    sub_total += len(add)
                    with timed("_subscribe_chunks", ex=ex_type, count=len(add)):
                        self._subscribe_chunks(ex_type, sorted(add))
            log("_sync_all_subscriptions.summary", unsub=unsub_total, sub=sub_total)

    def _run_ws_lane(self, lane: str) -> None:
        lane_key = str(lane or "").strip().upper()
        if lane_key not in _WS_LANES:
            return
        while not self._stop:
            try:
                from SmartApi.smartWebSocketV2 import SmartWebSocketV2
                from angelone.smart_api_client import (
                    API_KEY,
                    CLIENT_ID,
                    angel_session_tokens,
                )

                jwt_token, feed_token = angel_session_tokens()
            except Exception as exc:
                print(f"Angel market feed [{lane_key}]: session unavailable:", exc)
                if lane_key == _PRIMARY_CHAIN_LANE:
                    self._started = False
                    self._enter_neo_fallback("no session")
                return

            sws = SmartWebSocketV2(jwt_token, API_KEY, CLIENT_ID, feed_token)
            self._sws_by_lane[lane_key] = sws

            def on_open(_wsapp, *, _lane=lane_key) -> None:
                self._connected_by_lane[_lane] = True
                if _lane == _PRIMARY_CHAIN_LANE:
                    self.neo_fallback = False
                self._flush_pending_for_lane(_lane)

            def on_data(_wsapp, message) -> None:
                if not isinstance(message, dict):
                    return
                token, ltp = _parse_tick_ltp(message)
                if token is None or ltp is None:
                    return
                self._apply_tick(token, ltp, message=message)

            def on_error(_wsapp, error, *, _lane=lane_key) -> None:
                print(f"Angel market feed WS error [{_lane}]:", error)

            def on_close(_wsapp, *, _lane=lane_key) -> None:
                self._connected_by_lane[_lane] = False
                self._sws_by_lane.pop(_lane, None)

            sws.on_open = on_open
            sws.on_data = on_data
            sws.on_error = on_error
            sws.on_close = on_close
            try:
                sws.connect()
            except Exception as exc:
                print(f"Angel market feed WS connect failed [{lane_key}]:", exc)
                if lane_key == _PRIMARY_CHAIN_LANE:
                    self._enter_neo_fallback("connect failed")
                    return
            finally:
                self._connected_by_lane[lane_key] = False
                self._sws_by_lane.pop(lane_key, None)
            if self._stop:
                return
            if lane_key == _PRIMARY_CHAIN_LANE:
                self._enter_neo_fallback("ws closed")
                self._started = False
                return
            time.sleep(2.0)

    def _enter_neo_fallback(self, reason: str) -> None:
        if self.neo_fallback:
            return
        print(f"Angel chain fallback → Neo WS ({reason})")
        self.neo_fallback = True
        self._chain_token_ex.clear()
        self._chain_angel_to_neo.clear()
        cb = self._on_fallback
        if cb:
            try:
                if self._root is not None:
                    self._root.after(0, cb)
                else:
                    cb()
            except Exception:
                pass

    def _apply_tick(self, angel_token: str, ltp: float, *, message: dict | None = None) -> None:
        changed = False
        if self._index_enabled:
            for key, meta in ANGEL_INDEX_SPOT.items():
                if str(meta.get("token") or "") == angel_token:
                    prev = self._index_spot.get(key)
                    self._index_spot[key] = ltp
                    if isinstance(message, dict):
                        self._index_last_message[key] = message
                    self._push_index_spot(key, ltp, message=message)
                    if message is not None:
                        try:
                            from ui.live_debug_panel import notify_angel_index_ws_tick

                            notify_angel_index_ws_tick(message, index_key=key, ltp=ltp)
                        except Exception:
                            pass
                    if prev != ltp:
                        changed = True
                    if changed:
                        self._notify_ltps_updated()
                    return
        neo = self._chain_angel_to_neo.get(angel_token)
        if neo:
            self._push_neo_ltp(neo, ltp)
            changed = True
        if angel_token in self._strategy_token_ex:
            prev = self.ltp_by_token.get(angel_token)
            self.ltp_by_token[angel_token] = ltp
            if prev != ltp:
                changed = True
        if changed:
            self._notify_ltps_updated()

    def _push_neo_ltp(self, neo_token: str, ltp: float, *, refresh_ui: bool = True) -> None:
        from api.chain_quote import normalize_neo_token

        st = self._state
        if st is None:
            return
        tok = normalize_neo_token(neo_token)
        if not tok:
            return
        st.latest_ltps[tok] = ltp
        st.latest_ltps_tracker[tok] = ltp
        st.last_option_ltp_update_time = time.time()
        self._chain_ltp_by_neo[tok] = ltp
        store = getattr(st, "tick_ring_store", None)
        if store is not None and ltp > 0:
            store.append(tok, ltp=float(ltp), ts=time.time())
        if refresh_ui:
            self._schedule_chain_ltp_refresh()

    def replay_chain_ltps(self) -> None:
        """Repaint chain LTP from last Angel ticks after broker switch / resubscribe."""
        from api.chain_quote import normalize_neo_token

        st = self._state
        active_neo = frozenset(
            t
            for t in (
                normalize_neo_token(n) for n in self._chain_angel_to_neo.values()
            )
            if t
        )
        for neo, ltp in self._chain_ltp_by_neo.items():
            tok = normalize_neo_token(neo)
            if tok in active_neo and ltp > 0:
                self._push_neo_ltp(tok, ltp, refresh_ui=False)
        # Strategy legs overlap chain subs — Angel may not re-SEND SNAP for those tokens.
        for angel, neo in self._chain_angel_to_neo.items():
            tok = normalize_neo_token(neo)
            if not tok or tok not in active_neo:
                continue
            if st is not None:
                cell = st.latest_ltps_tracker.get(tok)
                if cell is not None:
                    try:
                        if float(cell) > 0:
                            continue
                    except (TypeError, ValueError):
                        pass
            strat_ltp = self.ltp_by_token.get(str(angel).strip())
            if strat_ltp is not None and strat_ltp > 0:
                self._push_neo_ltp(tok, strat_ltp, refresh_ui=False)
        self._schedule_chain_ltp_refresh()

    def bootstrap_chain_ltps_rest(self, rows: Iterable[Mapping[str, Any]]) -> None:
        """REST LTP for chain tokens still missing after cache replay (broker switch)."""
        from api.chain_quote import normalize_neo_token
        from shared.data.data_api_utils import parse_angel_market_ltp_by_token
        from ui.index_switch_perf import log, timed

        st = self._state
        if st is None or not self.chain_uses_angel():
            return
        client = getattr(st, "angelone_client", None)
        if client is None:
            try:
                from angelone.smart_api_client import ensure_angel_ready, smartApi

                client = ensure_angel_ready(smartApi)
            except Exception:
                return
        if client is None:
            return

        by_ex: dict[str, list[tuple[str, str]]] = defaultdict(list)
        for row in rows:
            neo = normalize_neo_token(row.get("neo_token"))
            angel = str(row.get("angel_token") or "").strip()
            if not angel or not neo:
                continue
            cached = self._chain_ltp_by_neo.get(neo)
            if cached is not None and cached > 0:
                continue
            try:
                ex = int(row.get("exchange_type") or 2)
            except (TypeError, ValueError):
                ex = 2
            by_ex[_ANGEL_EX_REST.get(ex, "NFO")].append((angel, neo))

        rest_tokens = sum(len(pairs) for pairs in by_ex.values())
        log("bootstrap_chain_ltps_rest", rest_tokens=rest_tokens, exchanges=list(by_ex.keys()))
        for ex_name, pairs in by_ex.items():
            angel_to_neo = {a: n for a, n in pairs}
            tokens = list(angel_to_neo.keys())
            for i in range(0, len(tokens), _SUBSCRIBE_CHUNK):
                chunk = tokens[i : i + _SUBSCRIBE_CHUNK]
                try:
                    with timed("getMarketData.LTP", ex=ex_name, chunk=len(chunk)):
                        resp = client.getMarketData("LTP", {ex_name: chunk})
                except Exception as exc:
                    print(f"Angel chain REST bootstrap ({ex_name}):", exc)
                    continue
                filled = 0
                for angel_tok, ltp in parse_angel_market_ltp_by_token(resp).items():
                    neo = angel_to_neo.get(angel_tok)
                    if neo and ltp > 0:
                        self._push_neo_ltp(neo, ltp, refresh_ui=False)
                        filled += 1
                log("bootstrap_chain_ltps_rest.chunk", ex=ex_name, filled=filled, requested=len(chunk))
        self._schedule_chain_ltp_refresh()

    def _force_resubscribe_chain_tokens(self) -> None:
        """Unsub + resub chain tokens so Angel WS sends fresh SNAP quotes."""
        if not self.online or not self._chain_token_ex:
            return
        by_ex: dict[int, list[str]] = defaultdict(list)
        for tok, ex in self._chain_token_ex.items():
            if tok:
                by_ex[int(ex)].append(str(tok))
        for ex_type, tokens in by_ex.items():
            tok_set = set(tokens)
            subscribed = self._subscribed_by_ex.setdefault(int(ex_type), set())
            refresh = sorted(tok_set & subscribed)
            if refresh:
                self._unsubscribe_chunks(ex_type, refresh)
                subscribed.difference_update(refresh)
            self._subscribe_chunks(ex_type, sorted(tok_set))

    def _schedule_chain_ltp_refresh(self) -> None:
        st = self._state
        root = self._root
        if st is None or root is None:
            return
        now = time.monotonic()
        if now - self._last_chain_ui_notify < _CHAIN_UI_DEBOUNCE_SEC:
            if self._chain_refresh_after_id is not None:
                return
        self._last_chain_ui_notify = now

        def _do() -> None:
            self._chain_refresh_after_id = None
            oc = getattr(st, "option_chain", None)
            if oc is not None and hasattr(oc, "refresh_chain_ltp_cells"):
                try:
                    oc.refresh_chain_ltp_cells()
                except Exception:
                    pass

        try:
            if self._chain_refresh_after_id is not None:
                root.after_cancel(self._chain_refresh_after_id)
        except Exception:
            pass
        try:
            self._chain_refresh_after_id = root.after(0, _do)
        except Exception:
            _do()

    def _push_index_spot(
        self,
        index_key: str,
        ltp: float,
        *,
        message: dict | None = None,
    ) -> None:
        if not self._index_enabled:
            return
        st = self._state
        if st is None:
            return
        from shared.data.data_api_utils import (
            index_change_from_angel_ws_message,
            index_display_prev_close,
            normalize_index_quote_fields,
            _valid_prev_close,
        )

        key = str(index_key or "").upper()
        refresh_header = False
        with self._index_data_lock:
            bucket = st.index_data.setdefault(key, {})
            bucket["price"] = ltp

            feed_prev = self._index_prev_close.get(key)
            if isinstance(message, dict):
                raw_close = message.get("closed_price") or message.get("close")
                if raw_close not in (None, ""):
                    try:
                        candidate = float(raw_close) / 100.0
                        valid = _valid_prev_close(candidate, ltp)
                        if valid is not None:
                            self._index_prev_close[key] = valid
                            feed_prev = valid
                    except (TypeError, ValueError):
                        pass

            prev_close = index_display_prev_close(
                key,
                ltp,
                feed_prev=feed_prev,
                index_data_row=bucket,
            )
            if prev_close is not None and key not in self._index_prev_close:
                self._index_prev_close[key] = prev_close

            cng, nc = index_change_from_angel_ws_message(
                message or {},
                ltp,
                prev_close=prev_close,
            )
            if abs(cng) <= 0.005 and abs(nc) <= 0.005 and prev_close is not None:
                cng = ltp - prev_close
            if abs(cng) > 0.005 or abs(nc) > 0.005 or prev_close is not None:
                _, c, n = normalize_index_quote_fields(
                    ltp, cng, nc if abs(nc) > 0.005 else 0.0
                )
                bucket["cng"] = f"{c:.2f}"
                bucket["nc"] = f"{n:.2f}"
                remembered = _valid_prev_close(ltp - c, ltp)
                if remembered is not None:
                    self._index_prev_close[key] = remembered
                elif prev_close is not None:
                    self._index_prev_close[key] = prev_close
                refresh_header = True
        if refresh_header:
            top_menu = getattr(st, "top_menu", None)
            if top_menu is not None:
                try:
                    top_menu.refresh_spot_price(st.index_data)
                except Exception:
                    pass
        self._append_index_tick_ring(key, ltp, message=message)
        self._maybe_rebuild_chain_on_angel_spot(index_key, ltp)

    def _index_tick_ts(self, message: dict | None) -> float:
        if isinstance(message, dict):
            for field in (
                "exchange_timestamp",
                "exch_feed_time",
                "last_traded_timestamp",
                "ltt",
            ):
                raw = message.get(field)
                if raw in (None, ""):
                    continue
                try:
                    v = float(raw)
                except (TypeError, ValueError):
                    continue
                if v > 10_000_000_000_000:
                    return v / 1_000_000.0
                if v > 10_000_000_000:
                    return v / 1000.0
                if v > 1_000_000_000:
                    return v
        return time.time()

    def _append_index_tick_ring(
        self,
        index_key: str,
        ltp: float,
        *,
        message: dict | None = None,
    ) -> None:
        st = self._state
        if st is None:
            return
        store = getattr(st, "tick_ring_store", None)
        if store is None:
            return
        key = str(index_key or "").upper()
        if key not in ("NIFTY", "SENSEX"):
            return
        try:
            px = float(ltp)
        except (TypeError, ValueError):
            return
        if px <= 0:
            return
        store.append(key, ltp=px, ts=self._index_tick_ts(message))

    def _maybe_rebuild_chain_on_angel_spot(self, index_key: str, ltp: float) -> None:
        """Rebuild left chain once when Angel index spot arrives after a Neo-spot build."""
        from api.chain_quote import option_chain_via_angel_enabled

        if not self._index_enabled or not option_chain_via_angel_enabled():
            return
        if not self.chain_uses_angel():
            return
        st = self._state
        if st is None:
            return
        key = str(index_key or "").upper()
        top_menu = getattr(st, "top_menu", None)
        if top_menu is None:
            return
        try:
            selected = str(top_menu.index_var.get() or "").upper()
        except Exception:
            return
        if selected != key:
            return
        if getattr(st, "_chain_build_in_progress", False):
            return
        if getattr(st, "chain_built_with_angel_spot", False):
            return
        ce = getattr(st, "ce_tree", None)
        if ce is not None and ce.get_children():
            return
        pending = getattr(st, "_chain_angel_spot_rebuild_pending", None)
        if pending is not None:
            return
        st._chain_angel_spot_rebuild_pending = True
        root = self._root

        def _rebuild() -> None:
            st._chain_angel_spot_rebuild_pending = False
            if getattr(st, "chain_built_with_angel_spot", False):
                return
            cb = getattr(top_menu, "update_tables", None)
            if callable(cb):
                try:
                    cb()
                except Exception as exc:
                    print("Chain rebuild on Angel index spot:", exc)

        if root is not None:
            try:
                root.after(0, _rebuild)
            except Exception:
                st._chain_angel_spot_rebuild_pending = False
        else:
            _rebuild()

    def _flush_pending_for_lane(self, lane: str) -> None:
        lane_key = str(lane or "").strip().upper()
        for ex_type, tokens in list(self._pending_by_ex.items()):
            if self._lane_for_exchange_type(ex_type) != lane_key:
                continue
            if tokens:
                add = tokens - self._subscribed_by_ex.get(int(ex_type), set())
                if add:
                    self._subscribe_chunks(int(ex_type), sorted(add))
            self._pending_by_ex[int(ex_type)] = set()
        desired = self._desired_by_exchange()
        for ex_type, token_set in desired.items():
            if self._lane_for_exchange_type(ex_type) != lane_key:
                continue
            add = token_set - self._subscribed_by_ex.get(ex_type, set())
            if add:
                self._subscribe_chunks(ex_type, sorted(add))

    def _subscribe_chunks(self, exchange_type: int, tokens: list[str]) -> None:
        lane = self._lane_for_exchange_type(exchange_type)
        if not self._connected_by_lane.get(lane):
            self._pending_by_ex[int(exchange_type)].update(
                t for t in tokens if t
            )
            return
        sws = self._sws_by_lane.get(lane)
        if sws is None or not tokens:
            return
        subscribed = self._subscribed_by_ex[int(exchange_type)]
        corr = _CORRELATION_BY_LANE.get(lane, _WS_CORRELATION_ID)
        for i in range(0, len(tokens), _SUBSCRIBE_CHUNK):
            chunk = [t for t in tokens[i : i + _SUBSCRIBE_CHUNK] if t not in subscribed]
            if not chunk:
                continue
            try:
                sws.subscribe(
                    corr,
                    _SUBSCRIBE_MODE,
                    [{"exchangeType": int(exchange_type), "tokens": chunk}],
                )
            except Exception as exc:
                print(f"Angel subscribe ex={exchange_type} [{lane}]:", exc)
                self._pending_by_ex[int(exchange_type)].update(chunk)
                continue
            subscribed.update(chunk)

    def _unsubscribe_chunks(self, exchange_type: int, tokens: list[str]) -> None:
        lane = self._lane_for_exchange_type(exchange_type)
        sws = self._sws_by_lane.get(lane)
        if sws is None or not tokens:
            return
        corr = _CORRELATION_BY_LANE.get(lane, _WS_CORRELATION_ID)
        for i in range(0, len(tokens), _SUBSCRIBE_CHUNK):
            chunk = tokens[i : i + _SUBSCRIBE_CHUNK]
            try:
                sws.unsubscribe(
                    corr,
                    _SUBSCRIBE_MODE,
                    [{"exchangeType": int(exchange_type), "tokens": chunk}],
                )
            except Exception as exc:
                print(f"Angel unsubscribe ex={exchange_type} [{lane}]:", exc)

    def _notify_ltps_updated(self) -> None:
        now = time.monotonic()
        if now - self._last_ui_notify < _UI_DEBOUNCE_SEC:
            return
        self._last_ui_notify = now
        cb = self._on_ltps_updated
        root = self._root
        if root is not None:
            try:
                root.after(0, cb or (lambda: None))
                return
            except Exception:
                pass
        if cb:
            try:
                cb()
            except Exception:
                pass


def build_strategy_angel_ws_subscription_report(
    *,
    manager: Any | None = None,
) -> str:
    """Text report: Angel WS subscriptions (strategy + chain + index)."""
    import datetime as dt

    feed = get_strategy_ltp_feed()
    ltps = get_strategy_ltp_cache()
    now = dt.datetime.now().strftime("%H:%M:%S")

    lines: list[str] = [
        "Angel WebSocket subscriptions (in-app)",
        f"Time: {now}",
        "",
    ]

    if feed is None:
        lines.append("Feed: not installed")
        return "\n".join(lines)

    lines.extend(
        [
            f"Feed started: {'yes' if feed.started else 'no'}",
            f"WS connected: {'yes' if feed.online else 'no'} "
            f"(NIFTY={feed._connected_by_lane.get('NIFTY')} "
            f"SENSEX={feed._connected_by_lane.get('SENSEX')})",
            f"Neo fallback: {'yes' if feed.neo_fallback else 'no'}",
            f"Subscribed: {len(feed.subscribed_tokens)}",
            f"Pending: {len(feed.pending_tokens)}",
            f"Strategy LTP cache: {len(ltps)}",
            f"Index spot (Angel): {feed._index_spot}",
            "",
        ]
    )

    token_meta: dict[str, list[str]] = {}
    if manager is not None:
        try:
            open_entries = manager.get_open_strategy_entries()
        except Exception:
            open_entries = []
        from research.strategy_math.strategy_tracker import leg_angel_token

        for entry in open_entries:
            cid = str(entry.get("cycle_id") or "?")
            label = str(entry.get("strategy_label") or entry.get("strategy_key") or "?")
            for leg in entry.get("legs") or []:
                tok = leg_angel_token(leg)
                if not tok:
                    continue
                sym = str(leg.get("angel_symbol") or leg.get("trading_symbol") or "?").strip()
                token_meta.setdefault(tok, []).append(f"{label} [{cid}] ({sym})")

    subscribed = sorted(feed.subscribed_tokens, key=lambda t: int(t) if t.isdigit() else t)
    if not subscribed:
        lines.append("No tokens subscribed.")
        return "\n".join(lines)

    lines.append("— Subscribed —")
    for tok in subscribed:
        ltp = ltps.get(tok)
        ltp_txt = f"{ltp:.2f}" if ltp is not None and ltp > 0 else "—"
        neo = feed._chain_angel_to_neo.get(tok)
        note = f" chain→neo {neo}" if neo else ""
        meta = token_meta.get(tok, [f"token {tok}{note}"])
        lines.append(f"  {tok:>6}  LTP {ltp_txt:>8}  {meta[0]}")

    return "\n".join(lines)

"""Chunked Angel SmartAPI getCandleData fetch (500-bar API cap)."""

from __future__ import annotations

import time
from datetime import date, datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

from shared.data.angel_rate_limit import (
    in_cooldown,
    is_rate_limit_error,
    log_rate_limit_once,
    mark_rate_limit,
)

from .angel_historic_store import (
    AngelHistoricStore,
    BARS_PER_TRADING_DAY,
    HISTORIC_INTERVALS_SEC,
    IST,
    angel_historic_store,
    expected_bar_count,
    historic_window,
)

_FETCH_SCOPE = "getCandleData:historic"
_ANGEL_MAX_ROWS = 500
_REQUEST_PAUSE_SEC = 0.35

# Conservative chunk sizes (calendar days) to stay under 500 bars per request.
_CHUNK_DAYS: dict[int, int] = {
    60: 1,
    180: 3,    # ~125 bars/day → 3d ≈ 375
    300: 5,
    900: 15,   # ~25 bars/day → 15d ≈ 375
    1800: 40,
    3600: 40,
    86400: 500,
}

# Per HTTP/CLI call — avoid one long browser request (1m ≈ 130+ API calls for 6m).
_DEFAULT_BATCH_DAYS: dict[int, int] = {
    60: 15,   # ~15 calendar days per click (~11 sessions × 375 bars)
    180: 45,
    300: 60,   # ~3000 bars @ 75 bars/day
    900: 90,
    1800: 180,
    3600: 180,
    86400: 730,
}
_DEFAULT_MAX_REQUESTS: dict[int, int] = {
    60: 18,   # 15d window ≈ 11 trading days + headroom
    180: 20,
    300: 12,   # 60d / 5d chunks ≈ 12 calls → ~3000 bars
    900: 12,
    1800: 30,
    3600: 15,
    86400: 5,
}
_DEFAULT_TARGET_BARS: dict[int, int] = {
    300: 3000,
}


def limits_from_target_bars(interval_sec: int, target_bars: int) -> tuple[int, int]:
    """Derive (batch_days, max_requests) to fetch about ``target_bars`` per click."""
    sec = int(interval_sec)
    target_bars = max(1, int(target_bars))
    per_day = BARS_PER_TRADING_DAY.get(sec, 1)
    trading_days = max(1, (target_bars + per_day - 1) // per_day)
    batch_days = max(1, int(trading_days * 7 / 5) + 3)
    chunk_days = _CHUNK_DAYS.get(sec, 5)
    max_requests = max(1, (trading_days + chunk_days - 1) // chunk_days + 2)
    return batch_days, max_requests


def resolve_fetch_limits(
    interval_sec: int,
    *,
    batch_days: int | None = None,
    max_requests: int | None = None,
    target_bars: int | None = None,
) -> tuple[int, int, int | None]:
    """Return (batch_days, max_requests, target_bars_used)."""
    sec = int(interval_sec)
    if batch_days is not None and max_requests is not None:
        return int(batch_days), int(max_requests), target_bars
    if target_bars is not None:
        bd, mr = limits_from_target_bars(sec, target_bars)
        return bd, mr, int(target_bars)
    if sec in _DEFAULT_TARGET_BARS and batch_days is None and max_requests is None:
        tb = _DEFAULT_TARGET_BARS[sec]
        bd, mr = limits_from_target_bars(sec, tb)
        return bd, mr, tb
    default_batch, default_max_req = default_fetch_limits(sec)
    return (
        int(batch_days) if batch_days is not None else default_batch,
        int(max_requests) if max_requests is not None else default_max_req,
        None,
    )


def default_fetch_limits(interval_sec: int) -> tuple[int, int]:
    """Return (batch_days, max_requests) for one fetch call."""
    sec = int(interval_sec)
    return (
        _DEFAULT_BATCH_DAYS.get(sec, 30),
        _DEFAULT_MAX_REQUESTS.get(sec, 30),
    )


def exchange_for_token(token: str, index_tokens: dict[str, str] | None = None) -> str:
    tok = str(token or "").strip()
    index_tokens = index_tokens or {
        "99926000": "NSE",   # NIFTY
        "99919000": "BSE",   # SENSEX
    }
    if tok in index_tokens:
        return index_tokens[tok]
    return "NFO"


def _parse_candle_rows(rows: list, interval_sec: int) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for c in rows:
        try:
            dt_text = str(c[0])[:19]
            try:
                dt_obj = datetime.strptime(dt_text, "%Y-%m-%d %H:%M:%S").replace(tzinfo=IST)
            except ValueError:
                dt_obj = datetime.fromisoformat(dt_text.replace(" ", "T")).replace(tzinfo=IST)
            unix_ts = int(dt_obj.timestamp())
        except (TypeError, ValueError, IndexError):
            continue
        volume = 0
        if len(c) > 5 and c[5] not in (None, ""):
            try:
                volume = int(float(c[5]))
            except (TypeError, ValueError):
                volume = 0
        out.append({
            "bucket_start": float(unix_ts),
            "open": float(c[1]),
            "high": float(c[2]),
            "low": float(c[3]),
            "close": float(c[4]),
            "volume": volume,
            "interval_sec": interval_sec,
        })
    return out


def _ensure_client(client: Any | None = None) -> tuple[Any | None, str | None]:
    try:
        if client is not None:
            from angelone.smart_api_client import ensure_angel_ready
            return ensure_angel_ready(client), None
        from angelone.smart_api_client import ensure_angel_session, smartApi
        ensure_angel_session()
        return smartApi, None
    except Exception as exc:
        return None, str(exc)


def fetch_candles_chunk(
    client: Any,
    *,
    exchange: str,
    token: str,
    interval_sec: int,
    from_dt: datetime,
    to_dt: datetime,
) -> tuple[list[dict[str, Any]], str | None]:
    angel_interval = HISTORIC_INTERVALS_SEC.get(int(interval_sec))
    if not angel_interval:
        return [], f"Unsupported interval_sec={interval_sec}"

    if in_cooldown(_FETCH_SCOPE):
        return [], "Angel API in rate-limit cooldown"

    historic_param = {
        "exchange": exchange,
        "symboltoken": str(token),
        "interval": angel_interval,
        "fromdate": from_dt.strftime("%Y-%m-%d %H:%M"),
        "todate": to_dt.strftime("%Y-%m-%d %H:%M"),
    }
    try:
        val = client.getCandleData(historic_param)
    except Exception as exc:
        if is_rate_limit_error(exc=exc):
            mark_rate_limit(_FETCH_SCOPE)
            log_rate_limit_once(_FETCH_SCOPE, str(exc))
        return [], str(exc)

    if not isinstance(val, dict):
        if is_rate_limit_error(resp=val):
            mark_rate_limit(_FETCH_SCOPE)
            log_rate_limit_once(_FETCH_SCOPE, str(val))
        return [], f"Unexpected response type: {type(val).__name__}"

    if val.get("status") is False:
        msg = val.get("message") or val.get("errorcode") or "status=false"
        if is_rate_limit_error(resp=val, message=str(msg)):
            mark_rate_limit(_FETCH_SCOPE)
            log_rate_limit_once(_FETCH_SCOPE, str(msg))
        return [], str(msg)

    rows = val.get("data") or []
    if not rows:
        return [], None
    return _parse_candle_rows(rows, interval_sec), None


def _cap_to_session_end(dt: datetime) -> datetime:
    """Use last completed session close when ``dt`` is outside market hours."""
    from datetime import time as dt_time

    d = dt.date()
    if dt.weekday() >= 5:
        d = d - timedelta(days=1)
        while d.weekday() >= 5:
            d -= timedelta(days=1)
        return _session_close(d)
    t = dt.time()
    if t < dt_time(9, 15):
        d = d - timedelta(days=1)
        while d.weekday() >= 5:
            d -= timedelta(days=1)
        return _session_close(d)
    if t > dt_time(15, 30):
        return _session_close(d)
    return dt


def expected_latest_session_day(*, now: datetime | None = None) -> date:
    """Trading day the Newest column should reach for 'up to date'."""
    now = now or datetime.now(tz=IST)
    end = _cap_to_session_end(now)
    return end.date()


def is_newest_up_to_date(
    newest_ts: float | None,
    *,
    interval_sec: int = 60,
    now: datetime | None = None,
) -> bool:
    """True when stored newest is on the expected latest session day (IST)."""
    if newest_ts is None:
        return False
    try:
        newest_dt = datetime.fromtimestamp(float(newest_ts), tz=IST)
    except (TypeError, ValueError, OSError, OverflowError):
        return False
    target_day = expected_latest_session_day(now=now)
    if newest_dt.date() < target_day:
        return False
    # Same session day: allow incomplete live bar lag of a few intervals.
    now = now or datetime.now(tz=IST)
    target_end = _cap_to_session_end(now)
    lag = float(target_end.timestamp()) - float(newest_ts)
    return lag <= max(float(interval_sec) * 3.0, 900.0)


def resolve_fetch_window(
    token: str,
    interval_sec: int,
    months: int,
    *,
    mode: str = "auto",
    store: AngelHistoricStore | None = None,
    batch_days: int | None = None,
) -> tuple[datetime, datetime, str, dict[str, Any]]:
    """Pick [window_start, window_end] and mode label.

    Modes: initial | extend_before | extend_after (alias: up_to_date) | auto.
    """
    store = store or angel_historic_store
    interval_sec = int(interval_sec)
    bounds = store.get_bounds(token, interval_sec)
    meta: dict[str, Any] = {"bounds_before": bounds}

    if batch_days is None:
        batch_days, _ = default_fetch_limits(interval_sec)
    batch_days = max(1, int(batch_days))

    _from_ts, to_ts, from_day, to_day = historic_window(months)
    full_start = datetime(from_day.year, from_day.month, from_day.day, 9, 15, tzinfo=IST)
    full_end = _cap_to_session_end(datetime.fromtimestamp(to_ts, tz=IST))
    meta["full_window_from"] = full_start.strftime("%Y-%m-%d %H:%M")
    meta["full_window_to"] = full_end.strftime("%Y-%m-%d %H:%M")

    resolved = str(mode or "auto").strip().lower()
    if resolved in ("up_to_date", "upto_date", "forward"):
        resolved = "extend_after"

    if resolved == "auto":
        cnt = int(bounds.get("bar_count") or 0)
        if cnt > 0:
            expected = expected_bar_count(interval_sec, from_day, to_day)
            cov = store.coverage_in_range(token, interval_sec, _from_ts, to_ts)
            window_pct = float(cov["bar_count"]) / expected if expected else 0.0
            resolved = "extend_before" if window_pct >= 0.85 else "initial"
        else:
            resolved = "initial"

    if resolved == "extend_after":
        newest_ts = bounds.get("newest_ts")
        if newest_ts is None:
            resolved = "initial"
        elif is_newest_up_to_date(newest_ts, interval_sec=interval_sec):
            meta["already_current"] = True
            meta["batch_days"] = batch_days
            # Empty window — caller skips API.
            return full_end, full_end, "extend_after", meta
        else:
            newest_dt = datetime.fromtimestamp(float(newest_ts), tz=IST)
            # Fill gap after stored newest toward today (pages start at full_end).
            window_start = newest_dt + timedelta(seconds=interval_sec)
            window_end = full_end
            if window_start >= window_end:
                meta["already_current"] = True
                meta["batch_days"] = batch_days
                return full_end, full_end, "extend_after", meta
            meta["extend_after"] = bounds.get("newest_time")
            meta["batch_days"] = batch_days
            return window_start, window_end, "extend_after", meta

    if resolved == "extend_before":
        oldest_ts = bounds.get("oldest_ts")
        if oldest_ts is None:
            resolved = "initial"
        else:
            oldest_dt = datetime.fromtimestamp(float(oldest_ts), tz=IST)
            window_end = oldest_dt - timedelta(seconds=interval_sec)
            # Go further back in history — do not cap at the 6m window start.
            window_start = window_end - timedelta(days=batch_days)
            meta["extend_before"] = bounds["oldest_time"]
            meta["batch_days"] = batch_days
            return window_start, window_end, "extend_before", meta

    # initial — fill recent window; if meaningful partial DB, continue backward
    window_end = full_end
    bar_count = int(bounds.get("bar_count") or 0)
    if bar_count > 0 and bounds.get("oldest_ts"):
        expected = expected_bar_count(interval_sec, from_day, to_day)
        cov = store.coverage_in_range(token, interval_sec, _from_ts, to_ts)
        window_pct = float(cov["bar_count"]) / expected if expected else 0.0
        if window_pct >= 0.01:
            oldest_dt = datetime.fromtimestamp(float(bounds["oldest_ts"]), tz=IST)
            window_end = oldest_dt - timedelta(seconds=interval_sec)
            meta["continue_fill"] = bounds["oldest_time"]
        else:
            meta["refill_from_now"] = True

    window_start = max(full_start, window_end - timedelta(days=batch_days))
    meta["batch_days"] = batch_days
    return window_start, window_end, "initial", meta


def _session_open(day: date) -> datetime:
    return datetime(day.year, day.month, day.day, 9, 15, tzinfo=IST)


def _session_close(day: date) -> datetime:
    return datetime(day.year, day.month, day.day, 15, 30, tzinfo=IST)


def _prev_session_end(from_dt: datetime) -> datetime:
    """Jump to the previous trading session close (skip weekends naively)."""
    day = from_dt.date() - timedelta(days=1)
    while day.weekday() >= 5:
        day -= timedelta(days=1)
    return _session_close(day)


def _session_chunk_start(chunk_end: datetime, chunk_days: int, window_start: datetime) -> datetime:
    """Align chunk start to 09:15 IST (Angel index candles expect session fromdate)."""
    end_day = chunk_end.date()
    start_day = end_day - timedelta(days=max(1, chunk_days) - 1)
    chunk_start = _session_open(start_day)
    return max(window_start, chunk_start)


def _fetch_chunks_backward(
    api: Any,
    *,
    exchange: str,
    token: str,
    interval_sec: int,
    window_start: datetime,
    window_end: datetime,
    progress_cb=None,
    max_requests: int | None = None,
) -> tuple[list[dict[str, Any]], int, str | None, bool, bool]:
    """Page backward from window_end toward window_start.

    Returns bars, requests, error, exhausted, stopped_early.
    """
    chunk_days = _CHUNK_DAYS.get(interval_sec, 5)
    all_bars: dict[float, dict[str, Any]] = {}
    chunk_end = window_end
    requests = 0
    last_error: str | None = None
    empty_streak = 0
    exhausted = False
    stopped_early = False

    while chunk_end > window_start:
        if max_requests and requests >= max_requests:
            stopped_early = True
            break

        chunk_start = _session_chunk_start(chunk_end, chunk_days, window_start)
        if chunk_start >= chunk_end:
            chunk_end = _prev_session_end(chunk_end)
            if chunk_end <= window_start:
                break
            continue

        bars, err = fetch_candles_chunk(
            api,
            exchange=exchange,
            token=token,
            interval_sec=interval_sec,
            from_dt=chunk_start,
            to_dt=chunk_end,
        )
        requests += 1
        if err:
            last_error = err
            break
        if not bars:
            empty_streak += 1
            if chunk_start <= window_start or empty_streak >= 3:
                exhausted = True
                break
            chunk_end = _prev_session_end(chunk_end)
            time.sleep(_REQUEST_PAUSE_SEC)
            continue

        empty_streak = 0
        for b in bars:
            all_bars[float(b["bucket_start"])] = b

        if progress_cb:
            progress_cb(len(all_bars), chunk_start, chunk_end)

        oldest = min(float(b["bucket_start"]) for b in bars)
        if len(bars) >= _ANGEL_MAX_ROWS:
            chunk_end = datetime.fromtimestamp(oldest, tz=IST) - timedelta(seconds=interval_sec)
        elif oldest <= chunk_start.timestamp() + interval_sec:
            chunk_end = _prev_session_end(chunk_start)
        else:
            chunk_end = datetime.fromtimestamp(oldest, tz=IST) - timedelta(seconds=interval_sec)

        if chunk_end <= window_start:
            break
        time.sleep(_REQUEST_PAUSE_SEC)

    if not all_bars and not last_error:
        exhausted = True

    if chunk_end > window_start and not exhausted and not last_error:
        stopped_early = True

    win_start_ts = window_start.timestamp()
    win_end_ts = window_end.timestamp()
    result_bars = [
        all_bars[k]
        for k in sorted(all_bars)
        if win_start_ts - 1 <= k <= win_end_ts + 1
    ]
    return result_bars, requests, last_error, exhausted, stopped_early


def fetch_historic_range(
    token: str,
    interval_sec: int,
    *,
    months: int = 6,
    exchange: str | None = None,
    client: Any | None = None,
    progress_cb=None,
    mode: str = "auto",
    store: AngelHistoricStore | None = None,
    batch_days: int | None = None,
    max_requests: int | None = None,
    target_bars: int | None = None,
) -> dict[str, Any]:
    """Fetch candles into a date window (one batch per call — click again for more).

    mode=auto (default): if DB already has bars, fetch *older* data before the
    stored oldest timestamp; otherwise fetch the recent ``months`` window.

    mode=extend_after / up_to_date: fill from stored newest toward today so the
    Newest column reaches the current session.
    """
    interval_sec = int(interval_sec)
    angel_interval = HISTORIC_INTERVALS_SEC.get(interval_sec)
    if not angel_interval:
        return {"ok": False, "error": f"Unsupported interval {interval_sec}", "bars": []}

    store = store or angel_historic_store
    exchange = exchange or exchange_for_token(token)
    batch_days, max_requests, target_used = resolve_fetch_limits(
        interval_sec,
        batch_days=batch_days,
        max_requests=max_requests,
        target_bars=target_bars,
    )

    window_start, window_end, fetch_mode, window_meta = resolve_fetch_window(
        token,
        interval_sec,
        months,
        mode=mode,
        store=store,
        batch_days=batch_days,
    )

    if window_meta.get("already_current") or window_end <= window_start:
        bounds_after = store.get_bounds(token, interval_sec)
        return {
            "ok": True,
            "error": None,
            "history_exhausted": False,
            "batch_limited": False,
            "batch_days": batch_days,
            "max_requests": max_requests,
            "target_bars": target_used,
            "more_batches": False,
            "already_current": True,
            "hint": "Newest already up to date for this interval.",
            "token": str(token),
            "exchange": exchange,
            "interval_sec": interval_sec,
            "angel_interval": angel_interval,
            "months": months,
            "fetch_mode": fetch_mode,
            "requests": 0,
            "bar_count": 0,
            "bars": [],
            "window_from": window_start.strftime("%Y-%m-%d %H:%M"),
            "window_to": window_end.strftime("%Y-%m-%d %H:%M"),
            "bounds_after": bounds_after,
            **window_meta,
        }

    api, login_err = _ensure_client(client)
    if api is None:
        return {
            "ok": False,
            "error": login_err or "Angel login failed",
            "token": str(token),
            "exchange": exchange,
            "interval_sec": interval_sec,
            "angel_interval": angel_interval,
            "months": months,
            "fetch_mode": fetch_mode,
            "requests": 0,
            "bar_count": 0,
            "bars": [],
            **window_meta,
        }

    result_bars, requests, last_error, exhausted, stopped_early = _fetch_chunks_backward(
        api,
        exchange=exchange,
        token=token,
        interval_sec=interval_sec,
        window_start=window_start,
        window_end=window_end,
        progress_cb=progress_cb,
        max_requests=max_requests,
    )

    bounds_after = store.get_bounds(token, interval_sec)
    # Prefer in-memory newest from this batch when store not yet upserted.
    batch_newest = max((float(b["bucket_start"]) for b in result_bars), default=None)
    effective_newest = batch_newest
    if bounds_after.get("newest_ts") is not None:
        store_newest = float(bounds_after["newest_ts"])
        effective_newest = (
            max(store_newest, batch_newest) if batch_newest is not None else store_newest
        )
    _from_ts, to_ts, from_day, to_day = historic_window(months)
    expected = expected_bar_count(interval_sec, from_day, to_day)
    cov = store.coverage_in_range(token, interval_sec, _from_ts, to_ts)
    window_pct = round(100.0 * float(cov["bar_count"]) / expected, 1) if expected else 0.0
    up_to_date = is_newest_up_to_date(effective_newest, interval_sec=interval_sec)
    if fetch_mode == "extend_after":
        more_batches = (
            not exhausted
            and not last_error
            and not up_to_date
            and (stopped_early or len(result_bars) > 0)
        )
    else:
        more_batches = (
            not exhausted
            and not last_error
            and (
                stopped_early
                or window_pct < 85.0
                or (
                    fetch_mode == "extend_before"
                    and bounds_after.get("oldest_ts") is not None
                    and float(bounds_after["oldest_ts"]) > _from_ts + interval_sec
                )
            )
        )
    hint = None
    if more_batches:
        hint = (
            f"Batch done ({requests} API calls, {len(result_bars)} bars). "
            "Click the same fetch link again for the next batch."
        )
    elif exhausted and not result_bars:
        hint = "No older data from Angel for this window."
    elif fetch_mode == "extend_after" and up_to_date:
        hint = "Newest is up to date."

    return {
        "ok": (last_error is None or len(result_bars) > 0) and not (
            fetch_mode == "extend_before" and len(result_bars) == 0 and exhausted
        ),
        "error": last_error,
        "history_exhausted": exhausted and len(result_bars) == 0,
        "batch_limited": stopped_early,
        "batch_days": batch_days,
        "max_requests": max_requests,
        "target_bars": target_used,
        "more_batches": more_batches,
        "up_to_date": up_to_date,
        "window_6m_coverage_pct": window_pct,
        "hint": hint,
        "token": str(token),
        "exchange": exchange,
        "interval_sec": interval_sec,
        "angel_interval": angel_interval,
        "months": months,
        "fetch_mode": fetch_mode,
        "requests": requests,
        "bar_count": len(result_bars),
        "bars": result_bars,
        "window_from": window_start.strftime("%Y-%m-%d %H:%M"),
        "window_to": window_end.strftime("%Y-%m-%d %H:%M"),
        "bounds_after": bounds_after,
        **window_meta,
    }


def fetch_months_history_batch(
    token: str,
    interval_sec: int,
    *,
    months: int = 3,
    exchange: str | None = None,
    client: Any | None = None,
    store: AngelHistoricStore | None = None,
) -> dict[str, Any]:
    """Fetch one ~``months`` chunk for an interval.

    First call (empty DB): last ``months`` toward today.
    Later calls: next older ``months`` before stored oldest (extend_before).
    """
    store = store or angel_historic_store
    months = max(1, int(months))
    batch_days = months * 31
    chunk_days = _CHUNK_DAYS.get(int(interval_sec), 5)
    max_requests = max(60, (batch_days // max(1, chunk_days)) + 15)
    bounds = store.get_bounds(token, interval_sec)
    mode = "extend_before" if int(bounds.get("bar_count") or 0) > 0 else "initial"
    result = fetch_historic_range(
        token,
        interval_sec,
        months=months,
        exchange=exchange,
        client=client,
        mode=mode,
        store=store,
        batch_days=batch_days,
        max_requests=max_requests,
    )
    result["history_batch_months"] = months
    result["history_batch_mode"] = mode
    return result


def probe_angel_availability(
    token: str,
    interval_sec: int = 300,
    *,
    exchange: str | None = None,
    client: Any | None = None,
) -> dict[str, Any]:
    """Lightweight live probe: can Angel return any candles for this token?"""
    exchange = exchange or exchange_for_token(token)
    api, login_err = _ensure_client(client)
    if api is None:
        return {
            "token": str(token),
            "exchange": exchange,
            "interval_sec": interval_sec,
            "angel_reachable": False,
            "probe_bars": 0,
            "error": login_err or "Angel login failed",
        }
    now = datetime.now(tz=IST)
    from_dt = now - timedelta(days=3)
    bars, err = fetch_candles_chunk(
        api,
        exchange=exchange,
        token=token,
        interval_sec=interval_sec,
        from_dt=from_dt,
        to_dt=now,
    )
    return {
        "token": str(token),
        "exchange": exchange,
        "interval_sec": interval_sec,
        "angel_reachable": err is None and len(bars) > 0,
        "probe_bars": len(bars),
        "error": err,
    }

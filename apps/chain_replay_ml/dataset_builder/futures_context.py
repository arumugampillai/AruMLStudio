"""Front-month futures token resolution for DayContext (Phase 1).

Resolves current-month FUTIDX once per trading day from ``token_day_meta``
(preferred) with ``contract_meta_provider`` fallback. Soft-fail when absent.
"""

from __future__ import annotations

import sqlite3
from datetime import date, datetime
from typing import Any, Callable


def _parse_day(raw: str | date | datetime | None) -> date | None:
    if raw is None:
        return None
    if isinstance(raw, datetime):
        return raw.date()
    if isinstance(raw, date):
        return raw
    text = str(raw or "").strip()
    if not text:
        return None
    for fmt in ("%Y-%m-%d", "%d-%b-%Y", "%d%b%Y", "%d/%m/%Y"):
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    try:
        import pandas as pd

        ts = pd.to_datetime(text, errors="coerce", dayfirst=True)
        if not pd.isna(ts):
            return ts.date()
    except Exception:
        pass
    return None


def _pick_front_month(
    candidates: list[dict[str, Any]],
    *,
    as_of: date,
) -> dict[str, Any] | None:
    """Prefer expiry in as_of calendar month; else nearest expiry >= as_of."""
    current_ym = (as_of.year, as_of.month)
    best_in_month: tuple[date, dict[str, Any]] | None = None
    best_any: tuple[date, dict[str, Any]] | None = None
    for row in candidates:
        exp = _parse_day(row.get("expiry"))
        if exp is None or exp < as_of:
            continue
        if best_any is None or exp < best_any[0]:
            best_any = (exp, row)
        if (exp.year, exp.month) == current_ym:
            if best_in_month is None or exp < best_in_month[0]:
                best_in_month = (exp, row)
    picked = best_in_month or best_any
    return picked[1] if picked else None


def _candidates_from_token_day_meta(
    conn: sqlite3.Connection,
    *,
    index_key: str,
    as_of_date: str,
) -> list[dict[str, Any]]:
    cols = {row[1] for row in conn.execute("PRAGMA table_info(token_day_meta)")}
    if "token" not in cols or "as_of_date" not in cols:
        return []
    has_inst = "instrument_type" in cols
    has_name = "name" in cols
    has_expiry = "expiry_date" in cols
    has_symbol = "trading_symbol" in cols
    if not (has_name and has_expiry):
        return []

    if has_inst:
        sql = """
            SELECT token, trading_symbol, expiry_date, instrument_type
            FROM token_day_meta
            WHERE as_of_date = ?
              AND UPPER(COALESCE(name, '')) = ?
              AND UPPER(COALESCE(instrument_type, '')) = 'FUTIDX'
        """
        rows = conn.execute(sql, (as_of_date, index_key.upper())).fetchall()
    else:
        # Legacy / sparse meta: futures have expiry, no CE/PE option_type.
        opt_col = "option_type" if "option_type" in cols else None
        if opt_col:
            sql = f"""
                SELECT token, trading_symbol, expiry_date, NULL
                FROM token_day_meta
                WHERE as_of_date = ?
                  AND UPPER(COALESCE(name, '')) = ?
                  AND ({opt_col} IS NULL OR {opt_col} = '')
                  AND expiry_date IS NOT NULL AND expiry_date != ''
                  AND (strike_price IS NULL OR strike_price = 0)
            """
            rows = conn.execute(sql, (as_of_date, index_key.upper())).fetchall()
        else:
            return []

    out: list[dict[str, Any]] = []
    for token, symbol, expiry, _inst in rows:
        tok = str(token or "").strip()
        if not tok:
            continue
        out.append({
            "token": tok,
            "symbol": str(symbol or "").strip() if has_symbol else "",
            "expiry": str(expiry or "").strip(),
        })
    return out


def _candidates_from_provider(index_key: str) -> list[dict[str, Any]]:
    try:
        from storage.contract_meta_provider import contract_meta_provider
    except Exception:
        return []
    if not hasattr(contract_meta_provider, "futures_for_name"):
        return []
    return list(contract_meta_provider.futures_for_name(index_key) or [])


def _token_has_ticks(
    conn: sqlite3.Connection,
    token: str,
    *,
    open_ts: float | None = None,
    close_ts: float | None = None,
) -> bool:
    if open_ts is not None and close_ts is not None:
        row = conn.execute(
            """
            SELECT 1 FROM ticks
            WHERE token = ? AND ts >= ? AND ts <= ?
              AND ltp IS NOT NULL AND ltp > 0
            LIMIT 1
            """,
            (token, open_ts, close_ts),
        ).fetchone()
    else:
        row = conn.execute(
            "SELECT 1 FROM ticks WHERE token = ? AND ltp IS NOT NULL AND ltp > 0 LIMIT 1",
            (token,),
        ).fetchone()
    return row is not None


def resolve_front_month_futures(
    conn: sqlite3.Connection,
    *,
    underlying: str,
    trading_day: str,
    normalize_index_name: Callable[[str], str],
    open_ts: float | None = None,
    close_ts: float | None = None,
) -> dict[str, str] | None:
    """Return ``{token, symbol, expiry}`` for front-month FUTIDX, or None."""
    index_key = normalize_index_name(underlying)
    as_of = _parse_day(trading_day)
    if as_of is None:
        return None

    candidates = _candidates_from_token_day_meta(
        conn, index_key=index_key, as_of_date=str(trading_day).strip()[:10],
    )
    if not candidates:
        candidates = _candidates_from_provider(index_key)

    # Prefer contracts that actually have ticks when session bounds are known.
    if open_ts is not None and close_ts is not None and candidates:
        with_ticks = [
            c for c in candidates
            if _token_has_ticks(conn, c["token"], open_ts=open_ts, close_ts=close_ts)
        ]
        if with_ticks:
            candidates = with_ticks

    picked = _pick_front_month(candidates, as_of=as_of)
    if picked is None:
        return None
    token = str(picked.get("token") or "").strip()
    if not token:
        return None
    return {
        "token": token,
        "symbol": str(picked.get("symbol") or "").strip(),
        "expiry": str(picked.get("expiry") or "").strip(),
    }


def emit_futures_timeline_features(
    raw: dict[str, Any],
    *,
    ts: float,
    futures_tl: Any | None,
) -> dict[str, Any]:
    """Broadcast futures Base levels from ``futures_tl`` as-of ``ts``.

    Phase 1: futures_ltp, futures_vwap.
    Phase 2: futures_day_volume, futures_bid, futures_ask, futures_spread.
    """
    out = dict(raw)
    null_keys = (
        "futures_ltp",
        "futures_vwap",
        "futures_day_volume",
        "futures_oi",
        "futures_bid",
        "futures_ask",
        "futures_spread",
    )
    tl = futures_tl
    if tl is None or not getattr(tl, "timestamps", None):
        for key in null_keys:
            out[key] = None
        return out

    ltp_fn = getattr(tl, "ltp_rupees_at", None)
    atp_fn = getattr(tl, "atp_rupees_at", None)
    vol_fn = getattr(tl, "volume_at", None)
    oi_fn = getattr(tl, "oi_at", None)
    book_fn = getattr(tl, "book_at", None)
    spread_fn = getattr(tl, "spread_paise_at", None)

    out["futures_ltp"] = ltp_fn(ts) if callable(ltp_fn) else None
    out["futures_vwap"] = atp_fn(ts) if callable(atp_fn) else None

    vol = vol_fn(ts) if callable(vol_fn) else None
    out["futures_day_volume"] = float(vol) if vol is not None else None
    oi = oi_fn(ts) if callable(oi_fn) else None
    out["futures_oi"] = float(oi) if oi is not None else None

    bid = ask = spread = None
    book = book_fn(ts) if callable(book_fn) else None
    if book is not None and getattr(book, "has_l1", False):
        bid_paise = int(book.bid_prices_paise[0] or 0)
        ask_paise = int(book.ask_prices_paise[0] or 0)
        if bid_paise > 0:
            bid = bid_paise / 100.0
        if ask_paise > 0:
            ask = ask_paise / 100.0
        if bid is not None and ask is not None:
            spread = max(0.0, ask - bid)
    if spread is None and callable(spread_fn):
        spread_paise = spread_fn(ts)
        if spread_paise is not None and int(spread_paise) > 0:
            spread = int(spread_paise) / 100.0

    out["futures_bid"] = bid
    out["futures_ask"] = ask
    out["futures_spread"] = spread
    return out


__all__ = ["resolve_front_month_futures", "emit_futures_timeline_features"]

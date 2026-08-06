"""Chain replay export helpers (no FastAPI dependency).

Used by ``main.py`` HTTP routes and ``download_chain_replay.py --local``.
"""
from __future__ import annotations

import json
import os
import sqlite3
from datetime import date, datetime
from typing import Any, Callable
from zoneinfo import ZoneInfo

from storage.contract_meta_provider import _derive_option_type, contract_meta_provider

IST = ZoneInfo("Asia/Kolkata")

MIN_CHAIN_OPTION_TOKENS = 10

CHAIN_TICK_FIELDS = [
    "token",
    "ts",
    "ltp",
    "day_volume",
    "ltq",
    "seq",
    "bid_prices",
    "bid_quantities",
    "ask_prices",
    "ask_quantities",
    "oi",
]


class ChainReplayError(Exception):
    """Raised for invalid parameters or missing data."""


def parse_expiry_date(expiry_str: Any) -> date | None:
    if expiry_str is None:
        return None
    if isinstance(expiry_str, date) and not isinstance(expiry_str, datetime):
        return expiry_str
    if isinstance(expiry_str, datetime):
        return expiry_str.date()
    text = str(expiry_str).strip()
    if not text:
        return None
    for fmt in ("%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y", "%d%b%Y", "%d-%b-%Y"):
        try:
            return datetime.strptime(text.upper(), fmt).date()
        except ValueError:
            continue
    return None


def ist_market_session_bounds(trading_day: str) -> tuple[float, float]:
    try:
        y, m, d = (int(p) for p in trading_day.split("-"))
        day = date(y, m, d)
    except (TypeError, ValueError) as exc:
        raise ChainReplayError(f"Bad date format (need YYYY-MM-DD): {trading_day}") from exc
    open_dt = datetime(day.year, day.month, day.day, 9, 15, 0, tzinfo=IST)
    close_dt = datetime(day.year, day.month, day.day, 15, 30, 0, tzinfo=IST)
    return open_dt.timestamp(), close_dt.timestamp()


def normalize_expiry_param(expiry: str) -> str:
    parsed = parse_expiry_date(expiry)
    if parsed is None:
        raise ChainReplayError(f"Bad expiry format (need YYYY-MM-DD): {expiry}")
    return parsed.strftime("%Y-%m-%d")


def parse_json_int_array(raw: Any) -> list[int]:
    if raw in (None, ""):
        return []
    if isinstance(raw, list):
        out: list[int] = []
        for item in raw:
            if item is None:
                continue
            try:
                out.append(int(item))
            except (TypeError, ValueError):
                continue
        return out
    try:
        parsed = json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return []
    if not isinstance(parsed, list):
        return []
    out: list[int] = []
    for item in parsed:
        if item is None:
            continue
        try:
            out.append(int(item))
        except (TypeError, ValueError):
            continue
    return out


def require_v1_ticks_schema(conn: sqlite3.Connection) -> None:
    cols = {row[1] for row in conn.execute("PRAGMA table_info(ticks)")}
    if "bid_prices" not in cols:
        raise ChainReplayError(
            "Chain replay requires v1 schema; run legacy_to_v1_import first."
        )


def index_prev_close_paise(
    conn: sqlite3.Connection,
    *,
    index_token: str,
    as_of_date: str,
) -> int | None:
    """Previous close for index spot change % (paise, v1 token_day_meta)."""
    row = conn.execute(
        """
        SELECT prev_close
        FROM token_day_meta
        WHERE token = ? AND as_of_date = ?
        """,
        (str(index_token).strip(), as_of_date),
    ).fetchone()
    if not row or row[0] is None:
        return None
    try:
        value = int(row[0])
    except (TypeError, ValueError):
        return None
    return value if value > 0 else None


def index_spot_meta(
    underlying: str,
    index_config: dict[str, dict[str, Any]],
    normalize_index_name: Callable[[str], str],
) -> tuple[str, dict[str, Any]]:
    index_key = normalize_index_name(underlying)
    cfg = index_config.get(index_key)
    if cfg is None:
        raise ChainReplayError(f"Unknown underlying: {underlying}")
    index_token = str(cfg.get("index_token") or "").strip()
    if not index_token:
        raise ChainReplayError(f"No index token configured for {underlying}")
    return index_token, {
        "symbol": str(cfg.get("display_symbol") or index_key),
        "type": "INDEX",
        "strike": None,
        "lot_size": None,
    }


def _normalize_chain_option_entry(entry: dict[str, Any]) -> dict[str, Any] | None:
    """Keep CE/PE strikes only; drop futures / invalid rows."""
    sym = str(entry.get("symbol") or "").strip()
    opt_type = str(entry.get("type") or "").strip().upper()
    if opt_type not in ("CE", "PE"):
        derived = _derive_option_type(sym)
        if not derived:
            return None
        opt_type = derived
    try:
        strike = int(entry.get("strike"))
    except (TypeError, ValueError):
        return None
    if strike <= 0:
        return None
    out = dict(entry)
    out["type"] = opt_type
    out["strike"] = strike
    return out


def _filter_chain_option_tokens(tokens: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    filtered: dict[str, dict[str, Any]] = {}
    for tok, meta in tokens.items():
        normalized = _normalize_chain_option_entry(meta)
        if normalized is not None:
            filtered[str(tok).strip()] = normalized
    return filtered


def chain_tokens_from_db_meta(
    conn: sqlite3.Connection,
    *,
    underlying: str,
    expiry: str,
    as_of_date: str,
    normalize_index_name: Callable[[str], str],
) -> dict[str, dict[str, Any]]:
    index_key = normalize_index_name(underlying)
    tokens: dict[str, dict[str, Any]] = {}
    rows = conn.execute(
        """
        SELECT token, trading_symbol, strike_price, option_type, lot_size, day_open
        FROM token_day_meta
        WHERE name = ? AND expiry_date = ? AND as_of_date = ?
        ORDER BY strike_price, option_type
        """,
        (index_key, expiry, as_of_date),
    ).fetchall()
    for token, symbol, strike, option_type, lot_size, day_open in rows:
        tok = str(token).strip()
        if not tok:
            continue
        entry: dict[str, Any] = {
            "symbol": symbol,
            "type": option_type or "OPT",
            "strike": strike,
            "lot_size": lot_size,
        }
        if day_open is not None:
            try:
                open_paise = int(day_open)
                if open_paise > 0:
                    entry["day_open"] = open_paise
            except (TypeError, ValueError):
                pass
        tokens[tok] = entry
    return tokens


def chain_tokens_from_provider(
    conn: sqlite3.Connection,
    *,
    underlying: str,
    expiry: str,
    normalize_index_name: Callable[[str], str],
    require_ticks: bool = True,
) -> dict[str, dict[str, Any]]:
    index_key = normalize_index_name(underlying)
    db_tokens = {str(row[0]) for row in conn.execute("SELECT DISTINCT token FROM ticks")}
    tokens: dict[str, dict[str, Any]] = {}
    for tok in contract_meta_provider.tokens_for_name_expiry(index_key, expiry):
        meta = contract_meta_provider.lookup(tok)
        if not meta:
            continue
        tok_s = str(tok).strip()
        if not tok_s:
            continue
        if require_ticks and tok_s not in db_tokens:
            continue
        opt_type = meta.get("option_type") or _derive_option_type(meta.get("trading_symbol"))
        tokens[tok_s] = {
            "symbol": meta.get("trading_symbol"),
            "type": opt_type or "OPT",
            "strike": meta.get("strike_price"),
            "lot_size": meta.get("lot_size"),
        }
    return tokens


def chain_expiries_with_option_ticks(
    conn: sqlite3.Connection,
    *,
    underlying: str,
    as_of_date: str,
    normalize_index_name: Callable[[str], str],
) -> list[str]:
    """Expiries with CE/PE ticks during the replay market session on ``as_of_date``."""
    index_key = normalize_index_name(underlying)
    open_ts, close_ts = ist_market_session_bounds(as_of_date)
    rows = conn.execute(
        """
        SELECT DISTINCT m.expiry_date
        FROM token_day_meta m
        INNER JOIN ticks t ON t.token = m.token
        WHERE m.name = ?
          AND m.as_of_date = ?
          AND m.option_type IN ('CE', 'PE')
          AND m.expiry_date IS NOT NULL
          AND m.expiry_date != ''
          AND t.ts >= ?
          AND t.ts <= ?
        ORDER BY m.expiry_date
        """,
        (index_key, as_of_date, open_ts, close_ts),
    ).fetchall()
    return [str(row[0]) for row in rows]


def chain_expiries_for_day(
    conn: sqlite3.Connection,
    *,
    underlying: str,
    normalize_index_name: Callable[[str], str],
    as_of_date: str | None = None,
) -> list[str]:
    index_key = normalize_index_name(underlying)
    expiries: set[str] = set()

    if as_of_date:
        for row in conn.execute(
            """
            SELECT DISTINCT expiry_date
            FROM token_day_meta
            WHERE name = ? AND as_of_date = ? AND expiry_date IS NOT NULL AND expiry_date != ''
            """,
            (index_key, as_of_date),
        ):
            expiries.add(str(row[0]))
        if expiries:
            with_ticks = set(
                chain_expiries_with_option_ticks(
                    conn,
                    underlying=underlying,
                    as_of_date=as_of_date,
                    normalize_index_name=normalize_index_name,
                )
            )
            tick_first = [e for e in sorted(with_ticks) if e in expiries]
            rest = [e for e in sorted(expiries) if e not in with_ticks]
            return tick_first + rest

    db_tokens = {str(row[0]) for row in conn.execute("SELECT DISTINCT token FROM ticks")}
    for tok in db_tokens:
        meta = contract_meta_provider.lookup(tok)
        if not meta:
            continue
        if str(meta.get("name") or "").upper() != index_key.upper():
            continue
        exp = meta.get("expiry_date")
        if exp:
            expiries.add(str(exp))
    return sorted(expiries)


def enrich_tokens_day_open(
    conn: sqlite3.Connection,
    tokens: dict[str, dict[str, Any]],
    *,
    as_of_date: str,
) -> None:
    """Attach day_open (paise) from token_day_meta when missing."""
    need = [tok for tok, meta in tokens.items() if meta.get("day_open") is None]
    if not need:
        return
    batch_size = 400
    for offset in range(0, len(need), batch_size):
        chunk = need[offset : offset + batch_size]
        placeholders = ",".join("?" for _ in chunk)
        rows = conn.execute(
            f"""
            SELECT token, day_open
            FROM token_day_meta
            WHERE as_of_date = ? AND token IN ({placeholders})
            """,
            [as_of_date, *chunk],
        ).fetchall()
        for token, day_open in rows:
            tok = str(token).strip()
            if not tok or tok not in tokens or day_open is None:
                continue
            try:
                open_paise = int(day_open)
            except (TypeError, ValueError):
                continue
            if open_paise > 0:
                tokens[tok]["day_open"] = open_paise


def resolve_chain_tokens(
    conn: sqlite3.Connection,
    *,
    underlying: str,
    expiry: str,
    as_of_date: str,
    index_config: dict[str, dict[str, Any]],
    normalize_index_name: Callable[[str], str],
) -> dict[str, dict[str, Any]]:
    index_token, index_meta = index_spot_meta(underlying, index_config, normalize_index_name)
    db_raw = chain_tokens_from_db_meta(
        conn,
        underlying=underlying,
        expiry=expiry,
        as_of_date=as_of_date,
        normalize_index_name=normalize_index_name,
    )
    option_tokens = _filter_chain_option_tokens(db_raw)

    if len(option_tokens) < MIN_CHAIN_OPTION_TOKENS:
        tick_only = _filter_chain_option_tokens(
            chain_tokens_from_provider(
                conn,
                underlying=underlying,
                expiry=expiry,
                normalize_index_name=normalize_index_name,
                require_ticks=True,
            )
        )
        for tok, meta in tick_only.items():
            option_tokens.setdefault(tok, meta)

    if len(option_tokens) < MIN_CHAIN_OPTION_TOKENS:
        full_chain = _filter_chain_option_tokens(
            chain_tokens_from_provider(
                conn,
                underlying=underlying,
                expiry=expiry,
                normalize_index_name=normalize_index_name,
                require_ticks=False,
            )
        )
        for tok, meta in full_chain.items():
            option_tokens.setdefault(tok, meta)

    if not option_tokens:
        available = chain_expiries_for_day(
            conn,
            underlying=underlying,
            normalize_index_name=normalize_index_name,
            as_of_date=as_of_date,
        )
        hint = (
            f" Available expiries: {', '.join(available)}."
            if available
            else " No expiries resolved (is contract_meta_provider loaded?)."
        )
        raise ChainReplayError(
            f"No option chain tokens for {underlying} expiry={expiry} on {as_of_date}.{hint}"
        )
    enrich_tokens_day_open(conn, option_tokens, as_of_date=as_of_date)
    tokens = {index_token: index_meta}
    for tok, meta in option_tokens.items():
        if tok != index_token:
            tokens[tok] = meta
    return tokens


def chain_tick_row_from_sql_row(row: tuple[Any, ...]) -> list[Any] | None:
    (
        token,
        ts,
        ltp,
        day_volume,
        ltq,
        seq,
        bid_prices,
        bid_quantities,
        ask_prices,
        ask_quantities,
        oi,
    ) = row
    if ltp is None:
        return None
    try:
        ltp_i = int(ltp)
    except (TypeError, ValueError):
        return None
    if ltp_i <= 0:
        return None
    try:
        ts_f = round(float(ts), 3)
    except (TypeError, ValueError):
        return None
    oi_i: int | None
    try:
        oi_i = int(oi) if oi is not None else None
    except (TypeError, ValueError):
        oi_i = None
    return [
        str(token),
        ts_f,
        ltp_i,
        int(day_volume) if day_volume is not None else None,
        int(ltq) if ltq is not None else 0,
        int(seq) if seq is not None else 0,
        parse_json_int_array(bid_prices),
        parse_json_int_array(bid_quantities),
        parse_json_int_array(ask_prices),
        parse_json_int_array(ask_quantities),
        oi_i,
    ]


def chain_replay_select_sql(token_list: list[str]) -> tuple[str, str]:
    placeholders = ",".join("?" for _ in token_list)
    count_sql = f"""
        SELECT COUNT(1)
        FROM ticks t
        WHERE t.token IN ({placeholders})
          AND t.ts >= ?
          AND t.ts <= ?
    """
    ticks_sql = f"""
        SELECT t.token, t.ts, t.ltp, t.day_volume, t.ltq, t.sequence_number,
               t.bid_prices, t.bid_quantities, t.ask_prices, t.ask_quantities,
               t.oi
        FROM ticks t
        WHERE t.token IN ({placeholders})
          AND t.ts >= ?
          AND t.ts <= ?
        ORDER BY t.ts ASC, t.sequence_number ASC
    """
    return count_sql, ticks_sql


def prepare_chain_replay_context(
    *,
    underlying: str,
    expiry_norm: str,
    resolved_day: str,
    db_path: str,
    index_config: dict[str, dict[str, Any]],
    normalize_index_name: Callable[[str], str],
) -> tuple[dict[str, Any], str, list[Any]]:
    open_ts, close_ts = ist_market_session_bounds(resolved_day)
    index_key = normalize_index_name(underlying)

    conn = sqlite3.connect(db_path)
    try:
        require_v1_ticks_schema(conn)
        available_expiries = chain_expiries_for_day(
            conn,
            underlying=underlying,
            normalize_index_name=normalize_index_name,
            as_of_date=resolved_day,
        )
        token_meta = resolve_chain_tokens(
            conn,
            underlying=underlying,
            expiry=expiry_norm,
            as_of_date=resolved_day,
            index_config=index_config,
            normalize_index_name=normalize_index_name,
        )
        token_list = list(token_meta.keys())
        count_sql, ticks_sql = chain_replay_select_sql(token_list)
        params: list[Any] = [*token_list, open_ts, close_ts]
        tick_count = int(conn.execute(count_sql, params).fetchone()[0])
        option_tick_token_count = int(
            conn.execute(
                """
                SELECT COUNT(DISTINCT t.token)
                FROM ticks t
                INNER JOIN token_day_meta m
                    ON m.token = t.token AND m.as_of_date = ?
                WHERE m.name = ?
                  AND m.expiry_date = ?
                  AND m.option_type IN ('CE', 'PE')
                """,
                (resolved_day, index_key, expiry_norm),
            ).fetchone()[0]
        )
        expiries_with_ticks = chain_expiries_with_option_ticks(
            conn,
            underlying=underlying,
            as_of_date=resolved_day,
            normalize_index_name=normalize_index_name,
        )
        index_prev_close: int | None = None
        for tok, tok_meta in token_meta.items():
            if tok_meta.get("type") == "INDEX":
                index_prev_close = index_prev_close_paise(
                    conn,
                    index_token=str(tok),
                    as_of_date=resolved_day,
                )
                break
    finally:
        conn.close()

    meta = {
        "schema": "v1",
        "underlying": index_key,
        "expiry": expiry_norm,
        "date": resolved_day,
        "db_path": os.path.basename(db_path),
        "token_count": len(token_meta),
        "tick_count": tick_count,
        "option_tick_token_count": option_tick_token_count,
        "expiries_with_ticks": expiries_with_ticks,
        "recommended_expiry": expiries_with_ticks[0] if expiries_with_ticks else expiry_norm,
        "market_open_ts": open_ts,
        "market_close_ts": close_ts,
        "available_expiries": available_expiries,
        "fields": list(CHAIN_TICK_FIELDS),
        "tokens": token_meta,
        "index_prev_close": index_prev_close,
        "meta_only": False,
    }
    return meta, ticks_sql, params


def stream_chain_replay_json(
    meta: dict[str, Any],
    db_path: str,
    ticks_sql: str,
    params: list[Any],
):
    def generate():
        conn = sqlite3.connect(db_path)
        try:
            yield '{"meta":'
            yield json.dumps(meta, ensure_ascii=False, separators=(",", ":"))
            yield ',"ticks":['
            first = True
            for row in conn.execute(ticks_sql, params):
                tick = chain_tick_row_from_sql_row(row)
                if tick is None:
                    continue
                if not first:
                    yield ","
                first = False
                yield json.dumps(tick, ensure_ascii=False, separators=(",", ":"))
            yield "]}"
        finally:
            conn.close()

    return generate


def export_chain_replay_to_file(
    *,
    meta: dict[str, Any],
    db_path: str,
    ticks_sql: str,
    params: list[Any],
    out_path: str,
) -> int:
    written = 0
    with open(out_path, "w", encoding="utf-8", newline="") as fh:
        for chunk in stream_chain_replay_json(meta, db_path, ticks_sql, params)():
            fh.write(chunk)
            written += len(chunk.encode("utf-8"))
    return written


def bootstrap_provider_from_options_df(options_df: Any) -> int:
    contract_meta_provider.clear()
    return contract_meta_provider.populate_from_options_df(options_df)


def _scrip_master_cache_path() -> str:
    from path_config import CHART_DATA_ROOT

    cache_dir = os.path.join(CHART_DATA_ROOT, "data", "cache")
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, "OpenAPIScripMaster.json")


def _load_scrip_master_rows() -> list[dict[str, Any]]:
    """Fetch Angel scrip master, with local cache + SSL fallback for CLI use."""
    cache_path = _scrip_master_cache_path()
    if os.path.isfile(cache_path) and os.path.getsize(cache_path) > 0:
        with open(cache_path, encoding="utf-8") as fh:
            cached = json.load(fh)
        if isinstance(cached, list) and cached:
            return cached

    try:
        import requests
    except ImportError as exc:
        raise ChainReplayError(
            "requests required for --local export when provider is empty"
        ) from exc

    url = "https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json"
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        resp = requests.get(url, headers=headers, timeout=120)
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        resp = requests.get(url, headers=headers, timeout=120, verify=False)
        resp.raise_for_status()
        data = resp.json()

    if not isinstance(data, list) or not data:
        raise ChainReplayError("Angel scrip master download returned no instruments.")

    with open(cache_path, "w", encoding="utf-8") as fh:
        json.dump(data, fh)
    return data


def bootstrap_provider_for_underlying(
    underlying: str,
    index_config: dict[str, dict[str, Any]],
    *,
    normalize_index_name: Callable[[str], str],
) -> None:
    """Best-effort provider bootstrap for offline CLI export."""
    if contract_meta_provider.size() > 0:
        return

    try:
        import pandas as pd
    except ImportError as exc:
        raise ChainReplayError(
            "pandas required for --local export when provider is empty"
        ) from exc

    index_key = normalize_index_name(underlying)
    cfg = index_config.get(index_key) or index_config.get("NIFTY") or {}
    title = str(cfg.get("title") or index_key).upper()
    exch = str(cfg.get("exch_seg") or "NFO").upper()

    df = pd.DataFrame(_load_scrip_master_rows())
    options = df[
        (df["name"].astype(str).str.upper() == title)
        & (df["exch_seg"].astype(str).str.upper() == exch)
        & (df["instrumenttype"] == "OPTIDX")
    ].copy()
    if options.empty and index_key == "SENSEX":
        options = df[
            (df["name"].astype(str).str.upper() == "SENSEX")
            & (df["instrumenttype"] == "OPTIDX")
        ].copy()
    if not options.empty:
        options["strike"] = options["strike"].astype(float) / 100
    bootstrap_provider_from_options_df(options)
    contract_meta_provider.populate_from_index_config(index_config)


def bootstrap_provider_minimal_nifty(index_config: dict[str, dict[str, Any]]) -> None:
    """Backward-compatible alias for NIFTY-only bootstrap."""
    bootstrap_provider_for_underlying(
        "NIFTY",
        index_config,
        normalize_index_name=lambda raw: str(raw or "NIFTY").strip().upper(),
    )

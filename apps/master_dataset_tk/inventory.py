"""Load chain source inventory from disk cache (no HTTP server)."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from chain_replay_ml.bs import days_to_expiry


def _data_dir(chart_dir: str) -> str:
    import os

    return os.path.join(chart_dir, "data")


def load_chain_inventory_rows(chart_dir: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Chain rows for master builds — same source as Create Dataset inventory API."""
    from storage.market_db_inventory_cache import load_cache

    data_dir = _data_dir(chart_dir)
    cache = load_cache(data_dir)
    if not cache:
        return [], {"source": "market_db_inventory.json", "last_updated": None}

    databases = cache.get("databases") or {}
    spot_keys: set[tuple[str, str]] = set()
    for trading_day, entry in databases.items():
        for row in (entry or {}).get("rows") or []:
            if str(row.get("kind") or "").lower() != "spot":
                continue
            market = str(row.get("index") or "").strip().upper()
            if market in ("NIFTY", "SENSEX", "BANKNIFTY"):
                spot_keys.add((trading_day, market))

    rows_out: list[dict[str, Any]] = []
    for trading_day in sorted(databases.keys(), reverse=True):
        entry = databases.get(trading_day) or {}
        db_file = str(entry.get("db_file") or "").strip()
        db_path = str(entry.get("db_path") or "").strip()
        for row in entry.get("rows") or []:
            if str(row.get("kind") or "").lower() != "chain":
                continue
            market = str(row.get("index") or "").strip().upper()
            if market not in ("NIFTY", "SENSEX", "BANKNIFTY"):
                continue
            expiry_raw = str(row.get("expiry") or "").strip()
            if not expiry_raw or expiry_raw.upper() == "SPOT":
                continue
            try:
                dte = int(days_to_expiry(trading_day, expiry_raw))
            except (TypeError, ValueError):
                dte = None
            spot_ok = (trading_day, market) in spot_keys
            rows_out.append({
                "type": "chain",
                "trading_day": trading_day,
                "date": trading_day,
                "market": market,
                "expiry": expiry_raw,
                "chain_ticks": int(row.get("tick_count") or row.get("ticks") or 0),
                "spot_available": spot_ok,
                "days_to_expiry": dte,
                "source_id": f"{trading_day}|{market}|{expiry_raw}",
                "db_file": db_file,
                "db_path": db_path,
            })

    meta = {
        "source": "market_db_inventory.json",
        "last_updated": cache.get("last_updated"),
        "row_count": len(rows_out),
    }
    return rows_out, meta


def filter_inventory_rows(rows: list[dict[str, Any]], market: str) -> list[dict[str, Any]]:
    m = str(market or "NIFTY").strip().upper()
    if m == "BOTH":
        return list(rows)
    return [r for r in rows if str(r.get("market") or "").upper() == m]

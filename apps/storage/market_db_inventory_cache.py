"""JSON cache for View DB market tick inventory.

On each request we always scan the data folder (cheap). SQLite tick counts are
reused from ``market_db_inventory.json`` when still valid:

* Historical days — cached while the DB file size/mtime is unchanged.
* Today before 15:30 — always re-queried (live writer still appending ticks).
* Today after 15:30 — cached once the file is stable and cache was built after
  the session close (or file mtime matches).
"""
from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any, Callable

CACHE_FILENAME = "market_db_inventory.json"
MARKET_CLOSE_HOUR = 15
MARKET_CLOSE_MINUTE = 30


def cache_file_path(data_dir: str) -> str:
    return os.path.join(data_dir, CACHE_FILENAME)


def _market_close_dt(day: datetime) -> datetime:
    return day.replace(
        hour=MARKET_CLOSE_HOUR,
        minute=MARKET_CLOSE_MINUTE,
        second=0,
        microsecond=0,
    )


def _parse_iso_ts(raw: str | None) -> datetime | None:
    if not raw:
        return None
    try:
        return datetime.fromisoformat(str(raw))
    except ValueError:
        return None


def load_cache(data_dir: str) -> dict[str, Any] | None:
    path = cache_file_path(data_dir)
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    databases = data.get("databases")
    if not isinstance(databases, dict):
        return None
    return data


def save_cache(data_dir: str, payload: dict[str, Any]) -> None:
    path = cache_file_path(data_dir)
    tmp = path + ".tmp"
    os.makedirs(data_dir, exist_ok=True)
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
        fh.write("\n")
    os.replace(tmp, path)


def _db_file_meta(db_path: str) -> tuple[int, float] | None:
    try:
        return os.path.getsize(db_path), os.path.getmtime(db_path)
    except OSError:
        return None


def _relative_db_path(db_path: str, data_dir: str) -> str:
    """Stable relative path for cache keys; basename when drives differ (Windows)."""
    try:
        return os.path.relpath(db_path, data_dir).replace("\\", "/")
    except ValueError:
        return os.path.basename(db_path)


def needs_reinventory(
    *,
    trading_day: str,
    db_path: str,
    entry: dict[str, Any] | None,
    cache_last_updated: datetime | None,
    now: datetime,
    data_dir: str,
) -> bool:
    """Return True when tick counts must be read from SQLite again."""
    if entry is None:
        return True

    rel_path = _relative_db_path(db_path, data_dir)
    if entry.get("db_path") != rel_path:
        return True
    if entry.get("db_file") != os.path.basename(db_path):
        return True
    if not entry.get("rows") and not entry.get("error"):
        return True

    meta = _db_file_meta(db_path)
    if meta is None:
        return True
    size_bytes, mtime = meta
    if int(entry.get("size_bytes") or -1) != size_bytes:
        return True

    stored_mtime = entry.get("mtime")
    if stored_mtime is not None and abs(float(stored_mtime) - mtime) > 1.0:
        return True

    today_str = now.strftime("%Y-%m-%d")
    if trading_day == today_str:
        close = _market_close_dt(now)
        if now < close:
            return True
        if cache_last_updated is None:
            return True
        if cache_last_updated.date() < now.date():
            return True
        if cache_last_updated < close:
            return True
        if mtime > cache_last_updated.timestamp() + 1.0:
            return True
        return False

    return False


def _flatten_inventory_rows(databases: list[dict]) -> list[dict[str, Any]]:
    flat_rows: list[dict[str, Any]] = []
    for entry in databases:
        for row in entry.get("rows") or []:
            flat_rows.append({
                **row,
                "db_file": entry["db_file"],
                "db_path": entry["db_path"],
                "db_total_ticks": entry["total_ticks"],
            })
    return flat_rows


def build_disk_only_response(*, data_dir: str) -> dict[str, Any] | None:
    """Return last persisted inventory without opening SQLite (instant after restart)."""
    cache = load_cache(data_dir)
    if not cache:
        return None
    cached_databases: dict[str, Any] = dict(cache.get("databases") or {})
    if not cached_databases:
        return None
    disk_days = sorted(cached_databases.keys(), reverse=True)
    databases = [cached_databases[day] for day in disk_days]
    return {
        "count": len(databases),
        "databases": databases,
        "rows": _flatten_inventory_rows(databases),
        "last_updated": cache.get("last_updated"),
        "cached": True,
        "stale": True,
    }


def build_inventory_response(
    *,
    data_dir: str,
    disk_days: list[str],
    resolve_db_path: Callable[[str], str | None],
    inventory_db: Callable[[str, str], dict],
    force: bool = False,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Folder-sync + cached SQLite inventory for View DB."""
    now = now or datetime.now()
    cache = None if force else load_cache(data_dir)
    cache_last_updated = _parse_iso_ts(cache.get("last_updated") if cache else None)
    cached_databases: dict[str, Any] = dict(cache.get("databases") or {}) if cache else {}

    disk_index: dict[str, str] = {}
    for day in disk_days:
        db_path = resolve_db_path(day)
        if db_path:
            disk_index[day] = db_path

    refreshed_any = False
    for day in list(cached_databases.keys()):
        if day not in disk_index:
            del cached_databases[day]
            refreshed_any = True

    for day, db_path in disk_index.items():
        prev = cached_databases.get(day)
        if force or needs_reinventory(
            trading_day=day,
            db_path=db_path,
            entry=prev,
            cache_last_updated=cache_last_updated,
            now=now,
            data_dir=data_dir,
        ):
            cached_databases[day] = inventory_db(day, db_path)
            refreshed_any = True
        else:
            cached_databases[day] = prev

    last_updated = now if refreshed_any or cache_last_updated is None else cache_last_updated
    cache_payload = {
        "last_updated": last_updated.isoformat(timespec="seconds"),
        "databases": {},
    }
    for day, entry in cached_databases.items():
        db_path = disk_index.get(day)
        meta = _db_file_meta(db_path) if db_path else None
        stored = dict(entry)
        if meta is not None:
            stored["mtime"] = meta[1]
        cache_payload["databases"][day] = stored

    save_cache(data_dir, cache_payload)

    databases = [cached_databases[day] for day in disk_days if day in cached_databases]
    flat_rows = _flatten_inventory_rows(databases)

    return {
        "count": len(databases),
        "databases": databases,
        "rows": flat_rows,
        "last_updated": cache_payload["last_updated"],
        "cached": not refreshed_any and cache is not None,
    }

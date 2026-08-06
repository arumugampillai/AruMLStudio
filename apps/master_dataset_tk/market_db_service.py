"""Market tick DB inventory — standalone (mirrors main app View DB)."""

from __future__ import annotations

import os
import re
import sqlite3
import time
from typing import Any, Callable

from chain_replay_ml.export_atm_pipeline import INDEX_CONFIG
from tick_data_paths import tick_search_dirs

_REPLAY_DB_DAY_ANY_RE = re.compile(r"^angel_market_(\d{4}-\d{2}-\d{2})(?:_[^.]+)?\.db$", re.I)


def data_dir(chart_dir: str) -> str:
    return os.path.join(chart_dir, "data")


def search_dirs(chart_dir: str) -> list[str]:
    return tick_search_dirs(chart_dir)


def db_has_ticks_table(db_path: str) -> bool:
    try:
        conn = sqlite3.connect(db_path)
        try:
            row = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='ticks'",
            ).fetchone()
            return row is not None
        finally:
            conn.close()
    except sqlite3.Error:
        return False


def db_candidates(chart_dir: str, day: str) -> list[str]:
    candidates: list[str] = []
    exact = f"angel_market_{day}.db"
    prefix = f"angel_market_{day}_"
    for root in search_dirs(chart_dir):
        if not os.path.isdir(root):
            continue
        exact_path = os.path.join(root, exact)
        if os.path.exists(exact_path):
            candidates.append(exact_path)
        try:
            for entry in os.listdir(root):
                if not entry.startswith(prefix) or not entry.endswith(".db"):
                    continue
                full = os.path.join(root, entry)
                if os.path.isfile(full):
                    candidates.append(full)
        except OSError:
            continue
    return candidates


def pick_db_path(chart_dir: str, day: str) -> str | None:
    dirs = search_dirs(chart_dir)
    primary = os.path.join(dirs[0], f"angel_market_{day}.db") if dirs else os.path.join(
        data_dir(chart_dir), f"angel_market_{day}.db",
    )
    if os.path.exists(primary) and db_has_ticks_table(primary):
        try:
            conn = sqlite3.connect(primary)
            try:
                count = conn.execute("SELECT COUNT(*) FROM ticks").fetchone()[0]
            finally:
                conn.close()
            if count > 0:
                return primary
        except sqlite3.Error:
            pass

    best: tuple[int, str] | None = None
    for path in db_candidates(chart_dir, day):
        if path == primary:
            continue
        if not os.path.exists(path):
            continue
        try:
            size = os.path.getsize(path)
        except OSError:
            continue
        if size <= 0 or not db_has_ticks_table(path):
            continue
        try:
            conn = sqlite3.connect(path)
            try:
                count = conn.execute("SELECT COUNT(*) FROM ticks").fetchone()[0]
            finally:
                conn.close()
        except sqlite3.Error:
            continue
        if int(count or 0) <= 0:
            continue
        if best is None or size > best[0]:
            best = (size, path)
    return best[1] if best else None


def available_replay_days(chart_dir: str) -> list[str]:
    found: set[str] = set()
    for root in search_dirs(chart_dir):
        if not os.path.isdir(root):
            continue
        for entry in os.listdir(root):
            m = _REPLAY_DB_DAY_ANY_RE.match(entry)
            if not m:
                continue
            full = os.path.join(root, entry)
            try:
                if os.path.getsize(full) <= 0:
                    continue
            except OSError:
                continue
            if db_has_ticks_table(full):
                found.add(m.group(1))
    return sorted(found, reverse=True)


def inventory_single_market_db(chart_dir: str, trading_day: str, db_path: str) -> dict[str, Any]:
    from storage.market_db_inventory_cache import _relative_db_path

    root = data_dir(chart_dir)
    rel_path = _relative_db_path(db_path, root)
    try:
        size_bytes = os.path.getsize(db_path)
    except OSError:
        size_bytes = 0

    rows: list[dict[str, Any]] = []
    total_ticks = 0
    error: str | None = None

    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        try:
            if not db_has_ticks_table(db_path):
                return {
                    "trading_day": trading_day,
                    "db_file": os.path.basename(db_path),
                    "db_path": rel_path,
                    "size_bytes": size_bytes,
                    "total_ticks": 0,
                    "rows": [],
                    "error": "No ticks table",
                }

            has_meta = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='token_day_meta'",
            ).fetchone()

            if has_meta:
                index_tokens = tuple(
                    str(cfg.get("index_token") or "").strip()
                    for cfg in INDEX_CONFIG.values()
                    if str(cfg.get("index_token") or "").strip()
                )
                chain_rows = conn.execute(
                    """
                    SELECT m.name AS index_name,
                           m.expiry_date AS expiry,
                           COUNT(t.id) AS tick_count
                    FROM token_day_meta m
                    INNER JOIN ticks t ON t.token = m.token
                    WHERE m.as_of_date = ?
                      AND m.name IS NOT NULL AND m.name != ''
                      AND m.expiry_date IS NOT NULL AND m.expiry_date != ''
                      AND m.option_type IN ('CE', 'PE')
                    GROUP BY m.name, m.expiry_date
                    ORDER BY m.name, m.expiry_date
                    """,
                    (trading_day,),
                ).fetchall()
                for r in chain_rows:
                    rows.append({
                        "trading_day": trading_day,
                        "index": str(r["index_name"]),
                        "expiry": str(r["expiry"]),
                        "tick_count": int(r["tick_count"]),
                        "kind": "chain",
                    })

                if index_tokens:
                    placeholders = ",".join("?" * len(index_tokens))
                    spot_rows = conn.execute(
                        f"""
                        SELECT m.name AS index_name,
                               COUNT(t.id) AS tick_count
                        FROM token_day_meta m
                        INNER JOIN ticks t ON t.token = m.token
                        WHERE m.as_of_date = ?
                          AND m.token IN ({placeholders})
                        GROUP BY m.name
                        ORDER BY m.name
                        """,
                        (trading_day, *index_tokens),
                    ).fetchall()
                    for r in spot_rows:
                        rows.append({
                            "trading_day": trading_day,
                            "index": str(r["index_name"]),
                            "expiry": "SPOT",
                            "tick_count": int(r["tick_count"]),
                            "kind": "spot",
                        })

            if rows:
                total_ticks = sum(int(r.get("tick_count") or 0) for r in rows)
            else:
                total_ticks = int(conn.execute("SELECT COUNT(*) FROM ticks").fetchone()[0])

            if not rows and total_ticks > 0:
                rows.append({
                    "trading_day": trading_day,
                    "index": "—",
                    "expiry": "—",
                    "tick_count": total_ticks,
                    "kind": "all",
                })
        finally:
            conn.close()
    except sqlite3.Error as exc:
        error = str(exc)

    return {
        "trading_day": trading_day,
        "db_file": os.path.basename(db_path),
        "db_path": rel_path,
        "size_bytes": size_bytes,
        "total_ticks": total_ticks,
        "rows": rows,
        "error": error,
    }


def load_inventory(chart_dir: str, *, force: bool = False) -> dict[str, Any]:
    from storage.market_db_inventory_cache import build_inventory_response

    root = data_dir(chart_dir)

    def resolve(day: str) -> str | None:
        return pick_db_path(chart_dir, day)

    def inventory_db(day: str, db_path: str) -> dict[str, Any]:
        return inventory_single_market_db(chart_dir, day, db_path)

    return build_inventory_response(
        data_dir=root,
        disk_days=available_replay_days(chart_dir),
        resolve_db_path=resolve,
        inventory_db=inventory_db,
        force=force,
    )


def load_inventory_disk_only(chart_dir: str) -> dict[str, Any] | None:
    from storage.market_db_inventory_cache import build_disk_only_response

    return build_disk_only_response(data_dir=data_dir(chart_dir))


def format_size_bytes(n: int | float | None) -> str:
    val = float(n or 0)
    if val < 1024:
        return f"{int(val)} B"
    if val < 1024 * 1024:
        return f"{val / 1024:.1f} KB"
    if val < 1024 * 1024 * 1024:
        return f"{val / (1024 * 1024):.1f} MB"
    return f"{val / (1024 * 1024 * 1024):.2f} GB"


def format_tick_count(n: int | float | None) -> str:
    val = int(n or 0)
    if val >= 1_000_000:
        return f"{val / 1_000_000:.2f}M"
    if val >= 1_000:
        return f"{val / 1_000:.1f}k"
    return str(val)


def probe_load_day_context(
    chart_dir: str,
    *,
    trading_day: str,
    market: str,
    expiry: str,
    interval_sec: int = 10,
    on_status: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    from chain_replay_ml.dataset_builder.day_context import SourceSpec, load_day_context
    from chain_replay_ml.dataset_builder.orchestrator import _source_from_dict

    src = _source_from_dict({
        "trading_day": trading_day,
        "market": market,
        "expiry": expiry,
    })
    if on_status:
        on_status(f"Loading {trading_day} · {market} · {expiry}…")
    t0 = time.perf_counter()
    ctx = load_day_context(chart_dir, src, feature_grid_step_sec=int(interval_sec))
    elapsed = time.perf_counter() - t0
    return {
        "ok": True,
        "elapsed_sec": round(elapsed, 3),
        "source_ticks": ctx.source_ticks,
        "spot_ticks": ctx.spot_ticks,
        "chain_ticks": ctx.chain_ticks,
        "db_path": ctx.db_path,
        "strikes": len(ctx.strike_mapping),
        "validation_lines": list(ctx.validation_lines),
    }

"""Probe option_low NULL path for the 9 affected tokens."""
from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path

MASTER = r"D:\data\master_dataset\master_dataset_nifty_3s.db"
DAY = "2026-07-24"

TOKENS = [
    "63913", "63911", "63909", "63907", "63905",
    "63948", "63903", "63901", "63899",
]


def main() -> None:
    conn = sqlite3.connect(MASTER)
    tables = [r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY 1"
    )]
    print("MASTER_TABLES", tables)
    has_tdm = "token_day_meta" in tables
    print("has_token_day_meta", has_tdm)

    # Samples side for affected tokens
    print("\n=== samples option OHLC for affected tokens ===")
    for tok in TOKENS:
        row = conn.execute(
            """
            SELECT token, symbol,
                   COUNT(*) n,
                   SUM(CASE WHEN option_low IS NULL THEN 1 ELSE 0 END),
                   SUM(CASE WHEN option_high IS NULL THEN 1 ELSE 0 END),
                   SUM(CASE WHEN option_open IS NULL THEN 1 ELSE 0 END),
                   SUM(CASE WHEN option_prev_close IS NULL THEN 1 ELSE 0 END),
                   MIN(ltp), MAX(ltp),
                   MIN(option_high), MAX(option_high),
                   MIN(option_open), MAX(option_open),
                   MIN(option_prev_close), MAX(option_prev_close),
                   MIN(option_low), MAX(option_low)
            FROM samples
            WHERE trading_day=? AND token=?
            """,
            (DAY, tok),
        ).fetchone()
        print(row)

    if has_tdm:
        cols = [r[1] for r in conn.execute("PRAGMA table_info(token_day_meta)")]
        print("\nTDM_COLS", cols)
        print("\n=== token_day_meta rows for affected tokens ===")
        ph = ",".join("?" for _ in TOKENS)
        # try as_of_date = DAY
        qcols = [c for c in (
            "token", "as_of_date", "trading_symbol", "symbol",
            "day_open", "day_high", "day_low", "prev_close",
            "open", "high", "low", "close",
        ) if c in cols]
        rows = conn.execute(
            f"SELECT {', '.join(qcols)} FROM token_day_meta "
            f"WHERE token IN ({ph}) ORDER BY token, as_of_date"
            if "as_of_date" in cols else
            f"SELECT {', '.join(qcols)} FROM token_day_meta WHERE token IN ({ph}) ORDER BY token",
            TOKENS,
        ).fetchall()
        print("qcols", qcols)
        for r in rows:
            print(r)

        # Compare: healthy token with option_low present
        healthy = conn.execute(
            """
            SELECT token, symbol FROM samples
            WHERE trading_day=? AND option_low IS NOT NULL
            LIMIT 1
            """,
            (DAY,),
        ).fetchone()
        print("\nHEALTHY_SAMPLE", healthy)
        if healthy and "as_of_date" in cols:
            h = conn.execute(
                f"SELECT {', '.join(qcols)} FROM token_day_meta "
                f"WHERE token=? AND as_of_date=?",
                (healthy[0], DAY),
            ).fetchall()
            print("HEALTHY_TDM", h)

    # Find replay DB for the day
    candidates = [
        rf"D:\data\angel_market_{DAY.replace('-', '')}.db",
        rf"D:\data\{DAY}\angel_market.db",
        rf"D:\data\replay\{DAY}.db",
        rf"D:\data\ticks\{DAY}.db",
    ]
    # search nearby
    tick_dir = os.environ.get("ARUMLSTUDIO_TICK_DATA_DIR", r"D:\data\ticks")
    for root in (r"D:\data", tick_dir):
        if not os.path.isdir(root):
            continue
        for dirpath, _, files in os.walk(root):
            for f in files:
                if f.endswith(".db") and ("2026" in f or "0724" in f or "07-24" in f or "0724" in dirpath):
                    candidates.append(os.path.join(dirpath, f))
            if len(candidates) > 40:
                break
        if len(candidates) > 40:
            break

    print("\nCANDIDATE_DBS")
    seen = set()
    for p in candidates:
        if p in seen:
            continue
        seen.add(p)
        exists = os.path.isfile(p)
        print(("OK" if exists else "--"), p)

    conn.close()


if __name__ == "__main__":
    main()

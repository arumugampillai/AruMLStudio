"""Null audit for one master trading day."""
from __future__ import annotations

import sqlite3

DB = r"D:/data/master_dataset/master_dataset_nifty_3s.db"
DAY = "2026-07-23"

PRIORITY = [
    "spot_high_ema20",
    "spot_low_ema20",
    "weighted_spot_close_ema",
    "spot_ema20_channel_width",
    "option_open",
    "option_high",
    "option_low",
    "option_prev_close",
    "futures_ltp",
    "futures_vwap",
    "spot_1m_ema9",
    "gamma_flip_spot",
    "spot_ema9",
    "ltp",
    "current_iv",
]


def main() -> None:
    conn = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    n = conn.execute(
        "SELECT COUNT(*) FROM samples WHERE trading_day=?", (DAY,)
    ).fetchone()[0]
    cols = [r[1] for r in conn.execute("PRAGMA table_info(samples)")]
    print(f"day={DAY} rows={n:,} cols={len(cols)}", flush=True)

    all_null: list[str] = []
    partial: list[tuple[str, float, int]] = []
    full = 0
    for c in cols:
        nn = (
            conn.execute(
                f'SELECT SUM(CASE WHEN "{c}" IS NOT NULL THEN 1 ELSE 0 END) '
                f"FROM samples WHERE trading_day=?",
                (DAY,),
            ).fetchone()[0]
            or 0
        )
        pct = 100.0 * nn / n if n else 0.0
        if nn == 0:
            all_null.append(c)
        elif nn < n:
            partial.append((c, round(pct, 2), int(nn)))
        else:
            full += 1

    print(f"fully_populated={full}", flush=True)
    print(f"partial_null={len(partial)}", flush=True)
    print(f"all_null={len(all_null)}", flush=True)

    print("--- PRIORITY ---", flush=True)
    for c in PRIORITY:
        if c not in cols:
            print(f"{c}: MISSING_FROM_SCHEMA", flush=True)
            continue
        nn = (
            conn.execute(
                f'SELECT SUM(CASE WHEN "{c}" IS NOT NULL THEN 1 ELSE 0 END) '
                f"FROM samples WHERE trading_day=?",
                (DAY,),
            ).fetchone()[0]
            or 0
        )
        print(f"{c}: {100.0 * nn / n:.2f}% ({nn:,}/{n:,})", flush=True)

    print("--- ALL-NULL ---", flush=True)
    for c in all_null:
        print(c, flush=True)

    print("--- PARTIAL worst 30 ---", flush=True)
    partial.sort(key=lambda x: x[1])
    for c, pct, nn in partial[:30]:
        print(f"{c}: {pct}% ({nn:,}/{n:,})", flush=True)

    conn.close()


if __name__ == "__main__":
    main()

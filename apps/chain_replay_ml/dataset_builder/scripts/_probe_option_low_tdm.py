"""Inspect token_day_meta + ticks for option_low-null tokens on 2026-07-24."""
from __future__ import annotations

import sqlite3

TICK_DB = r"D:\data\ticks\angel_market_2026-07-24.db"
DAY = "2026-07-24"
TOKENS = [
    "63913", "63911", "63909", "63907", "63905",
    "63948", "63903", "63901", "63899",
]

conn = sqlite3.connect(TICK_DB)
tables = [r[0] for r in conn.execute(
    "SELECT name FROM sqlite_master WHERE type='table' ORDER BY 1"
)]
print("TABLES", tables)
cols = [r[1] for r in conn.execute("PRAGMA table_info(token_day_meta)")]
print("TDM_COLS", cols)

ph = ",".join("?" for _ in TOKENS)
rows = conn.execute(
    f"""
    SELECT token, as_of_date, trading_symbol, option_type, strike_price,
           day_open, day_high, day_low, prev_close,
           first_seen_ts, last_updated_ts
    FROM token_day_meta
    WHERE as_of_date = ? AND token IN ({ph})
    ORDER BY token
    """,
    [DAY, *TOKENS],
).fetchall()
print("\n=== AFFECTED token_day_meta ===")
for r in rows:
    print(r)

# Healthy comparators: tokens with non-null day_low and similar
healthy = conn.execute(
    """
    SELECT token, trading_symbol, day_open, day_high, day_low, prev_close
    FROM token_day_meta
    WHERE as_of_date = ?
      AND day_low IS NOT NULL AND day_low > 0
      AND instrument_type LIKE '%OPT%'
    ORDER BY token
    LIMIT 5
    """,
    (DAY,),
).fetchall()
print("\n=== HEALTHY OPT token_day_meta sample ===")
for r in healthy:
    print(r)

# Count how many options have day_low null/0 vs positive
stats = conn.execute(
    """
    SELECT
      COUNT(*),
      SUM(CASE WHEN day_low IS NULL THEN 1 ELSE 0 END),
      SUM(CASE WHEN day_low = 0 THEN 1 ELSE 0 END),
      SUM(CASE WHEN day_low > 0 THEN 1 ELSE 0 END),
      SUM(CASE WHEN day_high IS NULL THEN 1 ELSE 0 END),
      SUM(CASE WHEN day_high = 0 THEN 1 ELSE 0 END),
      SUM(CASE WHEN day_high > 0 THEN 1 ELSE 0 END),
      SUM(CASE WHEN day_open IS NULL THEN 1 ELSE 0 END),
      SUM(CASE WHEN day_open = 0 THEN 1 ELSE 0 END),
      SUM(CASE WHEN day_open > 0 THEN 1 ELSE 0 END)
    FROM token_day_meta
    WHERE as_of_date = ?
    """,
    (DAY,),
).fetchone()
print("\nTDM_DAY_STATS count,low_null,low_0,low_pos,high_null,high_0,high_pos,open_null,open_0,open_pos")
print(stats)

# All tokens with day_low null or 0
bad = conn.execute(
    """
    SELECT token, trading_symbol, option_type, strike_price,
           day_open, day_high, day_low, prev_close
    FROM token_day_meta
    WHERE as_of_date = ?
      AND (day_low IS NULL OR day_low <= 0)
    ORDER BY day_low NULLS FIRST, token
    """,
    (DAY,),
).fetchall()
print(f"\n=== ALL bad day_low rows ({len(bad)}) ===")
for r in bad:
    print(r)

# Tick table schema + min/max LTP for affected
tick_cols = []
for tname in ("ticks", "tick", "market_ticks", "angel_ticks"):
    if tname in tables:
        tick_cols = [r[1] for r in conn.execute(f"PRAGMA table_info({tname})")]
        print(f"\nTICK_TABLE {tname} cols", tick_cols[:40])
        # find ltp-like
        ltp_col = next((c for c in tick_cols if c.lower() in ("ltp", "last_traded_price", "last_price")), None)
        print("ltp_col", ltp_col)
        if ltp_col:
            for tok in TOKENS[:3]:
                row = conn.execute(
                    f"""
                    SELECT COUNT(*), MIN("{ltp_col}"), MAX("{ltp_col}"),
                           MIN(CASE WHEN "{ltp_col}" > 0 THEN "{ltp_col}" END)
                    FROM {tname} WHERE token=?
                    """,
                    (tok,),
                ).fetchone()
                print("tick_stats", tok, row)
        break

conn.close()

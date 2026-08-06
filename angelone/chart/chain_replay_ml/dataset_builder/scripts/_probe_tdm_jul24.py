"""Check day_low for option_low-null tokens on 2026-07-24 tick DB."""
from __future__ import annotations

import sqlite3
from pathlib import Path

TOKENS = ["63913", "63911", "63909", "63907", "63905", "63948", "63903", "63901", "63899"]
DAY = "2026-07-24"
p = Path(rf"D:\data\ticks\angel_market_{DAY}.db")
print("exists", p, p.exists())
if not p.exists():
    raise SystemExit(1)
c = sqlite3.connect(f"file:{p}?mode=ro", uri=True)
cols = [r[1] for r in c.execute("PRAGMA table_info(token_day_meta)")]
print("cols", cols)
ph = ",".join("?" * len(TOKENS))
# as_of_date may be used instead of trading_day
for day_col in ("as_of_date", "trading_day"):
    if day_col not in cols:
        continue
    rows = c.execute(
        f"""
        SELECT token, day_open, day_high, day_low, prev_close
        FROM token_day_meta
        WHERE {day_col}=? AND CAST(token AS TEXT) IN ({ph})
        ORDER BY token
        """,
        (DAY, *TOKENS),
    ).fetchall()
    print(f"via {day_col}: {len(rows)} rows")
    for r in rows:
        print(" ", r)

# Also sample a healthy token that has option_low filled
healthy = c.execute(
    """
    SELECT token, day_open, day_high, day_low, prev_close
    FROM token_day_meta
    WHERE as_of_date=? AND day_low > 0
    LIMIT 5
    """,
    (DAY,),
).fetchall()
print("healthy samples", healthy)

# Count day_low <= 0 on that day
bad = c.execute(
    """
    SELECT COUNT(*),
           SUM(CASE WHEN day_low IS NULL OR day_low <= 0 THEN 1 ELSE 0 END)
    FROM token_day_meta WHERE as_of_date=?
    """,
    (DAY,),
).fetchone()
print("token_day_meta totals", bad)
c.close()

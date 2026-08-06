"""Compare Master option_low vs token_day_meta day_low for bad vs good tokens."""
from __future__ import annotations

import sqlite3

DAY = "2026-07-24"
BAD = ["63913", "63911", "63909"]
# pick a CE token that has option_low filled in master
m = sqlite3.connect(r"D:\data\master_dataset\master_dataset_nifty_3s.db")
good = m.execute(
    """
    SELECT token, COUNT(*), SUM(CASE WHEN option_low IS NULL THEN 1 ELSE 0 END)
    FROM samples WHERE trading_day=?
    GROUP BY token
    HAVING SUM(CASE WHEN option_low IS NULL THEN 1 ELSE 0 END)=0
    LIMIT 5
    """,
    (DAY,),
).fetchall()
print("good tokens", good)

t = sqlite3.connect(rf"D:\data\ticks\angel_market_{DAY}.db")
for tok in BAD + [str(g[0]) for g in good[:3]]:
    tdm = t.execute(
        """
        SELECT day_open, day_high, day_low, prev_close
        FROM token_day_meta WHERE as_of_date=? AND CAST(token AS TEXT)=?
        """,
        (DAY, str(tok)),
    ).fetchone()
    mast = m.execute(
        """
        SELECT
          SUM(CASE WHEN option_low IS NULL THEN 1 ELSE 0 END),
          SUM(CASE WHEN option_open IS NULL THEN 1 ELSE 0 END),
          MIN(option_low), MAX(option_low), MIN(option_open), MAX(option_open),
          COUNT(*)
        FROM samples WHERE trading_day=? AND CAST(token AS TEXT)=?
        """,
        (DAY, str(tok)),
    ).fetchone()
    print(f"token={tok}")
    print(f"  TDM open/high/low/pc={tdm}")
    print(f"  Master ol_null={mast[0]} oo_null={mast[1]} ol_range=({mast[2]},{mast[3]}) oo_range=({mast[4]},{mast[5]}) n={mast[6]}")

m.close()
t.close()

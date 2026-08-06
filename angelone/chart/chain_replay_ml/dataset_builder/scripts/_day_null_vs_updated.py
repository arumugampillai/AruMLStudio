import sqlite3
import json

conn = sqlite3.connect(r"file:D:/data/master_dataset/master_dataset_nifty_3s.db?mode=ro", uri=True)
rows = conn.execute(
    "SELECT trading_day, row_count, token_count, last_updated, status "
    "FROM master_dataset_days ORDER BY trading_day"
).fetchall()
for r in rows:
    print("\t".join(str(x) for x in r))

# per-day non-null for key cols vs last_updated
cols = [
    "spot_high_ema20",
    "option_open",
    "futures_ltp",
    "spot_1m_ema9",
    "gamma_flip_spot",
    "spot_ema9",
    "ltp",
]
print("--- per day nonnull pct ---")
for day, *_ in rows:
    parts = []
    for c in cols:
        n, nn = conn.execute(
            f'SELECT COUNT(*), SUM(CASE WHEN "{c}" IS NOT NULL THEN 1 ELSE 0 END) '
            f'FROM samples WHERE trading_day=?',
            (day,),
        ).fetchone()
        pct = 100.0 * (nn or 0) / n if n else 0
        parts.append(f"{c}:{pct:.0f}%")
    print(day, " | ".join(parts))
conn.close()

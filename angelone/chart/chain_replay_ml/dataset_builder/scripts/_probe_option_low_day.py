"""Deep-dive option_low NULLs on Master day."""
from __future__ import annotations

import sqlite3
from datetime import datetime
from zoneinfo import ZoneInfo

DB = r"D:\data\master_dataset\master_dataset_nifty_3s.db"
DAY = "2026-07-24"
IST = ZoneInfo("Asia/Kolkata")

conn = sqlite3.connect(DB)
print("=== Session OHLC null counts ===")
for c in [
    "option_open",
    "option_high",
    "option_low",
    "option_prev_close",
    "spot_open",
    "spot_high",
    "spot_low",
    "spot_prev_close",
]:
    n = conn.execute(
        f'SELECT SUM(CASE WHEN "{c}" IS NULL THEN 1 ELSE 0 END) '
        f"FROM samples WHERE trading_day=?",
        (DAY,),
    ).fetchone()[0]
    print(f"  {c}: {n}")

print("\n=== option_low by token ===")
rows = conn.execute(
    """
    SELECT token, option_type, strike,
      COUNT(*) n,
      SUM(CASE WHEN option_low IS NULL THEN 1 ELSE 0 END) ol_null,
      MIN(timestamp) t0, MAX(timestamp) t1
    FROM samples WHERE trading_day=?
    GROUP BY token
    ORDER BY ol_null DESC
    """,
    (DAY,),
).fetchall()
for r in rows:
    if r[4] <= 0:
        continue
    t0 = datetime.fromtimestamp(r[5], IST).strftime("%H:%M:%S")
    t1 = datetime.fromtimestamp(r[6], IST).strftime("%H:%M:%S")
    print(f"  {r[0]} {r[1]} K={r[2]} n={r[3]} ol_null={r[4]} span={t0}-{t1}")

bad = [r for r in rows if r[4] > 0]
print(f"\ntokens with option_low NULL: {len(bad)} / {len(rows)}")
print(f"sum ol_null: {sum(r[4] for r in bad)}")
full = sum(1 for r in bad if r[4] == r[3])
partial = sum(1 for r in bad if 0 < r[4] < r[3])
print(f"full-token-null tokens: {full}  partial: {partial}")

print("\n=== option_low NULL by IST hour ===")
for h, n in conn.execute(
    """
    SELECT CAST(strftime('%H', datetime(timestamp, 'unixepoch', '+5 hours', '+30 minutes')) AS INT) h,
           COUNT(*)
    FROM samples WHERE trading_day=? AND option_low IS NULL
    GROUP BY h ORDER BY h
    """,
    (DAY,),
).fetchall():
    print(f"  hour {h:02d}: {n}")

print("\n=== joint with other option OHLC ===")
print(
    conn.execute(
        """
        SELECT
          SUM(CASE WHEN option_low IS NULL AND option_open IS NULL THEN 1 ELSE 0 END) AS low_and_open_null,
          SUM(CASE WHEN option_low IS NULL AND option_open IS NOT NULL THEN 1 ELSE 0 END) AS low_null_open_ok,
          SUM(CASE WHEN option_low IS NOT NULL AND option_open IS NULL THEN 1 ELSE 0 END) AS low_ok_open_null,
          SUM(CASE WHEN option_low IS NULL AND option_high IS NULL THEN 1 ELSE 0 END) AS low_and_high_null,
          SUM(CASE WHEN option_low IS NULL AND option_prev_close IS NULL THEN 1 ELSE 0 END) AS low_and_pc_null,
          SUM(CASE WHEN option_low IS NULL AND ltp IS NOT NULL THEN 1 ELSE 0 END) AS low_null_ltp_ok
        FROM samples WHERE trading_day=?
        """,
        (DAY,),
    ).fetchone()
)

# After option_low forgiven: remaining incomplete outside warmup — top co-null features
print("\ndone")
conn.close()

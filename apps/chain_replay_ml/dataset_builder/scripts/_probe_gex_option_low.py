import json
import sqlite3
from datetime import datetime
from zoneinfo import ZoneInfo

DB = r"D:\data\master_dataset\master_dataset_nifty_3s.db"
DAY = "2026-07-24"
IST = ZoneInfo("Asia/Kolkata")
conn = sqlite3.connect(DB)

def ist(ts):
    if ts is None:
        return None
    return datetime.fromtimestamp(float(ts), tz=IST).strftime("%H:%M:%S")

# GEX family time coverage
for c in ("call_gex", "put_gex", "net_gex", "chain_gex", "gamma_flip_spot", "gamma_flip_distance"):
    row = conn.execute(
        f"""
        SELECT
          SUM(CASE WHEN "{c}" IS NULL THEN 1 ELSE 0 END),
          SUM(CASE WHEN "{c}" IS NOT NULL THEN 1 ELSE 0 END),
          MIN(CASE WHEN "{c}" IS NOT NULL THEN timestamp END),
          MAX(CASE WHEN "{c}" IS NOT NULL THEN timestamp END),
          MIN(CASE WHEN "{c}" IS NULL THEN timestamp END),
          MAX(CASE WHEN "{c}" IS NULL THEN timestamp END)
        FROM samples WHERE trading_day=?
        """,
        (DAY,),
    ).fetchone()
    print(
        f"{c:20} null={row[0]} ok={row[1]} ok_range={ist(row[2])}-{ist(row[3])} "
        f"null_range={ist(row[4])}-{ist(row[5])}"
    )

# After 12:30 — are GEX values null while IV present?
cut = datetime(2026, 7, 24, 12, 30, 0, tzinfo=IST).timestamp()
row = conn.execute(
    """
    SELECT
      COUNT(*),
      SUM(CASE WHEN gamma_flip_spot IS NULL THEN 1 ELSE 0 END),
      SUM(CASE WHEN call_gex IS NULL THEN 1 ELSE 0 END),
      SUM(CASE WHEN net_gex IS NULL THEN 1 ELSE 0 END),
      SUM(CASE WHEN current_iv IS NULL THEN 1 ELSE 0 END),
      SUM(CASE WHEN ltp IS NULL THEN 1 ELSE 0 END)
    FROM samples WHERE trading_day=? AND timestamp > ?
    """,
    (DAY, cut),
).fetchone()
print("AFTER_1230", dict(zip(
    ["rows", "gf_null", "call_gex_null", "net_gex_null", "iv_null", "ltp_null"], row
)))

# Before 12:30
row = conn.execute(
    """
    SELECT
      COUNT(*),
      SUM(CASE WHEN gamma_flip_spot IS NULL THEN 1 ELSE 0 END),
      SUM(CASE WHEN call_gex IS NULL THEN 1 ELSE 0 END),
      SUM(CASE WHEN net_gex IS NULL THEN 1 ELSE 0 END)
    FROM samples WHERE trading_day=? AND timestamp <= ?
    """,
    (DAY, cut),
).fetchone()
print("THRU_1230", dict(zip(["rows", "gf_null", "call_gex_null", "net_gex_null"], row)))

# option_low: which tokens, is day_low <=0 in source?
# Check distinct tokens with option_low null
tok = conn.execute(
    """
    SELECT token, symbol,
           COUNT(*) n,
           SUM(CASE WHEN option_low IS NULL THEN 1 ELSE 0 END) ol_null,
           MIN(ltp), MAX(ltp), AVG(ltp)
    FROM samples WHERE trading_day=?
    GROUP BY token
    HAVING ol_null > 0
    ORDER BY ol_null DESC
    """,
    (DAY,),
).fetchall()
print("option_low_null_tokens", len(tok))
for t in tok[:15]:
    print(" ", t)

# scenarios from deep json
deep = json.loads(open(
    r"chain_replay_ml/dataset_builder/scripts/null_audit_2026-07-24_deep.json", encoding="utf-8"
).read())
print("SCENARIOS", deep["scenarios_complete_rows"])
print("RECOVERY", deep["recovery_vs_baseline"])
print("BY_OPT", deep["by_option_type"])
print("TOK_SUM", deep["gamma_flip_token_summary"])

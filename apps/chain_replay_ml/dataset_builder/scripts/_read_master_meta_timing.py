import json
import sqlite3

conn = sqlite3.connect(r"file:D:/data/master_dataset/master_dataset_nifty_3s.db?mode=ro", uri=True)
keys = [r[0] for r in conn.execute("SELECT key FROM dataset_meta ORDER BY 1")]
print("meta_keys", keys)
for k in keys:
    if any(x in k.lower() for x in ("profil", "timing", "stats", "perf", "build")):
        v = conn.execute("SELECT value FROM dataset_meta WHERE key=?", (k,)).fetchone()[0]
        print("---", k)
        print((v or "")[:800])
days = conn.execute(
    "SELECT trading_day, row_count, last_updated FROM master_dataset_days ORDER BY last_updated DESC LIMIT 5"
).fetchall()
print("days", days)
conn.close()

import sqlite3
from pathlib import Path

db = r"D:/data/master_dataset/master_dataset_nifty_3s.db"
conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
tables = [r[0] for r in conn.execute(
    "SELECT name FROM sqlite_master WHERE type='table' ORDER BY 1"
)]
print("tables:", tables)
for t in tables:
    if any(x in t.lower() for x in ("day", "meta", "build", "insert", "fingerprint")):
        cols = [r[1] for r in conn.execute(f"PRAGMA table_info({t})")]
        print(t, cols)
        try:
            n = conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
            print("  rows", n)
            rows = conn.execute(f"SELECT * FROM {t} LIMIT 3").fetchall()
            for r in rows:
                print(" ", r[:12])
        except Exception as e:
            print(" ", e)
conn.close()

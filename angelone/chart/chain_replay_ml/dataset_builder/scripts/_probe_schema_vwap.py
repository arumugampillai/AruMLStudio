import json
import sqlite3

conn = sqlite3.connect(r"D:/data/master_dataset/master_dataset_nifty_3s.db")
print(conn.execute("SELECT key FROM dataset_meta").fetchall())
for key in ("master_config", "build_schema"):
    row = conn.execute("SELECT value FROM dataset_meta WHERE key=?", (key,)).fetchone()
    if not row:
        print(key, "MISSING")
        continue
    doc = json.loads(row[0])
    if key == "build_schema":
        feats = doc.get("feature_columns") or []
        print("build_schema n", len(feats))
        for f in ("option_vwap", "futures_vwap", "futures_ltp", "bid_ask_spread", "ltp"):
            print(f, "in", f in feats)
    else:
        print("master_config feature_count", doc.get("feature_count"))
conn.close()

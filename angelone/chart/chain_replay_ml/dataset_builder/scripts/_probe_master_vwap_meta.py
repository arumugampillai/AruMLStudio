"""Inspect master meta for VWAP column provenance."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

MASTER = Path(r"D:/data/master_dataset/master_dataset_nifty_3s.db")
conn = sqlite3.connect(str(MASTER))

for table in (
    "master_dataset_meta",
    "dataset_meta",
    "master_dataset_meta_history",
    "builder_progress",
):
    cols = [r[1] for r in conn.execute(f"PRAGMA table_info({table})")]
    print(table, cols)
    try:
        n = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        print("  rows", n)
        rows = conn.execute(f"SELECT * FROM {table} LIMIT 3").fetchall()
        for r in rows:
            s = str(r)
            if len(s) > 400:
                s = s[:400] + "..."
            print(" ", s)
    except Exception as exc:
        print(" ", exc)

# build_schema feature list contains vwap?
for key in ("build_schema", "master_config", "feature_columns", "created_at", "built_at"):
    try:
        row = conn.execute(
            "SELECT value FROM master_dataset_meta WHERE key=?", (key,)
        ).fetchone()
        if row:
            val = row[0]
            if key == "build_schema":
                doc = json.loads(val)
                feats = doc.get("feature_columns") or []
                print("build_schema feature_count", len(feats))
                print("option_vwap in schema", "option_vwap" in feats)
                print("futures_vwap in schema", "futures_vwap" in feats)
                print("futures_ltp in schema", "futures_ltp" in feats)
            else:
                print(key, str(val)[:300])
    except Exception as exc:
        print(key, exc)

# column order / creation: check if vwap columns are near end (late ADD COLUMN)
cols = [r[1] for r in conn.execute("PRAGMA table_info(samples)")]
for name in ("ltp", "futures_ltp", "futures_vwap", "option_vwap", "bid_ask_spread"):
    print(name, "cid", cols.index(name) if name in cols else None, "of", len(cols))

conn.close()

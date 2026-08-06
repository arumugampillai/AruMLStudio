"""Check token_day_meta day_low for option_low-null tokens."""
from __future__ import annotations

import sqlite3
from pathlib import Path

TOKENS = ["63913", "63911", "63909", "63907", "63905", "63948", "63903", "63901", "63899"]
DAY = "2026-07-24"

found = []
for p in Path(r"D:\data").rglob("*.db"):
    name = p.name.lower()
    if "master" in name or "bak" in name or "pre_rebuild" in name:
        continue
    try:
        if p.stat().st_size > 8_000_000_000:
            continue
    except OSError:
        continue
    try:
        c = sqlite3.connect(f"file:{p}?mode=ro", uri=True)
        tables = [
            r[0]
            for r in c.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        ]
        if "token_day_meta" in tables:
            found.append(p)
        c.close()
    except Exception:
        pass

print("token_day_meta DBs:", [str(p) for p in found[:15]])
for p in found[:5]:
    c = sqlite3.connect(f"file:{p}?mode=ro", uri=True)
    cols = [r[1] for r in c.execute("PRAGMA table_info(token_day_meta)")]
    print(p.name, "cols", cols[:20])
    qcols = [x for x in ("token", "trading_day", "day_low", "day_high", "day_open", "prev_close") if x in cols]
    if "token" not in cols:
        c.close()
        continue
    placeholders = ",".join("?" * len(TOKENS))
    day_col = "trading_day" if "trading_day" in cols else None
    if day_col:
        sql = (
            f"SELECT {', '.join(qcols)} FROM token_day_meta "
            f"WHERE trading_day=? AND token IN ({placeholders})"
        )
        rows = c.execute(sql, (DAY, *TOKENS)).fetchall()
    else:
        sql = f"SELECT {', '.join(qcols)} FROM token_day_meta WHERE token IN ({placeholders})"
        rows = c.execute(sql, TOKENS).fetchall()
    print(" rows", len(rows))
    for r in rows[:20]:
        print(" ", r)
    c.close()

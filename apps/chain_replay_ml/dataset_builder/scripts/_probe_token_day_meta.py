import os
import sqlite3
import sys

_apps_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _apps_dir not in sys.path:
    sys.path.insert(0, _apps_dir)
from tick_data_paths import resolve_tick_data_dir

root = os.environ.get("ARUMLSTUDIO_TICK_DATA_DIR") or resolve_tick_data_dir()
for day in ["2026-07-24", "2026-07-23", "2026-05-26", "2026-06-25", "2026-07-13"]:
    path = os.path.join(root, f"angel_market_{day}.db")
    if not os.path.isfile(path):
        print(day, "NO_DB")
        continue
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    tables = [
        r[0]
        for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY 1"
        )
    ]
    if "token_day_meta" not in tables:
        print(day, "no token_day_meta", tables[:10])
        conn.close()
        continue
    cols = [r[1] for r in conn.execute("PRAGMA table_info(token_day_meta)")]
    n = conn.execute("SELECT COUNT(*) FROM token_day_meta").fetchone()[0]
    open_cols = [c for c in cols if "open" in c.lower()]
    print(day, "rows", n, "open_cols", open_cols)
    if open_cols:
        c = open_cols[0]
        nn = conn.execute(
            f'SELECT SUM(CASE WHEN "{c}" IS NOT NULL THEN 1 ELSE 0 END) FROM token_day_meta'
        ).fetchone()[0]
        print("  nonnull", c, nn)
    conn.close()

import os
import sqlite3

root = r"c:/Users/admin/PycharmProjects/v1/AruNeo/angelone/chart/data"
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

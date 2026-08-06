"""Probe option session OHLC source for a trading day."""
from __future__ import annotations

import os
import sqlite3
import sys

sys.path.insert(0, r"c:/Users/admin/PycharmProjects/v1/AruNeo/angelone/chart")

from chain_replay_ml.export_atm_pipeline import replay_db_path
from chain_replay_ml.dataset_builder.session_ohlc import load_session_ohlc_by_token

CHART = r"c:/Users/admin/PycharmProjects/v1/AruNeo/angelone/chart"
DAY = "2026-07-23"
MDB = r"D:/data/master_dataset/master_dataset_nifty_3s.db"


def main() -> None:
    db = replay_db_path(CHART, DAY)
    print("replay_db", db, "exists", bool(db and os.path.isfile(db or "")), flush=True)
    if not db or not os.path.isfile(db):
        return
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    tables = [
        r[0]
        for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY 1"
        )
    ]
    print("has token_day_meta", "token_day_meta" in tables, flush=True)
    if "token_day_meta" not in tables:
        print("tables", tables[:20], flush=True)
        conn.close()
        return
    cols = [r[1] for r in conn.execute("PRAGMA table_info(token_day_meta)")]
    print("cols", cols, flush=True)
    dates = conn.execute(
        "SELECT as_of_date, COUNT(*) FROM token_day_meta GROUP BY 1 ORDER BY 1 DESC LIMIT 10"
    ).fetchall()
    print("tdm dates", dates, flush=True)
    n = conn.execute(
        "SELECT COUNT(*) FROM token_day_meta WHERE as_of_date=?", (DAY,)
    ).fetchone()[0]
    print("tdm rows for day", n, flush=True)
    sample = conn.execute(
        """
        SELECT token, typeof(token), day_open, day_high, day_low, prev_close
        FROM token_day_meta
        WHERE as_of_date=?
        LIMIT 8
        """,
        (DAY,),
    ).fetchall()
    for row in sample:
        print(" sample", row, flush=True)
    for c in ("day_open", "day_high", "day_low", "prev_close"):
        if c not in cols:
            print(c, "missing col", flush=True)
            continue
        nn = conn.execute(
            f'SELECT SUM(CASE WHEN "{c}" IS NOT NULL AND "{c}" > 0 THEN 1 ELSE 0 END) '
            f"FROM token_day_meta WHERE as_of_date=?",
            (DAY,),
        ).fetchone()[0]
        print(f"  {c} >0 : {nn}", flush=True)

    mconn = sqlite3.connect(f"file:{MDB}?mode=ro", uri=True)
    tokens = [
        str(r[0])
        for r in mconn.execute(
            "SELECT DISTINCT token FROM samples WHERE trading_day=? LIMIT 30",
            (DAY,),
        )
    ]
    print("master tokens sample", tokens[:8], flush=True)
    loaded = load_session_ohlc_by_token(conn, tokens, as_of_date=DAY)
    nonempty = sum(1 for v in loaded.values() if any(x is not None for x in v.values()))
    print("loaded nonempty", nonempty, "of", len(tokens), flush=True)
    for t in tokens[:5]:
        print(" ", t, loaded.get(t), flush=True)

    # Direct cast match test
    if tokens:
        t0 = tokens[0]
        row = conn.execute(
            """
            SELECT token, day_open, day_high, day_low, prev_close
            FROM token_day_meta
            WHERE as_of_date=? AND CAST(token AS TEXT)=?
            """,
            (DAY, t0),
        ).fetchone()
        print("cast match for", t0, row, flush=True)

    for c in ("spot_open", "spot_high", "option_open", "option_low"):
        try:
            nn = mconn.execute(
                f'SELECT SUM(CASE WHEN "{c}" IS NOT NULL THEN 1 ELSE 0 END) '
                f"FROM samples WHERE trading_day=?",
                (DAY,),
            ).fetchone()[0]
            print(f"master {c} nonnull", nn, flush=True)
        except Exception as e:
            print("master", c, e, flush=True)
    mconn.close()
    conn.close()


if __name__ == "__main__":
    main()

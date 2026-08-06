"""Reproduce ATP → option_vwap / futures_vwap from tick DB vs master."""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from chain_replay_ml.ticks import load_tick_timelines  # noqa: E402
from chain_replay_ml.dataset_builder.futures_context import (  # noqa: E402
    emit_futures_timeline_features,
)

TICK = Path(r"D:/data/ticks/angel_market_2026-07-24.db")
MASTER = Path(r"D:/data/master_dataset/master_dataset_nifty_3s.db")


def main() -> None:
    tconn = sqlite3.connect(str(TICK))
    mconn = sqlite3.connect(str(MASTER))

    # Option token with ATP in ticks and rows in master
    row = mconn.execute(
        """
        SELECT token, timestamp, ltp, option_vwap, futures_ltp, futures_vwap
        FROM samples
        WHERE trading_day='2026-07-24' AND ltp IS NOT NULL
        LIMIT 1
        """
    ).fetchone()
    print("master sample row:", row)
    token, ts, ltp, ov, fl, fv = row

    atp_stats = tconn.execute(
        "SELECT COUNT(*), SUM(atp>0), MIN(CASE WHEN atp>0 THEN atp END), MAX(atp) "
        "FROM ticks WHERE token=?",
        (token,),
    ).fetchone()
    print("option tick atp stats:", atp_stats)

    fut_token = tconn.execute(
        "SELECT token FROM token_day_meta WHERE instrument_type='FUTIDX' LIMIT 1"
    ).fetchone()[0]
    print("futures token:", fut_token)
    print(
        "futures tick atp:",
        tconn.execute(
            "SELECT COUNT(*), SUM(atp>0), MIN(CASE WHEN atp>0 THEN atp END), MAX(atp) "
            "FROM ticks WHERE token=?",
            (fut_token,),
        ).fetchone(),
    )

    # Load timelines around sample ts
    open_ts = float(ts) - 3600
    close_ts = float(ts) + 3600
    tls = load_tick_timelines(tconn, [token, fut_token], open_ts, close_ts)
    opt_tl = tls[token]
    fut_tl = tls[fut_token]
    print("opt timeline len", len(opt_tl.timestamps), "atps len", len(opt_tl.atps_paise))
    print("fut timeline len", len(fut_tl.timestamps), "atps len", len(fut_tl.atps_paise))
    print("opt atp_rupees_at", opt_tl.atp_rupees_at(float(ts)), "ltp_rupees_at", opt_tl.ltp_rupees_at(float(ts)))
    print("fut atp_rupees_at", fut_tl.atp_rupees_at(float(ts)), "ltp_rupees_at", fut_tl.ltp_rupees_at(float(ts)))
    print("positive atp samples in opt timeline", sum(1 for a in opt_tl.atps_paise if a > 0))
    print("positive atp samples in fut timeline", sum(1 for a in fut_tl.atps_paise if a > 0))

    emitted = emit_futures_timeline_features({}, ts=float(ts), futures_tl=fut_tl)
    print("emit futures:", emitted)

    # When was master built / schema?
    try:
        meta = mconn.execute(
            "SELECT key, value FROM meta WHERE key LIKE '%schema%' OR key LIKE '%built%' OR key LIKE '%version%' LIMIT 20"
        ).fetchall()
        print("master meta keys sample:", meta[:10])
    except Exception as exc:
        print("meta read:", exc)
        tables = mconn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        print("tables", tables)

    tconn.close()
    mconn.close()


if __name__ == "__main__":
    main()

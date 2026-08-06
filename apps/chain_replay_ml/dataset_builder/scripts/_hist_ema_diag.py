"""Historical Bar Check for spot_*_ema200 — diagnostics only."""
from __future__ import annotations

import os
import sqlite3
from datetime import datetime
from zoneinfo import ZoneInfo

from chain_replay_ml.dataset_builder.historic_spot_ema_context import (
    build_historic_spot_ema_book,
    historic_spot_ema_feature_name,
    resolve_historic_bars_db_path,
)
from chain_replay_ml.feature_policy.metadata import build_feature_policy_metadata

IST = ZoneInfo("Asia/Kolkata")
DAY = "2026-07-24"
MASTER = r"D:\data\master_dataset\master_dataset_nifty_3s.db"
CHART_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)


def _fmt(ts: float | None) -> str:
    if ts is None:
        return "(none)"
    return datetime.fromtimestamp(float(ts), tz=IST).strftime("%Y-%m-%d %H:%M:%S IST")


def main() -> None:
    db = resolve_historic_bars_db_path(CHART_DIR)
    print("chart_dir", CHART_DIR)
    print("historic_db", db, "exists", os.path.isfile(db) if db else False)

    print("\nFeature Policy metadata:")
    for n in (
        "spot_1m_ema200",
        "spot_3m_ema200",
        "spot_5m_ema200",
        "spot_15m_ema200",
        "spot_ema200",
        "ltp_ema200",
    ):
        m = build_feature_policy_metadata(n)
        print(
            f"  {n}: category={m.feature_category} lifecycle={m.lifecycle} "
            f"warmup_samples={m.intrinsic_warmup_samples} "
            f"warmup_sec={m.intrinsic_warmup_sec} mode={m.warmup_mode}"
        )

    book = build_historic_spot_ema_book(trading_day=DAY, chart_dir=CHART_DIR)
    print("\nHistorical Bar Check")
    print("=" * 60)
    for s in book.series:
        print(f"{s.label} bars loaded: {len(s.timestamps):,}")
        if s.timestamps:
            print(f"  Earliest bar: {_fmt(s.timestamps[0])}")
            print(f"  Latest bar:   {_fmt(s.timestamps[-1])}")
        ema = s.emas.get(200)
        first_i = next((i for i, v in enumerate(ema or ()) if v is not None), None)
        if first_i is None:
            print("  EMA200 first non-null: (never)")
        else:
            print(
                f"  EMA200 first non-null: {_fmt(s.timestamps[first_i])} "
                f"(bar index {first_i}/{len(s.timestamps) - 1})"
            )
        open_ts = datetime(2026, 7, 24, 9, 16, tzinfo=IST).timestamp()
        levels = book.levels_at(open_ts)
        key = historic_spot_ema_feature_name(s.label, 200)
        print(f"  asof 09:16 on {DAY}: {key}={levels.get(key)}")

    print("\nEMA200 first non-null summary:")
    for s in book.series:
        ema = s.emas.get(200)
        first_i = next((i for i, v in enumerate(ema or ()) if v is not None), None)
        print(
            f"  {s.label:4s}: "
            + (_fmt(s.timestamps[first_i]) if first_i is not None else "(never)")
        )

    print("\nMaster sample first non-null (any token):")
    conn = sqlite3.connect(MASTER)
    for feat in (
        "spot_1m_ema200",
        "spot_3m_ema200",
        "spot_5m_ema200",
        "spot_15m_ema200",
        "spot_ema200",
    ):
        row = conn.execute(
            f"""
            SELECT COUNT(*),
                   SUM(CASE WHEN "{feat}" IS NULL THEN 1 ELSE 0 END)
            FROM samples WHERE trading_day=?
            """,
            (DAY,),
        ).fetchone()
        first = conn.execute(
            f"""
            SELECT MIN(timestamp) FROM samples
            WHERE trading_day=? AND "{feat}" IS NOT NULL
            """,
            (DAY,),
        ).fetchone()[0]
        print(
            f"  {feat}: nulls={int(row[1] or 0):,}/{int(row[0]):,}  "
            f"first_non_null={_fmt(first) if first else '(never)'}"
        )

    print("\nspot_1m_ema200 NULL by IST hour:")
    for h, n in conn.execute(
        """
        SELECT CAST(strftime('%H', datetime(timestamp, 'unixepoch',
               '+5 hours', '+30 minutes')) AS INT),
               COUNT(*)
        FROM samples WHERE trading_day=? AND spot_1m_ema200 IS NULL
        GROUP BY 1 ORDER BY 1
        """,
        (DAY,),
    ):
        print(f"  hour {h:02d}: {n}")

    print("\nPer-token null count for spot_1m_ema200:")
    all_tok = conn.execute(
        """
        SELECT token, COUNT(*) n,
               SUM(CASE WHEN spot_1m_ema200 IS NULL THEN 1 ELSE 0 END) nn
        FROM samples WHERE trading_day=?
        GROUP BY token ORDER BY nn DESC
        """,
        (DAY,),
    ).fetchall()
    for r in all_tok[:15]:
        print(f"  token={r[0]} n={r[1]} null={r[2]}")
    pred200 = sum(min(200, r[1]) for r in all_tok)
    print(f"\nsum(min(200, n_token))={pred200}  actual_nulls={sum(r[2] for r in all_tok)}")
    conn.close()


if __name__ == "__main__":
    main()

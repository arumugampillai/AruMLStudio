"""Deeper attribution for gamma_flip / option_low / exclusive killers."""
from __future__ import annotations

import json
import sqlite3
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

DB = r"D:\data\master_dataset\master_dataset_nifty_3s.db"
DAY = "2026-07-24"
IST = ZoneInfo("Asia/Kolkata")
OUT = Path(__file__).with_name("null_audit_2026-07-24_deep.json")

conn = sqlite3.connect(DB)
cols = [r[1] for r in conn.execute("PRAGMA table_info(samples)")]
where = "trading_day = ?"
params = (DAY,)
total = int(conn.execute(f"SELECT COUNT(*) FROM samples WHERE {where}", params).fetchone()[0])

# 100% null
all_null = []
for c in cols:
    n = int(
        conn.execute(
            f'SELECT SUM(CASE WHEN "{c}" IS NULL THEN 1 ELSE 0 END) FROM samples WHERE {where}',
            params,
        ).fetchone()[0]
        or 0
    )
    if n >= total:
        all_null.append(c)

kept = [c for c in cols if c not in all_null]
nn = " AND ".join(f'"{c}" IS NOT NULL' for c in kept)

# Exclusive: incomplete under full No-Null, but complete if we exclude a column set
def complete_excluding(exclude: set[str]) -> int:
    use = [c for c in kept if c not in exclude]
    clause = " AND ".join(f'"{c}" IS NOT NULL' for c in use)
    return int(
        conn.execute(
            f"SELECT COUNT(*) FROM samples WHERE {where} AND ({clause})",
            params,
        ).fetchone()[0]
    )


scenarios = {
    "baseline_no_null": int(
        conn.execute(f"SELECT COUNT(*) FROM samples WHERE {where} AND ({nn})", params).fetchone()[0]
    ),
    "exclude_gamma_flip_pair": complete_excluding({"gamma_flip_spot", "gamma_flip_distance"}),
    "exclude_option_low": complete_excluding({"option_low"}),
    "exclude_gamma_and_option_low": complete_excluding(
        {"gamma_flip_spot", "gamma_flip_distance", "option_low"}
    ),
    "exclude_all_ema_star": complete_excluding(
        {c for c in kept if "_ema" in c or c.startswith("weighted_")}
    ),
    "exclude_gamma_option_low_ema": complete_excluding(
        {c for c in kept if c in {"gamma_flip_spot", "gamma_flip_distance", "option_low"}
         or "_ema" in c or c.startswith("weighted_")}
    ),
}

# gamma_flip null pattern by option_type / moneyness-ish
by_opt = conn.execute(
    f"""
    SELECT option_type,
           COUNT(*) AS n,
           SUM(CASE WHEN gamma_flip_spot IS NULL THEN 1 ELSE 0 END) AS gf_null,
           SUM(CASE WHEN option_low IS NULL THEN 1 ELSE 0 END) AS ol_null,
           SUM(CASE WHEN current_iv IS NULL THEN 1 ELSE 0 END) AS iv_null
    FROM samples WHERE {where}
    GROUP BY option_type
    """,
    params,
).fetchall()

# token-level: is gamma_flip null for whole tokens or intermittent?
tok = conn.execute(
    f"""
    SELECT token,
           COUNT(*) AS n,
           SUM(CASE WHEN gamma_flip_spot IS NULL THEN 1 ELSE 0 END) AS gf_null,
           MIN(CASE WHEN gamma_flip_spot IS NULL THEN timestamp END) AS first_null,
           MAX(CASE WHEN gamma_flip_spot IS NULL THEN timestamp END) AS last_null,
           MIN(CASE WHEN gamma_flip_spot IS NOT NULL THEN timestamp END) AS first_ok,
           MAX(CASE WHEN gamma_flip_spot IS NOT NULL THEN timestamp END) AS last_ok
    FROM samples WHERE {where}
    GROUP BY token
    ORDER BY gf_null DESC
    """,
    params,
).fetchall()

tok_summary = {
    "tokens": len(tok),
    "tokens_always_null_gf": sum(1 for t in tok if t[2] == t[1]),
    "tokens_never_null_gf": sum(1 for t in tok if t[2] == 0),
    "tokens_partial_gf": sum(1 for t in tok if 0 < t[2] < t[1]),
    "top10": [
        {
            "token": t[0],
            "rows": t[1],
            "gf_null": t[2],
            "gf_null_pct": round(100.0 * t[2] / t[1], 2),
        }
        for t in tok[:10]
    ],
}

# When gamma_flip is non-null, what's the time range
gf_ok = conn.execute(
    f"""
    SELECT MIN(timestamp), MAX(timestamp), COUNT(*)
    FROM samples WHERE {where} AND gamma_flip_spot IS NOT NULL
    """,
    params,
).fetchone()
gf_bad = conn.execute(
    f"""
    SELECT MIN(timestamp), MAX(timestamp), COUNT(*)
    FROM samples WHERE {where} AND gamma_flip_spot IS NULL
    """,
    params,
).fetchone()


def ist(ts):
    if ts is None:
        return None
    return datetime.fromtimestamp(float(ts), tz=IST).strftime("%H:%M:%S")


# option_low: correlation with other session OHLC
ol = conn.execute(
    f"""
    SELECT
      SUM(CASE WHEN option_low IS NULL THEN 1 ELSE 0 END),
      SUM(CASE WHEN option_high IS NULL THEN 1 ELSE 0 END),
      SUM(CASE WHEN option_open IS NULL THEN 1 ELSE 0 END),
      SUM(CASE WHEN option_prev_close IS NULL THEN 1 ELSE 0 END),
      SUM(CASE WHEN option_low IS NULL AND option_high IS NULL THEN 1 ELSE 0 END),
      SUM(CASE WHEN option_low IS NULL AND ltp IS NOT NULL THEN 1 ELSE 0 END)
    FROM samples WHERE {where}
    """,
    params,
).fetchone()

# Rows where ONLY gamma_flip_* are null among high-impact cols
# (approximate exclusive killer)
high_impact = [
    "gamma_flip_spot",
    "gamma_flip_distance",
    "option_low",
    "iv_ema300",
    "ltp_ema300",
    "iv_ema200",
    "ltp_ema200",
    "weighted_ltp_ema",
    "iv_rv_spread_10m",
    "current_iv",
]
# Count incomplete rows whose null set among high_impact is subset of gamma_flip pair
sel = ", ".join(f'"{c}"' for c in high_impact)
cur = conn.execute(f"SELECT {sel} FROM samples WHERE {where}", params)
only_gf = 0
gf_and_more = 0
no_gf_but_other = 0
complete_hi = 0
while True:
    batch = cur.fetchmany(8000)
    if not batch:
        break
    for row in batch:
        nulls = {high_impact[i] for i, v in enumerate(row) if v is None}
        if not nulls:
            complete_hi += 1
            continue
        only_flip = nulls <= {"gamma_flip_spot", "gamma_flip_distance"}
        if only_flip:
            only_gf += 1
        elif "gamma_flip_spot" in nulls or "gamma_flip_distance" in nulls:
            gf_and_more += 1
        else:
            no_gf_but_other += 1

# current_iv / greeks null counts
iv_greeks = {}
for c in (
    "current_iv",
    "delta",
    "gamma",
    "vega",
    "theta",
    "vanna",
    "charm",
    "speed",
    "bs_reiv_pred",
    "dgt_reiv_pred",
    "roll_iv",
):
    if c not in cols:
        continue
    iv_greeks[c] = int(
        conn.execute(
            f'SELECT SUM(CASE WHEN "{c}" IS NULL THEN 1 ELSE 0 END) FROM samples WHERE {where}',
            params,
        ).fetchone()[0]
        or 0
    )

# Spot/ltp should be zero
core = {}
for c in ("spot", "ltp", "bid_ask_spread", "futures_ltp", "delta", "gamma"):
    if c in cols:
        core[c] = int(
            conn.execute(
                f'SELECT SUM(CASE WHEN "{c}" IS NULL THEN 1 ELSE 0 END) FROM samples WHERE {where}',
                params,
            ).fetchone()[0]
            or 0
        )

report = {
    "scenarios_complete_rows": scenarios,
    "recovery_vs_baseline": {
        k: scenarios[k] - scenarios["baseline_no_null"] for k in scenarios if k != "baseline_no_null"
    },
    "by_option_type": [
        {
            "option_type": r[0],
            "rows": r[1],
            "gamma_flip_null": r[2],
            "option_low_null": r[3],
            "current_iv_null": r[4],
        }
        for r in by_opt
    ],
    "gamma_flip_token_summary": tok_summary,
    "gamma_flip_time": {
        "ok_count": gf_ok[2],
        "ok_first": ist(gf_ok[0]),
        "ok_last": ist(gf_ok[1]),
        "null_count": gf_bad[2],
        "null_first": ist(gf_bad[0]),
        "null_last": ist(gf_bad[1]),
    },
    "option_low_vs_ohlc": {
        "option_low_null": ol[0],
        "option_high_null": ol[1],
        "option_open_null": ol[2],
        "option_prev_close_null": ol[3],
        "both_low_high_null": ol[4],
        "low_null_but_ltp_present": ol[5],
    },
    "high_impact_partition": {
        "complete_on_high_impact_subset": complete_hi,
        "null_only_gamma_flip_pair": only_gf,
        "gamma_flip_plus_other_high_impact": gf_and_more,
        "other_high_impact_without_gamma_flip": no_gf_but_other,
    },
    "iv_greeks_nulls": iv_greeks,
    "core_nulls": core,
    "all_null_cols": all_null,
}
OUT.write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))

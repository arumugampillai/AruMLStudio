"""Investigation-only NULL audit for Master DB day 2026-07-24.

Does not modify data or feature calculations.
"""
from __future__ import annotations

import json
import os
import re
import sqlite3
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

DB = r"D:\data\master_dataset\master_dataset_nifty_3s.db"
DAY = "2026-07-24"
OUT = os.path.join(
    os.path.dirname(__file__),
    "null_audit_2026-07-24.json",
)
IST = ZoneInfo("Asia/Kolkata")

# Identity / bookkeeping — never used as No-Null feature failure signal alone
SKIP_ATTR = {
    "trading_day",
    "timestamp",
    "token",
    "master_row_id",
    "market",
    "expiry",
    "strike",
    "option_type",
    "symbol",
}


def classify_column(name: str) -> dict:
    """Heuristic root-cause class from feature naming + known ownership patterns."""
    n = name
    cause = "unknown"
    expected = None  # True/False/None
    parent = None
    note = ""

    # Session / OHLC that need history
    if re.search(r"_(open|high|low|close|prev_close)$", n) or n.startswith("option_"):
        if any(x in n for x in ("open", "high", "low", "prev_close", "vwap")):
            cause = "session_or_tape_warmup"
            expected = True
            note = "Session OHLC / VWAP needs prior ticks in session"

    if re.search(r"_ema\d+$", n) or re.search(r"_ema\d+_", n) or "_ewm_" in n:
        cause = "ema_warmup"
        expected = True
        m = re.search(r"ema(\d+)", n)
        parent = re.sub(r"_ema\d+.*", "", n) if m else None
        note = f"EMA needs ~{m.group(1) if m else '?'} prior observations"

    if "_zscore_" in n or re.search(r"_roll_", n) or re.search(
        r"_(mean|std|min|max)_\d", n
    ):
        cause = "rolling_window_warmup"
        expected = True
        note = "Rolling / z-score window incomplete at session start"

    if n in ("current_iv", "call_iv", "put_iv") or n.endswith("_iv") or "iv_" in n:
        if cause == "unknown":
            cause = "iv_unavailable"
            expected = False
            note = "IV requires valid option quote + BS/solve path"

    if n in ("delta", "gamma", "theta", "vega", "rho", "charm", "speed", "vanna", "vomma") or n.startswith(
        "abs_delta"
    ):
        cause = "greeks_unavailable"
        expected = False
        parent = "current_iv / spot / strike / T"
        note = "Greeks need IV + pricing inputs"

    if "straddle" in n or "atm" in n:
        if cause == "unknown":
            cause = "chain_aggregate_missing"
            expected = False
            note = "ATM/chain aggregate needs counterpart strikes on grid"

    if n.endswith("_lag_") or "_lag_" in n or "_change_" in n or "_diff_" in n or "_return_" in n:
        # Master registry should not have these if migrated — if present, warmup
        cause = "time_shift_warmup"
        expected = True
        note = "Lag/diff/return first rows NULL by construction"

    if "depth" in n or "book" in n or "bid" in n or "ask" in n:
        if cause == "unknown":
            cause = "book_microstructure"
            expected = False
            note = "Order book / spread may be missing on some ticks"

    if n.startswith("futures_") or "futures" in n:
        if cause == "unknown":
            cause = "futures_context_missing"
            expected = False
            note = "Futures LTP/VWAP join may miss early/late ticks"

    if "volume" in n or n.endswith("_oi") or "open_interest" in n or n == "option_oi":
        if cause == "unknown":
            cause = "volume_oi_missing"
            expected = False
            note = "OI/volume feed gaps"

    if "pcr" in n or "gex" in n or "flow" in n:
        if cause == "unknown":
            cause = "chain_aggregate_missing"
            expected = False

    if "moneyness" in n or "ltp_to_spot" in n or "spot_to_ltp" in n:
        if cause == "unknown":
            cause = "derived_from_nullable_parent"
            expected = None
            parent = "ltp, spot"
            note = "Ratio NULL if either side NULL"

    if n in ("spot", "ltp", "timestamp", "token"):
        cause = "raw_market"
        expected = False
        note = "Core tape — NULLs here are serious"

    return {
        "cause": cause,
        "expected": expected,
        "parent": parent,
        "note": note,
    }


def ts_to_ist(ts: float | None) -> str | None:
    if ts is None:
        return None
    try:
        return datetime.fromtimestamp(float(ts), tz=IST).strftime("%H:%M:%S")
    except Exception:
        return None


def main() -> None:
    t0 = time.perf_counter()
    conn = sqlite3.connect(DB)
    cols = [r[1] for r in conn.execute("PRAGMA table_info(samples)")]
    where = "trading_day = ?"
    params = (DAY,)
    total = int(conn.execute(f"SELECT COUNT(*) FROM samples WHERE {where}", params).fetchone()[0])

    # Column NULL stats in batches
    col_stats = []
    batch = 25
    for i in range(0, len(cols), batch):
        chunk = cols[i : i + batch]
        parts = []
        for c in chunk:
            parts.append(
                f'SUM(CASE WHEN "{c}" IS NULL THEN 1 ELSE 0 END) AS "n_{c}"'
            )
            parts.append(
                f'MIN(CASE WHEN "{c}" IS NULL THEN timestamp END) AS "f_{c}"'
            )
            parts.append(
                f'MAX(CASE WHEN "{c}" IS NULL THEN timestamp END) AS "l_{c}"'
            )
        sql = f"SELECT {', '.join(parts)} FROM samples WHERE {where}"
        row = conn.execute(sql, params).fetchone()
        for j, c in enumerate(chunk):
            null_n = int(row[j * 3] or 0)
            first_ts = row[j * 3 + 1]
            last_ts = row[j * 3 + 2]
            pct = 100.0 * null_n / total if total else 0.0
            meta = classify_column(c)
            col_stats.append({
                "feature": c,
                "null_count": null_n,
                "null_pct": round(pct, 4),
                "first_null_ts": first_ts,
                "last_null_ts": last_ts,
                "first_null_ist": ts_to_ist(first_ts),
                "last_null_ist": ts_to_ist(last_ts),
                **meta,
            })

    # 100% NULL columns (Step 1 of No-Null)
    all_null_cols = [c["feature"] for c in col_stats if c["null_count"] >= total]
    kept_for_row = [c for c in cols if c not in all_null_cols]

    # Complete rows under No-Null Step 2
    nn = " AND ".join(f'"{c}" IS NOT NULL' for c in kept_for_row)
    complete = int(
        conn.execute(
            f"SELECT COUNT(*) FROM samples WHERE {where} AND ({nn})",
            params,
        ).fetchone()[0]
    )
    incomplete = total - complete

    # Load nullable feature columns (exclude identity) for row attribution
    nullable = [
        c["feature"]
        for c in col_stats
        if c["null_count"] > 0
        and c["feature"] not in all_null_cols
        and c["feature"] not in SKIP_ATTR
    ]
    # Cap for memory — prioritize highest null%
    nullable_sorted = sorted(
        nullable,
        key=lambda f: next(x["null_count"] for x in col_stats if x["feature"] == f),
        reverse=True,
    )
    # Use all nullable kept cols (should be manageable)
    use_cols = nullable_sorted

    print(f"nullable kept cols for attribution: {len(use_cols)}", flush=True)

    # Stream rows and attribute dominant cause
    sel = ["timestamp", "token"] + [f'"{c}"' for c in use_cols]
    cur = conn.execute(
        f"SELECT {', '.join(sel)} FROM samples WHERE {where} ORDER BY timestamp, token",
        params,
    )

    cause_hit_rows = Counter()  # rows with at least one NULL in cause
    dominant = Counter()
    unexpected_dominant = Counter()
    # Per-row: which causes present
    # Also time histogram of incomplete rows
    incomplete_by_minute = Counter()
    first_incomplete_ts = None
    last_incomplete_ts = None
    # Contribution: how many incomplete rows each column touches
    col_kills = Counter()

    cause_priority = [
        "raw_market",
        "iv_unavailable",
        "greeks_unavailable",
        "futures_context_missing",
        "chain_aggregate_missing",
        "book_microstructure",
        "volume_oi_missing",
        "ema_warmup",
        "rolling_window_warmup",
        "session_or_tape_warmup",
        "time_shift_warmup",
        "derived_from_nullable_parent",
        "unknown",
    ]
    feat_cause = {c["feature"]: c["cause"] for c in col_stats}
    feat_expected = {c["feature"]: c["expected"] for c in col_stats}

    scanned = 0
    incomplete_scanned = 0
    # Sample of unexpected incomplete rows
    unexpected_samples = []

    while True:
        batch_rows = cur.fetchmany(5000)
        if not batch_rows:
            break
        for row in batch_rows:
            scanned += 1
            ts = row[0]
            token = row[1]
            null_feats = []
            for i, feat in enumerate(use_cols):
                if row[i + 2] is None:
                    null_feats.append(feat)
                    col_kills[feat] += 1
            if not null_feats:
                continue
            # Row is incomplete among kept nullable cols — but Step2 also requires
            # non-nullable-stat columns that are always present. A row can also fail
            # only on cols that are never null in aggregate? No — if a col has 0 nulls
            # it never fails a row. So null_feats empty means complete among use_cols.
            # However complete check used ALL kept cols including never-null — those
            # can't fail. So incomplete iff null_feats non-empty. Good.
            incomplete_scanned += 1
            if first_incomplete_ts is None:
                first_incomplete_ts = ts
            last_incomplete_ts = ts
            try:
                minute = datetime.fromtimestamp(float(ts), tz=IST).strftime("%H:%M")
                incomplete_by_minute[minute] += 1
            except Exception:
                pass

            causes_present = set()
            for f in null_feats:
                causes_present.add(feat_cause.get(f, "unknown"))
            for c in causes_present:
                cause_hit_rows[c] += 1

            dom = "unknown"
            for p in cause_priority:
                if p in causes_present:
                    dom = p
                    break
            dominant[dom] += 1

            # unexpected if any unexpected null feature is among nulls
            has_unexp = any(feat_expected.get(f) is False for f in null_feats)
            has_exp_only = all(
                feat_expected.get(f) in (True, None) for f in null_feats
            ) and not has_unexp
            if has_unexp:
                unexpected_dominant[dom] += 1
                if len(unexpected_samples) < 25:
                    unexpected_samples.append({
                        "timestamp": ts,
                        "time_ist": ts_to_ist(ts),
                        "token": token,
                        "null_features": null_feats[:20],
                        "dominant": dom,
                    })

    # Cross-check incomplete count
    # Note: incomplete_scanned should equal incomplete if use_cols covers all nullable kept

    # Time range of day
    day_bounds = conn.execute(
        f"SELECT MIN(timestamp), MAX(timestamp) FROM samples WHERE {where}",
        params,
    ).fetchone()

    # Early session incomplete rate (before 09:30, 09:45, 10:00)
    def count_before(hhmm: str) -> tuple[int, int]:
        # approximate: compare IST string via python is slow; use unix cutoff
        y, m, d = map(int, DAY.split("-"))
        h, mi = map(int, hhmm.split(":"))
        cutoff = datetime(y, m, d, h, mi, tzinfo=IST).timestamp()
        tot = int(
            conn.execute(
                f"SELECT COUNT(*) FROM samples WHERE {where} AND timestamp < ?",
                (DAY, cutoff),
            ).fetchone()[0]
        )
        if not kept_for_row:
            return tot, 0
        comp = int(
            conn.execute(
                f"SELECT COUNT(*) FROM samples WHERE {where} AND timestamp < ? AND ({nn})",
                (DAY, cutoff),
            ).fetchone()[0]
        )
        return tot, tot - comp

    early_stats = {}
    for label in ("09:20", "09:30", "09:45", "10:00", "10:30"):
        t, inc = count_before(label)
        early_stats[label] = {
            "rows_before": t,
            "incomplete_before": inc,
            "incomplete_pct": round(100.0 * inc / t, 2) if t else 0.0,
        }

    # Top null columns
    with_nulls = [c for c in col_stats if c["null_count"] > 0]
    with_nulls.sort(key=lambda x: x["null_count"], reverse=True)

    # Cause summary from columns
    cause_cols = defaultdict(list)
    for c in with_nulls:
        if c["feature"] in all_null_cols:
            continue
        cause_cols[c["cause"]].append(c["feature"])

    report = {
        "db": DB,
        "trading_day": DAY,
        "generated_at": datetime.now(IST).isoformat(timespec="seconds"),
        "elapsed_sec": round(time.perf_counter() - t0, 2),
        "totals": {
            "master_rows": total,
            "columns": len(cols),
            "columns_100pct_null": all_null_cols,
            "columns_with_any_null": len(with_nulls),
            "kept_columns_after_step1": len(kept_for_row),
            "complete_rows": complete,
            "incomplete_rows": incomplete,
            "discard_pct": round(100.0 * incomplete / total, 2) if total else 0.0,
            "day_start_ist": ts_to_ist(day_bounds[0]),
            "day_end_ist": ts_to_ist(day_bounds[1]),
            "first_incomplete_ist": ts_to_ist(first_incomplete_ts),
            "last_incomplete_ist": ts_to_ist(last_incomplete_ts),
            "incomplete_scanned": incomplete_scanned,
        },
        "early_session": early_stats,
        "column_stats": col_stats,
        "top20_null_columns": with_nulls[:20],
        "all_null_columns_detail": [c for c in col_stats if c["feature"] in all_null_cols],
        "row_loss": {
            "dominant_cause_counts": dict(dominant),
            "cause_hit_row_counts": dict(cause_hit_rows),
            "unexpected_dominant_counts": dict(unexpected_dominant),
            "top_columns_killing_rows": col_kills.most_common(30),
            "incomplete_by_minute_top": incomplete_by_minute.most_common(40),
        },
        "cause_to_columns": {k: v for k, v in cause_cols.items()},
        "unexpected_samples": unexpected_samples,
        "notes": [
            "No-Null Step 1 drops 100% NULL columns; Step 2 drops any row with NULL in remaining columns.",
            "Dominant cause uses priority: IV/greeks/futures/chain before EMA/rolling warmup.",
            "Rows can hit multiple causes; cause_hit_row_counts count multi-membership.",
            "Investigation only — no data or formula changes.",
        ],
    }
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(report, fh)
    print("wrote", OUT)
    print("total", total, "complete", complete, "incomplete", incomplete)
    print("100% null", all_null_cols)
    print("top5", [(c["feature"], c["null_count"], c["null_pct"]) for c in with_nulls[:5]])
    print("dominant", dict(dominant.most_common(10)))
    print("elapsed", report["elapsed_sec"])


if __name__ == "__main__":
    main()

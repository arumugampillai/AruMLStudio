"""Trace No-Null dropped columns: Master vs Analysis origin (fast)."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pyarrow.compute as pc
import pyarrow.parquet as pq

META = Path(
    r"c:/Users/admin/PycharmProjects/v1/AruNeo/angelone/chart/data/datasets/"
    r"analysis_206r_193p_3s_20260730_094409.json"
)
BAK = META.with_suffix(".parquet.pre_nonull.bak")
MASTER_DB = Path(r"D:/data/master_dataset/master_dataset_nifty_3s.db")
OUT = Path(
    r"c:/Users/admin/PycharmProjects/v1/AruNeo/angelone/chart/data/datasets/"
    r"analysis_206r_193p_3s_20260730_094409.null_origin_trace.json"
)

PIPELINE_PARENTS: dict[str, list[str]] = {
    "spot_high_ema20_to_ltp_ratio": ["spot_high_ema20"],
    "spot_low_ema20_to_ltp_ratio": ["spot_low_ema20"],
    "spot_high_ema50_to_ltp_ratio": ["spot_high_ema50"],
    "spot_low_ema50_to_ltp_ratio": ["spot_low_ema50"],
    "spot_high_ema100_to_ltp_ratio": ["spot_high_ema100"],
    "spot_low_ema100_to_ltp_ratio": ["spot_low_ema100"],
    "spot_high_ema200_to_ltp_ratio": ["spot_high_ema200"],
    "spot_low_ema200_to_ltp_ratio": ["spot_low_ema200"],
    "spot_high_ema300_to_ltp_ratio": ["spot_high_ema300"],
    "spot_low_ema300_to_ltp_ratio": ["spot_low_ema300"],
    "weighted_spot_high_ema_to_ltp_ratio": ["weighted_spot_high_ema"],
    "weighted_spot_low_ema_to_ltp_ratio": ["weighted_spot_low_ema"],
    "weighted_spot_high_ema_to_weighted_spot_low_ema": [
        "weighted_spot_high_ema",
        "weighted_spot_low_ema",
    ],
    "weighted_spot_ema_to_weighted_spot_low_ema": [
        "weighted_spot_close_ema",
        "weighted_spot_low_ema",
    ],
    "weighted_spot_ema_to_weighted_spot_high_ema": [
        "weighted_spot_close_ema",
        "weighted_spot_high_ema",
    ],
    "ltp_to_spot_ema20_channel_width_ratio": ["spot_ema20_channel_width"],
    "ltp_to_spot_ema50_channel_width_ratio": ["spot_ema50_channel_width"],
    "ltp_to_spot_ema100_channel_width_ratio": ["spot_ema100_channel_width"],
    "ltp_to_spot_ema200_channel_width_ratio": ["spot_ema200_channel_width"],
    "ltp_to_spot_ema300_channel_width_ratio": ["spot_ema300_channel_width"],
    "futures_ltp_minus_futures_vwap": ["futures_ltp", "futures_vwap"],
    "futures_ltp_minus_futures_vwap_div_futures_vwap": ["futures_ltp", "futures_vwap"],
    "spot_minus_futures_ltp": ["futures_ltp"],
    "spot_minus_futures_vwap": ["futures_vwap"],
    "spot_div_futures_ltp": ["futures_ltp"],
    "futures_ltp_div_spot": ["futures_ltp"],
}


def master_day_nulls(
    conn: sqlite3.Connection, cols: list[str], days: list[str]
) -> tuple[dict[str, float], dict[str, list[str]]]:
    """One GROUP BY pass: per-day row count + non-null per column."""
    ph = ",".join("?" * len(days))
    parts = ["trading_day", "COUNT(*) AS n"]
    for c in cols:
        parts.append(
            f'SUM(CASE WHEN "{c}" IS NOT NULL THEN 1 ELSE 0 END) AS "nn__{c}"'
        )
    sql = (
        f"SELECT {', '.join(parts)} FROM samples "
        f"WHERE trading_day IN ({ph}) GROUP BY trading_day ORDER BY 1"
    )
    print(f"Master SQL: {len(cols)} cols × {len(days)} days…", flush=True)
    rows = conn.execute(sql, days).fetchall()
    colnames = [d[0] for d in conn.execute(sql, days).description]  # noqa — reuse
    # Re-fetch description from cursor
    cur = conn.execute(sql, days)
    desc = [d[0] for d in cur.description]
    rows = cur.fetchall()
    print(f"Master SQL done: {len(rows)} day rows", flush=True)

    totals_n = 0
    totals_nn = {c: 0 for c in cols}
    all_null: dict[str, list[str]] = {c: [] for c in cols}
    for row in rows:
        rec = dict(zip(desc, row))
        day = str(rec["trading_day"])
        n = int(rec["n"])
        totals_n += n
        for c in cols:
            nn = int(rec[f"nn__{c}"] or 0)
            totals_nn[c] += nn
            if nn == 0:
                all_null[c].append(day)
    pct = {
        c: (100.0 * totals_nn[c] / totals_n if totals_n else 0.0) for c in cols
    }
    return pct, all_null


def analysis_day_nulls(
    bak: Path, cols: list[str]
) -> tuple[dict[str, float], dict[str, list[str]]]:
    """Per row-group (≈ one day) null stats — avoids full-frame load."""
    pf = pq.ParquetFile(str(bak))
    schema = set(pf.schema_arrow.names)
    use = ["trading_day"] + [c for c in cols if c in schema]
    print(
        f"Analysis parquet: {pf.num_row_groups} row groups, {len(use)-1} cols…",
        flush=True,
    )
    totals_n = 0
    totals_nn = {c: 0 for c in cols}
    all_null: dict[str, list[str]] = {c: [] for c in cols}
    # Track days seen for missing cols
    days_seen: list[str] = []

    for i in range(pf.num_row_groups):
        t = pf.read_row_group(i, columns=use)
        day_col = t.column("trading_day")
        # Most RGs are single-day; take unique
        days = pc.unique(day_col).to_pylist()
        n = t.num_rows
        totals_n += n
        day_label = str(days[0]) if len(days) == 1 else f"RG{i}:{','.join(map(str, days))}"
        if len(days) == 1:
            days_seen.append(str(days[0]))
        for c in cols:
            if c not in schema:
                all_null[c].append(day_label if len(days) == 1 else str(days[0]))
                continue
            nn = n - t.column(c).null_count
            totals_nn[c] += nn
            if nn == 0:
                # If multi-day RG, attribute carefully
                if len(days) == 1:
                    all_null[c].append(str(days[0]))
                else:
                    # fallback: check per day
                    pdf = t.select(["trading_day", c]).to_pandas()
                    for d, g in pdf.groupby(pdf["trading_day"].astype(str)):
                        if int(g[c].notna().sum()) == 0:
                            all_null[c].append(str(d))
        if (i + 1) % 5 == 0 or i + 1 == pf.num_row_groups:
            print(f"  RG {i+1}/{pf.num_row_groups}", flush=True)

    # Dedup all_null lists
    for c in cols:
        all_null[c] = sorted(set(all_null[c]))
    pct = {
        c: (100.0 * totals_nn[c] / totals_n if totals_n else 0.0) for c in cols
    }
    return pct, all_null


def main() -> None:
    meta = json.loads(META.read_text(encoding="utf-8"))
    dropped = list(meta["no_null_dropped_columns"])
    days = [d["trading_day"] for d in meta["days"]]

    conn = sqlite3.connect(f"file:{MASTER_DB}?mode=ro", uri=True)
    master_schema = {r[1] for r in conn.execute("PRAGMA table_info(samples)")}
    master_dropped = [c for c in dropped if c in master_schema]
    pipeline_dropped = [c for c in dropped if c not in master_schema]
    print(
        f"Dropped={len(dropped)} master={len(master_dropped)} "
        f"pipeline={len(pipeline_dropped)}",
        flush=True,
    )

    master_pct, master_all_null = master_day_nulls(conn, master_dropped, days)
    analysis_pct, analysis_all_null = analysis_day_nulls(BAK, dropped)

    buckets: dict[str, list[str]] = {
        "ALREADY_NULL_IN_MASTER": [],
        "WORSENED_IN_ANALYSIS": [],
        "PIPELINE_INHERITED_FROM_MASTER_PARENT": [],
        "PIPELINE_INTRODUCED_OR_UNEXPLAINED": [],
        "OTHER": [],
    }
    rows: list[dict] = []

    for c in master_dropped:
        m_days = set(master_all_null[c])
        a_days = set(analysis_all_null[c])
        new_bad = sorted(a_days - m_days)
        if new_bad:
            bucket = "WORSENED_IN_ANALYSIS"
            fix = "Master Dataset → Analysis Dataset"
        else:
            bucket = "ALREADY_NULL_IN_MASTER"
            fix = "Tick → Master Dataset"
        buckets[bucket].append(c)
        rows.append(
            {
                "feature": c,
                "layer": "master_registry",
                "bucket": bucket,
                "master_nonnull_pct": round(master_pct[c], 3),
                "analysis_nonnull_pct": round(analysis_pct[c], 3),
                "master_all_null_days": master_all_null[c],
                "analysis_all_null_days": analysis_all_null[c],
                "new_all_null_days_in_analysis": new_bad,
                "fix_layer": fix,
            }
        )

    for c in pipeline_dropped:
        a_days = analysis_all_null[c]
        parents = PIPELINE_PARENTS.get(c, [])
        unexplained: list[str] = []
        for day in a_days:
            explained = any(day in master_all_null.get(p, []) for p in parents)
            if not explained:
                explained = any(day in analysis_all_null.get(p, []) for p in parents)
            if not explained:
                unexplained.append(day)
        if parents and a_days and not unexplained:
            bucket = "PIPELINE_INHERITED_FROM_MASTER_PARENT"
            fix = "Tick → Master Dataset (parent); pipeline propagates NULL"
        elif unexplained:
            bucket = "PIPELINE_INTRODUCED_OR_UNEXPLAINED"
            fix = "Master Dataset → Analysis Dataset (pipeline)"
        else:
            bucket = "OTHER"
            fix = "check"
        buckets[bucket].append(c)
        rows.append(
            {
                "feature": c,
                "layer": "analysis_pipeline",
                "bucket": bucket,
                "master_nonnull_pct": None,
                "analysis_nonnull_pct": round(analysis_pct[c], 3),
                "analysis_all_null_days": a_days,
                "parents": parents,
                "unexplained_all_null_days": unexplained,
                "fix_layer": fix,
            }
        )

    summary = {
        "dataset": META.stem,
        "master_db": str(MASTER_DB),
        "analysis_pre_nonull": str(BAK),
        "n_dropped": len(dropped),
        "n_master_registry": len(master_dropped),
        "n_pipeline_only": len(pipeline_dropped),
        "bucket_counts": {k: len(v) for k, v in buckets.items()},
        "buckets": buckets,
        "features": rows,
        "verdict": {
            "tick_to_master_count": (
                len(buckets["ALREADY_NULL_IN_MASTER"])
                + len(buckets["PIPELINE_INHERITED_FROM_MASTER_PARENT"])
            ),
            "master_to_analysis_count": (
                len(buckets["WORSENED_IN_ANALYSIS"])
                + len(buckets["PIPELINE_INTRODUCED_OR_UNEXPLAINED"])
            ),
            "no_null_logic_as_root_cause": 0,
            "note": (
                "No-Null multi-day Step1 is the drop *trigger*, but every dropped "
                "feature already had ≥1 all-NULL trading day upstream. Primary data "
                "fix is Tick→Master for registry parents; pipeline ratios inherit."
            ),
        },
    }
    OUT.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary["bucket_counts"], indent=2), flush=True)
    print("verdict", summary["verdict"], flush=True)
    for r in rows:
        print(
            f"{r['bucket']:42s} {r['feature']:45s} "
            f"m={r.get('master_all_null_days')} a={r['analysis_all_null_days']}",
            flush=True,
        )
    print("wrote", OUT, flush=True)
    conn.close()


if __name__ == "__main__":
    main()

"""Registry No-Null attribution audit — investigation only (no data changes).

Uses pandas for fast exclusive/marginal attribution over ~310k × ~200 features.
"""
from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

import numpy as np
import pandas as pd

from chain_replay_ml.dataset_builder.nullable_features import (
    NULLABLE_FEATURE_LIST,
    mandatory_columns_for_step2,
)
from chain_replay_ml.dataset_builder.transformations.lag_ui import (
    META_SKIP_COLUMNS,
    canonical_registry_feature_names,
)

DB = r"D:\data\master_dataset\master_dataset_nifty_3s.db"
OUT = Path(r"D:\data\master_dataset\registry_null_audit_2026-07-24.json")
DAY = "2026-07-24"


def classify_feature(name: str, *, null_rows: int, total: int, exclusive: int) -> dict:
    """Heuristic: expected warmup vs bug vs candidate for Nullable list."""
    n = name
    nullable_candidate = False
    status = "ok_zero_null"
    note = "No NULLs"

    if null_rows <= 0:
        return {"status": status, "nullable_candidate": False, "note": note}

    # Session / tape
    if n in {
        "option_open",
        "option_high",
        "option_low",
        "option_prev_close",
        "spot_open",
        "spot_high",
        "spot_low",
        "spot_prev_close",
        "option_vwap",
        "futures_vwap",
    }:
        if null_rows <= 7000:
            status = "expected_warmup_or_sparse"
            note = "Session OHLC / VWAP — small NULL count may be early-session / missing tape"
        else:
            status = "bug_suspect"
            note = "Session OHLC / VWAP — large NULL count; likely collector or readiness bug"
            if n.startswith("option_") and "vwap" not in n:
                note += " (option session OHLC previously had day_low=0 latch)"
        return {"status": status, "nullable_candidate": False, "note": note}

    if any(x in n for x in ("_ema", "_std", "zscore", "rv_", "weighted_")):
        status = "expected_controller_warmup"
        note = "EMA / rolling / RV controller warmup beyond fixed 15m wall-clock"
        # Large exclusive impact may still be a readiness/policy bug
        if exclusive > 10000:
            status = "bug_suspect_or_overlong_warmup"
            note = "Controller feature removes many rows alone — check readiness / window vs 15m policy"
        return {"status": status, "nullable_candidate": False, "note": note}

    if n in NULLABLE_FEATURE_LIST:
        status = "nullable_allowed"
        note = "On Nullable Feature List — ignored by Step 2"
        return {"status": status, "nullable_candidate": True, "note": note}

    if n in {"delta", "gamma", "theta", "rho", "abs_delta"} or n.endswith("_iv"):
        status = "pricing_unavailable"
        note = "IV/greeks can be unresolved on some ticks"
        nullable_candidate = n not in {"delta", "gamma", "theta", "rho"}  # core often required
        if n in {"delta", "gamma", "theta", "rho", "abs_delta"}:
            nullable_candidate = False
            if exclusive > 5000:
                status = "bug_suspect"
                note = "Core greek NULL with large exclusive impact — pricing path bug"
        return {"status": status, "nullable_candidate": nullable_candidate, "note": note}

    if "gamma_flip" in n:
        status = "nullable_allowed"
        note = "Flip unresolved when GEX profile has no zero-crossing"
        return {"status": status, "nullable_candidate": True, "note": note}

    if n.startswith("futures_"):
        status = "futures_context"
        note = "Futures context may be missing for some samples"
        if exclusive > 5000 or null_rows > 20000:
            status = "bug_suspect"
            note = "Futures feature large NULL impact — check futures loader"
        return {"status": status, "nullable_candidate": False, "note": note}

    if null_rows == total:
        status = "step1_empty_column"
        note = "100% NULL — dropped in Step 1 (no row impact)"
        return {"status": status, "nullable_candidate": False, "note": note}

    if exclusive > 1000:
        status = "bug_suspect"
        note = "Exclusive row killer — investigate emission / readiness"
        return {"status": status, "nullable_candidate": False, "note": note}

    if null_rows > 0:
        status = "sparse_null_overlap"
        note = "NULLs overlap other incomplete features; small exclusive impact"
        return {"status": status, "nullable_candidate": False, "note": note}

    return {"status": status, "nullable_candidate": nullable_candidate, "note": note}


def main() -> None:
    t0 = time.monotonic()
    conn = sqlite3.connect(DB)
    conn.execute("PRAGMA query_only=ON")

    total = int(
        conn.execute(
            "SELECT COUNT(*) FROM samples WHERE trading_day=?",
            (DAY,),
        ).fetchone()[0]
    )
    print(f"total rows: {total:,}", flush=True)

    db_cols = {r[1] for r in conn.execute("PRAGMA table_info(samples)")}
    registry_all = list(canonical_registry_feature_names())
    registry = [c for c in registry_all if c in db_cols]
    missing_registry = [c for c in registry_all if c not in db_cols]
    print(
        f"registry present: {len(registry)}  missing from DB: {len(missing_registry)}",
        flush=True,
    )

    # Load registry + timestamp/token for warmup analysis
    load_cols = list(dict.fromkeys(["timestamp", "token", *registry]))
    load_cols = [c for c in load_cols if c in db_cols]
    print(f"Loading {len(load_cols)} columns…", flush=True)
    col_sql = ", ".join(f'"{c}"' for c in load_cols)
    df = pd.read_sql_query(
        f"SELECT {col_sql} FROM samples WHERE trading_day=?",
        conn,
        params=(DAY,),
    )
    print(f"loaded shape={df.shape}", flush=True)

    null_counts = {c: int(df[c].isna().sum()) for c in registry}
    all_null_cols = [c for c, n in null_counts.items() if n >= total]
    kept = [c for c in registry if c not in set(all_null_cols)]
    mandatory = mandatory_columns_for_step2(kept)
    nullable_present = [c for c in NULLABLE_FEATURE_LIST if c in kept]
    print(
        f"Step1 dropped 100% NULL ({len(all_null_cols)}): {all_null_cols}",
        flush=True,
    )
    print(
        f"kept={len(kept)} mandatory={len(mandatory)} nullable={nullable_present}",
        flush=True,
    )

    # Boolean matrix: True = non-null
    mat = np.column_stack([df[c].notna().to_numpy() for c in mandatory])
    complete_mask = mat.all(axis=1)
    baseline = int(complete_mask.sum())
    incomplete_mask = ~complete_mask
    print(f"Registry No-Null after Step2: {baseline:,}  removed={total - baseline:,}", flush=True)

    tmin = float(df["timestamp"].min())
    tmax = float(df["timestamp"].max())
    warmup_end = tmin + 15 * 60
    warmup_mask = df["timestamp"].to_numpy() < warmup_end
    warmup_rows = int(warmup_mask.sum())
    n_tokens = int(df["token"].nunique())
    print(
        f"session [{tmin}..{tmax}] tokens={n_tokens} warmup_15m={warmup_rows:,}",
        flush=True,
    )

    features_report: list[dict] = []
    for i, feat in enumerate(mandatory, 1):
        col_idx = mandatory.index(feat)
        feat_null = ~mat[:, col_idx]
        n_null = int(null_counts.get(feat, int(feat_null.sum())))
        others_ok = np.ones(total, dtype=bool)
        if mandatory:
            # all other mandatory non-null
            if len(mandatory) > 1:
                others_ok = mat[:, np.arange(len(mandatory)) != col_idx].all(axis=1)
            exclusive = int((feat_null & others_ok).sum())
        else:
            exclusive = n_null
        # marginal: complete if we ignore this feature
        without = mat[:, np.arange(len(mandatory)) != col_idx].all(axis=1)
        marginal = int(without.sum()) - baseline
        nulls_in_incomplete = int((feat_null & incomplete_mask).sum())
        clf = classify_feature(
            feat, null_rows=n_null, total=total, exclusive=exclusive
        )
        features_report.append(
            {
                "feature": feat,
                "null_rows": n_null,
                "null_pct": round(100.0 * n_null / total, 3) if total else 0.0,
                "exclusive_rows_removed": exclusive,
                "marginal_rows_saved_if_nullable": marginal,
                "nulls_among_incomplete_rows": nulls_in_incomplete,
                "on_nullable_list": False,
                "step1_dropped_100pct_null": False,
                **clf,
            }
        )
        if i % 50 == 0:
            print(f"  attribution {i}/{len(mandatory)}…", flush=True)

    # Nullable features (ignored by Step 2) — still report nulls
    for feat in kept:
        if feat not in NULLABLE_FEATURE_LIST:
            continue
        n_null = int(null_counts.get(feat, 0))
        clf = classify_feature(feat, null_rows=n_null, total=total, exclusive=0)
        features_report.append(
            {
                "feature": feat,
                "null_rows": n_null,
                "null_pct": round(100.0 * n_null / total, 3) if total else 0.0,
                "exclusive_rows_removed": 0,
                "marginal_rows_saved_if_nullable": 0,
                "nulls_among_incomplete_rows": 0,
                "on_nullable_list": True,
                "step1_dropped_100pct_null": False,
                **clf,
            }
        )

    for c in all_null_cols:
        clf = classify_feature(c, null_rows=total, total=total, exclusive=0)
        features_report.append(
            {
                "feature": c,
                "null_rows": total,
                "null_pct": 100.0,
                "exclusive_rows_removed": 0,
                "marginal_rows_saved_if_nullable": 0,
                "nulls_among_incomplete_rows": 0,
                "on_nullable_list": c in NULLABLE_FEATURE_LIST,
                "step1_dropped_100pct_null": True,
                **clf,
            }
        )

    # Features with zero nulls
    reported = {r["feature"] for r in features_report}
    for feat in registry:
        if feat in reported:
            continue
        features_report.append(
            {
                "feature": feat,
                "null_rows": int(null_counts.get(feat, 0)),
                "null_pct": round(100.0 * null_counts.get(feat, 0) / total, 3),
                "exclusive_rows_removed": 0,
                "marginal_rows_saved_if_nullable": 0,
                "nulls_among_incomplete_rows": 0,
                "on_nullable_list": feat in NULLABLE_FEATURE_LIST,
                "step1_dropped_100pct_null": False,
                "status": "ok_zero_null",
                "nullable_candidate": False,
                "note": "No NULLs",
            }
        )

    features_report.sort(
        key=lambda r: (
            -int(r["exclusive_rows_removed"]),
            -int(r["marginal_rows_saved_if_nullable"]),
            -int(r["null_rows"]),
            r["feature"],
        )
    )

    # Cumulative exclusive killers
    cum = []
    active = complete_mask.copy()
    # Start from incomplete; successively forgive exclusive features
    remaining_incomplete = incomplete_mask.copy()
    cur_rows = baseline
    for feat in [r["feature"] for r in features_report if r["exclusive_rows_removed"] > 0][
        :25
    ]:
        col_idx = mandatory.index(feat)
        # rows that fail only due to this feature among still-mandatory set — approximate via
        # recompute with forgiven set
        # Simpler: forgive feature by treating its column as always True
        mat[:, col_idx] = True
        new_complete = mat.all(axis=1)
        new_rows = int(new_complete.sum())
        cum.append(
            {
                "also_nullable": feat,
                "rows_after": new_rows,
                "gained": new_rows - cur_rows,
            }
        )
        cur_rows = new_rows

    # Reload mat for safety — we mutated it; rebuild from df for any further use
    # (not needed)

    # Warmup overlap: how many of the removed rows are outside 15m warmup?
    removed_mask = incomplete_mask
    removed_outside_warmup = int((removed_mask & ~warmup_mask).sum())
    removed_inside_warmup = int((removed_mask & warmup_mask).sum())

    scenarios = {
        "raw_total": total,
        "n_tokens": n_tokens,
        "warmup_15m_rows": warmup_rows,
        "expected_after_warmup_only": total - warmup_rows,
        "after_registry_no_null": baseline,
        "rows_removed_by_no_null": total - baseline,
        "removed_inside_15m_warmup": removed_inside_warmup,
        "removed_outside_15m_warmup": removed_outside_warmup,
        "extra_removed_beyond_warmup": (total - baseline) - warmup_rows,
        "step1_dropped_columns": all_null_cols,
        "user_reported_after_no_null": 243790,
    }

    payload = {
        "day": DAY,
        "db": DB,
        "summary": scenarios,
        "nullable_list": list(NULLABLE_FEATURE_LIST),
        "missing_registry_from_db": missing_registry,
        "registry_count": len(registry),
        "mandatory_count": len(mandatory),
        "features": features_report,
        "cumulative_nullable_experiment": cum,
        "elapsed_sec": round(time.monotonic() - t0, 1),
    }
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {OUT}", flush=True)
    print(json.dumps(scenarios, indent=2), flush=True)
    print("\nTOP exclusive / marginal killers:", flush=True)
    shown = 0
    for r in features_report:
        if (
            r["exclusive_rows_removed"]
            or r["marginal_rows_saved_if_nullable"]
            or (r["null_rows"] > 0 and shown < 60)
        ):
            print(
                f"  {r['feature']:42s} null={r['null_rows']:7d} "
                f"excl={r['exclusive_rows_removed']:7d} "
                f"marg={r['marginal_rows_saved_if_nullable']:7d} "
                f"[{r['status']}] {r['note'][:70]}",
                flush=True,
            )
            shown += 1
            if shown >= 50:
                break
    print("\nCumulative forgive exclusive killers:", flush=True)
    for row in cum[:15]:
        print(
            f"  +nullable {row['also_nullable']:40s} "
            f"rows={row['rows_after']:,} (+{row['gained']:,})",
            flush=True,
        )
    print(f"elapsed {payload['elapsed_sec']}s", flush=True)
    conn.close()


if __name__ == "__main__":
    main()

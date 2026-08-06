"""Recover orphan analysis parquet: day-by-day No-Null + registry metadata."""

from __future__ import annotations

import gc
import json
import os
import sys
from datetime import datetime, timezone

import pyarrow as pa
import pyarrow.parquet as pq

DATA_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "data")
)
NAME = "analysis_206r_193p_3s_20260730_094409"
PQ = os.path.join(DATA_DIR, "datasets", f"{NAME}.parquet")
OUT_TMP = PQ + ".nonull.tmp"
JSON_PATH = os.path.join(DATA_DIR, "datasets", f"{NAME}.json")
EXPECTED_PATH = os.path.join(DATA_DIR, "datasets", f"{NAME}.expected.json")

CHART_DIR = os.path.abspath(os.path.join(DATA_DIR, ".."))
if CHART_DIR not in sys.path:
    sys.path.insert(0, CHART_DIR)

from chain_replay_ml.dataset_builder.analysis_lab_store import register_dataset
from chain_replay_ml.dataset_builder.auditor import list_datasets
from chain_replay_ml.dataset_builder.non_null_filter import apply_non_null_filter_frame
from chain_replay_ml.dataset_builder.pipeline_features_config import (
    build_pipeline_features_transformation_config,
)
from chain_replay_ml.dataset_builder.schema_registry import load_feature_registry

IDENTITY = {
    "trading_day",
    "timestamp",
    "token",
    "strike",
    "option_type",
    "spot",
    "ltp",
    "symbol",
    "market",
    "expiry",
}


def main() -> None:
    print("source", PQ, "size_gb", round(os.path.getsize(PQ) / 1e9, 2))
    pf = pq.ParquetFile(PQ)
    schema_names = [str(n) for n in pf.schema_arrow.names]
    print("rows", pf.metadata.num_rows, "cols", len(schema_names))

    days_table = pq.read_table(PQ, columns=["trading_day"])
    days = sorted({str(x) for x in days_table.column(0).to_pylist() if x is not None})
    del days_table
    print("days", len(days), days[0], "...", days[-1])

    reg = load_feature_registry()
    registry_feats: set[str] = set()
    cols = reg.get("columns") or {}
    if isinstance(cols, dict):
        registry_feats = {str(k) for k in cols.keys()}
    groups = reg.get("groups") or {}
    if isinstance(groups, dict):
        for g in groups.values():
            if isinstance(g, dict):
                for f in g.get("features") or []:
                    registry_feats.add(str(f))
            elif isinstance(g, list):
                registry_feats.update(str(x) for x in g)

    pipeline_cols = [
        c
        for c in schema_names
        if c not in registry_feats
        and c not in IDENTITY
        and not str(c).startswith("future_ltp_")
        and not str(c).startswith("label_")
    ]
    print("pipeline_step2_candidates", len(pipeline_cols))

    print("Pass1: empty-column discovery…")
    empty_on_any_day: set[str] = set()
    for i, day in enumerate(days, 1):
        t = pq.read_table(PQ, filters=[("trading_day", "=", day)])
        n = t.num_rows
        for name in schema_names:
            if t.column(name).null_count >= n:
                empty_on_any_day.add(name)
        print(
            f"  day {i}/{len(days)} {day}: rows={n:,} empty_so_far={len(empty_on_any_day)}"
        )
        del t
        gc.collect()

    kept = [c for c in schema_names if c not in empty_on_any_day]
    dropped = [c for c in schema_names if c in empty_on_any_day]
    print("Step1 kept", len(kept), "dropped", len(dropped))
    if dropped[:20]:
        print("  dropped sample", dropped[:20])

    try:
        xform_cfg = build_pipeline_features_transformation_config()
    except TypeError:
        xform_cfg = build_pipeline_features_transformation_config(None)

    step2_scope = [c for c in pipeline_cols if c in set(kept)]
    print("step2_scope", len(step2_scope))

    print("Pass2: filter + write…")
    if os.path.isfile(OUT_TMP):
        os.remove(OUT_TMP)

    writer = None
    rows_before = 0
    rows_after = 0
    rows_by_day_before: dict[str, int] = {}
    rows_by_day_after: dict[str, int] = {}
    final_cols: list[str] | None = None

    for i, day in enumerate(days, 1):
        df = pq.read_table(PQ, filters=[("trading_day", "=", day)]).to_pandas()
        rows_before += len(df)
        rows_by_day_before[day] = len(df)
        nn = apply_non_null_filter_frame(
            df,
            step2_columns=step2_scope,
            transformation_config=xform_cfg,
        )
        out = nn["frame"]
        keep_here = [c for c in kept if c in out.columns]
        out = out.loc[:, keep_here].copy()
        rows_after += len(out)
        rows_by_day_after[day] = len(out)
        print(f"  day {i}/{len(days)} {day}: {len(df):,} -> {len(out):,}")
        if out.empty:
            del df, out, nn
            gc.collect()
            continue
        table = pa.Table.from_pandas(out, preserve_index=False)
        if final_cols is None:
            final_cols = list(table.schema.names)
            writer = pq.ParquetWriter(OUT_TMP, table.schema, compression="zstd")
        else:
            missing = [c for c in final_cols if c not in table.schema.names]
            for c in missing:
                table = table.append_column(c, pa.nulls(table.num_rows))
            table = table.select(final_cols)
        writer.write_table(table)
        del df, out, nn, table
        gc.collect()

    if writer is not None:
        writer.close()

    if not os.path.isfile(OUT_TMP):
        raise SystemExit("No output written — all rows dropped?")

    bak = PQ + ".pre_nonull.bak"
    if os.path.isfile(bak):
        os.remove(bak)
    os.replace(PQ, bak)
    os.replace(OUT_TMP, PQ)
    print("wrote", PQ, "size_gb", round(os.path.getsize(PQ) / 1e9, 2))
    print("rows", rows_before, "->", rows_after, "dropped_rows", rows_before - rows_after)

    pf2 = pq.ParquetFile(PQ)
    col_names = [str(n) for n in pf2.schema_arrow.names]
    feature_cols = [
        c
        for c in col_names
        if c not in IDENTITY
        and not str(c).startswith("future_ltp_")
        and not str(c).startswith("label_")
    ]
    target_cols = [
        c
        for c in col_names
        if str(c).startswith("future_ltp_") or str(c).startswith("label_")
    ]

    days_meta = []
    sources = []
    for day in days:
        days_meta.append(
            {
                "trading_day": day,
                "market": "NIFTY",
                "expiry": "",
                "source_id": f"NIFTY_{day}",
            }
        )
        sources.append(
            {
                "trading_day": day,
                "market": "NIFTY",
                "expiry": "",
                "status": "loaded",
                "rows": int(rows_by_day_after.get(day, 0)),
            }
        )

    meta = {
        "dataset_name": NAME,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "version": 2,
        "builder_version": "1.4.2-recovery",
        "export_source": "analysis_orphan_nonull_recovery",
        "dataset_kind": "analysis",
        "keep_pipeline_owned": True,
        "storage_backend": "parquet",
        "market": "NIFTY",
        "days": days_meta,
        "trading_days": len(days),
        "sources": sources,
        "sample_interval_sec": 3,
        "sampling": {"interval_sec": 3, "method": "fixed_interval"},
        "feature_profile": "custom",
        "feature_columns": feature_cols,
        "enabled_features": feature_cols,
        "transformed_feature_columns": [c for c in pipeline_cols if c in set(col_names)],
        "feature_count": len(feature_cols),
        "prediction_target_columns": target_cols,
        "target_count": len(target_cols),
        "row_count": int(rows_after),
        "column_count": len(col_names),
        "export_stats": {
            "rows_exported": int(rows_after),
            "rows_per_day": {k: int(v) for k, v in rows_by_day_after.items()},
            "rows_before_nonull": int(rows_before),
        },
        "master_filter": {
            "all_days": True,
            "no_null_data": True,
            "premium_enabled": False,
        },
        "no_null_dropped_columns": dropped,
        "no_null_report": {
            "rows_before": int(rows_before),
            "rows_after": int(rows_after),
            "empty_columns_removed": len(dropped),
            "columns_removed": dropped,
            "incomplete_rows_removed": int(rows_before - rows_after),
            "rows_by_day_before": {k: int(v) for k, v in rows_by_day_before.items()},
            "rows_by_day_after": {k: int(v) for k, v in rows_by_day_after.items()},
            "stage": "pipeline_post_transformation_recovery",
            "step2_scope": "pipeline_only",
            "ok": True,
        },
        "recovery_note": (
            "Recovered after python crash during full-frame No-Null load. "
            "No-Null applied day-by-day. Pre-filter parquet saved as .pre_nonull.bak."
        ),
    }

    with open(JSON_PATH, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
        fh.write("\n")
    print("wrote", JSON_PATH)

    expected = {
        "dataset_name": NAME,
        "market": "NIFTY",
        "sample_interval_sec": 3,
        "trading_days": days,
        "feature_count": len(feature_cols),
        "row_count": int(rows_after),
    }
    with open(EXPECTED_PATH, "w", encoding="utf-8") as fh:
        json.dump(expected, fh, indent=2)
        fh.write("\n")
    print("wrote", EXPECTED_PATH)

    reg_row = register_dataset(
        DATA_DIR,
        PQ,
        name=NAME,
        relative_path=f"datasets/{NAME}.parquet",
    )
    print(
        "analysis.db registered",
        reg_row.get("dataset_id"),
        "rows",
        reg_row.get("rows"),
        "features",
        reg_row.get("features"),
    )

    rows = [r for r in list_datasets(DATA_DIR) if r.get("dataset_name") == NAME]
    print("registry list", rows[0] if rows else None)
    print("DONE")


if __name__ == "__main__":
    main()

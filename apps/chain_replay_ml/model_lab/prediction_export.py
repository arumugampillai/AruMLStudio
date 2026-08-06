"""Export Model Lab prediction dataset to CSV / Parquet / SQLite."""

from __future__ import annotations

import os
import sqlite3
from typing import Any


def _load_frame(db_path: str, *, data_dir: str | None = None):
    import pandas as pd

    from .prediction_feature_store import PredictionFeatureStore
    from .store import ModelLabStore

    with ModelLabStore(db_path) as store:
        access = PredictionFeatureStore.from_store(store, data_dir=data_dir)
        if not access.is_referenced():
            conn = store.conn
            return pd.read_sql_query("SELECT * FROM prediction_dataset", conn)

        # Materialize outcomes + joined features (registry names) for export
        cols = store._prediction_table_columns()
        outcome_cols = [c for c in cols if not str(c).startswith("sf_")]
        rows = access.fetch_rows(outcome_cols=outcome_cols)
        return pd.DataFrame(rows)


def export_prediction_dataset(
    db_path: str,
    out_path: str,
    *,
    fmt: str | None = None,
    data_dir: str | None = None,
) -> dict[str, Any]:
    if not os.path.isfile(db_path):
        return {"ok": False, "error": f"Lab DB not found: {db_path}"}

    ext = (fmt or os.path.splitext(out_path)[1].lstrip(".")).lower()
    if ext in ("csv",):
        df = _load_frame(db_path, data_dir=data_dir)
        df.to_csv(out_path, index=False)
    elif ext in ("parquet", "pq"):
        df = _load_frame(db_path, data_dir=data_dir)
        df.to_parquet(out_path, index=False)
    elif ext in ("sqlite", "db"):
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        if os.path.isfile(out_path):
            os.remove(out_path)
        src = sqlite3.connect(db_path)
        try:
            dest = sqlite3.connect(out_path)
            try:
                src.backup(dest)
                # Keep research tables only
                tables = {
                    r[0]
                    for r in dest.execute(
                        "SELECT name FROM sqlite_master WHERE type='table'"
                    ).fetchall()
                }
                for name in tables:
                    if name not in (
                        "prediction_dataset",
                        "prediction_dataset_summary",
                        "model_lab_info",
                    ):
                        dest.execute(f'DROP TABLE IF EXISTS "{name}"')
                dest.commit()
            finally:
                dest.close()
        finally:
            src.close()
    else:
        return {"ok": False, "error": f"Unsupported export format: {ext}"}

    return {"ok": True, "path": out_path, "format": ext}

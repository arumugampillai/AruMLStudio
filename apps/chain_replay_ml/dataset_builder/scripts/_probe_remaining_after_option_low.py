"""Remaining incomplete rows after forgiving option_low."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import sqlite3

from chain_replay_ml.dataset_builder.nullable_features import mandatory_columns_for_step2
from chain_replay_ml.dataset_builder.transformations.lag_ui import (
    canonical_registry_feature_names,
)

DB = r"D:\data\master_dataset\master_dataset_nifty_3s.db"
DAY = "2026-07-24"
AUDIT = Path(r"D:\data\master_dataset\registry_null_audit_2026-07-24.json")

conn = sqlite3.connect(DB)
db_cols = {r[1] for r in conn.execute("PRAGMA table_info(samples)")}
registry = [c for c in canonical_registry_feature_names() if c in db_cols]
audit = json.loads(AUDIT.read_text(encoding="utf-8"))
dropped = set(audit["summary"]["step1_dropped_columns"])
kept = [c for c in registry if c not in dropped]
mandatory = mandatory_columns_for_step2(kept)
# forgive option_low
mandatory_no_ol = [c for c in mandatory if c != "option_low"]

cols = list(dict.fromkeys(["timestamp", "token", *mandatory]))
col_sql = ", ".join(f'"{c}"' for c in cols)
df = pd.read_sql_query(
    f"SELECT {col_sql} FROM samples WHERE trading_day=?",
    conn,
    params=(DAY,),
)
total = len(df)
tmin = float(df["timestamp"].min())
warmup = df["timestamp"].to_numpy() < (tmin + 15 * 60)

mat = np.column_stack([df[c].notna().to_numpy() for c in mandatory_no_ol])
complete = mat.all(axis=1)
incomplete = ~complete
print(f"after forgiving option_low: complete={complete.sum():,} incomplete={incomplete.sum():,}")
print(f"  of incomplete: inside_warmup={(incomplete & warmup).sum():,} outside={(incomplete & ~warmup).sum():,}")

# Among incomplete outside warmup, which features are still null most often?
out = incomplete & ~warmup
print("\nTop co-null features among incomplete OUTSIDE 15m warmup (option_low forgiven):")
scores = []
for i, c in enumerate(mandatory_no_ol):
    n = int((~mat[:, i] & out).sum())
    if n:
        scores.append((n, c))
scores.sort(reverse=True)
for n, c in scores[:30]:
    print(f"  {c:42s} null_in_remaining={n}")

conn.close()

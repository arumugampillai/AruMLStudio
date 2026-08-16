"""Fast Master-only all-null day trace for dropped analysis columns."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import sys
import os

_apps_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _apps_dir not in sys.path:
    sys.path.insert(0, _apps_dir)
from master_dataset_tk.project_config import resolve_master_data_dir

_DATASETS_DIR = Path(resolve_master_data_dir())
META = _DATASETS_DIR / "analysis_206r_193p_3s_20260730_094409.json"
OUT = META.with_name("_master_null_days.json")
MASTER_DB = r"D:/data/master_dataset/master_dataset_nifty_3s.db"


def main() -> None:
    meta = json.loads(META.read_text(encoding="utf-8"))
    dropped = list(meta["no_null_dropped_columns"])
    days = [d["trading_day"] for d in meta["days"]]
    conn = sqlite3.connect(f"file:{MASTER_DB}?mode=ro", uri=True)
    schema = {r[1] for r in conn.execute("PRAGMA table_info(samples)")}
    mcols = [c for c in dropped if c in schema]
    pipeline = [c for c in dropped if c not in schema]
    print(f"master={len(mcols)} pipeline={len(pipeline)}", flush=True)

    ph = ",".join("?" * len(days))
    parts = ["trading_day", "COUNT(*) AS n"]
    for i, c in enumerate(mcols):
        parts.append(
            f'SUM(CASE WHEN "{c}" IS NOT NULL THEN 1 ELSE 0 END) AS nn_{i}'
        )
    sql = (
        f"SELECT {', '.join(parts)} FROM samples "
        f"WHERE trading_day IN ({ph}) GROUP BY trading_day ORDER BY 1"
    )
    print("querying master…", flush=True)
    rows = conn.execute(sql, days).fetchall()
    print(f"got {len(rows)} days", flush=True)

    tot_n = 0
    tot_nn = [0] * len(mcols)
    all_null: dict[str, list[str]] = {c: [] for c in mcols}
    for row in rows:
        day = str(row[0])
        n = int(row[1])
        tot_n += n
        for i, c in enumerate(mcols):
            nn = int(row[2 + i] or 0)
            tot_nn[i] += nn
            if nn == 0:
                all_null[c].append(day)

    out = {
        "master_all_null": all_null,
        "master_pct": {
            c: round(100.0 * tot_nn[i] / tot_n, 3) if tot_n else 0.0
            for i, c in enumerate(mcols)
        },
        "pipeline_only": pipeline,
        "days": days,
    }
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    for c in mcols:
        print(f"{c}\t{out['master_pct'][c]}\t{all_null[c]}", flush=True)
    print("wrote", OUT, flush=True)
    conn.close()


if __name__ == "__main__":
    main()

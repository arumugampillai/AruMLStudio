"""Fast Analysis pre-NoNull all-null days via row groups."""
from __future__ import annotations

import json
from pathlib import Path

import pyarrow.compute as pc
import pyarrow.parquet as pq

import sys
import os

_apps_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _apps_dir not in sys.path:
    sys.path.insert(0, _apps_dir)
from master_dataset_tk.project_config import resolve_master_data_dir

_DATASETS_DIR = Path(resolve_master_data_dir())
META = _DATASETS_DIR / "analysis_206r_193p_3s_20260730_094409.json"
BAK = META.with_suffix(".parquet.pre_nonull.bak")
OUT = META.with_name("_analysis_null_days.json")


def main() -> None:
    meta = json.loads(META.read_text(encoding="utf-8"))
    dropped = list(meta["no_null_dropped_columns"])
    pf = pq.ParquetFile(str(BAK))
    schema = set(pf.schema_arrow.names)
    use = ["trading_day"] + [c for c in dropped if c in schema]
    print(f"RGs={pf.num_row_groups} cols={len(use)-1}", flush=True)

    tot_n = 0
    tot_nn = {c: 0 for c in dropped}
    all_null: dict[str, list[str]] = {c: [] for c in dropped}

    for i in range(pf.num_row_groups):
        t = pf.read_row_group(i, columns=use)
        days = [str(x) for x in pc.unique(t.column("trading_day")).to_pylist()]
        n = t.num_rows
        tot_n += n
        print(f"RG{i} day={days} n={n}", flush=True)
        if len(days) == 1:
            day = days[0]
            for c in dropped:
                if c not in schema:
                    all_null[c].append(day)
                    continue
                nn = n - t.column(c).null_count
                tot_nn[c] += nn
                if nn == 0:
                    all_null[c].append(day)
        else:
            pdf = t.to_pandas()
            for day, g in pdf.groupby(pdf["trading_day"].astype(str)):
                for c in dropped:
                    if c not in g.columns:
                        all_null[c].append(str(day))
                        continue
                    nn = int(g[c].notna().sum())
                    tot_nn[c] += nn
                    if nn == 0:
                        all_null[c].append(str(day))

    for c in dropped:
        all_null[c] = sorted(set(all_null[c]))
    out = {
        "analysis_all_null": all_null,
        "analysis_pct": {
            c: round(100.0 * tot_nn[c] / tot_n, 3) if tot_n else 0.0
            for c in dropped
        },
    }
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    for c in dropped:
        print(f"{c}\t{out['analysis_pct'][c]}\t{all_null[c]}", flush=True)
    print("wrote", OUT, flush=True)


if __name__ == "__main__":
    main()

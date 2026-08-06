"""Emit canvas-ready feature rows JSON."""
from __future__ import annotations

import json
from pathlib import Path

d = json.loads(
    Path(r"D:\data\master_dataset\registry_null_audit_2026-07-24.json").read_text(
        encoding="utf-8"
    )
)
rows = []
for f in d["features"]:
    if f["null_rows"] == 0:
        continue
    rows.append(
        {
            "feature": f["feature"],
            "null_rows": f["null_rows"],
            "exclusive": f["exclusive_rows_removed"],
            "marginal": f["marginal_rows_saved_if_nullable"],
            "nullable": f["on_nullable_list"],
            "status": f["status"],
            "nullable_candidate": bool(f.get("nullable_candidate", False)),
            "note": f["note"],
        }
    )
Path(r"D:\data\master_dataset\_canvas_embed.json").write_text(
    json.dumps({"summary": d["summary"], "rows": rows, "zero_null": 101}, indent=2),
    encoding="utf-8",
)
print("rows", len(rows))

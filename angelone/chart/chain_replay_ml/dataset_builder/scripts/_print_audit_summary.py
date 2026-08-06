"""Print exclusive killers and status summary from audit JSON."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

p = Path(r"D:\data\master_dataset\registry_null_audit_2026-07-24.json")
d = json.loads(p.read_text(encoding="utf-8"))
feats = d["features"]
with_null = [f for f in feats if f["null_rows"] > 0]
excl = [f for f in feats if f["exclusive_rows_removed"] > 0]
print("features_with_nulls", len(with_null), "of", len(feats))
print("exclusive_killers", len(excl))
for f in excl:
    print(
        f"  {f['feature']}: null={f['null_rows']} excl={f['exclusive_rows_removed']} "
        f"marg={f['marginal_rows_saved_if_nullable']} status={f['status']}"
    )
print("status", Counter(f["status"] for f in with_null))
print("zero_null", sum(1 for f in feats if f["null_rows"] == 0))
print("summary", json.dumps(d["summary"], indent=2))
print("cum", json.dumps(d["cumulative_nullable_experiment"][:5], indent=2))

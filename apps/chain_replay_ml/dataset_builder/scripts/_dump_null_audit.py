import json
from pathlib import Path

r = json.loads(Path("chain_replay_ml/dataset_builder/scripts/null_audit_2026-07-24.json").read_text())
print("TOTALS", json.dumps(r["totals"], indent=2))
print("EARLY", json.dumps(r["early_session"], indent=2))
print("DOMINANT", r["row_loss"]["dominant_cause_counts"])
print("CAUSE_HITS", r["row_loss"]["cause_hit_row_counts"])
print("TOP_KILLS")
for f, n in r["row_loss"]["top_columns_killing_rows"][:30]:
    print(f"  {f:42} {n}")
print("TOP_NULL_COLS")
cols = sorted(r["column_stats"], key=lambda x: -x["null_count"])
for c in cols[:45]:
    if c["null_count"] <= 0:
        continue
    print(
        f"  {c['feature']:42} {c['null_count']:7} {c['null_pct']:6.2f}% "
        f"{c['cause']:26} {c['first_null_ist']}-{c['last_null_ist']}"
    )
print("CAUSE_COLS")
for k, v in sorted(r["cause_to_columns"].items(), key=lambda kv: -len(kv[1])):
    print(f"  {k}: {len(v)} -> {v[:12]}")
print("MINUTE_TOP", r["row_loss"]["incomplete_by_minute_top"][:20])

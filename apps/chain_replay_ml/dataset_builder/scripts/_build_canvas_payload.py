import json
from pathlib import Path

base = json.loads(Path("chain_replay_ml/dataset_builder/scripts/null_audit_2026-07-24.json").read_text())
deep = json.loads(Path("chain_replay_ml/dataset_builder/scripts/null_audit_2026-07-24_deep.json").read_text())

# Reclassify columns properly for report
def refine(c):
    f = c["feature"]
    n = c["null_count"]
    if f in ("option_vwap", "futures_vwap"):
        return {**c, "cause": "feed_never_populated", "expected": False,
                "note": "100% NULL all day — column not wired / feed missing. Dropped in No-Null Step 1 (no row loss)."}
    if f in ("gamma_flip_spot", "gamma_flip_distance"):
        return {**c, "cause": "gamma_flip_no_sign_change", "expected": True,
                "note": "GEX totals present all day; flip OK only 09:17:45–12:30. After 12:30 cum(net GEX) has no zero crossing (documented nullable)."}
    if f == "option_low":
        return {**c, "cause": "session_day_low_missing", "expected": False,
                "note": "9 tokens always NULL; option_high/open/prev_close OK; LTP present. token_day_meta.day_low missing or ≤0."}
    if "_ema" in f or f.startswith("weighted_"):
        return {**c, "cause": "ema_warmup", "expected": True,
                "note": "EMA / weighted EMA needs prior observations; spot EMAs clear by ~09:20–09:30."}
    if "score_" in f or "sample_count_" in f or f.startswith("opt_rv") or "rv_" in f:
        return {**c, "cause": "rolling_window_warmup", "expected": True,
                "note": "Rolling score/RV window incomplete early session."}
    if f in ("current_iv", "vega", "vanna", "charm", "speed") or "iv_" in f or f.endswith("_iv"):
        return {**c, "cause": "iv_or_iv_derived", "expected": False if n > 5000 else True,
                "note": "IV solve / IV-derived gaps on some ticks."}
    if f in ("vega", "vanna", "charm", "speed"):
        return {**c, "cause": "greeks_need_iv", "expected": False,
                "note": "Tracks current_iv nulls (3823)."}
    return c

cols = [refine(c) for c in base["column_stats"]]
with_null = sorted([c for c in cols if c["null_count"] > 0], key=lambda x: -x["null_count"])

# Build slim payload for canvas
payload = {
    "totals": base["totals"],
    "early_session": base["early_session"],
    "deep": {
        "scenarios": deep["scenarios_complete_rows"],
        "recovery": deep["recovery_vs_baseline"],
        "high_impact_partition": deep["high_impact_partition"],
        "gamma_flip_time": deep["gamma_flip_time"],
        "option_low_vs_ohlc": deep["option_low_vs_ohlc"],
        "iv_greeks_nulls": deep["iv_greeks_nulls"],
        "core_nulls": deep["core_nulls"],
        "by_option_type": deep["by_option_type"],
        "tok_sum": deep["gamma_flip_token_summary"],
    },
    "top40": [
        {
            "feature": c["feature"],
            "null_count": c["null_count"],
            "null_pct": c["null_pct"],
            "cause": c.get("cause"),
            "expected": c.get("expected"),
            "first": c.get("first_null_ist"),
            "last": c.get("last_null_ist"),
            "note": c.get("note"),
        }
        for c in with_null[:40]
    ],
    "all_null_columns": base["totals"]["columns_100pct_null"],
    "gex_probe": {
        "call_gex_null": 0,
        "net_gex_null": 0,
        "gamma_flip_null": 156744,
        "gamma_flip_ok_until": "12:30:00",
        "after_1230_rows": 147000,
        "after_1230_gf_null": 147000,
        "thru_1230_gf_null": 9744,
    },
}
Path("chain_replay_ml/dataset_builder/scripts/null_audit_canvas_payload.json").write_text(
    json.dumps(payload)
)
print("payload keys", payload.keys())
print("top5", [c["feature"] for c in payload["top40"][:5]])
print("recovery", payload["deep"]["recovery"])

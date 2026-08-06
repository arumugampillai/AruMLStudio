"""Compare two Create Model packages (Dataset Engine off vs on).

After running the same Create Model job twice — once with
``ARUNEO_DATASET_ENGINE=off`` and once with ``on`` — point this at the
two package directories. It checks the observation checklist:

  Dataset checksum / shape, feature count, training rows,
  holdout metrics, walk-forward metrics, selected features,
  dataset_load metrics.

Usage (from angelone/chart)::

  set PYTHONPATH=.
  python -m chain_replay_ml.dataset_engine.compare_create_model_runs ^
      --off path/to/models/Run_OFF ^
      --on  path/to/models/Run_ON
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


_HOLDOUT_KEYS = (
    "rmse",
    "mae",
    "r2",
    "directional_accuracy_pct",
    "mape",
    "medae",
)
_WF_MEAN_KEYS = (
    "mean_rmse",
    "mean_mae",
    "mean_r2",
    "mean_directional_accuracy_pct",
    "mean_mape",
)
# Identity across backends (pandas has no partition prune counters).
_LOAD_COMPARE_KEYS = (
    "rows_returned",
    "columns_returned",
)
_LOAD_IMPROVE_OR_EQUAL = (
    "load_time_sec",
    "peak_rss_mb",
)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    with path.open(encoding="utf-8") as fh:
        data = json.load(fh)
    return data if isinstance(data, dict) else {}


def _read_text(path: Path) -> str | None:
    if not path.is_file():
        return None
    return path.read_text(encoding="utf-8").strip()


def _floats_close(a: Any, b: Any, *, rtol: float, atol: float) -> bool:
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    try:
        fa, fb = float(a), float(b)
    except (TypeError, ValueError):
        return a == b
    if math.isnan(fa) and math.isnan(fb):
        return True
    return math.isclose(fa, fb, rel_tol=rtol, abs_tol=atol)


def _holdout_block(metrics: dict[str, Any]) -> dict[str, Any]:
    test = metrics.get("test")
    if isinstance(test, dict) and test:
        return test
    validation = metrics.get("validation")
    return validation if isinstance(validation, dict) else {}


def _wf_block(metrics: dict[str, Any], package: Path) -> dict[str, Any]:
    prod = metrics.get("production_walk_forward")
    if isinstance(prod, dict) and prod:
        return prod
    summary = _read_json(package / "walk_forward" / "summary.json")
    agg = summary.get("aggregated")
    return agg if isinstance(agg, dict) else {}


def _selected_features(package: Path, metadata: dict[str, Any]) -> list[str]:
    csv_path = package / "walk_forward" / "selected_features.csv"
    if csv_path.is_file():
        lines = csv_path.read_text(encoding="utf-8").splitlines()
        if len(lines) >= 2:
            header = [c.strip().lower() for c in lines[0].split(",")]
            try:
                name_i = header.index("feature")
            except ValueError:
                name_i = 0
            sel_i = header.index("selected") if "selected" in header else None
            out: list[str] = []
            for row in lines[1:]:
                parts = [p.strip() for p in row.split(",")]
                if not parts or not parts[0]:
                    continue
                if sel_i is not None and sel_i < len(parts):
                    if parts[sel_i].lower() not in {"yes", "true", "1", "y"}:
                        continue
                out.append(parts[name_i] if name_i < len(parts) else parts[0])
            if out:
                return out
    feats = metadata.get("features") or metadata.get("selected_features")
    if isinstance(feats, list):
        return [str(x) for x in feats]
    return []


def _dataset_load(metadata: dict[str, Any], training_meta: dict[str, Any]) -> dict[str, Any]:
    for block in (metadata, training_meta):
        load = block.get("dataset_load")
        if isinstance(load, dict) and load:
            return load
    nested = training_meta.get("metadata")
    if isinstance(nested, dict):
        load = nested.get("dataset_load")
        if isinstance(load, dict):
            return load
    return {}


def compare_create_model_packages(
    off_dir: str | Path,
    on_dir: str | Path,
    *,
    rtol: float = 1e-9,
    atol: float = 1e-6,
) -> dict[str, Any]:
    """Return a structured pass/fail report for an off vs on Create Model pair."""
    off_p = Path(off_dir)
    on_p = Path(on_dir)

    off_meta = _read_json(off_p / "metadata.json")
    on_meta = _read_json(on_p / "metadata.json")
    off_metrics = _read_json(off_p / "metrics.json")
    on_metrics = _read_json(on_p / "metrics.json")
    off_tm = _read_json(off_p / "training_metadata.json")
    on_tm = _read_json(on_p / "training_metadata.json")

    checks: list[dict[str, Any]] = []

    def add(category: str, expected: str, ok: bool, detail: Any = None) -> None:
        checks.append(
            {
                "category": category,
                "expected": expected,
                "ok": bool(ok),
                "detail": detail,
            }
        )

    off_schema = _read_text(off_p / "schema_hash.txt")
    on_schema = _read_text(on_p / "schema_hash.txt")
    off_val = _read_text(off_p / "validation_hash.txt")
    on_val = _read_text(on_p / "validation_hash.txt")
    add(
        "Dataset checksum / shape",
        "Identical",
        off_schema == on_schema and off_val == on_val and off_schema is not None,
        {
            "schema_hash_match": off_schema == on_schema,
            "validation_hash_match": off_val == on_val,
            "off_schema": off_schema,
            "on_schema": on_schema,
        },
    )

    off_fc = off_meta.get("feature_count")
    on_fc = on_meta.get("feature_count")
    add(
        "Feature count",
        "Identical",
        off_fc == on_fc and off_fc is not None,
        {"off": off_fc, "on": on_fc},
    )

    off_rows = off_meta.get("row_count")
    on_rows = on_meta.get("row_count")
    add(
        "Training rows",
        "Identical",
        off_rows == on_rows and off_rows is not None,
        {"off": off_rows, "on": on_rows},
    )

    off_hold = _holdout_block(off_metrics)
    on_hold = _holdout_block(on_metrics)
    hold_diffs = {
        k: {"off": off_hold.get(k), "on": on_hold.get(k)}
        for k in _HOLDOUT_KEYS
        if not _floats_close(off_hold.get(k), on_hold.get(k), rtol=rtol, atol=atol)
    }
    add(
        "Holdout metrics (MAE/RMSE/etc.)",
        "Identical (within floating-point tolerance)",
        bool(off_hold) and bool(on_hold) and not hold_diffs,
        {"diffs": hold_diffs or None, "keys_compared": list(_HOLDOUT_KEYS)},
    )

    off_wf = _wf_block(off_metrics, off_p)
    on_wf = _wf_block(on_metrics, on_p)
    wf_diffs = {
        k: {"off": off_wf.get(k), "on": on_wf.get(k)}
        for k in _WF_MEAN_KEYS
        if not _floats_close(off_wf.get(k), on_wf.get(k), rtol=rtol, atol=atol)
    }
    # Non-WF packages: both empty → skip as N/A pass with note
    if not off_wf and not on_wf:
        add(
            "Walk-forward metrics",
            "Identical (within floating-point tolerance)",
            True,
            {"note": "No walk-forward block on either package (skipped)"},
        )
    else:
        add(
            "Walk-forward metrics",
            "Identical (within floating-point tolerance)",
            bool(off_wf) and bool(on_wf) and not wf_diffs,
            {"diffs": wf_diffs or None, "keys_compared": list(_WF_MEAN_KEYS)},
        )

    off_feats = _selected_features(off_p, off_meta)
    on_feats = _selected_features(on_p, on_meta)
    add(
        "Selected features",
        "Identical, or explain any intentional nondeterminism",
        off_feats == on_feats and bool(off_feats),
        {
            "off_count": len(off_feats),
            "on_count": len(on_feats),
            "match": off_feats == on_feats,
            "only_off": sorted(set(off_feats) - set(on_feats))[:20],
            "only_on": sorted(set(on_feats) - set(off_feats))[:20],
        },
    )

    off_load = _dataset_load(off_meta, off_tm)
    on_load = _dataset_load(on_meta, on_tm)
    load_identity_ok = True
    load_detail: dict[str, Any] = {"off": off_load or None, "on": on_load or None}
    if not off_load and not on_load:
        add(
            "Dataset load metrics",
            "Improved or equal",
            True,
            {"note": "dataset_load missing on both packages (re-run after wiring)"},
        )
    else:
        for k in _LOAD_COMPARE_KEYS:
            if off_load.get(k) != on_load.get(k):
                load_identity_ok = False
                load_detail.setdefault("identity_diffs", {})[k] = {
                    "off": off_load.get(k),
                    "on": on_load.get(k),
                }
        improve: dict[str, Any] = {}
        improve_ok = True
        for k in _LOAD_IMPROVE_OR_EQUAL:
            a, b = off_load.get(k), on_load.get(k)
            if a is None or b is None:
                continue
            try:
                fa, fb = float(a), float(b)
            except (TypeError, ValueError):
                continue
            better_or_eq = fb <= fa * 1.05  # allow 5% noise
            improve[k] = {"off": fa, "on": fb, "improved_or_equal": better_or_eq}
            if not better_or_eq:
                improve_ok = False
        load_detail["improve_or_equal"] = improve
        add(
            "Dataset load metrics",
            "Improved or equal",
            load_identity_ok and improve_ok,
            load_detail,
        )

    overall = all(c["ok"] for c in checks)
    return {
        "ok": overall,
        "off_dir": str(off_p),
        "on_dir": str(on_p),
        "checks": checks,
        "transparent_replacement": overall,
    }


def _print_report(report: dict[str, Any]) -> None:
    print("Create Model pair comparison")
    print(f"  off: {report['off_dir']}")
    print(f"  on:  {report['on_dir']}")
    print()
    print(f"{'Category':<42} {'OK':<5} Expected")
    print("-" * 90)
    for c in report["checks"]:
        flag = "PASS" if c["ok"] else "FAIL"
        print(f"{c['category']:<42} {flag:<5} {c['expected']}")
        if not c["ok"] and c.get("detail") is not None:
            print(f"    detail: {json.dumps(c['detail'], default=str)[:500]}")
    print("-" * 90)
    print(
        "Transparent replacement:"
        f" {'YES' if report['transparent_replacement'] else 'NO'}"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare Create Model packages (Dataset Engine off vs on)."
    )
    parser.add_argument("--off", required=True, help="Package dir trained with ENGINE=off")
    parser.add_argument("--on", required=True, help="Package dir trained with ENGINE=on")
    parser.add_argument("--rtol", type=float, default=1e-9)
    parser.add_argument("--atol", type=float, default=1e-6)
    parser.add_argument("--json", action="store_true", help="Print full JSON report")
    args = parser.parse_args(argv)

    report = compare_create_model_packages(
        args.off, args.on, rtol=args.rtol, atol=args.atol
    )
    if args.json:
        print(json.dumps(report, indent=2, default=str))
    else:
        _print_report(report)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

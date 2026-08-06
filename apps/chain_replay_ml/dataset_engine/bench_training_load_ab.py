"""Separate-process A/B for Model Builder dataset load (fair Peak RSS).

Usage (from angelone/chart):

  set PYTHONPATH=.
  python -m chain_replay_ml.dataset_engine.bench_training_load_ab ^
      --data-dir path/to/chart/data ^
      --dataset analysis_206r_193p_3s_20260730_094409 ^
      --premium-min 15 --premium-max 100 ^
      --features feat_a,feat_b ^
      --target future_ltp_60

If --features/--target omitted, picks first available numeric-ish columns
from the parquet schema (same heuristic as the internal bench helper).

Each backend runs in its own Python process so Peak RSS is not polluted
by the other path.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


def _worker_payload() -> dict[str, Any]:
    """Executed only in child processes (argv JSON blob)."""
    import logging

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    raw = json.loads(os.environ["ARUNEO_LOAD_AB_PAYLOAD"])
    os.environ["ARUNEO_DATASET_ENGINE"] = raw["backend_flag"]  # off | on

    from chain_replay_ml.training.config import TrainingConfig
    from chain_replay_ml.training.dataset_loader import load_training_xy
    from chain_replay_ml.training.load_backend import process_rss_mb
    import time

    cfg = TrainingConfig(
        dataset=raw["dataset"],
        features=list(raw["features"]),
        target=str(raw["target"]),
        premium_selection_enabled=bool(raw.get("premium_enabled", False)),
        premium_min=raw.get("premium_min"),
        premium_max=raw.get("premium_max"),
    )
    rss0 = process_rss_mb()
    t0 = time.perf_counter()
    X, y, _feats, meta, _exp, _ctx = load_training_xy(raw["data_dir"], cfg)
    elapsed = time.perf_counter() - t0
    rss1 = process_rss_mb()
    load = dict(meta.get("dataset_load") or {})
    out = {
        "backend_flag": raw["backend_flag"],
        "dataset_load": load,
        "wall_time_sec": round(elapsed, 6),
        "rss_before_mb": None if rss0 is None else round(rss0, 3),
        "rss_after_mb": None if rss1 is None else round(rss1, 3),
        "peak_rss_mb": load.get("peak_rss_mb")
        if load.get("peak_rss_mb") is not None
        else (
            None
            if rss0 is None or rss1 is None
            else round(max(rss0, rss1), 3)
        ),
        "x_shape": list(X.shape),
        "y_len": int(len(y)),
        "x_checksum": float(X.to_numpy(dtype=float, copy=False).sum()),
        "y_checksum": float(y.to_numpy(dtype=float, copy=False).sum()),
    }
    print(json.dumps(out), flush=True)
    return out


def _run_child(payload: dict[str, Any]) -> dict[str, Any]:
    env = os.environ.copy()
    env["ARUNEO_LOAD_AB_PAYLOAD"] = json.dumps(payload)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(Path(__file__).resolve().parents[2]), env.get("PYTHONPATH", "")]
    )
    # Re-enter this module as worker
    cmd = [
        sys.executable,
        "-c",
        "from chain_replay_ml.dataset_engine.bench_training_load_ab import _worker_payload; _worker_payload()",
    ]
    proc = subprocess.run(
        cmd,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"child backend={payload['backend_flag']} failed rc={proc.returncode}\n"
            f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        )
    # Last JSON line is the result (logging may precede it)
    lines = [ln.strip() for ln in proc.stdout.splitlines() if ln.strip().startswith("{")]
    if not lines:
        raise RuntimeError(f"no JSON result from child\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}")
    return json.loads(lines[-1])


def _default_features_target(parquet_path: Path) -> tuple[list[str], str]:
    import pyarrow.parquet as pq

    cols = list(pq.read_schema(str(parquet_path)).names)
    skip = {
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
        "master_row_id",
        "ltp_to_spot_ratio",
    }
    feats = [
        c
        for c in cols
        if c not in skip and not str(c).startswith("future_")
    ][:5]
    targets = [c for c in cols if str(c).startswith("future_ltp")]
    if not feats or not targets:
        raise SystemExit(f"Could not infer features/target from {parquet_path}")
    return feats, targets[0]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", required=True, help="chart data dir (contains datasets/)")
    ap.add_argument("--dataset", required=True, help="dataset stem (no .parquet)")
    ap.add_argument("--features", default="", help="comma-separated feature columns")
    ap.add_argument("--target", default="", help="target column")
    ap.add_argument("--premium-min", type=float, default=None)
    ap.add_argument("--premium-max", type=float, default=None)
    args = ap.parse_args(argv)

    data_dir = os.path.abspath(args.data_dir)
    parquet = Path(data_dir) / "datasets" / f"{args.dataset}.parquet"
    if not parquet.is_file():
        raise SystemExit(f"Parquet not found: {parquet}")

    if args.features.strip() and args.target.strip():
        features = [c.strip() for c in args.features.split(",") if c.strip()]
        target = args.target.strip()
    else:
        features, target = _default_features_target(parquet)

    premium_enabled = args.premium_min is not None and args.premium_max is not None
    base = {
        "data_dir": data_dir,
        "dataset": args.dataset,
        "features": features,
        "target": target,
        "premium_enabled": premium_enabled,
        "premium_min": args.premium_min,
        "premium_max": args.premium_max,
    }

    print("=== Process A: ARUNEO_DATASET_ENGINE=off ===", flush=True)
    off = _run_child({**base, "backend_flag": "off"})
    print("=== Process B: ARUNEO_DATASET_ENGINE=on ===", flush=True)
    on = _run_child({**base, "backend_flag": "on"})

    checksum_match = (
        abs(off["x_checksum"] - on["x_checksum"]) < 1e-6
        and abs(off["y_checksum"] - on["y_checksum"]) < 1e-6
        and off["x_shape"] == on["x_shape"]
    )
    summary = {
        "dataset": args.dataset,
        "features": features,
        "target": target,
        "checksum_match": checksum_match,
        "off": {
            "wall_time_sec": off["wall_time_sec"],
            "peak_rss_mb": off["peak_rss_mb"],
            "dataset_load": off["dataset_load"],
            "x_shape": off["x_shape"],
        },
        "on": {
            "wall_time_sec": on["wall_time_sec"],
            "peak_rss_mb": on["peak_rss_mb"],
            "dataset_load": on["dataset_load"],
            "x_shape": on["x_shape"],
        },
    }
    print("=== SUMMARY ===", flush=True)
    print(json.dumps(summary, indent=2), flush=True)
    print(
        f"\nGate: checksum_match={checksum_match} "
        f"load_off={off['wall_time_sec']:.3f}s load_on={on['wall_time_sec']:.3f}s "
        f"rss_off={off['peak_rss_mb']} rss_on={on['peak_rss_mb']}",
        flush=True,
    )
    return 0 if checksum_match else 2


if __name__ == "__main__":
    # Child re-entry uses _worker_payload via -c; parent uses main().
    if os.environ.get("ARUNEO_LOAD_AB_PAYLOAD"):
        _worker_payload()
    else:
        raise SystemExit(main())

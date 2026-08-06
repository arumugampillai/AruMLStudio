"""CLI: python -m chain_replay_ml.prediction_meta [--data-dir PATH] ..."""

from __future__ import annotations

import argparse
import json
import os
import sys

from tick_pipeline import DATA_DIR

from .builder import build_prediction_meta_dataset, resolve_prediction_meta_db_path


def _progress(payload: dict) -> None:
    phase = payload.get("phase", "")
    done = payload.get("rows_done", 0)
    total = payload.get("rows_total", 0)
    if phase == "batch":
        pct = payload.get("pct", 0)
        print(f"\r[{pct:5.1f}%] {done}/{total} rows", end="", flush=True)
    elif phase == "complete":
        print(f"\nComplete: {done}/{total} rows")
        stats = payload.get("stats") or {}
        print(json.dumps(stats, indent=2))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build prediction meta dataset from master SQLite")
    parser.add_argument("--data-dir", default=DATA_DIR, help="Chart data directory")
    parser.add_argument("--market", default="NIFTY")
    parser.add_argument("--interval-sec", type=int, default=3)
    parser.add_argument("--master-db", default=None, help="Override master dataset path")
    parser.add_argument("--output-db", default=None, help="Override output SQLite path")
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument("--no-resume", action="store_true", help="Rebuild from scratch")
    parser.add_argument("--no-path-enrich", action="store_true", help="Skip tick-path outcome columns")
    args = parser.parse_args(argv)

    data_dir = os.path.abspath(args.data_dir)
    try:
        result = build_prediction_meta_dataset(
            data_dir,
            market=args.market,
            sampling_interval_sec=args.interval_sec,
            master_db_path=args.master_db,
            output_db_path=args.output_db,
            batch_size=args.batch_size,
            resume=not args.no_resume,
            enrich_path_outcomes=not args.no_path_enrich,
            on_progress=_progress,
        )
    except Exception as exc:
        print(f"Build failed: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

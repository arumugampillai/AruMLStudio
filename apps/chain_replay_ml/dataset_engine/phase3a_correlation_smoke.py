"""Phase 3A smoke — Correlation Engine vs Pandas on real analysis parquets."""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any

from chain_replay_ml.dataset_builder.analysis_correlation import (
    compare_correlation_matrices,
    compute_correlation_frame,
)
from chain_replay_ml.dataset_engine.phase2_observation import _chart_data_dir, _evidence_dir


_LOG = logging.getLogger("phase3a_correlation")


def _append(path: Path, entry: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, default=str) + "\n")


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=str(_chart_data_dir()))
    parser.add_argument(
        "--datasets",
        default="analysis_206r_212p_3s_20260728_110059,analysis_206r_193p_3s_20260730_025644",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=0,
        help="0 = full frame for parity; else cap (Engine LIMIT vs legacy sample may diverge)",
    )
    args = parser.parse_args(argv)
    data_dir = Path(args.data_dir)
    out = _evidence_dir(data_dir) / "phase3a_correlation.jsonl"
    max_rows = None if int(args.max_rows) <= 0 else int(args.max_rows)

    failures = 0
    for name in [s.strip() for s in args.datasets.split(",") if s.strip()]:
        path = data_dir / "datasets" / f"{name}.parquet"
        if not path.is_file():
            _LOG.warning("skip missing %s", path)
            continue
        _LOG.info("Correlation parity dataset=%s max_rows=%s", name, max_rows)
        t0 = time.perf_counter()
        report = compare_correlation_matrices(str(path), max_rows=max_rows)
        # Also one auto-path load for telemetry under default env.
        corr, feats = compute_correlation_frame(str(path), max_rows=max_rows or 150_000)
        load = dict((getattr(corr, "attrs", None) or {}).get("dataset_load") or {})
        entry = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "dataset": name,
            "max_rows": max_rows,
            "wall_sec": round(time.perf_counter() - t0, 3),
            "parity_ok": bool(report.get("ok")),
            "matrices_close": bool(report.get("matrices_close")),
            "max_abs_diff": report.get("max_abs_diff"),
            "features": len(feats),
            "auto_dataset_load": load,
            "engine_load": report.get("engine_load"),
            "pandas_load": report.get("pandas_load"),
        }
        _append(out, entry)
        _LOG.info(
            "recorded parity_ok=%s max_abs_diff=%s auto_backend=%s → %s",
            entry["parity_ok"],
            entry["max_abs_diff"],
            load.get("backend"),
            out,
        )
        if not entry["parity_ok"]:
            failures += 1

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

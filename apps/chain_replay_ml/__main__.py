"""CLI: export Phase 1 canonical rows from chain replay SQLite."""

from __future__ import annotations

import argparse
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
from path_config import ensure_ml_studio_paths

ensure_ml_studio_paths()

from storage.chain_replay_export import ChainReplayError

from chain_replay_ml.pipeline import (
    default_out_path,
    delta_profile_suffix,
    export_day_features,
    write_csv,
    write_parquet,
)
from chain_replay_ml.constants import DEFAULT_DELTA_PROFILE_TARGET
from chain_replay_ml.reanchor import ReanchorThresholds


def _parse_strikes(raw: str | None) -> list[float] | None:
    if not raw:
        return None
    return [float(s.strip()) for s in raw.split(",") if s.strip()]


def _parse_types(raw: str | None) -> list[str] | None:
    if not raw:
        return None
    return [t.strip().upper() for t in raw.split(",") if t.strip()]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Export Phase 1 ML features (1m rows) from chain replay SQLite.",
    )
    parser.add_argument("--underlying", default="NIFTY")
    parser.add_argument("--expiry", required=True, help="YYYY-MM-DD")
    parser.add_argument("--date", required=True, help="Replay trading day YYYY-MM-DD")
    parser.add_argument("--strikes", help="Comma-separated strikes in rupees, e.g. 24100,24200")
    parser.add_argument("--types", help="CE,PE (default both)")
    parser.add_argument("--atm-band", type=int, default=None, help="ATM ± N strikes (overrides --strikes)")
    parser.add_argument(
        "--delta-profile",
        nargs="?",
        type=float,
        const=DEFAULT_DELTA_PROFILE_TARGET,
        metavar="DELTA",
        help=(
            "Export all CE/PE in delta band (target, ATM] at seed time "
            f"(default target={DEFAULT_DELTA_PROFILE_TARGET}). "
            "Overrides --strikes and --atm-band."
        ),
    )
    parser.add_argument("--iv-threshold", type=float, default=2.0)
    parser.add_argument("--spot-threshold", type=float, default=0.3)
    parser.add_argument("--max-roll-age", type=float, default=15.0)
    parser.add_argument("--skip-warmup", action="store_true", help="Drop first 15 minutes")
    parser.add_argument("--format", choices=("csv", "parquet"), default="csv")
    parser.add_argument("--out", default=None, help="Output file path")
    args = parser.parse_args(argv)

    thresholds = ReanchorThresholds(
        iv_pct=args.iv_threshold,
        spot_pct=args.spot_threshold,
        max_age_min=args.max_roll_age,
    )

    try:
        t0 = time.monotonic()
        rows, profile_meta = export_day_features(
            chart_dir=CHART_DIR,
            underlying=args.underlying,
            expiry=args.expiry,
            date=args.date,
            strikes=_parse_strikes(args.strikes) if args.delta_profile is None else None,
            types=_parse_types(args.types) if args.delta_profile is None else None,
            atm_band=args.atm_band if args.delta_profile is None else None,
            delta_profile=args.delta_profile,
            thresholds=thresholds,
            skip_warmup=args.skip_warmup,
        )
        if not rows:
            print("No rows exported (check ticks / filters).", file=sys.stderr)
            return 1

        ext = "parquet" if args.format == "parquet" else "csv"
        profile_suffix = (
            delta_profile_suffix(args.delta_profile) if args.delta_profile is not None else None
        )
        out_path = args.out or default_out_path(
            CHART_DIR,
            args.underlying.upper(),
            args.expiry,
            args.date,
            ext,
            profile_suffix=profile_suffix,
        )
        if args.format == "parquet":
            write_parquet(out_path, rows)
        else:
            write_csv(out_path, rows)

        labeled = sum(1 for r in rows if r.get("residual_5m") is not None)
        print(
            f"Exported {len(rows)} rows ({labeled} with residual_5m) "
            f"in {time.monotonic() - t0:.1f}s -> {out_path}"
        )
        if profile_meta and profile_meta.get("picked"):
            ce_d = profile_meta.get("atm_ce_delta")
            pe_d = profile_meta.get("atm_pe_delta")
            lo = profile_meta.get("target_delta")
            print(
                f"Delta profile band (09:16 seed): "
                f"CE {lo} < d <= {ce_d}, PE {pe_d} <= d < -{lo}"
            )
            print(f"  {len(profile_meta['picked'])} options:")
            for pick in profile_meta["picked"]:
                print(
                    f"  {pick['type']} {pick['strike']:.0f} "
                    f"delta={pick['delta']}  {pick.get('symbol')}"
                )
        return 0
    except ChainReplayError as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

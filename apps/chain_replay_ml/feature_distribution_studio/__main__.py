"""CLI for Feature Distribution Studio compute.

Usage (from angelone/chart)::

  set PYTHONPATH=.
  python -m chain_replay_ml.feature_distribution_studio \\
      --data-dir data \\
      --model DatasetEngine_CM_20260731_015113_ENGINE_ON
"""

from __future__ import annotations

import argparse
import json
import logging


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    parser = argparse.ArgumentParser(description="Feature Distribution Studio (compute)")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--model", required=True, help="Model package name")
    parser.add_argument("--holdout-max-rows", type=int, default=20_000)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    from chain_replay_ml.feature_distribution_studio import (
        run_feature_distribution_studio,
    )

    def on_progress(ev: dict) -> None:
        logging.getLogger("fds").info(
            "stage=%s %s",
            ev.get("stage"),
            {k: v for k, v in ev.items() if k != "stage"},
        )

    result = run_feature_distribution_studio(
        data_dir=args.data_dir,
        model_name=args.model,
        holdout_max_rows=args.holdout_max_rows,
        progress=on_progress,
    )
    if args.json:
        print(
            json.dumps(
                {
                    "ok": result.ok,
                    "error": result.error,
                    "model_name": result.model_name,
                    "artifacts_dir": result.artifacts_dir,
                    "meta": result.meta,
                    "comparison_preview": result.comparison[:15],
                },
                indent=2,
                default=str,
            )
        )
    else:
        print(f"ok={result.ok} error={result.error}")
        print(f"artifacts={result.artifacts_dir}")
        if result.comparison:
            print(
                f"{'feature':<36} {'null%':>7} {'mean':>12} {'p50':>12} {'skew':>8}"
            )
            for row in result.comparison[:20]:
                mean = row.get("mean")
                p50 = row.get("p50")
                skew = row.get("skew")
                print(
                    f"{str(row.get('feature')):<36} "
                    f"{float(row.get('null_pct') or 0):>7.2f} "
                    f"{(float(mean) if mean is not None else float('nan')):>12.4f} "
                    f"{(float(p50) if p50 is not None else float('nan')):>12.4f} "
                    f"{(float(skew) if skew is not None else float('nan')):>8.3f}"
                )
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

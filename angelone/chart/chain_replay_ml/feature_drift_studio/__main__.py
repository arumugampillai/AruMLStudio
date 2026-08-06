"""CLI for Feature Drift Studio compute.

Usage (from angelone/chart)::

  set PYTHONPATH=.
  python -m chain_replay_ml.feature_drift_studio \\
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
    parser = argparse.ArgumentParser(description="Feature Drift Studio (compute)")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--model", required=True, help="Model package name")
    parser.add_argument("--holdout-max-rows", type=int, default=20_000)
    parser.add_argument("--wf-max-rows", type=int, default=50_000)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    from chain_replay_ml.feature_drift_studio import run_feature_drift_studio

    def on_progress(ev: dict) -> None:
        logging.getLogger("drift").info(
            "stage=%s %s",
            ev.get("stage"),
            {k: v for k, v in ev.items() if k != "stage"},
        )

    result = run_feature_drift_studio(
        data_dir=args.data_dir,
        model_name=args.model,
        holdout_max_rows=args.holdout_max_rows,
        wf_max_rows=args.wf_max_rows,
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
        meta = result.meta or {}
        print(
            f"feature_drift={meta.get('feature_drift_pct')}% "
            f"similarity={meta.get('similarity_pct')}%"
        )
        if result.comparison:
            print(
                f"{'feature':<28} {'drift%':>7} {'KS':>7} {'W':>8} {'risk':>8} {'risk_sc':>8}"
            )
            for row in result.comparison[:20]:
                dp = row.get("drift_pct")
                ks = row.get("ks_statistic")
                w = row.get("wasserstein_distance")
                print(
                    f"{str(row.get('feature')):<28} "
                    f"{(float(dp) if dp is not None else float('nan')):>7.1f} "
                    f"{(float(ks) if ks is not None else float('nan')):>7.4f} "
                    f"{(float(w) if w is not None else float('nan')):>8.4f} "
                    f"{str(row.get('risk') or ''):>8} "
                    f"{float(row.get('risk_score') or 0):>8.2f}"
                )
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

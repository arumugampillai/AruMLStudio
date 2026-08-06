"""CLI for Feature Importance Studio compute (Milestone 1–3).

Usage (from angelone/chart)::

  set PYTHONPATH=.
  python -m chain_replay_ml.feature_importance_studio \\
      --data-dir data \\
      --model DatasetEngine_CM_20260731_015113_ENGINE_ON
"""

from __future__ import annotations

import argparse
import json
import logging
import sys


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    parser = argparse.ArgumentParser(description="Feature Importance Studio (compute)")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--model", required=True, help="Model package name")
    parser.add_argument("--holdout-max-rows", type=int, default=20_000)
    parser.add_argument("--permutation-repeats", type=int, default=3)
    parser.add_argument("--shap-sample", type=int, default=400)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    from chain_replay_ml.feature_importance_studio import run_feature_importance_studio

    def on_progress(ev: dict) -> None:
        logging.getLogger("fis").info(
            "stage=%s %s",
            ev.get("stage"),
            {k: v for k, v in ev.items() if k != "stage"},
        )

    result = run_feature_importance_studio(
        data_dir=args.data_dir,
        model_name=args.model,
        holdout_max_rows=args.holdout_max_rows,
        permutation_n_repeats=args.permutation_repeats,
        shap_sample_size=args.shap_sample,
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
            print(f"{'feature':<40} {'gain':>10} {'perm':>10} {'shap':>10}")
            for row in result.comparison[:20]:
                print(
                    f"{str(row.get('feature')):<40} "
                    f"{float(row.get('gain') or 0):>10.4f} "
                    f"{float(row.get('permutation_mean') or 0):>10.6f} "
                    f"{float(row.get('shap_mean_abs') or 0):>10.6f}"
                )
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

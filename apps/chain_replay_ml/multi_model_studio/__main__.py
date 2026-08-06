"""CLI for Multi-model Feature Studio (join-only).

Usage (from angelone/chart)::

  set PYTHONPATH=.
  python -m chain_replay_ml.multi_model_studio \\
      --data-dir data --model-a ModelA --model-b ModelB
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
    parser = argparse.ArgumentParser(description="Multi-model Feature Studio (join)")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--model-a", required=True)
    parser.add_argument("--model-b", required=True)
    parser.add_argument(
        "--require",
        default="",
        help="Comma list: importance,distribution,drift (optional hard requirements)",
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    from chain_replay_ml.multi_model_studio import run_multi_model_studio

    require = tuple(
        p.strip().lower() for p in str(args.require or "").split(",") if p.strip()
    )

    def on_progress(ev: dict) -> None:
        logging.getLogger("mms").info("stage=%s", ev.get("stage"))

    result = run_multi_model_studio(
        data_dir=args.data_dir,
        model_a=args.model_a,
        model_b=args.model_b,
        require=require,
        progress=on_progress,
    )
    if args.json:
        print(
            json.dumps(
                {
                    "ok": result.ok,
                    "error": result.error,
                    "model_a": result.model_a,
                    "model_b": result.model_b,
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
            f"common={meta.get('common_count')} "
            f"only_a={meta.get('only_a_count')} only_b={meta.get('only_b_count')}"
        )
        if result.comparison:
            print(
                f"{'feature':<32} {'rkA':>5} {'rkB':>5} {'Δrk':>6} "
                f"{'riskA':>8} {'riskB':>8}"
            )
            for row in result.comparison[:20]:
                print(
                    f"{str(row.get('feature')):<32} "
                    f"{str(row.get('rank_gain_a') or '—'):>5} "
                    f"{str(row.get('rank_gain_b') or '—'):>5} "
                    f"{str(row.get('rank_gain_delta') if row.get('rank_gain_delta') is not None else '—'):>6} "
                    f"{str(row.get('risk_a') or '—'):>8} "
                    f"{str(row.get('risk_b') or '—'):>8}"
                )
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

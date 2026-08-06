"""CLI for Recommendation Engine → Experiment Planner artifacts.

Usage (from angelone/chart)::

  set PYTHONPATH=.
  python -m chain_replay_ml.recommendation_engine --data-dir data --model <Name>
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
    parser = argparse.ArgumentParser(
        description="Recommendation Engine (Experiment Planner artifacts)"
    )
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--require",
        default="",
        help="Comma list: importance,distribution,drift,diagnostics",
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    from chain_replay_ml.recommendation_engine import run_recommendation_engine

    require = tuple(
        p.strip().lower() for p in str(args.require or "").split(",") if p.strip()
    )

    def on_progress(ev: dict) -> None:
        logging.getLogger("rec").info("stage=%s", ev.get("stage"))

    result = run_recommendation_engine(
        data_dir=args.data_dir,
        model_name=args.model,
        require=require,
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
                    "summary": result.summary,
                    "suggestions_preview": result.suggestions[:15],
                },
                indent=2,
                default=str,
            )
        )
    else:
        print(f"ok={result.ok} error={result.error}")
        print(f"artifacts={result.artifacts_dir}")
        s = result.summary or {}
        print(
            f"suggestions={s.get('total_suggestions')} "
            f"high={s.get('high_priority')} med={s.get('medium_priority')} "
            f"low={s.get('low_priority')}"
        )
        for sug in result.suggestions[:10]:
            print(
                f"- [{sug.get('priority')}] {sug.get('category')}: {sug.get('title')} "
                f"(evidence={sug.get('evidence_score')})"
            )
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

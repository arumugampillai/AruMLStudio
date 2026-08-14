"""CLI: diagnose Analysis Dataset feature-source losses.

Usage (from ``apps`` with PYTHONPATH=.):

    python -m chain_replay_ml.dataset_builder.scripts.diagnose_feature_sources \\
        --data-dir D:/data/chart/data \\
        --dataset YOUR_DATASET_NAME \\
        --pipeline PL_0005
"""

from __future__ import annotations

import argparse
import json
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Diagnose feature source losses")
    parser.add_argument("--data-dir", required=True, help="Chart data directory (…/chart/data)")
    parser.add_argument("--dataset", default="", help="Dataset name for parquet/metadata compare")
    parser.add_argument("--pipeline", default="", help="Experimental pipeline id (e.g. PL_0005)")
    parser.add_argument("--json-out", default="", help="Optional path to write JSON report")
    args = parser.parse_args(argv)

    from chain_replay_ml.dataset_builder.feature_source_diagnostic import (
        diagnose_feature_sources,
        format_diagnostic_report,
    )

    report = diagnose_feature_sources(
        args.data_dir,
        dataset_name=args.dataset or None,
        pipeline_id=args.pipeline or None,
    )
    text = format_diagnostic_report(report)
    print(text)

    if args.json_out:
        payload = {
            "interval_summaries": [s.__dict__ for s in report.interval_summaries],
            "base_reason_counts": report.base_reason_counts,
            "other_reason_counts": report.other_reason_counts,
            "base_missing": [t.__dict__ for t in report.base_missing],
            "other_missing": [t.__dict__ for t in report.other_missing],
            "config_notes": report.config_notes,
        }
        with open(args.json_out, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, ensure_ascii=False)
            fh.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

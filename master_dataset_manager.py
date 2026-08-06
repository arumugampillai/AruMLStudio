#!/usr/bin/env python3
"""ML Research Studio — standalone Tkinter app (no chart server required).

Run from repo root:

    python master_dataset_manager.py
    python master_dataset_manager.py --chart-dir D:\\MyResearch\\project_data
    pythonw master_dataset_manager.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
APPS_DIR = ROOT / "apps"
if str(APPS_DIR) not in sys.path:
    sys.path.insert(0, str(APPS_DIR))

from master_dataset_tk.app import main


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ML Research Studio")
    parser.add_argument(
        "--chart-dir",
        dest="chart_dir",
        default=None,
        help="Project data folder (contains data/). Defaults to last-used or bundled apps/.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    main(chart_dir=args.chart_dir)

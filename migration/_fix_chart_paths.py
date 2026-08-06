"""One-off path fix after apps/ restructure (run from repo root)."""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "apps"

REPLACEMENTS: list[tuple[re.Pattern[str], str]] = [
    (
        re.compile(
            r"^_CHART_DIR = os\.path\.dirname\(os\.path\.dirname\(os\.path\.dirname\(os\.path\.abspath\(__file__\)\)\)\)\s*$",
            re.M,
        ),
        "from path_config import CHART_DATA_ROOT as _CHART_DIR",
    ),
    (
        re.compile(
            r"^_CHART_DIR = os\.path\.dirname\(os\.path\.dirname\(os\.path\.abspath\(__file__\)\)\)\s*$",
            re.M,
        ),
        "from path_config import CHART_DATA_ROOT as _CHART_DIR",
    ),
    (
        re.compile(
            r"^_CHART = os\.path\.dirname\(os\.path\.dirname\(os\.path\.dirname\(os\.path\.abspath\(__file__\)\)\)\)\s*$",
            re.M,
        ),
        "from path_config import CHART_DATA_ROOT as _CHART",
    ),
    (
        re.compile(
            r"^_CURR_DIR = os\.path\.dirname\(os\.path\.abspath\(__file__\)\)\s*\n_CHART_DIR = os\.path\.abspath\(os\.path\.join\(_CURR_DIR, \"\.\.\"\)\)\s*$",
            re.M,
        ),
        "from path_config import CHART_DATA_ROOT as _CHART_DIR",
    ),
]


def main() -> None:
    changed: list[str] = []
    for py in ROOT.rglob("*.py"):
        if "__pycache__" in py.parts:
            continue
        text = py.read_text(encoding="utf-8")
        orig = text
        for pat, repl in REPLACEMENTS:
            text = pat.sub(repl, text)
        if "from path_config import CHART_DATA_ROOT as _CHART_DIR" in text and "target_backfill" in py.name:
            pass
        if text != orig:
            py.write_text(text, encoding="utf-8")
            changed.append(str(py.relative_to(ROOT.parent)))
    print(f"Updated {len(changed)} files")
    for c in changed:
        print(c)


if __name__ == "__main__":
    main()

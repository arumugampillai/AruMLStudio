"""Fix sys.path bootstraps after apps/ restructure."""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "apps"

OLD_BLOCK = re.compile(
    r"if _CHART_DIR not in sys\.path:\s*\n\s*sys\.path\.insert\(0, _CHART_DIR\)\s*",
    re.M,
)
NEW_BLOCK = (
    "from path_config import ensure_ml_studio_paths\n"
    "ensure_ml_studio_paths()\n"
)

OLD_PROJ = re.compile(
    r"_PROJ_DIR = os\.path\.abspath\(os\.path\.join\(_CURR_DIR, \"\.\.\", \"\.\.\", \"\.\.\"\)\)\s*\n\s*"
    r"if _CHART_DIR not in sys\.path:\s*\n\s*sys\.path\.insert\(0, _CHART_DIR\)\s*"
    r"if _PROJ_DIR not in sys\.path:\s*\n\s*sys\.path\.insert\(0, _PROJ_DIR\)\s*",
    re.M,
)


def main() -> None:
    n = 0
    for py in ROOT.rglob("*.py"):
        if "__pycache__" in py.parts:
            continue
        text = py.read_text(encoding="utf-8")
        orig = text
        text = OLD_PROJ.sub(
            "from path_config import ensure_ml_studio_paths\nensure_ml_studio_paths()\n",
            text,
        )
        if "from path_config import CHART_DATA_ROOT as _CHART_DIR" in text and OLD_BLOCK.search(text):
            text = OLD_BLOCK.sub(NEW_BLOCK, text)
        if text != orig:
            py.write_text(text, encoding="utf-8")
            n += 1
            print(py.relative_to(ROOT.parent))
    print("fixed", n)


if __name__ == "__main__":
    main()

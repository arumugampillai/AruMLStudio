"""Resolve feature implementation paths and open them in the IDE."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


def chart_package_root() -> Path:
    """Return angelone/chart (parent of master_dataset_tk)."""
    return Path(__file__).resolve().parent.parent


def resolve_module_file(module_path: str) -> Path | None:
    rel = str(module_path or "").replace("\\", "/").strip().lstrip("/")
    if not rel:
        return None
    candidate = chart_package_root() / rel
    return candidate if candidate.is_file() else None


def _function_name(function_ref: str | None) -> str:
    return str(function_ref or "").replace("()", "").strip()


def find_source_line(
    path: Path,
    *,
    feature_name: str,
    function_ref: str | None = None,
) -> int:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return 1

    patterns = [
        f'"{feature_name}"',
        f"'{feature_name}'",
        f"{feature_name} =",
        f'"{feature_name}":',
        f"'{feature_name}':",
        f'"{feature_name}"',
    ]
    for idx, line in enumerate(lines, start=1):
        for pat in patterns:
            if pat in line:
                return idx

    fn = _function_name(function_ref)
    if fn:
        fn_re = re.compile(rf"^\s*def\s+{re.escape(fn)}\s*\(")
        for idx, line in enumerate(lines, start=1):
            if fn_re.match(line):
                return idx
    return 1


def resolve_source_location(
    *,
    feature_name: str,
    module_path: str,
    function_ref: str | None = None,
) -> dict[str, Any]:
    file_path = resolve_module_file(module_path)
    if file_path is None:
        return {
            "ok": False,
            "module": module_path,
            "function": function_ref or "",
            "error": f"File not found: {module_path}",
        }
    line = find_source_line(
        file_path,
        feature_name=feature_name,
        function_ref=function_ref,
    )
    return {
        "ok": True,
        "module": module_path,
        "function": function_ref or "",
        "path": str(file_path),
        "line": line,
        "label": f"{module_path}:{line}",
    }


def open_source_location(location: dict[str, Any]) -> bool:
    if not location.get("ok"):
        return False
    path = str(location.get("path") or "")
    line = int(location.get("line") or 1)
    if not path or not os.path.isfile(path):
        return False

    path = os.path.abspath(path)
    goto = f"{path}:{line}"

    if sys.platform == "win32":
        candidates = [
            ["cursor", "-g", goto],
            ["cursor.cmd", "-g", goto],
            ["code", "-g", goto],
            ["pycharm64.exe", f"--line={line}", path],
            ["pycharm.exe", f"--line={line}", path],
        ]
    elif sys.platform == "darwin":
        candidates = [
            ["cursor", "-g", goto],
            ["code", "-g", goto],
            ["open", "-a", "Cursor", goto],
            ["open", "-a", "PyCharm", path],
        ]
    else:
        candidates = [
            ["cursor", "-g", goto],
            ["code", "-g", goto],
            ["xdg-open", path],
        ]

    for cmd in candidates:
        exe = cmd[0]
        if shutil.which(exe) is None and not os.path.isfile(exe):
            continue
        try:
            subprocess.Popen(cmd, shell=False)  # noqa: S603
            return True
        except OSError:
            continue

    try:
        if sys.platform == "win32":
            os.startfile(path)  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            subprocess.run(["open", path], check=False)
        else:
            subprocess.run(["xdg-open", path], check=False)
        return True
    except OSError:
        return False

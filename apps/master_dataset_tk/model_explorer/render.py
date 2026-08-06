"""Official xgboost.to_graphviz rendering and Graphviz export."""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
from dataclasses import dataclass
from typing import Any


class RenderError(RuntimeError):
    """Raised when Graphviz rendering or export fails."""


@dataclass(frozen=True)
class GraphvizStatus:
    python_graphviz: bool
    system_dot: bool
    message: str

    @property
    def available(self) -> bool:
        return self.python_graphviz and self.system_dot


_WINDOWS_DOT_CANDIDATES = (
    r"C:\Program Files\Graphviz\bin\dot.exe",
    r"C:\Program Files (x86)\Graphviz\bin\dot.exe",
    os.path.expandvars(r"%LOCALAPPDATA%\Programs\Graphviz\bin\dot.exe"),
    os.path.expandvars(r"%ProgramFiles%\Graphviz\bin\dot.exe"),
    os.path.expandvars(r"%ProgramFiles(x86)%\Graphviz\bin\dot.exe"),
)


def _windows_install_hint() -> str:
    return (
        "Install Graphviz, then restart Studio. Windows: "
        "`winget install --id Graphviz.Graphviz -e --source winget` "
        "(or https://graphviz.org/download/). "
        "Ensure `C:\\Program Files\\Graphviz\\bin` is on PATH, or keep the "
        "default install location so Model Explorer can auto-discover `dot.exe`."
    )


def _prepend_dot_bin(dot_exe: str) -> None:
    """Expose ``dot`` to the Python graphviz package via PATH / GRAPHVIZ_DOT."""
    abs_dot = os.path.abspath(dot_exe)
    bin_dir = os.path.dirname(abs_dot)
    path = os.environ.get("PATH") or ""
    parts = [p for p in path.split(os.pathsep) if p]
    if bin_dir not in parts:
        os.environ["PATH"] = bin_dir + (os.pathsep + path if path else "")
    os.environ["GRAPHVIZ_DOT"] = abs_dot


def _ensure_dot_on_path() -> str | None:
    """Return absolute path to ``dot`` if found; prepend its bin dir to PATH when needed."""
    env_dot = (os.environ.get("GRAPHVIZ_DOT") or "").strip()
    if env_dot and os.path.isfile(env_dot):
        _prepend_dot_bin(env_dot)
        return os.path.abspath(env_dot)

    which = shutil.which("dot")
    if which:
        _prepend_dot_bin(which)
        return which

    if sys.platform.startswith("win"):
        for candidate in _WINDOWS_DOT_CANDIDATES:
            if candidate and os.path.isfile(candidate):
                _prepend_dot_bin(candidate)
                return os.path.abspath(candidate)
    return None


def check_graphviz() -> GraphvizStatus:
    """Report whether Python graphviz + system ``dot`` are usable."""
    py_ok = False
    try:
        import graphviz  # noqa: F401

        py_ok = True
    except ImportError:
        pass

    dot_path = _ensure_dot_on_path()
    sys_ok = bool(dot_path)

    if py_ok and sys_ok:
        msg = f"Graphviz ready (dot={dot_path})"
    elif not py_ok and not sys_ok:
        msg = (
            "Graphviz missing: install the Python package `graphviz` "
            "(pip install graphviz) and the system binary. "
            + _windows_install_hint()
        )
    elif not py_ok:
        msg = "Python package `graphviz` is not installed (pip install graphviz)."
    else:
        msg = (
            "System Graphviz binary `dot` not found on PATH. "
            + _windows_install_hint()
        )
    return GraphvizStatus(python_graphviz=py_ok, system_dot=sys_ok, message=msg)


def to_graphviz_source(
    booster: Any,
    tree_id: int = 0,
    *,
    rankdir: str = "TB",
    yes_color: str = "#0000FF",
    no_color: str = "#FF0000",
    **kwargs: Any,
) -> Any:
    """Return the official ``xgboost.to_graphviz`` Source object (no custom drawing)."""
    try:
        import xgboost as xgb
    except ImportError as exc:  # pragma: no cover
        raise RenderError("xgboost is not installed") from exc

    status = check_graphviz()
    if not status.python_graphviz:
        raise RenderError(status.message)

    try:
        return xgb.to_graphviz(
            booster,
            num_trees=int(tree_id),
            rankdir=rankdir,
            yes_color=yes_color,
            no_color=no_color,
            **kwargs,
        )
    except Exception as exc:
        raise RenderError(f"xgboost.to_graphviz failed: {exc}") from exc


def render_tree_png(
    booster: Any,
    tree_id: int = 0,
    *,
    out_dir: str | None = None,
    **kwargs: Any,
) -> str:
    """Render one tree to a PNG file via Graphviz; return absolute path."""
    status = check_graphviz()
    if not status.available:
        raise RenderError(status.message)

    source = to_graphviz_source(booster, tree_id, **kwargs)
    work = out_dir or tempfile.mkdtemp(prefix="xgb_explorer_")
    os.makedirs(work, exist_ok=True)
    base = os.path.join(work, f"tree_{int(tree_id)}")
    try:
        # graphviz.Source.render returns path without or with extension depending on version
        rendered = source.render(filename=base, format="png", cleanup=True)
    except Exception as exc:
        raise RenderError(
            f"Graphviz render failed (is `dot` on PATH?): {exc}"
        ) from exc
    path = str(rendered)
    if not path.lower().endswith(".png"):
        path = path + ".png"
    if not os.path.isfile(path):
        raise RenderError(f"Graphviz did not produce PNG at {path}")
    return os.path.abspath(path)


def export_tree(
    booster: Any,
    tree_id: int,
    dest_path: str,
    *,
    fmt: str | None = None,
    **kwargs: Any,
) -> str:
    """Export one tree to PNG, SVG, or PDF via Graphviz.

    ``fmt`` is inferred from ``dest_path`` extension when omitted.
    """
    status = check_graphviz()
    if not status.available:
        raise RenderError(status.message)

    dest = os.path.abspath(dest_path)
    ext = (fmt or os.path.splitext(dest)[1].lstrip(".")).lower()
    if ext not in ("png", "svg", "pdf"):
        raise RenderError(f"unsupported export format {ext!r}; use png, svg, or pdf")

    source = to_graphviz_source(booster, tree_id, **kwargs)
    out_dir = os.path.dirname(dest) or "."
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(dest))[0]
    base = os.path.join(out_dir, stem)
    try:
        rendered = source.render(filename=base, format=ext, cleanup=True)
    except Exception as exc:
        raise RenderError(f"Graphviz export failed: {exc}") from exc

    produced = str(rendered)
    if not produced.lower().endswith(f".{ext}"):
        produced = produced + f".{ext}"
    # Normalize to exact dest_path if Graphviz wrote sibling name
    if os.path.abspath(produced) != dest and os.path.isfile(produced):
        if os.path.isfile(dest):
            os.remove(dest)
        os.replace(produced, dest)
        return dest
    if not os.path.isfile(produced):
        raise RenderError(f"Graphviz did not produce {ext.upper()} at {produced}")
    return os.path.abspath(produced)

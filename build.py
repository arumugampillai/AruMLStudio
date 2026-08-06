#!/usr/bin/env python3
"""AruNeo — safety-first standalone build workflow for ML Research Studio.

=======================================================================
 SAFETY CONTRACT (read this before touching anything below)
=======================================================================
This script builds a standalone Windows EXE for the "ML Research Studio"
Tkinter app using PyInstaller. It is intentionally paranoid about what it
is allowed to touch on disk:

    ALLOWED (may be created / modified / deleted by this script):
        <project_root>/build/
        <project_root>/dist/
        <project_root>/releases/
        <project_root>/release_logs/

    EVERYTHING ELSE IS READ-ONLY, including but not limited to:
        - all project source code
        - databases (*.db), parquet files, datasets, models
        - application logs (logs/, angelone/logs, etc.)
        - the model/feature registry
        - configuration files (config/, *.json project config, etc.)

Every single filesystem *mutation* (mkdir / write / copy / delete/rmtree)
in this file goes through one of the small `guarded_*` helper functions
below, and every one of those helpers calls `assert_allowed()` first.
`assert_allowed()` is a pure, easily-unit-testable function that raises
`UnsafeOperationError` the instant a target path would land outside the
four allowed roots. There is no other way to touch disk in this script —
this is the load-bearing safety mechanism, treat it as such.

Reads (validation, discovery, dependency/dataset sanity checks) are not
restricted to those four folders — this script needs to *look* at project
source, databases, etc. to validate the project — but it must NEVER open
those files for writing, and never delete/rename/move them. Database
checks in particular are limited to `Path.exists()`, `os.access()` and
directory listings; a database file is never opened as a DB connection.

=======================================================================
 USAGE
=======================================================================
    python build.py                       # full pipeline (same as `build`)
    python build.py validate [--dry-run]
    python build.py clean    [--auto] [--dry-run]
    python build.py build    [--auto] [--dry-run] [--keep N]
    python build.py verify   [--dry-run] [--release PATH]

Flags:
    --auto      Do not ask for confirmation before deleting build/ or dist/.
    --dry-run   Print exactly what would happen. Creates/deletes/modifies
                nothing on disk (no exceptions, including log files).
    --keep N    Number of most-recent releases/ folders to retain
                (default 10). Only used by the `build` pipeline.
    --release   For `verify`: path to a specific release folder or .exe.
                Defaults to the most recent folder under releases/.

See build.bat for a double-click-friendly wrapper that forwards all CLI
arguments to `python build.py`.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

# ----------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------

# The only sub-folders of the project root this script is ever allowed to
# create, modify, or delete. See assert_allowed() below — this tuple is
# the single source of truth for the safety contract.
ALLOWED_SUBDIRS: tuple[str, ...] = ("build", "dist", "releases", "release_logs")

DEFAULT_KEEP_RELEASES = 10
SMOKE_TEST_WAIT_SECONDS = 10
DEFAULT_APP_NAME = "ML Research Studio"

# Common launcher filenames to look for at the project root, in priority
# order, BEFORE falling back to a generic content-based scan. This repo's
# actual entry point (master_dataset_manager.py) is listed first because
# discovery already found it by searching the codebase; the more generic
# names are kept for portability if this script is reused elsewhere.
KNOWN_ENTRY_NAMES: tuple[str, ...] = (
    "master_dataset_manager.py",
    "main.py",
    "app.py",
    "ml_research_studio.py",
    "studio.py",
)

# Folders to never descend into while scanning the repo for discovery
# artifacts (.spec, .ico, version.py, etc.) — keeps discovery fast and
# avoids ever "discovering" something inside our own build output.
_SCAN_IGNORE_DIRS = {
    "build", "dist", "releases", "release_logs",
    ".git", ".venv", "venv", "env", "__pycache__", "node_modules",
    "catboost_info",
}

# Extension point: if a real PyInstaller build fails at runtime with
# ModuleNotFoundError for a heavy/optional dependency (pandas, sklearn,
# xgboost, lightgbm, catboost, matplotlib, shap, optuna, ...), add the
# relevant `--collect-all <pkg>` / `--hidden-import <module>` flags here.
# PyInstaller usually resolves pure-Python imports automatically via
# static analysis (helped by --paths below); packages with C extensions
# or plugin-style dynamic loading sometimes need an explicit nudge.
#
# --additional-hooks-dir / --runtime-hook (pyi_hooks/) work around a
# PyInstaller + Python 3.12.0 frozen-importer bug where
# scipy.stats._distn_infrastructure crashes on import with
# "NameError: name 'obj' is not defined" (pyinstaller/pyinstaller#7992,
# fixed upstream in CPython 3.12.1). sklearn.metrics — used by
# chain_replay_ml/training/evaluator.py, reachable from the Create Dataset
# panel's import chain — pulls scipy.stats in transitively, so this
# previously crashed the packaged EXE at startup. See pyi_hooks/ for
# details. Keep these even after upgrading Python, unless you've confirmed
# the target build Python is >= 3.12.1 (or 3.11.x) and re-tested a cold
# frozen build.
#
# pyi_hooks/hook-xgboost.py and pyi_hooks/hook-catboost.py (picked up
# automatically via the same --additional-hooks-dir, by PyInstaller's
# hook-<module>.py naming convention) fix a separate packaging gap: both
# libraries ship compiled native libraries as package *data* rather than
# as importable extension modules, so PyInstaller's static import-graph
# analysis never discovers them and the frozen EXE fails at first use
# with "Cannot find XGBoost Library in the candidate path" (or the
# catboost equivalent). See those hook files for details.
_PYI_HOOKS_DIR = Path(__file__).resolve().parent / "pyi_hooks"
EXTRA_PYINSTALLER_ARGS: list[str] = [
    "--additional-hooks-dir", str(_PYI_HOOKS_DIR),
    "--runtime-hook", str(_PYI_HOOKS_DIR / "rth_scipy_distn_fix.py"),
]

# Windows defaults subprocess text mode to the locale codec (cp1252), but
# PyInstaller/git may emit UTF-8 or other non-cp1252 bytes. Always decode
# explicitly so log capture never raises UnicodeDecodeError.
_SUBPROCESS_TEXT_KWARGS: dict[str, object] = {
    "text": True,
    "encoding": "utf-8",
    "errors": "replace",
}


# ----------------------------------------------------------------------
# Exceptions
# ----------------------------------------------------------------------

class UnsafeOperationError(RuntimeError):
    """Raised when an operation would touch a path outside the allowed roots."""


class ValidationError(RuntimeError):
    """Raised when discovery/validation cannot proceed at all (fatal)."""


class BuildError(RuntimeError):
    """Raised when the PyInstaller build step fails or produces unexpected output."""


# ----------------------------------------------------------------------
# Logging — structured [INFO] / [WARN] / [ERROR] lines, optionally teed
# to a build log file under release_logs/ (itself opened via the same
# guarded helpers as everything else).
# ----------------------------------------------------------------------

class _TeeLogger:
    def __init__(self) -> None:
        self._fh = None

    def bind(self, path: Path, project_root: Path) -> None:
        assert_allowed(path, project_root)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = path.open("a", encoding="utf-8")

    def unbind(self) -> None:
        if self._fh is not None:
            try:
                self._fh.close()
            finally:
                self._fh = None

    def write_line(self, line: str) -> None:
        print(line)
        if self._fh is not None:
            self._fh.write(line + "\n")
            self._fh.flush()


_LOGGER = _TeeLogger()


def log_info(msg: str) -> None:
    _LOGGER.write_line(f"[INFO] {msg}")


def log_warn(msg: str) -> None:
    _LOGGER.write_line(f"[WARN] {msg}")


def log_error(msg: str) -> None:
    _LOGGER.write_line(f"[ERROR] {msg}")


# ----------------------------------------------------------------------
# Filesystem safety guard — the ONLY approved way to mutate disk.
# ----------------------------------------------------------------------

def is_path_allowed(path: Path, project_root: Path) -> bool:
    """Pure path-math check: is `path` inside one of the allowed roots?

    No filesystem I/O is performed (Path.resolve() only normalizes the
    path string; it does not require the path to exist). This makes the
    function trivially unit-testable.
    """
    root = project_root.resolve()
    try:
        resolved = path.resolve()
    except OSError:
        resolved = Path(os.path.normpath(str(path)))

    try:
        resolved.relative_to(root)
    except ValueError:
        return False  # Not even under the project root — definitely not allowed.

    for sub in ALLOWED_SUBDIRS:
        allowed_root = root / sub
        if resolved == allowed_root or allowed_root in resolved.parents:
            return True
    return False


def assert_allowed(path: Path, project_root: Path) -> None:
    if not is_path_allowed(path, project_root):
        allowed = ", ".join(f"{d}/" for d in ALLOWED_SUBDIRS)
        raise UnsafeOperationError(
            f"Refusing to touch '{path}': it is outside the allowed build "
            f"roots ({allowed}) under project root '{project_root}'. "
            "This is the safety guard working as intended — if you believe "
            "this path really should be writable, it does not belong in "
            "this build tool."
        )


def guarded_mkdir(path: Path, project_root: Path, *, dry_run: bool) -> None:
    assert_allowed(path, project_root)
    if dry_run:
        log_info(f"[DRY-RUN] Would create directory: {path}")
        return
    path.mkdir(parents=True, exist_ok=True)
    log_info(f"Created directory: {path}")


def guarded_rmtree(path: Path, project_root: Path, *, dry_run: bool) -> None:
    assert_allowed(path, project_root)
    if not path.exists():
        return
    if dry_run:
        log_info(f"[DRY-RUN] Would delete: {path}")
        return
    shutil.rmtree(path)
    log_info(f"Deleted: {path}")


def guarded_copy_file(src: Path, dst: Path, project_root: Path, *, dry_run: bool) -> None:
    assert_allowed(dst, project_root)
    if dry_run:
        log_info(f"[DRY-RUN] Would copy {src} -> {dst}")
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    log_info(f"Copied {src} -> {dst}")


def guarded_copytree_contents(src_dir: Path, dst_dir: Path, project_root: Path, *, dry_run: bool) -> None:
    """Copy the *contents* of src_dir into dst_dir (used for onedir PyInstaller output)."""
    assert_allowed(dst_dir, project_root)
    if dry_run:
        log_info(f"[DRY-RUN] Would copy contents of {src_dir} -> {dst_dir}")
        return
    dst_dir.mkdir(parents=True, exist_ok=True)
    for item in src_dir.iterdir():
        target = dst_dir / item.name
        if item.is_dir():
            shutil.copytree(item, target, dirs_exist_ok=True)
        else:
            shutil.copy2(item, target)
    log_info(f"Copied contents of {src_dir} -> {dst_dir}")


def guarded_write_text(path: Path, content: str, project_root: Path, *, dry_run: bool) -> None:
    assert_allowed(path, project_root)
    if dry_run:
        log_info(f"[DRY-RUN] Would write file: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    log_info(f"Wrote file: {path}")


# ----------------------------------------------------------------------
# Discovery — pathlib-only, no hard-coded absolute paths. Everything
# below reads files; it never writes anything.
# ----------------------------------------------------------------------

@dataclass
class ProjectLayout:
    root: Path
    entry: Path
    spec: Path | None
    icon: Path | None
    pathex_dirs: list[Path] = field(default_factory=list)
    app_package_dirs: list[Path] = field(default_factory=list)
    app_name: str = DEFAULT_APP_NAME


def discover_project_root(start: Path) -> Path:
    """Walk upward from `start` looking for a `.git` directory.

    Falls back to `start` itself (build.py's own folder) since this
    script is designed to live at the project root.
    """
    current = start.resolve()
    for candidate in (current, *current.parents):
        if (candidate / ".git").exists():
            return candidate
    return current


_ENTRY_SIGNAL_RE = re.compile(r"mainloop\(\)|tk\.Tk\(|Tk\(\)|ttk\.")


def discover_entry_script(root: Path) -> Path:
    for name in KNOWN_ENTRY_NAMES:
        candidate = root / name
        if candidate.is_file():
            return candidate

    for candidate in sorted(root.glob("run_*.py")):
        return candidate

    scored: list[tuple[int, Path]] = []
    for candidate in sorted(root.glob("*.py")):
        try:
            text = candidate.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        score = 0
        if "__main__" in text:
            score += 1
        if _ENTRY_SIGNAL_RE.search(text):
            score += 2
        if "ML Research Studio" in text:
            score += 5
        if score:
            scored.append((score, candidate))

    if scored:
        scored.sort(key=lambda item: (-item[0], item[1].name))
        return scored[0][1]

    raise ValidationError(
        "Could not auto-discover a Tkinter entry script at the project root.\n"
        "  Fix: place your launcher script (e.g. main.py) at the project "
        "root, or add its filename to KNOWN_ENTRY_NAMES near the top of build.py."
    )


def discover_spec_file(root: Path) -> Path | None:
    matches: list[Path] = []
    for spec_path in root.rglob("*.spec"):
        rel_parts = spec_path.relative_to(root).parts[:-1]
        if any(part in _SCAN_IGNORE_DIRS for part in rel_parts):
            continue
        matches.append(spec_path)
    if not matches:
        return None
    if len(matches) > 1:
        log_warn(f"Multiple .spec files found; using {matches[0]} (others: {matches[1:]})")
    return matches[0]


def discover_icon(root: Path, spec: Path | None) -> Path | None:
    if spec is not None:
        try:
            text = spec.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            text = ""
        m = re.search(r"icon\s*=\s*\[?['\"]([^'\"]+\.ico)['\"]", text)
        if m:
            candidate = (spec.parent / m.group(1)).resolve()
            if candidate.is_file():
                return candidate

    matches: list[Path] = []
    for ico_path in root.rglob("*.ico"):
        rel_parts = ico_path.relative_to(root).parts[:-1]
        if any(part in _SCAN_IGNORE_DIRS for part in rel_parts):
            continue
        matches.append(ico_path)
    return matches[0] if matches else None


# --- Pathex / import-folder discovery -----------------------------------
#
# This repo's launcher builds its import path dynamically, e.g.:
#
#     ROOT = Path(__file__).resolve().parent
#     CHART_DIR = ROOT / "angelone" / "chart"
#     sys.path.insert(0, str(CHART_DIR))
#
# Rather than hard-coding the literal "angelone/chart" string, we do a
# small best-effort static parse of that pattern so discovery keeps
# working if the layout ever changes.

_ROOT_ASSIGN_RE = re.compile(
    r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*Path\(__file__\)\.resolve\(\)((?:\.parent)*)\s*$"
)
_JOIN_ASSIGN_RE = re.compile(
    r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*([A-Za-z_][A-Za-z0-9_]*)((?:\s*/\s*['\"][^'\"]+['\"])+)\s*$"
)
_SEGMENT_RE = re.compile(r"['\"]([^'\"]+)['\"]")
_SYS_PATH_INSERT_RE = re.compile(
    r"sys\.path\.insert\(\s*\d+\s*,\s*(?:str\()?([A-Za-z_][A-Za-z0-9_]*)\)?\s*\)"
)
_IMPORT_RE = re.compile(r"^\s*(?:from|import)\s+([A-Za-z_][A-Za-z0-9_]*)", re.MULTILINE)
_VERSION_ATTR_RE = re.compile(r"__version__\s*=\s*['\"]([^'\"]+)['\"]")
_TITLE_RE = re.compile(r"\.title\(\s*f?['\"]([^'\"{]+)['\"]")


def _parse_path_variables(entry_text: str, entry_dir: Path) -> dict[str, Path]:
    variables: dict[str, Path] = {}
    for raw_line in entry_text.splitlines():
        line = raw_line.strip()

        m_root = _ROOT_ASSIGN_RE.match(line)
        if m_root:
            var_name, parents = m_root.group(1), m_root.group(2)
            base = entry_dir
            for _ in range(parents.count(".parent") - 1):
                base = base.parent
            variables[var_name] = base
            continue

        m_join = _JOIN_ASSIGN_RE.match(line)
        if m_join:
            var_name, base_var, join_tail = m_join.group(1), m_join.group(2), m_join.group(3)
            if base_var in variables:
                segments = _SEGMENT_RE.findall(join_tail)
                variables[var_name] = variables[base_var].joinpath(*segments)
    return variables


def discover_pathex_dirs(entry: Path) -> list[Path]:
    text = entry.read_text(encoding="utf-8", errors="ignore")
    variables = _parse_path_variables(text, entry.parent)

    dirs: list[Path] = []
    for match in _SYS_PATH_INSERT_RE.finditer(text):
        var_name = match.group(1)
        if var_name in variables:
            dirs.append(variables[var_name])

    if entry.parent not in dirs:
        dirs.insert(0, entry.parent)

    seen: set[Path] = set()
    unique: list[Path] = []
    for d in dirs:
        if d not in seen:
            seen.add(d)
            unique.append(d)
    return unique


def discover_app_package_dirs(entry: Path, pathex_dirs: list[Path]) -> list[Path]:
    text = entry.read_text(encoding="utf-8", errors="ignore")
    imported_names = set(_IMPORT_RE.findall(text))

    package_dirs: list[Path] = []
    for pathex in pathex_dirs:
        for name in imported_names:
            candidate = pathex / name
            if candidate.is_dir() and (candidate / "__init__.py").is_file():
                package_dirs.append(candidate)
    return package_dirs


def _iter_py_files(dirs: list[Path], limit: int | None = None):
    count = 0
    for d in dirs:
        for py_file in d.rglob("*.py"):
            yield py_file
            count += 1
            if limit is not None and count >= limit:
                return


def discover_app_name(app_package_dirs: list[Path]) -> str:
    for py_file in _iter_py_files(app_package_dirs, limit=500):
        try:
            text = py_file.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        m = _TITLE_RE.search(text)
        if m:
            name = m.group(1).strip()
            if name:
                return name
    return DEFAULT_APP_NAME


def discover_layout(root: Path) -> ProjectLayout:
    entry = discover_entry_script(root)
    spec = discover_spec_file(root)
    icon = discover_icon(root, spec)
    pathex_dirs = discover_pathex_dirs(entry)
    app_package_dirs = discover_app_package_dirs(entry, pathex_dirs)
    app_name = discover_app_name(app_package_dirs) if app_package_dirs else DEFAULT_APP_NAME
    return ProjectLayout(
        root=root,
        entry=entry,
        spec=spec,
        icon=icon,
        pathex_dirs=pathex_dirs,
        app_package_dirs=app_package_dirs,
        app_name=app_name,
    )


# ----------------------------------------------------------------------
# Version resolution: __version__ -> version.py -> git tag -> git hash
# -> "Development Build"
# ----------------------------------------------------------------------

def _run_git(root: Path, args: list[str]) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args], cwd=root, capture_output=True, timeout=10,
            **_SUBPROCESS_TEXT_KWARGS,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    value = result.stdout.strip()
    return value or None


def _iter_version_py(root: Path):
    for candidate in root.rglob("version.py"):
        rel_parts = candidate.relative_to(root).parts[:-1]
        if any(part in _SCAN_IGNORE_DIRS for part in rel_parts):
            continue
        yield candidate


def resolve_app_version(layout: ProjectLayout) -> str:
    candidates = [layout.entry, *_iter_py_files(layout.app_package_dirs, limit=500)]
    for path in candidates:
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        m = _VERSION_ATTR_RE.search(text)
        if m:
            log_info(f"Resolved app version from {path.name}: {m.group(1)}")
            return m.group(1)

    for version_file in _iter_version_py(layout.root):
        try:
            text = version_file.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        m = _VERSION_ATTR_RE.search(text)
        if m:
            log_info(f"Resolved app version from {version_file}: {m.group(1)}")
            return m.group(1)

    tag = _run_git(layout.root, ["describe", "--tags"])
    if tag:
        log_info(f"Resolved app version from git tag: {tag}")
        return tag

    commit = _run_git(layout.root, ["rev-parse", "--short", "HEAD"])
    if commit:
        log_info(f"Resolved app version from git commit hash: {commit}")
        return commit

    log_warn("No version signal found (no __version__, version.py, or git repo); using 'Development Build'.")
    return "Development Build"


def get_git_commit(root: Path) -> str | None:
    return _run_git(root, ["rev-parse", "HEAD"])


def get_pyinstaller_version() -> str | None:
    try:
        result = subprocess.run(
            [sys.executable, "-m", "PyInstaller", "--version"],
            capture_output=True, timeout=30, **_SUBPROCESS_TEXT_KWARGS,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return (result.stdout.strip() or result.stderr.strip()) or None


# ----------------------------------------------------------------------
# Validate (read-only)
# ----------------------------------------------------------------------

def _check_dependency_sanity(layout: ProjectLayout) -> None:
    """Best-effort, side-effect-free import sanity check.

    Uses importlib.util.find_spec() (module lookup only — this does not
    execute module code) on third-party names referenced by the app's
    own package(s). Missing modules are reported as warnings, not hard
    failures, since not every referenced name is necessarily required
    for a successful freeze.
    """
    stdlib = getattr(sys, "stdlib_module_names", frozenset())
    local_names = {p.name for p in layout.app_package_dirs} | {layout.entry.stem}

    imported: set[str] = set()
    for py_file in _iter_py_files(layout.app_package_dirs, limit=500):
        try:
            text = py_file.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        imported.update(_IMPORT_RE.findall(text))

    third_party = sorted(
        name for name in imported
        if name not in stdlib and name not in local_names and not name.startswith("_")
    )

    missing: list[str] = []
    for name in third_party:
        try:
            found = importlib.util.find_spec(name)
        except (ImportError, ValueError, ModuleNotFoundError):
            found = None
        if found is None:
            missing.append(name)

    if missing:
        log_warn(
            "Possibly missing/unimportable third-party module(s) referenced by the app: "
            + ", ".join(missing)
            + " (the PyInstaller build may fail — install missing packages first)."
        )
    else:
        log_info(f"Dependency sanity check OK ({len(third_party)} third-party module(s) importable).")


# Packages whose compiled native library is shipped as package *data*
# rather than as a normal importable extension module — PyInstaller's
# static import-graph analysis can silently miss these unless a
# pyi_hooks/hook-<package>.py collects them explicitly (see hook-xgboost.py
# / hook-catboost.py). Mapping is package name -> glob patterns for the
# native library file(s), searched relative to the installed package dir.
_NATIVE_LIB_GLOBS: dict[str, tuple[str, ...]] = {
    "xgboost": ("lib/xgboost.dll", "lib/libxgboost.so", "lib/libxgboost.dylib"),
    "lightgbm": ("lib_lightgbm.dll", "lib_lightgbm.so", "lib_lightgbm.dylib"),
    "catboost": ("_catboost*.pyd", "_catboost*.so"),
}


def _iter_reachable_py_files(pathex_dirs: list[Path], limit: int = 4000):
    """Yields .py files under any of `pathex_dirs`, skipping ignored dirs.

    Broader than `_iter_py_files(app_package_dirs)` — this follows the same
    roots PyInstaller's own analysis starts from (`--paths`), so it also
    reaches packages the entry script only imports *transitively* (e.g.
    chain_replay_ml, imported by master_dataset_tk rather than directly by
    the launcher script).
    """
    count = 0
    for pathex in pathex_dirs:
        for py_file in pathex.rglob("*.py"):
            rel_parts = py_file.relative_to(pathex).parts[:-1]
            if any(part in _SCAN_IGNORE_DIRS for part in rel_parts):
                continue
            yield py_file
            count += 1
            if count >= limit:
                return


def _check_native_libraries(layout: ProjectLayout) -> None:
    """Best-effort, read-only check that ML packages' native libs exist.

    This only checks packages that are (a) actually installed in the
    current interpreter and (b) referenced somewhere in the app's own
    source tree (including transitively-imported packages reachable via
    `--paths`, not just the launcher's direct imports) — it never fails
    validation outright, since a missing native lib for a package the app
    doesn't even import isn't this build's problem. The goal is to surface
    a missing/renamed .dll/.so *before* a multi-minute PyInstaller build,
    instead of discovering it only after launching the frozen EXE.
    """
    imported: set[str] = set()
    for py_file in _iter_reachable_py_files(layout.pathex_dirs):
        try:
            text = py_file.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        imported.update(_IMPORT_RE.findall(text))

    for package, patterns in _NATIVE_LIB_GLOBS.items():
        if package not in imported:
            continue

        try:
            spec = importlib.util.find_spec(package)
        except (ImportError, ValueError, ModuleNotFoundError):
            spec = None
        if spec is None or not spec.submodule_search_locations:
            log_warn(
                f"{package} is imported by the app but is not installed in this "
                f"interpreter ({sys.executable}) — install it before building, or "
                "the frozen EXE will fail as soon as it's used."
            )
            continue

        pkg_dir = Path(next(iter(spec.submodule_search_locations)))
        found = next((m for pattern in patterns for m in pkg_dir.glob(pattern)), None)
        if found is not None:
            log_info(f"{package} native library found: {found}")
        else:
            log_warn(
                f"{package} is installed but its native library "
                f"({' / '.join(patterns)}) was not found under {pkg_dir} — "
                "the frozen EXE may fail at runtime. Re-check the pip install, "
                "or that pyi_hooks/hook-" + package + ".py still matches the "
                "installed layout."
            )


# Sidecar data files loaded at runtime via Path(__file__).with_name(...)
# (or equivalent) that must both (a) physically exist next to their .py
# module and (b) be collected into the frozen EXE by a matching
# pyi_hooks/hook-<package>.py — otherwise the packaged app crashes with
# FileNotFoundError the first time that code path runs, even though it
# works fine from a normal checkout. See pyi_hooks/hook-chain_replay_ml.py
# for the full story (this originally broke the OHLC Aggregation
# transformation's ohlc_history_profiles.json). Mapping is package name ->
# relative paths (from the package's own directory) of the files it needs.
_CRITICAL_PACKAGE_DATA_FILES: dict[str, tuple[str, ...]] = {
    "chain_replay_ml": (
        "dataset_builder/transformations/ohlc_history_profiles.json",
        "dataset_builder/transformations/horizon_policy.json",
    ),
}


def _check_packaged_data_files(layout: ProjectLayout) -> None:
    """Best-effort, read-only check for the __file__-sibling-data-file class
    of packaging bug.

    Never fails validation outright — a missing hook or data file is a
    packaging gap to fix, not a reason to block every build — but it
    surfaces the exact issue *before* a multi-minute PyInstaller build
    instead of only after launching the frozen EXE and hitting the crash.
    """
    for package, rel_paths in _CRITICAL_PACKAGE_DATA_FILES.items():
        pkg_dir = next(
            (pathex / package for pathex in layout.pathex_dirs if (pathex / package).is_dir()),
            None,
        )
        if pkg_dir is None:
            continue  # This package isn't part of the current build; nothing to check.

        hook_path = _PYI_HOOKS_DIR / f"hook-{package}.py"
        if not hook_path.is_file():
            log_warn(
                f"{package} ships sidecar data file(s) loaded via Path(__file__) "
                f"but no {hook_path.name} was found under {_PYI_HOOKS_DIR} — "
                "the frozen EXE will likely fail with FileNotFoundError the "
                "first time that code path runs. Add a PyInstaller collection "
                "hook (see pyi_hooks/hook-chain_replay_ml.py for the pattern)."
            )
            continue

        for rel_path in rel_paths:
            data_path = pkg_dir / rel_path
            if data_path.is_file():
                log_info(f"Packaged data file OK: {package}/{rel_path}")
            else:
                log_warn(
                    f"Expected sidecar data file missing on disk: {data_path} — "
                    f"{package} code that loads it via Path(__file__) will raise "
                    "FileNotFoundError (in the frozen EXE, and in a normal "
                    "checkout alike)."
                )


def _check_datasets(layout: ProjectLayout) -> None:
    """STRICTLY READ-ONLY dataset/database checks.

    Only Path.exists(), os.access(), and directory listing are used —
    databases are never opened as DB connections, never vacuumed,
    migrated, moved, or modified in any way.
    """
    for pathex in layout.pathex_dirs:
        data_dir = pathex / "data"
        if not data_dir.exists():
            log_warn(f"No data directory at {data_dir} (created automatically on first run).")
            continue
        if not os.access(data_dir, os.R_OK):
            log_warn(f"Data directory exists but is not readable: {data_dir}")
            continue
        db_files = list(data_dir.rglob("*.db"))
        parquet_files = list(data_dir.rglob("*.parquet"))
        log_info(
            f"Data directory OK (read-only check): {data_dir} "
            f"({len(db_files)} .db file(s), {len(parquet_files)} .parquet file(s))"
        )
        if not db_files and not parquet_files:
            log_warn(f"Data directory {data_dir} has no .db/.parquet files yet (fine for a fresh setup).")


def run_validate(layout: ProjectLayout) -> list[str]:
    """Runs all validation checks; returns a list of hard-failure messages (empty == pass)."""
    failures: list[str] = []

    log_info(f"Project root detected: {layout.root}")
    log_info(f"Entry script detected: {layout.entry.relative_to(layout.root)}")

    if not layout.entry.is_file():
        failures.append(
            f"Entry script missing: {layout.entry}\n"
            "  Fix: restore the launcher script, or update KNOWN_ENTRY_NAMES in build.py."
        )

    if layout.spec is not None:
        log_info(f"PyInstaller spec detected: {layout.spec.relative_to(layout.root)} (will be used for the build)")
    else:
        log_info("No PyInstaller .spec found — build.py will generate PyInstaller CLI arguments automatically.")

    if layout.icon is not None:
        log_info(f"Icon detected: {layout.icon.relative_to(layout.root)}")
    else:
        log_warn("No .ico icon found — the build will proceed without a custom application icon.")

    if not layout.app_package_dirs:
        failures.append(
            "Could not resolve the application's first-party package directory.\n"
            "  Fix: ensure the entry script adds the correct folder to sys.path before "
            "importing its app package (see master_dataset_manager.py for the expected pattern)."
        )
    for pkg_dir in layout.app_package_dirs:
        if pkg_dir.is_dir():
            log_info(f"Application package OK: {pkg_dir.relative_to(layout.root)}")
        else:
            failures.append(f"Required application package folder missing: {pkg_dir}")

    for pathex in layout.pathex_dirs:
        if pathex.is_dir():
            label = "." if pathex == layout.root else str(pathex.relative_to(layout.root))
            log_info(f"Required import path OK: {label}")
        else:
            failures.append(f"Required import path missing: {pathex}")

    log_info(f"App name resolved: {layout.app_name}")

    pyi_version = get_pyinstaller_version()
    if pyi_version is None:
        failures.append(
            "PyInstaller is not installed/importable in this Python environment.\n"
            f"  Fix: \"{sys.executable}\" -m pip install pyinstaller pyinstaller-hooks-contrib"
        )
    else:
        log_info(f"PyInstaller available: {pyi_version}")

    log_info(f"Python interpreter: {sys.executable} ({platform.python_version()})")

    _check_dependency_sanity(layout)
    _check_native_libraries(layout)
    _check_packaged_data_files(layout)
    _check_datasets(layout)

    if failures:
        log_error(f"Validation FAILED with {len(failures)} issue(s):")
        for i, msg in enumerate(failures, start=1):
            log_error(f"  {i}. {msg}")
    else:
        log_info("Validation PASSED.")
    return failures


# ----------------------------------------------------------------------
# Clean
# ----------------------------------------------------------------------

def _dir_size_human(path: Path) -> str:
    total = 0
    try:
        for f in path.rglob("*"):
            if f.is_file():
                try:
                    total += f.stat().st_size
                except OSError:
                    pass
    except OSError:
        pass
    size = float(total)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024 or unit == "TB":
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


def run_clean(root: Path, *, auto: bool, dry_run: bool) -> bool:
    """Deletes ONLY build/ and dist/ under the project root. Nothing else, ever."""
    targets = [root / "build", root / "dist"]
    existing = [t for t in targets if t.exists()]

    if not existing:
        log_info("Nothing to clean: build/ and dist/ do not exist.")
        return True

    log_info("The following paths will be removed:")
    for t in existing:
        log_info(f"  - {t}  ({_dir_size_human(t)})")

    if dry_run:
        log_info("[DRY-RUN] No files were deleted.")
        return True

    if not auto:
        answer = input("Proceed with deleting the paths listed above? [y/N]: ").strip().lower()
        if answer not in ("y", "yes"):
            log_warn("Clean aborted by user — no changes made.")
            return False

    for t in existing:
        guarded_rmtree(t, root, dry_run=False)
    log_info("Clean complete.")
    return True


# ----------------------------------------------------------------------
# Build (PyInstaller invocation + release packaging)
# ----------------------------------------------------------------------

def _pyinstaller_command(layout: ProjectLayout) -> list[str]:
    root = layout.root
    cmd = [
        sys.executable, "-m", "PyInstaller", "--noconfirm",
        "--distpath", str(root / "dist"),
        "--workpath", str(root / "build"),
        "--specpath", str(root / "build"),
    ]
    if layout.spec is not None:
        # An existing .spec already encodes name/icon/onefile/hidden-imports —
        # respect it as-is and only redirect output into the allowed folders.
        cmd.append(str(layout.spec))
        return cmd

    cmd += [str(layout.entry), "--name", layout.app_name, "--onefile", "--windowed"]
    if layout.icon is not None:
        cmd += ["--icon", str(layout.icon)]
    for pathex in layout.pathex_dirs:
        cmd += ["--paths", str(pathex)]
    cmd += EXTRA_PYINSTALLER_ARGS
    return cmd


def _log_planned_pyinstaller_command(layout: ProjectLayout) -> None:
    cmd = _pyinstaller_command(layout)
    rendered = " ".join(f'"{c}"' if " " in c else c for c in cmd)
    log_info("[DRY-RUN] Planned PyInstaller command:")
    log_info(f"  {rendered}")


def _run_pyinstaller(layout: ProjectLayout) -> int:
    cmd = _pyinstaller_command(layout)
    log_info("Running: " + " ".join(cmd))
    process = subprocess.Popen(
        cmd, cwd=layout.root, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        bufsize=1, **_SUBPROCESS_TEXT_KWARGS,
    )
    assert process.stdout is not None
    for line in process.stdout:
        _LOGGER.write_line(f"[PYI] {line.rstrip()}")
    process.wait()
    return process.returncode


def locate_built_exe(dist_dir: Path, app_name: str) -> Path:
    onefile_candidate = dist_dir / f"{app_name}.exe"
    if onefile_candidate.is_file():
        return onefile_candidate

    onedir_candidate = dist_dir / app_name / f"{app_name}.exe"
    if onedir_candidate.is_file():
        return onedir_candidate

    matches = list(dist_dir.rglob("*.exe")) if dist_dir.exists() else []
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise BuildError(f"No .exe found under {dist_dir} after PyInstaller build.")
    raise BuildError(
        f"Multiple .exe files found under {dist_dir}; cannot determine which to release: {matches}"
    )


def copy_to_release(
    exe_path: Path, dist_dir: Path, releases_root: Path, project_root: Path, *, dry_run: bool,
) -> tuple[Path, Path]:
    stamp = datetime.now().strftime("%Y-%m-%d_%H%M")
    release_dir = releases_root / stamp
    suffix = 1
    while release_dir.exists():
        suffix += 1
        release_dir = releases_root / f"{stamp}_{suffix}"

    guarded_mkdir(release_dir, project_root, dry_run=dry_run)

    is_onedir = exe_path.parent != dist_dir
    if is_onedir:
        guarded_copytree_contents(exe_path.parent, release_dir, project_root, dry_run=dry_run)
        dest_exe = release_dir / exe_path.name
    else:
        dest_exe = release_dir / exe_path.name
        guarded_copy_file(exe_path, dest_exe, project_root, dry_run=dry_run)

    return release_dir, dest_exe


def write_release_info(
    release_dir: Path, *, project_root: Path, layout: ProjectLayout, app_version: str,
    pyinstaller_version: str, build_started: datetime, build_ended: datetime, dry_run: bool,
) -> dict:
    info = {
        "app_name": layout.app_name,
        "app_version": app_version,
        "git_commit": get_git_commit(project_root),
        "build_time_local": build_ended.isoformat(timespec="seconds"),
        "build_duration_seconds": round((build_ended - build_started).total_seconds(), 1),
        "python_version": platform.python_version(),
        "pyinstaller_version": pyinstaller_version,
        "entry_script": str(layout.entry.relative_to(project_root)),
        "spec_file": str(layout.spec.relative_to(project_root)) if layout.spec else None,
        "platform": platform.platform(),
    }
    path = release_dir / "release_info.json"
    guarded_write_text(path, json.dumps(info, indent=2), project_root, dry_run=dry_run)
    return info


def prune_old_releases(releases_root: Path, keep: int, project_root: Path, *, dry_run: bool) -> None:
    """Deletes the oldest release folders beyond `keep`, ONLY inside releases/."""
    if not releases_root.exists() or keep <= 0:
        return
    dirs = sorted((d for d in releases_root.iterdir() if d.is_dir()), key=lambda d: d.name)
    excess = dirs[:-keep] if len(dirs) > keep else []
    for old in excess:
        log_info(f"Pruning old release (keeping latest {keep}): {old}")
        guarded_rmtree(old, project_root, dry_run=dry_run)


def _print_summary(*, success: bool, exe_path: Path | None, started_at: datetime, log_path: Path, dry_run: bool) -> None:
    duration = datetime.now() - started_at
    log_info("=" * 64)
    log_info("BUILD SUMMARY" + (" (DRY-RUN — nothing was created/modified)" if dry_run else ""))
    log_info(f"  Result:   {'SUCCESS' if success else 'FAILED'}")
    if exe_path is not None:
        log_info(f"  EXE:      {exe_path}")
    log_info(f"  Duration: {duration}")
    log_info(f"  Log:      {log_path}")
    log_info("=" * 64)


def run_build(layout: ProjectLayout, *, auto: bool, dry_run: bool, keep: int) -> bool:
    """Full pipeline: Validate -> Clean -> Build -> Verify. Stops on first failure."""
    root = layout.root
    started_at = datetime.now()
    log_path = root / "release_logs" / f"build_{started_at:%Y%m%d_%H%M%S}.log"

    if not dry_run:
        guarded_mkdir(root / "release_logs", root, dry_run=False)
        _LOGGER.bind(log_path, root)
        log_info(f"Build log: {log_path}")
    else:
        log_info(f"[DRY-RUN] Build log would be written to: {log_path}")

    try:
        log_info("=== Stage 1/4: Validate ===")
        failures = run_validate(layout)
        if failures:
            log_error("Aborting build: validation failed. Fix the issues above and re-run.")
            return False

        log_info("=== Stage 2/4: Clean ===")
        if not run_clean(root, auto=auto, dry_run=dry_run):
            log_error("Aborting build: clean step was not completed.")
            return False

        log_info("=== Stage 3/4: Build (PyInstaller) ===")
        if dry_run:
            _log_planned_pyinstaller_command(layout)
            log_info("[DRY-RUN] Skipping actual PyInstaller invocation and release packaging.")
            log_info("=== Stage 4/4: Verify ===")
            log_info("[DRY-RUN] Skipping verify — no EXE was produced.")
            _print_summary(success=True, exe_path=None, started_at=started_at, log_path=log_path, dry_run=True)
            return True

        pyi_version = get_pyinstaller_version() or "unknown"
        exit_code = _run_pyinstaller(layout)
        if exit_code != 0:
            log_error(f"PyInstaller exited with code {exit_code}. See log for details: {log_path}")
            return False

        exe_path = locate_built_exe(root / "dist", layout.app_name)
        log_info(f"Build produced: {exe_path}")

        app_version = resolve_app_version(layout)
        release_dir, dest_exe = copy_to_release(
            exe_path, root / "dist", root / "releases", root, dry_run=False,
        )
        log_info(f"Release copied to: {release_dir}")

        ended_at = datetime.now()
        write_release_info(
            release_dir, project_root=root, layout=layout, app_version=app_version,
            pyinstaller_version=pyi_version, build_started=started_at, build_ended=ended_at,
            dry_run=False,
        )

        prune_old_releases(root / "releases", keep, root, dry_run=False)

        log_info("=== Stage 4/4: Verify ===")
        verify_ok = run_verify(dest_exe, root)
        _print_summary(success=verify_ok, exe_path=dest_exe, started_at=started_at, log_path=log_path, dry_run=False)
        if not verify_ok:
            log_error("Build succeeded but the smoke test failed. Investigate before distributing this release.")
        return verify_ok
    finally:
        _LOGGER.unbind()


# ----------------------------------------------------------------------
# Verify
# ----------------------------------------------------------------------

def find_latest_release(releases_root: Path) -> Path | None:
    if not releases_root.exists():
        return None
    dirs = sorted((d for d in releases_root.iterdir() if d.is_dir()), key=lambda d: d.name)
    return dirs[-1] if dirs else None


def resolve_verify_target(root: Path, release_arg: str | None) -> Path | None:
    if release_arg:
        candidate = Path(release_arg)
        if not candidate.is_absolute():
            candidate = root / candidate
        if candidate.is_file() and candidate.suffix.lower() == ".exe":
            return candidate
        if candidate.is_dir():
            matches = list(candidate.rglob("*.exe"))
            if len(matches) == 1:
                return matches[0]
            if not matches:
                log_error(f"No .exe found under: {candidate}")
                return None
            log_error(f"Multiple .exe files found under {candidate}; specify the exact file with --release.")
            return None
        log_error(f"--release path not found: {candidate}")
        return None

    latest = find_latest_release(root / "releases")
    if latest is None:
        log_error(
            "No releases found under releases/.\n"
            "  Fix: run `python build.py build` first, or pass --release to point at a specific EXE/folder."
        )
        return None
    matches = list(latest.rglob("*.exe"))
    if not matches:
        log_error(f"No .exe found in latest release folder: {latest}")
        return None
    if len(matches) > 1:
        log_warn(f"Multiple .exe files found in {latest}; using {matches[0]}")
    return matches[0]


def run_verify(exe_path: Path, project_root: Path, *, dry_run: bool = False) -> bool:
    """Three-stage smoke test: exists+size -> launch+wait 10s -> terminate cleanly."""
    log_info(f"=== Verify: {exe_path} ===")

    if not exe_path.is_file():
        log_error(f"Stage 1/3 FAILED — EXE not found: {exe_path}")
        return False
    size_mb = exe_path.stat().st_size / (1024 * 1024)
    log_info(f"Stage 1/3 PASSED — EXE exists ({size_mb:.1f} MB): {exe_path}")

    if dry_run:
        log_info("[DRY-RUN] Skipping launch/terminate stages — nothing was run.")
        return True

    smoke_log_dir = project_root / "release_logs"
    guarded_mkdir(smoke_log_dir, project_root, dry_run=False)
    smoke_log_path = smoke_log_dir / f"verify_{datetime.now():%Y%m%d_%H%M%S}.log"
    log_info(f"Launching for a {SMOKE_TEST_WAIT_SECONDS}s smoke test, output captured to: {smoke_log_path}")

    with smoke_log_path.open("w", encoding="utf-8") as smoke_fh:
        try:
            process = subprocess.Popen(
                [str(exe_path)], cwd=exe_path.parent, stdout=smoke_fh, stderr=subprocess.STDOUT,
            )
        except OSError as exc:
            log_error(f"Stage 2/3 FAILED — could not launch EXE: {exc}")
            return False

        time.sleep(SMOKE_TEST_WAIT_SECONDS)
        return_code = process.poll()

        if return_code is not None:
            log_error(
                f"Stage 2/3 FAILED — process exited within {SMOKE_TEST_WAIT_SECONDS}s "
                f"(exit code {return_code}). See {smoke_log_path} for output."
            )
            return False
        log_info(f"Stage 2/3 PASSED — process still running after {SMOKE_TEST_WAIT_SECONDS}s.")

        log_info("Stage 3/3 — terminating smoke-test process...")
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            log_warn("Process did not exit after terminate(); killing it.")
            process.kill()
            process.wait(timeout=5)

    log_info("Stage 3/3 PASSED — process terminated cleanly.")
    return True


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

COMMANDS = {"validate", "clean", "build", "verify"}


def _build_arg_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--auto", action="store_true",
        help="Do not prompt for confirmation before deleting build/ or dist/.",
    )
    common.add_argument(
        "--dry-run", action="store_true",
        help="Print exactly what would happen; create/delete/modify nothing.",
    )

    parser = argparse.ArgumentParser(
        prog="build.py",
        description="AruNeo ML Research Studio — safety-first standalone build workflow.",
    )
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("validate", parents=[common], help="Read-only project validation.")
    sub.add_parser("clean", parents=[common], help="Delete build/ and dist/ only.")

    p_build = sub.add_parser("build", parents=[common], help="Full pipeline: validate -> clean -> build -> verify.")
    p_build.add_argument(
        "--keep", type=int, default=DEFAULT_KEEP_RELEASES,
        help=f"Number of most-recent releases/ folders to keep (default {DEFAULT_KEEP_RELEASES}).",
    )

    p_verify = sub.add_parser("verify", parents=[common], help="Smoke-test the latest (or specified) release EXE.")
    p_verify.add_argument(
        "--release", default=None,
        help="Path to a release folder or .exe to verify. Defaults to the latest release.",
    )

    return parser


def cmd_validate(args: argparse.Namespace) -> int:
    root = discover_project_root(Path(__file__).parent)
    layout = discover_layout(root)
    failures = run_validate(layout)
    return 0 if not failures else 1


def cmd_clean(args: argparse.Namespace) -> int:
    root = discover_project_root(Path(__file__).parent)
    ok = run_clean(root, auto=args.auto, dry_run=args.dry_run)
    return 0 if ok else 1


def cmd_build(args: argparse.Namespace) -> int:
    root = discover_project_root(Path(__file__).parent)
    layout = discover_layout(root)
    ok = run_build(layout, auto=args.auto, dry_run=args.dry_run, keep=args.keep)
    return 0 if ok else 1


def cmd_verify(args: argparse.Namespace) -> int:
    root = discover_project_root(Path(__file__).parent)
    exe_path = resolve_verify_target(root, args.release)
    if exe_path is None:
        return 1
    ok = run_verify(exe_path, root, dry_run=args.dry_run)
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] not in COMMANDS:
        # Default: `python build.py [flags...]` runs the full Build Release pipeline.
        argv = ["build", *argv]

    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    dispatch = {
        "validate": cmd_validate,
        "clean": cmd_clean,
        "build": cmd_build,
        "verify": cmd_verify,
    }

    try:
        return dispatch[args.command](args)
    except UnsafeOperationError as exc:
        log_error(f"SAFETY GUARD TRIPPED: {exc}")
        return 1
    except (ValidationError, BuildError) as exc:
        log_error(str(exc))
        return 1
    except KeyboardInterrupt:
        log_warn("Interrupted by user.")
        return 130


if __name__ == "__main__":
    sys.exit(main())

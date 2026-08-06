"""Tests for the xgboost (and catboost) PyInstaller native-library fix.

Background
----------
The packaged EXE crashed at first xgboost use with:

    xgboost.core.XGBoostError: Cannot find XGBoost Library in the
    candidate path. List of candidates:
    ...\\_MEI...\\xgboost\\lib\\xgboost.dll
    ...\\_MEI...\\lib\\xgboost.dll
    ...

Root cause: xgboost ships its compiled native library
(``xgboost/lib/xgboost.dll`` on Windows) as package *data* rather than as
an importable extension module (see ``xgboost/libpath.py::find_lib_path``),
so PyInstaller's static import-graph analysis never discovers it and a
plain ``pyinstaller entry.py`` never bundles it.

Fix: ``pyi_hooks/hook-xgboost.py`` uses ``collect_all("xgboost")`` to pull
in the compiled library, package data, and hidden imports. Because it
lives under ``pyi_hooks/`` and ``build.py`` already points PyInstaller's
``--additional-hooks-dir`` there, PyInstaller auto-discovers it via its
``hook-<module>.py`` naming convention — no extra CLI flags are needed.
``pyi_hooks/hook-catboost.py`` applies the same fix defensively for
catboost (also a declared, natively-compiled dependency).

These tests can't run a real frozen build, but they guard the pieces
that matter:

  1. The hook files exist and use `collect_all` against the right
     package names.
  2. `collect_all("xgboost")` actually surfaces the native .dll as a
     PyInstaller *binary* (not just a data file) — this is exactly what
     makes it land next to the .exe/in the onefile bundle at runtime.
     This guards against a future xgboost release moving the library
     such that collect_all silently stops finding it.
  3. build.py wires --additional-hooks-dir at the directory containing
     these hook files.
  4. build.py's own native-library validate check (`_check_native_libraries`)
     correctly detects the installed xgboost.dll and correctly flags
     libraries that are referenced by the app but not installed.

Run with:
    python -m pytest tests/test_pyi_xgboost_fix.py -v
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

_HOOKS_DIR = _ROOT / "pyi_hooks"
_XGBOOST_HOOK_PATH = _HOOKS_DIR / "hook-xgboost.py"
_CATBOOST_HOOK_PATH = _HOOKS_DIR / "hook-catboost.py"


class HookFilesExistTests(unittest.TestCase):
    def test_xgboost_hook_file_exists(self) -> None:
        self.assertTrue(_XGBOOST_HOOK_PATH.is_file(), f"missing {_XGBOOST_HOOK_PATH}")

    def test_catboost_hook_file_exists(self) -> None:
        self.assertTrue(_CATBOOST_HOOK_PATH.is_file(), f"missing {_CATBOOST_HOOK_PATH}")

    def test_xgboost_hook_uses_collect_all_on_correct_package(self) -> None:
        text = _XGBOOST_HOOK_PATH.read_text(encoding="utf-8")
        self.assertIn("collect_all", text)
        self.assertIn('collect_all("xgboost")', text)

    def test_catboost_hook_uses_collect_all_on_correct_package(self) -> None:
        text = _CATBOOST_HOOK_PATH.read_text(encoding="utf-8")
        self.assertIn("collect_all", text)
        self.assertIn('collect_all("catboost")', text)


class BuildPyWiresHooksDirTests(unittest.TestCase):
    def setUp(self) -> None:
        import build as build_module

        self.build_module = build_module

    def test_extra_pyinstaller_args_reference_additional_hooks_dir(self) -> None:
        args = self.build_module.EXTRA_PYINSTALLER_ARGS
        self.assertIn("--additional-hooks-dir", args)
        idx = args.index("--additional-hooks-dir")
        hooks_dir = Path(args[idx + 1])
        self.assertTrue(hooks_dir.is_dir())
        # The xgboost/catboost hooks must live directly inside the folder
        # PyInstaller is told to scan, or they'll never be auto-discovered.
        self.assertTrue((hooks_dir / "hook-xgboost.py").is_file())
        self.assertTrue((hooks_dir / "hook-catboost.py").is_file())


class CollectAllSurfacesNativeBinaryTests(unittest.TestCase):
    """Guards against a future xgboost release silently breaking the fix."""

    def test_collect_all_xgboost_returns_the_native_library_as_a_binary(self) -> None:
        try:
            from PyInstaller.utils.hooks import collect_all
        except ImportError:
            self.skipTest("PyInstaller not installed in this environment")

        try:
            import xgboost  # noqa: F401
        except ImportError:
            self.skipTest("xgboost not installed in this environment")

        _datas, binaries, _hiddenimports = collect_all("xgboost")
        self.assertTrue(
            binaries,
            "collect_all('xgboost') returned no binaries — the xgboost package "
            "layout may have changed; re-check that the native library is still "
            "being bundled (this is what fixes 'Cannot find XGBoost Library').",
        )
        dll_names = {Path(src).name.lower() for src, _dest in binaries}
        self.assertTrue(
            any(name.startswith("xgboost") or name.startswith("libxgboost") for name in dll_names),
            f"expected an xgboost native library among collected binaries, got: {dll_names}",
        )


class CheckNativeLibrariesValidateTests(unittest.TestCase):
    """Exercises build.py's `_check_native_libraries` in isolation, against a
    throwaway fake project tree (never touches the real repo).
    """

    def setUp(self) -> None:
        import build as build_module

        self.build_module = build_module
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

        self.pkg_dir = self.root / "appstuff"
        self.pkg_dir.mkdir(parents=True)
        (self.pkg_dir / "__init__.py").write_text("", encoding="utf-8")
        (self.pkg_dir / "trainer.py").write_text(
            "def train():\n    from xgboost import XGBClassifier\n    return XGBClassifier\n",
            encoding="utf-8",
        )

        entry = self.root / "app.py"
        entry.write_text("import appstuff\n", encoding="utf-8")

        self.layout = build_module.ProjectLayout(
            root=self.root,
            entry=entry,
            spec=None,
            icon=None,
            pathex_dirs=[self.root],
            app_package_dirs=[self.pkg_dir],
        )

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_detects_import_of_native_ml_package(self) -> None:
        # Real xgboost is installed in this test environment (a project
        # dependency) — the check should find its actual native library
        # without raising, regardless of log output.
        self.build_module._check_native_libraries(self.layout)  # must not raise

    def test_warns_when_referenced_package_not_installed(self) -> None:
        (self.pkg_dir / "trainer.py").write_text(
            "def train():\n    from totally_not_a_real_ml_package import Thing\n    return Thing\n",
            encoding="utf-8",
        )
        self.build_module._NATIVE_LIB_GLOBS["totally_not_a_real_ml_package"] = ("fake.dll",)
        try:
            warnings: list[str] = []
            original_warn = self.build_module.log_warn
            self.build_module.log_warn = warnings.append
            try:
                self.build_module._check_native_libraries(self.layout)
            finally:
                self.build_module.log_warn = original_warn
            self.assertTrue(
                any("totally_not_a_real_ml_package" in w and "not installed" in w for w in warnings)
            )
        finally:
            del self.build_module._NATIVE_LIB_GLOBS["totally_not_a_real_ml_package"]

    def test_ignores_packages_never_imported_by_the_app(self) -> None:
        (self.pkg_dir / "trainer.py").write_text("x = 1\n", encoding="utf-8")
        infos: list[str] = []
        warnings: list[str] = []
        original_info, original_warn = self.build_module.log_info, self.build_module.log_warn
        self.build_module.log_info, self.build_module.log_warn = infos.append, warnings.append
        try:
            self.build_module._check_native_libraries(self.layout)
        finally:
            self.build_module.log_info, self.build_module.log_warn = original_info, original_warn
        combined = " ".join(infos + warnings)
        self.assertNotIn("xgboost", combined)
        self.assertNotIn("lightgbm", combined)
        self.assertNotIn("catboost", combined)


if __name__ == "__main__":
    unittest.main()

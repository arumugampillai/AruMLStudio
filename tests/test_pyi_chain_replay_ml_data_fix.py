"""Tests for the chain_replay_ml packaged-JSON-sidecar PyInstaller fix.

Background
----------
The packaged EXE crashed as soon as the OHLC Aggregation transformation
loaded its history profiles, with:

    FileNotFoundError: ...\\_MEI...\\chain_replay_ml\\dataset_builder\\
    transformations\\ohlc_history_profiles.json

Root cause: ``dataset_builder/transformations/ohlc_history_profiles.py``
(and ``horizon_policy.py``) load a JSON config file that ships as a
sibling of their ``.py`` source via ``Path(__file__).with_name(...)``.
PyInstaller's static import-graph analysis only follows Python imports,
so a plain ``pyinstaller entry.py`` never bundled these JSON files, and
the frozen module's ``__file__`` under ``_MEIPASS`` had no matching JSON
sibling on disk.

Fix: ``pyi_hooks/hook-chain_replay_ml.py`` uses a scoped
``collect_data_files("chain_replay_ml", subdir="dataset_builder/transformations",
includes=["*.json"])`` to bundle exactly those two config files (and any
future JSON added to that same folder) — deliberately narrower than
``collect_data_files("chain_replay_ml")`` / ``--collect-data
chain_replay_ml``, which would also sweep in generated/machine-local/
sensitive files elsewhere in the package (e.g. a live
``data/replay_session.db``). ``build.py``'s ``_check_packaged_data_files``
validate step also warns before a build if either file goes missing or
the hook itself disappears.

These tests can't run a real frozen build, but they guard the pieces
that matter:

  1. The hook file exists and actually surfaces both known config files
     as PyInstaller data entries with the destination path that keeps
     them siblings of their .py module inside the frozen bundle.
  2. It does NOT sweep in unrelated/sensitive files from elsewhere in the
     package (regression test for the "blanket collect" mistake).
  3. build.py's ``_check_packaged_data_files`` validate step detects both
     the present-and-correct case and the missing-file / missing-hook
     cases.

Run with:
    python -m pytest tests/test_pyi_chain_replay_ml_data_fix.py -v
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

_APPS_DIR = _ROOT / "apps"
if str(_APPS_DIR) not in sys.path:
    sys.path.insert(0, str(_APPS_DIR))

_HOOKS_DIR = _ROOT / "pyi_hooks"
_HOOK_PATH = _HOOKS_DIR / "hook-chain_replay_ml.py"

_EXPECTED_REL_PATHS = (
    "dataset_builder/transformations/ohlc_history_profiles.json",
    "dataset_builder/transformations/horizon_policy.json",
)


class HookFileExistsTests(unittest.TestCase):
    def test_hook_file_exists(self) -> None:
        self.assertTrue(_HOOK_PATH.is_file(), f"missing {_HOOK_PATH}")

    def test_hook_uses_collect_data_files_scoped_to_transformations(self) -> None:
        text = _HOOK_PATH.read_text(encoding="utf-8")
        self.assertIn("collect_data_files", text)
        self.assertIn('"chain_replay_ml"', text)
        self.assertIn("dataset_builder/transformations", text)


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
        self.assertTrue((hooks_dir / "hook-chain_replay_ml.py").is_file())


class CollectDataFilesFindsExpectedConfigsTests(unittest.TestCase):
    """Actually invokes the same PyInstaller helper the hook uses, against
    the real chain_replay_ml package in this repo.
    """

    def setUp(self) -> None:
        try:
            from PyInstaller.utils.hooks import collect_data_files
        except ImportError:
            self.skipTest("PyInstaller not installed in this environment")
        try:
            import chain_replay_ml  # noqa: F401
        except ImportError as exc:
            self.skipTest(f"chain_replay_ml not importable: {exc}")
        self.collect_data_files = collect_data_files

    def test_collects_both_known_config_files(self) -> None:
        datas = self.collect_data_files(
            "chain_replay_ml",
            subdir="dataset_builder/transformations",
            includes=["*.json"],
        )
        collected_names = {Path(src).name for src, _dest in datas}
        self.assertIn("ohlc_history_profiles.json", collected_names)
        self.assertIn("horizon_policy.json", collected_names)

    def test_destination_keeps_json_as_sibling_of_its_module(self) -> None:
        """The (src, dest) tuple's dest must be the package-relative directory
        so PyInstaller re-creates ``chain_replay_ml/dataset_builder/
        transformations/<file>.json`` next to the extracted
        ``ohlc_history_profiles.py`` — exactly what ``Path(__file__).with_name``
        needs to find it in the frozen bundle.
        """
        datas = self.collect_data_files(
            "chain_replay_ml",
            subdir="dataset_builder/transformations",
            includes=["*.json"],
        )
        by_name = {Path(src).name: dest for src, dest in datas}
        for name in ("ohlc_history_profiles.json", "horizon_policy.json"):
            dest_parts = Path(by_name[name]).parts
            self.assertEqual(
                dest_parts[-3:],
                ("chain_replay_ml", "dataset_builder", "transformations"),
                f"unexpected destination for {name}: {by_name[name]}",
            )

    def test_does_not_sweep_in_unrelated_or_sensitive_files(self) -> None:
        """Regression test for the blanket-collect mistake: a plain
        ``collect_data_files("chain_replay_ml")`` (no subdir/includes) also
        bundles generated/machine-local/sensitive files that must never
        ship in the EXE, e.g. a live replay-session SQLite database. The
        scoped call the hook actually uses must not include any of those.
        """
        datas = self.collect_data_files(
            "chain_replay_ml",
            subdir="dataset_builder/transformations",
            includes=["*.json"],
        )
        collected_names = {Path(src).name for src, _dest in datas}
        self.assertEqual(collected_names, {"ohlc_history_profiles.json", "horizon_policy.json"})
        for src, _dest in datas:
            self.assertNotIn("replay_session", src)
            self.assertNotIn("__pycache__", src)


class CheckPackagedDataFilesValidateTests(unittest.TestCase):
    """Exercises build.py's `_check_packaged_data_files` in isolation,
    against a throwaway fake project tree (never touches the real repo).
    """

    def setUp(self) -> None:
        import build as build_module

        self.build_module = build_module
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

        self.pkg_dir = self.root / "chain_replay_ml"
        self.transforms_dir = self.pkg_dir / "dataset_builder" / "transformations"
        self.transforms_dir.mkdir(parents=True)
        (self.pkg_dir / "__init__.py").write_text("", encoding="utf-8")

        self.hooks_dir = self.root / "pyi_hooks"
        self.hooks_dir.mkdir(parents=True)
        (self.hooks_dir / "hook-chain_replay_ml.py").write_text(
            "datas = []\n", encoding="utf-8"
        )

        entry = self.root / "app.py"
        entry.write_text("import chain_replay_ml\n", encoding="utf-8")

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

    def _write_expected_json_files(self) -> None:
        for rel_path in _EXPECTED_REL_PATHS:
            (self.root / "chain_replay_ml" / rel_path).write_text("{}", encoding="utf-8")

    def test_passes_silently_when_files_and_hook_present(self) -> None:
        self._write_expected_json_files()
        with self._patched_hooks_dir():
            warnings: list[str] = []
            original_warn = self.build_module.log_warn
            self.build_module.log_warn = warnings.append
            try:
                self.build_module._check_packaged_data_files(self.layout)
            finally:
                self.build_module.log_warn = original_warn
        self.assertEqual(warnings, [])

    def test_warns_when_a_critical_json_file_is_missing(self) -> None:
        # Only write one of the two expected files.
        (self.transforms_dir / "ohlc_history_profiles.json").write_text("{}", encoding="utf-8")
        with self._patched_hooks_dir():
            warnings: list[str] = []
            original_warn = self.build_module.log_warn
            self.build_module.log_warn = warnings.append
            try:
                self.build_module._check_packaged_data_files(self.layout)
            finally:
                self.build_module.log_warn = original_warn
        self.assertTrue(
            any("horizon_policy.json" in w and "missing" in w for w in warnings),
            f"expected a missing-file warning, got: {warnings}",
        )

    def test_warns_when_hook_file_is_missing(self) -> None:
        self._write_expected_json_files()
        (self.hooks_dir / "hook-chain_replay_ml.py").unlink()
        with self._patched_hooks_dir():
            warnings: list[str] = []
            original_warn = self.build_module.log_warn
            self.build_module.log_warn = warnings.append
            try:
                self.build_module._check_packaged_data_files(self.layout)
            finally:
                self.build_module.log_warn = original_warn
        self.assertTrue(
            any("hook-chain_replay_ml.py" in w for w in warnings),
            f"expected a missing-hook warning, got: {warnings}",
        )

    def _patched_hooks_dir(self):
        return _patch_module_attr(self.build_module, "_PYI_HOOKS_DIR", self.hooks_dir)


class _patch_module_attr:
    """Small context manager to temporarily override a module attribute."""

    def __init__(self, module, name: str, value) -> None:
        self._module = module
        self._name = name
        self._value = value

    def __enter__(self) -> None:
        self._original = getattr(self._module, self._name)
        setattr(self._module, self._name, self._value)

    def __exit__(self, *exc_info: object) -> None:
        setattr(self._module, self._name, self._original)


if __name__ == "__main__":
    unittest.main()

"""Tests for the scipy.stats._distn_infrastructure PyInstaller crash fix.

Background
----------
The packaged EXE crashed at startup with:

    Failed to execute script 'master_dataset_manager' due to unhandled
    exception: name 'obj' is not defined

Root cause: scipy/stats/_distn_infrastructure.py ends with a module-level
docstring-cleanup loop (`for obj in [...]: ... / del obj`) that, under
PyInstaller's frozen importer on Python 3.12.0, evaluates the comprehension
to an empty list, leaving `obj` unbound before the trailing `del obj`
(pyinstaller/pyinstaller#7992). This app pulls scipy.stats in transitively
via `sklearn.metrics` (chain_replay_ml/training/evaluator.py), which is
reached from master_dataset_tk/create_dataset_panel.py's import chain.

Fix: pyi_hooks/hook-scipy.stats._distn_infrastructure.py forces the .py
source to be bundled, and pyi_hooks/rth_scipy_distn_fix.py (registered as a
PyInstaller runtime hook via build.py's EXTRA_PYINSTALLER_ARGS) source-patches
the unsafe `del obj` before the module executes.

These tests can't reproduce PyInstaller's frozen-importer bug directly (that
requires an actual frozen build), but they guard the pieces that matter:

  1. The runtime hook's source-patch logic transforms the known-bad pattern
     correctly.
  2. The installed scipy version still contains the exact pattern the patch
     targets (so a scipy upgrade that changes this code doesn't silently
     make the patch a no-op).
  3. build.py actually wires the hook directory + runtime hook into the
     PyInstaller invocation.
  4. The real import chain that surfaces the bug (create_dataset_panel ->
     build_service -> ... -> evaluator -> sklearn.metrics -> scipy.stats)
     still imports cleanly under a normal (non-frozen) interpreter.

Run with:
    python -m pytest tests/test_pyi_scipy_distn_fix.py -v
"""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

_APPS_DIR = _ROOT / "apps"
if str(_APPS_DIR) not in sys.path:
    sys.path.insert(0, str(_APPS_DIR))

_HOOKS_DIR = _ROOT / "pyi_hooks"
_RTH_PATH = _HOOKS_DIR / "rth_scipy_distn_fix.py"
_HOOK_FILE_PATH = _HOOKS_DIR / "hook-scipy.stats._distn_infrastructure.py"


def _load_module_from_path(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class RuntimeHookFilesExistTests(unittest.TestCase):
    def test_runtime_hook_file_exists(self) -> None:
        self.assertTrue(_RTH_PATH.is_file(), f"missing {_RTH_PATH}")

    def test_collection_hook_file_exists(self) -> None:
        self.assertTrue(_HOOK_FILE_PATH.is_file(), f"missing {_HOOK_FILE_PATH}")

    def test_collection_hook_forces_py_source_bundling(self) -> None:
        text = _HOOK_FILE_PATH.read_text(encoding="utf-8")
        self.assertIn('module_collection_mode = "pyz+py"', text)


class PatchSourceLogicTests(unittest.TestCase):
    """Loads rth_scipy_distn_fix.py in isolation (no sys.meta_path pollution
    needed — we only exercise the pure `_patch_source` function).
    """

    @classmethod
    def setUpClass(cls) -> None:
        # Loading the module installs a meta_path finder as a side effect
        # (harmless — it only intercepts scipy.stats._distn_infrastructure,
        # and delegates to whatever finder would have handled it anyway).
        cls.hook_module = _load_module_from_path("_rth_scipy_distn_fix_under_test", _RTH_PATH)

    @classmethod
    def tearDownClass(cls) -> None:
        # Remove the finder instance(s) this test installed so repeated test
        # runs / other tests don't accumulate duplicate finders.
        sys.meta_path[:] = [
            f for f in sys.meta_path
            if type(f).__module__ != cls.hook_module.__name__
        ]

    def test_patches_known_bad_pattern(self) -> None:
        source = (
            "docdict_discrete['default'] = _doc_default_disc\n"
            "\n"
            "# clean up all the separate docstring elements\n"
            "for obj in [s for s in dir() if s.startswith('_doc_')]:\n"
            "    exec('del ' + obj)\n"
            "del obj\n"
            "\n"
            "\n"
            "def _moment(data, n, mu=None):\n"
            "    pass\n"
        )
        patched = self.hook_module._patch_source(source)
        self.assertNotIn("\ndel obj\n", patched)
        self.assertIn("globals().pop('obj', None)", patched)
        # Patched source must still be valid, executable Python.
        compile(patched, "<patched-scipy-stub>", "exec")

    def test_leaves_unrelated_source_untouched(self) -> None:
        source = "x = 1\ny = 2\n"
        self.assertEqual(self.hook_module._patch_source(source), source)

    def test_patched_module_survives_obj_unbound(self) -> None:
        """Simulates the exact frozen-importer failure mode: the for-loop
        body never runs (empty iterable), so `obj` is never bound. The
        patched `globals().pop('obj', None)` must not raise, unlike the
        original `del obj`.
        """
        source = (
            "for obj in []:\n"
            "    pass\n"
            "del obj\n"
        )
        with self.assertRaises(NameError):
            exec(compile(source, "<unpatched>", "exec"), {})

        patched = self.hook_module._patch_source(source)
        namespace: dict = {}
        exec(compile(patched, "<patched>", "exec"), namespace)  # must not raise


class InstalledScipyStillMatchesPatchTargetTests(unittest.TestCase):
    """Guards against a scipy upgrade silently making the patch a no-op."""

    def test_installed_scipy_source_contains_targeted_pattern(self) -> None:
        try:
            import scipy.stats._distn_infrastructure as distn_mod
        except ImportError:
            self.skipTest("scipy not installed in this environment")

        source_path = Path(distn_mod.__file__)
        source = source_path.read_text(encoding="utf-8")
        self.assertIn(
            "\ndel obj\n",
            source,
            "scipy.stats._distn_infrastructure no longer contains the "
            "targeted 'del obj' pattern — the PyInstaller runtime hook "
            "patch may now be a silent no-op; re-check whether the "
            "NameError still reproduces in a frozen build before removing "
            "the workaround.",
        )


class BuildPyWiresHooksTests(unittest.TestCase):
    def setUp(self) -> None:
        import build as build_module

        self.build_module = build_module

    def test_extra_pyinstaller_args_reference_hooks_dir(self) -> None:
        args = self.build_module.EXTRA_PYINSTALLER_ARGS
        self.assertIn("--additional-hooks-dir", args)
        self.assertIn("--runtime-hook", args)

    def test_referenced_hook_paths_exist_on_disk(self) -> None:
        args = self.build_module.EXTRA_PYINSTALLER_ARGS
        for flag, value in zip(args, args[1:]):
            if flag in ("--additional-hooks-dir", "--runtime-hook"):
                self.assertTrue(Path(value).exists(), f"{flag} target missing: {value}")


class ImportChainSmokeTest(unittest.TestCase):
    """Sanity check for the exact import chain that surfaces the bug under
    PyInstaller: create_dataset_panel -> build_service ->
    chain_replay_ml.dataset_builder.dataset_pipeline ->
    chain_replay_ml.training.orchestrator -> .evaluator -> sklearn.metrics
    -> scipy.stats. Under a normal interpreter this must import cleanly;
    the bug only manifests inside a frozen build.
    """

    def test_evaluator_module_imports_sklearn_and_scipy_cleanly(self) -> None:
        try:
            import chain_replay_ml.training.evaluator  # noqa: F401
        except ImportError as exc:
            self.skipTest(f"optional ML dependency chain unavailable: {exc}")

    def test_build_service_imports_cleanly(self) -> None:
        try:
            import master_dataset_tk.build_service  # noqa: F401
        except ImportError as exc:
            self.skipTest(f"optional ML dependency chain unavailable: {exc}")


if __name__ == "__main__":
    unittest.main()

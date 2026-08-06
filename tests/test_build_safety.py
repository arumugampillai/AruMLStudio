"""Unit tests for build.py's safety guard and dry-run behavior.

These tests intentionally avoid invoking PyInstaller or touching the real
project tree — everything runs against a throwaway temp directory. They
only exercise:

  1. The allowed-path guard (`is_path_allowed` / `assert_allowed`) that
     rejects any write/delete target outside build/, dist/, releases/,
     release_logs/.
  2. That --dry-run mode creates/deletes/modifies nothing on disk.

Run with:
    python -m unittest tests.test_build_safety -v
or:
    python -m pytest tests/test_build_safety.py -v
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import build as build_module  # noqa: E402


class AllowedPathGuardTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.project_root = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_allows_writes_inside_each_allowed_root(self) -> None:
        for sub in ("build", "dist", "releases", "release_logs"):
            target = self.project_root / sub / "nested" / "thing.txt"
            self.assertTrue(build_module.is_path_allowed(target, self.project_root), msg=sub)

    def test_rejects_writes_to_project_source(self) -> None:
        target = self.project_root / "angelone" / "chart" / "master_dataset_tk" / "app.py"
        self.assertFalse(build_module.is_path_allowed(target, self.project_root))

    def test_rejects_writes_to_project_root_itself(self) -> None:
        target = self.project_root / "some_new_file.txt"
        self.assertFalse(build_module.is_path_allowed(target, self.project_root))

    def test_rejects_writes_to_a_lookalike_sibling_folder(self) -> None:
        # "releases_backup" starts with "releases" but is NOT the releases/
        # folder itself — must not be treated as allowed.
        target = self.project_root / "releases_backup" / "thing.txt"
        self.assertFalse(build_module.is_path_allowed(target, self.project_root))

    def test_rejects_writes_outside_project_root_entirely(self) -> None:
        target = self.project_root.parent / "outside.txt"
        self.assertFalse(build_module.is_path_allowed(target, self.project_root))

    def test_assert_allowed_raises_for_disallowed_path(self) -> None:
        target = self.project_root / "data" / "master_dataset.db"
        with self.assertRaises(build_module.UnsafeOperationError):
            build_module.assert_allowed(target, self.project_root)

    def test_assert_allowed_passes_silently_for_allowed_path(self) -> None:
        target = self.project_root / "releases" / "2026-01-01_0000" / "App.exe"
        build_module.assert_allowed(target, self.project_root)  # must not raise

    def test_guarded_rmtree_refuses_disallowed_target_and_leaves_it_untouched(self) -> None:
        target = self.project_root / "config"
        target.mkdir(parents=True)
        (target / "cred.py").write_text("SECRET = 1")
        with self.assertRaises(build_module.UnsafeOperationError):
            build_module.guarded_rmtree(target, self.project_root, dry_run=False)
        self.assertTrue((target / "cred.py").exists())

    def test_guarded_write_text_refuses_disallowed_target(self) -> None:
        target = self.project_root / "angelone" / "chart" / "data" / "hacked.txt"
        with self.assertRaises(build_module.UnsafeOperationError):
            build_module.guarded_write_text(target, "oops", self.project_root, dry_run=False)
        self.assertFalse(target.exists())


class DryRunCreatesNothingTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.project_root = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_guarded_mkdir_dry_run_creates_no_directory(self) -> None:
        target = self.project_root / "build" / "work"
        build_module.guarded_mkdir(target, self.project_root, dry_run=True)
        self.assertFalse(target.exists())

    def test_guarded_write_text_dry_run_creates_no_file(self) -> None:
        target = self.project_root / "releases" / "2026-01-01_0000" / "release_info.json"
        build_module.guarded_write_text(target, "{}", self.project_root, dry_run=True)
        self.assertFalse(target.exists())
        self.assertFalse(target.parent.exists())

    def test_guarded_copy_file_dry_run_creates_no_file(self) -> None:
        src = self.project_root / "dist" / "App.exe"
        src.parent.mkdir(parents=True)
        src.write_bytes(b"fake-exe-bytes")
        dst = self.project_root / "releases" / "2026-01-01_0000" / "App.exe"
        build_module.guarded_copy_file(src, dst, self.project_root, dry_run=True)
        self.assertFalse(dst.exists())

    def test_guarded_rmtree_dry_run_does_not_delete(self) -> None:
        target = self.project_root / "build"
        target.mkdir(parents=True)
        (target / "marker.txt").write_text("keep me")
        build_module.guarded_rmtree(target, self.project_root, dry_run=True)
        self.assertTrue((target / "marker.txt").exists())

    def test_run_clean_dry_run_leaves_build_and_dist_untouched(self) -> None:
        for sub in ("build", "dist"):
            d = self.project_root / sub
            d.mkdir(parents=True)
            (d / "marker.txt").write_text("keep me")

        result = build_module.run_clean(self.project_root, auto=True, dry_run=True)

        self.assertTrue(result)
        for sub in ("build", "dist"):
            self.assertTrue((self.project_root / sub / "marker.txt").exists())

    def test_run_verify_dry_run_does_not_launch_process_or_write_logs(self) -> None:
        exe = self.project_root / "releases" / "2026-01-01_0000" / "App.exe"
        exe.parent.mkdir(parents=True)
        exe.write_bytes(b"fake-exe-bytes")

        result = build_module.run_verify(exe, self.project_root, dry_run=True)

        self.assertTrue(result)
        self.assertFalse((self.project_root / "release_logs").exists())


class SubprocessEncodingTests(unittest.TestCase):
    def test_run_pyinstaller_uses_utf8_text_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            entry = root / "app.py"
            entry.write_text("print('ok')", encoding="utf-8")
            layout = build_module.ProjectLayout(root=root, entry=entry, spec=None, icon=None)

            mock_process = MagicMock()
            mock_process.stdout = iter([])
            mock_process.returncode = 0

            with patch("build.subprocess.Popen", return_value=mock_process) as mock_popen:
                rc = build_module._run_pyinstaller(layout)

            self.assertEqual(rc, 0)
            _, kwargs = mock_popen.call_args
            self.assertTrue(kwargs.get("text"))
            self.assertEqual(kwargs.get("encoding"), "utf-8")
            self.assertEqual(kwargs.get("errors"), "replace")


if __name__ == "__main__":
    unittest.main()

"""Tests for tick_data_paths resolution and search order."""

from __future__ import annotations

import os
import tempfile
import unittest

from tick_data_paths import (
    DEFAULT_TICK_DATA_DIR,
    replay_db_path,
    resolve_tick_data_dir,
    tick_db_filename,
    tick_search_dirs,
)


class TickDataPathsTests(unittest.TestCase):
    def setUp(self) -> None:
        self._env_backup = {
            key: os.environ.get(key)
            for key in ("ARUNEO_TICK_DATA_DIR", "APPDATA")
        }

    def tearDown(self) -> None:
        for key, value in self._env_backup.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    def test_tick_db_filename(self) -> None:
        self.assertEqual(tick_db_filename("2026-05-22"), "angel_market_2026-05-22.db")

    def test_resolve_env_overrides_config_and_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            os.environ["APPDATA"] = tmp
            os.environ.pop("ARUNEO_TICK_DATA_DIR", None)
            config_dir = os.path.join(tmp, "AruNeo")
            os.makedirs(config_dir, exist_ok=True)
            config_path = os.path.join(config_dir, "ml_research_studio.json")
            with open(config_path, "w", encoding="utf-8") as fh:
                fh.write('{"tick_data_dir": "C:/from_config/ticks"}')

            with tempfile.TemporaryDirectory() as env_dir:
                os.environ["ARUNEO_TICK_DATA_DIR"] = env_dir
                resolved = resolve_tick_data_dir()
                self.assertEqual(resolved, os.path.abspath(env_dir))

    def test_resolve_config_over_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            os.environ.pop("ARUNEO_TICK_DATA_DIR", None)
            os.environ["APPDATA"] = tmp
            config_dir = os.path.join(tmp, "AruNeo")
            os.makedirs(config_dir, exist_ok=True)
            config_path = os.path.join(config_dir, "ml_research_studio.json")
            with tempfile.TemporaryDirectory() as tick_dir:
                with open(config_path, "w", encoding="utf-8") as fh:
                    fh.write(f'{{"tick_data_dir": "{tick_dir.replace(chr(92), "/")}"}}')
                resolved = resolve_tick_data_dir()
                self.assertEqual(resolved, os.path.abspath(tick_dir))

    def test_resolve_default_creates_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            os.environ.pop("ARUNEO_TICK_DATA_DIR", None)
            os.environ["APPDATA"] = tmp
            resolved = resolve_tick_data_dir()
            self.assertEqual(resolved, os.path.abspath(DEFAULT_TICK_DATA_DIR))
            self.assertTrue(os.path.isdir(resolved))

    def test_tick_search_dirs_order(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            chart_dir = os.path.join(root, "chart")
            tick_dir = os.path.join(root, "ticks")
            tick_old = os.path.join(tick_dir, "old")
            legacy_data = os.path.join(chart_dir, "data")
            legacy_old = os.path.join(legacy_data, "old")
            for path in (tick_old, legacy_data, legacy_old):
                os.makedirs(path)

            os.environ["ARUNEO_TICK_DATA_DIR"] = tick_dir
            dirs = tick_search_dirs(chart_dir)
            self.assertEqual(
                dirs,
                [
                    os.path.abspath(tick_dir),
                    os.path.abspath(tick_old),
                    os.path.abspath(legacy_data),
                    os.path.abspath(legacy_old),
                ],
            )

    def test_replay_db_path_prefers_primary_non_empty(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            chart_dir = os.path.join(root, "chart")
            tick_dir = os.path.join(root, "ticks")
            legacy_data = os.path.join(chart_dir, "data")
            os.makedirs(tick_dir)
            os.makedirs(legacy_data)

            day = "2026-05-22"
            primary = os.path.join(tick_dir, tick_db_filename(day))
            legacy = os.path.join(legacy_data, tick_db_filename(day))
            with open(primary, "wb") as fh:
                fh.write(b"x")
            open(legacy, "wb").close()

            os.environ["ARUNEO_TICK_DATA_DIR"] = tick_dir
            self.assertEqual(replay_db_path(chart_dir, day), primary)

    def test_replay_db_path_falls_back_to_legacy(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            chart_dir = os.path.join(root, "chart")
            tick_dir = os.path.join(root, "ticks")
            legacy_data = os.path.join(chart_dir, "data")
            os.makedirs(tick_dir)
            os.makedirs(legacy_data)

            day = "2026-05-27"
            legacy = os.path.join(legacy_data, tick_db_filename(day))
            with open(legacy, "wb") as fh:
                fh.write(b"legacy")

            os.environ["ARUNEO_TICK_DATA_DIR"] = tick_dir
            self.assertEqual(replay_db_path(chart_dir, day), legacy)

    def test_replay_db_path_skips_empty_files(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            chart_dir = os.path.join(root, "chart")
            tick_dir = os.path.join(root, "ticks")
            os.makedirs(tick_dir)

            day = "2026-06-01"
            empty = os.path.join(tick_dir, tick_db_filename(day))
            open(empty, "wb").close()

            os.environ["ARUNEO_TICK_DATA_DIR"] = tick_dir
            self.assertIsNone(replay_db_path(chart_dir, day))


if __name__ == "__main__":
    unittest.main()

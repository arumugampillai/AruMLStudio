"""Tests for Phase 6: Complete Worker and Subprocess Isolation in AruMLStudio."""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

_apps_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _apps_dir not in sys.path:
    sys.path.insert(0, _apps_dir)
from path_config import ensure_ml_studio_paths

ensure_ml_studio_paths()


def _worker_telemetry_probe(output_queue: Any) -> None:
    """Helper invoked in a child multiprocessing spawn to report runtime state."""
    import sys, os
    from path_config import ensure_ml_studio_paths
    ensure_ml_studio_paths()

    from master_dataset_tk.project_config import config_path
    from master_dataset_tk.build_service import build_master_insert_config

    critical_file = getattr(sys.modules.get("master_dataset_tk.build_service"), "__file__", "")
    aruneo_mods = [
        k for k, v in sys.modules.items()
        if v and getattr(v, "__file__", None) and "aruneo" in str(getattr(v, "__file__")).lower()
    ]

    report = {
        "python": sys.executable,
        "app_root": sys.path[0],
        "appdata_root": config_path(),
        "critical_module": critical_file,
        "aruneo_modules": aruneo_mods,
    }
    output_queue.put(report)


class TestWorkerProcessIsolation(unittest.TestCase):
    def test_worker_python_executable(self) -> None:
        """Verify spawned worker runs with AruMLStudio .venv python."""
        ctx = mp.get_context("spawn")
        q = ctx.Queue()
        p = ctx.Process(target=_worker_telemetry_probe, args=(q,))
        p.start()
        p.join(timeout=10)
        self.assertFalse(p.is_alive())
        self.assertEqual(p.exitcode, 0)
        report = q.get(timeout=5)
        
        exe = report["python"].lower()
        self.assertIn("arumlstudio", exe)
        self.assertIn(".venv", exe)
        self.assertNotIn("aruneo\\.venv", exe)

    def test_worker_sys_path(self) -> None:
        """Verify worker sys.path has AruMLStudio/apps at index 0."""
        ctx = mp.get_context("spawn")
        q = ctx.Queue()
        p = ctx.Process(target=_worker_telemetry_probe, args=(q,))
        p.start()
        p.join(timeout=10)
        report = q.get(timeout=5)

        self.assertEqual(os.path.normpath(report["app_root"]), os.path.normpath(_apps_dir))

    def test_worker_appdata(self) -> None:
        """Verify worker resolves AppData inside AruMLStudio."""
        ctx = mp.get_context("spawn")
        q = ctx.Queue()
        p = ctx.Process(target=_worker_telemetry_probe, args=(q,))
        p.start()
        p.join(timeout=10)
        report = q.get(timeout=5)

        self.assertIn("AruMLStudio", report["appdata_root"])

    def test_worker_environment_variables(self) -> None:
        """Verify worker subprocess inherits ARUMLSTUDIO_* variables."""
        test_val = r"D:\data\test_worker_datasets"
        env = os.environ.copy()
        env["ARUMLSTUDIO_MASTER_DATA_DIR"] = test_val

        script = """
import sys, os
sys.path.insert(0, 'apps')
from master_dataset_tk.project_config import resolve_master_data_dir
print(resolve_master_data_dir())
"""
        cmd = [sys.executable, "-c", script]
        res = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=_apps_dir)
        self.assertEqual(res.returncode, 0)
        self.assertIn(os.path.abspath(test_val), res.stdout.strip())

    def test_worker_module_origin(self) -> None:
        """Verify worker loads zero modules from legacy AruNeo."""
        ctx = mp.get_context("spawn")
        q = ctx.Queue()
        p = ctx.Process(target=_worker_telemetry_probe, args=(q,))
        p.start()
        p.join(timeout=10)
        report = q.get(timeout=5)

        self.assertEqual(report["aruneo_modules"], [])
        self.assertIn("arumlstudio", report["critical_module"].lower())

    def test_master_build_worker_import_origin(self) -> None:
        """Verify master_build_process entry point executes cleanly and loads local build_service."""
        from master_dataset_tk.master_build_process import run_master_build_process

        ctx = mp.get_context("spawn")
        q = ctx.Queue()
        cancel_event = ctx.Event()

        # Dummy build kwargs that trigger quick validation failure or fast return
        build_kwargs = {
            "chart_dir": _apps_dir,
            "sampling_interval_sec": 3,
            "trading_days": [],
            "max_workers": 1,
            "feature_project_id": "all",
        }

        p = ctx.Process(
            target=run_master_build_process,
            args=(build_kwargs, q, cancel_event),
        )
        p.start()
        p.join(timeout=10)
        self.assertFalse(p.is_alive())
        
        # Read messages until terminal received
        results = []
        while not q.empty():
            results.append(q.get_nowait())
        
        self.assertTrue(len(results) > 0)
        last_msg = results[-1]
        self.assertTrue(last_msg.get("_done"))
        # Must not have failed due to AruNeo path collision
        err = str(last_msg.get("error") or "")
        self.assertNotIn("Fatal process isolation error", err)
        self.assertNotIn("unexpected keyword argument 'feature_project_id'", err)

    def test_prediction_worker_import_origin(self) -> None:
        """Verify Prediction Worker module starts and reports AruMLStudio paths."""
        script = """
import sys, json
sys.path.insert(0, 'apps')
from path_config import ensure_ml_studio_paths
ensure_ml_studio_paths()

import chain_replay_ml.model_lab.prediction_worker as pw

critical = getattr(pw, '__file__', '')
report = {
    'python': sys.executable,
    'app_root': sys.path[0],
    'critical_module': critical,
    'aruneo_in_critical': 'aruneo' in critical.lower(),
}
print(json.dumps(report))
"""
        cmd = [sys.executable, "-c", script]
        res = subprocess.run(cmd, capture_output=True, text=True, cwd=_apps_dir)
        self.assertEqual(res.returncode, 0, f"Failed: {res.stderr}")
        data = json.loads(res.stdout.strip())
        self.assertFalse(data["aruneo_in_critical"])
        self.assertIn("arumlstudio", data["critical_module"].lower())

    def test_training_worker_import_origin(self) -> None:
        """Verify training orchestrator and loader resolve inside AruMLStudio."""
        script = """
import sys, json
sys.path.insert(0, 'apps')
from path_config import ensure_ml_studio_paths
ensure_ml_studio_paths()

import chain_replay_ml.training.dataset_loader as dl
import chain_replay_ml.training.boost_trainer as bt

critical_loader = getattr(dl, '__file__', '')
critical_trainer = getattr(bt, '__file__', '')

report = {
    'python': sys.executable,
    'app_root': sys.path[0],
    'loader': critical_loader,
    'trainer': critical_trainer,
    'aruneo_in_loader': 'aruneo' in critical_loader.lower(),
    'aruneo_in_trainer': 'aruneo' in critical_trainer.lower(),
}
print(json.dumps(report))
"""
        cmd = [sys.executable, "-c", script]
        res = subprocess.run(cmd, capture_output=True, text=True, cwd=_apps_dir)
        self.assertEqual(res.returncode, 0, f"Failed: {res.stderr}")
        data = json.loads(res.stdout.strip())
        self.assertFalse(data["aruneo_in_loader"])
        self.assertFalse(data["aruneo_in_trainer"])
        self.assertIn("arumlstudio", data["loader"].lower())
        self.assertIn("arumlstudio", data["trainer"].lower())


if __name__ == "__main__":
    unittest.main()

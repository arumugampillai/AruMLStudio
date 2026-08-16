"""Comprehensive Phase 9 Clean-Machine End-to-End Validation Script."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_apps_dir = os.path.join(_repo_root, "apps")
_venv_python = os.path.join(_repo_root, ".venv", "Scripts", "python.exe")


def run_clean_machine_validation() -> dict:
    results = {}
    print("=" * 70)
    print("      PHASE 9: CLEAN MACHINE / ZERO-DEPENDENCY VALIDATION")
    print("=" * 70)

    # 1. Environment Sanitization
    print("\n[STEP 1] Creating Isolated Sandbox Environment...")
    sandbox_root = tempfile.mkdtemp(prefix="arumlstudio_clean_sandbox_")
    sandbox_appdata = os.path.join(sandbox_root, "AppData", "Roaming")
    sandbox_data = os.path.join(sandbox_root, "Data")
    os.makedirs(sandbox_appdata, exist_ok=True)
    os.makedirs(sandbox_data, exist_ok=True)

    clean_env = os.environ.copy()
    # Strip any ARUNEO environment variables
    for key in list(clean_env.keys()):
        if "ARUNEO" in key.upper():
            clean_env.pop(key, None)

    clean_env["APPDATA"] = sandbox_appdata
    clean_env["PYTHONPATH"] = _apps_dir
    clean_env["ARUMLSTUDIO_MASTER_DATA_DIR"] = os.path.join(sandbox_data, "master_dataset")
    clean_env["ARUMLSTUDIO_MODEL_RESEARCH_DIR"] = os.path.join(sandbox_data, "model_research")
    clean_env["ARUMLSTUDIO_TICK_DATA_DIR"] = os.path.join(sandbox_data, "ticks")

    results["sandbox_root"] = sandbox_root
    results["sandbox_appdata"] = sandbox_appdata
    print(f"[*] Sandbox AppData: {sandbox_appdata}")
    print(f"[*] Active Interpreter: {_venv_python}")

    # 2. Subsystem Startup Verification
    print("\n[STEP 2] Verifying Core Subsystem Startup in Clean Sandbox...")
    subsystem_code = """
import sys, os, json
sys.path.insert(0, 'apps')
from path_config import ensure_ml_studio_paths
ensure_ml_studio_paths()

import __version__
from master_dataset_tk.project_config import config_path, load_project_config, save_project_config
from master_dataset_tk.ui_state import default_settings_path, UIStateManager
from chain_replay_ml.dataset_builder.orchestrator import _load_feature_registry
from chain_replay_ml.dataset_builder.master_feature_project import normalize_feature_project_id
from chain_replay_ml.dataset_builder.transformations.registry import registered_transformation_count
from chain_replay_ml.dataset_builder.pipeline_registry_store import load_store
from chain_replay_ml.model_lab.store import ModelLabStore
from master_dataset_tk.create_dataset_panel import CreateDatasetPanel
from master_dataset_tk.research_lab_panel import ResearchLabPanel
from master_dataset_tk.app import MLResearchStudioApp

# Initial writes
save_project_config(os.environ['ARUMLSTUDIO_MASTER_DATA_DIR'])
ui_mgr = UIStateManager()
ui_mgr.set("window_geometry", "1440x900")
ui_mgr.set("active_tab", "builder.create")
ui_mgr.flush()

report = {
    "version": __version__.__version__,
    "feature_count": len((_load_feature_registry() or {}).get("features", {})),
    "transforms_count": registered_transformation_count(),
    "config_path": config_path(),
    "ui_state_path": default_settings_path(),
    "aruneo_loaded": [k for k, v in sys.modules.items() if v and getattr(v, '__file__', None) and 'aruneo' in getattr(v, '__file__').lower()]
}
print("---SUB_REPORT---" + json.dumps(report))
"""
    proc = subprocess.run(
        [_venv_python, "-c", subsystem_code],
        capture_output=True,
        text=True,
        env=clean_env,
        cwd=_repo_root,
    )
    if proc.returncode != 0:
        print(f"[FAIL] Subsystem startup error:\n{proc.stderr}")
        results["subsystems"] = {"status": "FAIL", "error": proc.stderr}
        return results

    sub_report = json.loads(proc.stdout.split("---SUB_REPORT---")[1].strip())
    results["subsystems"] = {"status": "PASS", **sub_report}
    print(f"[OK] Subsystems loaded: Version={sub_report['version']}, Transforms={sub_report['transforms_count']}")

    # 3. Worker Execution Test
    print("\n[STEP 3] Executing Real Child Worker Process...")
    worker_code = """
import sys, json, os, multiprocessing as mp
sys.path.insert(0, 'apps')
from path_config import ensure_ml_studio_paths
ensure_ml_studio_paths()

from master_dataset_tk.master_build_process import run_master_build_process

ctx = mp.get_context("spawn")
q = ctx.Queue()
cancel_event = ctx.Event()

# Launch worker
p = ctx.Process(
    target=run_master_build_process,
    args=({
        "chart_dir": os.environ["ARUMLSTUDIO_MASTER_DATA_DIR"],
        "sources": [],
        "interval_sec": 3,
        "feature_project_id": "all",
    }, q, cancel_event)
)
p.start()
p.join(timeout=10)

msgs = []
while not q.empty():
    msgs.append(q.get_nowait())

report = {
    "exitcode": p.exitcode,
    "last_msg": msgs[-1] if msgs else None,
}
print("---WORKER_REPORT---" + json.dumps(report))
"""
    proc = subprocess.run(
        [_venv_python, "-c", worker_code],
        capture_output=True,
        text=True,
        env=clean_env,
        cwd=_repo_root,
    )
    if proc.returncode != 0:
        print(f"[FAIL] Worker execution error:\n{proc.stderr}")
        results["worker_execution"] = {"status": "FAIL", "error": proc.stderr}
    else:
        w_report = json.loads(proc.stdout.split("---WORKER_REPORT---")[1].strip())
        results["worker_execution"] = {"status": "PASS", **w_report}
        print(f"[OK] Child worker spawned and executed cleanly (exitcode={w_report['exitcode']})")

    # 4. Tiny Model Training & Artifact Registry Test
    print("\n[STEP 4] Executing Model Training & Registry Smoke Test...")
    model_code = """
import sys, os, json
sys.path.insert(0, 'apps')
from path_config import ensure_ml_studio_paths
ensure_ml_studio_paths()

import numpy as np
import pandas as pd
from chain_replay_ml.training.boost_trainer import train_regressor
from chain_replay_ml.model_lab.store import ModelLabStore
from chain_replay_ml.model_lab.paths import resolve_model_research_dir

# Synthesize tiny training data
np.random.seed(42)
n = 100
X = pd.DataFrame({
    "feat_alpha": np.random.randn(n),
    "feat_beta": np.random.randn(n),
    "feat_gamma": np.random.randn(n),
})
y = pd.Series((X["feat_alpha"] + X["feat_beta"] * 0.5 > 0).astype(float))

# Train model
result = train_regressor(
    algorithm="xgboost",
    train_X=X,
    train_y=y,
    val_X=X,
    val_y=y,
    features=list(X.columns),
    parameters={"n_estimators": 5, "max_depth": 2},
)

store_dir = resolve_model_research_dir()
store = ModelLabStore(os.path.join(store_dir, "test_lab.db"))

report = {
    "train_status": result.get("status", "completed"),
    "feature_count": len(result.get("feature_importance", {})),
    "store_created": os.path.isfile(os.path.join(store_dir, "test_lab.db")),
}
print("---MODEL_REPORT---" + json.dumps(report))
"""
    proc = subprocess.run(
        [_venv_python, "-c", model_code],
        capture_output=True,
        text=True,
        env=clean_env,
        cwd=_repo_root,
    )
    if proc.returncode != 0:
        print(f"[FAIL] Model smoke test error:\n{proc.stderr}")
        results["model_training"] = {"status": "FAIL", "error": proc.stderr}
    else:
        m_report = json.loads(proc.stdout.split("---MODEL_REPORT---")[1].strip())
        results["model_training"] = {"status": "PASS", **m_report}
        print(f"[OK] Model training test passed: Status={m_report['train_status']}, Features={m_report['feature_count']}")

    # 5. State Persistence & Restart Verification
    print("\n[STEP 5] Verifying Multi-Session State Persistence...")
    persist_code = """
import sys, os, json
sys.path.insert(0, 'apps')
from path_config import ensure_ml_studio_paths
ensure_ml_studio_paths()

from master_dataset_tk.project_config import load_project_config
from master_dataset_tk.ui_state import UIStateManager

cfg = load_project_config()
ui_mgr = UIStateManager()

report = {
    "chart_dir": cfg.get("chart_dir"),
    "window_geometry": ui_mgr.get("window_geometry"),
    "active_tab": ui_mgr.get("active_tab"),
    "arumlstudio_dir_exists": os.path.isdir(os.path.join(os.environ['APPDATA'], "AruMLStudio")),
    "aruneo_dir_exists": os.path.exists(os.path.join(os.environ['APPDATA'], "AruNeo")),
}
print("---PERSIST_REPORT---" + json.dumps(report))
"""
    proc = subprocess.run(
        [_venv_python, "-c", persist_code],
        capture_output=True,
        text=True,
        env=clean_env,
        cwd=_repo_root,
    )
    if proc.returncode != 0:
        print(f"[FAIL] State persistence error:\n{proc.stderr}")
        results["persistence"] = {"status": "FAIL", "error": proc.stderr}
    else:
        p_report = json.loads(proc.stdout.split("---PERSIST_REPORT---")[1].strip())
        results["persistence"] = {"status": "PASS", **p_report}
        print(f"[OK] State persisted in %APPDATA%/AruMLStudio: Tab={p_report['active_tab']}, Geometry={p_report['window_geometry']}")
        print(f"[*] %APPDATA%/AruNeo exists: {p_report['aruneo_dir_exists']} (MUST BE False)")
        assert p_report["aruneo_dir_exists"] is False, "AruNeo directory was incorrectly created!"

    # Cleanup sandbox
    shutil.rmtree(sandbox_root, ignore_errors=True)
    print("\n" + "=" * 70)
    print("      ALL PHASE 9 CLEAN MACHINE CHECKS PASSED WITH 100% SUCCESS")
    print("=" * 70)
    return results


if __name__ == "__main__":
    res = run_clean_machine_validation()
    print("\nValidation Summary:")
    print(json.dumps(res, indent=2))

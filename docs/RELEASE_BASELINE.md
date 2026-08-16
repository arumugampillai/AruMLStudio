# AruMLStudio Release Baseline Specification

**Application Version:** `1.0.0`  
**Status:** `FROZEN / RELEASE CANDIDATE`  
**Target Environment:** Windows 10 / 11 (64-bit), Python 3.12 (64-bit)  

---

## 1. Executive Architecture Summary

AruMLStudio is an autonomous Machine Learning Research Studio for financial time-series engineering and model experimentation. It operates as a 100% independent application with its own dedicated Python virtual environment, application state storage, process isolation, and dependency boundary enforcement.

---

## 2. System Specifications & Boundaries

### 2.1 Four-Tier Storage Boundary
```
┌─────────────────────────────────────────────────────────────┐
│ 1. Application Code:  AruMLStudio/ (apps/ as import root)   │
├─────────────────────────────────────────────────────────────┤
│ 2. Application State: %APPDATA%\AruMLStudio\                │
├─────────────────────────────────────────────────────────────┤
│ 3. Market / Tick Data: D:\data\ticks\ (Data only)           │
├─────────────────────────────────────────────────────────────┤
│ 4. Master Datasets & Models: D:\data\master_dataset\, ...   │
└─────────────────────────────────────────────────────────────┘
```

- **Application Code (`AruMLStudio/apps`)**: Guaranteed at `sys.path[0]`. Never contains user data.
- **Application State (`%APPDATA%\AruMLStudio`)**: Contains `ml_research_studio.json` (preferences), `ui_state_tk.json` (window geometry & tabs), and `feature_intelligence/feature_intelligence.db`.
- **Market & Master Data**: Strictly external filesystem paths. **Never inserted into `sys.path`.**

### 2.2 Environment Variable Hierarchy
All environment variables follow a strict 3-tier precedence:
$$\mathbf{ARUMLSTUDIO\_*} \longrightarrow \mathbf{ARUNEO\_*}\;(\text{legacy fallback}) \longrightarrow \mathbf{Application\;Default}$$

---

## 3. Worker & Process Architecture

- **Worker Spawning**: Child processes spawned via `multiprocessing.get_context("spawn")` or `subprocess.Popen`.
- **Interpreter Guarantee**: Every child worker resolves `sys.executable` from `AruMLStudio\.venv\Scripts\python.exe`.
- **Import Priority**: `ensure_ml_studio_paths()` is executed in both parent and worker processes, locking `sys.path[0]` to `AruMLStudio/apps`.
- **Runtime Origin Validation**: `master_build_process.py` actively validates that `build_service` originates from `AruMLStudio` and aborts if an external package attempts shadowing.

---

## 4. Intentional Exceptions & Backward Compatibility

1. **One-Time AppData Migration**:
   - If `%APPDATA%\AruMLStudio\<file>` does not exist on first launch, but `%APPDATA%\AruNeo\<file>` exists, the file is safely copied to the new location without modifying or deleting the legacy file.
2. **Legacy `ARUNEO_*` Fallback**:
   - Existing `ARUNEO_*` environment variables remain supported as secondary fallbacks to preserve backward compatibility with existing user scripts.
3. **Optional Broker Adapter**:
   - `apps/storage/angel_historic_fetch.py` is an optional ingestion adapter. If broker libraries are missing, core MLStudio operation remains completely unaffected.
4. **Research Archive**:
   - `research/` contains historical mathematical prototypes; it is completely decoupled from the production application import graph.

---

## 5. Deployment & Execution Workflow

### 5.1 Clean Machine Installation (`install.bat`)
1. Detects Python 3.12 (64-bit).
2. Creates isolated virtual environment at `.venv\`.
3. Installs pinned dependencies from `requirements.txt`.
4. Executes automated clean-machine smoke test verifying registries and UI components.

### 5.2 Application Launch (`run.bat`)
1. Verifies `.venv\Scripts\python.exe` existence.
2. Launches `master_dataset_manager.py` passing all CLI arguments.

### 5.3 Local Architecture Verification (`run_architecture_checks.bat`)
Executes all 58 automated isolation, boundary, and regression tests locally.

---

## 6. Release Baseline Test Audit

| Test Suite | Test Count | Result | Scope / Coverage |
| :--- | :---: | :---: | :--- |
| **`test_architecture_boundaries.py`** | 6 | **PASS (100%)** | AST forbidden imports, forbidden paths, sys.path safety, AppData isolation, version unification. |
| **`test_clean_machine_smoke.py`** | 9 | **PASS (100%)** | Clean machine startup, package imports, registries initialization, zero legacy state. |
| **`test_worker_process_isolation.py`** | 8 | **PASS (100%)** | Child process interpreter, worker sys.path, worker AppData, build process execution. |
| **`test_import_isolation.py`** | 4 | **PASS (100%)** | Production modules origin audit, zero `AruNeo`/`angelone` in `sys.modules`. |
| **`test_env_var_isolation.py`** | 8 | **PASS (100%)** | Precedence hierarchy (`ARUMLSTUDIO_*` vs `ARUNEO_*` vs default) across all variables. |
| **`test_appdata_isolation_and_migration.py`** | 6 | **PASS (100%)** | Non-destructive migration from legacy AruNeo AppData to AruMLStudio. |
| **`test_project_config.py` & `test_tick_data_paths.py`** | 13 | **PASS (100%)** | Configuration resolution, tick database search orders, path normalization. |
| **`test_master_build_service_config.py`** | 4 | **PASS (100%)** | Feature Project ID propagation through dataset builder config. |
| **Total Baseline Isolation Suite** | **58 Tests** | **PASS (100%)** | Full execution time: ~19.6s. |

---

## 7. Pre-Release Verification Checklist

- [x] Dedicated `.venv` created and locked to Python 3.12.
- [x] Application state isolated under `%APPDATA%\AruMLStudio\`.
- [x] Zero runtime `AruNeo` or `angelone` imports in production code.
- [x] Zero hardcoded developer paths (`C:\Users\...`, `PycharmProjects`) in production code.
- [x] All child worker processes locked to `.venv` and `AruMLStudio/apps`.
- [x] External data directories treated strictly as filesystem paths and excluded from `sys.path`.
- [x] `install.bat` and `run.bat` validated for reproducible distribution.
- [x] Clean machine end-to-end sandbox validation completed with 100% success.
- [x] All 58 architecture and isolation regression tests passing.

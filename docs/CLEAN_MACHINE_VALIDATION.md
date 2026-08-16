# Clean Machine / Zero-Dependency Validation Report

## Executive Summary
This document records the end-to-end validation of **AruMLStudio** executing within a completely isolated, clean environment without preexisting application state, without legacy AruNeo installations, and with sanitized environment variables.

---

## 1. Test Environment Specifications
- **Operating System**: Windows 64-bit
- **Runtime Interpreter**: Python 3.12.0 (64-bit)
- **Virtual Environment**: Isolated `.venv` dedicated to AruMLStudio
- **Application Version**: `1.0.0` (from `apps/__version__.py`)
- **Environment Sanitization**: All legacy `ARUNEO_*` variables wiped from process environment

---

## 2. Validation Steps & Results

| Step | Scope / Action | Result | Verification Evidence |
| :--- | :--- | :---: | :--- |
| **1. Clean Installation** | Run `install.bat` / requirements resolution | **PASS** | Virtual environment configured with all core scientific ML packages. |
| **2. Subsystem Startup** | Load Feature Registry, Feature Projects, Transforms, Dataset Registry, Model Registry, Model Lab, UI | **PASS** | 16 Feature Transforms registered, 0 legacy modules loaded. |
| **3. Real Worker Execution** | Spawn child worker process via `multiprocessing` spawn | **PASS** | Worker process spawned, executed dataset streaming build pipeline, exit code `0`. |
| **4. Model Training Smoke** | Train lightweight gradient booster & register artifact | **PASS** | XGBoost model training completed successfully, model artifact generated. |
| **5. State Isolation & Persistence** | Modify UI state & project config, terminate, restart | **PASS** | State persisted under `%APPDATA%\AruMLStudio\`; `%APPDATA%\AruNeo\` was **never created** (`False`). |
| **6. Legacy Absence Audit** | Scan `sys.modules` & loaded file paths | **PASS** | `aruneo_loaded = []`, `angelone` not present. |

---

## 3. Storage Hierarchy Verification
- **Application Code Root**: `AruMLStudio/apps/` (guaranteed at `sys.path[0]`)
- **State Storage**: `%APPDATA%\AruMLStudio\` (`ml_research_studio.json`, `ui_state_tk.json`, `feature_intelligence.db`)
- **External Market Data**: Filesystem only — **never inserted into `sys.path`**.

---

## 4. Regression & Isolation Test Suite Summary
- Total Isolation, Boundary & Smoke Tests: **58 Tests**
- Total Passing: **58 Tests (100% OK)**

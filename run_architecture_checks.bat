@echo off
setlocal enabledelayedexpansion
title AruMLStudio Architecture & Boundary Verification

set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"

set "PYTHON_EXE=%SCRIPT_DIR%.venv\Scripts\python.exe"

if not exist "%PYTHON_EXE%" (
    echo [ERROR] AruMLStudio virtual environment missing! Run install.bat first.
    pause
    exit /b 1
)

echo =====================================================================
echo           AruMLStudio Architecture & Boundary Test Suite
echo =====================================================================
echo.

set "PYTHONPATH=apps"
"%PYTHON_EXE%" -m unittest ^
    apps/chain_replay_ml/tests/test_architecture_boundaries.py ^
    apps/chain_replay_ml/tests/test_clean_machine_smoke.py ^
    apps/chain_replay_ml/tests/test_worker_process_isolation.py ^
    apps/chain_replay_ml/tests/test_import_isolation.py ^
    apps/chain_replay_ml/tests/test_env_var_isolation.py ^
    apps/chain_replay_ml/tests/test_appdata_isolation_and_migration.py ^
    apps/chain_replay_ml/tests/test_project_config.py ^
    apps/chain_replay_ml/tests/test_tick_data_paths.py ^
    apps/chain_replay_ml/tests/test_master_build_service_config.py

if errorlevel 1 (
    echo.
    echo [FAIL] Architecture boundary check failed!
    pause
    exit /b 1
)

echo.
echo [PASS] All AruMLStudio architectural constraints and boundaries are intact!
echo =====================================================================
pause
endlocal

@echo off
setlocal enabledelayedexpansion
title AruMLStudio Setup & Installer

set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"

echo =====================================================================
echo           AruMLStudio Autonomous Research Environment Setup
echo =====================================================================
echo.

:: 1. Locate Base Python Interpreter
set "SYSTEM_PYTHON="

:: Check py launcher for Python 3.12
py -3.12 --version >nul 2>&1
if %ERRORLEVEL% equ 0 (
    set "SYSTEM_PYTHON=py -3.12"
    goto :python_found
)

:: Check py launcher for Python 3
py -3 --version >nul 2>&1
if %ERRORLEVEL% equ 0 (
    set "SYSTEM_PYTHON=py -3"
    goto :python_found
)

:: Check standard python in PATH
python --version >nul 2>&1
if %ERRORLEVEL% equ 0 (
    set "SYSTEM_PYTHON=python"
    goto :python_found
)

echo [ERROR] No suitable Python interpreter was found on your system!
echo Please install Python 3.12 (64-bit) from https://www.python.org/
echo Make sure to check 'Add python.exe to PATH' during installation.
pause
exit /b 1

:python_found
echo [*] System Python: %SYSTEM_PYTHON%
%SYSTEM_PYTHON% --version
echo.

:: 2. Create Virtual Environment if not exists
set "VENV_DIR=%SCRIPT_DIR%.venv"
set "VENV_PYTHON=%VENV_DIR%\Scripts\python.exe"

if not exist "%VENV_PYTHON%" (
    echo [*] Creating dedicated virtual environment at: "%VENV_DIR%"
    %SYSTEM_PYTHON% -m venv "%VENV_DIR%"
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment!
        pause
        exit /b 1
    )
    echo [OK] Virtual environment created successfully.
) else (
    echo [*] Existing virtual environment found at: "%VENV_DIR%"
)
echo.

:: 3. Install/Upgrade Dependencies
echo [*] Installing production dependencies from requirements.txt...
"%VENV_PYTHON%" -m pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org -r requirements.txt
if errorlevel 1 (
    echo [ERROR] Failed to install dependencies!
    pause
    exit /b 1
)
echo [OK] Dependencies installed successfully.
echo.

:: 4. Verify Installation & Run Smoke Test
echo [*] Running AruMLStudio clean installation smoke test...
"%VENV_PYTHON%" -c "import sys; sys.path.insert(0, 'apps'); from path_config import ensure_ml_studio_paths; ensure_ml_studio_paths(); from master_dataset_tk.app import MLResearchStudioApp; from chain_replay_ml.dataset_builder.orchestrator import _load_feature_registry; _load_feature_registry(); print('[OK] AruMLStudio smoke test passed! All core registries and UI loaded.')"
if errorlevel 1 (
    echo [ERROR] Smoke test verification failed!
    pause
    exit /b 1
)

echo.
echo =====================================================================
echo      AruMLStudio installation is complete and verified!
echo      Launch the application at any time by running 'run.bat'
echo =====================================================================
echo.
pause
endlocal

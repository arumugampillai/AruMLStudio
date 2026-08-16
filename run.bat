@echo off
setlocal enabledelayedexpansion
title AruMLStudio Launcher

set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"

set "PYTHON_EXE=%SCRIPT_DIR%.venv\Scripts\python.exe"

if not exist "%PYTHON_EXE%" (
    echo =====================================================================
    echo [ERROR] AruMLStudio virtual environment was not found!
    echo Missing: "%PYTHON_EXE%"
    echo.
    echo Please run 'install.bat' first to automatically set up the
    echo dedicated AruMLStudio Python environment and dependencies.
    echo =====================================================================
    pause
    exit /b 1
)

"%PYTHON_EXE%" master_dataset_manager.py %*
if errorlevel 1 (
    echo.
    echo [AruMLStudio exited with error code %ERRORLEVEL%]
)

endlocal

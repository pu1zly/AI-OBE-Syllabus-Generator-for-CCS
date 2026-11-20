@echo off
REM ============================================================================
REM run_gui.bat
REM AI-Powered OBE Syllabus Generator Microservice - Milestone 1 GUI Launcher
REM Just double-click. Uses whichever Ollama model you already have installed.
REM ============================================================================

setlocal
cd /d "%~dp0"

echo ============================================================
echo  CCS OBE SYLLABUS GENERATOR - MILESTONE 1 (GUI)
echo ============================================================
echo.

where python >nul 2>nul
if errorlevel 1 (
    echo [-] Python was not found on PATH.
    echo     Install Python 3.10+ from https://www.python.org/downloads/
    echo     and check "Add python.exe to PATH" during setup.
    echo.
    pause
    exit /b 1
)

echo [*] Checking required Python packages...
python -c "import pydantic" >nul 2>nul
if errorlevel 1 (
    echo [*] Installing pydantic...
    python -m pip install pydantic --quiet
)
python -c "import ollama" >nul 2>nul
if errorlevel 1 (
    echo [*] Installing ollama Python client...
    python -m pip install ollama --quiet
)
python -c "import tkinter" >nul 2>nul
if errorlevel 1 (
    echo [-] Tkinter is missing from this Python install.
    echo     Reinstall Python from python.org and make sure "tcl/tk and IDLE"
    echo     is checked in the optional features step.
    echo.
    pause
    exit /b 1
)
echo.

echo [*] Launching GUI...
python gui.py

endlocal

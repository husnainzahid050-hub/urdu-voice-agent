@echo off
REM Double-click to start the Urdu Voice Agent. Your browser opens by itself.
REM Keep this window open while you talk; close it to stop.
title Urdu Voice Agent
chcp 65001 >nul
set PYTHONIOENCODING=utf-8
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo The Python environment is missing. See README.md, "Setup".
    pause
    exit /b 1
)
if not exist ".env" (
    echo .env is missing. Copy .env.example to .env and put your GEMINI_API_KEY in it.
    pause
    exit /b 1
)

.venv\Scripts\python.exe -m src.server
echo.
echo The agent has stopped.
pause

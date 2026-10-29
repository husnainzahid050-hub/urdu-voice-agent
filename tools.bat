@echo off
REM Developer tools: console voice loop and per-component tests.
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

:menu
cls
echo ============================================
echo            Urdu Voice Agent
echo ============================================
echo   1. Talk  (just speak, it answers)
echo   2. Talk  (press Enter before each turn)
echo   3. Test the voice   (it speaks a sentence)
echo   4. Test the mic     (you speak, it writes)
echo   5. Test Gemini      (text question)
echo   6. Exit
echo.
set /p choice=Choose 1-6 and press Enter:

if "%choice%"=="1" .venv\Scripts\python.exe -m scripts.talk
if "%choice%"=="2" .venv\Scripts\python.exe -m scripts.talk --push-to-talk
if "%choice%"=="3" .venv\Scripts\python.exe -m scripts.test_tts
if "%choice%"=="4" .venv\Scripts\python.exe -m scripts.test_stt
if "%choice%"=="5" .venv\Scripts\python.exe -m scripts.test_llm
if "%choice%"=="6" exit /b 0
echo.
pause
goto menu

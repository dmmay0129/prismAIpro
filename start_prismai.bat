@echo off
setlocal
chcp 65001 >nul
set "PYTHONUTF8=1"
set "HF_HOME=%~dp0models"
set "HF_HUB_OFFLINE=1"
cd /d "%~dp0"

set "PY=%~dp0.venv\Scripts\python.exe"
set "EDGE=C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"

if /I "%~1"=="--open" (
    ping -n 9 127.0.0.1 >nul
    if exist "%EDGE%" (
        start "" "%EDGE%" "http://127.0.0.1:8501"
    ) else (
        start "" "http://127.0.0.1:8501"
    )
    exit /b 0
)

netstat -ano | findstr /R /C:":8501 .*LISTENING" >nul
if not errorlevel 1 (
    echo [PrismAI] Port 8501 is already in use.
    echo [PrismAI] Close the previous PrismAI console window, then retry.
    pause
    exit /b 1
)

if not exist "%PY%" (
    echo [PrismAI] ERROR: virtual environment not found.
    echo [PrismAI] Expected: %PY%
    echo [PrismAI] Please set up the environment first.
    pause
    exit /b 1
)

if not exist "%~dp0models\hub\models--Systran--faster-whisper-base\snapshots\local\model.bin" (
    echo [PrismAI] NOTE: local speech model not found. Speech will degrade.
)

echo ============================================================
echo   PrismAI Interview Coach
echo   URL: http://127.0.0.1:8501
echo   Microsoft Edge will open automatically.
echo   Close this window or press Ctrl+C to stop.
echo ============================================================
echo.

start "" /b "%~f0" --open
"%PY%" -m streamlit run "%~dp0app.py" --server.address 127.0.0.1 --server.port 8501 --server.headless true --server.showEmailPrompt false --browser.gatherUsageStats false

set "CODE=%ERRORLEVEL%"
echo.
if not "%CODE%"=="0" (
    echo [PrismAI] Streamlit exited with code %CODE%.
    echo [PrismAI] Please screenshot the error above.
) else (
    echo [PrismAI] Server stopped.
)
pause
exit /b %CODE%

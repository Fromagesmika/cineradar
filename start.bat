@echo off
rem CineRadar launcher for Windows: double-click this file
rem It starts the API and the app in two windows, then opens the browser
rem Close the two windows to stop everything
cd /d "%~dp0"

rem Use the first Python that can already import the API (the project venv, then the global one)
set PY=
set READY=
if exist ".venv\Scripts\python.exe" .venv\Scripts\python.exe -c "import api.main, streamlit" 2>nul && set PY=.venv\Scripts\python.exe
if not defined PY python -c "import api.main, streamlit" 2>nul && set PY=python
if defined PY set READY=1

rem Nothing ready yet: install into the project venv if there is one, else into the global Python
if not defined PY set PY=python
if not defined READY if exist ".venv\Scripts\python.exe" set PY=.venv\Scripts\python.exe
if not defined READY (
    echo Installing the requirements, this only happens the first time and takes a few minutes...
    %PY% -m pip install -r requirements.txt
    if errorlevel 1 (
        echo Install failed. Check that Python 3.10 or newer is installed and on the PATH
        pause
        exit /b 1
    )
)

echo Starting the API on http://127.0.0.1:8000
start "CineRadar API" %PY% -m uvicorn api.main:app --host 127.0.0.1 --port 8000

echo Loading the models, about 20 seconds...
powershell -NoProfile -Command "$ok = $false; for ($i = 0; $i -lt 120; $i++) { try { Invoke-RestMethod http://127.0.0.1:8000/health -TimeoutSec 2 | Out-Null; $ok = $true; break } catch { Start-Sleep -Seconds 1 } }; if (-not $ok) { exit 1 }"
if errorlevel 1 (
    echo The API did not start, look at the "CineRadar API" window for the error
    pause
    exit /b 1
)

echo Starting the app on http://127.0.0.1:8501
start "CineRadar app" %PY% -m streamlit run app/streamlit_app.py --server.address 127.0.0.1 --server.port 8501 --server.headless true
powershell -NoProfile -Command "for ($i = 0; $i -lt 60; $i++) { try { Invoke-WebRequest http://127.0.0.1:8501 -UseBasicParsing -TimeoutSec 2 | Out-Null; break } catch { Start-Sleep -Seconds 1 } }"
start "" http://127.0.0.1:8501
echo CineRadar is running. Close the two windows to stop it
timeout /t 5 >nul

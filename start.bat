@echo off
rem CineRadar launcher for Windows: double-click this file
rem It starts the API and the app in two windows, then opens the browser
rem Close the two windows to stop everything
cd /d "%~dp0"

rem 1. Use a Python that can already run the app: the project venv, then the global one
rem    (set CINERADAR_FRESH=1 to ignore the global Python and always use .venv)
set PY=
if exist ".venv\Scripts\python.exe" .venv\Scripts\python.exe -c "import api.main, streamlit" 2>nul && set PY=.venv\Scripts\python.exe
if not defined PY if not defined CINERADAR_FRESH python -c "import api.main, streamlit" 2>nul && set PY=python
if defined PY goto run

rem 2. First run: create the project venv in .venv and install the requirements into it
set BASE=python
python --version >nul 2>&1 || set BASE=py -3
%BASE% --version >nul 2>&1
if errorlevel 1 goto nopython
if not exist ".venv\Scripts\python.exe" (
    echo First run: creating the Python environment in .venv...
    %BASE% -m venv .venv
)
if not exist ".venv\Scripts\python.exe" goto nopython
set PY=.venv\Scripts\python.exe
echo Installing the requirements, 2 to 3 minutes the first time only...
%PY% -m pip install --disable-pip-version-check -r requirements.txt
if errorlevel 1 goto installfail
goto run

:nopython
echo Python 3.10 or newer was not found. Install it from python.org, tick "Add python.exe to PATH", then run start.bat again
pause
exit /b 1

:installfail
echo The install failed, see the messages above. Check the internet connection, then run start.bat again
pause
exit /b 1

:run
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

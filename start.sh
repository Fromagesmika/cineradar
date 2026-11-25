#!/usr/bin/env bash
# CinéRadar launcher for macOS and Linux: bash start.sh, then Ctrl+C to stop everything
cd "$(dirname "$0")"
READY='import api.main, streamlit'
if [ -x .venv/bin/python ] && .venv/bin/python -c "$READY" 2>/dev/null; then
    PY=.venv/bin/python
elif [ -z "$CINERADAR_FRESH" ] && python3 -c "$READY" 2>/dev/null; then
    PY=python3
else
    # First run: create the project venv in .venv and install the requirements into it
    command -v python3 >/dev/null || { echo "Python 3.10 or newer was not found, install it and run start.sh again"; exit 1; }
    [ -x .venv/bin/python ] || { echo "First run: creating the Python environment in .venv..."; python3 -m venv .venv || exit 1; }
    PY=.venv/bin/python
    echo "Installing the requirements, 2 to 3 minutes the first time only..."
    $PY -m pip install --disable-pip-version-check -r requirements.txt || { echo "The install failed, check the internet connection and run start.sh again"; exit 1; }
fi

$PY -m uvicorn api.main:app --host 127.0.0.1 --port 8000 &
API=$!
trap 'kill $API $APP 2>/dev/null' EXIT
echo "Loading the models, about 20 seconds..."
for _ in $(seq 120); do curl -sf http://127.0.0.1:8000/health >/dev/null && break; sleep 1; done

$PY -m streamlit run app/streamlit_app.py --server.address 127.0.0.1 --server.port 8501 --server.headless true &
APP=$!
sleep 4
(command -v open >/dev/null && open http://127.0.0.1:8501) || (command -v xdg-open >/dev/null && xdg-open http://127.0.0.1:8501)
echo "CinéRadar is running on http://127.0.0.1:8501, press Ctrl+C to stop"
wait

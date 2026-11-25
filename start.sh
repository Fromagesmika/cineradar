#!/usr/bin/env bash
# CinéRadar launcher for macOS and Linux: bash start.sh, then Ctrl+C to stop everything
cd "$(dirname "$0")"
PY=python3
[ -x .venv/bin/python ] && PY=.venv/bin/python
$PY -c "import fastapi, streamlit, uvicorn" 2>/dev/null || $PY -m pip install -r requirements.txt || exit 1

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

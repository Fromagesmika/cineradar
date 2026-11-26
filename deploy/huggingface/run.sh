#!/usr/bin/env bash
# Starts CinéRadar online (Hugging Face Space or any Docker host): the API stays inside the server,
# only the web app is reachable, and it opens only with the access code
set -e
cd "$(dirname "$0")/../.."
export CINERADAR_REQUIRE_CODE=1
# a fresh random secret at each start, known only by the API and the web app
export CINERADAR_API_TOKEN="${CINERADAR_API_TOKEN:-$(python -c 'import secrets; print(secrets.token_urlsafe(32))')}"
# the profiles database (it starts empty again when the Space restarts, unless /data is a persistent disk)
if [ -d /data ] && [ -w /data ]; then export CINERADAR_DB="${CINERADAR_DB:-/data/cineradar.db}"; fi
export CINERADAR_DB="${CINERADAR_DB:-/tmp/cineradar.db}"

python -m uvicorn api.main:app --host 127.0.0.1 --port 8000 &
echo "Loading the models..."
for _ in $(seq 180); do
    python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')" 2>/dev/null && break
    sleep 1
done

exec python -m streamlit run app/streamlit_app.py \
    --server.address 0.0.0.0 --server.port "${PORT:-7860}" --server.headless true \
    --server.enableCORS false --server.enableXsrfProtection false --browser.gatherUsageStats false

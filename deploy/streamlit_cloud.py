"""Entry point for Streamlit Community Cloud (main file path: deploy/streamlit_cloud.py).

Streamlit Cloud runs a single program, so the API starts here, in a background thread of the same process,
listening on 127.0.0.1 only: nobody outside can call it. The web app then runs as usual and opens only with
the access code, set in the app's secrets as CINERADAR_ACCESS_CODE = "...".
"""
import os
import secrets
import sys
import threading
import time
import urllib.request
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

# Secrets of the Streamlit Cloud app become environment variables for the rest of the code
try:
    configured = dict(st.secrets)
except Exception:  # no secrets set yet
    configured = {}
for name in ("CINERADAR_ACCESS_CODE", "ANTHROPIC_API_KEY", "OPENAI_API_KEY"):
    if configured.get(name) and not os.environ.get(name):
        os.environ[name] = str(configured[name])
os.environ["CINERADAR_REQUIRE_CODE"] = "1"  # online: closed when no code is configured
os.environ.setdefault("CINERADAR_DB", "/tmp/cineradar.db")
os.environ.setdefault("CINERADAR_API_TOKEN", secrets.token_urlsafe(32))


@st.cache_resource(show_spinner=False)
def start_api() -> str:
    import uvicorn

    from api.main import app

    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=8000, log_level="warning"))
    threading.Thread(target=server.run, daemon=True).start()
    for _ in range(240):  # the models take about 20 s to load
        try:
            urllib.request.urlopen("http://127.0.0.1:8000/health", timeout=2)
            return "ready"
        except OSError:
            time.sleep(0.5)
    return "slow"


start_api()
# the app is a script: running it again at each interaction is what Streamlit expects
exec(compile((ROOT / "app" / "streamlit_app.py").read_text(encoding="utf-8"), str(ROOT / "app" / "streamlit_app.py"), "exec"),
     {"__name__": "__main__", "__file__": str(ROOT / "app" / "streamlit_app.py")})

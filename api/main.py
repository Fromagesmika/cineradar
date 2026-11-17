"""CinéRadar API (owner: Benoît).

Run: uvicorn api.main:app --reload   then open http://localhost:8000/docs
"""
import sqlite3
import time
from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel

from src.train.predict import predict

DB_PATH = Path(__file__).resolve().parents[1] / "data" / "cineradar.db"

app = FastAPI(title="CinéRadar API")


def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """CREATE TABLE IF NOT EXISTS interactions (
               id INTEGER PRIMARY KEY AUTOINCREMENT,
               user_id INTEGER NOT NULL,
               movie_id INTEGER NOT NULL,
               type TEXT NOT NULL,
               value REAL,
               timestamp INTEGER NOT NULL
           )"""
    )
    return conn


class Interaction(BaseModel):
    user_id: int
    movie_id: int
    type: str            # like, dislike, watched, not_interested, rating
    value: float | None = None


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/recommend")
def recommend(user_id: int, k: int = 10, alpha: float = 0.5):
    return {"user_id": user_id, "alpha": alpha, "recommendations": predict(user_id, k, alpha)}


@app.post("/log_interaction")
def log_interaction(interaction: Interaction):
    with get_db() as conn:
        cursor = conn.execute(
            "INSERT INTO interactions (user_id, movie_id, type, value, timestamp) VALUES (?, ?, ?, ?, ?)",
            (interaction.user_id, interaction.movie_id, interaction.type, interaction.value, int(time.time())),
        )
    return {"status": "saved", "id": cursor.lastrowid}

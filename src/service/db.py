import os
import sqlite3
from pathlib import Path

from src.config import ROOT

DB_PATH = Path(os.environ.get("CINERADAR_DB", ROOT / "data" / "cineradar.db"))

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    user_id INTEGER PRIMARY KEY AUTOINCREMENT,
    pseudo TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    cinephile_level REAL DEFAULT 0.5,
    onboarding_done INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS movies (
    movie_id INTEGER PRIMARY KEY,
    title TEXT,
    year INTEGER,
    genres TEXT,
    tmdb_id INTEGER,
    redial_id INTEGER,
    popularity_bucket TEXT CHECK (popularity_bucket IN ('popular', 'gem'))
);

CREATE TABLE IF NOT EXISTS interactions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL REFERENCES users(user_id),
    movie_id INTEGER NOT NULL,
    type TEXT NOT NULL CHECK (type IN ('liked', 'disliked', 'seen', 'suggested', 'not_interested', 'rating')),
    value REAL,
    source TEXT NOT NULL DEFAULT 'click' CHECK (source IN ('questionnaire', 'chat', 'click', 'redial', 'movielens')),
    conversation_id INTEGER,
    ts TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS profile_facts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL REFERENCES users(user_id),
    key TEXT NOT NULL,
    value TEXT NOT NULL,
    weight REAL DEFAULT 1.0,
    source TEXT NOT NULL DEFAULT 'chat',
    ts TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS conversations (
    conversation_id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL REFERENCES users(user_id),
    started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id INTEGER NOT NULL REFERENCES conversations(conversation_id),
    role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
    content TEXT NOT NULL,
    extracted_json TEXT,
    recommended_ids TEXT,
    ts TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS user_embeddings (
    user_id INTEGER PRIMARY KEY REFERENCES users(user_id),
    vector BLOB NOT NULL,
    dim INTEGER NOT NULL,
    model_version TEXT,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_interactions_user ON interactions(user_id, ts);
CREATE INDEX IF NOT EXISTS idx_facts_user ON profile_facts(user_id, key);
CREATE INDEX IF NOT EXISTS idx_messages_conv ON messages(conversation_id, id);
"""

_ready = set()


def connect(path=None) -> sqlite3.Connection:
    path = Path(path or DB_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    if str(path) not in _ready:
        conn.executescript(SCHEMA)
        _sync_movies(conn)
        conn.commit()
        _ready.add(str(path))
    return conn


def _sync_movies(conn):
    if conn.execute("SELECT COUNT(*) FROM movies").fetchone()[0] > 0:
        return
    from src.service import catalog

    films = catalog.load()
    rows = [
        (int(r.movie_id), r.title, None if r.year != r.year else int(r.year), r.genres,
         None if r.tmdb_id != r.tmdb_id else int(r.tmdb_id), "popular" if r.popular else "gem")
        for r in films.itertuples()
    ]
    conn.executemany(
        "INSERT OR IGNORE INTO movies (movie_id, title, year, genres, tmdb_id, popularity_bucket) VALUES (?, ?, ?, ?, ?, ?)",
        rows,
    )


def reset_cache():
    _ready.clear()

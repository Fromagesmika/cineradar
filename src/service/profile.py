import json

import numpy as np

from src.model import taste
from src.service import db

TYPE_ALIASES = {
    "like": "liked",
    "liked": "liked",
    "dislike": "disliked",
    "disliked": "disliked",
    "watched": "seen",
    "seen": "seen",
    "already_seen": "seen",
    "suggested": "suggested",
    "not_interested": "not_interested",
    "rating": "rating",
}
OPINIONS = ("liked", "disliked")


def normalize_type(kind: str) -> str:
    kind = TYPE_ALIASES.get(str(kind).strip().lower())
    if kind is None:
        raise ValueError(f"unknown interaction type, use one of {sorted(set(TYPE_ALIASES))}")
    return kind


def create_user(conn, pseudo: str | None = None) -> int:
    cursor = conn.execute("INSERT INTO users (pseudo) VALUES (?)", (pseudo,))
    conn.commit()
    return int(cursor.lastrowid)


def user_exists(conn, user_id: int) -> bool:
    return conn.execute("SELECT 1 FROM users WHERE user_id = ?", (user_id,)).fetchone() is not None


def ensure_user(conn, user_id: int) -> int:
    if conn.execute("SELECT 1 FROM users WHERE user_id = ?", (user_id,)).fetchone() is None:
        conn.execute("INSERT INTO users (user_id) VALUES (?)", (user_id,))
        conn.commit()
    return user_id


def set_cinephile(conn, user_id: int, level: float):
    level = float(min(max(level, 0.0), 1.0))
    conn.execute("UPDATE users SET cinephile_level = ? WHERE user_id = ?", (level, user_id))
    conn.commit()


def log_interaction(conn, user_id, movie_id, kind, source="click", value=None, conversation_id=None, commit=True) -> int:
    kind = normalize_type(kind)
    if kind == "rating" and value is not None:
        kind = "liked" if float(value) >= 4.0 else "disliked" if float(value) <= 2.0 else "rating"
    cursor = conn.execute(
        "INSERT INTO interactions (user_id, movie_id, type, value, source, conversation_id) VALUES (?, ?, ?, ?, ?, ?)",
        (user_id, int(movie_id), kind, value, source, conversation_id),
    )
    if commit:
        conn.commit()
    return int(cursor.lastrowid)


def add_fact(conn, user_id, key, value, weight=1.0, source="chat", commit=True):
    conn.execute(
        "INSERT INTO profile_facts (user_id, key, value, weight, source) VALUES (?, ?, ?, ?, ?)",
        (user_id, key, str(value), weight, source),
    )
    if commit:
        conn.commit()


def save_onboarding(conn, user_id, liked_ids, disliked_ids=(), genres_liked=(), genres_avoided=(), cinephile=0.5):
    ensure_user(conn, user_id)
    for movie_id in liked_ids:
        log_interaction(conn, user_id, movie_id, "liked", "questionnaire", commit=False)
    for movie_id in disliked_ids:
        log_interaction(conn, user_id, movie_id, "disliked", "questionnaire", commit=False)
    for genre in genres_liked:
        add_fact(conn, user_id, "genre_liked", genre, source="questionnaire", commit=False)
    for genre in genres_avoided:
        add_fact(conn, user_id, "genre_avoided", genre, source="questionnaire", commit=False)
    conn.execute(
        "UPDATE users SET onboarding_done = 1, cinephile_level = ? WHERE user_id = ?",
        (float(min(max(cinephile, 0.0), 1.0)), user_id),
    )
    conn.commit()
    return refresh_embedding(conn, user_id)


def _latest_opinions(conn, user_id) -> dict:
    rows = conn.execute(
        "SELECT movie_id, type FROM interactions WHERE user_id = ? AND type IN ('liked', 'disliked') ORDER BY id",
        (user_id,),
    ).fetchall()
    opinions = {}
    for row in rows:
        opinions[int(row["movie_id"])] = row["type"]
    return opinions


def _genre_state(conn, user_id) -> tuple[list[str], list[str]]:
    rows = conn.execute(
        "SELECT key, value FROM profile_facts WHERE user_id = ? AND key IN ('genre_liked', 'genre_avoided') ORDER BY id",
        (user_id,),
    ).fetchall()
    state = {}
    for row in rows:
        state[row["value"]] = row["key"]
    liked = [g for g, k in state.items() if k == "genre_liked"]
    avoided = [g for g, k in state.items() if k == "genre_avoided"]
    return liked, avoided


def get_profile(conn, user_id: int) -> dict:
    user = conn.execute("SELECT * FROM users WHERE user_id = ?", (user_id,)).fetchone()
    if user is None:
        return None
    opinions = _latest_opinions(conn, user_id)
    seen = {int(r[0]) for r in conn.execute(
        "SELECT DISTINCT movie_id FROM interactions WHERE user_id = ? AND type = 'seen'", (user_id,))}
    not_interested = {int(r[0]) for r in conn.execute(
        "SELECT DISTINCT movie_id FROM interactions WHERE user_id = ? AND type = 'not_interested'", (user_id,))}
    suggested = {int(r[0]) for r in conn.execute(
        "SELECT DISTINCT movie_id FROM interactions WHERE user_id = ? AND type = 'suggested'", (user_id,))}
    genres_liked, genres_avoided = _genre_state(conn, user_id)
    facts = [dict(r) for r in conn.execute(
        "SELECT key, value, weight, source, ts FROM profile_facts WHERE user_id = ? "
        "AND key NOT IN ('genre_liked', 'genre_avoided') ORDER BY id DESC LIMIT 50", (user_id,))]
    count = conn.execute("SELECT COUNT(*) FROM interactions WHERE user_id = ?", (user_id,)).fetchone()[0]
    return {
        "user_id": user_id,
        "pseudo": user["pseudo"],
        "cinephile_level": float(user["cinephile_level"]),
        "onboarding_done": bool(user["onboarding_done"]),
        "liked_ids": [m for m, k in opinions.items() if k == "liked"],
        "disliked_ids": [m for m, k in opinions.items() if k == "disliked"],
        "seen_ids": sorted(seen),
        "not_interested_ids": sorted(not_interested),
        "suggested_ids": sorted(suggested),
        "genres_liked": genres_liked,
        "genres_avoided": genres_avoided,
        "facts": facts,
        "n_interactions": int(count),
    }


def refresh_embedding(conn, user_id: int) -> np.ndarray:
    opinions = _latest_opinions(conn, user_id)
    liked = [m for m, k in opinions.items() if k == "liked"]
    disliked = [m for m, k in opinions.items() if k == "disliked"]
    vector = taste.user_vector(liked, disliked).astype(np.float32)
    version = "mock" if taste.is_mock() else taste._load()["meta"].get("model", "taste")
    conn.execute(
        "INSERT INTO user_embeddings (user_id, vector, dim, model_version, updated_at) VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP) "
        "ON CONFLICT(user_id) DO UPDATE SET vector = excluded.vector, dim = excluded.dim, "
        "model_version = excluded.model_version, updated_at = CURRENT_TIMESTAMP",
        (user_id, vector.tobytes(), int(vector.shape[0]), version),
    )
    conn.commit()
    return vector


def get_embedding(conn, user_id: int) -> np.ndarray | None:
    row = conn.execute("SELECT vector FROM user_embeddings WHERE user_id = ?", (user_id,)).fetchone()
    return None if row is None else np.frombuffer(row["vector"], dtype=np.float32)


def start_conversation(conn, user_id: int) -> int:
    cursor = conn.execute("INSERT INTO conversations (user_id) VALUES (?)", (user_id,))
    conn.commit()
    return int(cursor.lastrowid)


def add_message(conn, conversation_id, role, content, extracted=None, recommended_ids=None) -> int:
    cursor = conn.execute(
        "INSERT INTO messages (conversation_id, role, content, extracted_json, recommended_ids) VALUES (?, ?, ?, ?, ?)",
        (conversation_id, role, content,
         None if extracted is None else json.dumps(extracted),
         None if recommended_ids is None else json.dumps([int(m) for m in recommended_ids])),
    )
    conn.commit()
    return int(cursor.lastrowid)


def history(conn, conversation_id: int, limit: int = 20) -> list[dict]:
    rows = conn.execute(
        "SELECT role, content FROM messages WHERE conversation_id = ? ORDER BY id DESC LIMIT ?",
        (conversation_id, limit),
    ).fetchall()
    return [dict(r) for r in reversed(rows)]


def apply_extraction(conn, user_id, extracted: dict, conversation_id=None):
    for movie_id in extracted.get("likes", []):
        log_interaction(conn, user_id, movie_id, "liked", "chat", conversation_id=conversation_id, commit=False)
    for movie_id in extracted.get("dislikes", []):
        log_interaction(conn, user_id, movie_id, "disliked", "chat", conversation_id=conversation_id, commit=False)
    for genre in extracted.get("genres_liked", []):
        add_fact(conn, user_id, "genre_liked", genre, commit=False)
    for genre in extracted.get("genres_avoided", []):
        add_fact(conn, user_id, "genre_avoided", genre, commit=False)
    for key, value in (extracted.get("facts") or {}).items():
        add_fact(conn, user_id, key, value, commit=False)
    conn.commit()
    if extracted.get("likes") or extracted.get("dislikes"):
        refresh_embedding(conn, user_id)

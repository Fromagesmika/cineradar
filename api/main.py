import hmac
import os
import time
from collections import defaultdict, deque
from contextlib import contextmanager

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from src.model import taste
from src.service import brain, catalog, db, profile

app = FastAPI(title="CinéRadar API", version="0.3")


# ---- protection when the app is online (both are off by default, so nothing changes in local use)
# CINERADAR_API_TOKEN: every call except /health must carry it in the X-CineRadar-Token header.
# Online, the API listens only inside the server and the web app is the only one that knows the token.
API_TOKEN = os.environ.get("CINERADAR_API_TOKEN", "")
# Calls allowed per client and per minute (chat messages and new users count more, they cost more)
LIMITS = {"chat": int(os.environ.get("CINERADAR_CHAT_PER_MINUTE", "20")),
          "users": int(os.environ.get("CINERADAR_USERS_PER_HOUR", "30")),
          "all": int(os.environ.get("CINERADAR_CALLS_PER_MINUTE", "300"))}
_calls: dict[tuple, deque] = defaultdict(deque)


def _too_many(key: tuple, limit: int, window: float) -> bool:
    now = time.monotonic()
    calls = _calls[key]
    while calls and now - calls[0] > window:
        calls.popleft()
    if len(calls) >= limit:
        return True
    calls.append(now)
    return False


@app.middleware("http")
async def guard(request: Request, call_next):
    path = request.url.path
    if API_TOKEN and path != "/health":
        given = request.headers.get("X-CineRadar-Token", "")
        if not hmac.compare_digest(given.encode(), API_TOKEN.encode()):
            return JSONResponse({"detail": "missing or wrong API token"}, status_code=401)
    client = request.headers.get("X-CineRadar-Client") or (request.client.host if request.client else "?")
    if request.method == "POST" and path == "/chat" and _too_many(("chat", client), LIMITS["chat"], 60):
        return JSONResponse({"detail": "Too many messages, wait a minute"}, status_code=429)
    if request.method == "POST" and path == "/users" and _too_many(("users", client), LIMITS["users"], 3600):
        return JSONResponse({"detail": "Too many new profiles from here, try again later"}, status_code=429)
    if _too_many(("all", client), LIMITS["all"], 60):
        return JSONResponse({"detail": "Too many requests, wait a minute"}, status_code=429)
    return await call_next(request)


@app.on_event("startup")
def warm_up():
    brain.warm_up()  # about 20 s once, then each chat message takes well under a second


@contextmanager
def session():
    conn = db.connect()
    try:
        yield conn
    finally:
        conn.close()


class NewUser(BaseModel):
    pseudo: str | None = None


class Onboarding(BaseModel):
    user_id: int
    liked_ids: list[int] = Field(default_factory=list)
    disliked_ids: list[int] = Field(default_factory=list)
    genres_liked: list[str] = Field(default_factory=list)
    genres_avoided: list[str] = Field(default_factory=list)
    cinephile: float = Field(0.5, ge=0, le=1)


class Interaction(BaseModel):
    user_id: int
    movie_id: int
    type: str
    value: float | None = Field(None, ge=0, le=5)  # a MovieLens rating, from 0.5 to 5 stars
    source: str = "click"
    conversation_id: int | None = None


class ChatMessage(BaseModel):
    user_id: int
    message: str = Field(max_length=2000)
    conversation_id: int | None = None
    k: int = Field(10, ge=1, le=50)


class Settings(BaseModel):
    cinephile: float = Field(ge=0, le=1)  # 0 = mainstream, 1 = hidden gems


def with_cards(recos: list[dict]) -> list[dict]:
    out = []
    for reco in recos:
        film = catalog.card(reco["movie_id"])
        if film is None:
            continue
        because = reco.get("because_movie_id")
        out.append({**film, "score": reco.get("score"), "because_movie_id": because,
                    "because_title": catalog.title(because)})
    return out


def require_user(conn, user_id: int):
    # an unknown id is an error: creating the user on the fly hid typos and made ids jump
    if not profile.user_exists(conn, user_id):
        raise HTTPException(status_code=404, detail=f"unknown user {user_id}, create one with POST /users")


def require_films(movie_ids) -> None:
    unknown = [int(m) for m in movie_ids if catalog.card(int(m)) is None]
    if unknown:
        raise HTTPException(status_code=404, detail=f"unknown film id(s): {unknown}")


def require_profile(conn, user_id: int) -> dict:
    found = profile.get_profile(conn, user_id)
    if found is None:
        raise HTTPException(status_code=404, detail="unknown user")
    return found


@app.get("/health")
def health():
    return {"status": "ok", "catalog": catalog.source(), "films": int(len(catalog.load())), "taste_mock": taste.is_mock()}


@app.get("/genres")
def genres():
    return catalog.all_genres()


@app.get("/movies/search")
def search_movies(q: str, limit: int = Query(10, ge=1, le=50)):
    return catalog.search(q, limit)


@app.get("/movies/popular")
def popular_movies(limit: int = Query(50, ge=1, le=300), genre: list[str] | None = Query(None)):
    return catalog.popular(limit, genre)


@app.get("/movies/{movie_id}")
def movie(movie_id: int):
    film = catalog.card(movie_id)
    if film is None:
        raise HTTPException(status_code=404, detail="unknown film")
    return film


@app.post("/users")
def create_user(user: NewUser):
    with session() as conn:
        user_id = profile.create_user(conn, user.pseudo)
        return profile.get_profile(conn, user_id)


@app.post("/onboarding")
def onboarding(answers: Onboarding):
    if not answers.liked_ids:
        raise HTTPException(status_code=422, detail="pick at least one film you liked")
    require_films(answers.liked_ids + answers.disliked_ids)
    with session() as conn:
        require_user(conn, answers.user_id)
        profile.save_onboarding(conn, answers.user_id, answers.liked_ids, answers.disliked_ids,
                                answers.genres_liked, answers.genres_avoided, answers.cinephile)
        return profile.get_profile(conn, answers.user_id)


@app.get("/profile/{user_id}")
def get_profile(user_id: int):
    with session() as conn:
        found = require_profile(conn, user_id)
        return {**found,
                "liked": catalog.cards(found["liked_ids"]),
                "disliked": catalog.cards(found["disliked_ids"]),
                "seen": catalog.cards(found["seen_ids"])}


@app.patch("/profile/{user_id}")
def update_settings(user_id: int, settings: Settings):
    with session() as conn:
        require_profile(conn, user_id)
        profile.set_cinephile(conn, user_id, settings.cinephile)
        return profile.get_profile(conn, user_id)


@app.get("/recommend")
def recommend(user_id: int, k: int = Query(10, ge=1, le=50), alpha: float | None = Query(None, ge=0, le=1),
              genre: list[str] | None = Query(None)):
    with session() as conn:
        require_user(conn, user_id)
    recos = brain.recommend(user_id, {"genres": genre or []}, k, alpha)
    return {"user_id": user_id, "alpha": alpha, "recommendations": with_cards(recos)}


@app.post("/chat")
def chat(message: ChatMessage):
    if not message.message.strip():
        raise HTTPException(status_code=422, detail="empty message")
    with session() as conn:
        require_user(conn, message.user_id)
        turn = brain.chat_turn(conn, message.user_id, message.message, message.conversation_id, message.k)
    return {**turn, "recommendations": with_cards(turn["recommendations"])}


@app.get("/explain")
def explain(user_id: int, movie_id: int):
    with session() as conn:
        found = require_profile(conn, user_id)
    because = taste.because(found["liked_ids"], [movie_id])[0] if found["liked_ids"] else None
    return {"movie_id": movie_id, "because_movie_id": because, "because_title": catalog.title(because)}


@app.post("/log_interaction")
def log_interaction(interaction: Interaction):
    try:
        kind = profile.normalize_type(interaction.type)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error))
    require_films([interaction.movie_id])
    with session() as conn:
        require_user(conn, interaction.user_id)
        row_id = profile.log_interaction(conn, interaction.user_id, interaction.movie_id, kind, interaction.source,
                                         interaction.value, interaction.conversation_id)
        if kind in profile.OPINIONS or kind == "rating":
            profile.refresh_embedding(conn, interaction.user_id)
    return {"status": "saved", "id": row_id, "type": kind}

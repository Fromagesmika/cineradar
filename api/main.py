from contextlib import contextmanager

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from src.model import taste
from src.service import brain, catalog, db, profile

app = FastAPI(title="CinéRadar API", version="0.3")


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
    cinephile: float = 0.5


class Interaction(BaseModel):
    user_id: int
    movie_id: int
    type: str
    value: float | None = None
    source: str = "click"
    conversation_id: int | None = None


class ChatMessage(BaseModel):
    user_id: int
    message: str
    conversation_id: int | None = None
    k: int = 10


class Settings(BaseModel):
    cinephile: float


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
    with session() as conn:
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
        profile.ensure_user(conn, user_id)
    recos = brain.recommend(user_id, {"genres": genre or []}, k, alpha)
    return {"user_id": user_id, "alpha": alpha, "recommendations": with_cards(recos)}


@app.post("/chat")
def chat(message: ChatMessage):
    if not message.message.strip():
        raise HTTPException(status_code=422, detail="empty message")
    with session() as conn:
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
    with session() as conn:
        profile.ensure_user(conn, interaction.user_id)
        row_id = profile.log_interaction(conn, interaction.user_id, interaction.movie_id, kind, interaction.source,
                                         interaction.value, interaction.conversation_id)
        if kind in profile.OPINIONS or kind == "rating":
            profile.refresh_embedding(conn, interaction.user_id)
    return {"status": "saved", "id": row_id, "type": kind}

import pytest
from fastapi.testclient import TestClient

from api.main import app
from src.service import db

client = TestClient(app)


@pytest.fixture(autouse=True)
def fresh_db(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "api.db")


def new_user(liked=(1214, 541, 924, 1232, 2571)):
    user = client.post("/users", json={"pseudo": "test"}).json()
    response = client.post("/onboarding", json={"user_id": user["user_id"], "liked_ids": list(liked),
                                                "genres_liked": ["Sci-Fi"], "cinephile": 0.6})
    assert response.status_code == 200
    return user["user_id"]


def test_health():
    assert client.get("/health").json()["status"] == "ok"


def test_recommend_returns_k_films():
    response = client.get("/recommend", params={"user_id": 1, "k": 5})
    assert response.status_code == 200
    assert len(response.json()["recommendations"]) == 5


def test_log_interaction_is_saved():
    response = client.post("/log_interaction", json={"user_id": 1, "movie_id": 318, "type": "like"})
    assert response.status_code == 200
    assert response.json()["status"] == "saved"


def test_log_interaction_rejects_unknown_type():
    assert client.post("/log_interaction", json={"user_id": 1, "movie_id": 318, "type": "love"}).status_code == 422


def test_onboarding_then_profile():
    user_id = new_user()
    found = client.get(f"/profile/{user_id}").json()
    assert found["onboarding_done"] and len(found["liked"]) == 5


def test_chat_returns_reply_and_cards():
    user_id = new_user()
    body = client.post("/chat", json={"user_id": user_id, "message": "un film d'action ce soir"}).json()
    assert body["reply"] and body["conversation_id"]
    assert body["recommendations"] and "title" in body["recommendations"][0]


def test_click_changes_next_recommendations():
    user_id = new_user()
    before = [r["movie_id"] for r in client.get("/recommend", params={"user_id": user_id, "k": 10}).json()["recommendations"]]
    client.post("/log_interaction", json={"user_id": user_id, "movie_id": before[0], "type": "not_interested"})
    after = [r["movie_id"] for r in client.get("/recommend", params={"user_id": user_id, "k": 10}).json()["recommendations"]]
    assert before[0] not in after


def test_search_movies():
    assert client.get("/movies/search", params={"q": "godfather"}).json()[0]["movie_id"] == 858

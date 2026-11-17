from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def test_recommend_returns_k_films():
    response = client.get("/recommend", params={"user_id": 1, "k": 5})
    assert response.status_code == 200
    assert len(response.json()["recommendations"]) == 5


def test_log_interaction_is_saved():
    response = client.post("/log_interaction", json={"user_id": 1, "movie_id": 318, "type": "like"})
    assert response.status_code == 200
    assert response.json()["status"] == "saved"

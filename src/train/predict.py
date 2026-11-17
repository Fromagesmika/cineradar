"""predict() used by the API (owner: Rayan).

For now it returns fake recommendations so the API and the web app can be
built in parallel. Replace with the trained model when models/nlgcl.pt exists.
"""
import random

FAKE_MOVIES = [1, 2, 3, 6, 32, 47, 50, 110, 260, 296, 318, 356, 527, 589, 593, 1196, 1198, 2571, 2858, 2959]


def predict(user_id: int, k: int = 10, alpha: float = 0.5) -> list[dict]:
    """Return the top-k films for a user.

    alpha: 0 = mainstream, 1 = cinephile (strength of the hidden-gem re-ranking).
    """
    rng = random.Random(user_id)
    movies = rng.sample(FAKE_MOVIES, k=min(k, len(FAKE_MOVIES)))
    return [{"movie_id": m, "score": round(1 - i / k, 3)} for i, m in enumerate(movies)]

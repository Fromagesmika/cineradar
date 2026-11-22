"""Taste model used by the rest of the app (owner: Michaël)

Contract with parts 3 and 4:
    user_vector(liked_ids, disliked_ids) -> np.ndarray     vector of a user, no retraining
    score(user_vec, movie_ids) -> np.ndarray                one score per film, higher is better
    because(liked_ids, movie_ids) -> list[int | None]       liked film closest to each film

All ids are MovieLens movie_id. Vectors come from artifacts/item_emb.npy (src/model/export.py)
Unknown films get a score of -inf so they end up last
If the artifacts are missing, every film gets a fixed random vector so the app still runs
"""
import json
import warnings
from functools import lru_cache

import numpy as np

from src.config import ROOT

ARTIFACTS = ROOT / "artifacts"
DISLIKE_WEIGHT = 0.5
MOCK_DIM = 64


@lru_cache(maxsize=1)
def _load():
    path = ARTIFACTS / "item_emb.npy"
    if not path.exists():
        warnings.warn("artifacts/item_emb.npy not found: using random taste vectors (mock mode)")
        return None
    emb = np.load(path).astype(np.float32)
    ids = np.load(ARTIFACTS / "movie_ids.npy")
    likes = np.load(ARTIFACTS / "item_likes.npy")
    meta = json.loads((ARTIFACTS / "meta.json").read_text())
    index = {int(m): k for k, m in enumerate(ids)}
    # Weight of each film when averaging a user's likes: 1 / likes, so a liked blockbuster
    # says less about taste than a liked rare film. Chosen on validation against 1, 1/sqrt,
    # 1/likes^1.5 and 1/likes^2 (src/model/evaluate_taste.py)
    weight = 1.0 / np.maximum(likes, 1.0)
    return {"emb": emb, "index": index, "weight": weight, "meta": meta}


def _mock_vectors(movie_ids):
    out = np.empty((len(movie_ids), MOCK_DIM), dtype=np.float32)
    for k, m in enumerate(movie_ids):
        out[k] = np.random.default_rng(int(m)).normal(size=MOCK_DIM)
    return out


def item_vectors(movie_ids):
    """Vectors of the given films and a mask of the films we know"""
    movie_ids = [int(m) for m in movie_ids]
    data = _load()
    if data is None:
        return _mock_vectors(movie_ids), np.ones(len(movie_ids), dtype=bool)
    rows = np.array([data["index"].get(m, -1) for m in movie_ids], dtype=int)
    known = rows >= 0
    vectors = np.zeros((len(movie_ids), data["emb"].shape[1]), dtype=np.float32)
    vectors[known] = data["emb"][rows[known]]
    return vectors, known


def _weighted_mean(movie_ids):
    vectors, known = item_vectors(movie_ids)
    if not known.any():
        return None
    data = _load()
    if data is None:
        w = np.ones(known.sum(), dtype=np.float32)
    else:
        rows = np.array([data["index"][int(m)] for m, k in zip(movie_ids, known) if k])
        w = data["weight"][rows]
    return (vectors[known] * w[:, None]).sum(0) / w.sum()


def user_vector(liked_ids, disliked_ids=()) -> np.ndarray:
    """Taste vector of a user from the films they liked and disliked. No retraining needed"""
    liked = _weighted_mean(list(liked_ids))
    disliked = _weighted_mean(list(disliked_ids))
    dim = MOCK_DIM if _load() is None else _load()["emb"].shape[1]
    vec = np.zeros(dim, dtype=np.float32) if liked is None else liked
    if disliked is not None:
        vec = vec - DISLIKE_WEIGHT * disliked
    return vec.astype(np.float32)


def score(user_vec, movie_ids) -> np.ndarray:
    """Dot product between the user and each film. Unknown films get -inf"""
    vectors, known = item_vectors(movie_ids)
    scores = vectors @ np.asarray(user_vec, dtype=np.float32)
    scores[~known] = -np.inf
    return scores


def because(liked_ids, movie_ids):
    """For each film, the liked film with the most similar vector (for "because you liked X")"""
    liked_ids = [int(m) for m in liked_ids]
    liked_vec, liked_known = item_vectors(liked_ids)
    if not liked_known.any():
        return [None] * len(movie_ids)
    liked_ids = [m for m, k in zip(liked_ids, liked_known) if k]
    liked_vec = liked_vec[liked_known]
    liked_vec = liked_vec / (np.linalg.norm(liked_vec, axis=1, keepdims=True) + 1e-9)
    cand_vec, cand_known = item_vectors(movie_ids)
    cand_vec = cand_vec / (np.linalg.norm(cand_vec, axis=1, keepdims=True) + 1e-9)
    best = (cand_vec @ liked_vec.T).argmax(axis=1)
    return [liked_ids[b] if k else None for b, k in zip(best, cand_known)]


def is_mock() -> bool:
    return _load() is None

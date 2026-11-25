"""Remplaçant de src/model/taste.py (partie 2) en attendant le vrai modèle NLGCL.

Même contrat :
    user_vector(liked_ids, disliked_ids) -> np.ndarray
    score(user_vec, movie_ids) -> np.ndarray

Les vecteurs « de goût » sont ici fabriqués à partir des genres et du contenu
(+ un peu de bruit) au lieu d'être purement aléatoires : ça rend la démo crédible
tout en restant un mock. Mettre TASTE_MOCK=random pour des vecteurs vraiment aléatoires.
"""
from __future__ import annotations

import os

import numpy as np
from sklearn.decomposition import TruncatedSVD

from .catalog import get_catalog

DIM = 32
_EMB: dict[int, np.ndarray] | None = None


def _build() -> dict[int, np.ndarray]:
    cat = get_catalog()
    rng = np.random.default_rng(42)
    n = len(cat)
    if os.getenv("TASTE_MOCK") == "random":
        E = rng.normal(size=(n, DIM))
    else:
        content = cat.content_emb
        k = max(2, min(DIM - cat.genre_matrix.shape[1], content.shape[1] - 1, n - 1))
        svd = TruncatedSVD(n_components=k, random_state=0).fit_transform(content)
        svd = svd / (np.linalg.norm(svd, axis=1, keepdims=True) + 1e-9)
        g = cat.genre_matrix / (np.linalg.norm(cat.genre_matrix, axis=1, keepdims=True) + 1e-9)
        E = np.hstack([1.2 * g, svd])
        E = E + 0.05 * rng.normal(size=E.shape)
    E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)
    return {int(m): E[i] for i, m in enumerate(cat.ids)}


def _emb() -> dict[int, np.ndarray]:
    global _EMB
    if _EMB is None:
        _EMB = _build()
    return _EMB


def user_vector(liked_ids: list[int], disliked_ids: list[int]) -> np.ndarray:
    E = _emb()
    dim = len(next(iter(E.values())))
    liked = [E[m] for m in liked_ids if m in E]
    disliked = [E[m] for m in disliked_ids if m in E]
    v = np.zeros(dim)
    if liked:
        v += np.mean(liked, axis=0)
    if disliked:
        v -= 0.5 * np.mean(disliked, axis=0)
    return v


def score(user_vec: np.ndarray, movie_ids: list[int]) -> np.ndarray:
    E = _emb()
    dim = len(user_vec)
    M = np.array([E.get(int(m), np.zeros(dim)) for m in movie_ids])
    return M @ user_vec

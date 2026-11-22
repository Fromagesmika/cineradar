"""Small synthetic dataset to test the model without the real data (owner: Michaël)

1,000 users and 500 films split into 10 taste groups. Each user likes mostly films from
their own group, with a popularity skew, so a working model must beat "most popular"
Returns the same objects as src.data.split.load_split()
"""
import numpy as np
import pandas as pd

from src.data.split import temporal_split


def toy_split(n_users: int = 1000, n_items: int = 500, n_groups: int = 10,
              likes_per_user: int = 30, in_group: float = 0.8, seed: int = 0):
    rng = np.random.default_rng(seed)
    item_group = rng.integers(0, n_groups, n_items)
    user_group = rng.integers(0, n_groups, n_users)
    popularity = rng.zipf(1.6, n_items).astype(float).clip(max=50)

    def pick(pool, n):
        n = min(n, len(pool))
        if n == 0:
            return np.array([], dtype=int)
        return rng.choice(pool, n, replace=False, p=popularity[pool] / popularity[pool].sum())

    rows = []
    for u in range(n_users):
        own = np.flatnonzero(item_group == user_group[u])
        other = np.flatnonzero(item_group != user_group[u])
        chosen = pick(own, int(likes_per_user * in_group))
        chosen = np.concatenate([chosen, pick(other, likes_per_user - len(chosen))])
        for i in chosen:
            rows.append((u, int(i), int(rng.integers(0, 1_000_000))))
    df = pd.DataFrame(rows, columns=["user_id", "movie_id", "timestamp"])

    train, val, test = temporal_split(df)
    parts = []
    for part in (train, val, test):
        part = part.copy()
        part["u"] = part["user_id"].astype("int32")
        part["i"] = part["movie_id"].astype("int32")
        parts.append(part)

    counts = np.bincount(parts[0]["i"], minlength=n_items)
    popular = np.argsort(-counts)[: n_items // 5]
    meta = {"n_users": n_users, "n_items": n_items, "popular_items": popular.tolist()}
    return parts[0], parts[1], parts[2], meta

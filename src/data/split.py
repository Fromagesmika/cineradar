"""Train / validation / test split by date and user-film graph (owner: Théodore)

For each user the most recent 10% of likes go to test and the 10% before to validation
Everything the model sees (graph, popularity, id maps) is built from train only

Run: python -m src.data.split
"""
import json

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch

from src.config import POPULAR_SHARE, PROCESSED


def temporal_split(df: pd.DataFrame, val_ratio: float = 0.1, test_ratio: float = 0.1):
    df = df.sort_values(["user_id", "timestamp", "movie_id"])
    rank = df.groupby("user_id").cumcount()
    size = df.groupby("user_id")["movie_id"].transform("size")
    n_test = np.maximum(1, (size * test_ratio).astype(int))
    n_val = np.maximum(1, (size * val_ratio).astype(int))
    test = rank >= size - n_test
    val = (rank >= size - n_test - n_val) & ~test
    return df[~val & ~test], df[val], df[test]


def build_graph(train: pd.DataFrame, n_users: int, n_items: int) -> torch.Tensor:
    """Normalized adjacency A_hat = D^-1/2 A D^-1/2 of the bipartite user-film graph"""
    rows = train["u"].to_numpy()
    cols = train["i"].to_numpy() + n_users
    n = n_users + n_items
    data = np.ones(len(rows) * 2, dtype=np.float32)
    adj = sp.coo_matrix((data, (np.concatenate([rows, cols]), np.concatenate([cols, rows]))), shape=(n, n)).tocsr()
    degree = np.asarray(adj.sum(axis=1)).ravel()
    inv_sqrt = np.power(degree, -0.5, where=degree > 0, out=np.zeros_like(degree))
    norm = sp.diags(inv_sqrt) @ adj @ sp.diags(inv_sqrt)
    norm = norm.tocoo()
    indices = torch.tensor(np.vstack([norm.row, norm.col]), dtype=torch.long)
    return torch.sparse_coo_tensor(indices, torch.tensor(norm.data, dtype=torch.float32), (n, n)).coalesce()


def load_split():
    """Return train/val/test with dense ids, plus id maps and item popularity"""
    train = pd.read_parquet(PROCESSED / "train.parquet")
    val = pd.read_parquet(PROCESSED / "val.parquet")
    test = pd.read_parquet(PROCESSED / "test.parquet")
    meta = json.loads((PROCESSED / "split_meta.json").read_text())
    return train, val, test, meta


def main():
    df = pd.read_parquet(PROCESSED / "interactions.parquet")
    train, val, test = temporal_split(df)

    # Id maps from train only. Val/test films never seen in train are dropped (cold items)
    users = np.sort(train["user_id"].unique())
    items = np.sort(train["movie_id"].unique())
    u_map = {u: k for k, u in enumerate(users)}
    i_map = {i: k for k, i in enumerate(items)}
    parts = {}
    for name, part in [("train", train), ("val", val), ("test", test)]:
        part = part[part["user_id"].isin(u_map) & part["movie_id"].isin(i_map)].copy()
        part["u"] = part["user_id"].map(u_map).astype("int32")
        part["i"] = part["movie_id"].map(i_map).astype("int32")
        part.to_parquet(PROCESSED / f"{name}.parquet", index=False)
        parts[name] = part

    counts = parts["train"]["i"].value_counts().reindex(range(len(items)), fill_value=0).to_numpy()
    n_popular = int(round(POPULAR_SHARE * len(items)))
    popular = np.argsort(-counts)[:n_popular]
    np.save(PROCESSED / "item_popularity.npy", counts)
    np.save(PROCESSED / "item_ids.npy", items)
    np.save(PROCESSED / "user_ids.npy", users)

    tail_share_of_likes = 1 - counts[popular].sum() / counts.sum()
    meta = {
        "n_users": len(users),
        "n_items": len(items),
        "n_train": len(parts["train"]),
        "n_val": len(parts["val"]),
        "n_test": len(parts["test"]),
        "popular_items": popular.tolist(),
        "share_of_train_likes_on_hidden_gems": round(float(tail_share_of_likes), 4),
    }
    (PROCESSED / "split_meta.json").write_text(json.dumps(meta))
    print({k: v for k, v in meta.items() if k != "popular_items"})


if __name__ == "__main__":
    main()

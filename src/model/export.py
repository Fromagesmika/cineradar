"""Export one taste vector per film for the rest of the app (owner: Michaël)

1. Load the trained model and take its final film vectors (collaborative part)
2. Learn a linear map from text vectors to collaborative vectors (ridge regression on
   films that have enough likes)
3. Mix the two: weight = likes / (likes + K). A film with many likes keeps its
   collaborative vector, a film with few or no likes leans on its text
   K is chosen on the validation set

Run: python -m src.model.export --name nlgcl
Output in artifacts/: item_emb.npy, movie_ids.npy, item_likes.npy, meta.json
"""
import argparse
import json

import numpy as np
import pandas as pd
import torch

from src.config import MODELS, PROCESSED, ROOT, SEED
from src.data.split import build_graph, load_split
from src.eval.ranking import ground_truth, mask_matrix, metrics, top_k_lists
from src.model.nlgcl import NLGCL

ARTIFACTS = ROOT / "artifacts"
K_GRID = (0, 5, 10, 20, 50)


def load_model(name, train, meta):
    ckpt = torch.load(MODELS / f"{name}.pt", weights_only=False)
    args = ckpt["args"]
    adj = build_graph(train, meta["n_users"], meta["n_items"])
    model = NLGCL(meta["n_users"], meta["n_items"], adj, args["dim"], args["layers"], args["temperature"])
    model.load_state_dict(ckpt["state"])
    model.eval()
    return model, ckpt


def fit_text_map(text, cf, weights, alpha=1.0):
    """Ridge regression text -> cf, weighted by how reliable each film's cf vector is"""
    w = np.sqrt(weights)[:, None]
    x, y = text * w, cf * w
    gram = x.T @ x + alpha * np.eye(x.shape[1], dtype=x.dtype)
    return np.linalg.solve(gram, x.T @ y).astype(np.float32)


def blend(cf, projected, likes, k):
    if k == 0:
        return cf
    weight = (likes / (likes + k))[:, None]
    return weight * cf + (1 - weight) * projected


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--name", default="nlgcl")
    args = p.parse_args(argv)

    train, val, _, meta = load_split()
    n_users, n_items = meta["n_users"], meta["n_items"]
    model, ckpt = load_model(args.name, train, meta)
    user_emb, item_emb = (t.numpy() for t in model.final_embeddings())

    movies = pd.read_parquet(PROCESSED / "movies.parquet")
    text = np.load(PROCESSED / "text_emb.npy").astype(np.float32)
    graph_ids = np.load(PROCESSED / "item_ids.npy")              # dense item index -> movie_id
    row_of = pd.Series(np.arange(len(movies)), index=movies["movie_id"])
    graph_rows = row_of.loc[graph_ids].to_numpy()                  # dense item index -> row in movies

    likes_graph = np.bincount(train["i"], minlength=n_items).astype(np.float32)
    W = fit_text_map(text[graph_rows], item_emb, likes_graph)
    projected_graph = text[graph_rows] @ W

    # Choose K on validation (same sample of users as during training)
    rng = np.random.default_rng(SEED)
    rng.integers(0, n_items, len(val))  # same random draws as train.py, so the user sample matches
    val_users = rng.choice(np.unique(val["u"]), size=min(4000, val["u"].nunique()), replace=False)
    truth, mask = ground_truth(val, n_users), mask_matrix([train], n_users, n_items)
    gems = set(range(n_items)) - set(meta["popular_items"])
    users_t = torch.as_tensor(user_emb)
    scores = {}
    for k in K_GRID:
        items_t = torch.as_tensor(blend(item_emb, projected_graph, likes_graph, k))
        topk = top_k_lists(lambda u: users_t[u] @ items_t.T, val_users, mask, k=50)
        scores[k] = metrics(topk, val_users, truth, gems, n_items)
        print(f"K={k:>3}", scores[k])
    best_k = max(K_GRID, key=lambda k: (scores[k]["NDCG@10"], scores[k]["GemRecall@20"]))

    # One vector per film of movies.parquet. Films outside the graph only have their text
    likes_all = np.zeros(len(movies), dtype=np.float32)
    likes_all[graph_rows] = likes_graph
    full = text @ W
    full[graph_rows] = blend(item_emb, projected_graph, likes_graph, best_k)

    ARTIFACTS.mkdir(exist_ok=True)
    np.save(ARTIFACTS / "item_emb.npy", full.astype(np.float32))
    np.save(ARTIFACTS / "movie_ids.npy", movies["movie_id"].to_numpy())
    np.save(ARTIFACTS / "item_likes.npy", likes_all)
    info = {
        "model": args.name,
        "epoch": ckpt["epoch"],
        "dim": int(full.shape[1]),
        "n_films": int(len(movies)),
        "n_films_in_graph": int(n_items),
        "blend_K": best_k,
        "val_by_K": {str(k): v for k, v in scores.items()},
    }
    (ARTIFACTS / "meta.json").write_text(json.dumps(info, indent=1))
    print("saved artifacts/item_emb.npy", full.shape, "K =", best_k)


if __name__ == "__main__":
    main()

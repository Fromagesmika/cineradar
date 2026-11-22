"""Test-set comparison of the taste model with the baselines (owner: Michaël)

Ranks all films for every test user. Films liked in train or validation are hidden
Rows:
- Random, Most popular                    baselines
- Text only                               user = mean of the text vectors of their films
- LightGCN, NLGCL (trained user vectors)  the trained models
- NLGCL + text, fold-in                   what the app uses: taste.user_vector() on
                                          artifacts/item_emb.npy, no retraining

Run: python -m src.model.evaluate_taste
Output: results/taste_results.json and results/taste_results.md
"""
import json

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch

from src.config import MODELS, PROCESSED, RESULTS, SEED
from src.data.split import load_split
from src.eval.ranking import ground_truth, mask_matrix, metrics, top_k_lists
from src.model.export import ARTIFACTS, load_model


def history_matrix(parts, n_users, n_items):
    df = pd.concat(parts)
    return sp.csr_matrix((np.ones(len(df), dtype=np.float32), (df["u"], df["i"])), shape=(n_users, n_items))


def fold_in(history, item_vecs, item_weight):
    """Weighted mean of the vectors of each user's liked films, as in taste.user_vector()"""
    weighted = history.multiply(item_weight[None, :]).tocsr()
    sums = np.asarray(weighted.sum(axis=1)).ravel()
    sums[sums == 0] = 1
    return (weighted @ item_vecs) / sums[:, None]


def main():
    train, val, test, meta = load_split()
    n_users, n_items = meta["n_users"], meta["n_items"]
    gems = set(range(n_items)) - set(meta["popular_items"])
    users = np.unique(test["u"])
    truth = ground_truth(test, n_users)
    mask = mask_matrix([train, val], n_users, n_items)
    history = history_matrix([train, val], n_users, n_items)

    def run(score_fn):
        return metrics(top_k_lists(score_fn, users, mask, k=50), users, truth, gems, n_items)

    rows = {}
    gen = torch.Generator().manual_seed(SEED)
    rows["Random"] = run(lambda u: torch.rand(len(u), n_items, generator=gen))
    counts = torch.as_tensor(np.bincount(train["i"], minlength=n_items), dtype=torch.float32)
    rows["Most popular"] = run(lambda u: counts.expand(len(u), -1))

    # Text only
    movies = pd.read_parquet(PROCESSED / "movies.parquet")
    row_of = pd.Series(np.arange(len(movies)), index=movies["movie_id"])
    graph_rows = row_of.loc[np.load(PROCESSED / "item_ids.npy")].to_numpy()
    text = np.load(PROCESSED / "text_emb.npy").astype(np.float32)[graph_rows]
    text_users = torch.as_tensor(fold_in(history, text, np.ones(n_items, dtype=np.float32)))
    text_items = torch.as_tensor(text)
    rows["Text only"] = run(lambda u: text_users[u] @ text_items.T)

    for name, label in [("lightgcn", "LightGCN"), ("nlgcl", "NLGCL")]:
        if not (MODELS / f"{name}.pt").exists():
            print(f"skip {name}: models/{name}.pt not found")
            continue
        model, _ = load_model(name, train, meta)
        user_emb, item_emb = model.final_embeddings()
        rows[label] = run(lambda u, U=user_emb, I=item_emb: U[u] @ I.T)

    if (ARTIFACTS / "item_emb.npy").exists():
        art = json.loads((ARTIFACTS / "meta.json").read_text())
        items = np.load(ARTIFACTS / "item_emb.npy")[graph_rows]
        likes = np.load(ARTIFACTS / "item_likes.npy")[graph_rows]
        users_fold = torch.as_tensor(fold_in(history, items, 1 / np.maximum(likes, 1)).astype(np.float32))
        items_t = torch.as_tensor(items)
        rows[f"{art['model'].upper()} + text, fold-in (app)"] = run(lambda u: users_fold[u] @ items_t.T)

    (RESULTS / "taste_results.json").write_text(json.dumps(rows, indent=1))
    table = pd.DataFrame(rows).T
    lines = ["| Method | " + " | ".join(table.columns) + " |", "|---" * (len(table.columns) + 1) + "|"]
    lines += [f"| {name} | " + " | ".join(f"{v:.4f}" for v in row) + " |" for name, row in table.iterrows()]
    (RESULTS / "taste_results.md").write_text("\n".join(lines) + "\n")
    print(f"test users: {len(users)}")
    print(table.to_string())


if __name__ == "__main__":
    main()

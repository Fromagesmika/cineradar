"""Full-ranking evaluation over all films (owner: Théodore)

Films already liked in train (and in val, when we evaluate on test) are masked
"""
import numpy as np
import torch


def ground_truth(df, n_users):
    truth = [[] for _ in range(n_users)]
    for u, i in zip(df["u"].to_numpy(), df["i"].to_numpy()):
        truth[u].append(i)
    return truth


def mask_matrix(dfs, n_users, n_items):
    """Boolean users x items mask of films to hide from the ranking"""
    mask = torch.zeros(n_users, n_items, dtype=torch.bool)
    for df in dfs:
        mask[torch.as_tensor(df["u"].to_numpy(), dtype=torch.long), torch.as_tensor(df["i"].to_numpy(), dtype=torch.long)] = True
    return mask


def top_k_lists(score_fn, users, mask, k=50, chunk=2048):
    """score_fn(user_tensor) -> scores (len(users) x n_items). Returns np.array of top-k item ids"""
    out = []
    for start in range(0, len(users), chunk):
        batch = torch.as_tensor(users[start:start + chunk], dtype=torch.long)
        scores = score_fn(batch).clone()
        scores[mask[batch]] = -torch.inf
        out.append(torch.topk(scores, k, dim=1).indices.numpy())
    return np.concatenate(out) if out else np.zeros((0, k), dtype=int)


def metrics(topk, users, truth, hidden_gems, n_items, ks=(10, 20)):
    """NDCG@10, Recall@10/20, hidden-gem recall, coverage@10 and hidden-gem share@10"""
    gems = np.zeros(n_items, dtype=bool)
    gems[list(hidden_gems)] = True
    discounts = 1 / np.log2(np.arange(2, 52))
    ndcg, rec = [], {k: [] for k in ks}
    gem_rec = []
    for row, u in zip(topk, users):
        rel = set(truth[u])
        if not rel:
            continue
        hits = np.array([i in rel for i in row[:max(ks)]])
        idcg = discounts[: min(len(rel), 10)].sum()
        ndcg.append((hits[:10] * discounts[:10]).sum() / idcg)
        for k in ks:
            rec[k].append(hits[:k].sum() / len(rel))
        rel_gems = {i for i in rel if gems[i]}
        if rel_gems:
            gem_rec.append(len(rel_gems & set(row[:20])) / len(rel_gems))
    top10 = topk[:, :10]
    result = {
        "NDCG@10": float(np.mean(ndcg)),
        **{f"Recall@{k}": float(np.mean(v)) for k, v in rec.items()},
        "GemRecall@20": float(np.mean(gem_rec)) if gem_rec else 0.0,
        "Coverage@10": float(len(np.unique(top10)) / n_items),
        "GemShare@10": float(gems[top10].mean()),
    }
    return {k: round(v, 4) for k, v in result.items()}

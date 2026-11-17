"""Evaluation metrics and simple baselines (owner: Théodore).

recommendations: {user_id: [movie_id, ...]}  ranked list, best first
ground_truth:    {user_id: {movie_id, ...}}  films liked in the test set
"""
import math
import random


def ndcg_at_k(recommendations, ground_truth, k=10):
    scores = []
    for user, relevant in ground_truth.items():
        if not relevant:
            continue
        ranked = recommendations.get(user, [])[:k]
        dcg = sum(1 / math.log2(i + 2) for i, item in enumerate(ranked) if item in relevant)
        idcg = sum(1 / math.log2(i + 2) for i in range(min(len(relevant), k)))
        scores.append(dcg / idcg)
    return sum(scores) / len(scores) if scores else 0.0


def recall_at_k(recommendations, ground_truth, k=20):
    scores = []
    for user, relevant in ground_truth.items():
        if not relevant:
            continue
        ranked = recommendations.get(user, [])[:k]
        scores.append(len(set(ranked) & relevant) / len(relevant))
    return sum(scores) / len(scores) if scores else 0.0


def coverage_at_k(recommendations, n_items, k=10):
    """Share of the catalogue recommended to at least one user."""
    shown = {item for ranked in recommendations.values() for item in ranked[:k]}
    return len(shown) / n_items


def hidden_gem_share_at_k(recommendations, hidden_gems, k=10):
    """Average share of hidden gems (non-popular films) in the top-k."""
    shares = [
        sum(item in hidden_gems for item in ranked[:k]) / k
        for ranked in recommendations.values()
        if ranked
    ]
    return sum(shares) / len(shares) if shares else 0.0


# ----- Baselines (the model must beat both) -----

def random_baseline(users, items, seen, k=20, seed=42):
    rng = random.Random(seed)
    items = list(items)
    recs = {}
    for user in users:
        candidates = [i for i in rng.sample(items, min(len(items), k + len(seen.get(user, ())))) if i not in seen.get(user, ())]
        recs[user] = candidates[:k]
    return recs


def most_popular_baseline(users, item_counts, seen, k=20):
    ranking = sorted(item_counts, key=item_counts.get, reverse=True)
    return {u: [i for i in ranking if i not in seen.get(u, ())][:k] for u in users}


if __name__ == "__main__":
    # Quick check with fake predictions
    recs = {1: [10, 20, 30], 2: [40, 50, 60]}
    truth = {1: {10, 99}, 2: {70}}
    print("NDCG@10  :", round(ndcg_at_k(recs, truth), 4))
    print("Recall@20:", round(recall_at_k(recs, truth), 4))

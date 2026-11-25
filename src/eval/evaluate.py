"""Evaluation harness for the whole system on real dialogues (part 1)

    evaluate(recommend_fn) -> pd.DataFrame (one row of metrics)

recommend_fn has the contract of src/reco/engine.py:
    recommend_fn(user_id, intent, k, cinephile) -> [{"movie_id", "score", "because_movie_id"}, ...]

Test cases come from the ReDial test seekers (src/data/redial.py), never seen in train:
- request: what the seeker wrote before the recommender suggested a film, read by the part 3
  extractor (rules mode, no API key) when it is there
- profile: the films the seeker mentioned and liked or disliked, passed as liked_ids / disliked_ids
- targets: the films the recommender suggested and the seeker did not dislike

Metrics
- Recall@1/10/50 and NDCG@10 on the targets
- GemShare@10: share of hidden gems (not in the 20% most rated MovieLens films) in the top 10
- Coverage@10: share of the catalog that appears in at least one top 10
- Jaccard@10: two different seekers send the same request ("an action film"); mean overlap of their
  top 10. Lower means more personal

The MovieLens offline results of the taste model are in results/taste_results.md
(src/model/evaluate_taste.py)

Run: python -m src.eval.evaluate             (baselines, taste model alone, full system, ablations)
Output: results/redial_results.json and results/redial_results.md
"""
import json
import os
import re
import time

import numpy as np
import pandas as pd

from src.config import PROCESSED, RESULTS, SEED

N_CASES = 300
N_PAIRS = 60
K_MAX = 50
SAME_REQUEST = {"intent": "recommend", "genres": ["Action"], "query": ""}


# ------------------------------------------------------------------ test cases

def _clean_text(text: str, names: dict) -> str:
    return re.sub(r"@(\d+)", lambda m: names.get(int(m.group(1)), "a movie"), text)


def build_cases(n_cases: int = N_CASES, seed: int = SEED) -> list[dict]:
    mentions = pd.read_parquet(PROCESSED / "redial_mentions.parquet")
    dialogues = pd.read_parquet(PROCESSED / "redial_dialogues.parquet").set_index("conversation_id")
    links = pd.read_parquet(PROCESSED / "redial_links.parquet")
    movies = pd.read_parquet(PROCESSED / "movies.parquet", columns=["movie_id", "clean_title"])
    title = dict(zip(movies["movie_id"], movies["clean_title"]))
    names = {int(r): title.get(m, re.sub(r"\s*\(\d{4}\)\s*$", "", n))
             for r, m, n in zip(links["redial_id"], links["movie_id"], links["redial_name"])}

    test = mentions[(mentions["split"] == "test") & mentions["movie_id"].notna()]
    cases = []
    for conv, group in test.groupby("conversation_id"):
        seeker = group[group["mentioned_by"] == "seeker"]
        targets = group[(group["mentioned_by"] == "recommender") & (group["liked"] >= 0)]
        if targets.empty:
            continue
        first_suggestion = int(targets["position"].min())
        messages = json.loads(dialogues.loc[conv, "messages"])
        request = " ".join(_clean_text(m["text"], names) for m in messages[:first_suggestion]
                           if m["speaker"] == "seeker")
        cases.append({
            "conversation_id": int(conv),
            "seeker_id": int(group["seeker_id"].iloc[0]),
            "request": request,
            "liked": [int(m) for m in seeker.loc[seeker["liked"] == 1, "movie_id"]],
            "disliked": [int(m) for m in seeker.loc[seeker["liked"] == -1, "movie_id"]],
            "targets": [int(m) for m in targets["movie_id"]],
        })
    rng = np.random.default_rng(seed)
    if len(cases) > n_cases:
        cases = [cases[i] for i in sorted(rng.choice(len(cases), n_cases, replace=False))]
    return cases


def _extractor():
    """Part 3 extractor in rules mode (no network) if it is there, else genre keywords only"""
    os.environ.setdefault("LLM_OFFLINE", "1")
    try:
        from src.llm.agent import extract
        return lambda text: extract(text, [])
    except ImportError:
        return lambda text: {"intent": "recommend", "query": text}


def case_intent(case: dict, extract) -> dict:
    intent = dict(extract(case["request"])) if case["request"].strip() else {"intent": "recommend"}
    intent["liked_ids"] = list(dict.fromkeys(list(intent.get("liked_ids") or []) + case["liked"]))
    intent["disliked_ids"] = list(dict.fromkeys(list(intent.get("disliked_ids") or []) + case["disliked"]))
    intent["n"] = K_MAX
    return intent


# ------------------------------------------------------------------ metrics

def _gems() -> set:
    movies = pd.read_parquet(PROCESSED / "movies.parquet", columns=["movie_id", "popularity_bucket"])
    return set(movies.loc[movies["popularity_bucket"] == "gem", "movie_id"])


def _ndcg(ranked, targets, k=10):
    gains = [1 / np.log2(i + 2) for i, m in enumerate(ranked[:k]) if m in targets]
    ideal = sum(1 / np.log2(i + 2) for i in range(min(len(targets), k)))
    return sum(gains) / ideal if ideal else 0.0


def _ids(recos) -> list[int]:
    return [int(r["movie_id"]) for r in recos]


def evaluate(recommend_fn, cases: list[dict] | None = None, n_pairs: int = N_PAIRS, cinephile: float = 0.5,
             catalog_size: int | None = None) -> pd.DataFrame:
    cases = cases if cases is not None else build_cases()
    extract = _extractor()
    gems = _gems()
    recall = {1: [], 10: [], 50: []}
    ndcg, gem_share, shown = [], [], set()
    start = time.time()
    for i, case in enumerate(cases):
        # negative ids: users that do not exist in the app database, so only the case's profile counts
        ranked = _ids(recommend_fn(-(i + 1), case_intent(case, extract), K_MAX, cinephile))
        targets = set(case["targets"])
        for k in recall:
            recall[k].append(len(targets & set(ranked[:k])) / len(targets))
        ndcg.append(_ndcg(ranked, targets))
        top = ranked[:10]
        gem_share.append(np.mean([m in gems for m in top]) if top else 0.0)
        shown |= set(top)

    # personalization: same request, two different seekers with at least 3 liked films
    rng = np.random.default_rng(SEED)
    profiles = [c for c in cases if len(c["liked"]) >= 3]
    jaccard = []
    for _ in range(n_pairs if len(profiles) >= 2 else 0):
        a, b = rng.choice(len(profiles), 2, replace=False)
        lists = []
        for j, case in ((a, profiles[a]), (b, profiles[b])):
            intent = {**SAME_REQUEST, "liked_ids": case["liked"], "disliked_ids": case["disliked"]}
            lists.append(set(_ids(recommend_fn(-(10_000 + int(j)), intent, 10, cinephile))))
        union = lists[0] | lists[1]
        jaccard.append(len(lists[0] & lists[1]) / len(union) if union else 1.0)

    if catalog_size is None:
        catalog_size = len(pd.read_parquet(PROCESSED / "movies.parquet", columns=["movie_id"]))
    return pd.DataFrame([{
        "Recall@1": np.mean(recall[1]), "Recall@10": np.mean(recall[10]), "Recall@50": np.mean(recall[50]),
        "NDCG@10": np.mean(ndcg), "GemShare@10": np.mean(gem_share),
        "Coverage@10": len(shown) / catalog_size, "Jaccard@10": np.mean(jaccard) if jaccard else np.nan,
        "cases": len(cases), "sec/request": (time.time() - start) / max(len(cases), 1),
    }])


# ------------------------------------------------------------------ baselines

def random_recommend(seed: int = SEED):
    movies = pd.read_parquet(PROCESSED / "movies.parquet", columns=["movie_id", "n_ratings"])
    pool = movies.loc[movies["n_ratings"] >= 20, "movie_id"].to_numpy()
    rng = np.random.default_rng(seed)

    def rec(user_id, intent, k=10, cinephile=0.5):
        return [{"movie_id": int(m), "score": 0.0, "because_movie_id": None} for m in rng.choice(pool, k, replace=False)]
    return rec


def popular_recommend():
    movies = pd.read_parquet(PROCESSED / "movies.parquet", columns=["movie_id", "n_ratings"])
    ranked = movies.sort_values("n_ratings", ascending=False)["movie_id"].astype(int).tolist()

    def rec(user_id, intent, k=10, cinephile=0.5):
        blocked = set(intent.get("liked_ids") or []) | set(intent.get("disliked_ids") or [])
        return [{"movie_id": m, "score": 0.0, "because_movie_id": None} for m in ranked[: k + len(blocked)]
                if m not in blocked][:k]
    return rec


def taste_recommend():
    """Part 2 alone: dot product with the user's taste vector, among films with 20+ ratings"""
    from src.model import taste
    movies = pd.read_parquet(PROCESSED / "movies.parquet", columns=["movie_id", "n_ratings"])
    pool = movies.loc[movies["n_ratings"] >= 20, "movie_id"].astype(int).to_numpy()
    popular = popular_recommend()

    def rec(user_id, intent, k=10, cinephile=0.5):
        liked, disliked = intent.get("liked_ids") or [], intent.get("disliked_ids") or []
        if not liked:
            return popular(user_id, intent, k)
        scores = taste.score(taste.user_vector(liked, disliked), pool)
        scores[np.isin(pool, liked + disliked)] = -np.inf
        top = np.argsort(-scores)[:k]
        return [{"movie_id": int(pool[i]), "score": float(scores[i]), "because_movie_id": None} for i in top]
    return rec


def main():
    cases = build_cases()
    print(f"{len(cases)} test dialogues, {sum(len(c['targets']) for c in cases)} target films")
    systems = {"Random": random_recommend(), "Most popular": popular_recommend(),
               "Taste model alone (part 2)": taste_recommend()}
    try:
        from src.reco import engine
        from src.reco.memory import InMemoryProvider
        engine.set_memory_provider(InMemoryProvider())  # empty memory: only the dialogue counts
        labels = {"llm_seul": "Request only (part 3, no taste)", "+gout": "+ taste vectors",
                  "+descriptions": "+ film descriptions", "+memoire": "+ memory",
                  "+pepites (complet)": "+ hidden-gem re-ranking (full system)"}
        for key, config in engine.ABLATIONS.items():
            systems[labels.get(key, key)] = engine.make_recommend(config)
        full = engine.make_recommend(engine.FULL)
        systems["Full system, slider on hidden gems (0.9)"] = (
            lambda user_id, intent, k=10, cinephile=0.5: full(user_id, intent, k, 0.9))
    except ImportError:
        print("part 3 not found: baselines only")

    rows = []
    for name, fn in systems.items():
        row = evaluate(fn, cases)
        row.insert(0, "System", name)
        rows.append(row)
        print(row.to_string(index=False, header=len(rows) == 1))
    table = pd.concat(rows, ignore_index=True)
    (RESULTS / "redial_results.json").write_text(table.to_json(orient="records", indent=1))
    cols = [c for c in table.columns if c != "System"]
    lines = ["| System | " + " | ".join(cols) + " |", "|---" * (len(cols) + 1) + "|"]
    for _, r in table.iterrows():
        lines.append(f"| {r['System']} | " + " | ".join(
            f"{r[c]:.0f}" if c == "cases" else f"{r[c]:.4f}" for c in cols) + " |")
    (RESULTS / "redial_results.md").write_text("\n".join(lines) + "\n")
    print("saved results/redial_results.md")


if __name__ == "__main__":
    main()

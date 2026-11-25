"""Ablations : LLM seul / + goût / + descriptions / + mémoire / + pépites.

    python -m src.reco.ablations

- Si le harnais de la partie 1 existe (src/eval/evaluate.py avec une fonction `evaluate(recommend_fn)`),
  chaque variante y est branchée et on concatène les tableaux.
- Sinon, petit banc d'essai interne sur profils synthétiques (contrôle de cohérence, PAS les
  résultats officiels du rapport) : Recall@10 sur films cachés, part de pépites, couverture,
  et personnalisation (Jaccard des top 10 entre profils pour la même demande : plus bas = plus personnalisé).
"""
from __future__ import annotations

import importlib
import itertools

import numpy as np
import pandas as pd

from .catalog import get_catalog
from .engine import ABLATIONS, make_recommend, recommend_detailed
from .memory import build_memory


def _official(variants) -> pd.DataFrame | None:
    try:
        ev = importlib.import_module("src.eval.evaluate")
    except Exception:
        return None
    fn = getattr(ev, "evaluate", None)
    if fn is None:
        return None
    rows = []
    for name, cfg in variants.items():
        res = fn(make_recommend(cfg))
        res = res if isinstance(res, pd.DataFrame) else pd.DataFrame([res])
        res.insert(0, "variante", name)
        rows.append(res)
    return pd.concat(rows, ignore_index=True)


def _synthetic_profiles(n_users: int = 40, n_liked: int = 5, n_hidden: int = 2, seed: int = 0):
    """Chaque profil part d'un film « graine » et aime ses voisins (contenu + genres)."""
    cat = get_catalog()
    rng = np.random.default_rng(seed)
    sims = cat.content_emb @ cat.content_emb.T
    sims = np.asarray(sims.todense()) if hasattr(sims, "todense") else np.asarray(sims)
    sims = sims + 0.5 * (cat.genre_matrix @ cat.genre_matrix.T) / np.maximum(cat.genre_matrix.sum(1), 1)
    profiles = []
    for u in range(n_users):
        seed_i = rng.integers(len(cat))
        neigh = np.argsort(-sims[seed_i])[: n_liked + n_hidden + 3]
        neigh = rng.permutation(neigh[: n_liked + n_hidden])
        liked = [int(cat.ids[i]) for i in neigh[:n_liked]]
        hidden = [int(cat.ids[i]) for i in neigh[n_liked:]]
        h = cat.get(hidden[0])
        g = h["genres"][:1]
        q = h["tags"][0] if h["tags"] and rng.random() < 0.5 else ""  # une demande sur deux avec une ambiance
        profiles.append({"user_id": 10_000 + u, "liked": liked, "hidden": hidden, "genre": g, "query": q,
                         "cinephile": float(rng.choice([0.2, 0.5, 0.8]))})
    return profiles


def _internal(variants, k: int = 10) -> pd.DataFrame:
    cat = get_catalog()
    profiles = _synthetic_profiles()
    rows = []
    for name, cfg in variants.items():
        recalls, gems, shown, action_lists = [], [], set(), []
        for p in profiles:
            mem = build_memory(p["user_id"], [{"movie_id": m, "type": "onboarding_like"} for m in p["liked"]],
                               cinephile=p["cinephile"])
            recos = recommend_detailed(p["user_id"], {"genres": p["genre"], "query": p["query"]}, k=k,
                                       cinephile=p["cinephile"], config=cfg, memory=mem)
            ids = [r["movie_id"] for r in recos]
            recalls.append(len(set(ids) & set(p["hidden"])) / len(p["hidden"]))
            gems.append(np.mean([cat.get(m)["popularity"] <= 0.25 for m in ids]) if ids else 0)
            shown |= set(ids)
            same = recommend_detailed(p["user_id"], {"genres": ["Action"]}, k=k,
                                      cinephile=p["cinephile"], config=cfg, memory=mem)
            action_lists.append({r["movie_id"] for r in same})
        jac = [len(a & b) / max(len(a | b), 1) for a, b in itertools.combinations(action_lists, 2)]
        rows.append({
            "variante": name,
            "Recall@10": round(float(np.mean(recalls)), 3),
            "part_pepites": round(float(np.mean(gems)), 3),
            "couverture": round(len(shown) / len(cat), 3),
            "Jaccard_meme_demande": round(float(np.mean(jac)), 3),
        })
    return pd.DataFrame(rows)


def run(variants=None) -> pd.DataFrame:
    variants = variants or ABLATIONS
    df = _official(variants)
    if df is None:
        print("(harnais de la partie 1 absent -> banc d'essai interne sur profils synthétiques)")
        df = _internal(variants)
    return df


if __name__ == "__main__":
    pd.set_option("display.width", 140)
    print(run().to_string(index=False))

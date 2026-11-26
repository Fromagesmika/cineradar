"""Moteur de recommandation (partie 3).

    recommend(user_id, intent, k=10, cinephile=0.5) -> [{"movie_id", "score", "because_movie_id"}, ...]

Pipeline :
  1. Mémoire utilisateur (questionnaire + clics + notes + chat), lue via un MemoryProvider.
  2. Génération de candidats selon l'intention : genres, mots-clés/tags, recherche sémantique,
     films de référence (« comme Heat »), période.
  3. Classement : intention + goût (partie 2) + mémoire (affinités de genres, films refusés,
     lassitude) puis re-ranking « pépites » piloté par le curseur cinéphile.
  4. Diversification légère et explication (« parce que tu as aimé… »).

Chaque brique peut être coupée via EngineConfig pour les ablations.
"""
from __future__ import annotations

import importlib
import os
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from .catalog import CANONICAL_GENRES, get_catalog, normalize_text
from .memory import InMemoryProvider, MemoryProvider, SQLiteProvider, UserMemory

# --------------------------------------------------------------------------- goût (partie 2)


def _load_taste():
    """Utilise src/model/taste.py (partie 2) s'il existe, sinon le mock."""
    if os.getenv("TASTE_BACKEND") != "mock":
        try:
            mod = importlib.import_module("src.model.taste")
            if hasattr(mod, "user_vector") and hasattr(mod, "score"):
                return mod
        except Exception:
            pass
    from . import taste_mock

    return taste_mock


taste = _load_taste()


def _safe_taste_scores(user_vec, movie_ids) -> np.ndarray:
    try:
        s = np.asarray(taste.score(user_vec, list(map(int, movie_ids))), dtype=float)
        return np.nan_to_num(s)
    except Exception:
        return np.zeros(len(movie_ids))


_UNIT_CACHE: dict = {}


def _taste_cosine(ref_ids, movie_ids) -> np.ndarray:
    """Similarité cosinus entre les films de référence et chaque film. Le produit scalaire favorise
    les films très populaires (vecteurs de grande norme) : pour « comme X », on veut des films proches de X"""
    if not hasattr(taste, "item_vectors"):
        return _safe_taste_scores(taste.user_vector(ref_ids, []), movie_ids)
    key = (id(taste), id(movie_ids), len(movie_ids))
    if key not in _UNIT_CACHE:
        vecs, known = taste.item_vectors(list(map(int, movie_ids)))
        unit = vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9)
        _UNIT_CACHE.clear()
        _UNIT_CACHE[key] = (unit, {int(m): i for i, m in enumerate(movie_ids)}, known)
    unit, row, known = _UNIT_CACHE[key]
    refs = [row[int(m)] for m in ref_ids if int(m) in row and known[row[int(m)]]]
    if not refs:
        return np.zeros(len(movie_ids))
    query = unit[refs].mean(axis=0)
    return np.where(known, unit @ query, 0.0)


# --------------------------------------------------------------------------- mémoire

_provider: MemoryProvider | None = None
DEMO_MEMORY = InMemoryProvider()


def set_memory_provider(p: MemoryProvider) -> None:
    global _provider
    _provider = p


def get_memory_provider() -> MemoryProvider:
    global _provider
    if _provider is None:
        from src.service import db as app_db  # base de la partie 4 (data/cineradar.db)

        path = os.getenv("DB_PATH", str(app_db.DB_PATH))
        _provider = SQLiteProvider(path) if Path(path).exists() else DEMO_MEMORY
    return _provider


# --------------------------------------------------------------------------- config


@dataclass(frozen=True)
class EngineConfig:
    use_taste: bool = True      # vecteurs de goût (partie 2)
    use_content: bool = True    # descriptions : recherche sémantique sur résumés + tags
    use_memory: bool = True     # matrice de mémoire (affinités genres, refus, lassitude)
    use_gems: bool = True       # re-ranking pépites selon le curseur
    use_diversity: bool = True
    w_intent: float = 1.0
    w_taste: float = 0.8
    w_memory: float = 0.5
    w_gems: float = 0.35
    w_quality: float = 0.5
    w_diversity: float = 0.15
    pool_size: int = 300


FULL = EngineConfig()
ABLATIONS = {
    "llm_seul": EngineConfig(use_taste=False, use_content=False, use_memory=False, use_gems=False, use_diversity=False),
    "+gout": EngineConfig(use_content=False, use_memory=False, use_gems=False, use_diversity=False),
    "+descriptions": EngineConfig(use_memory=False, use_gems=False, use_diversity=False),
    "+memoire": EngineConfig(use_gems=False, use_diversity=False),
    "+pepites (complet)": FULL,
}


# --------------------------------------------------------------------------- helpers


_GENRE_WORDS = {w for g in CANONICAL_GENRES for w in normalize_text(g, False).split()} | {
    "film", "films", "movie", "movies", "un", "une", "de", "d", "des", "ce", "soir", "tonight", "a", "an",
    "comedie", "drame", "horreur", "guerre", "aventure", "fantastique", "polar", "sf", "science", "fiction",
    "romantique", "policier", "thriller", "documentaire", "famille", "animation", "mystere", "musical"}


def _minmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return x
    lo, hi = x.min(), x.max()
    return (x - lo) / (hi - lo) if hi - lo > 1e-12 else np.zeros_like(x)


def _norm_genres(gs) -> list[str]:
    out = []
    for g in gs or []:
        for cg in CANONICAL_GENRES:
            if normalize_text(g) == normalize_text(cg):
                out.append(cg)
    return list(dict.fromkeys(out))


def _ids(xs) -> list[int]:
    out = []
    for x in xs or []:
        try:
            out.append(int(x))
        except (TypeError, ValueError):
            pass
    return out


# --------------------------------------------------------------------------- coeur


def recommend_detailed(
    user_id: int,
    intent: dict | None,
    k: int = 10,
    cinephile: float | None = 0.5,
    config: EngineConfig = FULL,
    memory: UserMemory | None = None,
) -> list[dict]:
    """Comme recommend() mais renvoie aussi le détail des scores (debug, ablations, rapport)."""
    cat = get_catalog()
    intent = dict(intent or {})
    mem = memory if memory is not None else get_memory_provider().get(user_id)
    ids = cat.ids
    n = len(ids)

    # ---- curseur : valeur passée par l'appli, sinon mémoire, + décalage demandé dans le message
    c = cinephile if cinephile is not None else (mem.cinephile if mem.cinephile is not None else 0.5)
    c = float(np.clip(c + float(intent.get("cinephile_shift") or 0.0), 0.0, 1.0))

    # ---- signaux de l'intention
    genres = _norm_genres(intent.get("genres"))
    excluded_genres = set(_norm_genres(intent.get("excluded_genres")))
    if config.use_memory:
        excluded_genres |= mem.excluded_genres() - set(genres)
    ref_ids = [m for m in _ids(intent.get("reference_ids")) if m in cat.id_to_row]
    msg_liked = _ids(intent.get("liked_ids"))
    msg_disliked = _ids(intent.get("disliked_ids"))
    keywords = [normalize_text(w) for w in intent.get("keywords") or [] if normalize_text(w)]
    query = " ".join([intent.get("query") or "", *keywords]).strip()
    if intent.get("source") == "rules":
        # en mode règles, `query` est le message brut (souvent en français, avec « salut », « ce soir »)
        # alors que les résumés sont en anglais : seuls les mots-clés traduits par les règles sont utiles
        query = " ".join(keywords)
    # une demande qui ne contient qu'un genre (« un film d'action ») est déjà couverte par le filtre
    # de genre : la recherche sur le mot « action » ne ferait que favoriser les résumés qui le contiennent
    if genres and not set(normalize_text(query, False).split()) - _GENRE_WORDS:
        query = ""

    # origine (« un film coréen ») et personnes (« un film de Nolan », « avec Tom Hanks »)
    origins = {str(o) for o in intent.get("origins") or [] if o}
    directors = [str(d) for d in intent.get("directors") or [] if d]
    actors = [str(a) for a in intent.get("actors") or [] if a]
    origin_match = np.isin(cat.origins, list(origins)).astype(float) if origins and hasattr(cat, "origins") else np.zeros(n)
    people_match = (cat.people_rows(directors, actors).astype(float)
                    if (directors or actors) and hasattr(cat, "people_rows") else np.zeros(n))
    wants_people = bool(people_match.any())
    wants_origin = bool(origin_match.any())

    liked = list(dict.fromkeys(msg_liked + (mem.liked_ids if config.use_memory or config.use_taste else [])))
    disliked = list(dict.fromkeys(msg_disliked + (mem.disliked_ids if config.use_memory or config.use_taste else [])))

    # ---- filtres durs
    mask = np.ones(n, dtype=bool)
    blocked = set(liked) | set(disliked) | set(ref_ids) | set(_ids(intent.get("exclude_ids")))
    blocked |= mem.seen | mem.not_interested
    if blocked:
        mask &= ~np.isin(ids, np.fromiter(blocked, dtype=ids.dtype))
    if excluded_genres:
        ex = np.array([CANONICAL_GENRES.index(g) for g in excluded_genres])
        mask &= cat.genre_matrix[:, ex].sum(axis=1) == 0
    years = cat.df["year"].to_numpy(dtype=float)

    def year_mask(slack: float) -> np.ndarray:
        ym = np.ones(n, dtype=bool)
        if intent.get("year_min"):
            ym &= np.nan_to_num(years, nan=0) >= float(intent["year_min"]) - slack
        if intent.get("year_max"):
            ym &= np.nan_to_num(years, nan=9999) <= float(intent["year_max"]) + slack
        return ym

    # ---- score d'intention (calculé sur tout le catalogue, vectorisé)
    if genres:
        gi = np.array([CANONICAL_GENRES.index(g) for g in genres])
        genre_match = cat.genre_matrix[:, gi].sum(axis=1) / len(gi)
    else:
        genre_match = np.zeros(n)

    tag_match = np.zeros(n)
    if keywords:
        for i, h in enumerate(cat.keyword_haystack):
            tag_match[i] = sum(1 for w in keywords if w in h) / len(keywords)

    sem = cat.semantic_scores(query) if (config.use_content and query) else np.zeros(n)

    ref_sim = np.zeros(n)
    if ref_ids:
        if config.use_content:
            ref_sim += _minmax(cat.content_similarity(ref_ids))
        if config.use_taste:
            ref_sim += _minmax(_taste_cosine(ref_ids, ids))
        if not config.use_content and not config.use_taste:  # LLM seul : genres partagés
            ref_g = cat.genre_matrix[[cat.id_to_row[m] for m in ref_ids]].max(axis=0)
            ref_sim = cat.genre_matrix @ ref_g / max(ref_g.sum(), 1)

    intent_score = (1.0 * genre_match + 0.6 * tag_match + 1.0 * _minmax(sem) + 1.2 * _minmax(ref_sim)
                    + 1.5 * origin_match + 2.0 * people_match)

    # ---- pool de candidats (on relâche les contraintes si trop peu de résultats)
    def pool(require_genre: bool, slack: float | None, require_origin: bool = True, require_people: bool = True):
        m = mask.copy()
        if require_genre and genres:
            m &= genre_match > 0
        if slack is not None:
            m &= year_mask(slack)
        if require_origin and wants_origin:
            m &= origin_match > 0
        if require_people and wants_people:
            m &= people_match > 0
        return np.where(m)[0]

    # relâchement progressif : période élargie, puis sans période, puis sans genre imposé, et en dernier
    # recours sans l'origine ni la personne demandées (ce sont les contraintes les plus explicites)
    steps = [(True, 0), (True, 5), (True, 10), (True, None), (False, 0), (False, None),
             (False, None, False, True), (False, None, False, False)]
    # une personne n'a souvent que quelques films : mieux vaut 3 films de Nolan que 10 films au hasard
    enough = min(k, 3) if wants_people else k
    cand = pool(*steps[0])
    for step in steps[1:]:
        if len(cand) >= enough:
            break
        cand = pool(*step)
    if len(cand) == 0:
        return []
    # ---- goût (partie 2)
    user_vec = None
    if config.use_taste:
        if liked or disliked:
            user_vec = taste.user_vector(liked[:50], disliked[:50])
        elif mem.embedding is not None:
            user_vec = mem.embedding
    has_taste = user_vec is not None and np.any(user_vec)

    # à intention égale (ex. tous les films d'action), le goût départage les candidats, pas la
    # popularité : sinon le pool ne contient que les films les plus connus et le curseur n'a plus d'effet
    tie = 1e-3 * cat.df["pop"].to_numpy()[cand]
    if has_taste:
        tie = tie + 0.5 * _minmax(_safe_taste_scores(user_vec, ids[cand]))
    order = np.argsort(-(intent_score[cand] + tie))
    cand = cand[order[: config.pool_size]]
    cand_ids = ids[cand]
    taste_s = _safe_taste_scores(user_vec, cand_ids) if has_taste else np.zeros(len(cand))

    # ---- mémoire : affinités de genres, films refusés, lassitude
    mem_s = np.zeros(len(cand))
    if config.use_memory:
        gv = mem.genre_vector()
        G = cat.genre_matrix[cand]
        mem_s += (G @ gv) / np.sqrt(np.maximum(G.sum(axis=1), 1))
        if mem.not_interested:
            ni = cat.content_similarity(list(mem.not_interested))[cand]
            mem_s -= 0.8 * ni
        fatigue = np.array([mem.shown.get(int(m), 0) for m in cand_ids], dtype=float)
        mem_s -= 0.3 * np.minimum(fatigue, 3)

    I, T, M = _minmax(intent_score[cand]), _minmax(taste_s), mem_s
    # « comme X » : la référence prime sur le goût général
    damp = 0.5 if ref_ids else 1.0
    rel = config.w_intent * I
    if config.use_taste:
        rel = rel + damp * config.w_taste * T
    if config.use_memory:
        rel = rel + damp * config.w_memory * np.clip(M, -1.5, 1.5)

    # ---- pépites : c=1 pousse la longue traîne, c=0 pousse les classiques, c=0.5 neutre
    pop = cat.df["pop"].to_numpy()[cand]
    quality = cat.df["quality"].to_numpy()[cand]
    gem_adj = (2 * c - 1) * (0.5 - pop) * 2 if config.use_gems else np.zeros(len(cand))
    if config.use_gems:
        # une pépite doit être bien notée : le bonus « peu connu » est pondéré par la qualité,
        # et la qualité compte d'autant plus que l'utilisateur est cinéphile
        gem_adj = np.where(gem_adj > 0, gem_adj * 2 * quality, gem_adj)
    final = rel + config.w_gems * gem_adj + 1e-3 * pop
    if config.use_gems:
        # films mal notés reculent, films très bien notés avancent (note moyenne MovieLens)
        final = final + config.w_quality * (quality - 0.5)

    # ---- sélection avec diversification (MMR léger sur le contenu)
    top = np.argsort(-final)[: max(5 * k, 50)]
    chosen: list[int] = []
    if config.use_diversity and len(top) > k:
        E = cat.content_emb[cand[top]]
        S = E @ E.T
        S = np.asarray(S.todense()) if hasattr(S, "todense") else np.asarray(S)
        # même règle qu'avant (pénalité = similarité max avec un film déjà choisi), en vectorisé
        base = final[top].astype(float)
        penalty = np.zeros(len(top))
        free = np.ones(len(top), dtype=bool)
        # « asiatique » : varier les pays plutôt que 10 films japonais
        top_origin = cat.origins[cand[top]] if len(origins) > 1 else None
        same_origin = np.zeros(len(top))
        while free.any() and len(chosen) < k:
            values = np.where(free, base - config.w_diversity * penalty - 0.12 * same_origin, -np.inf)
            best = int(np.argmax(values))
            chosen.append(top[best])
            free[best] = False
            penalty = np.maximum(penalty, S[:, best])
            if top_origin is not None:
                same_origin += top_origin == top_origin[best]
    else:
        chosen = list(top[:k])

    chosen_ids = [int(cand_ids[j]) for j in chosen]
    because = _because(chosen_ids, ref_ids + [m for m in liked if m not in ref_ids], config)

    out = []
    for j, mid, b in zip(chosen, chosen_ids, because):
        out.append({
            "movie_id": mid,
            "score": round(float(final[j]), 4),
            "because_movie_id": b,
            "_detail": {
                "intent": round(float(I[j]), 3),
                "taste": round(float(T[j]), 3),
                "memory": round(float(M[j]), 3),
                "gem": round(float(gem_adj[j]), 3),
                "popularity": round(float(pop[j]), 2),
                "cinephile": c,
            },
        })
    return out


def _because(reco_ids: list[int], anchors: list[int], config: EngineConfig) -> list[int | None]:
    """Pour chaque reco, le film aimé (ou de référence) le plus proche — pour « parce que tu as aimé… »."""
    cat = get_catalog()
    anchors = [a for a in anchors if a in cat.id_to_row][:30]
    if not anchors or not reco_ids:
        return [None] * len(reco_ids)
    sims = np.zeros((len(anchors), len(reco_ids)))
    rows_r = [cat.id_to_row[r] for r in reco_ids]
    for a_i, a in enumerate(anchors):
        cs = cat.content_emb[cat.id_to_row[a]] @ cat.content_emb[rows_r].T
        cs = np.asarray(cs.todense()).ravel() if hasattr(cs, "todense") else np.asarray(cs).ravel()
        ts = _safe_taste_scores(taste.user_vector([a], []), reco_ids) if config.use_taste else 0
        ga = cat.genre_matrix[cat.id_to_row[a]]
        gs = cat.genre_matrix[rows_r] @ ga / max(ga.sum(), 1)
        sims[a_i] = cs + 0.5 * np.asarray(ts) + 0.3 * gs
    best = sims.argmax(axis=0)
    out = []
    for j, b in enumerate(best):
        out.append(int(anchors[b]) if sims[b, j] > 0.35 else None)
    return out


def recommend(user_id: int, intent: dict, k: int = 10, cinephile: float = 0.5, **kwargs) -> list[dict]:
    """Fonction du contrat. Renvoie [{"movie_id", "score", "because_movie_id"}]."""
    recos = recommend_detailed(user_id, intent, k=k, cinephile=cinephile, **kwargs)
    return [{key: r[key] for key in ("movie_id", "score", "because_movie_id")} for r in recos]


def make_recommend(config: EngineConfig):
    """Fabrique une fonction recommend() pour une variante d'ablation (branchable sur le harnais de la partie 1)."""

    def _rec(user_id: int, intent: dict, k: int = 10, cinephile: float = 0.5, **kw):
        return recommend(user_id, intent, k=k, cinephile=cinephile, config=config, **kw)

    return _rec


__all__ = [
    "recommend", "recommend_detailed", "make_recommend", "EngineConfig", "ABLATIONS", "FULL",
    "set_memory_provider", "get_memory_provider", "DEMO_MEMORY", "replace",
]

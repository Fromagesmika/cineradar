import importlib
import json
import re

import numpy as np

from src.model import taste
from src.service import catalog, db, profile

GEM_STRENGTH = 0.8
SHORTLIST = 150
SUGGESTED_PENALTY = 0.5
GENRE_BONUS = 0.25
FRIEND = "Radar"
GENRE_FR = {"Action": "action", "Adventure": "aventure", "Animation": "animation", "Children": "famille",
            "Comedy": "comédie", "Crime": "polar", "Documentary": "documentaire", "Drama": "drame",
            "Fantasy": "fantasy", "Film-Noir": "film noir", "Horror": "horreur", "Musical": "comédie musicale",
            "Mystery": "mystère", "Romance": "romance", "Sci-Fi": "science-fiction", "Thriller": "thriller",
            "War": "guerre", "Western": "western"}

GENRE_WORDS = {
    "Action": ["action"],
    "Adventure": ["aventure", "aventures", "adventure"],
    "Animation": ["animation", "anime", "dessin anime", "cartoon"],
    "Children": ["enfant", "enfants", "famille", "familial", "kids", "children", "family"],
    "Comedy": ["comedie", "comedies", "comedy", "drole", "funny", "rire"],
    "Crime": ["crime", "policier", "polar", "gangster", "mafia", "braquage", "heist"],
    "Documentary": ["documentaire", "documentary", "docu"],
    "Drama": ["drame", "drama", "dramatique"],
    "Fantasy": ["fantasy", "fantastique", "magie", "magic"],
    "Film-Noir": ["film noir"],
    "Horror": ["horreur", "horror", "epouvante", "flippant", "scary", "peur"],
    "Musical": ["comedie musicale", "musical"],
    "Mystery": ["mystere", "mystery", "enquete"],
    "Romance": ["romance", "romantique", "romantic", "love story", "histoire d amour"],
    "Sci-Fi": ["sf", "science fiction", "sci fi", "scifi", "espace", "space"],
    "Thriller": ["thriller", "suspense"],
    "War": ["guerre", "war"],
    "Western": ["western", "westerns"],
}
DISLIKE_CUES = ["deteste", "detestais", "pas aime", "aime pas", "pas adore", "pas fan", "nul", "decu", "ennuye",
                "hate", "hated", "didn t like", "don t like", "do not like", "disliked", "boring", "awful"]
LIKE_CUES = ["adore", "aime", "aimais", "kiffe", "kiff", "fan de", "favori", "prefere", "coup de coeur",
             "i love", "i loved", "i like", "i liked", "i enjoyed", "i really like", "my favorite", "fan of"]
LOCAL_NEGATIONS = ["pas de", "pas d", "sans", "pas trop", "no", "not", "without", "rien de", "evite"]
GEM_CUES = ["pepite", "peu connu", "meconnu", "inconnu", "surprends", "surprise", "original", "obscure",
            "hidden gem", "underrated", "sous cote", "pointu", "cinephile", "rare"]
MAINSTREAM_CUES = ["connu", "populaire", "grand public", "blockbuster", "mainstream", "safe", "classique du"]
FRENCH_WORDS = {"je", "j", "un", "une", "le", "la", "les", "des", "pour", "ce", "soir", "film", "moi", "tu", "et",
                "avec", "pas", "veux", "envie", "aime", "quel", "quelque", "chose", "mais", "bonjour", "salut"}
COMMON_WORDS = {"action", "drama", "love", "crime", "horror", "comedy", "war", "romance", "western", "thriller",
                "tonight", "movie", "film", "films", "something", "anything", "space", "family", "mystery", "rare",
                "original", "the movie", "this", "that", "home", "time", "life", "help", "nice", "cool", "fear"}


def _clauses(message: str) -> list[str]:
    return [c for c in re.split(r"[.;!?\n]+|,|\bmais\b|\bbut\b|\bpar contre\b", message, flags=re.I) if c.strip()]


def _has(text: str, cues) -> bool:
    padded = f" {text} "
    return any(f" {cue} " in padded for cue in cues)


def _genres_in(text: str):
    padded = f" {text} "
    found = []
    for genre, words in GENRE_WORDS.items():
        for word in words:
            start = padded.find(f" {word} ")
            if start >= 0:
                found.append((genre, padded[:start].strip()))
                break
    if any(g == "Musical" for g, _ in found):
        found = [(g, p) for g, p in found if g != "Comedy" or "comedie musicale" not in text]
    return found


def _titles_in(clause_key: str):
    films = catalog.load()
    out = []
    for movie_id, start, end in catalog.find_titles(clause_key):
        key = films.loc[movie_id, "short_key"]
        if key in COMMON_WORDS or any(key == w for words in GENRE_WORDS.values() for w in words):
            continue
        if " " not in key and not films.loc[movie_id, "in_graph"]:
            continue
        out.append((movie_id, clause_key[:start].strip()))
    return out


def _years(key: str):
    decade = re.search(r"\b(?:annees|years)\s*(\d0)\b|\b(\d0)s\b|\b(19\d0|20[0-2]0)s?\b", key)
    if decade:
        value = next(g for g in decade.groups() if g)
        start = int(value) if len(value) == 4 else (1900 + int(value) if int(value) >= 20 else 2000 + int(value))
        return start, start + 9
    if _has(key, ["recent", "recents", "nouveau", "nouveaux", "new", "latest", "dernier", "derniers"]):
        return 2010, None
    if _has(key, ["vieux", "ancien", "anciens", "old", "classique", "classic", "classics", "noir et blanc"]):
        return None, 1980
    return None, None


def fallback_extract(message: str, history=None) -> dict:
    result = {
        "intent": {"genres": [], "avoid_genres": [], "reference_ids": [], "min_year": None, "max_year": None,
                   "cinephile_shift": 0.0},
        "likes": [], "dislikes": [], "genres_liked": [], "genres_avoided": [], "facts": {},
    }
    intent = result["intent"]
    for clause in _clauses(message):
        key = catalog.normalize(clause)
        for movie_id, before in _titles_in(key):
            if _has(before, DISLIKE_CUES):
                result["dislikes"].append(movie_id)
            elif _has(before, LIKE_CUES):
                result["likes"].append(movie_id)
            else:
                intent["reference_ids"].append(movie_id)
        for genre, before in _genres_in(key):
            tail = " ".join(before.split()[-3:])
            if _has(before, DISLIKE_CUES):
                result["genres_avoided"].append(genre)
            elif _has(tail, LOCAL_NEGATIONS):
                intent["avoid_genres"].append(genre)
            elif _has(before, LIKE_CUES) and not _has(before, ["envie", "want", "veux", "cherche", "looking"]):
                result["genres_liked"].append(genre)
            else:
                intent["genres"].append(genre)
    key = catalog.normalize(message)
    intent["min_year"], intent["max_year"] = _years(key)
    if _has(key, GEM_CUES):
        intent["cinephile_shift"] = 0.3
    elif _has(key, MAINSTREAM_CUES):
        intent["cinephile_shift"] = -0.3
    if intent["genres"]:
        result["facts"]["asked_genres"] = ", ".join(intent["genres"])
    for name in ("likes", "dislikes", "genres_liked", "genres_avoided"):
        result[name] = list(dict.fromkeys(result[name]))
    for name in ("genres", "avoid_genres", "reference_ids"):
        intent[name] = list(dict.fromkeys(intent[name]))
    return result


def _zscore(values: np.ndarray) -> np.ndarray:
    finite = np.isfinite(values)
    out = np.full(values.shape, -np.inf, dtype=np.float64)
    if finite.sum() == 0:
        return out
    std = values[finite].std() or 1.0
    out[finite] = (values[finite] - values[finite].mean()) / std
    return out


def _candidates(user: dict, intent: dict, k: int):
    films = catalog.load()
    pool = films[films["in_graph"]] if films["in_graph"].sum() >= k else films
    exclude = set(user["liked_ids"]) | set(user["disliked_ids"]) | set(user["seen_ids"]) \
        | set(user["not_interested_ids"]) | set(intent.get("reference_ids", []))
    pool = pool[~pool["movie_id"].isin(exclude)]
    wanted = set(intent.get("genres") or [])
    avoided = (set(intent.get("avoid_genres") or []) | set(user["genres_avoided"])) - wanted
    if avoided:
        pool = pool[~pool["genre_set"].map(lambda g: bool(g & avoided))]
    min_year, max_year = intent.get("min_year"), intent.get("max_year")
    if min_year:
        pool = pool[pool["year"] >= min_year]
    if max_year:
        pool = pool[pool["year"] <= max_year]
    if wanted:
        matching = pool[pool["genre_set"].map(lambda g: wanted <= g)]
        if len(matching) < k:
            matching = pool[pool["genre_set"].map(lambda g: bool(g & wanted))]
        if len(matching) >= min(k, 1):
            pool = matching
    return pool


def rank(user: dict, intent: dict, k: int = 10, cinephile: float | None = None) -> list[dict]:
    intent = intent or {}
    level = user["cinephile_level"] if cinephile is None else cinephile
    level = float(np.clip(level + intent.get("cinephile_shift", 0.0), 0.0, 1.0))
    pool = _candidates(user, intent, k)
    if pool.empty:
        return []
    ids = pool["movie_id"].to_numpy()
    liked = list(user["liked_ids"]) + list(intent.get("reference_ids", []))
    disliked = list(user["disliked_ids"])
    if liked or disliked:
        base = taste.score(taste.user_vector(liked, disliked), ids).astype(np.float64)
    else:
        base = np.log1p(pool["likes"].to_numpy())
    score = _zscore(base)
    if user["genres_liked"]:
        fav = set(user["genres_liked"])
        score = score + GENRE_BONUS * pool["genre_set"].map(lambda g: min(len(g & fav), 2)).to_numpy()
    suggested = set(user.get("suggested_ids", []))
    if suggested:
        score = score - SUGGESTED_PENALTY * np.isin(ids, list(suggested))
    order = np.argsort(-score)[: max(SHORTLIST, k)]
    order = order[np.isfinite(score[order])]
    if len(order) == 0:
        return []
    rarity = _zscore(-np.log1p(pool["likes"].to_numpy()[order]))
    final = score[order] + (2 * level - 1) * GEM_STRENGTH * rarity
    top = order[np.argsort(-final)[:k]]
    top_final = np.sort(final)[::-1][:k]
    reasons = taste.because(liked, ids[top]) if liked else [None] * len(top)
    return [
        {"movie_id": int(ids[i]), "score": round(float(s), 4), "because_movie_id": None if r is None else int(r)}
        for i, s, r in zip(top, top_final, reasons)
    ]


def fallback_recommend(user_id: int, intent: dict | None = None, k: int = 10, cinephile: float | None = None) -> list[dict]:
    conn = db.connect()
    profile.ensure_user(conn, user_id)
    return rank(profile.get_profile(conn, user_id), intent or {}, k, cinephile)


def _is_french(message: str) -> bool:
    words = set(catalog.normalize(message).split())
    return len(words & FRENCH_WORDS) >= 2 or bool(re.search(r"[éèêàçù]", message))


def fallback_reply(message: str, user: dict, recos: list[dict], extracted: dict | None = None) -> str:
    fr = _is_french(message)
    extracted = extracted or {}
    intent = extracted.get("intent", {})
    lines = []
    learned = [catalog.title(m) for m in extracted.get("likes", [])]
    hated = [catalog.title(m) for m in extracted.get("dislikes", [])]
    if learned:
        lines.append(("Noté, tu as aimé " if fr else "Got it, you liked ") + ", ".join(filter(None, learned)) + ".")
    if hated:
        lines.append(("Ok, je retiens que tu n'as pas aimé " if fr else "Ok, I'll remember you didn't like ")
                     + ", ".join(filter(None, hated)) + ".")
    if not recos:
        lines.append("Je n'ai rien trouvé avec ces critères, tu veux élargir un peu ?" if fr
                     else "I couldn't find anything with these filters, want to widen the search?")
        return " ".join(lines)
    genres = ", ".join(GENRE_FR.get(g, g) if fr else g for g in intent.get("genres", []))
    if genres:
        lines.append((f"Côté {genres}, voilà ce que je te propose selon tes goûts :" if fr
                      else f"For some {genres} that fits your taste, here's what I'd pick:"))
    else:
        lines.append("Voilà ce que je te conseille :" if fr else "Here's what I'd recommend:")
    for reco in recos[:3]:
        film = catalog.card(reco["movie_id"])
        if film is None:
            continue
        year = f" ({film['year']})" if film["year"] else ""
        gem = (" (pépite)" if fr else " (hidden gem)") if film["hidden_gem"] else ""
        because = catalog.title(reco.get("because_movie_id"))
        why = (f" parce que tu as aimé {because}" if fr else f" because you liked {because}") if because else ""
        lines.append(f"- **{film['title']}**{year}{gem}{why}")
    if len(user["liked_ids"]) < 5:
        lines.append("Dis-m'en plus : c'est quoi un film que tu as adoré récemment ?" if fr
                     else "Tell me more: what's a film you loved recently?")
    else:
        lines.append("Dis-moi si tu les as déjà vus, j'affinerai la prochaine fois." if fr
                     else "Tell me if you've seen them already, I'll fine-tune next time.")
    return "\n".join(lines)


# ---------------------------------------------------------------- part 3 (src/llm/agent.py, src/reco/engine.py)
# When part 3 is there, it does the understanding, ranking and reply. This file then only
# translates between its format and the database. Without it, the simple version above is used

INTENT_KEYS = {"avoid_genres": "excluded_genres", "min_year": "year_min", "max_year": "year_max"}


def _part3():
    try:
        return importlib.import_module("src.llm.agent"), importlib.import_module("src.reco.engine")
    except ImportError:
        return None


def _use_database(conn=None):
    """Part 3 reads the user's memory (clicks, questionnaire, facts) from this database"""
    from src.reco.memory import SQLiteProvider

    path = conn.execute("PRAGMA database_list").fetchone()[2] if conn is not None else str(db.DB_PATH)
    _part3()[1].set_memory_provider(SQLiteProvider(path))


def warm_up():
    """Load the catalogs and taste vectors once at start-up, so the first chat message is fast"""
    catalog.load()
    taste.is_mock()
    if _part3():
        from src.reco.catalog import get_catalog
        get_catalog()


def to_engine_intent(intent: dict | None) -> dict:
    """Request in this file's format -> part 3 format"""
    return {INTENT_KEYS.get(k, k): v for k, v in (intent or {}).items()}


def from_agent(intent: dict) -> dict:
    """Part 3 extraction -> what the database stores (same keys as fallback_extract)"""
    genres_liked, genres_avoided, facts = [], [], {}
    for fact in intent.get("profile_facts", []):
        if fact["key"] == "genre_like":
            genres_liked.append(fact["value"])
        elif fact["key"] == "genre_dislike":
            genres_avoided.append(fact["value"])
        else:
            facts[fact["key"]] = fact["value"]
    return {
        "intent": {"genres": intent.get("genres", []), "avoid_genres": intent.get("excluded_genres", []),
                   "reference_ids": intent.get("reference_ids", []), "min_year": intent.get("year_min"),
                   "max_year": intent.get("year_max"), "cinephile_shift": intent.get("cinephile_shift", 0.0)},
        "likes": intent.get("liked_ids", []),
        "dislikes": intent.get("disliked_ids", []),
        "seen": intent.get("seen_ids", []),
        "genres_liked": genres_liked,
        "genres_avoided": genres_avoided,
        "facts": facts,
        "agent_intent": intent,
    }


def _agent_history(conn, conversation_id: int, limit: int = 20) -> list[dict]:
    """Chat history in part 3 format: assistant turns carry the request and the films shown,
    so that "another one" or "more recent" refine the previous request"""
    rows = conn.execute(
        "SELECT role, content, extracted_json, recommended_ids FROM messages WHERE conversation_id = ? "
        "ORDER BY id DESC LIMIT ?", (conversation_id, limit)).fetchall()
    out, last_intent = [], None
    for row in reversed(rows):
        if row["role"] == "user":
            extracted = json.loads(row["extracted_json"] or "{}")
            last_intent = extracted.get("agent_intent")
            out.append({"role": "user", "content": row["content"]})
        else:
            recos = json.loads(row["recommended_ids"] or "[]")
            out.append({"role": "assistant", "content": row["content"], "intent": last_intent or {}, "recos": recos})
    return out


def extract(message: str, history=None) -> dict:
    part3 = _part3()
    return from_agent(part3[0].extract(message, history or [])) if part3 else fallback_extract(message, history)


def recommend(user_id: int, intent: dict | None = None, k: int = 10, cinephile: float | None = None) -> list[dict]:
    part3 = _part3()
    if part3 is None:
        return fallback_recommend(user_id, intent, k, cinephile)
    _use_database()
    return part3[1].recommend(user_id, to_engine_intent(intent), k, cinephile)


def reply(message: str, user: dict, recos: list[dict], extracted: dict | None = None) -> str:
    part3 = _part3()
    return part3[0].reply(message, user, recos) if part3 else fallback_reply(message, user, recos, extracted)


def chat_turn(conn, user_id: int, message: str, conversation_id: int | None = None, k: int = 10) -> dict:
    profile.ensure_user(conn, user_id)
    if conversation_id is None:
        conversation_id = profile.start_conversation(conn, user_id)
    part3 = _part3()
    if part3 is None:
        return _fallback_turn(conn, user_id, message, conversation_id, k)

    _use_database(conn)
    turn = part3[0].chat_turn(user_id, message, _agent_history(conn, conversation_id), cinephile=None, k=k)
    extracted = from_agent(turn["intent"])
    recos = turn["recos"]
    profile.add_message(conn, conversation_id, "user", message, extracted=extracted)
    profile.apply_extraction(conn, user_id, extracted, conversation_id)
    for movie_id in extracted["seen"]:
        profile.log_interaction(conn, user_id, movie_id, "seen", "chat", conversation_id=conversation_id, commit=False)
    for reco in recos[:3]:  # the 3 films named in the reply; shown films move back if ignored
        profile.log_interaction(conn, user_id, reco["movie_id"], "suggested", "chat",
                                conversation_id=conversation_id, commit=False)
    conn.commit()
    profile.add_message(conn, conversation_id, "assistant", turn["reply"],
                        recommended_ids=[r["movie_id"] for r in recos])
    return {"conversation_id": conversation_id, "reply": turn["reply"], "recommendations": recos,
            "extracted": extracted}


def _fallback_turn(conn, user_id: int, message: str, conversation_id: int, k: int) -> dict:
    past = profile.history(conn, conversation_id)
    extracted = fallback_extract(message, past)
    profile.add_message(conn, conversation_id, "user", message, extracted=extracted)
    profile.apply_extraction(conn, user_id, extracted, conversation_id)
    recos = fallback_recommend(user_id, extracted.get("intent", {}), k)
    user = profile.get_profile(conn, user_id)
    answer = fallback_reply(message, user, recos, extracted)
    for reco in recos[:3]:
        profile.log_interaction(conn, user_id, reco["movie_id"], "suggested", "chat",
                                conversation_id=conversation_id, commit=False)
    conn.commit()
    profile.add_message(conn, conversation_id, "assistant", answer, recommended_ids=[r["movie_id"] for r in recos])
    return {"conversation_id": conversation_id, "reply": answer, "recommendations": recos, "extracted": extracted}

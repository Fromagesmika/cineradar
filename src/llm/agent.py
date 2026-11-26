"""Cerveau conversationnel (partie 3).

Fonctions du contrat :
    extract(message, history) -> dict      # message -> intention JSON
    reply(message, profile, recos) -> str  # réponse « ami » basée UNIQUEMENT sur les recos fournies

Bonus pour la partie 4 :
    chat_turn(user_id, message, history) -> {"reply", "recos", "intent", "memory_updates"}
    -> un seul appel pour /chat ; `memory_updates` liste ce qu'il faut écrire dans la base
       (interactions + profile_facts) pour que la matrice de mémoire évolue.

Format de `history` : [{"role": "user"|"assistant", "content": str,
                        "intent": dict (optionnel, sur les tours assistant),
                        "recos": [movie_id, ...] (optionnel, sur les tours assistant)}]
Stocker `intent` et `recos` avec chaque message assistant permet le « un autre / plus récent ».
"""
from __future__ import annotations

import json
import random
import re

from src.reco import engine
from src.reco.catalog import CANONICAL_GENRES, get_catalog, normalize_text
from src.reco.memory import InMemoryProvider

from . import client
from .prompts import EXTRACTION_SYSTEM, FRIEND_SYSTEM
from .rules import extract_rules

LIST_FIELDS = ["genres", "excluded_genres", "keywords", "reference_titles", "liked_titles",
               "disliked_titles", "seen_titles", "profile_facts", "directors", "actors", "origins"]
FR_GENRE = {
    "Action": "action", "Adventure": "aventure", "Animation": "animation", "Children": "famille",
    "Comedy": "comédie", "Crime": "polar", "Documentary": "documentaire", "Drama": "drame",
    "Fantasy": "fantastique", "Film-Noir": "film noir", "Horror": "horreur", "IMAX": "IMAX",
    "Musical": "comédie musicale", "Mystery": "mystère", "Romance": "romance", "Sci-Fi": "SF",
    "Thriller": "thriller", "War": "film de guerre", "Western": "western",
}


# =========================================================================== extract


def _last_assistant(history: list[dict] | None) -> dict | None:
    for h in reversed(history or []):
        if h.get("role") == "assistant" and (h.get("intent") or h.get("recos")):
            return h
    return None


def _normalize_intent(d: dict, source: str) -> dict:
    cat = get_catalog()
    out = {
        "intent": d.get("intent") or "recommend",
        "query": d.get("query") or "",
        "year_min": d.get("year_min"),
        "year_max": d.get("year_max"),
        "n": int(d.get("n") or 10),
        "source": source,
    }
    for f in LIST_FIELDS:
        v = d.get(f) or []
        out[f] = v if isinstance(v, list) else [v]
    try:
        out["cinephile_shift"] = float(d.get("cinephile_shift") or 0.0)
    except (TypeError, ValueError):
        out["cinephile_shift"] = 0.0
    # genres -> noms MovieLens
    canon = {normalize_text(g): g for g in CANONICAL_GENRES}
    canon.update({"science fiction": "Sci-Fi", "scifi": "Sci-Fi", "children s": "Children", "family": "Children"})
    for f in ("genres", "excluded_genres"):
        out[f] = list(dict.fromkeys(canon[normalize_text(g)] for g in out[f] if normalize_text(g) in canon))
    # titres -> movie_id
    for kind in ("reference", "liked", "disliked", "seen"):
        ids = [i for i in (d.get(f"{kind}_ids") or []) if isinstance(i, int)]
        for t in out[f"{kind}_titles"]:
            mid = cat.find_title(str(t))
            if mid is not None:
                ids.append(mid)
        out[f"{kind}_ids"] = list(dict.fromkeys(ids))
    # origines : codes connus seulement (« ja », « ko »…)
    from src.reco.origin import ORIGIN_WORDS
    known = set().union(*ORIGIN_WORDS.values())
    out["origins"] = sorted({str(o).lower() for o in out["origins"] if str(o).lower() in known})
    out["directors"] = [str(x) for x in out["directors"] if str(x).strip()]
    out["actors"] = [str(x) for x in out["actors"] if str(x).strip()]
    # faits de profil propres
    facts = []
    for f in out["profile_facts"]:
        if isinstance(f, dict) and f.get("key") and f.get("value") not in (None, ""):
            facts.append({"key": str(f["key"]), "value": str(f["value"])})
    out["profile_facts"] = facts
    return out


def _extract_llm(message: str, history: list[dict] | None) -> dict | None:
    if not client.is_available():
        return None
    msgs = []
    for h in (history or [])[-6:]:
        content = h.get("content", "")
        if h.get("role") == "assistant" and h.get("intent"):
            content += f"\n[intention précédente: {json.dumps(h['intent'], ensure_ascii=False)}]"
        msgs.append({"role": "user" if h.get("role") == "user" else "assistant", "content": content or "…"})
    # l'API Anthropic exige d'alterner et de commencer par user : on aplatit l'historique
    context = "\n".join(f"{m['role']}: {m['content']}" for m in msgs)
    user = (f"Historique récent :\n{context}\n\n" if context else "") + f"Dernier message : {message}\n\nJSON :"
    raw = client.complete(EXTRACTION_SYSTEM, [{"role": "user", "content": user}], max_tokens=600,
                          temperature=0.0, json_mode=True)
    data = client.parse_json(raw)
    return _normalize_intent(data, "llm") if data else None


def _merge_refine(intent: dict, history: list[dict] | None) -> dict:
    """« un autre », « plus récent »… : on part de la demande précédente et on applique les changements."""
    prev_turn = _last_assistant(history)
    if not prev_turn:
        return intent
    # les avis et faits de profil ne sont pas recopiés : ils sont déjà en mémoire
    per_turn = {"liked_ids", "disliked_ids", "seen_ids", "liked_titles", "disliked_titles",
                "seen_titles", "profile_facts"}
    prev = {k: v for k, v in (prev_turn.get("intent") or {}).items() if k not in per_turn}
    merged = dict(prev)
    for k, v in intent.items():
        if k in ("intent", "source"):
            continue
        if isinstance(v, list):
            merged[k] = list(dict.fromkeys((prev.get(k) or []) + v))
        elif v not in (None, "", 0, 0.0):
            merged[k] = v
    merged["excluded_genres"] = merged.get("excluded_genres") or []
    merged["genres"] = [g for g in merged.get("genres", []) if g not in merged["excluded_genres"]]
    merged["intent"] = "refine"
    merged["source"] = intent.get("source")
    merged["exclude_ids"] = list(dict.fromkeys((prev.get("exclude_ids") or []) + list(prev_turn.get("recos") or [])))
    return merged


def extract(message: str, history: list[dict] | None = None) -> dict:
    """Message -> intention structurée. LLM si une clé est configurée, sinon règles."""
    rules = extract_rules(message, history)
    intent = _extract_llm(message, history)
    if intent is None:
        intent = _normalize_intent(rules, "rules")
    else:
        # filet de sécurité : personnes et origines trouvées par les règles, titres ratés par le LLM
        for f in ("directors", "actors", "origins"):
            intent[f] = list(dict.fromkeys(intent.get(f, []) + rules.get(f, [])))
        for kind in ("reference", "liked", "disliked", "seen"):
            intent[f"{kind}_ids"] = list(dict.fromkeys(intent[f"{kind}_ids"] + [
                m for m in rules[f"{kind}_ids"] if not any(m in intent[f"{o}_ids"] for o in ("reference", "liked", "disliked", "seen"))
            ]))
    if intent["intent"] == "refine" or (rules["intent"] == "refine" and intent["source"] == "rules"):
        intent = _merge_refine(intent, history)
    elif intent["intent"] == "feedback" and _last_assistant(history):
        # simple avis : on garde le contexte de la demande en cours (genres, ambiance…)
        merged = _merge_refine(intent, history)
        merged["intent"] = "feedback"
        merged["exclude_ids"] = []
        intent = merged
    return intent


# =========================================================================== reply


def _cards(recos: list[dict], profile: dict | None = None, intent: dict | None = None) -> list[dict]:
    cat = get_catalog()
    liked = {cat.find_title(t) for t in (profile or {}).get("liked_titles", [])}
    asked = [(n, "director") for n in (intent or {}).get("directors", [])] + \
            [(n, "actor") for n in (intent or {}).get("actors", [])]
    cards = []
    for r in recos:
        m = cat.get(r["movie_id"])
        if not m:
            continue
        b = cat.get(r["because_movie_id"]) if r.get("because_movie_id") is not None else None
        cards.append({
            "title": m["title"], "year": m["year"], "genres": m["genres"],
            "overview": m["overview"][:300], "pepite": m["popularity"] <= 0.25,
            "because": b["title"] if b else None,
            # vrai « parce que tu as aimé » seulement si le film est dans les goûts connus
            "because_liked": bool(b and b["movie_id"] in liked),
            "directors": m.get("directors", [])[:2],
            # la personne demandée (« un film de Nolan ») si elle est dans ce film
            "person": next(((n, role) for n, role in asked
                            if cat.resolve_person(n) in {cat.resolve_person(x) for x in
                                                         (m.get("directors", []) if role == "director" else m.get("cast", []))}),
                           None),
        })
    return cards


def _profile_summary(profile: dict | None) -> dict:
    p = profile or {}
    return {k: p[k] for k in ("name", "liked_titles", "favorite_genres", "avoided_genres", "facts") if p.get(k)}


FR_GENRE_PHRASE = {
    "Action": "un film d'action", "Adventure": "un film d'aventure", "Animation": "un film d'animation",
    "Children": "un film à voir en famille", "Comedy": "une comédie", "Crime": "un polar",
    "Documentary": "un documentaire", "Drama": "un drame", "Fantasy": "un film fantastique",
    "Film-Noir": "un film noir", "Horror": "un film d'horreur", "Musical": "une comédie musicale",
    "Mystery": "un film à mystère", "Romance": "une romance", "Sci-Fi": "un film de SF",
    "Thriller": "un thriller", "War": "un film de guerre", "Western": "un western",
}

# ---- langue de la réponse : celle du message (français par défaut)
FR_HINTS = set("je j tu un une le la les des du pour ce soir moi et avec pas veux voudrais envie aime adore quel "
               "quelque chose mais bonjour salut merci plutot truc film films comme genre donne propose conseille "
               "mon ma mes qui que est sur dans sans peu moins plus tres".split())
EN_HINTS = set("i i'm im me my you a an the for tonight something some with without not no but like want would love "
               "loved hate please movie movies film films show give recommend any anything good more less less "
               "recent old funny scary hi hello thanks thank what can".split())


def detect_language(message: str) -> str:
    """'en' si le message est en anglais, sinon 'fr' (la langue par défaut de l'appli)."""
    raw = message or ""
    if re.search(r"[éèêàçùâîôœ]", raw):
        return "fr"
    words = re.findall(r"[a-z']+", raw.lower())
    fr = sum(w in FR_HINTS for w in words)
    en = sum(w in EN_HINTS for w in words)
    return "en" if en > fr else "fr"


EN_GENRE_PHRASE = {
    "Action": "an action film", "Adventure": "an adventure film", "Animation": "an animated film",
    "Children": "a family film", "Comedy": "a comedy", "Crime": "a crime film", "Documentary": "a documentary",
    "Drama": "a drama", "Fantasy": "a fantasy film", "Film-Noir": "a film noir", "Horror": "a horror film",
    "Musical": "a musical", "Mystery": "a mystery", "Romance": "a romance", "Sci-Fi": "a sci-fi film",
    "Thriller": "a thriller", "War": "a war film", "Western": "a western",
}
EN_OPENERS = ["Ok, I've got just the thing!", "Oh, I have a few ideas.", "I know exactly what you mean.",
              "Good call, check these out:", "Here's what I'd pick:"]
EN_CLOSERS = [
    "Something recent or a classic next time?",
    "Does that sound good, or should I try another mood?",
    "Watching alone or with someone?",
    "Lighter or darker next time?",
    "Any of these tempting? Tell me if you've already seen them.",
]
TEXTS = {
    "fr": {"none": ("Je n'ai rien trouvé qui colle parfaitement… Tu peux me donner un film que tu as aimé, "
                    "ou une ambiance (drôle, sombre, action, SF…) ?"),
           "more": "Et {n} autre{s} idée{s} dans les fiches juste en dessous.",
           "thanks": "Avec plaisir ! Dis-moi ce que tu en as pensé, ça m'aide à mieux viser la prochaine fois.",
           "hello": "Salut ! Tu as envie de quoi ce soir ? Un genre, une ambiance, ou un film que tu as adoré ?",
           "same": "Si ça te plaît, dans le même esprit :",
           "noted": "Noté, je m'en souviendrai ! "},
    "en": {"none": ("I couldn't find anything that really fits… Can you give me a film you loved, "
                    "or a mood (funny, dark, action, sci-fi…)?"),
           "more": "And {n} more idea{s} in the cards just below.",
           "thanks": "My pleasure! Tell me what you thought of it, it helps me aim better next time.",
           "hello": "Hi! What do you feel like watching tonight? A genre, a mood, or a film you loved?",
           "same": "If you like it, in the same spirit:",
           "noted": "Got it, I'll remember that! "},
}

OPENERS = [
    "Ok, j'ai ce qu'il te faut !", "Alors là, j'ai quelques idées.", "Je vois exactement le genre.",
    "Bonne pioche, regarde ça :", "Tiens, je te propose ça :",
]
CLOSERS = [
    "Tu veux plutôt quelque chose de récent ou un classique ?",
    "Ça te parle, ou je pars sur une autre ambiance ?",
    "Tu regardes ça seul ou accompagné ?",
    "Plus léger ou plus sombre, la prochaine fois ?",
    "Un de ceux-là te tente ? Dis-moi si tu les as déjà vus.",
]


def _line(c: dict, i: int = 0, lang: str = "fr") -> str:
    year = f" ({c['year']})" if c["year"] else ""
    phrases = EN_GENRE_PHRASE if lang == "en" else FR_GENRE_PHRASE
    genre = phrases.get(c["genres"][0], "a film" if lang == "en" else "un film") if c["genres"] else \
        ("a film" if lang == "en" else "un film")
    b = c["because"]
    person = c.get("person")
    if lang == "en":
        if person:
            name, role = person
            why = [f"directed by {name}", f"by {name}", f"another {name}"][i % 3] if role == "director" else \
                  [f"with {name}", f"starring {name}", f"another one with {name}"][i % 3]
        elif b and c.get("because_liked"):
            why = [f"because you liked {b}", f"since you loved {b}", f"if you enjoyed {b}, you'll like this one"][i % 3]
        elif b:
            why = [f"in the same vein as {b}", f"close to {b}"][i % 2]
        else:
            why = [f"{genre} that fits what you asked", f"{genre} I'd recommend without hesitation"][i % 2]
        gem = ", a real hidden gem" if c["pepite"] else ""
        return f"• **{c['title']}**{year}: {why}{gem}."
    if person:
        name, role = person
        why = [f"signé {name}", f"de {name}", f"encore un {name}"][i % 3] if role == "director" else \
              [f"avec {name}", f"{name} au casting", f"encore un film avec {name}"][i % 3]
    elif b and c.get("because_liked"):
        why = [f"parce que tu as aimé {b}", f"vu que tu as adoré {b}", f"si {b} t'a plu, celui-là aussi"][i % 3]
    elif b:
        why = [f"dans la même veine que {b}", f"proche de {b}"][i % 2]
    else:
        why = [f"{genre} qui colle à ta demande", f"{genre} que je te conseille les yeux fermés"][i % 2]
    gem = ", une vraie pépite peu connue" if c["pepite"] else ""
    return f"• **{c['title']}**{year} : {why}{gem}."


def _template_reply(message: str, profile: dict | None, recos: list[dict], n_highlight: int = 3,
                    intent: dict | None = None) -> str:
    lang = detect_language(message)
    cards = _cards(recos, profile, intent)
    if not cards:
        return TEXTS[lang]["none"]
    name = (profile or {}).get("name")
    opener = random.choice(EN_OPENERS if lang == "en" else OPENERS)
    if name:
        opener = f"{name}, " + opener[0].lower() + opener[1:]
    lines = [opener, *(_line(c, i, lang) for i, c in enumerate(cards[:n_highlight]))]
    if len(cards) > n_highlight:
        rest = len(cards) - n_highlight
        lines.append(TEXTS[lang]["more"].format(n=rest, s="s" if rest > 1 else ""))
    lines.append(random.choice(EN_CLOSERS if lang == "en" else CLOSERS))
    return "\n".join(lines)


def _reply_is_valid(text: str, recos: list[dict], profile: dict | None) -> bool:
    """Vérifie que le LLM ne cite aucun film hors de la liste (ni n'invente)."""
    cat = get_catalog()
    allowed = {r["movie_id"] for r in recos} | {r.get("because_movie_id") for r in recos}
    for t in (profile or {}).get("liked_titles", []):
        mid = cat.find_title(t)
        if mid:
            allowed.add(mid)
    mentioned = set(cat.titles_in_text(text))
    reco_ids = {r["movie_id"] for r in recos}
    return bool(mentioned & reco_ids) and mentioned <= allowed


def reply(message: str, profile: dict | None, recos: list[dict], n_highlight: int = 3,
          intent: dict | None = None) -> str:
    """Réponse d'ami, uniquement à partir des recos fournies (+ explication « parce que tu as aimé… »)."""
    if not recos:
        return _template_reply(message, profile, recos)
    if client.is_available():
        payload = {
            "langue_de_la_reponse": "anglais" if detect_language(message) == "en" else "français",
            "message_utilisateur": message,
            "profil": _profile_summary(profile),
            "films_a_proposer_dans_cet_ordre": [
                {k: v for k, v in c.items() if v not in (None, False, [])} for c in _cards(recos, profile, intent)
            ],
        }
        text = client.complete(
            FRIEND_SYSTEM.format(n_highlight=n_highlight),
            [{"role": "user", "content": json.dumps(payload, ensure_ascii=False)}],
            max_tokens=400, temperature=0.7,
        )
        if text and _reply_is_valid(text, recos, profile):
            return text.strip()
    return _template_reply(message, profile, recos, n_highlight, intent)


# =========================================================================== orchestration


def memory_updates_from(intent: dict, recos: list[dict]) -> dict:
    """Ce que la partie 4 doit écrire dans la base après ce tour (matrice de mémoire)."""
    interactions = []
    for mid in intent.get("liked_ids", []):
        interactions.append({"movie_id": mid, "type": "chat_like", "source": "chat"})
    for mid in intent.get("disliked_ids", []):
        interactions.append({"movie_id": mid, "type": "chat_dislike", "source": "chat"})
    for mid in intent.get("seen_ids", []):
        interactions.append({"movie_id": mid, "type": "seen", "source": "chat"})
    for r in recos:
        interactions.append({"movie_id": r["movie_id"], "type": "shown", "source": "chat"})
    return {"interactions": interactions, "profile_facts": intent.get("profile_facts", [])}


def apply_updates(provider: InMemoryProvider, user_id: int, updates: dict) -> None:
    for it in updates["interactions"]:
        provider.log(user_id, it["movie_id"], it["type"])
    for f in updates["profile_facts"]:
        provider.add_fact(user_id, f["key"], f["value"])


def _info_reply(intent: dict, lang: str = "fr") -> str | None:
    cat = get_catalog()
    ids = intent.get("reference_ids") or intent.get("liked_ids") or intent.get("seen_ids")
    if not ids:
        return None
    m = cat.get(ids[0])
    genres = ", ".join(m["genres"]) if lang == "en" else ", ".join(FR_GENRE.get(g, g) for g in m["genres"])
    return f"**{m['title']}** ({m['year']}){'' if lang == 'en' else ' '}: {genres}. {m['overview']}"


def chat_turn(user_id: int, message: str, history: list[dict] | None = None,
              cinephile: float | None = None, k: int = 10) -> dict:
    """Un tour de conversation complet : extraction -> recommandations -> réponse -> mises à jour mémoire."""
    intent = extract(message, history)
    lang = detect_language(message)
    provider = engine.get_memory_provider()

    if intent["intent"] == "chitchat":
        if re.search(r"\b(merci|thanks|thank you)\b", normalize_text(message)):
            text = TEXTS[lang]["thanks"]
        else:
            text = TEXTS[lang]["hello"]
        return {"reply": text, "recos": [], "intent": intent, "memory_updates": memory_updates_from(intent, [])}

    # on applique d'abord ce que le message nous apprend (films aimés, faits), pour que CE tour en profite
    pre = memory_updates_from(intent, [])
    if isinstance(provider, InMemoryProvider):
        apply_updates(provider, user_id, pre)

    reco_intent = dict(intent)
    if intent["intent"] == "info" and not intent.get("reference_ids"):
        reco_intent["reference_ids"] = intent.get("liked_ids") or intent.get("seen_ids")
    recos = engine.recommend(user_id, reco_intent, k=min(k, intent.get("n") or k), cinephile=cinephile)
    mem = provider.get(user_id)
    profile = mem.to_profile()

    text = reply(message, profile, recos, intent=intent)
    if intent["intent"] == "info":
        info = _info_reply(intent, lang)
        if info:
            text = info + f"\n\n{TEXTS[lang]['same']}\n" + "\n".join(
                _line(c, i, lang) for i, c in enumerate(_cards(recos, profile, intent)[:3]))
    elif intent["intent"] == "feedback" and recos:
        text = TEXTS[lang]["noted"] + text

    updates = memory_updates_from(intent, recos)
    if isinstance(provider, InMemoryProvider):
        apply_updates(provider, user_id, {"interactions": [i for i in updates["interactions"] if i["type"] == "shown"],
                                          "profile_facts": []})
    return {"reply": text, "recos": recos, "intent": intent, "memory_updates": updates}


__all__ = ["extract", "reply", "chat_turn", "memory_updates_from", "apply_updates", "detect_language"]

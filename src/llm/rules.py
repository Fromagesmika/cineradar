"""Extraction sans LLM (mode « sans clé API ») : mots-clés + expressions régulières.

Moins fin qu'un LLM mais suffisant pour la démo et comme repli si l'API tombe.
"""
from __future__ import annotations

import datetime as dt
import re

import numpy as np

from src.reco.catalog import get_catalog
from src.reco.catalog import normalize_text as _nt


def normalize_text(s: str) -> str:  # messages : on garde les articles (« un autre »)
    return _nt(s, strip_article=False)

# expression (texte normalisé, sans accents) -> genre MovieLens. Les plus longues d'abord.
GENRE_LEXICON = {
    "comedie musicale": "Musical", "comedies musicales": "Musical", "musical": "Musical",
    "science fiction": "Sci-Fi", "sci fi": "Sci-Fi", "scifi": "Sci-Fi", "sf": "Sci-Fi", "futuriste": "Sci-Fi",
    "dessin anime": "Animation", "dessins animes": "Animation", "animation": "Animation", "anime": "Animation",
    "film noir": "Film-Noir",
    "film de guerre": "War", "films de guerre": "War", "guerre": "War", "war": "War",
    "action": "Action",
    "aventure": "Adventure", "aventures": "Adventure", "adventure": "Adventure",
    "enfant": "Children", "enfants": "Children", "famille": "Children", "familial": "Children", "gamins": "Children",
    "comedie": "Comedy", "comedies": "Comedy", "drole": "Comedy", "droles": "Comedy", "rire": "Comedy",
    "marrant": "Comedy", "humour": "Comedy", "comedy": "Comedy",
    "policier": "Crime", "polar": "Crime", "polars": "Crime", "gangster": "Crime", "gangsters": "Crime",
    "mafia": "Crime", "crime": "Crime", "braquage": "Crime",
    "documentaire": "Documentary", "docu": "Documentary", "documentaires": "Documentary",
    "drame": "Drama", "drames": "Drama", "emouvant": "Drama", "pleurer": "Drama", "drama": "Drama",
    "fantastique": "Fantasy", "fantasy": "Fantasy", "magie": "Fantasy",
    "horreur": "Horror", "epouvante": "Horror", "flippant": "Horror", "peur": "Horror", "horror": "Horror",
    "mystere": "Mystery", "enquete": "Mystery", "whodunit": "Mystery",
    "romance": "Romance", "romantique": "Romance", "amour": "Romance", "comedie romantique": "Romance",
    "thriller": "Thriller", "suspense": "Thriller", "thrillers": "Thriller",
    "western": "Western", "westerns": "Western",
}

# ambiances / thèmes en français -> tags (anglais, comme MovieLens/TMDB)
KEYWORD_LEXICON = {
    "braquage": "heist", "casse": "heist", "vengeance": "revenge", "venger": "revenge",
    "espace": "space", "spatial": "space", "zombie": "zombies", "zombies": "zombies",
    "voyage dans le temps": "time travel", "boucle temporelle": "time loop",
    "arts martiaux": "martial arts", "kung fu": "martial arts", "baston": "martial arts",
    "tueur a gages": "hitman", "tueur en serie": "serial killer", "serial killer": "serial killer",
    "coreen": "korean", "coreens": "korean", "coreenne": "korean", "coreennes": "korean",
    "francais": "french", "francaise": "french", "francaises": "french",
    "japonais": "japanese", "japonaise": "japanese", "hongkongais": "hong kong",
    "course poursuite": "car chase", "courses poursuites": "car chase", "voitures": "car chase",
    "super heros": "superhero", "superheros": "superhero", "robot": "robot", "robots": "robot",
    "intelligence artificielle": "artificial intelligence", "ia": "artificial intelligence",
    "feel good": "feel-good", "bonne humeur": "feel-good", "leger": "feel-good", "detente": "feel-good",
    "sombre": "dark", "glauque": "disturbing", "derangeant": "disturbing",
    "retournement": "twist", "twist": "twist", "psychologique": "psychological", "prise de tete": "cerebral",
    "intelligent": "cerebral", "prison": "prison", "mafia": "mafia", "yakuza": "yakuza",
    "post apo": "post-apocalyptic", "post apocalyptique": "post-apocalyptic", "dystopie": "dystopia",
    "extraterrestre": "aliens", "extraterrestres": "aliens", "aliens": "aliens",
    "espion": "spy", "espionnage": "spy", "cyberpunk": "cyberpunk", "nostalgique": "melancholic",
    "melancolique": "melancholic", "amitie": "friendship", "culte": "cult",
}

GEM_WORDS = ["pepite", "pepites", "meconnu", "meconnus", "peu connu", "pointu", "pointus", "inde",
             "independant", "film d auteur", "underground", "rare", "original", "sortir des sentiers"]
MAINSTREAM_WORDS = ["connu", "connus", "grand public", "blockbuster", "blockbusters", "populaire",
                    "populaires", "incontournable", "incontournables", "succes", "valeur sure"]
NEG_WORDS = {"pas", "sans", "evite", "eviter", "evites", "deteste", "jamais", "rien", "marre", "supporte", "aucun"}

TRIG_DISLIKE = r"(pas aime|pas kiffe|pas adore|deteste|detestee|nul|horrible|ennuye|ennuyeux|deçu|decu|pas fan)"
TRIG_REFERENCE = r"(comme|genre|style|similaire|ressemble|dans la veine|du meme genre|a la|facon|proche de)"
TRIG_LIKE = r"(adore|aime|kiffe|love|j ai bien aime|fan de|prefere|coup de coeur|genial|top)"
TRIG_SEEN = r"(deja vu|vu|regarde)"

STOPWORDS = set("""a ai as au aux avec ce ces c ca cet cette d dans de des du elle en et est etait il ils
j je l la le les leur lui m ma me mes moi mon n ne ni nous on ou par pas plus pour qu que qui s sa se ses si son
sur t ta te tes toi ton tu un une vos votre vous y film films truc trucs chose quelque veux voudrais aimerais
envie cherche propose proposes recommande conseille conseilles donne donnes moi peux regarder voir mater soir
ce soir bien tres trop genre style comme autre autres un peu plutot qqbrk""".split())

REFINE_WORDS = ["un autre", "d autres", "autre chose", "encore", "plus recent", "moins", "plutot", "et en",
                "pareil mais", "la meme chose"]
CHITCHAT = ["salut", "bonjour", "coucou", "hello", "merci", "cool", "super", "top merci", "ok", "bonsoir"]
REQUEST_WORDS = ["film", "films", "conseil", "conseille", "propose", "recommande", "envie", "cherche",
                 "regarder", "mater", "voir", "idee", "suggestion"]


def _blank(text: str, phrase: str) -> str:
    return re.sub(rf"\b{re.escape(phrase)}\b", " " * len(phrase), text)


def _find_titles(raw: str, norm: str) -> list[tuple[int, int]]:
    """Renvoie [(movie_id, position)] des titres trouvés dans le message."""
    cat = get_catalog()
    tokens = norm.split()
    found: list[tuple[int, int]] = []
    used = [False] * len(tokens)
    # 1) titres entre guillemets
    for q in re.findall(r"[\"«“']([^\"»”']{2,60})[\"»”']", raw):
        mid = cat.find_title(q)
        if mid:
            k = normalize_text(q)
            pos = norm.find(k)
            found.append((mid, max(pos, 0)))
    # 2) n-grammes du plus long au plus court
    caps = {normalize_text(w) for w in re.findall(r"\b[A-Z][\w'’-]+", raw)}
    char_pos = []
    p = 0
    for t in tokens:
        p = norm.find(t, p)
        char_pos.append(p)
        p += len(t)
    for n in range(min(8, len(tokens)), 0, -1):
        for i in range(len(tokens) - n + 1):
            if any(used[i : i + n]):
                continue
            key = " ".join(tokens[i : i + n])
            mid = cat.title_keys.get(key)
            if mid is None:
                continue
            if n == 1:
                # un seul mot : il faut une majuscule ou un déclencheur juste avant (évite « Heat », « Up »…)
                before = " ".join(tokens[max(0, i - 3) : i])
                if key not in caps and not re.search(rf"{TRIG_REFERENCE}|{TRIG_LIKE}|{TRIG_DISLIKE}|{TRIG_SEEN}", before):
                    continue
                if len(key) < 3:
                    continue
            found.append((mid, char_pos[i]))
            for j in range(i, i + n):
                used[j] = True
    # dédoublonnage
    seen, out = set(), []
    for mid, pos in sorted(found, key=lambda x: x[1]):
        if mid not in seen:
            seen.add(mid)
            out.append((mid, pos))
    return out


def _classify_title(norm: str, pos: int) -> str:
    """Regarde ce qui précède le titre dans la même proposition."""
    before = norm[:pos]
    clause = re.split(r"\b(mais|et puis|par contre|alors que|qqbrk)\b", before)[-1][-60:]
    if re.search(TRIG_DISLIKE, clause):
        return "disliked"
    if re.search(TRIG_REFERENCE, clause):
        return "reference"
    if re.search(r"deja vu|j ai vu|vu ", clause) and not re.search(TRIG_LIKE, clause):
        for after in norm[pos:].split("qqbrk")[:2]:  # la proposition qui suit le titre
            if re.search(TRIG_DISLIKE, after):
                return "disliked"
            if re.search(r"\b(adore|aime|kiffe|genial|trop bien|excellent)\b", after):
                return "liked"
        return "seen"
    if re.search(TRIG_LIKE, clause):
        return "liked"
    after = norm[pos : pos + 80]
    if re.search(r"\b(j ai pas aime|nul|horrible|ennuyeux|bof)\b", after):
        return "disliked"
    if re.search(r"\b(j ai adore|genial|trop bien|incroyable|chef d oeuvre)\b", after):
        return "liked"
    return "reference"


def extract_rules(message: str, history: list[dict] | None = None) -> dict:
    raw = message or ""
    # On découpe par phrases/virgules pour garder les ponctuations utiles aux déclencheurs
    norm = normalize_text(raw.replace(",", " , ").replace(".", " . "))
    norm_clauses = normalize_text(re.sub(r"[.,;!?]", " qqbrk ", raw))
    out = {
        "intent": "recommend", "genres": [], "excluded_genres": [], "keywords": [], "query": raw.strip(),
        "reference_titles": [], "liked_titles": [], "disliked_titles": [], "seen_titles": [],
        "reference_ids": [], "liked_ids": [], "disliked_ids": [], "seen_ids": [],
        "year_min": None, "year_max": None, "cinephile_shift": 0.0, "profile_facts": [], "n": 10,
        "source": "rules",
    }
    cat = get_catalog()

    # ---- titres (on les retire ensuite du texte pour ne pas confondre « Drive » et un genre)
    work = norm_clauses
    for mid, pos in _find_titles(raw, norm_clauses):
        kind = _classify_title(norm_clauses, pos)
        title = cat.get(mid)["title"]
        out[f"{kind}_ids"].append(mid)
        out[f"{kind}_titles"].append(title)
        for k, v in cat.title_keys.items():
            if v == mid:
                work = _blank(work, k)

    # ---- genres (avec négation et préférences durables)
    tokens_text = work
    for phrase in sorted(GENRE_LEXICON, key=len, reverse=True):
        for m in re.finditer(rf"\b{re.escape(phrase)}\b", tokens_text):
            g = GENRE_LEXICON[phrase]
            before_words = tokens_text[: m.start()].split("qqbrk")[-1].split()[-4:]
            negated = any(w in NEG_WORDS for w in before_words)
            prefix = tokens_text[: m.start()].split("qqbrk")[-1]
            det = r"\s+(les?|la|l|des)?\s*(films?|trucs?)?\s*(d|de|du)?\s*$"
            persistent_dislike = re.search(r"(deteste|supporte pas|aime pas|horreur des?|j aime pas)" + det, prefix)
            persistent_like = re.search(r"(j adore|fan de|je kiffe|passionne de|j aime)" + det, prefix)
            if negated or persistent_dislike:
                out["excluded_genres"].append(g)
                if persistent_dislike:
                    out["profile_facts"].append({"key": "genre_dislike", "value": g})
            else:
                out["genres"].append(g)
                if persistent_like and "envie" not in tokens_text:
                    out["profile_facts"].append({"key": "genre_like", "value": g})
        tokens_text = _blank(tokens_text, phrase) if phrase not in KEYWORD_LEXICON else tokens_text
    out["genres"] = [g for g in dict.fromkeys(out["genres"]) if g not in out["excluded_genres"]]
    out["excluded_genres"] = list(dict.fromkeys(out["excluded_genres"]))

    # ---- mots-clés / ambiances
    for phrase in sorted(KEYWORD_LEXICON, key=len, reverse=True):
        if re.search(rf"\b{re.escape(phrase)}\b", work):
            out["keywords"].append(KEYWORD_LEXICON[phrase])
    out["keywords"] = list(dict.fromkeys(out["keywords"]))

    # ---- époque
    year_now = dt.date.today().year
    m = re.search(r"annees? (\d{2,4})", norm)
    if m:
        d = int(m.group(1))
        d = d + 1900 if d < 100 and d >= 30 else (d + 2000 if d < 100 else d)
        out["year_min"], out["year_max"] = d, d + 9
    if m2 := re.search(r"(avant|pre) (\d{4})", norm):
        out["year_max"] = int(m2.group(2))
    if m3 := re.search(r"(apres|depuis|post) (\d{4})", norm):
        out["year_min"] = int(m3.group(2))
    if re.search(r"\b(recent|recents|recente|recentes|nouveau|nouveaute|nouveautes|recemment|cette annee|newer|new)\b", norm):
        # « récent » = les 6 dernières années du catalogue (MovieLens 25M s'arrête en 2019)
        last_year = int(min(year_now, np.nanmax(cat.df["year"].to_numpy(dtype=float))))
        out["year_min"] = last_year - 6
    if re.search(r"\b(vieux|ancien|anciens|classique noir et blanc|old school)\b", norm):
        out["year_max"] = out["year_max"] or 1985

    # ---- curseur
    if any(re.search(rf"\b{w}\b", norm) for w in GEM_WORDS):
        out["cinephile_shift"] = 0.35
    elif any(re.search(rf"\b{w}\b", norm) for w in MAINSTREAM_WORDS) and "peu connu" not in norm:
        out["cinephile_shift"] = -0.35

    # ---- infos de profil
    if re.search(r"\b(avec (mes|les|ma|mon) (enfants|gosses|petits|fils|fille|neveux)|en famille)\b", norm):
        out["profile_facts"].append({"key": "watch_with", "value": "famille"})
        out["excluded_genres"] = list(dict.fromkeys(out["excluded_genres"] + ["Horror"]))
    if re.search(r"\b(avec (ma|mon) (copine|copain|femme|mari|meuf|mec|cherie|cheri)|en couple|en amoureux)\b", norm):
        out["profile_facts"].append({"key": "watch_with", "value": "couple"})
    if re.search(r"\b(entre potes|avec (mes|des) (potes|amis))\b", norm):
        out["profile_facts"].append({"key": "watch_with", "value": "amis"})
    if m := re.search(r"\bj ai (\d{1,2}) ans\b", norm):
        out["profile_facts"].append({"key": "age", "value": m.group(1)})
    if re.search(r"\b(en vo|version originale|sous titre)", norm):
        out["profile_facts"].append({"key": "language_pref", "value": "VO"})

    # ---- intention
    has_request = any(re.search(rf"\b{w}\b", norm) for w in REQUEST_WORDS) or out["genres"] or out["keywords"]
    if re.search(r"\b(c est quoi|parle moi de|de quoi parle|ca raconte quoi|c est bien)\b", norm) and (
        out["liked_ids"] or out["reference_ids"] or out["seen_ids"]
    ):
        out["intent"] = "info"
    elif out["reference_ids"]:
        out["intent"] = "similar"
        # « comme X mais plus récent » -> après l'année de X
        if re.search(r"\b(plus recent|more recent|newer)", norm):
            y = cat.get(out["reference_ids"][0])["year"]
            if y:
                out["year_min"] = y + 1
    elif any(re.search(rf"\b{w}\b", norm) for w in REFINE_WORDS) and history:
        out["intent"] = "refine"
    elif (out["liked_ids"] or out["disliked_ids"] or out["seen_ids"]) and not has_request:
        out["intent"] = "feedback"
    elif not has_request and (len(norm.split()) <= 4 and any(norm.startswith(w) for w in CHITCHAT)):
        out["intent"] = "chitchat"

    # requête sémantique : le message sans les titres ni les mots vides
    words = [w for w in work.split() if w not in STOPWORDS and len(w) > 2]
    out["query"] = " ".join(words) if len(words) >= 1 else ""
    return out

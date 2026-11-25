"""Origine des films (pays / langue d'origine), pour les demandes du type « un film coréen ».

Le catalogue exporté ne contient pas la langue d'origine. Si une colonne `original_language` (code ISO
de TMDB, ex. "ja", "ko", "fr") est présente, elle fait foi. Sinon l'origine est déduite de trois indices
déjà dans le catalogue :
  1. le titre original que MovieLens ajoute entre parenthèses aux films étrangers
     (« Seven Samurai (Shichinin no samurai) », « Amelie (Fabuleux destin d'Amélie Poulain, Le) »),
     et l'écriture du titre (hangul, kana, cyrillique…) ;
  2. les tags (« japan », « bollywood », « korean »…) ;
  3. les noms du réalisateur et des acteurs (« Song Kang-ho », « Toshirō Mifune »…), puis les autres
     films du même réalisateur.
C'est une heuristique : elle rate des films et en classe parfois mal, mais elle suffit pour filtrer.

Codes utilisés : en, fr, it, es, de, pt, ru, sv, da, nl, ja, ko, zh, hi, th, other.
"""
from __future__ import annotations

import re
import unicodedata
from collections import Counter, defaultdict

import pandas as pd

ASIA = {"ja", "ko", "zh", "hi", "th"}
EUROPE = {"fr", "it", "es", "de", "pt", "ru", "sv", "da", "nl"}

# --------------------------------------------------------------------------- tags
TAG_ORIGIN = {
    "japan": "ja", "japanese": "ja", "anime": "ja", "studio ghibli": "ja", "ghibli": "ja", "samurai": "ja",
    "yakuza": "ja", "kurosawa": "ja", "miyazaki": "ja", "j-horror": "ja", "japanese horror": "ja",
    "korea": "ko", "korean": "ko", "south korea": "ko", "k-drama": "ko",
    "china": "zh", "chinese": "zh", "hong kong": "zh", "wuxia": "zh", "taiwan": "zh",
    "hong kong action": "zh", "shaw brothers": "zh",
    "india": "hi", "bollywood": "hi", "tollywood": "hi",
    "thailand": "th", "thai": "th", "muay thai": "th",
    "france": "fr", "french": "fr", "nouvelle vague": "fr", "french new wave": "fr",
    "italy": "it", "italian": "it", "spaghetti western": "it", "giallo": "it", "fellini": "it",
    "spain": "es", "spanish": "es", "almodovar": "es", "mexico": "es", "mexican": "es", "argentina": "es",
    "germany": "de", "german": "de", "russia": "ru", "russian": "ru", "soviet": "ru", "ussr": "ru",
    "sweden": "sv", "swedish": "sv", "bergman": "sv", "denmark": "da", "danish": "da", "dogme 95": "da",
    "brazil": "pt", "brazilian": "pt", "portugal": "pt",
}
# un tag de lieu (« japan ») peut décrire un film américain tourné là-bas : il compte moins
WEAK_TAGS = {"japan", "china", "hong kong", "india", "thailand", "france", "italy", "spain", "germany",
             "russia", "mexico", "korea", "south korea", "samurai", "yakuza", "anime", "paris", "wuxia"}

# --------------------------------------------------------------------------- titre original
STOPWORDS = {
    "fr": "le la les des du un une et est pour sur dans avec pas qui que mon ma mes ce cette au aux l d qu nous vous ils elle".split(),
    "it": "il lo gli della dello delle degli di del un una per con non che nel nella sono come vita bella nuovo "
          "amore uomo notte tutti cosa questo ogni anni giorni storia".split(),
    "es": "el los las la del un una y por para con no que en mi su como mas comunidad antena hijo hija mujer "
          "amor vida muerte noche".split(),
    "de": "der die das den dem des und ein eine einer mit von zu nicht ist im auf fur rennt leben andere".split(),
    "pt": "o os as do dos das um uma para com nao em meu cidade deus vida central".split(),
    "sv": "och en ett det den att av med som pa om jag".split(),
    "da": "og en et det den at af med som pa om jeg ikke".split(),
    "nl": "de het een van en met niet op voor".split(),
}
# mots qui n'existent qu'en anglais : un titre entre parenthèses en anglais est un « a.k.a. »
ENGLISH = set("the of and a an to in for with my is are you your his her love man woman story".split())

KANA = re.compile(r"[぀-ヿ]")
HANGUL = re.compile(r"[가-힯ᄀ-ᇿ]")
HAN = re.compile(r"[一-鿿]")
CYRILLIC = re.compile(r"[Ѐ-ӿ]")
THAI = re.compile(r"[฀-๿]")
DEVANAGARI = re.compile(r"[ऀ-ॿ]")

# romanisations : japonais (Hepburn), pinyin, coréen
JA_SYLLABLES = re.compile(r"^(?:(?:ky|gy|sh|ch|ny|hy|my|ry|by|py|ts|[kgsztdnhbpmrwyjf])?[aeiou]+|n)+$")
JA_WORDS = {"no", "wa", "monogatari", "san", "kun", "sama", "ga", "wo", "ni", "to", "yo", "kara", "made"}
PINYIN = re.compile(
    r"^(?:zh|ch|sh|[bpmfdtnlgkhjqxrzcsyw])?(?:iang|iong|uang|ang|eng|ing|ong|ian|iao|uai|uan|ai|ei|ao|ou|an|en|"
    r"in|un|ia|ie|iu|ua|uo|ue|ui|er|a|e|i|o|u|v)(?:n|ng)?$")
ACCENTS = {"fr": "éêâîœ", "it": "òù", "es": "ñáíóú¿¡", "de": "äöüß", "pt": "ãõç", "sv": "å"}
KO_SYLLABLES = re.compile(
    r"^(?:(?:kk|tt|pp|ss|jj|ch|[gkndrmbsjhtpl])?(?:yeo|yae|wae|wa|wo|we|wi|ui|eo|eu|ae|oe|ya|yo|yu|ye|a|e|i|o|u)"
    r"(?:ng|n|m|l|k|t|p|g|b|d)?)+$")
KO_MARK = re.compile(r"(eo|ae|oe|ui)")
YUE = re.compile(r"^(?:gw|kw|ng|ch|sh|ts|[bpmfdtnlgkhwzcsjy])?(?:aa|oe|eo|eu|[aeiou]{1,2})(?:ng|n|m|p|t|k|i|u)?$")
RU_NAME = re.compile(r"(ov|ova|ev|eva|sky|skiy|skaya|skii|enko|yev|yeva|in|ina|ich)$")
RU_FIRST = set("aleksandr aleksei alexei andrei andrey anatoliy boris dmitri dmitriy fyodor igor ivan konstantin "
               "mikhail nikita nikolai nikolay oleg pavel sergei sergey vladimir vasili viktor yuri yevgeni "
               "anna olga natalya tatyana yelena irina svetlana lyudmila".split())


def _ascii(s: str) -> str:
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()


def _alt_titles(title: str) -> list[str]:
    """Les parenthèses d'un titre MovieLens, sans l'année ni les « a.k.a. »."""
    out = []
    for inner in re.findall(r"\(([^()]+)\)", str(title)):
        inner = inner.strip()
        if re.fullmatch(r"\d{4}", inner) or inner.lower().startswith(("a.k.a", "aka ")):
            continue
        out.append(inner)
    return out


def _lang_of_text(text: str) -> str | None:
    if HANGUL.search(text):
        return "ko"
    if KANA.search(text):
        return "ja"
    if HAN.search(text):
        return "zh"
    if CYRILLIC.search(text):
        return "ru"
    if THAI.search(text):
        return "th"
    if DEVANAGARI.search(text):
        return "hi"
    words = re.findall(r"[a-z]+", _ascii(text))
    if not words:
        return None
    if sum(w in ENGLISH for w in words) >= max(1, len(words) // 3):
        return None  # titre anglais alternatif (« a.k.a. »)
    lower = text.lower()
    if any(ch in lower for ch in "ōūô") and all(JA_SYLLABLES.match(w) for w in words):
        return "ja"
    if len(words) >= 2 and any(w in JA_WORDS for w in words) and all(JA_SYLLABLES.match(w) for w in words):
        return "ja"
    scores = {lang: sum(w in sw for w in words) for lang, sw in STOPWORDS.items()}
    for lang, chars in ACCENTS.items():
        scores[lang] += 2 * sum(ch in lower for ch in chars)
    best = max(scores, key=scores.get)
    if scores[best] >= 1 and list(scores.values()).count(scores[best]) == 1:
        return best
    if len(words) >= 2 and all(PINYIN.match(w) for w in words):
        return "zh"
    if len(words) >= 2 and any(KO_MARK.search(w) for w in words) and all(KO_SYLLABLES.match(w) for w in words):
        return "ko"
    if len(words) >= 2 and all(len(w) <= 5 and YUE.match(w) for w in words):
        return "zh"  # cantonais (« Mou gaan dou »)
    if len(words) >= 2 and all(JA_SYLLABLES.match(w) for w in words) and sum(map(len, words)) / len(words) >= 4.5:
        return "ja"
    if len(words) == 1 and KO_SYLLABLES.match(words[0]) and re.search(r"(eo|ae|ui)", words[0]) and len(words[0]) >= 7:
        return "ko"  # « Gisaengchung »
    return "other"  # un titre étranger dont on ne reconnaît pas la langue


# --------------------------------------------------------------------------- noms
KO_SURNAMES = set("kim lee park choi jung jeong kang cho jo yoon yun jang lim im han oh seo shin kwon hwang ahn an "
                  "song yoo yu hong jeon jun ko go moon mun yang son bae baek heo nam sim noh ha kwak sung cha "
                  "joo woo gu min ryu na jin uhm um yeon ma won bong".split())
ZH_SURNAMES = set("zhang wang li liu chen yang huang zhao zhou wu xu sun zhu ma hu guo he lin gao luo zheng liang "
                  "xie tang han feng deng cao peng zeng xiao tian dong pan yuan cai jiang yu du ye cheng wei su lu "
                  "ding shen ren yao jin chan wong leung lau cheung chow ng tsui yuen yen tse kwan lai lam tam "
                  "fung mok chiu chu ho siu hui tong gong jet chung yeung kwok shu".split())
HI_SURNAMES = set("khan kapoor kumar singh sharma chopra bachchan roshan rao reddy mukherjee padukone kaif joshi "
                  "madhavan dutt devgn shetty patel gupta verma agarwal bhatt johar dixit rai chatterjee banerjee "
                  "malhotra mehta desai iyer menon nair pillai rajinikanth haasan ghosh kashyap akhtar dhawan hooda".split())
TH_NAME = re.compile(r"^[a-z]{6,}(kul|sakul|porn|pong|chai|phan|wat|siri|sak|thorn|rat|anon|wong)$")


def _person_lang(name: str) -> str | None:
    raw = name.strip()
    if not raw:
        return None
    if HANGUL.search(raw):
        return "ko"
    if KANA.search(raw):
        return "ja"
    if HAN.search(raw):
        return "zh"
    if CYRILLIC.search(raw):
        return "ru"
    parts = re.findall(r"[a-z]+(?:-[a-z]+)?", _ascii(raw))
    if len(parts) < 2:
        return None
    first, last = parts[0], parts[-1]
    plain = [p.replace("-", "") for p in parts]
    # « Song Kang-ho », « Chow Yun-fat », « Tony Leung Chiu-wai » : nom de famille puis prénom à trait d'union
    if "-" in last:
        given = last.replace("-", "")
        if first in KO_SURNAMES and KO_SYLLABLES.match(given) and (KO_MARK.search(given) or first not in ZH_SURNAMES):
            return "ko"
        if first in ZH_SURNAMES or parts[-2] in ZH_SURNAMES:
            return "zh"
    if len(plain) == 3 and plain[0] in KO_SURNAMES and all(KO_SYLLABLES.match(p) for p in plain[1:]) \
            and len(plain[1]) + len(plain[2]) <= 8 and plain[0] not in ZH_SURNAMES:
        return "ko"  # « Bong Joon Ho »
    if last in HI_SURNAMES or first in HI_SURNAMES:
        return "hi"
    if TH_NAME.match(last) and len(first) >= 4 and last not in ZH_SURNAMES:
        return "th"
    if any(ch in raw for ch in "ōūôû"):
        return "ja"
    if all(len(p) >= 3 and JA_SYLLABLES.match(p) for p in plain) and sum(map(len, plain)) >= 10:
        return "ja"
    if plain[0] in ZH_SURNAMES and all(PINYIN.match(p) for p in plain[1:]) and len(plain) <= 3:
        return "zh"  # « Zhang Ziyi » (pinyin, nom d'abord)
    if plain[-1] in ZH_SURNAMES and all(PINYIN.match(p) or YUE.match(p) for p in plain):
        return "zh"  # « Yimou Zhang »
    if first in RU_FIRST or RU_NAME.search(last) and len(last) >= 6 and RU_NAME.search(last).group(1) not in ("in", "ina"):
        return "ru" if (first in RU_FIRST or last.endswith(("skiy", "skaya", "enko", "ovich", "yev", "yeva"))) else None
    if any(ch in raw for ch in "áíóúñ"):
        return "es?"  # indice faible : nom accentué à l'espagnole
    if sum(ch in raw for ch in "éèçêë") >= 1:
        return "fr?"  # indice faible : prénom ou nom accentué à la française
    return None


def _split(x) -> list[str]:
    if not isinstance(x, str):
        return []
    return [v.strip() for v in x.split(",") if v.strip()]


# --------------------------------------------------------------------------- calcul


def infer_origins(df: pd.DataFrame) -> list[str]:
    """Un code d'origine par film (même ordre que df). "en" quand aucun indice ne dit le contraire."""
    if "original_language" in df and df["original_language"].notna().mean() > 0.5:
        return [str(x) if isinstance(x, str) and x else "en" for x in df["original_language"]]

    n = len(df)
    votes: list[Counter] = [Counter() for _ in range(n)]
    titles = df["title"].astype(str).tolist()
    tags = df["tags"].tolist() if "tags" in df else [[]] * n
    directors = df["directors"].tolist() if "directors" in df else [""] * n
    cast = df["cast"].tolist() if "cast" in df else [""] * n

    for i in range(n):
        for alt in _alt_titles(titles[i]):
            lang = _lang_of_text(alt)
            if lang:
                votes[i][lang] += 1 if lang == "other" else 3
        lang = _lang_of_text(re.sub(r"\([^()]*\)", " ", titles[i])) if re.search(r"[^\x00-ɏ]", titles[i]) else None
        if lang:
            votes[i][lang] += 3
        tg = tags[i] if isinstance(tags[i], (list, tuple)) else _split(tags[i])
        weak = Counter()
        for t in tg:
            t = _ascii(t).strip()
            lang = TAG_ORIGIN.get(t)
            if lang and t in WEAK_TAGS:
                weak[lang] = 1  # un tag de lieu ne suffit jamais seul
            elif lang:
                votes[i][lang] += 2
        votes[i].update(weak)
        people = _split(directors[i])[:2] + _split(cast[i])[:4]
        langs = [_person_lang(p) for p in people]
        found = Counter(l for l in langs if l)
        if found:
            top, count = found.most_common(1)[0]
            # la majorité de l'équipe (pas un seul acteur asiatique dans un film américain)
            if count >= 2 and count * 2 >= len(people):
                if top.endswith("?"):
                    votes[i][top[:-1]] += 1
                else:
                    votes[i][top] += count

    def decide(v: Counter) -> str:
        if not v:
            return "en"
        lang, score = v.most_common(1)[0]
        if lang == "other":
            return "other"  # titre original étranger, langue non reconnue
        return lang if score >= 2 else "en"

    origin = [decide(v) for v in votes]
    # un réalisateur tourne en général dans son pays : on propage à ses autres films
    by_director: dict[str, Counter] = defaultdict(Counter)
    for i in range(n):
        for d in _split(directors[i])[:1]:
            by_director[d][origin[i]] += 1
    for i in range(n):
        if origin[i] != "en":
            continue
        for d in _split(directors[i])[:1]:
            c = by_director[d]
            foreign = {k: v for k, v in c.items() if k not in ("en", "other")}
            if foreign:
                lang, count = max(foreign.items(), key=lambda kv: kv[1])
                if count >= 2 and count >= c["en"]:
                    origin[i] = lang
    return origin


# --------------------------------------------------------------------------- demandes

# mot du message (normalisé, sans accents) -> codes d'origine
ORIGIN_WORDS = {
    "asiatique": ASIA, "asiatiques": ASIA, "asie": ASIA, "asian": ASIA, "asia": ASIA,
    "japonais": {"ja"}, "japonaise": {"ja"}, "japonaises": {"ja"}, "japon": {"ja"}, "japanese": {"ja"},
    "japan": {"ja"}, "nippon": {"ja"},
    "coreen": {"ko"}, "coreens": {"ko"}, "coreenne": {"ko"}, "coreennes": {"ko"}, "coree": {"ko"},
    "korean": {"ko"}, "korea": {"ko"},
    "chinois": {"zh"}, "chinoise": {"zh"}, "chinoises": {"zh"}, "chine": {"zh"}, "chinese": {"zh"},
    "china": {"zh"}, "hongkongais": {"zh"}, "hong kong": {"zh"}, "taiwanais": {"zh"}, "cantonais": {"zh"},
    "indien": {"hi"}, "indienne": {"hi"}, "indiens": {"hi"}, "bollywood": {"hi"}, "indian": {"hi"},
    "thai": {"th"}, "thailandais": {"th"}, "thailandaise": {"th"},
    "europeen": EUROPE, "europeens": EUROPE, "europeenne": EUROPE, "european": EUROPE,
    "francais": {"fr"}, "francaise": {"fr"}, "francaises": {"fr"}, "french": {"fr"},
    "italien": {"it"}, "italienne": {"it"}, "italiens": {"it"}, "italian": {"it"},
    "espagnol": {"es"}, "espagnole": {"es"}, "spanish": {"es"}, "hispanique": {"es"},
    "mexicain": {"es"}, "argentin": {"es"},
    "allemand": {"de"}, "allemande": {"de"}, "german": {"de"},
    "russe": {"ru"}, "russes": {"ru"}, "russian": {"ru"}, "sovietique": {"ru"}, "sovietiques": {"ru"},
    "suedois": {"sv"}, "danois": {"da"}, "scandinave": {"sv", "da"}, "bresilien": {"pt"},
    "americain": {"en"}, "americains": {"en"}, "americaine": {"en"}, "hollywoodien": {"en"},
    "american": {"en"}, "anglais": {"en"}, "britannique": {"en"}, "british": {"en"},
}


def origins_in(norm_text: str) -> list[str]:
    """Codes d'origine demandés dans un message déjà normalisé (minuscules, sans accents)."""
    found: set[str] = set()
    for word, codes in sorted(ORIGIN_WORDS.items(), key=lambda kv: -len(kv[0])):
        if re.search(rf"\b{re.escape(word)}\b", norm_text):
            # « en français » parle de la langue de la réponse, pas du film
            if word in ("francais", "anglais") and re.search(rf"\b(en|in) {word}\b", norm_text) \
                    and not re.search(rf"\bfilms? {word}", norm_text):
                continue
            found |= codes
    return sorted(found)


__all__ = ["infer_origins", "origins_in", "ORIGIN_WORDS", "ASIA", "EUROPE"]

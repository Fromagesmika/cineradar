"""Matrice de mémoire : ce que le moteur sait de l'utilisateur à l'instant t.

La base (partie 4) stocke les événements bruts. Ici on les transforme en une
« mémoire » exploitable par le classement :

- liked / disliked      : films aimés / détestés (questionnaire, boutons, notes, chat), avec poids
- seen, not_interested  : à exclure ou à pénaliser
- shown                 : films déjà recommandés (pour ne pas reproposer sans cesse les mêmes)
- genre_affinity        : vecteur utilisateur × genre, recalculé à chaque requête avec décroissance
                          temporelle → c'est la ligne de l'utilisateur dans la « 3e matrice »
- facts                 : infos de profil (« regarde avec ses enfants », « déteste l'horreur »…)
- cinephile             : curseur mainstream (0) ↔ cinéphile (1)

Deux fournisseurs : InMemoryProvider (tests, démo, mocks) et SQLiteProvider (base de Benoît).
"""
from __future__ import annotations

import json
import math
import sqlite3
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from .catalog import CANONICAL_GENRES, get_catalog

# Poids de chaque type d'événement. >0 = signal positif, <0 = négatif.
EVENT_WEIGHTS = {
    "onboarding_like": 1.0,
    "like": 1.0,
    "chat_like": 0.8,        # « j'ai adoré X » dans le chat
    "onboarding_dislike": -1.0,
    "dislike": -1.0,
    "chat_dislike": -0.8,
    "not_interested": -0.4,  # pas forcément détesté : juste pas maintenant
    "seen": 0.0,
    "shown": 0.0,            # recommandé (impression)
    "recommended": 0.0,
}
HALF_LIFE_DAYS = 60.0


@dataclass
class UserMemory:
    user_id: int
    liked: dict[int, float] = field(default_factory=dict)
    disliked: dict[int, float] = field(default_factory=dict)
    seen: set[int] = field(default_factory=set)
    not_interested: set[int] = field(default_factory=set)
    shown: dict[int, int] = field(default_factory=dict)  # movie_id -> nb de fois montré
    genre_affinity: dict[str, float] = field(default_factory=dict)
    facts: dict[str, list[str]] = field(default_factory=dict)
    cinephile: float | None = None
    embedding: np.ndarray | None = None  # vecteur sauvegardé par la partie 4 (optionnel)

    @property
    def liked_ids(self) -> list[int]:
        return [m for m, _ in sorted(self.liked.items(), key=lambda x: -x[1])]

    @property
    def disliked_ids(self) -> list[int]:
        return list(self.disliked)

    def genre_vector(self) -> np.ndarray:
        return np.array([self.genre_affinity.get(g, 0.0) for g in CANONICAL_GENRES])

    def excluded_genres(self) -> set[str]:
        return {g for g in self.facts.get("genre_dislike", [])}

    def to_profile(self) -> dict:
        """Résumé lisible (utile pour le prompt « ami »)."""
        cat = get_catalog()
        names = lambda ids: [cat.get(m)["title"] for m in ids if cat.get(m)]
        top_genres = sorted(self.genre_affinity.items(), key=lambda x: -x[1])
        return {
            "user_id": self.user_id,
            "liked_titles": names(self.liked_ids[:10]),
            "disliked_titles": names(self.disliked_ids[:10]),
            "favorite_genres": [g for g, s in top_genres if s > 0.2][:5],
            "avoided_genres": [g for g, s in top_genres if s < -0.2][:5],
            "facts": self.facts,
            "cinephile": self.cinephile,
        }


def build_memory(
    user_id: int,
    events: list[dict],
    facts: list[tuple[str, str]] | None = None,
    cinephile: float | None = None,
    now: float | None = None,
    embedding: np.ndarray | None = None,
) -> UserMemory:
    """events : dicts {movie_id, type, value?, ts?}. `value` = note (0.5..5) si type == 'rating'."""
    cat = get_catalog()
    now = now or time.time()
    mem = UserMemory(user_id=user_id, cinephile=cinephile, embedding=embedding)
    item_score: dict[int, float] = defaultdict(float)
    genre_acc = np.zeros(len(CANONICAL_GENRES))
    genre_norm = 1e-9

    for e in sorted(events, key=lambda e: str(e.get("ts") or 0)):
        if e.get("movie_id") is None:  # titre introuvable dans le catalogue
            continue
        mid = int(e["movie_id"])
        typ = str(e.get("type", "")).lower()
        ts = e.get("ts") or now
        if isinstance(ts, str):
            try:
                from datetime import datetime

                ts = datetime.fromisoformat(ts).timestamp()
            except ValueError:
                ts = now
        decay = 0.5 ** (max(0.0, now - float(ts)) / 86400.0 / HALF_LIFE_DAYS)
        # onboarding = déclaration explicite, on ne la fait pas trop décroître
        if typ.startswith("onboarding"):
            decay = max(decay, 0.7)

        if typ == "rating":
            r = float(e.get("value") or 3)
            w = (r - 3.0) / 2.0  # 5 -> +1, 1 -> -1
            mem.seen.add(mid)
        else:
            w = EVENT_WEIGHTS.get(typ, 0.0)

        if typ in ("seen", "rating", "like", "dislike"):
            mem.seen.add(mid)
        if typ == "not_interested":
            mem.not_interested.add(mid)
        if typ in ("shown", "recommended"):
            mem.shown[mid] = mem.shown.get(mid, 0) + 1

        if w != 0.0:
            item_score[mid] += w * decay
            row = cat.id_to_row.get(mid)
            if row is not None:
                g = cat.genre_matrix[row]
                if g.sum() > 0:
                    genre_acc += w * decay * g / math.sqrt(g.sum())
                    genre_norm += abs(w) * decay

    for mid, s in item_score.items():
        if s > 0.05:
            mem.liked[mid] = s
        elif (s < -0.05 and mid not in mem.not_interested) or s < -0.5:
            mem.disliked[mid] = -s

    # Faits de profil
    for key, value in facts or []:
        mem.facts.setdefault(key, [])
        if value not in mem.facts[key]:
            mem.facts[key].append(value)

    genre_vec = genre_acc / genre_norm
    for g in mem.facts.get("genre_like", []):
        if g in CANONICAL_GENRES:
            genre_vec[CANONICAL_GENRES.index(g)] += 0.6
    for g in mem.facts.get("genre_dislike", []):
        if g in CANONICAL_GENRES:
            genre_vec[CANONICAL_GENRES.index(g)] -= 1.0
    mem.genre_affinity = {g: float(np.clip(v, -1.5, 1.5)) for g, v in zip(CANONICAL_GENRES, genre_vec)}

    if mem.cinephile is None and mem.facts.get("cinephile"):
        try:
            mem.cinephile = float(mem.facts["cinephile"][-1])
        except ValueError:
            pass
    return mem


class MemoryProvider(Protocol):
    def get(self, user_id: int) -> UserMemory: ...


class InMemoryProvider:
    """Mémoire en RAM. Sert de mock à la base de la partie 4 (et aux tests)."""

    def __init__(self):
        self.events: dict[int, list[dict]] = defaultdict(list)
        self.facts: dict[int, list[tuple[str, str]]] = defaultdict(list)
        self.cinephile: dict[int, float] = {}

    def log(self, user_id: int, movie_id: int, type: str, value: float | None = None, ts: float | None = None):
        self.events[user_id].append({"movie_id": movie_id, "type": type, "value": value, "ts": ts or time.time()})

    def add_fact(self, user_id: int, key: str, value: str):
        self.facts[user_id].append((key, value))

    def onboarding(self, user_id: int, liked: list[int], disliked: list[int] = (),
                   genres_liked: list[str] = (), genres_disliked: list[str] = (), cinephile: float = 0.5):
        for m in liked:
            self.log(user_id, m, "onboarding_like")
        for m in disliked:
            self.log(user_id, m, "onboarding_dislike")
        for g in genres_liked:
            self.add_fact(user_id, "genre_like", g)
        for g in genres_disliked:
            self.add_fact(user_id, "genre_dislike", g)
        self.cinephile[user_id] = cinephile

    def get(self, user_id: int) -> UserMemory:
        return build_memory(user_id, self.events.get(user_id, []), self.facts.get(user_id, []),
                            self.cinephile.get(user_id))


class SQLiteProvider:
    """Lit la base SQLite de la partie 4 (src/service/db.py).

    Les types de la base (liked, disliked, seen, suggested, not_interested, rating) et leur source
    (questionnaire, chat, click) sont traduits vers les événements de build_memory. Pour les genres,
    seul le dernier avis compte (un genre aimé puis évité est évité), comme dans src/service/profile.py.
    """

    SQL_EVENTS = """
        SELECT movie_id,
               CASE WHEN type = 'liked' AND source = 'questionnaire' THEN 'onboarding_like'
                    WHEN type = 'disliked' AND source = 'questionnaire' THEN 'onboarding_dislike'
                    WHEN type = 'liked' AND source = 'chat' THEN 'chat_like'
                    WHEN type = 'disliked' AND source = 'chat' THEN 'chat_dislike'
                    WHEN type = 'liked' THEN 'like'
                    WHEN type = 'disliked' THEN 'dislike'
                    WHEN type = 'suggested' THEN 'shown'
                    ELSE type END,
               value, ts
        FROM interactions WHERE user_id = ? ORDER BY id"""
    SQL_FACTS = """
        SELECT CASE key WHEN 'genre_liked' THEN 'genre_like' WHEN 'genre_avoided' THEN 'genre_dislike'
                        ELSE key END,
               value
        FROM profile_facts f
        WHERE user_id = ? AND (key NOT IN ('genre_liked', 'genre_avoided') OR id = (
            SELECT MAX(id) FROM profile_facts g WHERE g.user_id = f.user_id AND g.value = f.value
            AND g.key IN ('genre_liked', 'genre_avoided')))
        ORDER BY id"""
    SQL_CINEPHILE = "SELECT cinephile_level FROM users WHERE user_id = ?"
    SQL_EMB = "SELECT vector FROM user_embeddings WHERE user_id = ?"

    def __init__(self, db_path: str):
        self.db_path = db_path

    def _q(self, con, sql, uid):
        try:
            return con.execute(sql, (uid,)).fetchall()
        except sqlite3.Error:
            return []

    def get(self, user_id: int) -> UserMemory:
        con = sqlite3.connect(self.db_path)
        try:
            events = [{"movie_id": r[0], "type": r[1], "value": r[2], "ts": r[3]}
                      for r in self._q(con, self.SQL_EVENTS, user_id)]
            facts = [(r[0], r[1]) for r in self._q(con, self.SQL_FACTS, user_id)]
            row = self._q(con, self.SQL_CINEPHILE, user_id)
            cinephile = float(row[0][0]) if row and row[0][0] is not None else None
            emb_row = self._q(con, self.SQL_EMB, user_id)
            emb = None
            if emb_row and emb_row[0][0] is not None:
                raw = emb_row[0][0]
                emb = np.array(json.loads(raw)) if isinstance(raw, str) else np.frombuffer(raw, dtype=np.float32)
        finally:
            con.close()
        return build_memory(user_id, events, facts, cinephile, embedding=emb)

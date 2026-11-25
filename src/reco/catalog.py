"""Catalogue de films : chargement, normalisation, recherche par titre et recherche sémantique.

Source : le catalogue de l'appli (src/service/catalog.py, mêmes identifiants MovieLens), sauf
RECO_CATALOG=mock (mini-catalogue de 79 films des tests) ou MOVIES_PATH=fichier.parquet.
Colonnes attendues (contrat) : movie_id, title, year, tmdb_id, redial_id, genres, tags,
overview, poster_url, popularity_bucket. `genres` et `tags` peuvent être des listes ou des
chaînes séparées par "|" / ";" / ",". `aliases` (titres français) est optionnelle.
"""
from __future__ import annotations

import difflib
import os
import re
import unicodedata
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from .mock_catalog import load_mock_catalog

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MOVIES_PATH = ROOT / "data" / "processed" / "movies.parquet"

# Genres MovieLens (la partie 1 garde ces noms)
CANONICAL_GENRES = [
    "Action", "Adventure", "Animation", "Children", "Comedy", "Crime", "Documentary",
    "Drama", "Fantasy", "Film-Noir", "Horror", "IMAX", "Musical", "Mystery", "Romance",
    "Sci-Fi", "Thriller", "War", "Western",
]

# Correspondance popularity_bucket -> popularité dans [0, 1] (1 = très connu)
_BUCKET_TO_POP = {
    "head": 1.0, "top": 1.0, "popular": 1.0, "high": 1.0, "mainstream": 1.0,
    "mid": 0.5, "medium": 0.5, "torso": 0.5,
    "tail": 0.0, "low": 0.0, "long_tail": 0.0, "niche": 0.0, "gem": 0.0,
}


def normalize_text(s: str, strip_article: bool = True) -> str:
    """minuscules, sans accents, sans ponctuation (+ sans article initial pour les titres)."""
    if not isinstance(s, str):
        return ""
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    s = s.lower()
    s = re.sub(r"\(\d{4}\)", " ", s)  # "Heat (1995)" -> "heat"
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    if strip_article:
        s = re.sub(r"^(the|a|an|le|la|les|l|un|une) ", "", s.strip())
    return re.sub(r"\s+", " ", s).strip()


def _split(x) -> list[str]:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return []
    if isinstance(x, (list, tuple, np.ndarray)):
        return [str(v).strip() for v in x if str(v).strip()]
    return [v.strip() for v in re.split(r"[|;,]", str(x)) if v.strip()]


class Catalog:
    def __init__(self, df: pd.DataFrame):
        df = df.copy().reset_index(drop=True)
        df["movie_id"] = df["movie_id"].astype(int)
        df["genres"] = df["genres"].apply(_split)
        df["tags"] = df["tags"].apply(_split) if "tags" in df else [[] for _ in range(len(df))]
        df["aliases"] = df["aliases"].apply(_split) if "aliases" in df else [[] for _ in range(len(df))]
        df["overview"] = df.get("overview", pd.Series([""] * len(df))).fillna("").astype(str)
        df["year"] = pd.to_numeric(df.get("year"), errors="coerce")
        df["pop"] = self._popularity(df)
        df["quality"] = self._quality(df)
        self.df = df
        self.ids = df["movie_id"].to_numpy()
        self.id_to_row = {int(m): i for i, m in enumerate(self.ids)}
        self.genre_matrix = np.array(
            [[1.0 if g in gs else 0.0 for g in CANONICAL_GENRES] for gs in df["genres"]]
        )
        self.keyword_haystack = [
            f" {normalize_text(' '.join([t, *tg, *gs]), False)} "
            for t, tg, gs in zip(df["title"], df["tags"], df["genres"])
        ]
        self._build_title_index()
        self._build_content_index()

    # ------------------------------------------------------------------ popularité
    @staticmethod
    def _popularity(df: pd.DataFrame) -> pd.Series:
        """Popularité dans [0,1]. Utilise `popularity` numérique si dispo (percentile), sinon le bucket."""
        if "popularity" in df and pd.api.types.is_numeric_dtype(df["popularity"]):
            return df["popularity"].rank(pct=True).fillna(0.5)
        if "popularity_bucket" in df:
            def conv(b):
                if isinstance(b, (int, float)) and not pd.isna(b):
                    # bucket numérique (ex. décile 0..9) -> on normalise plus bas
                    return float(b)
                return _BUCKET_TO_POP.get(str(b).lower().strip(), 0.5)
            s = df["popularity_bucket"].apply(conv).astype(float)
            if s.max() > 1.0:  # buckets numériques
                s = (s - s.min()) / max(s.max() - s.min(), 1e-9)
            return s
        return pd.Series(0.5, index=df.index)

    @staticmethod
    def _quality(df: pd.DataFrame) -> pd.Series:
        """Qualité dans [0,1] : percentile de la note moyenne bayésienne (une pépite = peu connue ET bien notée).
        0.5 partout si les notes sont absentes (catalogue mock ou liste de secours)."""
        if "mean_rating" not in df or df["mean_rating"].isna().all():
            return pd.Series(0.5, index=df.index)
        n = pd.to_numeric(df.get("n_ratings"), errors="coerce").fillna(0)
        mean = pd.to_numeric(df["mean_rating"], errors="coerce")
        prior = float((mean * n).sum() / max(n.sum(), 1))
        bayes = (n * mean.fillna(prior) + 50 * prior) / (n + 50)
        return bayes.rank(pct=True)

    # ------------------------------------------------------------------ titres
    def _build_title_index(self):
        self.title_keys: dict[str, int] = {}
        for mid, title, aliases in zip(self.ids, self.df["title"], self.df["aliases"]):
            for t in [title, *aliases]:
                k = normalize_text(t)
                if k and k not in self.title_keys:
                    self.title_keys[k] = int(mid)
        # titres courts : « Mad Max: Fury Road » -> « mad max » (si ce titre n'existe pas déjà seul)
        for mid, title in zip(self.ids, self.df["title"]):
            head = re.split(r":| - |\(", str(title))[0]
            k = normalize_text(head)
            if k and len(k) >= 4 and k not in self.title_keys:
                self.title_keys[k] = int(mid)
        self._keys_list = list(self.title_keys)

    def find_title(self, text: str, cutoff: float = 0.82) -> int | None:
        """Retrouve un film à partir d'un titre approximatif (accents, casse, fautes légères)."""
        k = normalize_text(text)
        if not k:
            return None
        if k in self.title_keys:
            return self.title_keys[k]
        m = difflib.get_close_matches(k, self._keys_list, n=1, cutoff=cutoff)
        return self.title_keys[m[0]] if m else None

    def titles_in_text(self, text: str) -> list[int]:
        """Films dont le titre apparaît tel quel dans un texte (pour vérifier la réponse du LLM)."""
        t = f" {normalize_text(text, strip_article=False)} "
        # un titre d'un seul mot (« Heat », « Chose ») ne compte que s'il est écrit avec une majuscule
        caps = {normalize_text(w, False) for w in re.findall(r"\b[A-Z0-9][\w'’-]*", text or "")}
        found = []
        for k, mid in self.title_keys.items():
            if len(k) >= 3 and f" {k} " in t and (" " in k or k in caps):
                found.append(mid)
        return list(dict.fromkeys(found))

    # ------------------------------------------------------------------ contenu
    def _content_text(self, row) -> str:
        genres = " ".join(row["genres"]) * 2
        tags = " ".join(row["tags"]) * 2
        return f"{row['title']} {' '.join(row['aliases'])} {genres} {tags} {row['overview']}"

    def _build_content_index(self):
        texts = [self._content_text(r) for _, r in self.df.iterrows()]
        self._st_model = None
        if os.getenv("USE_SENTENCE_TRANSFORMERS") == "1":
            try:  # embeddings multilingues si dispo (Colab)
                from sentence_transformers import SentenceTransformer

                self._st_model = SentenceTransformer(
                    os.getenv("ST_MODEL", "paraphrase-multilingual-MiniLM-L12-v2")
                )
                self.content_emb = self._st_model.encode(texts, normalize_embeddings=True)
                return
            except Exception:
                self._st_model = None
        self._tfidf = TfidfVectorizer(
            strip_accents="unicode", lowercase=True, ngram_range=(1, 2), min_df=1, sublinear_tf=True
        )
        self.content_emb = normalize(self._tfidf.fit_transform(texts))

    def semantic_scores(self, query: str) -> np.ndarray:
        """Similarité cosinus entre une requête libre et chaque film (ordre de self.ids)."""
        if not query or not query.strip():
            return np.zeros(len(self.ids))
        if self._st_model is not None:
            q = self._st_model.encode([query], normalize_embeddings=True)
            return (self.content_emb @ q.T).ravel()
        q = normalize(self._tfidf.transform([query]))
        return np.asarray((self.content_emb @ q.T).todense()).ravel()

    def content_similarity(self, movie_ids: list[int]) -> np.ndarray:
        """Similarité de contenu moyenne entre des films de référence et tout le catalogue."""
        rows = [self.id_to_row[m] for m in movie_ids if m in self.id_to_row]
        if not rows:
            return np.zeros(len(self.ids))
        ref = self.content_emb[rows]
        sims = ref @ self.content_emb.T
        sims = np.asarray(sims.todense()) if hasattr(sims, "todense") else np.asarray(sims)
        return sims.mean(axis=0)

    # ------------------------------------------------------------------ accès
    def get(self, movie_id: int) -> dict | None:
        i = self.id_to_row.get(int(movie_id))
        if i is None:
            return None
        r = self.df.iloc[i]
        return {
            "movie_id": int(r["movie_id"]),
            "title": r["title"],
            "year": None if pd.isna(r["year"]) else int(r["year"]),
            "genres": list(r["genres"]),
            "tags": list(r["tags"]),
            "overview": r["overview"],
            "poster_url": r.get("poster_url"),
            "popularity": float(r["pop"]),
        }

    def __len__(self):
        return len(self.ids)


MIN_RATINGS = 20  # films with fewer MovieLens ratings are too obscure to recommend (and too many)


def _app_catalog() -> pd.DataFrame:
    """Same films and ids as the app (src/service/catalog.py): export, movies.parquet or seed list"""
    from src.service import catalog as app_catalog

    films = app_catalog.load().reset_index(drop=True)
    if films["n_ratings"].sum() > 0:
        films = films[films["n_ratings"] >= MIN_RATINGS]
    return films.sort_values("popularity", ascending=False)


@lru_cache(maxsize=2)
def _cached(path: str | None, backend: str | None) -> Catalog:
    if backend == "mock":
        return Catalog(load_mock_catalog())
    if path and Path(path).exists():
        return Catalog(pd.read_parquet(path))
    return Catalog(_app_catalog())


def get_catalog(path: str | None = None) -> Catalog:
    """RECO_CATALOG=mock gives the 79-film test catalog of part 3, MOVIES_PATH a parquet file"""
    return _cached(path or os.getenv("MOVIES_PATH"), os.getenv("RECO_CATALOG"))


get_catalog.cache_clear = _cached.cache_clear

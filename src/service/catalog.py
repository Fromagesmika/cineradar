import re
import unicodedata
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import POPULAR_SHARE, PROCESSED, ROOT

ARTIFACTS = ROOT / "artifacts"
EXPORTED = ARTIFACTS / "catalog.csv.gz"
SEED = Path(__file__).resolve().parent / "seed_movies.csv"
POSTER_BASE = "https://image.tmdb.org/t/p/w342"
ARTICLES = ("the", "a", "an", "le", "la", "les", "l", "un", "une", "der", "die", "das", "el", "il")
COLUMNS = ["movie_id", "title", "year", "genres", "tags", "overview", "poster_url", "tmdb_id", "redial_id",
           "directors", "cast", "n_ratings", "mean_rating", "original_language"]


def normalize(text) -> str:
    text = unicodedata.normalize("NFKD", str(text)).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def fix_title(title) -> str:
    title = str(title).strip()
    match = re.match(r"^(.*), (The|A|An|Le|La|Les|L'|Un|Une|Der|Die|Das|El|Il)( \(.*\))?$", title)
    if match:
        head, article, tail = match.groups()
        joiner = "" if article.endswith("'") else " "
        return f"{article}{joiner}{head}{tail or ''}"
    return title


def short_key(key: str) -> str:
    words = key.split()
    return " ".join(words[1:]) if len(words) > 1 and words[0] in ARTICLES else key


def _read_source() -> tuple[pd.DataFrame, str]:
    if EXPORTED.exists():
        return pd.read_csv(EXPORTED), "export"
    movies_path = PROCESSED / "movies.parquet"
    if movies_path.exists():
        movies = pd.read_parquet(movies_path)
        if "clean_title" in movies.columns:
            movies = movies.assign(title=movies["clean_title"])
        return movies, "processed"
    return pd.read_csv(SEED), "seed"


def _with_columns(movies: pd.DataFrame) -> pd.DataFrame:
    movies = movies.copy()
    if "poster_path" in movies.columns and "poster_url" not in movies.columns:
        movies["poster_url"] = movies["poster_path"].map(lambda p: f"{POSTER_BASE}{p}" if isinstance(p, str) and p else None)
    for column in COLUMNS:
        if column not in movies.columns:
            movies[column] = None
    movies = movies[COLUMNS].copy()
    movies["movie_id"] = movies["movie_id"].astype(int)
    movies["title"] = movies["title"].map(fix_title)
    movies["year"] = pd.to_numeric(movies["year"], errors="coerce")
    movies["tmdb_id"] = pd.to_numeric(movies["tmdb_id"], errors="coerce")
    for column in ("genres", "tags", "overview", "directors", "cast"):
        movies[column] = movies[column].fillna("").astype(str)
    movies["n_ratings"] = pd.to_numeric(movies["n_ratings"], errors="coerce").fillna(0)
    movies["mean_rating"] = pd.to_numeric(movies["mean_rating"], errors="coerce")
    movies["poster_url"] = movies["poster_url"].where(movies["poster_url"].notna(), None)
    return movies


def _likes() -> dict:
    ids_path, likes_path = ARTIFACTS / "movie_ids.npy", ARTIFACTS / "item_likes.npy"
    if not (ids_path.exists() and likes_path.exists()):
        return {}
    return dict(zip(np.load(ids_path).tolist(), np.load(likes_path).tolist()))


@lru_cache(maxsize=1)
def load() -> pd.DataFrame:
    movies, source = _read_source()
    movies = _with_columns(movies)
    likes = _likes()
    movies["likes"] = movies["movie_id"].map(likes).fillna(0.0).astype(float)
    if source == "seed" and not likes:
        movies["likes"] = np.linspace(len(movies), 1, len(movies))
    in_graph = movies["likes"] > 0
    # Popularity: number of ratings in the full MovieLens 25M (part 1) when known, else likes in the graph
    count = movies["n_ratings"] if movies["n_ratings"].sum() > 0 else movies["likes"]
    rated = count > 0
    threshold = count[rated].quantile(1 - POPULAR_SHARE) if rated.any() else np.inf
    movies["popularity"] = count
    movies["popular"] = count >= threshold
    movies["in_graph"] = in_graph
    movies["key"] = movies["title"].map(normalize)
    movies["short_key"] = movies["key"].map(short_key)
    movies["genre_set"] = movies["genres"].map(lambda g: frozenset(x.strip() for x in g.split(",") if x.strip()))
    movies.attrs["source"] = source
    return movies.set_index("movie_id", drop=False).sort_values(["popularity", "likes"], ascending=False)


def source() -> str:
    return load().attrs.get("source", "seed")


def all_genres() -> list[str]:
    genres = set().union(*load()["genre_set"])
    return sorted(g for g in genres if g not in {"IMAX", "(no genres listed)"})


def card(movie_id: int) -> dict | None:
    films = load()
    if movie_id not in films.index:
        return None
    row = films.loc[movie_id]
    return {
        "movie_id": int(row["movie_id"]),
        "title": row["title"],
        "year": None if pd.isna(row["year"]) else int(row["year"]),
        "genres": row["genres"],
        "overview": row["overview"],
        "poster_url": row["poster_url"],
        "directors": row["directors"],
        "mean_rating": None if pd.isna(row["mean_rating"]) else round(float(row["mean_rating"]), 1),
        "hidden_gem": not bool(row["popular"]),
    }


def cards(movie_ids) -> list[dict]:
    return [c for c in (card(int(m)) for m in movie_ids) if c is not None]


def title(movie_id) -> str | None:
    found = card(int(movie_id)) if movie_id is not None else None
    return found["title"] if found else None


def search(query: str, limit: int = 10) -> list[dict]:
    key = normalize(query)
    if not key:
        return []
    films = load()
    hits = films[films["key"].str.contains(key, regex=False) | films["short_key"].str.contains(key, regex=False)]
    starts = hits["key"].str.startswith(key) | hits["short_key"].str.startswith(key)
    hits = pd.concat([hits[starts], hits[~starts]])
    return cards(hits["movie_id"].head(limit))


def popular(limit: int = 50, genres=None) -> list[dict]:
    films = load()
    films = films[films["in_graph"]] if films["in_graph"].any() else films
    films = films.sort_values("likes", ascending=False)
    if genres:
        wanted = set(genres)
        films = films[films["genre_set"].map(lambda g: bool(g & wanted))]
    return cards(films["movie_id"].head(limit))


def find_titles(text: str, min_length: int = 4, max_films: int = 20000) -> list[tuple[int, int, int]]:
    key = f" {normalize(text)} "
    films = load().head(max_films)
    found = []
    for movie_id, full, short in zip(films["movie_id"], films["key"], films["short_key"]):
        for title_key in dict.fromkeys((full, short)):
            if len(title_key) < min_length:
                continue
            start = key.find(f" {title_key} ")
            if start >= 0:
                found.append((int(movie_id), start, start + len(title_key) + 1))
                break
    found.sort(key=lambda f: (f[1], -(f[2] - f[1])))
    kept, end = [], -1
    for movie_id, s, e in found:
        if s >= end:
            kept.append((movie_id, s, e))
            end = e
    return kept

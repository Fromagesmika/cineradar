"""Add TMDB details to every MovieLens film (part 1)

Source: a public dump of the TMDB catalogue (data/raw/tmdb/tmdb_movies.csv, downloaded by
scripts/download_data.py), so no TMDB API key is needed. Films are matched with the tmdbId
of MovieLens links.csv, then with the IMDb id when the TMDB id is missing

Writes data/processed/tmdb.parquet and adds these columns to movies.parquet:
tmdb_id, overview, poster_url, directors, cast, n_ratings, mean_rating, popularity_bucket

Run: python -m src.data.tmdb   (after src.data.preprocess)
"""
import numpy as np
import pandas as pd

from src.config import ML_DIR, POPULAR_SHARE, PROCESSED, RAW

TMDB_CSV = RAW / "tmdb" / "tmdb_movies.csv"
POSTER_BASE = "https://image.tmdb.org/t/p/w342"
ADDED = ["tmdb_id", "overview", "poster_url", "directors", "cast", "n_ratings", "mean_rating", "popularity_bucket"]


def first(text, n):
    if not isinstance(text, str):
        return ""
    return ", ".join([t.strip() for t in text.split(",") if t.strip()][:n])


def load_tmdb() -> pd.DataFrame:
    cols = ["id", "tconst", "overview", "poster_path", "directors", "cast", "original_language"]
    tmdb = pd.read_csv(TMDB_CSV, usecols=cols, dtype={"tconst": str}, low_memory=False)
    tmdb["id"] = pd.to_numeric(tmdb["id"], errors="coerce")
    tmdb = tmdb.dropna(subset=["id"]).drop_duplicates("id")
    tmdb["id"] = tmdb["id"].astype(int)
    tmdb["cast"] = tmdb["cast"].map(lambda c: first(c, 4))
    tmdb["directors"] = tmdb["directors"].map(lambda d: first(d, 2))
    return tmdb


def match(links: pd.DataFrame, tmdb: pd.DataFrame) -> pd.DataFrame:
    by_tmdb = links.merge(tmdb, left_on="tmdbId", right_on="id", how="inner")
    rest = links[~links["movieId"].isin(by_tmdb["movieId"])].copy()
    rest["tconst"] = rest["imdbId"].map(lambda i: f"tt{int(i):07d}")
    by_imdb = rest.merge(tmdb.dropna(subset=["tconst"]).drop_duplicates("tconst"), on="tconst", how="inner")
    found = pd.concat([by_tmdb, by_imdb], ignore_index=True)
    return pd.DataFrame({
        "movie_id": found["movieId"],
        "tmdb_id": found["id"],
        "overview": found["overview"].fillna(""),
        "poster_path": found["poster_path"],
        "directors": found["directors"].fillna(""),
        "cast": found["cast"].fillna(""),
        "original_language": found["original_language"],
    }).drop_duplicates("movie_id")


def rating_stats() -> pd.DataFrame:
    """Number of ratings and mean rating of each film in the full MovieLens 25M"""
    ratings = pd.read_csv(ML_DIR / "ratings.csv", usecols=["movieId", "rating"],
                          dtype={"movieId": "int32", "rating": "float32"})
    return ratings.groupby("movieId")["rating"].agg(["size", "mean"])


def main():
    links = pd.read_csv(ML_DIR / "links.csv")
    details = match(links, load_tmdb())
    details.to_parquet(PROCESSED / "tmdb.parquet", index=False)

    movies = pd.read_parquet(PROCESSED / "movies.parquet")
    movies = movies.drop(columns=[c for c in ADDED if c in movies.columns])
    movies = movies.merge(details.drop(columns=["original_language"]), on="movie_id", how="left")
    movies["poster_url"] = movies.pop("poster_path").map(
        lambda p: f"{POSTER_BASE}{p}" if isinstance(p, str) and p.startswith("/") else None)
    for col in ("overview", "directors", "cast"):
        movies[col] = movies[col].fillna("")
    stats = rating_stats()
    movies["n_ratings"] = movies["movie_id"].map(stats["size"]).fillna(0).astype(int)
    movies["mean_rating"] = movies["movie_id"].map(stats["mean"]).round(2)

    # Same rule as the graph: the 20% most rated films (among films with ratings) are popular
    rated = movies["n_ratings"] > 0
    threshold = np.quantile(movies.loc[rated, "n_ratings"], 1 - POPULAR_SHARE)
    movies["popularity_bucket"] = np.where(movies["n_ratings"] >= threshold, "popular", "gem")
    movies.to_parquet(PROCESSED / "movies.parquet", index=False)

    has = movies["overview"].str.len() > 0
    print(f"{len(details):,} MovieLens films matched with TMDB")
    print(f"summaries: {has.sum():,} / {len(movies):,} films, posters: {movies['poster_url'].notna().sum():,}")
    print(f"popular threshold: {threshold:.0f} ratings")


if __name__ == "__main__":
    main()

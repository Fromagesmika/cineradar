import pandas as pd

from src.config import ML_DIR, PROCESSED
from src.service.catalog import EXPORTED

KEEP = ["movie_id", "title", "year", "genres", "tags", "tmdb_id", "overview", "poster_path"]


def main():
    movies = pd.read_parquet(PROCESSED / "movies.parquet")
    movies = movies.assign(title=movies["clean_title"]) if "clean_title" in movies.columns else movies
    links = ML_DIR / "links.csv"
    if links.exists() and "tmdb_id" not in movies.columns:
        ids = pd.read_csv(links).rename(columns={"movieId": "movie_id", "tmdbId": "tmdb_id"})
        movies = movies.merge(ids[["movie_id", "tmdb_id"]], on="movie_id", how="left")
    tmdb = PROCESSED / "tmdb.parquet"
    if tmdb.exists():
        details = pd.read_parquet(tmdb)
        extra = [c for c in ("overview", "poster_path") if c in details.columns]
        movies = movies.drop(columns=[c for c in extra if c in movies.columns]).merge(
            details[["movie_id", *extra]], on="movie_id", how="left")
    movies = movies[[c for c in KEEP if c in movies.columns]]
    if "overview" in movies.columns:
        movies["overview"] = movies["overview"].fillna("").str.slice(0, 600)
    movies.to_csv(EXPORTED, index=False, compression="gzip")
    print(f"{len(movies):,} films written to {EXPORTED} ({EXPORTED.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()

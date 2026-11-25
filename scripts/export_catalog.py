"""Write artifacts/catalog.csv.gz, the film catalog the app uses without the raw data

Run once after the data pipeline (src.data.preprocess, src.data.tmdb, src.data.redial):
python -m scripts.export_catalog
"""
import pandas as pd

from src.config import PROCESSED
from src.service.catalog import EXPORTED

KEEP = ["movie_id", "title", "year", "genres", "tags", "tmdb_id", "redial_id", "overview", "poster_url",
        "directors", "cast", "n_ratings", "mean_rating"]


def main():
    movies = pd.read_parquet(PROCESSED / "movies.parquet")
    movies = movies.assign(title=movies["clean_title"]) if "clean_title" in movies.columns else movies
    movies = movies[[c for c in KEEP if c in movies.columns]]
    if "overview" in movies.columns:
        movies["overview"] = movies["overview"].fillna("").str.slice(0, 400)
    EXPORTED.parent.mkdir(exist_ok=True)
    movies.to_csv(EXPORTED, index=False, compression="gzip")
    print(f"{len(movies):,} films written to {EXPORTED} ({EXPORTED.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()

"""Clean MovieLens 25M (owner: Benoît)

Outputs in data/processed/
- interactions.parquet: user_id, movie_id, timestamp (likes only, sampled users, 10-core)
- movies.parquet: movie_id, title, year, genres, tags (top Tag Genome tags), text

Run: python -m src.data.preprocess
"""
import re

import numpy as np
import pandas as pd

from src.config import LIKE_THRESHOLD, MIN_LIKES, ML_DIR, N_USERS, PROCESSED, SEED


def k_core(df: pd.DataFrame, k: int) -> pd.DataFrame:
    """Drop users and films until each one has at least k interactions"""
    while True:
        users = df["user_id"].map(df["user_id"].value_counts())
        movies = df["movie_id"].map(df["movie_id"].value_counts())
        keep = (users >= k) & (movies >= k)
        if keep.all():
            return df
        df = df[keep]


def split_title(title: str):
    match = re.match(r"^(.*)\s+\((\d{4})\)\s*$", title.strip())
    if match:
        return match.group(1).strip(), int(match.group(2))
    return title.strip(), None


def build_movie_table(movie_ids: set) -> pd.DataFrame:
    movies = pd.read_csv(ML_DIR / "movies.csv").rename(columns={"movieId": "movie_id"})
    parsed = movies["title"].map(split_title)
    movies["clean_title"] = parsed.map(lambda t: t[0])
    movies["year"] = parsed.map(lambda t: t[1])
    movies["genres"] = movies["genres"].str.replace("|", ", ", regex=False).replace("(no genres listed)", "")

    # Top 8 Tag Genome tags per film (relevance score from MovieLens)
    tags = pd.read_csv(ML_DIR / "genome-tags.csv")
    scores = pd.read_csv(ML_DIR / "genome-scores.csv", dtype={"movieId": "int32", "tagId": "int16", "relevance": "float32"})
    scores = scores[scores["movieId"].isin(movie_ids)]
    top = scores.sort_values(["movieId", "relevance"], ascending=[True, False]).groupby("movieId").head(8)
    top = top.merge(tags, on="tagId")
    top_tags = top.groupby("movieId")["tag"].apply(lambda s: ", ".join(s)).rename("tags")
    movies = movies.merge(top_tags, left_on="movie_id", right_index=True, how="left")
    movies["tags"] = movies["tags"].fillna("")

    movies["text"] = (
        movies["clean_title"] + " (" + movies["year"].astype("Int64").astype(str).replace("<NA>", "") + "). "
        + "Genres: " + movies["genres"] + ". Tags: " + movies["tags"]
    )
    return movies[["movie_id", "title", "clean_title", "year", "genres", "tags", "text"]]


def main():
    ratings = pd.read_csv(
        ML_DIR / "ratings.csv",
        dtype={"userId": "int32", "movieId": "int32", "rating": "float32", "timestamp": "int64"},
    ).rename(columns={"userId": "user_id", "movieId": "movie_id"})
    print(f"raw ratings: {len(ratings):,}")

    likes = ratings[ratings["rating"] >= LIKE_THRESHOLD][["user_id", "movie_id", "timestamp"]]
    rng = np.random.default_rng(SEED)
    eligible = likes["user_id"].value_counts()
    eligible = eligible[eligible >= MIN_LIKES].index.to_numpy()
    sampled = rng.choice(eligible, size=min(N_USERS, len(eligible)), replace=False)
    likes = k_core(likes[likes["user_id"].isin(sampled)], MIN_LIKES)
    likes.to_parquet(PROCESSED / "interactions.parquet", index=False)

    movies = build_movie_table(set(likes["movie_id"]) | set(pd.read_csv(ML_DIR / "movies.csv")["movieId"]))
    movies.to_parquet(PROCESSED / "movies.parquet", index=False)

    stats = {
        "users": likes["user_id"].nunique(),
        "films": likes["movie_id"].nunique(),
        "likes": len(likes),
        "density_%": round(100 * len(likes) / likes["user_id"].nunique() / likes["movie_id"].nunique(), 3),
    }
    print(stats)


if __name__ == "__main__":
    main()

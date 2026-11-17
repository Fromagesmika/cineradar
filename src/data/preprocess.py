"""Clean MovieLens 25M (owner: Benoît).

Input : data/raw/ml-25m/ratings.csv
Output: data/processed/interactions.parquet  (user_id, movie_id, rating, timestamp)
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw" / "ml-25m"
PROCESSED = ROOT / "data" / "processed"

LIKE_THRESHOLD = 4.0   # a rating >= 4 stars counts as a "like"
MIN_LIKES = 10         # keep users and films with at least 10 likes
MIN_YEAR = 2010        # keep recent interactions only
N_USERS = 30_000       # sample of users, so training stays fast


def k_core(df: pd.DataFrame, k: int) -> pd.DataFrame:
    """Repeat filtering until every user and film has at least k interactions."""
    while True:
        user_counts = df["user_id"].map(df["user_id"].value_counts())
        movie_counts = df["movie_id"].map(df["movie_id"].value_counts())
        keep = (user_counts >= k) & (movie_counts >= k)
        if keep.all():
            return df
        df = df[keep]


def main():
    ratings = pd.read_csv(RAW / "ratings.csv")
    ratings = ratings.rename(columns={"userId": "user_id", "movieId": "movie_id"})

    # TODO (Benoît): check these choices on the real data and report the final sizes.
    ratings = ratings[pd.to_datetime(ratings["timestamp"], unit="s").dt.year >= MIN_YEAR]
    likes = ratings[ratings["rating"] >= LIKE_THRESHOLD]
    users = likes["user_id"].drop_duplicates().sample(
        n=min(N_USERS, likes["user_id"].nunique()), random_state=42
    )
    likes = k_core(likes[likes["user_id"].isin(users)], MIN_LIKES)

    PROCESSED.mkdir(parents=True, exist_ok=True)
    likes.to_parquet(PROCESSED / "interactions.parquet", index=False)
    print(f"users={likes.user_id.nunique()} films={likes.movie_id.nunique()} likes={len(likes)}")


if __name__ == "__main__":
    main()

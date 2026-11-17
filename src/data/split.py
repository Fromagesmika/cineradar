"""Train / validation / test split by date + user-film graph (owner: Théodore).

For each user: the most recent 10% of likes go to test, the 10% before to
validation, the rest to train. The model must never see test data.
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"


def temporal_split(df: pd.DataFrame, val_ratio: float = 0.1, test_ratio: float = 0.1):
    df = df.sort_values(["user_id", "timestamp"])
    rank = df.groupby("user_id").cumcount()
    size = df.groupby("user_id")["movie_id"].transform("size")
    n_test = np.maximum(1, (size * test_ratio).astype(int))
    n_val = np.maximum(1, (size * val_ratio).astype(int))
    test = rank >= size - n_test
    val = (rank >= size - n_test - n_val) & ~test
    return df[~val & ~test], df[val], df[test]


def build_graph(train: pd.DataFrame):
    """TODO (Théodore): map ids to indices and build the normalized adjacency
    matrix A_hat = D^-1/2 A D^-1/2 of the user-film graph (train data only)."""
    raise NotImplementedError


def main():
    df = pd.read_parquet(PROCESSED / "interactions.parquet")
    train, val, test = temporal_split(df)
    for name, part in [("train", train), ("val", val), ("test", test)]:
        part.to_parquet(PROCESSED / f"{name}.parquet", index=False)
        print(name, len(part))


if __name__ == "__main__":
    main()

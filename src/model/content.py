"""Text embeddings of every film, from its title, genres, tags and TMDB summary (owner: Michaël)

Films with few ratings have weak collaborative vectors. Their text vector gives a second
source of information, used in src/model/export.py

Run: python -m src.model.content
Output: data/processed/text_emb.npy (one row per film of movies.parquet, same order)
"""
import numpy as np
import pandas as pd

from src.config import PROCESSED, TEXT_MODEL


def film_texts(movies: pd.DataFrame) -> list[str]:
    texts = movies["text"].fillna("")
    if "overview" in movies.columns:  # added by part 1 once TMDB is downloaded
        texts = texts + " " + movies["overview"].fillna("")
    return texts.tolist()


def main():
    from sentence_transformers import SentenceTransformer

    movies = pd.read_parquet(PROCESSED / "movies.parquet")
    model = SentenceTransformer(TEXT_MODEL, device="cpu")
    emb = model.encode(film_texts(movies), batch_size=256, show_progress_bar=True, normalize_embeddings=True)
    np.save(PROCESSED / "text_emb.npy", emb.astype(np.float16))
    print("text embeddings:", emb.shape)


if __name__ == "__main__":
    main()

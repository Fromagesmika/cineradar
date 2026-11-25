"""Download the raw data into data/raw/ (only needed to rebuild the data or retrain)

- MovieLens 25M (about 250 MB): ratings, films, tags
- ReDial (about 6 MB): movie-recommendation dialogues
- TMDB (about 280 MB): film summaries, posters, directors and cast, from a public dump of the
  TMDB catalogue on Hugging Face, so no TMDB API key is needed

Run: python scripts/download_data.py            (everything)
     python scripts/download_data.py redial     (one dataset: movielens, redial or tmdb)
"""
import io
import sys
import zipfile
from pathlib import Path

import requests

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
MOVIELENS = "https://files.grouplens.org/datasets/movielens/ml-25m.zip"
REDIAL = "https://github.com/ReDialData/website/raw/data/redial_dataset.zip"
TMDB = ("https://huggingface.co/datasets/HenryWaltson/TMDB-IMDB-Movies-Dataset/resolve/main/"
        "TMDB%20%20IMDB%20Movies%20Dataset.csv")


def unzip(url: str, target: Path):
    print(f"Downloading {url}...")
    response = requests.get(url, timeout=900)
    response.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        archive.extractall(target)


def movielens():
    if (RAW_DIR / "ml-25m" / "ratings.csv").exists():
        print("MovieLens already downloaded")
        return
    unzip(MOVIELENS, RAW_DIR)


def redial():
    target = RAW_DIR / "redial"
    if (target / "train_data.jsonl").exists():
        print("ReDial already downloaded")
        return
    unzip(REDIAL, target)


def tmdb():
    target = RAW_DIR / "tmdb" / "tmdb_movies.csv"
    if target.exists():
        print("TMDB already downloaded")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {TMDB}...")
    with requests.get(TMDB, stream=True, timeout=900) as response:
        response.raise_for_status()
        with open(target, "wb") as out:
            for chunk in response.iter_content(chunk_size=1 << 20):
                out.write(chunk)


def main():
    steps = {"movielens": movielens, "redial": redial, "tmdb": tmdb}
    for name in sys.argv[1:] or steps:
        steps[name]()
    print("Done")


if __name__ == "__main__":
    main()

"""Download MovieLens 25M into data/raw/ (owner: Benoît)."""
import io
import zipfile
from pathlib import Path

import requests

URL = "https://files.grouplens.org/datasets/movielens/ml-25m.zip"
RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"


def main():
    target = RAW_DIR / "ml-25m"
    if target.exists():
        print(f"Already downloaded: {target}")
        return
    print(f"Downloading {URL} (about 250 MB)...")
    response = requests.get(URL, timeout=600)
    response.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        archive.extractall(RAW_DIR)
    print(f"Done: {target}")


if __name__ == "__main__":
    main()

"""ReDial: real movie-recommendation dialogues, linked to MovieLens (part 1)

ReDial (Li et al., NeurIPS 2018) has about 11,000 English dialogues between a seeker and a
recommender. After each dialogue, the seeker said for every film mentioned whether it was
suggested, seen and liked. We use them to test the whole chat system on real requests

1. Link each ReDial film to a MovieLens film by normalized title and year
2. One row per (dialogue, film) with the seeker's answers, who mentioned the film first and when
3. Split by seeker (initiatorWorkerId): 80% of seekers for train, 20% for test, so no seeker
   is in both

Run: python -m src.data.redial
Outputs in data/processed/: redial_links.parquet, redial_mentions.parquet, redial_dialogues.parquet
"""
import json
import re
import unicodedata
import zlib
from urllib.parse import unquote

import numpy as np
import pandas as pd

from src.config import PROCESSED, REDIAL_DIR

TEST_SHARE = 0.2
ARTICLES = ("the", "a", "an")
LIKED = {0: -1, 1: 1, 2: 0}  # ReDial answer -> -1 disliked, 1 liked, 0 did not say


NUMBERS = {"one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "six": "6", "seven": "7",
           "eight": "8", "nine": "9", "ten": "10", "ii": "2", "iii": "3"}


def norm_title(title: str) -> str:
    """Compact key: lower case, no accents, no leading article, digits for numbers, no spaces"""
    title = unicodedata.normalize("NFKD", unquote(str(title))).encode("ascii", "ignore").decode().lower()
    title = title.replace("&", " and ")
    match = re.match(r"^(.*), (the|a|an)$", title.strip())  # MovieLens writes "Matrix, The"
    if match:
        title = f"{match.group(2)} {match.group(1)}"
    words = re.sub(r"[^a-z0-9 ]+", "", re.sub(r"[-:/,.!?]+", " ", title)).split()
    if words and words[0] in ARTICLES:
        words = words[1:]
    return "".join(NUMBERS.get(w, w) for w in words)


def title_keys(title: str) -> list[str]:
    """Main title plus the alternate titles MovieLens gives in brackets"""
    alts = re.findall(r"\((.*?)\)", str(title))
    main = re.sub(r"\(.*?\)", " ", str(title))
    return [k for k in dict.fromkeys(norm_title(t) for t in [main, *alts]) if k]


def split_name(name: str):
    match = re.match(r"^(.*?)\s*\((\d{4})\)\s*$", str(name).strip())
    return (match.group(1), int(match.group(2))) if match else (str(name).strip(), None)


def link_movies(movies: pd.DataFrame) -> pd.DataFrame:
    redial = pd.read_csv(REDIAL_DIR / "movies_with_mentions.csv")
    parsed = redial["movieName"].map(split_name)
    redial["key"] = parsed.map(lambda p: norm_title(p[0]))
    redial["year"] = parsed.map(lambda p: p[1])

    keys = [(m, k, y) for m, t, y in zip(movies["movie_id"], movies["clean_title"], movies["year"])
            for k in title_keys(t)]
    ml = pd.DataFrame(keys, columns=["movie_id", "key", "year"])
    by_key_year = ml.dropna(subset=["year"]).drop_duplicates(["key", "year"]).set_index(["key", "year"])["movie_id"]
    films_per_key = ml.groupby("key")["movie_id"].nunique()
    by_key = ml[ml["key"].map(films_per_key) == 1].drop_duplicates("key").set_index("key")["movie_id"]
    by_year = {y: g for y, g in ml.dropna(subset=["year"]).groupby("year")}

    def find(row):
        key, year = row["key"], row["year"]
        if pd.notna(year):
            for y in (year, year - 1, year + 1):  # release year often differs by one
                if (key, float(y)) in by_key_year.index:
                    return by_key_year[(key, float(y))]
            # "Avengers: Infinity War" -> "Avengers: Infinity War - Part I", same year, only one candidate
            same_year = by_year.get(float(year))
            if same_year is not None and len(key) >= 6:
                hits = same_year.loc[same_year["key"].str.startswith(key), "movie_id"].unique()
                if len(hits) == 1:
                    return hits[0]
        return by_key.get(key)  # title found only once in MovieLens

    redial["movie_id"] = redial.apply(find, axis=1)
    return redial.rename(columns={"movieId": "redial_id", "movieName": "redial_name", "nbMentions": "mentions"})[
        ["redial_id", "redial_name", "mentions", "movie_id"]]


def is_test(worker_id: int) -> bool:
    return zlib.crc32(str(worker_id).encode()) % 100 < TEST_SHARE * 100


def read_dialogues(links: dict):
    mentions, dialogues = [], []
    for file in ("train_data.jsonl", "test_data.jsonl"):
        for line in open(REDIAL_DIR / file, encoding="utf8"):
            d = json.loads(line)
            seeker = d["initiatorWorkerId"]
            answers = d.get("initiatorQuestions") or d.get("respondentQuestions") or {}
            first = {}
            for position, msg in enumerate(d["messages"]):
                for rid in re.findall(r"@(\d+)", msg["text"]):
                    first.setdefault(rid, (position, "seeker" if msg["senderWorkerId"] == seeker else "recommender"))
            for rid, (position, who) in first.items():
                a = answers.get(rid) or {}
                mentions.append({
                    "conversation_id": int(d["conversationId"]),
                    "seeker_id": seeker,
                    "redial_id": int(rid),
                    "movie_id": links.get(int(rid)),
                    "mentioned_by": who,
                    "position": position,
                    "suggested": int(a.get("suggested", 0)),
                    "seen": int(a.get("seen", 2)),
                    "liked": LIKED.get(a.get("liked", 2), 0),
                })
            text = []
            for msg in d["messages"]:
                speaker = "seeker" if msg["senderWorkerId"] == seeker else "recommender"
                text.append({"speaker": speaker, "text": msg["text"]})
            dialogues.append({
                "conversation_id": int(d["conversationId"]),
                "seeker_id": seeker,
                "split": "test" if is_test(seeker) else "train",
                "messages": json.dumps(text),
            })
    return pd.DataFrame(mentions), pd.DataFrame(dialogues)


def main():
    movies = pd.read_parquet(PROCESSED / "movies.parquet")
    links = link_movies(movies)
    links.to_parquet(PROCESSED / "redial_links.parquet", index=False)
    mapping = {int(r): int(m) for r, m in zip(links["redial_id"], links["movie_id"]) if pd.notna(m)}

    mentions, dialogues = read_dialogues(mapping)
    mentions["movie_id"] = mentions["movie_id"].astype("Int64")
    mentions = mentions.merge(dialogues[["conversation_id", "split"]], on="conversation_id")
    mentions.to_parquet(PROCESSED / "redial_mentions.parquet", index=False)
    dialogues.to_parquet(PROCESSED / "redial_dialogues.parquet", index=False)

    # redial_id column in movies.parquet (the most mentioned ReDial film when several match)
    best = links.dropna(subset=["movie_id"]).sort_values("mentions", ascending=False).drop_duplicates("movie_id")
    movies = movies.drop(columns=["redial_id"], errors="ignore")
    movies["redial_id"] = movies["movie_id"].map(best.set_index("movie_id")["redial_id"]).astype("Int64")
    movies.to_parquet(PROCESSED / "movies.parquet", index=False)

    linked = links["movie_id"].notna()
    weighted = links.loc[linked, "mentions"].sum() / links["mentions"].sum()
    print(f"ReDial films linked to MovieLens: {linked.sum():,} / {len(links):,} "
          f"({100 * weighted:.1f}% of mentions)")
    print(f"dialogues: {len(dialogues):,} ({(dialogues['split'] == 'test').sum():,} test), "
          f"seekers: {dialogues['seeker_id'].nunique():,}")
    print(f"mentions: {len(mentions):,}, mapped: {mentions['movie_id'].notna().mean():.1%}")


if __name__ == "__main__":
    main()

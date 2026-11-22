"""Paths and shared settings"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
ML_DIR = RAW / "ml-25m"
REDIAL_DIR = RAW / "redial"
PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"
RESULTS = ROOT / "results"

for folder in (PROCESSED, MODELS, RESULTS):
    folder.mkdir(parents=True, exist_ok=True)

SEED = 42
LIKE_THRESHOLD = 4.0      # a rating of 4 stars or more is a like
N_USERS = 15_000          # MovieLens users kept, so training fits on a laptop CPU
MIN_LIKES = 10            # k-core: every user and film keeps at least 10 likes
POPULAR_SHARE = 0.20      # top 20% films by train likes are "popular", the rest are hidden gems
TEXT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# CinéRadar

CinéRadar is a film friend for cinephiles. You chat with it, it learns your taste, and it suggests films you would like but would probably not find alone

Under the hood, a graph model (NLGCL, Xu et al., RecSys 2025) learns taste from 25 million MovieLens ratings. A chat layer turns your message into a request, and a re-ranking step pushes well-rated films that few people have seen. A slider lets you move from mainstream picks to hidden gems

RecSys Startup Sprint, Group 18: Théodore BRICOUT, Benoît QUARANTA, Michaël LOPEZ, Rayan DAOUDI

## Run the app in 10 minutes

You need Python 3.10 or newer and an internet connection for the install. You do not need the raw data, the trained model or an API key: the film catalog and the film vectors are committed in `artifacts/`

**Shortest way**: get the code (`git clone` or unzip the ZIP), then double-click `start.bat` on Windows, or run `bash start.sh` on Mac or Linux. On the first run, the script creates a Python environment in `.venv` and installs the requirements into it (about 2 minutes), then starts everything and opens the app. The steps below do the same by hand

1. Get the code and install the dependencies (1 to 3 minutes). If you start from the ZIP, unzip it and start from `cd cineradar`

   Windows, in PowerShell or cmd
   ```
   git clone https://github.com/Fromagesmika/cineradar.git
   cd cineradar
   python -m venv .venv
   .venv\Scripts\python -m pip install -r requirements.txt
   ```
   Mac or Linux
   ```bash
   git clone https://github.com/Fromagesmika/cineradar.git
   cd cineradar
   python3 -m venv .venv
   .venv/bin/python -m pip install -r requirements.txt
   ```
2. Start CinéRadar, from the same folder

   Windows: `start.bat` (or double-click `start.bat` in the file explorer)

   Mac or Linux: `bash start.sh`

   The script starts the API, waits about 20 seconds while it loads the 62,000 films, starts the web app and opens http://127.0.0.1:8501 in your browser. To stop, close the two windows it opened (on Mac and Linux, press Ctrl+C)
3. Try it
   - Pick a nickname and press Start
   - Pick at least 5 films you loved (the search box finds any film), and if you like, the genres you love or avoid and a few films you disliked. Then press Build my profile
   - On the Chat page, click one of the example requests or write your own, in French or in English, for example "un film comme Heat, mais moins connu" or "something like Blade Runner but more recent"
   - Under each film, the Liked, Disliked, Seen and Not for me buttons change what comes next. The slider in the sidebar moves from mainstream picks to hidden gems. The For you page recommends films without a request, and My profile shows what the app has learned about you

To start the two parts by hand instead, activate the environment and run `uvicorn api.main:app` in one terminal and `streamlit run app/streamlit_app.py` in a second one, then open http://127.0.0.1:8501. The API documentation is at http://127.0.0.1:8000/docs. On Windows, use 127.0.0.1 rather than localhost: localhost first tries IPv6, and every call then waits 2 seconds

If the API window shows an error about port 8000 or 8501, another program already uses that port: close it and start again. If `python` is not found on Windows, use `py -3` instead

No API key is needed. Without one, the chat reads messages with rules and answers with sentence templates. With an `ANTHROPIC_API_KEY` (or an `OPENAI_API_KEY`, see `src/llm/client.py`) set before starting the API, a language model reads the messages and writes the answers. It may only talk about the films the ranking gave it, and the code falls back to the rules if it names another film or if the call fails

## How it works

A chat turn goes through four parts

1. Data (`src/data/`). MovieLens 25M gives the ratings. TMDB gives a summary, a poster, the director and the cast for 61,351 of the 62,423 films. ReDial gives 11,348 real English dialogues between a person looking for a film and a person recommending one, and 94% of the film mentions in them are linked to MovieLens films
2. Taste model (`src/model/`). NLGCL learns one vector per user and per film from the likes (4 stars or more) of 15,000 MovieLens users. Every film also gets a vector learned from its text (title, genres, tags, summary), which is the only signal for films outside the graph. A new user needs no retraining: their vector is a weighted mean of the films they liked, minus half of the films they disliked
3. Chat brain (`src/llm/`, `src/reco/`). The message becomes a request: genres, keywords, reference films ("like Heat"), period, films you loved or hated. Candidates are picked for the request, then ranked by request match, taste vector and user memory. The hidden-gem step then moves films up or down according to the slider and the average MovieLens rating, so a hidden gem is a film that is both little known and well rated. The reply explains each pick ("because you liked X")
4. Database, API and app (`src/service/`, `api/`, `app/`). Every answer to the questionnaire, every film named in the chat and every click (liked, disliked, seen, not for me) is stored in SQLite. The next recommendation reads this memory, so a click changes what comes next

### Database

| Table | Content |
|---|---|
| `users` | nickname, slider value, questionnaire done |
| `movies` | film catalog, popular or hidden gem |
| `interactions` | one row per opinion (`liked`, `disliked`, `seen`, `suggested`, `not_interested`, `rating`) with its source (`questionnaire`, `chat`, `click`) |
| `profile_facts` | genres loved or avoided, facts learned in the chat |
| `conversations`, `messages` | chat history, with the request read from each message and the films shown |
| `user_embeddings` | current taste vector of each user |

The database is created on first start in `data/cineradar.db`. The `CINERADAR_DB` environment variable changes its path

### API

| Endpoint | Use |
|---|---|
| `POST /users`, `POST /onboarding` | create a user, save the questionnaire |
| `POST /chat` | one message to the film friend: reply and film cards |
| `GET /recommend?user_id=&k=&alpha=&genre=` | recommendations without chat (`alpha` is the slider) |
| `POST /log_interaction` | save a click |
| `GET /profile/{id}`, `PATCH /profile/{id}` | read the profile, change the slider |
| `GET /explain?user_id=&movie_id=` | liked film closest to a film |
| `GET /movies/search?q=`, `GET /movies/{id}`, `GET /genres` | catalog |

## Results

### Taste model on MovieLens

Test set: the last 10% of each user's likes in time order, 14,969 users, the 6,930 films of the graph ranked for each user. Hidden gems are the films outside the 20% most liked

| Method | NDCG@10 | Recall@20 | GemRecall@20 | Coverage@10 |
|---|---|---|---|---|
| Random | 0.0013 | 0.0026 | 0.0027 | 1.0000 |
| Most popular | 0.0375 | 0.0796 | 0.0000 | 0.0136 |
| Text vectors only | 0.0094 | 0.0217 | 0.0179 | 0.2179 |
| LightGCN | 0.0549 | 0.1204 | 0.0058 | 0.2991 |
| NLGCL | 0.0563 | 0.1216 | 0.0030 | 0.2170 |
| NLGCL + text, user vector without retraining (used in the app) | 0.0563 | 0.1199 | 0.0052 | 0.2649 |

NLGCL beats the popularity baseline by 50% on NDCG@10, and the vector the app computes for a new user does as well as the trained one. On its own, the model rarely recommends hidden gems, so the system adds a re-ranking step. Full table: `results/taste_results.md`, learning curves: `results/curves.png`

### Whole system on ReDial dialogues

300 dialogues from 75 test seekers who are not used anywhere else. The system reads what the seeker wrote before the first suggestion, together with the films they said they liked, and must find the films the recommender suggested (950 films, about 3 per dialogue, among 18,430 candidates). Jaccard@10 measures how much the top 10s of two different seekers overlap when both ask for "an action film": lower means more personal

| System | Recall@10 | Recall@20 | Recall@50 | NDCG@10 | GemShare@10 | Jaccard@10 |
|---|---|---|---|---|---|---|
| Random | 0.0000 | 0.0000 | 0.0029 | 0.0000 | 0.3550 | 0.0000 |
| Most popular | 0.0135 | 0.0218 | 0.0574 | 0.0113 | 0.0000 | 1.0000 |
| Taste model alone | 0.0341 | 0.0597 | 0.1181 | 0.0280 | 0.0910 | 0.0683 |
| Request only, no taste | 0.0361 | 0.0597 | 0.1070 | 0.0256 | 0.0280 | 1.0000 |
| + taste vectors | 0.0350 | 0.0594 | 0.1132 | 0.0244 | 0.1703 | 0.1322 |
| + film descriptions | 0.0448 | 0.0842 | 0.1483 | 0.0339 | 0.1393 | 0.1322 |
| + memory | 0.0448 | 0.0842 | 0.1483 | 0.0339 | 0.1393 | 0.1322 |
| + hidden-gem re-ranking (full system, slider in the middle) | 0.0486 | 0.0898 | 0.1488 | 0.0373 | 0.0883 | 0.1671 |
| Full system, slider on hidden gems (0.9) | 0.0252 | 0.0399 | 0.0808 | 0.0171 | 0.5947 | 0.1595 |

The full system scores three times the popularity baseline on NDCG@10 (0.037 against 0.011), and two users asking the same thing share about one film in six. The memory row equals the row above because the test seekers have no click history; the tests check its effect instead (`test_click_changes_next_recommendations`). With the slider on hidden gems, 59% of the top 10 are hidden gems and NDCG@10 (0.017) stays above the popularity baseline: the user chooses how much accuracy to trade for discovery. All metrics are in `results/redial_results.md`

## Rebuild the data and retrain (optional)

This part needs PyTorch and sentence-transformers, which the app itself does not use: `pip install -r requirements-train.txt`

This takes about one hour on a laptop CPU and about 1.5 GB of downloads

```bash
python scripts/download_data.py                                      # MovieLens 25M, ReDial, TMDB into data/raw/
python -m src.data.preprocess                                        # likes of 15,000 users, film table
python -m src.data.tmdb                                              # summaries, posters, rating counts
python -m src.data.redial                                            # link ReDial films to MovieLens, split by seeker
python -m src.data.split                                             # train / validation / test split, graph
python -m src.model.content                                          # text vector of every film
python -m src.train.train --name lightgcn --cl-weight 0 --lr 5e-3    # LightGCN (backup model)
python -m src.train.train --name nlgcl --cl-weight 0.1 --lr 5e-3     # NLGCL
python -m src.model.export --name nlgcl                              # artifacts/item_emb.npy
python -m scripts.export_catalog                                     # artifacts/catalog.csv.gz
python -m src.model.evaluate_taste                                   # MovieLens results
python -m src.eval.evaluate                                          # ReDial results and ablations
```

One epoch takes about 15 s for LightGCN and 1 to 2 minutes for NLGCL. To try the training code without the data, train on a small synthetic graph: `python -m src.train.train --toy --name toy --batch 512 --lr 5e-3`

## Use the parts from Python

```python
from src.model import taste                        # part 2
vec = taste.user_vector(liked_ids=[318, 858], disliked_ids=[2571])   # MovieLens movie_id
scores = taste.score(vec, [1, 2, 3])               # higher is better

from src.reco.engine import recommend              # part 3
recommend(user_id, {"genres": ["Crime"], "liked_ids": [858]}, k=10, cinephile=0.8)
# -> [{"movie_id", "score", "because_movie_id"}, ...]

from src.eval.evaluate import evaluate             # part 1
table = evaluate(recommend)                        # any function with the contract above
```

More detail on the chat brain (request format, memory, score formula) is in `README_partie3.md`, in French

## Project structure

```
cineradar/
├── api/            FastAPI backend
├── app/            Streamlit web app (theme in .streamlit/config.toml)
├── artifacts/      film catalog and film vectors used by the app (committed)
├── data/           raw and processed data, SQLite database (not in git)
├── models/         trained models (not in git)
├── results/        evaluation tables and learning curves
├── scripts/        downloads, catalog export, terminal chat demo
├── src/
│   ├── data/       MovieLens cleaning, TMDB, ReDial, split and graph
│   ├── eval/       metrics and the evaluation harness
│   ├── model/      NLGCL, text vectors, export, taste.py
│   ├── train/      training loop
│   ├── llm/        message reading and reply (rules or language model)
│   ├── reco/       candidates, ranking, hidden gems, user memory
│   └── service/    database, profile, catalog, chat turn
├── tests/          55 tests
├── start.bat       installs on the first run, then starts the API and the app (Windows)
└── start.sh        the same for Mac and Linux
```

## Tests

```bash
pytest
```

The tests take about 2 minutes. They check the model code on a synthetic graph, the chat brain on a small catalog of 79 films, and the database and API on the real catalog, including that a click changes the next recommendations

With only `requirements.txt` installed, the model tests (tests/test_taste.py) are skipped because they need PyTorch. Install `requirements-train.txt` to run all 55

## Limits

- Without an API key, the rules understand short requests well but miss complex sentences, and French titles are not recognised ("Le Parrain" must be written "The Godfather")
- Hidden gems come from MovieLens users, who mostly rate English-language films released before 2020
- The ReDial scores are low in absolute terms because each recommender suggested about 3 films among 18,430 possible ones

## Data sources

- MovieLens 25M, GroupLens Research (F. M. Harper and J. A. Konstan, 2015)
- ReDial, Li et al., "Towards Deep Conversational Recommendations", NeurIPS 2018
- TMDB metadata from a public dump on Hugging Face (`HenryWaltson/TMDB-IMDB-Movies-Dataset`). This product uses TMDB data but is not endorsed or certified by TMDB
- NLGCL, Xu et al., RecSys 2025

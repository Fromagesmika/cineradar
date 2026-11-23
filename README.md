# CinéRadar

Hidden-gem film recommendation for cinephiles.
CinéRadar recommends films that match your taste **and** that you would probably not find alone.
It uses the NLGCL model (Xu et al., RecSys 2025) and adds a "hidden-gem" re-ranking step.

RecSys Startup Sprint, Group 18.

| Member | Role |
|---|---|
| Théodore BRICOUT | Project Lead: data split, graph, evaluation |
| Benoît QUARANTA | Data Engineer + Systems Engineer: data, database, API, web app |
| Michaël LOPEZ | Lead ML Engineer: NLGCL model |
| Rayan DAOUDI | ML Engineer: training, `predict()`, re-ranking |

## Project structure

```
cineradar/
├── api/            FastAPI backend (/chat, /recommend, /onboarding, /log_interaction, ...)
├── app/            Streamlit web app (questionnaire, chat with the film friend, profile)
├── data/
│   ├── raw/        downloaded datasets (not in git)
│   └── processed/  cleaned data (not in git)
├── models/         trained model files (not in git)
├── scripts/        download and utility scripts
├── src/
│   ├── data/       cleaning + train/test split + graph
│   ├── eval/       metrics (NDCG@10, Recall@20) and baselines
│   ├── model/      NLGCL model
│   ├── service/    SQLite database, user profile, film catalog, chat brain
│   └── train/      training loop and predict()
└── tests/          unit tests
```

## Run the project locally

1. Clone the repository
   ```bash
   git clone <repository-url>
   cd cineradar
   ```
2. Create a virtual environment and install the dependencies
   ```bash
   python -m venv .venv
   # Windows: .venv\Scripts\activate    |    Mac/Linux: source .venv/bin/activate
   pip install -r requirements.txt
   ```
3. Download the data (MovieLens 25M, about 250 MB)
   ```bash
   python scripts/download_data.py
   ```
4. Start the API
   ```bash
   uvicorn api.main:app --reload
   ```
   Open http://localhost:8000/docs to test the endpoints.
5. Start the web app (in a second terminal)
   ```bash
   streamlit run app/streamlit_app.py
   ```
   Open http://localhost:8501, pick a nickname, answer the questionnaire and start chatting

The app runs without the data: the taste vectors are committed in `artifacts/`, and the database
(`data/cineradar.db`) is created on first start

## Re-train the model (optional)

```bash
python -m src.data.preprocess                                        # clean the data
python -m src.data.split                                             # train / validation / test split
python -m src.model.content                                          # text vector of every film
python -m src.train.train --name lightgcn --cl-weight 0 --lr 5e-3    # LightGCN (backup model)
python -m src.train.train --name nlgcl --cl-weight 0.1 --lr 5e-3     # NLGCL
python -m src.model.export --name nlgcl                              # writes artifacts/item_emb.npy
python -m src.model.evaluate_taste                                   # test results vs baselines
python -m src.model.plot_curves lightgcn nlgcl                       # results/curves.png
```

On a laptop CPU, one epoch takes about 15 s for LightGCN and about 1 to 2 min for NLGCL

To try the model without the data, train on a small synthetic graph (1,000 users x 500 films):

```bash
python -m src.train.train --toy --name toy --batch 512 --lr 5e-3
```

## Taste model (part 2)

`src/model/taste.py` is the only file the other parts need. It reads `artifacts/item_emb.npy`, which is committed, so nobody has to retrain

```python
from src.model import taste

vec = taste.user_vector(liked_ids=[318, 858], disliked_ids=[2571])  # MovieLens movie_id
scores = taste.score(vec, [1, 2, 3])          # higher is better, unknown films get -inf
reasons = taste.because([318, 858], [1, 2, 3]) # liked film closest to each film
```

Every film of MovieLens has a vector. Films with many likes use the vector learned by the graph model, and films with few or no likes lean on a vector computed from their title, genres and tags (see `src/model/export.py`). A new user does not need retraining: their vector is the average of the films they liked. If `artifacts/` is missing, `taste.py` switches to random vectors so the app still runs

## Database, API and web app (part 4)

### How the profile evolves

1. **Questionnaire** (first visit): films you loved and hated, genres you love and avoid,
   mainstream ↔ hidden-gem slider
2. **Every chat message** is read by the brain: films you mention as loved or hated, genres,
   the film you want tonight. Lasting facts go to the database, the request only drives this answer
3. **Every click** on a film (👍 liked, 👎 disliked, 👁️ seen, 🚫 not interested) goes to `/log_interaction`
4. After each new opinion, the user's taste vector is recomputed with `taste.user_vector()`
   (no retraining) and stored in `user_embeddings`

So two users asking for "an action film" get different lists: the candidates match the request,
the order comes from each user's taste vector, then the hidden-gem re-ranking applies their slider

### Database (`src/service/db.py`, SQLite)

| Table | Content |
|---|---|
| `users` | nickname, cinephile level (slider), onboarding done |
| `movies` | film catalog, popular or hidden gem |
| `interactions` | one row per opinion: `liked`, `disliked`, `seen`, `suggested`, `not_interested`, `rating`, with its source (`questionnaire`, `chat`, `click`, `redial`, `movielens`) |
| `profile_facts` | genres loved or avoided and facts learned in the chat |
| `conversations`, `messages` | chat history, with what was extracted and what was recommended |
| `user_embeddings` | current taste vector of each user |

The `liked` / `seen` / `suggested` labels are the same as the ReDial questionnaire, so ReDial
dialogues can be loaded into the same table (`source = 'redial'`)

The database path can be changed with the `CINERADAR_DB` environment variable

### API

| Endpoint | Use |
|---|---|
| `GET /health` | status, catalog source, mock mode |
| `GET /genres`, `GET /movies/search?q=`, `GET /movies/popular`, `GET /movies/{id}` | catalog |
| `POST /users` | create a user |
| `POST /onboarding` | save the questionnaire |
| `GET /profile/{id}`, `PATCH /profile/{id}` | read the profile, change the slider |
| `POST /chat` | one message to the film friend: reply + recommendations |
| `GET /recommend?user_id=&k=&alpha=&genre=` | recommendations without chat |
| `GET /explain?user_id=&movie_id=` | "because you liked X" |
| `POST /log_interaction` | save a click |

### Where part 3 plugs in

`src/service/brain.py` uses part 3 automatically when these functions exist, and a simple
built-in version (keywords + taste model + hidden-gem re-ranking, no API key needed) otherwise:

```python
src/llm/agent.py    extract(message, history) -> dict      reply(message, profile, recos) -> str
src/reco/engine.py  recommend(user_id, intent, k, cinephile) -> list[{"movie_id", "score", "because_movie_id"}]
```

`extract` returns `{"intent": {"genres", "avoid_genres", "reference_ids", "min_year", "max_year",
"cinephile_shift"}, "likes", "dislikes", "genres_liked", "genres_avoided", "facts"}`.
The profile of a user is `src.service.profile.get_profile(db.connect(), user_id)`

### Film catalog

Without the data, the app uses 86 well-known films (`src/service/seed_movies.csv`).
To use the full MovieLens catalog (62,000 films, with TMDB summaries and posters if
`data/processed/tmdb.parquet` exists), run once after preprocessing and commit the file:

```bash
python -m scripts.export_catalog      # writes artifacts/catalog.csv.gz
```

## Run the tests

```bash
pytest
```

## Status

Work in progress. See the GitHub issues for the task list.

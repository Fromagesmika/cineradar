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
├── api/            FastAPI backend (/recommend, /log_interaction)
├── app/            Streamlit web interface
├── data/
│   ├── raw/        downloaded datasets (not in git)
│   └── processed/  cleaned data (not in git)
├── models/         trained model files (not in git)
├── scripts/        download and utility scripts
├── src/
│   ├── data/       cleaning + train/test split + graph
│   ├── eval/       metrics (NDCG@10, Recall@20) and baselines
│   ├── model/      NLGCL model
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

## Run the tests

```bash
pytest
```

## Status

Work in progress. See the GitHub issues for the task list.

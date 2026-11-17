# CinéRadar 🎬

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
python -m src.data.preprocess      # clean the data
python -m src.data.split           # train / validation / test split
python -m src.train.train          # train NLGCL, saves models/nlgcl.pt
```

## Run the tests

```bash
pytest
```

## Status

Work in progress. See the GitHub issues for the task list.

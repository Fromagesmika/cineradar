"""Tests de la partie 3. Lancer depuis la racine : `python -m pytest -q`."""
import os

os.environ["LLM_OFFLINE"] = "1"
os.environ["TASTE_BACKEND"] = "mock"

import numpy as np
import pytest

from src.llm import agent, client
from src.reco import engine
from src.reco.catalog import get_catalog
from src.reco.memory import InMemoryProvider

cat = get_catalog()
T = cat.find_title


@pytest.fixture
def mem():
    p = InMemoryProvider()
    engine.set_memory_provider(p)
    # Profil A : SF grand public. Profil B : polar d'auteur / cinéma asiatique.
    p.onboarding(1, [T(x) for x in ["The Matrix", "Inception", "Interstellar", "Avengers: Endgame"]], cinephile=0.2)
    p.onboarding(2, [T(x) for x in ["Le Samouraï", "Oldboy", "Heat", "Memories of Murder"]], cinephile=0.9)
    yield p
    engine.set_memory_provider(engine.DEMO_MEMORY)


def titles(recos):
    return [cat.get(r["movie_id"])["title"] for r in recos]


def test_contract_shape(mem):
    recos = engine.recommend(1, {"genres": ["Action"]}, k=10, cinephile=0.5)
    assert len(recos) == 10
    for r in recos:
        assert set(r) == {"movie_id", "score", "because_movie_id"}
        assert isinstance(r["movie_id"], int)


def test_critere_termine_deux_profils_action(mem):
    """« Deux profils différents qui demandent un film d'action obtiennent deux listes différentes et pertinentes. »"""
    intent = agent.extract("Je veux un film d'action", [])
    a = engine.recommend(1, intent, k=10, cinephile=0.2)
    b = engine.recommend(2, intent, k=10, cinephile=0.9)
    ia, ib = {r["movie_id"] for r in a}, {r["movie_id"] for r in b}
    assert len(ia & ib) / len(ia | ib) < 0.3  # listes différentes
    for r in a + b:  # pertinentes : ce sont des films d'action
        assert "Action" in cat.get(r["movie_id"])["genres"]
    # et cohérentes avec chaque profil
    assert np.mean([cat.get(r["movie_id"])["popularity"] for r in a]) > np.mean(
        [cat.get(r["movie_id"])["popularity"] for r in b])
    assert sum("Sci-Fi" in cat.get(r["movie_id"])["genres"] for r in a) > sum(
        "Sci-Fi" in cat.get(r["movie_id"])["genres"] for r in b)


def test_un_clic_change_la_reco_suivante(mem):
    before = engine.recommend(1, {"genres": ["Action"]}, k=10)
    first = before[0]["movie_id"]
    mem.log(1, first, "not_interested")
    after = engine.recommend(1, {"genres": ["Action"]}, k=10)
    assert first not in [r["movie_id"] for r in after]
    assert [r["movie_id"] for r in before] != [r["movie_id"] for r in after]


def test_seen_and_liked_are_not_recommended(mem):
    mem.log(1, T("Die Hard"), "seen")
    ids = [r["movie_id"] for r in engine.recommend(1, {"genres": ["Action"]}, k=20)]
    assert T("Die Hard") not in ids
    assert T("The Matrix") not in ids  # déjà aimé


def test_genre_dislike_fact_persists(mem):
    mem.add_fact(1, "genre_dislike", "Horror")
    ids = [r["movie_id"] for r in engine.recommend(1, {}, k=30)]
    assert all("Horror" not in cat.get(m)["genres"] for m in ids)


def test_family_excludes_horror(mem):
    intent = agent.extract("un film pour regarder avec mes enfants", [])
    assert "Horror" in intent["excluded_genres"]
    assert {"key": "watch_with", "value": "famille"} in intent["profile_facts"]


def test_cinephile_slider(mem):
    lo = engine.recommend(1, {"genres": ["Action"]}, k=10, cinephile=0.0)
    hi = engine.recommend(1, {"genres": ["Action"]}, k=10, cinephile=1.0)
    pop = lambda rs: np.mean([cat.get(r["movie_id"])["popularity"] for r in rs])
    assert pop(hi) < pop(lo)


def test_because_points_to_liked_movie(mem):
    recos = engine.recommend(1, {"genres": ["Action"]}, k=10)
    liked = set(mem.get(1).liked)
    assert any(r["because_movie_id"] for r in recos)
    assert all(r["because_movie_id"] in liked for r in recos if r["because_movie_id"])


def test_similar_to_reference(mem):
    intent = agent.extract("un film comme Heat", [])
    assert intent["reference_ids"] == [T("Heat")]
    ids = [r["movie_id"] for r in engine.recommend(1, intent, k=5)]
    assert T("Heat") not in ids
    assert any("Crime" in cat.get(m)["genres"] for m in ids)


def test_refine_excludes_previous(mem):
    r1 = agent.chat_turn(1, "un film d'action", [])
    h = [{"role": "user", "content": "un film d'action"},
         {"role": "assistant", "content": r1["reply"], "intent": r1["intent"],
          "recos": [r["movie_id"] for r in r1["recos"]]}]
    r2 = agent.chat_turn(1, "un autre", h)
    assert r2["intent"]["intent"] == "refine"
    assert not ({r["movie_id"] for r in r1["recos"]} & {r["movie_id"] for r in r2["recos"]})
    assert all("Action" in cat.get(r["movie_id"])["genres"] for r in r2["recos"])


def test_offline_reply_only_mentions_recos(mem):
    recos = engine.recommend(1, {"genres": ["Action"]}, k=10)
    text = agent.reply("un film d'action", mem.get(1).to_profile(), recos)
    allowed = {r["movie_id"] for r in recos} | {r["because_movie_id"] for r in recos}
    assert set(cat.titles_in_text(text)) <= allowed
    assert set(cat.titles_in_text(text)) & {r["movie_id"] for r in recos}


def test_chat_turn_returns_memory_updates(mem):
    out = agent.chat_turn(1, "J'ai adoré Heat mais j'ai détesté Titanic, un polar ?", [])
    types = {(i["movie_id"], i["type"]) for i in out["memory_updates"]["interactions"]}
    assert (T("Heat"), "chat_like") in types and (T("Titanic"), "chat_dislike") in types
    assert sum(t == "shown" for _, t in types) == len(out["recos"])
    assert T("Heat") in mem.get(1).liked  # la mémoire de démo est mise à jour


# --------------------------------------------------------------- chemin LLM (client simulé)


def test_llm_extract_path(monkeypatch, mem):
    monkeypatch.setattr(client, "is_available", lambda: True)
    monkeypatch.setattr(client, "complete", lambda *a, **k: '```json\n{"intent":"recommend","genres":["science fiction","action"],'
                        '"liked_titles":["inception"],"cinephile_shift":0.35}\n```')
    d = agent.extract("bla", [])
    assert d["source"] == "llm"
    assert d["genres"] == ["Sci-Fi", "Action"]
    assert d["liked_ids"] == [T("Inception")]


def test_llm_reply_hallucination_falls_back(monkeypatch, mem):
    recos = engine.recommend(1, {"genres": ["Action"]}, k=5)
    monkeypatch.setattr(client, "is_available", lambda: True)
    monkeypatch.setattr(client, "complete", lambda *a, **k: "Regarde Titanic, c'est top !")  # hors liste
    text = agent.reply("action", mem.get(1).to_profile(), recos)
    assert "Titanic" not in text
    assert set(cat.titles_in_text(text)) & {r["movie_id"] for r in recos}


def test_llm_reply_accepted_when_valid(monkeypatch, mem):
    recos = engine.recommend(1, {"genres": ["Action"]}, k=5)
    good = f"Fonce sur {cat.get(recos[0]['movie_id'])['title']}, tu vas adorer."
    monkeypatch.setattr(client, "is_available", lambda: True)
    monkeypatch.setattr(client, "complete", lambda *a, **k: good)
    assert agent.reply("action", mem.get(1).to_profile(), recos) == good

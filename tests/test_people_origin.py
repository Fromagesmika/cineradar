"""Réalisateurs, acteurs et origine des films, sur le vrai catalogue (« un film de Nolan », « un film coréen »)."""
import os

os.environ.setdefault("LLM_OFFLINE", "1")

import pandas as pd
import pytest

from src.llm.rules import extract_rules
from src.reco import engine
from src.reco.catalog import get_catalog
from src.reco.memory import InMemoryProvider
from src.reco.origin import infer_origins, origins_in

ASIA = {"ja", "ko", "zh", "hi", "th"}


@pytest.fixture
def mem():
    p = InMemoryProvider()
    engine.set_memory_provider(p)
    p.onboarding(1, [1214, 541, 924, 1232, 2571], cinephile=0.5)  # Alien, Blade Runner, 2001, Stalker, Matrix
    yield p
    engine.set_memory_provider(engine.DEMO_MEMORY)


def films(intent, k=8):
    cat = get_catalog()
    return [cat.get(r["movie_id"]) for r in engine.recommend(1, intent, k=k)]


def test_origin_words_in_messages():
    assert set(origins_in("un film asiatique")) == ASIA
    assert origins_in("un film coreen") == ["ko"]
    assert origins_in("something japanese") == ["ja"]
    assert origins_in("reponds en francais") == []  # la langue de la réponse, pas l'origine du film
    assert origins_in("un film francais") == ["fr"]


def test_origin_is_inferred_from_the_catalog():
    cat = get_catalog()
    expected = {"Spirited Away": "ja", "Seven Samurai": "ja", "Memories of Murder": "ko", "Train to Busan": "ko",
                "Crouching Tiger, Hidden Dragon": "zh", "In the Mood For Love": "zh", "3 Idiots": "hi",
                "Amelie": "fr", "Life Is Beautiful": "it", "Lost in Translation": "en", "Kill Bill: Vol. 2": "en"}
    for title, origin in expected.items():
        row = cat.df[cat.df["title"].str.startswith(title)].iloc[0]
        assert row["origin"] == origin, title


def test_original_language_column_wins_when_present():
    df = pd.DataFrame({"title": ["A", "B"], "tags": [[], []], "directors": ["", ""], "cast": ["", ""],
                       "original_language": ["ko", "en"]})
    assert infer_origins(df) == ["ko", "en"]


def test_asian_request_returns_asian_films_from_several_countries(mem):
    found = films(extract_rules("un film de cinéma asiatique"))
    assert len(found) == 8 and all(f["origin"] in ASIA for f in found)
    assert len({f["origin"] for f in found}) >= 3


def test_country_and_genre_together(mem):
    found = films(extract_rules("un thriller coréen"))
    assert all(f["origin"] == "ko" and "Thriller" in f["genres"] for f in found)


def test_director_request(mem):
    intent = extract_rules("un film de Christopher Nolan")
    assert intent["directors"] == ["Christopher Nolan"]
    assert all("Christopher Nolan" in f["directors"] for f in films(intent, 5))
    assert extract_rules("un film de nolan")["directors"] == ["Christopher Nolan"]
    assert extract_rules("un film de Miyazaki")["directors"] == ["Hayao Miyazaki"]


def test_actor_request(mem):
    intent = extract_rules("un truc avec Tom Hanks")
    assert intent["actors"] == ["Tom Hanks"] and not intent["directors"]
    assert all("Tom Hanks" in f["cast"] for f in films(intent, 5))
    # « un film de Tom Hanks » : il a surtout joué, c'est l'acteur qu'on veut
    assert extract_rules("un film de Tom Hanks")["actors"] == ["Tom Hanks"]


def test_director_and_actor_together(mem):
    found = films(extract_rules("un film de Scorsese avec De Niro"), 5)
    assert all("Martin Scorsese" in f["directors"] and "Robert De Niro" in f["cast"] for f in found)


def test_ordinary_words_are_not_people():
    for message in ("un film de guerre", "un film d'action ce soir", "Donne moi une comédie", "un film de noel"):
        intent = extract_rules(message)
        assert not intent["directors"] and not intent["actors"], message


def test_reply_names_the_person():
    from src.llm import agent
    turn = agent.chat_turn(1, "un film de Christopher Nolan")
    assert "Nolan" in turn["reply"]


def test_reply_language_follows_the_message():
    from src.llm import agent
    assert agent.detect_language("something like Blade Runner but more recent") == "en"
    assert agent.detect_language("un film comme Heat") == "fr"
    assert agent.detect_language("un thriller coréen") == "fr"
    assert agent.detect_language("Ghibli movies please") == "en"
    en = agent.chat_turn(1, "a Korean thriller please")["reply"]
    fr = agent.chat_turn(1, "un thriller coréen stp")["reply"]
    assert any(w in en for w in ("because", "since", "pick", "ideas", "thing", "Good call", "know exactly"))
    assert "parce que" in fr or "idées" in fr or "veine" in fr or "propose" in fr or "Bonne pioche" in fr


def test_english_requests_are_understood():
    intent = extract_rules("a funny movie from the 90s, nothing scary")
    assert intent["genres"] == ["Comedy"] and intent["excluded_genres"] == ["Horror"]
    assert (intent["year_min"], intent["year_max"]) == (1990, 1999)
    assert extract_rules("Ghibli movies please")["keywords"] == ["studio ghibli"]

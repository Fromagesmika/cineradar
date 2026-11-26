import pytest

from src.service import brain, catalog, db, profile

SCIFI_FAN = [1214, 541, 924, 1232, 2571]
CRIME_FAN = [858, 1221, 1213, 296, 1089]


@pytest.fixture
def conn(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "test.db")
    connection = db.connect()
    yield connection
    connection.close()


def test_schema_and_movies_are_created(conn):
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
    assert {"users", "movies", "interactions", "profile_facts", "conversations", "messages", "user_embeddings"} <= tables
    assert conn.execute("SELECT COUNT(*) FROM movies").fetchone()[0] == len(catalog.load())


def test_onboarding_fills_profile_and_embedding(conn):
    user_id = profile.create_user(conn, "alice")
    profile.save_onboarding(conn, user_id, SCIFI_FAN, [858], ["Sci-Fi"], ["Horror"], 0.7)
    found = profile.get_profile(conn, user_id)
    assert found["onboarding_done"]
    assert set(found["liked_ids"]) == set(SCIFI_FAN)
    assert found["disliked_ids"] == [858]
    assert found["genres_liked"] == ["Sci-Fi"] and found["genres_avoided"] == ["Horror"]
    assert found["cinephile_level"] == pytest.approx(0.7)
    assert profile.get_embedding(conn, user_id) is not None


def test_latest_opinion_wins(conn):
    user_id = profile.create_user(conn)
    profile.log_interaction(conn, user_id, 2571, "like")
    profile.log_interaction(conn, user_id, 2571, "dislike")
    found = profile.get_profile(conn, user_id)
    assert found["liked_ids"] == [] and found["disliked_ids"] == [2571]


def test_unknown_type_is_rejected():
    with pytest.raises(ValueError):
        profile.normalize_type("love")


def test_extract_reads_likes_dislikes_and_intent():
    out = brain.fallback_extract("J'ai adoré Alien mais j'ai détesté The Matrix. Un film d'action ce soir, pas d'horreur")
    assert out["likes"] == [1214]
    assert out["dislikes"] == [2571]
    assert out["intent"]["genres"] == ["Action"]
    assert out["intent"]["avoid_genres"] == ["Horror"]


def test_extract_reference_film_and_era():
    out = brain.fallback_extract("something like Stalker but recent")
    assert out["intent"]["reference_ids"] == [1232]
    assert out["intent"]["min_year"] == 2010


def test_same_request_gives_different_lists_to_different_profiles(conn):
    a = profile.create_user(conn, "scifi")
    b = profile.create_user(conn, "crime")
    profile.save_onboarding(conn, a, SCIFI_FAN, genres_liked=["Sci-Fi"])
    profile.save_onboarding(conn, b, CRIME_FAN, genres_liked=["Crime"])
    first = [r["movie_id"] for r in brain.chat_turn(conn, a, "un film d'action ce soir")["recommendations"]]
    second = [r["movie_id"] for r in brain.chat_turn(conn, b, "un film d'action ce soir")["recommendations"]]
    assert first != second
    for movie_id in first + second:
        assert "Action" in catalog.load().loc[movie_id, "genre_set"]


def test_recommendations_skip_known_films(conn):
    user_id = profile.create_user(conn)
    profile.save_onboarding(conn, user_id, SCIFI_FAN)
    profile.log_interaction(conn, user_id, 1200, "seen")
    recos = [r["movie_id"] for r in brain.fallback_recommend(user_id, {}, 20)]
    assert not set(recos) & (set(SCIFI_FAN) | {1200})


def test_chat_turn_is_stored(conn):
    user_id = profile.create_user(conn)
    turn = brain.chat_turn(conn, user_id, "I loved Blade Runner, any sci-fi idea?")
    assert turn["reply"] and turn["recommendations"]
    assert 541 in profile.get_profile(conn, user_id)["liked_ids"]
    roles = [m["role"] for m in profile.history(conn, turn["conversation_id"])]
    assert roles == ["user", "assistant"]
    suggested = conn.execute("SELECT COUNT(*) FROM interactions WHERE type = 'suggested'").fetchone()[0]
    assert suggested == 3


def test_cinephile_slider_pushes_hidden_gems(conn):
    user_id = profile.create_user(conn)
    profile.save_onboarding(conn, user_id, SCIFI_FAN)
    user = profile.get_profile(conn, user_id)
    likes = catalog.load()["likes"]
    mainstream = [r["movie_id"] for r in brain.rank(user, {}, 10, cinephile=0.0)]
    cinephile = [r["movie_id"] for r in brain.rank(user, {}, 10, cinephile=1.0)]
    assert likes.loc[cinephile].mean() < likes.loc[mainstream].mean()


def test_card_without_poster_has_no_nan():
    # 1,981 films have no TMDB poster: a NaN there made /chat answer 500
    missing = catalog.load()["poster_url"].isna()
    movie_id = int(catalog.load().index[missing][0])
    assert catalog.card(movie_id)["poster_url"] is None

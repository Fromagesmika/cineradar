"""Part 1: linking ReDial titles to MovieLens and the evaluation metrics (no data needed)"""
import pytest

from src.data.redial import norm_title, split_name, title_keys
from src.eval.evaluate import _ndcg, case_intent


@pytest.mark.parametrize("redial, movielens", [
    ("The Matrix", "Matrix, The"),
    ("WALL-E", "WALL·E"),
    ("Why Him%3F", "Why Him?"),
    ("John Wick: Chapter 2", "John Wick: Chapter Two"),
    ("Amélie", "Amelie"),
])
def test_same_film_gets_same_key(redial, movielens):
    assert norm_title(redial) == norm_title(movielens)


def test_alternate_titles_in_brackets_are_keys():
    assert norm_title("Dumb and Dumber") in title_keys("Dumb & Dumber (Dumb and Dumber)")


def test_split_name_reads_the_year():
    assert split_name("Headhunter  (2009)") == ("Headhunter", 2009)
    assert split_name("Star Wars") == ("Star Wars", None)


def test_ndcg():
    assert _ndcg([1, 2, 3], {1}) == pytest.approx(1.0)
    assert _ndcg([2, 1, 3], {1}) == pytest.approx(1 / 1.585, rel=1e-3)
    assert _ndcg([4, 5, 6], {1}) == 0.0


def test_case_profile_is_passed_to_the_recommender():
    case = {"request": "", "liked": [1, 2], "disliked": [3], "targets": [4]}
    intent = case_intent(case, lambda text: {})
    assert intent["liked_ids"] == [1, 2] and intent["disliked_ids"] == [3]

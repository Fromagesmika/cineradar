from src.eval.metrics import (
    coverage_at_k,
    most_popular_baseline,
    ndcg_at_k,
    recall_at_k,
)


def test_perfect_ranking_has_ndcg_one():
    assert ndcg_at_k({1: [10, 20]}, {1: {10, 20}}) == 1.0


def test_no_hit_has_zero_scores():
    assert ndcg_at_k({1: [1, 2]}, {1: {9}}) == 0.0
    assert recall_at_k({1: [1, 2]}, {1: {9}}) == 0.0


def test_recall_half():
    assert recall_at_k({1: [10, 30]}, {1: {10, 20}}) == 0.5


def test_coverage():
    assert coverage_at_k({1: [1, 2], 2: [2, 3]}, n_items=6) == 0.5


def test_most_popular_skips_seen_films():
    recs = most_popular_baseline([1], {10: 5, 20: 3, 30: 1}, seen={1: {10}}, k=2)
    assert recs[1] == [20, 30]

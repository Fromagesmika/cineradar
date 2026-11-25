import json

import numpy as np
import pytest
torch = pytest.importorskip("torch")  # the model tests need requirements-train.txt

from src.config import MODELS, RESULTS
from src.data.split import build_graph
from src.eval.ranking import ground_truth, mask_matrix, metrics, top_k_lists
from src.model import taste
from src.model.nlgcl import NLGCL, SymmetricSparseMM
from src.model.toy import toy_split


def test_sparse_matmul_gradient_matches_torch():
    train, _, _, meta = toy_split(n_users=50, n_items=30, likes_per_user=8)
    adj = build_graph(train, meta["n_users"], meta["n_items"])
    x = torch.randn(80, 4, requires_grad=True)
    torch.sparse.mm(adj, x).pow(2).sum().backward()
    expected = x.grad.clone()
    x.grad = None
    SymmetricSparseMM.apply(adj.to_sparse_csr(), x).pow(2).sum().backward()
    assert torch.allclose(expected, x.grad, atol=1e-5)


def test_nlgcl_forward_and_losses_run():
    train, _, _, meta = toy_split(n_users=50, n_items=30, likes_per_user=8)
    model = NLGCL(meta["n_users"], meta["n_items"], build_graph(train, meta["n_users"], meta["n_items"]), dim=8)
    layers = model.propagate()
    final = model.readout(layers)
    users, pos, neg = torch.tensor([0, 1]), torch.tensor([2, 3]), torch.tensor([4, 5])
    loss = model.bpr_loss(final, users, pos, neg) + model.contrastive_loss(layers, users, pos)
    loss.backward()
    assert torch.isfinite(loss) and model.embedding.weight.grad is not None


def test_toy_training_beats_most_popular():
    from src.train.train import main

    name = "pytest_toy"
    try:
        main(["--toy", "--name", name, "--batch", "512", "--cl-batch", "256", "--epochs", "8", "--lr", "5e-3"])
        train, val, _, meta = toy_split()
        n_users, n_items = meta["n_users"], meta["n_items"]
        ckpt = torch.load(MODELS / f"{name}.pt", weights_only=False)
        model = NLGCL(n_users, n_items, build_graph(train, n_users, n_items))
        model.load_state_dict(ckpt["state"])
        user_emb, item_emb = model.final_embeddings()

        users = np.unique(val["u"])
        truth, mask = ground_truth(val, n_users), mask_matrix([train], n_users, n_items)
        gems = set(range(n_items)) - set(meta["popular_items"])
        counts = torch.as_tensor(np.bincount(train["i"], minlength=n_items), dtype=torch.float32)
        popular = metrics(top_k_lists(lambda u: counts.expand(len(u), -1), users, mask), users, truth, gems, n_items)
        trained = metrics(top_k_lists(lambda u: user_emb[u] @ item_emb.T, users, mask), users, truth, gems, n_items)
        assert trained["NDCG@10"] > 2 * popular["NDCG@10"]
    finally:
        (MODELS / f"{name}.pt").unlink(missing_ok=True)
        (RESULTS / f"{name}_history.json").unlink(missing_ok=True)


@pytest.fixture
def fake_artifacts(tmp_path, monkeypatch):
    # Films 1 and 2 point one way, film 3 the other way, film 4 is in between
    emb = np.array([[1, 0], [0.9, 0.1], [-1, 0], [0, 1]], dtype=np.float32)
    np.save(tmp_path / "item_emb.npy", emb)
    np.save(tmp_path / "movie_ids.npy", np.array([1, 2, 3, 4]))
    np.save(tmp_path / "item_likes.npy", np.array([10, 10, 10, 0], dtype=np.float32))
    (tmp_path / "meta.json").write_text(json.dumps({"model": "test"}))
    monkeypatch.setattr(taste, "ARTIFACTS", tmp_path)
    taste._load.cache_clear()
    yield
    taste._load.cache_clear()


def test_user_vector_and_score(fake_artifacts):
    vec = taste.user_vector([1])
    scores = taste.score(vec, [2, 3, 999])
    assert scores[0] > scores[1]
    assert scores[2] == -np.inf


def test_dislikes_push_films_down(fake_artifacts):
    plain = taste.score(taste.user_vector([4]), [3])[0]
    with_dislike = taste.score(taste.user_vector([4], disliked_ids=[3]), [3])[0]
    assert with_dislike < plain


def test_because_picks_closest_liked_film(fake_artifacts):
    assert taste.because([1, 3], [2, 999]) == [1, None]


def test_mock_mode_without_artifacts(tmp_path, monkeypatch):
    monkeypatch.setattr(taste, "ARTIFACTS", tmp_path)
    taste._load.cache_clear()
    with pytest.warns(UserWarning):
        assert taste.is_mock()
    vec = taste.user_vector([1, 2])
    assert vec.shape == (taste.MOCK_DIM,)
    assert np.isfinite(taste.score(vec, [5, 6])).all()
    taste._load.cache_clear()

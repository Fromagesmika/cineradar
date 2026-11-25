"""Part 3 tests run on the 79-film mock catalog with mock taste vectors, the others on the real app
catalog. test_part3.py sets RECO_CATALOG and TASTE_BACKEND when it is imported, so this switches the
backend before each test and rebuilds the catalog only when the backend changes"""
import importlib
import os

import pytest

from src.reco import catalog as reco_catalog

_current = {"backend": None}


@pytest.fixture(autouse=True)
def _reco_backend(request, monkeypatch):
    mock = request.module.__name__.endswith("test_part3")
    backend = "mock" if mock else None
    if backend:
        monkeypatch.setenv("RECO_CATALOG", "mock")
    else:
        monkeypatch.delenv("RECO_CATALOG", raising=False)
        monkeypatch.delenv("TASTE_BACKEND", raising=False)
    if _current["backend"] != backend:
        reco_catalog.get_catalog.cache_clear()
        importlib.import_module("src.reco.taste_mock")._EMB = None
        _current["backend"] = backend
    engine = importlib.import_module("src.reco.engine")
    if mock:
        monkeypatch.setattr(engine, "taste", importlib.import_module("src.reco.taste_mock"))
    else:
        monkeypatch.setattr(engine, "taste", importlib.import_module("src.model.taste"))
    yield

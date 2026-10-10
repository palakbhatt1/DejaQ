import hashlib
import re

import numpy as np
import pytest

from app import detector, llm
from app import store as store_mod


def fake_embed(texts):
    """Deterministic bag-of-words hashing embedder: same word set => cosine 1.0. No model download."""
    out = np.zeros((len(texts), 384), dtype="float32")
    for i, t in enumerate(texts):
        for w in re.findall(r"\w+", t.lower()):
            out[i, int(hashlib.md5(w.encode()).hexdigest(), 16) % 384] += 1.0
        out[i] /= np.linalg.norm(out[i]) or 1.0
    return out


@pytest.fixture(autouse=True)
def _isolate(monkeypatch):
    monkeypatch.setattr(store_mod, "embed", fake_embed)
    monkeypatch.setattr(llm, "GEMINI_KEY", "")   # no real LLM unless a test mocks ask_json
    detector._cache.clear()

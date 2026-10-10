"""Needs the real embedding model: RUN_SLOW=1 pytest -m slow -p no:cacheprovider"""
import json
import os
from pathlib import Path

import pytest

from app import store as store_mod
from app.detector import check
from app.store import Store

pytestmark = [pytest.mark.slow, pytest.mark.skipif(not os.getenv("RUN_SLOW"), reason="set RUN_SLOW=1")]


def test_no_false_duplicates_on_reflection_corpus(monkeypatch):
    from app.embeddings import embed
    monkeypatch.setattr(store_mod, "embed", embed)   # undo the autouse fake
    corpus = json.load(open(Path(__file__).parents[1] / "eval" / "reflection_corpus.json"))
    s, verdicts = Store(), {}
    for item in corpus:
        if item["expected"] != "original":
            verdicts[item["id"]] = check(s, item["question"])["verdict"]
        s.add(item["question"])
    assert verdicts["Q10"] == "exact_duplicate"
    for qid in ("Q5", "Q6", "Q7", "Q9"):      # genuinely different questions must never be called duplicates
        assert verdicts[qid] not in {"exact_duplicate", "semantic_duplicate"}, qid

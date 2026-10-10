from app import detector, llm
from app.detector import check
from app.store import Store


def make(*qs, tags=None):
    s = Store()
    for q in qs:
        s.add(q, tags)
    return s


def test_exact_duplicate_ignores_formatting():
    s = make("Define the reflection of light.")
    r = check(s, "  DEFINE the reflection of light ?? ")
    assert r["verdict"] == "exact_duplicate" and r["stage"] == "normalize"


def test_empty_store_is_different():
    assert check(Store(), "What is reflection of light?")["verdict"] == "different"


def test_unrelated_is_different_without_llm_call(monkeypatch):
    monkeypatch.setattr(llm, "ask_json", lambda *a, **k: (_ for _ in ()).throw(AssertionError("judge called")))
    s = make("What is reflection of light?")
    assert check(s, "Name the capital of France today")["verdict"] == "different"


def test_high_similarity_auto_duplicate_skips_judge(monkeypatch):
    monkeypatch.setattr(llm, "ask_json", lambda *a, **k: (_ for _ in ()).throw(AssertionError("judge called")))
    s = make("What is reflection of light?")
    r = check(s, "light of reflection is what")            # same words, reordered
    assert r["verdict"] == "semantic_duplicate" and r["stage"] == "embedding"


def test_numeric_mismatch_forces_judge_even_above_auto_threshold(monkeypatch):
    calls = []
    monkeypatch.setattr(detector, "AUTO_DUP_T", 0.9)
    monkeypatch.setattr(llm, "ask_json", lambda p, **k: calls.append(p) or {"label": "different", "match": None, "reason": "given value differs"})
    s = make("what is the angle of reflection")
    r = check(s, "what is the angle of reflection 30")     # cosine ~0.93, but a number appears
    assert calls and r["verdict"] == "different" and r["numeric_guard"] is True


def test_judge_receives_top_neighbors_and_match_is_resolved(monkeypatch):
    seen = {}
    monkeypatch.setattr(detector, "REVIEW_T", 0.3)
    monkeypatch.setattr(detector, "AUTO_DUP_T", 0.99)

    def judge(prompt, **k):
        seen["prompt"] = prompt
        return {"label": "duplicate", "match": 2, "reason": "same concept"}

    monkeypatch.setattr(llm, "ask_json", judge)
    s = make("what is reflection of light", "state the laws of reflection of light")
    r = check(s, "define reflection of light please")
    assert "1." in seen["prompt"] and "2." in seen["prompt"]
    assert r["verdict"] == "semantic_duplicate" and r["stage"] == "llm"
    assert r["match"] in {"what is reflection of light", "state the laws of reflection of light"}


def test_ambiguous_label_maps_to_review(monkeypatch):
    monkeypatch.setattr(detector, "REVIEW_T", 0.3)
    monkeypatch.setattr(detector, "AUTO_DUP_T", 0.99)
    monkeypatch.setattr(llm, "ask_json", lambda *a, **k: {"label": "ambiguous", "match": None, "reason": "needs context"})
    s = make("what is reflection of light")
    assert check(s, "what is reflection")["verdict"] == "review"


def test_no_llm_borderline_degrades_to_review(monkeypatch):
    monkeypatch.setattr(detector, "REVIEW_T", 0.3)
    monkeypatch.setattr(detector, "AUTO_DUP_T", 0.99)
    monkeypatch.setattr(detector, "FALLBACK_DUP_T", 0.99)
    s = make("what is reflection of light")
    r = check(s, "what is reflection")
    assert r["verdict"] == "review" and r["stage"] == "no-llm"


def test_judge_results_are_cached(monkeypatch):
    n = {"c": 0}
    monkeypatch.setattr(detector, "REVIEW_T", 0.3)
    monkeypatch.setattr(detector, "AUTO_DUP_T", 0.99)

    def judge(*a, **k):
        n["c"] += 1
        return {"label": "different", "match": None, "reason": "x"}

    monkeypatch.setattr(llm, "ask_json", judge)
    s = make("what is reflection of light")
    check(s, "what is reflection")
    check(s, "what is reflection")
    assert n["c"] == 1


def test_tags_never_affect_the_verdict():
    a = make("Define the reflection of light.", tags={"topic": "optics"})
    b = make("Define the reflection of light.", tags={"completely": "different", "keys": 1})
    q = "define reflection of light"
    assert check(a, q)["verdict"] == check(b, q)["verdict"]

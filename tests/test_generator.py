from app import llm
from app.generator import generate
from app.store import Store


def fake_llm(prompt, **k):
    if "subtopics" in prompt and "Reply JSON" in prompt and "distinct subtopics" in prompt:
        return {"subtopics": ["laws of reflection", "spherical mirrors", "plane mirror images"]}
    return {"questions": [
        {"question": "light of reflection is what", "subtopic": "definition"},          # reorder of an existing question
        {"question": "Describe how a concave mirror forms a real image", "subtopic": "spherical mirrors"},
        {"question": "Describe how a concave mirror forms a real image!", "subtopic": "spherical mirrors"},  # in-batch repeat
        {"question": "List three uses of plane mirrors in daily life", "subtopic": "plane mirror images"},
    ]}


def test_generation_rejects_duplicates_and_adds_accepted(monkeypatch):
    monkeypatch.setattr(llm, "ask_json", fake_llm)
    s = Store()
    s.add("What is reflection of light?")
    out = generate(s, "reflection of light", n=2)
    acc = [a["question"] for a in out["accepted"]]
    assert len(acc) == 2 and all("concave" in q or "plane mirrors" in q for q in acc)
    rej = {r["question"]: r["verdict"] for r in out["rejected"]}
    assert rej["light of reflection is what"] == "semantic_duplicate"
    assert len(s.items) == 3
    assert out["plan"] and out["plan"][0]["coverage"] <= out["plan"][-1]["coverage"]
    assert all(i["tags"].get("source") == "generated" for i in s.items[1:])


def test_generation_reports_llm_failure(monkeypatch):
    monkeypatch.setattr(llm, "ask_json", lambda *a, **k: None)
    out = generate(Store(), "optics", n=3)
    assert out["accepted"] == [] and "error" in out

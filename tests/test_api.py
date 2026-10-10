from fastapi.testclient import TestClient

from app.main import create_app
from app.store import Store


def client():
    return TestClient(create_app(Store(), seed=False))


def test_health():
    r = client().get("/api/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_add_then_duplicate_is_rejected_with_409():
    c = client()
    assert c.post("/api/questions", json={"question": "What is reflection of light?", "tags": {"a": 1}}).status_code == 200
    r = c.post("/api/questions", json={"question": "what is reflection of light", "tags": {}})
    assert r.status_code == 409 and r.json()["detail"]["verdict"] == "exact_duplicate"


def test_check_never_mutates():
    c = client()
    c.post("/api/check", json={"question": "What is reflection of light?"})
    assert c.get("/api/questions").json() == []


def test_tag_filter_is_filter_only():
    c = client()
    c.post("/api/questions", json={"question": "What is reflection of light?", "tags": {"topic": "optics"}})
    c.post("/api/questions", json={"question": "State Newton's second law of motion", "tags": {"topic": "mechanics"}})
    r = c.get("/api/questions", params={"tag_key": "topic", "tag_value": "optics"}).json()
    assert len(r) == 1 and r[0]["tags"]["topic"] == "optics"
    assert len(c.get("/api/questions").json()) == 2


def test_validation():
    assert client().post("/api/check", json={"question": ""}).status_code == 422


def test_batch_check():
    c = client()
    c.post("/api/questions", json={"question": "What is reflection of light?"})
    r = c.post("/api/check/batch", json={"questions": ["what is reflection of light", "Unrelated query here"]})
    assert r.status_code == 200
    res = r.json()["results"]
    assert len(res) == 2
    assert res[0]["verdict"] == "exact_duplicate"
    assert res[1]["verdict"] == "different"


def test_delete_question():
    c = client()
    add_res = c.post("/api/questions", json={"question": "What is reflection of light?"}).json()
    qid = add_res["question"]["id"]
    assert len(c.get("/api/questions").json()) == 1

    del_res = c.delete(f"/api/questions/{qid}")
    assert del_res.status_code == 200
    assert len(c.get("/api/questions").json()) == 0

    del_again = c.delete(f"/api/questions/{qid}")
    assert del_again.status_code == 404


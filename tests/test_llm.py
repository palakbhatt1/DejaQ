import httpx

from app import llm


class Resp:
    status_code = 200

    def raise_for_status(self):
        pass

    def json(self):
        return {"candidates": [{"content": {"parts": [{"text": '{"ok": true}'}]}}]}


def test_retries_then_succeeds(monkeypatch):
    monkeypatch.setattr(llm, "GEMINI_KEY", "k")
    monkeypatch.setattr(llm.time, "sleep", lambda s: None)
    calls = {"n": 0}

    def post(*a, **k):
        calls["n"] += 1
        if calls["n"] == 1:
            raise httpx.ConnectError("boom")
        return Resp()

    monkeypatch.setattr(llm.httpx, "post", post)
    assert llm.ask_json("x") == {"ok": True} and calls["n"] == 2


def test_gives_up_after_retries(monkeypatch):
    monkeypatch.setattr(llm, "GEMINI_KEY", "k")
    monkeypatch.setattr(llm, "LLM_RETRIES", 3)
    monkeypatch.setattr(llm.time, "sleep", lambda s: None)
    monkeypatch.setattr(llm.httpx, "post", lambda *a, **k: (_ for _ in ()).throw(httpx.ConnectError("down")))
    assert llm.ask_json("x") is None


def test_no_key_returns_none():
    assert llm.ask_json("x") is None

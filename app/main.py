import json
import os
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import llm
from .config import DATA_PATH
from .detector import check
from .generator import generate
from .store import Store

SEED_PATH = "data/seed_questions.json"


class CheckReq(BaseModel):
    question: str = Field(min_length=3, max_length=500)


class GenReq(BaseModel):
    topic: str = Field(min_length=2, max_length=200)
    n: int = Field(5, ge=1, le=15)


class AddReq(BaseModel):
    question: str = Field(min_length=3, max_length=500)
    tags: dict[str, Any] = {}
    force: bool = False   # allow adding a `review` question after a human decision


class BatchCheckReq(BaseModel):
    questions: list[str] = Field(min_length=1, max_length=50)


def create_app(store: Store | None = None, seed: bool = True) -> FastAPI:
    store = store if store is not None else Store(DATA_PATH)

    @asynccontextmanager
    async def lifespan(_):
        if seed and not store.items and os.path.exists(SEED_PATH):
            with open(SEED_PATH) as f:
                for q in json.load(f):
                    store.add(q["question"], q.get("tags"), persist=False)
        yield

    app = FastAPI(title="DejaQ", description="Intelligent question generation with redundancy prevention and duplicate detection", lifespan=lifespan)

    @app.get("/api/health")
    def health():
        return {"status": "ok", "questions": len(store.items), "llm": llm.available()}

    @app.post("/api/check")   # detector only, never mutates the store
    def _check(b: CheckReq):
        return check(store, b.question)

    @app.post("/api/check/batch")
    def _check_batch(b: BatchCheckReq):
        results = []
        for q in b.questions:
            q_clean = q.strip()
            if len(q_clean) >= 3:
                results.append({"question": q_clean, **check(store, q_clean)})
            else:
                results.append({"question": q_clean, "verdict": "error", "reason": "Question must be at least 3 characters"})
        return {"results": results}

    @app.post("/api/generate")
    def _generate(b: GenReq):
        return generate(store, b.topic, b.n)

    @app.post("/api/questions")
    def _add(b: AddReq):
        r = check(store, b.question)
        if r["verdict"] != "different" and not (b.force and r["verdict"] == "review"):
            raise HTTPException(409, detail=r)   # duplicates are never added; `review` needs force
        return {"question": store.add(b.question, b.tags), "check": r}

    @app.delete("/api/questions/{qid}")
    def _delete(qid: str):
        if not store.delete(qid):
            raise HTTPException(404, detail="Question not found")
        return {"status": "deleted", "id": qid}

    @app.get("/api/questions")
    def _list(tag_key: str | None = None, tag_value: str | None = None):
        return [{"id": i["id"], "question": i["text"], "tags": i["tags"]}
                for i in store.filter(tag_key, tag_value)]

    if os.path.isdir("static"):
        app.mount("/", StaticFiles(directory="static", html=True), name="ui")
    return app


app = create_app()
